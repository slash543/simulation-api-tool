"""
FastAPI route handlers for simulation, DOE, extraction, and ML endpoints.

Endpoints
---------
GET  /templates                – list available simulation templates
GET  /templates/{name}         – get one template's config
POST /simulations/run          – submit one simulation (async via Celery)
POST /simulations/run/sync     – run one simulation synchronously (blocking)
GET  /simulations/{task_id}    – poll Celery task status
POST /doe/run                  – submit a DOE campaign (async via Celery)
GET  /doe/{task_id}            – poll DOE campaign task status
POST /extract                  – extract from an existing xplt (async)
POST /ml/predict               – predict pressure for one speed (ML)
POST /ml/predict/batch         – predict pressure for multiple speeds (ML)
GET  /health                   – health check
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from digital_twin_ui.app.api.schemas.simulation import (
    DOERequest,
    DOEResultResponse,
    ExtractionRequest,
    HealthResponse,
    MLPredictBatchRequest,
    MLPredictBatchResponse,
    MLPredictRequest,
    MLPredictResponse,
    PressureResultResponse,
    SimulationRequest,
    SimulationResultResponse,
    TaskResponse,
    TaskStatusResponse,
    TemplateInfo,
    TemplateListResponse,
)
from celery.result import AsyncResult

from digital_twin_ui.app.core.config import get_settings
from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.extraction.xplt_parser import extract_contact_pressure
from digital_twin_ui.simulation.simulation_runner import SimulationRunner
from digital_twin_ui.tasks.simulation_tasks import (
    extract_results_task,
    run_doe_campaign_task,
    run_full_pipeline_task,
    run_simulation_task,
)

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_celery_app():
    from digital_twin_ui.tasks.celery_app import celery_app
    return celery_app


def _task_status(task_id: str) -> TaskStatusResponse:
    """Fetch current state of a Celery task."""
    app = _get_celery_app()
    res = AsyncResult(task_id, app=app)
    state = res.state

    if state == "PENDING":
        return TaskStatusResponse(task_id=task_id, status="PENDING")
    if state == "STARTED":
        return TaskStatusResponse(task_id=task_id, status="RUNNING")
    if state == "SUCCESS":
        return TaskStatusResponse(task_id=task_id, status="SUCCESS", result=res.result)
    if state == "FAILURE":
        return TaskStatusResponse(
            task_id=task_id,
            status="FAILURE",
            error=str(res.result),
        )
    return TaskStatusResponse(task_id=task_id, status=state)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health_check() -> HealthResponse:
    """Return API liveness status."""
    return HealthResponse()


# ---------------------------------------------------------------------------
# Template registry endpoints
# ---------------------------------------------------------------------------

@router.get("/templates", response_model=TemplateListResponse, tags=["templates"])
async def list_templates() -> TemplateListResponse:
    """
    Return the list of available simulation templates with their configurations.

    Each template describes a FEB file, its speed range, number of steps, and
    per-step displacement magnitudes.
    """
    from digital_twin_ui.simulation.template_registry import get_template_registry

    registry = get_template_registry()
    templates = [
        TemplateInfo(
            name=tc.name,
            label=tc.label,
            n_steps=tc.n_steps,
            speed_range_min=tc.speed_range.min_mm_s,
            speed_range_max=tc.speed_range.max_mm_s,
            displacements_mm=tc.displacements_mm,
        )
        for tc in registry.all_configs()
    ]
    return TemplateListResponse(templates=templates)


@router.get(
    "/templates/{name}",
    response_model=TemplateInfo,
    tags=["templates"],
)
async def get_template(name: str) -> TemplateInfo:
    """
    Return the configuration for a single simulation template.

    Args:
        name: Template name (e.g. ``DT_BT_14Fr_FO_10E_IR12``).
    """
    from digital_twin_ui.simulation.template_registry import get_template_registry
    from fastapi import HTTPException

    registry = get_template_registry()
    try:
        tc = registry.get(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    return TemplateInfo(
        name=tc.name,
        label=tc.label,
        n_steps=tc.n_steps,
        speed_range_min=tc.speed_range.min_mm_s,
        speed_range_max=tc.speed_range.max_mm_s,
        displacements_mm=tc.displacements_mm,
    )


# ---------------------------------------------------------------------------
# Single simulation — async
# ---------------------------------------------------------------------------

@router.post(
    "/simulations/run",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["simulation"],
)
async def submit_simulation(body: SimulationRequest) -> TaskResponse:
    """
    Submit a simulation job to the Celery task queue.

    Returns immediately with a ``task_id``, ``run_id``, and the ``run_dir``
    path where results will be written. Poll status via
    ``GET /simulations/{task_id}``.
    """
    # Generate run_id and pre-create the run directory so the path is known
    # before the worker starts, allowing callers to watch the folder.
    cfg = get_settings()
    run_id = body.run_id or (
        "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_" + uuid.uuid4().hex[:4]
    )
    run_dir = cfg.runs_dir_abs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    xplt_path = run_dir / "input.xplt"

    logger.info(
        "Submitting simulation",
        speed_mm_s=body.speed_mm_s,
        template=body.template,
        run_id=run_id,
    )

    if body.extract:
        task = run_full_pipeline_task.delay(
            speed_mm_s=body.speed_mm_s,
            run_id=run_id,
            log_mlflow=body.log_mlflow,
            template=body.template,
            speeds_mm_s=body.speeds_mm_s,
            dwell_time_s=body.dwell_time_s,
        )
    else:
        task = run_simulation_task.delay(
            speed_mm_s=body.speed_mm_s,
            run_id=run_id,
            template=body.template,
            speeds_mm_s=body.speeds_mm_s,
            dwell_time_s=body.dwell_time_s,
        )

    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Simulation queued for template={body.template} speed={body.speed_mm_s} mm/s",
        run_id=run_id,
        run_dir=str(run_dir),
        xplt_path=str(xplt_path),
    )


# ---------------------------------------------------------------------------
# Single simulation — synchronous (blocking)
# ---------------------------------------------------------------------------

@router.post(
    "/simulations/run/sync",
    response_model=SimulationResultResponse,
    tags=["simulation"],
)
async def run_simulation_sync(body: SimulationRequest) -> SimulationResultResponse:
    """
    Run one simulation synchronously (blocking).

    Useful for development; use the async endpoint for production.
    """
    logger.info(
        "Running simulation synchronously",
        speed_mm_s=body.speed_mm_s,
        template=body.template,
    )
    runner = SimulationRunner()
    result = await runner.run_async(
        speed_mm_s=body.speed_mm_s,
        run_id=body.run_id,
        template=body.template,
        speeds_mm_s=body.speeds_mm_s,
        dwell_time_s=body.dwell_time_s,
    )

    extraction_dict = None
    if body.extract and result.succeeded and result.xplt_file.exists():
        try:
            pressure = extract_contact_pressure(result.xplt_file)
            extraction_dict = pressure.as_dict()
        except Exception as exc:
            logger.warning("Extraction failed in sync endpoint", error=str(exc))

    mlflow_run_id = None
    if body.log_mlflow and extraction_dict is not None:
        try:
            from digital_twin_ui.experiments.mlflow_manager import MLflowManager
            mgr = MLflowManager()
            mlflow_run_id = mgr.log_simulation_run(
                run_name=result.run_id,
                speed_mm_s=body.speed_mm_s,
                max_pressure=extraction_dict.get("max_pressure", 0.0),
                mean_pressures=extraction_dict.get("mean_pressure", []),
                times=extraction_dict.get("times", []),
            )
        except Exception as exc:
            logger.warning("MLflow logging failed in sync endpoint", error=str(exc))

    return SimulationResultResponse(
        run_id=result.run_id,
        status=result.status.value,
        speed_mm_s=result.speed_mm_s,
        duration_s=result.duration_s,
        error_message=result.error_message,
        extraction=extraction_dict,
        mlflow_run_id=mlflow_run_id,
        run_dir=str(result.run_dir),
        xplt_path=str(result.xplt_file),
    )


# ---------------------------------------------------------------------------
# Poll simulation task status
# ---------------------------------------------------------------------------

@router.get(
    "/simulations/{task_id}",
    response_model=TaskStatusResponse,
    tags=["simulation"],
)
async def get_simulation_status(task_id: str) -> TaskStatusResponse:
    """Poll the status of an async simulation task."""
    return _task_status(task_id)


# ---------------------------------------------------------------------------
# DOE campaign — async
# ---------------------------------------------------------------------------

@router.post(
    "/doe/run",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["doe"],
)
async def submit_doe_campaign(body: DOERequest) -> TaskResponse:
    """
    Submit a Design-of-Experiments campaign to the task queue.

    Returns immediately with a ``task_id``.
    """
    if body.speed_min >= body.speed_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="speed_min must be less than speed_max",
        )

    logger.info(
        "Submitting DOE campaign",
        n_samples=body.n_samples,
        speed_min=body.speed_min,
        speed_max=body.speed_max,
        template=body.template,
    )

    task = run_doe_campaign_task.delay(
        n_samples=body.n_samples,
        speed_min=body.speed_min,
        speed_max=body.speed_max,
        sampler=body.sampler,
        seed=body.seed,
        extract=body.extract,
        log_mlflow=body.log_mlflow,
        template=body.template,
        max_perturbation=body.max_perturbation,
        dwell_time_s=body.dwell_time_s,
    )

    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"DOE campaign queued: {body.n_samples} samples",
    )


# ---------------------------------------------------------------------------
# Poll DOE task status
# ---------------------------------------------------------------------------

@router.get(
    "/doe/{task_id}",
    response_model=TaskStatusResponse,
    tags=["doe"],
)
async def get_doe_status(task_id: str) -> TaskStatusResponse:
    """Poll the status of a DOE campaign task."""
    return _task_status(task_id)


# ---------------------------------------------------------------------------
# Post-hoc extraction from an xplt file
# ---------------------------------------------------------------------------

@router.post(
    "/extract",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["extraction"],
)
async def submit_extraction(body: ExtractionRequest) -> TaskResponse:
    """
    Extract contact pressure from an existing ``.xplt`` file (async).

    The file must already exist on the server filesystem.
    """
    xplt = Path(body.xplt_path)
    if not xplt.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"xplt file not found: {body.xplt_path}",
        )

    task = extract_results_task.delay(
        xplt_path=str(xplt),
        variable_name=body.variable_name,
    )

    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Extraction queued for {xplt.name}",
    )


@router.post(
    "/extract/sync",
    response_model=PressureResultResponse,
    tags=["extraction"],
)
async def extract_sync(body: ExtractionRequest) -> PressureResultResponse:
    """
    Extract contact pressure from an xplt file synchronously (blocking).
    """
    xplt = Path(body.xplt_path)
    if not xplt.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"xplt file not found: {body.xplt_path}",
        )

    try:
        result = extract_contact_pressure(xplt, variable_name=body.variable_name)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Extraction failed: {exc}",
        )

    return PressureResultResponse(
        variable_name=result.variable_name,
        times=list(result.times),
        max_pressure=float(result.max_pressure),
        mean_pressure=list(result.mean_pressure),
        n_faces=result.n_faces,
        source_path=str(result.source_path) if result.source_path else None,
    )


# ---------------------------------------------------------------------------
# ML prediction endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/ml/predict",
    response_model=MLPredictResponse,
    tags=["ml"],
)
async def ml_predict(body: MLPredictRequest) -> MLPredictResponse:
    """
    Predict peak contact pressure for a given insertion speed using the
    trained MLP model.

    The default checkpoint (``models/pressure_mlp.pt``) must exist.
    """
    from digital_twin_ui.ml.inference import PressurePredictor

    try:
        predictor = PressurePredictor.from_default_checkpoint()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {exc}",
        )

    predicted = predictor.predict(body.speed_mm_s)
    return MLPredictResponse(
        speed_mm_s=body.speed_mm_s,
        predicted_max_pressure=predicted,
    )


@router.post(
    "/ml/predict/batch",
    response_model=MLPredictBatchResponse,
    tags=["ml"],
)
async def ml_predict_batch(body: MLPredictBatchRequest) -> MLPredictBatchResponse:
    """
    Predict peak contact pressure for a list of insertion speeds.
    """
    from digital_twin_ui.ml.inference import PressurePredictor

    try:
        predictor = PressurePredictor.from_default_checkpoint()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {exc}",
        )

    predictions_values = predictor.predict_batch(body.speeds_mm_s)
    predictions = [
        MLPredictResponse(speed_mm_s=s, predicted_max_pressure=p)
        for s, p in zip(body.speeds_mm_s, predictions_values)
    ]
    return MLPredictBatchResponse(predictions=predictions)
