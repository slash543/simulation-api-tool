"""
FastAPI route handlers for simulation, DOE, extraction, and ML endpoints.

Endpoints
---------
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
)
from celery.result import AsyncResult

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

    Returns immediately with a ``task_id`` that can be polled via
    ``GET /simulations/{task_id}``.
    """
    logger.info("Submitting simulation", speed_mm_s=body.speed_mm_s)

    if body.extract:
        task = run_full_pipeline_task.delay(
            speed_mm_s=body.speed_mm_s,
            run_id=body.run_id,
            log_mlflow=body.log_mlflow,
        )
    else:
        task = run_simulation_task.delay(
            speed_mm_s=body.speed_mm_s,
            run_id=body.run_id,
        )

    return TaskResponse(
        task_id=task.id,
        status="PENDING",
        message=f"Simulation queued for speed={body.speed_mm_s} mm/s",
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
    logger.info("Running simulation synchronously", speed_mm_s=body.speed_mm_s)
    runner = SimulationRunner()
    result = runner.run(speed_mm_s=body.speed_mm_s, run_id=body.run_id)

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
    )

    task = run_doe_campaign_task.delay(
        n_samples=body.n_samples,
        speed_min=body.speed_min,
        speed_max=body.speed_max,
        sampler=body.sampler,
        seed=body.seed,
        extract=body.extract,
        log_mlflow=body.log_mlflow,
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
