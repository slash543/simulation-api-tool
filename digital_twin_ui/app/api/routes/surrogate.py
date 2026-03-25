"""
FastAPI surrogate-model endpoints.

Routes
------
GET  /surrogate/models              — list MLflow surrogate runs
POST /surrogate/predict             — batch contact-pressure prediction
POST /surrogate/csar                — CSAR vs insertion depth (surrogate)
POST /surrogate/predict-vtp         — annotate VTP with predicted pressures
POST /surrogate/csar-from-vtp       — CSAR from VTP geometry
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from digital_twin_ui.app.api.schemas.surrogate import (
    BandCSARSeries,
    CSARFromVTPRequest,
    CSARFromVTPResponse,
    CSARRequest,
    CSARResponse,
    ListSurrogateModelsResponse,
    PredictRequest,
    PredictResponse,
    PredictVTPRequest,
    PredictVTPResponse,
    SurrogateModelInfo,
)
from digital_twin_ui.surrogate.csar_engine import CSAREngine, build_insertion_depths, load_reference_facets
from digital_twin_ui.surrogate.predictor import (
    SurrogatePredictor,
    default_model_dir,
    is_model_available,
    list_mlflow_runs,
)
from digital_twin_ui.surrogate.vtp_processor import VTPProcessor, compute_csar_from_vtp

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/surrogate", tags=["surrogate"])

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Data root inside the container (bind-mounted from ./data/surrogate on host)
_SURROGATE_DATA_ROOT = Path(os.getenv("SURROGATE_DATA_PATH", "/app/surrogate_data"))
_RUNS_ROOT = Path(os.getenv("RUNS_PATH", "/app/runs"))

# Host-side path to the surrogate data directory (shown to users)
_SURROGATE_HOST_PATH = os.getenv("SURROGATE_HOST_PATH", "./data/surrogate")
_RUNS_HOST_PATH = os.getenv("RUNS_HOST_PATH", "./runs")


def _to_host_path(container_path: str) -> str:
    """Translate a container-internal path to a host-visible path."""
    p = str(container_path)
    if p.startswith(str(_SURROGATE_DATA_ROOT)):
        rel = p[len(str(_SURROGATE_DATA_ROOT)):]
        return _SURROGATE_HOST_PATH.rstrip("/") + rel
    if p.startswith(str(_RUNS_ROOT)):
        rel = p[len(str(_RUNS_ROOT)):]
        return _RUNS_HOST_PATH.rstrip("/") + rel
    return p


def _resolve_path(raw: str) -> Path:
    """
    Resolve a user-supplied path string to an absolute container path.

    Supports paths starting with:
    - /app/surrogate_data/...  (container-absolute)
    - /app/runs/...            (container-absolute)
    - data/surrogate/...       (relative to surrogate data root)
    - relative paths           (resolved relative to surrogate data root)
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    # Try relative to surrogate data root first
    candidate = _SURROGATE_DATA_ROOT / p
    if candidate.exists():
        return candidate
    # Try relative to working directory
    return Path.cwd() / p


def _default_reference_facets_path() -> Path:
    """Default path for the reference facets CSV used for CSAR computation."""
    return _SURROGATE_DATA_ROOT / "training" / "reference_facets.csv"


@lru_cache(maxsize=1)
def _get_predictor_cached() -> SurrogatePredictor:
    """Cache the 'latest' predictor so it is not reloaded on every request."""
    return SurrogatePredictor.load_latest()


def _get_predictor(run_id: Optional[str] = None) -> SurrogatePredictor:
    """
    Return the appropriate predictor.

    If *run_id* is None, returns the cached 'latest' predictor.
    If *run_id* is given, loads from MLflow (not cached — intended for occasional use).
    """
    if run_id is None:
        if not is_model_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Surrogate model not available. "
                    "Train a model using the full_pipeline notebook and ensure "
                    "the artifacts are saved to data/surrogate/models/latest/."
                ),
            )
        return _get_predictor_cached()

    # Load specific run from MLflow
    try:
        return SurrogatePredictor.load_from_run(run_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not load model for run_id={run_id}: {exc}",
        ) from exc


def _build_band_series(csar_df: pd.DataFrame) -> dict[str, BandCSARSeries]:
    """Convert CSAR DataFrame rows to BandCSARSeries objects keyed by label."""
    result = {}
    for label, group in csar_df.groupby("band_label"):
        g = group.sort_values("insertion_depth_mm")
        result[str(label)] = BandCSARSeries(
            label=str(label),
            zmin_mm=float(g["zmin_mm"].iloc[0]),
            zmax_mm=float(g["zmax_mm"].iloc[0]),
            insertion_depths_mm=g["insertion_depth_mm"].tolist(),
            csar=[None if (v is not None and np.isnan(v)) else (None if v is None else float(v)) for v in g["csar"]],
            contact_area_mm2=g["contact_area_mm2"].tolist(),
            total_area_mm2=float(g["total_area_mm2"].iloc[0]),
            n_contact_facets=g["n_contact_facets"].tolist(),
            n_total_facets=int(g["n_total_facets"].iloc[0]),
            mean_cp_MPa=g["mean_cp_MPa"].tolist(),
            max_cp_MPa=g["max_cp_MPa"].tolist(),
        )
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/models",
    response_model=ListSurrogateModelsResponse,
    summary="List trained surrogate models from MLflow",
)
def list_surrogate_models() -> ListSurrogateModelsResponse:
    """
    Return recent surrogate-lab MLflow runs plus availability of the 'latest' model.

    The 'latest' model is what all prediction endpoints use by default.
    It is populated by copying artifacts after training in the full_pipeline notebook.
    """
    runs = list_mlflow_runs(n=20)
    models = []
    for r in runs:
        if "error" in r:
            logger.warning("MLflow listing error: %s", r["error"])
            continue
        models.append(
            SurrogateModelInfo(
                run_id=r["run_id"],
                status=r["status"],
                start_time=r.get("start_time"),
                metrics={k: float(v) for k, v in r.get("metrics", {}).items()},
                params={k: str(v) for k, v in r.get("params", {}).items()},
                artifact_uri=r.get("artifact_uri"),
            )
        )
    return ListSurrogateModelsResponse(
        models=models,
        latest_available=is_model_available(),
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Batch contact-pressure prediction",
)
def predict_contact_pressure(req: PredictRequest) -> PredictResponse:
    """
    Predict contact pressure [MPa] for a set of facets at given insertion depths.

    Provide either:
    - ``facets``: inline list of FacetRow objects (each includes insertion_depth)
    - ``facets_csv_path``: path to a CSV with columns centroid_x/y/z, facet_area,
      insertion_depth (container-accessible path)

    Returns per-facet predicted contact pressure in MPa.
    """
    predictor = _get_predictor(req.run_id)

    if req.facets is not None:
        df = pd.DataFrame([f.model_dump() for f in req.facets])
    elif req.facets_csv_path is not None:
        csv_path = _resolve_path(req.facets_csv_path)
        if not csv_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CSV not found: {req.facets_csv_path}",
            )
        df = pd.read_csv(csv_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either 'facets' or 'facets_csv_path'.",
        )

    try:
        cp = predictor.predict(df)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return PredictResponse(
        n_facets=len(cp),
        contact_pressure_MPa=cp.tolist(),
        run_id_used=req.run_id,
        model_dir=str(predictor.model_dir),
    )


@router.post(
    "/csar",
    response_model=CSARResponse,
    summary="Compute CSAR vs insertion depth using surrogate model",
)
def compute_csar(req: CSARRequest) -> CSARResponse:
    """
    Compute Contact Surface Area Ratio (CSAR) vs insertion depth.

    Uses the trained surrogate model to predict contact pressure for all
    reference facets at each insertion depth, then computes CSAR per Z band.

    The reference facets CSV must exist at ``data/surrogate/training/reference_facets.csv``
    (default) or be specified via ``facets_csv_path``.

    CSAR is defined as:
        CSAR(depth, band) = Σ area_i [cp_i > threshold] / Σ area_i [all in band]
    """
    predictor = _get_predictor(req.run_id)

    # Resolve facets source
    if req.facets_csv_path:
        facets_path = _resolve_path(req.facets_csv_path)
    else:
        facets_path = _default_reference_facets_path()

    if not facets_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Reference facets file not found: {facets_path}. "
                "Export a reference_facets.csv from the full_pipeline notebook first "
                "(see: data/surrogate/training/reference_facets.csv)."
            ),
        )

    try:
        facets_df = load_reference_facets(facets_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Build insertion depths
    if req.insertion_depths_mm:
        depths = [float(d) for d in req.insertion_depths_mm]
    else:
        depths = build_insertion_depths(req.max_depth_mm, req.depth_step_mm)

    z_bands_dicts = [
        {"zmin": b.zmin, "zmax": b.zmax, "label": b.effective_label()}
        for b in req.z_bands
    ]

    try:
        engine = CSAREngine(predictor, cp_threshold=req.cp_threshold)
        csar_df = engine.compute_csar_vs_depth(facets_df, depths, z_bands_dicts)
    except Exception as exc:
        logger.exception("CSAR computation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSAR computation failed: {exc}",
        ) from exc

    return CSARResponse(
        insertion_depths_mm=sorted(csar_df["insertion_depth_mm"].unique().tolist()),
        bands=_build_band_series(csar_df),
        n_facets_total=len(facets_df),
        run_id_used=req.run_id,
        facets_source=str(facets_path),
    )


@router.post(
    "/predict-vtp",
    response_model=PredictVTPResponse,
    summary="Annotate VTP file with surrogate-predicted contact pressures",
)
def predict_vtp(req: PredictVTPRequest) -> PredictVTPResponse:
    """
    Read a VTP file, predict contact pressure on each facet at *insertion_depth_mm*,
    and save a new VTP file with the predicted values.

    The input VTP can be from:
    - xplt-parser's export_vtp() output in runs/ directory
    - data/surrogate/results/ or any container-accessible path

    The output VTP will have:
    - Same geometry (points, connectivity) as the input
    - ``contact_pressure_MPa`` CellData array populated with predictions
    """
    predictor = _get_predictor(req.run_id)

    vtp_path = _resolve_path(req.vtp_path)
    if not vtp_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VTP file not found: {req.vtp_path}",
        )

    # Determine output path
    if req.output_path:
        out_path = _resolve_path(req.output_path)
    else:
        out_path = vtp_path.parent / (vtp_path.stem + "_predicted.vtp")

    try:
        vtp = VTPProcessor.read(vtp_path)
        written = VTPProcessor.predict_and_save(
            vtp, predictor, req.insertion_depth_mm, out_path
        )
    except Exception as exc:
        logger.exception("VTP prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"VTP prediction failed: {exc}",
        ) from exc

    return PredictVTPResponse(
        output_vtp_path=str(written),
        host_output_path=_to_host_path(str(written)),
        n_faces=vtp.n_faces,
        insertion_depth_mm=req.insertion_depth_mm,
        run_id_used=req.run_id,
    )


@router.post(
    "/csar-from-vtp",
    response_model=CSARFromVTPResponse,
    summary="Compute CSAR vs insertion depth from VTP geometry",
)
def csar_from_vtp(req: CSARFromVTPRequest) -> CSARFromVTPResponse:
    """
    Read facet geometry from a VTP file and compute CSAR vs insertion depth.

    Useful when you have a VTP file exported from a simulation but want to
    evaluate CSAR for arbitrary insertion depths using the surrogate model —
    without running new FEM simulations.
    """
    predictor = _get_predictor(req.run_id)

    vtp_path = _resolve_path(req.vtp_path)
    if not vtp_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VTP file not found: {req.vtp_path}",
        )

    try:
        vtp = VTPProcessor.read(vtp_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse VTP file: {exc}",
        ) from exc

    if req.insertion_depths_mm:
        depths = [float(d) for d in req.insertion_depths_mm]
    else:
        depths = build_insertion_depths(req.max_depth_mm, req.depth_step_mm)

    z_bands_dicts = [
        {"zmin": b.zmin, "zmax": b.zmax, "label": b.effective_label()}
        for b in req.z_bands
    ]

    try:
        csar_df = compute_csar_from_vtp(
            vtp, predictor, depths, z_bands_dicts, req.cp_threshold
        )
    except Exception as exc:
        logger.exception("CSAR-from-VTP computation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSAR computation failed: {exc}",
        ) from exc

    return CSARFromVTPResponse(
        insertion_depths_mm=sorted(csar_df["insertion_depth_mm"].unique().tolist()),
        bands=_build_band_series(csar_df),
        n_facets=vtp.n_faces,
        vtp_source=str(vtp_path),
        run_id_used=req.run_id,
    )
