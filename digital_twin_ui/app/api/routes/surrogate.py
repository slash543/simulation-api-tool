"""
FastAPI surrogate-model endpoints.

Routes
------
GET  /surrogate/models              — list MLflow surrogate runs
GET  /surrogate/list-vtps           — list available VTP files in runs/ and surrogate_data/
POST /surrogate/predict             — batch contact-pressure prediction
POST /surrogate/csar                — CSAR vs insertion depth (surrogate)
POST /surrogate/predict-vtp         — annotate VTP with predicted pressures
POST /surrogate/csar-from-vtp       — CSAR from VTP geometry
POST /surrogate/csar-plot-from-vtp  — CSAR plot (PNG) from VTP geometry
POST /surrogate/analyse-from-vtp    — combined CSAR + peak-pressure plot (user-facing)
"""
from __future__ import annotations

import base64
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from digital_twin_ui.app.api.schemas.surrogate import (
    AnalyseContactRequest,
    AnalyseContactResponse,
    BandCSARSeries,
    BandSummary,
    CSARFromVTPRequest,
    CSARFromVTPResponse,
    CSARPlotFromVTPRequest,
    CSARPlotFromVTPResponse,
    CSARRequest,
    CSARResponse,
    ListSurrogateModelsResponse,
    ListVTPFilesResponse,
    PredictRequest,
    PredictResponse,
    PredictVTPRequest,
    PredictVTPResponse,
    RegisteredModelInfo,
    RegisteredModelVersion,
    SurrogateModelInfo,
    VTPFileInfo,
)
from digital_twin_ui.surrogate.csar_engine import CSAREngine, build_insertion_depths, load_reference_facets
from digital_twin_ui.surrogate.predictor import (
    SurrogatePredictor,
    default_model_dir,
    is_model_available,
    list_mlflow_runs,
    list_registered_models,
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


def _get_predictor(
    run_id: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    model_stage: Optional[str] = None,
) -> SurrogatePredictor:
    """
    Return the appropriate predictor.

    Priority order:
      1. *registered_model_name* — load from MLflow Model Registry
      2. *run_id*                — load specific MLflow run artifacts
      3. default                 — cached 'latest' local model; auto-falls back
                                   to SURROGATE_REGISTRY_MODEL_NAME env var
    """
    # 1. Registry-named model (cross-VM: trained on another machine, registered centrally)
    if registered_model_name:
        try:
            return SurrogatePredictor.load_from_registry(
                model_name=registered_model_name,
                stage=model_stage or None,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not load registered model '{registered_model_name}': {exc}",
            ) from exc

    # 2. Specific run_id
    if run_id is not None:
        try:
            return SurrogatePredictor.load_from_run(run_id)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not load model for run_id={run_id}: {exc}",
            ) from exc

    # 3. Default: cached local 'latest', with env-var registry fallback
    if not is_model_available():
        registry_name = os.getenv("SURROGATE_REGISTRY_MODEL_NAME", "")
        if registry_name:
            try:
                return SurrogatePredictor.load_from_registry(model_name=registry_name)
            except Exception as exc:
                logger.warning(
                    "Registry fallback for '%s' failed: %s", registry_name, exc
                )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Surrogate model not available. "
                "Either train a model (full_pipeline notebook) or set "
                "SURROGATE_REGISTRY_MODEL_NAME to a registered model name."
            ),
        )
    return _get_predictor_cached()


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
    Return recent surrogate-lab MLflow runs, registered models, and 'latest' availability.

    - ``models``: recent training runs (sorted newest-first)
    - ``registered_models``: models registered in the MLflow Model Registry
    - ``latest_available``: True if data/surrogate/models/latest/ has all artifacts

    On a fresh VM where a model has been registered via ``mlflow.register_model()``,
    ``registered_models`` will be populated even if ``latest_available`` is False.
    Pass ``registered_model_name`` in prediction requests to use a registry model.
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

    # Also list registered models from the MLflow Model Registry
    raw_registered = list_registered_models()
    registered = []
    for rm in raw_registered:
        if "error" in rm:
            logger.warning("MLflow registry error: %s", rm["error"])
            continue
        registered.append(
            RegisteredModelInfo(
                name=rm["name"],
                description=rm.get("description", ""),
                tags=rm.get("tags", {}),
                latest_versions=[
                    RegisteredModelVersion(
                        version=str(v["version"]),
                        stage=v.get("stage", ""),
                        run_id=v.get("run_id"),
                        status=v.get("status"),
                    )
                    for v in rm.get("latest_versions", [])
                ],
            )
        )

    return ListSurrogateModelsResponse(
        models=models,
        latest_available=is_model_available(),
        registered_models=registered,
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
    predictor = _get_predictor(req.run_id, req.registered_model_name, req.model_stage)

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
    predictor = _get_predictor(req.run_id, req.registered_model_name, req.model_stage)

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
    predictor = _get_predictor(req.run_id, req.registered_model_name, req.model_stage)

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
    predictor = _get_predictor(req.run_id, req.registered_model_name, req.model_stage)

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


# ---------------------------------------------------------------------------
# CSAR plot from VTP
# ---------------------------------------------------------------------------

def _make_csar_plot(csar_df: pd.DataFrame, title: str) -> bytes:
    """
    Generate a CSAR-vs-depth plot from the computed DataFrame.

    Returns raw PNG bytes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, group in csar_df.groupby("band_label"):
        g = group.sort_values("insertion_depth_mm")
        csar_vals = [v if (v is not None and not np.isnan(v)) else np.nan for v in g["csar"]]
        ax.plot(g["insertion_depth_mm"], csar_vals, marker="o", markersize=3, label=str(label))

    ax.set_xlabel("Insertion depth [mm]", fontsize=12)
    ax.set_ylabel("CSAR (contact area fraction)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@router.post(
    "/csar-plot-from-vtp",
    response_model=CSARPlotFromVTPResponse,
    summary="Compute CSAR vs insertion depth from VTP geometry and return a plot",
)
def csar_plot_from_vtp(req: CSARPlotFromVTPRequest) -> CSARPlotFromVTPResponse:
    """
    Read facet geometry from a VTP file, compute CSAR vs insertion depth using
    the surrogate model, and return a PNG plot.

    The plot is saved to disk and also returned as a base64-encoded PNG string
    so the MCP agent can display it inline (if the client supports it).

    Z bands define which axial regions of the catheter to analyse.  Each band
    becomes a separate curve in the plot.
    """
    predictor = _get_predictor(req.run_id, req.registered_model_name, req.model_stage)

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

    # Generate plot
    plot_title = req.title or f"CSAR vs Insertion Depth — {vtp_path.stem}"
    try:
        png_bytes = _make_csar_plot(csar_df, title=plot_title)
    except Exception as exc:
        logger.exception("Plot generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plot generation failed: {exc}",
        ) from exc

    # Save PNG to disk
    if req.output_path:
        out_path = _resolve_path(req.output_path)
    else:
        plots_dir = _SURROGATE_DATA_ROOT / "results" / "csar_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        stem = vtp_path.stem
        n_bands = len(req.z_bands)
        out_path = plots_dir / f"{stem}_csar_{n_bands}bands.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)
    logger.info("Saved CSAR plot to %s", out_path)

    png_b64 = base64.b64encode(png_bytes).decode("ascii")

    return CSARPlotFromVTPResponse(
        plot_path=str(out_path),
        host_plot_path=_to_host_path(str(out_path)),
        plot_png_b64=png_b64,
        insertion_depths_mm=sorted(csar_df["insertion_depth_mm"].unique().tolist()),
        bands=_build_band_series(csar_df),
        n_facets=vtp.n_faces,
        vtp_source=str(vtp_path),
        run_id_used=req.run_id,
    )


# ---------------------------------------------------------------------------
# Combined contact analysis (CSAR + peak pressure, single 2-panel plot)
# ---------------------------------------------------------------------------

def _make_combined_plot(csar_df: pd.DataFrame, title: str) -> bytes:
    """
    Generate a 2-panel figure:
      Top: CSAR vs insertion depth per Z band
      Bottom: Peak contact pressure vs insertion depth per Z band
    Returns raw PNG bytes.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_csar, ax_cp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for label, group in csar_df.groupby("band_label"):
        g = group.sort_values("insertion_depth_mm")
        depths = g["insertion_depth_mm"].tolist()
        csar_vals = [v if (v is not None and not np.isnan(v)) else np.nan for v in g["csar"]]
        cp_vals = g["max_cp_MPa"].tolist()
        ax_csar.plot(depths, csar_vals, marker="o", markersize=3, label=str(label))
        ax_cp.plot(depths, cp_vals, marker="s", markersize=3, label=str(label))

    ax_csar.set_ylabel("CSAR (contact area fraction)", fontsize=11)
    ax_csar.set_ylim(0.0, 1.05)
    ax_csar.set_title(title, fontsize=13)
    ax_csar.grid(True, alpha=0.4)
    ax_csar.legend(loc="upper left", fontsize=9)

    ax_cp.set_xlabel("Insertion depth [mm]", fontsize=11)
    ax_cp.set_ylabel("Peak contact pressure [MPa]", fontsize=11)
    ax_cp.grid(True, alpha=0.4)
    ax_cp.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_band_summaries(csar_df: pd.DataFrame) -> dict[str, BandSummary]:
    """Extract key statistics from the CSAR DataFrame per band."""
    summaries: dict[str, BandSummary] = {}
    for label, group in csar_df.groupby("band_label"):
        g = group.sort_values("insertion_depth_mm")
        csar_vals = np.array(
            [v if (v is not None and not np.isnan(v)) else np.nan for v in g["csar"]],
            dtype=float,
        )
        cp_vals = g["max_cp_MPa"].values.astype(float)
        depths = g["insertion_depth_mm"].values

        valid_csar = ~np.isnan(csar_vals)
        peak_csar = float(np.nanmax(csar_vals)) if valid_csar.any() else None
        depth_at_peak_csar = float(depths[np.nanargmax(csar_vals)]) if valid_csar.any() else None
        peak_cp = float(np.nanmax(cp_vals)) if len(cp_vals) > 0 else None
        depth_at_peak_cp = float(depths[np.nanargmax(cp_vals)]) if len(cp_vals) > 0 else None

        # First contact: shallowest depth where CSAR > 0
        contact_mask = valid_csar & (csar_vals > 0)
        first_contact = float(depths[contact_mask][0]) if contact_mask.any() else None

        summaries[str(label)] = BandSummary(
            label=str(label),
            zmin_mm=float(g["zmin_mm"].iloc[0]),
            zmax_mm=float(g["zmax_mm"].iloc[0]),
            n_total_facets=int(g["n_total_facets"].iloc[0]),
            total_area_mm2=float(g["total_area_mm2"].iloc[0]),
            peak_csar=peak_csar,
            depth_at_peak_csar_mm=depth_at_peak_csar,
            peak_pressure_MPa=peak_cp,
            depth_at_peak_pressure_mm=depth_at_peak_cp,
            first_contact_depth_mm=first_contact,
        )
    return summaries


@router.post(
    "/analyse-from-vtp",
    response_model=AnalyseContactResponse,
    summary="Combined CSAR + peak-pressure analysis from VTP (user-facing)",
)
def analyse_contact_from_vtp(req: AnalyseContactRequest) -> AnalyseContactResponse:
    """
    One-stop contact analysis: reads a VTP file, uses the surrogate model to
    predict contact pressure across a range of insertion depths, then generates
    a two-panel PNG plot showing:
      - CSAR (contact surface area ratio) vs insertion depth per Z band
      - Peak contact pressure vs insertion depth per Z band

    The user only needs to specify Z bands (axial regions of interest) —
    no knowledge of facet centroids or mesh geometry is required.

    Z=0 is the catheter tip; larger Z values are more proximal (towards the handle).
    """
    predictor = _get_predictor(req.run_id, req.registered_model_name, req.model_stage)

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

    depths = (
        [float(d) for d in req.insertion_depths_mm]
        if req.insertion_depths_mm
        else build_insertion_depths(req.max_depth_mm, req.depth_step_mm)
    )
    z_bands_dicts = [
        {"zmin": b.zmin, "zmax": b.zmax, "label": b.effective_label()}
        for b in req.z_bands
    ]

    try:
        csar_df = compute_csar_from_vtp(vtp, predictor, depths, z_bands_dicts, req.cp_threshold)
    except Exception as exc:
        logger.exception("CSAR computation failed in analyse-from-vtp")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSAR computation failed: {exc}",
        ) from exc

    plot_title = req.title or f"Contact Analysis — {vtp_path.stem}"
    try:
        png_bytes = _make_combined_plot(csar_df, title=plot_title)
    except Exception as exc:
        logger.exception("Combined plot generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plot generation failed: {exc}",
        ) from exc

    if req.output_path:
        out_path = _resolve_path(req.output_path)
    else:
        plots_dir = _SURROGATE_DATA_ROOT / "results" / "analysis_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"{vtp_path.stem}_contact_analysis.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)
    logger.info("Saved combined contact analysis plot to %s", out_path)

    return AnalyseContactResponse(
        plot_path=str(out_path),
        host_plot_path=_to_host_path(str(out_path)),
        plot_png_b64=base64.b64encode(png_bytes).decode("ascii"),
        insertion_depths_mm=sorted(csar_df["insertion_depth_mm"].unique().tolist()),
        bands=_build_band_series(csar_df),
        band_summaries=_build_band_summaries(csar_df),
        n_facets=vtp.n_faces,
        vtp_source=str(vtp_path),
        run_id_used=req.run_id,
    )


# ---------------------------------------------------------------------------
# List available VTP files
# ---------------------------------------------------------------------------

@router.get(
    "/list-vtps",
    response_model=ListVTPFilesResponse,
    summary="List available VTP files in runs/ and surrogate_data/",
)
def list_vtp_files(max_files: int = 50) -> ListVTPFilesResponse:
    """
    Scan for VTP files in the runs directory and surrogate data directory.

    Returns a list of available VTP files that can be used as input to
    predict-vtp, csar-from-vtp, and analyse-from-vtp endpoints.

    Ordered newest-first by file modification time.
    """
    search_dirs = [_RUNS_ROOT, _SURROGATE_DATA_ROOT]
    found: list[tuple[float, Path]] = []

    for base in search_dirs:
        if base.exists():
            for vtp_file in base.rglob("*.vtp"):
                try:
                    found.append((vtp_file.stat().st_mtime, vtp_file))
                except OSError:
                    pass

    # Sort newest first, then truncate
    found.sort(key=lambda x: x[0], reverse=True)
    found = found[:max_files]

    files = [
        VTPFileInfo(
            path=str(p),
            host_path=_to_host_path(str(p)),
            size_kb=round(p.stat().st_size / 1024, 1),
            stem=p.stem,
        )
        for _, p in found
    ]

    return ListVTPFilesResponse(
        vtp_files=files,
        total=len(files),
        search_dirs=[str(d) for d in search_dirs],
    )
