"""
Pydantic request / response schemas for the surrogate-model API endpoints.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class ZBand(BaseModel):
    """A Z-axis band defining a region of interest on the catheter surface."""

    zmin: float = Field(..., description="Minimum Z coordinate of the band [mm]")
    zmax: float = Field(..., description="Maximum Z coordinate of the band [mm]")
    label: str = Field(
        default="",
        description="Optional human-readable label (e.g. 'distal tip'). "
        "Auto-generated from zmin/zmax if empty.",
    )

    def effective_label(self) -> str:
        return self.label or f"z[{self.zmin:.0f},{self.zmax:.0f}]"


class FacetRow(BaseModel):
    """A single facet with geometry features for surrogate prediction."""

    centroid_x: float = Field(..., description="Facet centroid X coordinate [mm]")
    centroid_y: float = Field(..., description="Facet centroid Y coordinate [mm]")
    centroid_z: float = Field(..., description="Facet centroid Z coordinate [mm]")
    facet_area: float = Field(..., description="Facet surface area [mm²]")
    insertion_depth: float = Field(..., description="Catheter insertion depth [mm]")


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

class SurrogateModelInfo(BaseModel):
    run_id: str
    status: str
    start_time: Optional[int] = None
    metrics: dict[str, float] = {}
    params: dict[str, str] = {}
    artifact_uri: Optional[str] = None


class RegisteredModelVersion(BaseModel):
    """A single version entry from the MLflow Model Registry."""
    version: str
    stage: str
    run_id: Optional[str] = None
    status: Optional[str] = None


class RegisteredModelInfo(BaseModel):
    """A registered model and its latest versions from the MLflow registry."""
    name: str
    description: str = ""
    tags: dict[str, str] = {}
    latest_versions: list[RegisteredModelVersion] = []


class ListSurrogateModelsResponse(BaseModel):
    models: list[SurrogateModelInfo]
    latest_available: bool = Field(
        ...,
        description="True if data/surrogate/models/latest/ has all required artifacts.",
    )
    registered_models: list[RegisteredModelInfo] = Field(
        default_factory=list,
        description="Models registered in the MLflow Model Registry.",
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Batch contact-pressure prediction request."""

    facets: Optional[list[FacetRow]] = Field(
        None,
        description="Inline facets to predict. Provide either this or facets_csv_path.",
    )
    facets_csv_path: Optional[str] = Field(
        None,
        description="Host-visible path to a CSV file with facet geometry + insertion_depth. "
        "Container path: if the CSV is in data/surrogate/ it is accessible at "
        "/app/surrogate_data/.",
    )
    run_id: Optional[str] = Field(
        None,
        description="MLflow run ID to use. If None, loads the 'latest' trained model.",
    )
    registered_model_name: Optional[str] = Field(
        None,
        description="MLflow registered model name (e.g. 'CatheterCSARSurrogate'). "
        "If set, loads the latest registered version instead of run_id or local 'latest'.",
    )
    model_stage: Optional[str] = Field(
        None,
        description="Registry stage to load: 'Production', 'Staging', or None for latest version.",
    )


class PredictResponse(BaseModel):
    n_facets: int
    contact_pressure_MPa: list[float]
    run_id_used: Optional[str] = None
    model_dir: str


# ---------------------------------------------------------------------------
# CSAR computation
# ---------------------------------------------------------------------------

class CSARRequest(BaseModel):
    """Compute CSAR vs insertion depth using surrogate model predictions."""

    z_bands: list[ZBand] = Field(
        ...,
        description="Z-axis bands defining surface regions to analyse.",
        min_length=1,
    )
    insertion_depths_mm: Optional[list[float]] = Field(
        None,
        description="Insertion depth sample points [mm]. "
        "If omitted, a uniform grid from 0 to max_depth_mm is used.",
    )
    max_depth_mm: float = Field(
        300.0,
        description="Upper bound for auto-generated depth grid [mm]. "
        "Used only when insertion_depths_mm is omitted.",
    )
    depth_step_mm: float = Field(
        5.0,
        description="Step size for auto-generated depth grid [mm].",
    )
    facets_csv_path: Optional[str] = Field(
        None,
        description="Path to reference facets CSV (inside the container). "
        "If omitted, uses data/surrogate/training/reference_facets.csv.",
    )
    run_id: Optional[str] = Field(
        None,
        description="MLflow run ID. None = latest trained model.",
    )
    registered_model_name: Optional[str] = Field(
        None,
        description="MLflow registered model name. Overrides run_id when set.",
    )
    model_stage: Optional[str] = Field(
        None,
        description="Registry stage: 'Production', 'Staging', or None for latest.",
    )
    cp_threshold: float = Field(
        0.0,
        description="Predicted cp threshold [MPa] for 'in contact' classification.",
    )


class BandCSARSeries(BaseModel):
    label: str
    zmin_mm: float
    zmax_mm: float
    insertion_depths_mm: list[float]
    csar: list[Optional[float]]
    contact_area_mm2: list[float]
    total_area_mm2: float
    n_contact_facets: list[int]
    n_total_facets: int
    mean_cp_MPa: list[float]
    max_cp_MPa: list[float]


class CSARResponse(BaseModel):
    insertion_depths_mm: list[float]
    bands: dict[str, BandCSARSeries]
    n_facets_total: int
    run_id_used: Optional[str] = None
    facets_source: str


# ---------------------------------------------------------------------------
# VTP prediction
# ---------------------------------------------------------------------------

class PredictVTPRequest(BaseModel):
    """Annotate a VTP file with surrogate-predicted contact pressures."""

    vtp_path: str = Field(
        ...,
        description="Path to the input VTP file. "
        "Container path under /app/surrogate_data/ or /app/runs/.",
    )
    insertion_depth_mm: float = Field(
        ...,
        description="Catheter insertion depth [mm] at which to predict.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Path for the output VTP file. "
        "If omitted, saves next to the input file with '_predicted' suffix.",
    )
    run_id: Optional[str] = Field(
        None,
        description="MLflow run ID. None = latest trained model.",
    )
    registered_model_name: Optional[str] = Field(
        None,
        description="MLflow registered model name. Overrides run_id when set.",
    )
    model_stage: Optional[str] = Field(
        None,
        description="Registry stage: 'Production', 'Staging', or None for latest.",
    )


class PredictVTPResponse(BaseModel):
    output_vtp_path: str
    host_output_path: str
    n_faces: int
    insertion_depth_mm: float
    run_id_used: Optional[str] = None


# ---------------------------------------------------------------------------
# CSAR from VTP
# ---------------------------------------------------------------------------

class CSARFromVTPRequest(BaseModel):
    """Compute CSAR vs insertion depth using geometry from a VTP file."""

    vtp_path: str = Field(
        ...,
        description="Container path to the VTP file (geometry source).",
    )
    z_bands: list[ZBand] = Field(..., min_length=1)
    insertion_depths_mm: Optional[list[float]] = None
    max_depth_mm: float = 300.0
    depth_step_mm: float = 5.0
    run_id: Optional[str] = None
    registered_model_name: Optional[str] = Field(
        None,
        description="MLflow registered model name. Overrides run_id when set.",
    )
    model_stage: Optional[str] = None
    cp_threshold: float = 0.0


class CSARFromVTPResponse(BaseModel):
    insertion_depths_mm: list[float]
    bands: dict[str, BandCSARSeries]
    n_facets: int
    vtp_source: str
    run_id_used: Optional[str] = None


# ---------------------------------------------------------------------------
# CSAR plot from VTP
# ---------------------------------------------------------------------------

class CSARPlotFromVTPRequest(BaseModel):
    """Compute CSAR vs insertion depth from a VTP file and generate a plot."""

    vtp_path: str = Field(
        ...,
        description="Container path to the VTP file (geometry source).",
    )
    z_bands: list[ZBand] = Field(
        ...,
        min_length=1,
        description="Z-axis bands to plot. Each band becomes a separate curve.",
    )
    insertion_depths_mm: Optional[list[float]] = Field(
        None,
        description="Depth sample points [mm]. None = auto grid.",
    )
    max_depth_mm: float = Field(300.0, description="Upper bound for auto depth grid [mm].")
    depth_step_mm: float = Field(5.0, description="Step for auto depth grid [mm].")
    run_id: Optional[str] = Field(None, description="MLflow run ID. None = latest model.")
    registered_model_name: Optional[str] = Field(
        None,
        description="MLflow registered model name. Overrides run_id when set.",
    )
    model_stage: Optional[str] = Field(
        None,
        description="Registry stage: 'Production', 'Staging', or None for latest.",
    )
    cp_threshold: float = Field(0.0, description="Contact threshold [MPa].")
    output_path: Optional[str] = Field(
        None,
        description="Where to save the PNG. Defaults to surrogate_data/results/csar_plots/.",
    )
    title: Optional[str] = Field(None, description="Optional plot title override.")


class CSARPlotFromVTPResponse(BaseModel):
    plot_path: str = Field(..., description="Container path to the saved PNG file.")
    host_plot_path: str = Field(..., description="Host-visible path to the PNG file.")
    plot_png_b64: str = Field(..., description="Base64-encoded PNG for inline display.")
    insertion_depths_mm: list[float]
    bands: dict[str, BandCSARSeries]
    n_facets: int
    vtp_source: str
    run_id_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Combined contact analysis (CSAR + peak pressure, one call)
# ---------------------------------------------------------------------------

class AnalyseContactRequest(BaseModel):
    """
    One-shot combined analysis: CSAR + peak-pressure vs insertion depth from a VTP.

    Generates a two-panel plot (CSAR per band, peak pressure per band) and
    returns a concise statistics summary — no knowledge of facets/centroids needed.
    """

    vtp_path: str = Field(
        ...,
        description="Container path to the VTP file (geometry source). "
        "Files in /app/runs/ or /app/surrogate_data/ are accessible.",
    )
    z_bands: list[ZBand] = Field(
        ...,
        min_length=1,
        description=(
            "Axial regions to analyse (Z = catheter depth in mm from tip). "
            "Z=0 is the distal tip; larger Z values are more proximal. "
            "Example: [{\"zmin\": 0, \"zmax\": 50, \"label\": \"tip\"}]"
        ),
    )
    insertion_depths_mm: Optional[list[float]] = Field(
        None,
        description="Specific insertion depth sample points [mm]. "
        "None = auto grid from 0 to max_depth_mm.",
    )
    max_depth_mm: float = Field(300.0, description="Upper bound for auto depth grid [mm].")
    depth_step_mm: float = Field(5.0, description="Step size for auto depth grid [mm].")
    run_id: Optional[str] = Field(None, description="MLflow run ID. None = latest model.")
    registered_model_name: Optional[str] = Field(
        None,
        description="MLflow registered model name. Overrides run_id when set.",
    )
    model_stage: Optional[str] = Field(
        None,
        description="Registry stage: 'Production', 'Staging', or None for latest.",
    )
    cp_threshold: float = Field(0.0, description="Threshold [MPa] for 'in contact'.")
    output_path: Optional[str] = Field(None, description="Optional output PNG path.")
    title: Optional[str] = Field(None, description="Optional plot title.")


class BandSummary(BaseModel):
    """Key statistics for a single Z band."""

    label: str
    zmin_mm: float
    zmax_mm: float
    n_total_facets: int
    total_area_mm2: float
    peak_csar: Optional[float] = Field(None, description="Maximum CSAR across all depths.")
    depth_at_peak_csar_mm: Optional[float] = Field(None, description="Depth at peak CSAR.")
    peak_pressure_MPa: Optional[float] = Field(None, description="Maximum predicted pressure.")
    depth_at_peak_pressure_mm: Optional[float] = Field(None, description="Depth at peak pressure.")
    first_contact_depth_mm: Optional[float] = Field(
        None, description="Shallowest depth with any contact (CSAR > 0)."
    )


class AnalyseContactResponse(BaseModel):
    plot_path: str = Field(..., description="Container path to the saved PNG file.")
    host_plot_path: str = Field(..., description="Host-visible path to the PNG file.")
    plot_png_b64: str = Field(..., description="Base64-encoded PNG for inline display.")
    insertion_depths_mm: list[float]
    bands: dict[str, BandCSARSeries]
    band_summaries: dict[str, BandSummary] = Field(
        ..., description="Concise per-band statistics (peaks, first-contact depth)."
    )
    n_facets: int
    vtp_source: str
    run_id_used: Optional[str] = None


# ---------------------------------------------------------------------------
# List VTP files
# ---------------------------------------------------------------------------

class VTPFileInfo(BaseModel):
    path: str = Field(..., description="Container path to the VTP file.")
    host_path: str = Field(..., description="Host-visible path to the VTP file.")
    size_kb: float
    stem: str


class ListVTPFilesResponse(BaseModel):
    vtp_files: list[VTPFileInfo]
    total: int
    search_dirs: list[str]
