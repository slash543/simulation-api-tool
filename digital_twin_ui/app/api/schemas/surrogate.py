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


class ListSurrogateModelsResponse(BaseModel):
    models: list[SurrogateModelInfo]
    latest_available: bool = Field(
        ...,
        description="True if data/surrogate/models/latest/ has all required artifacts.",
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
    cp_threshold: float = 0.0


class CSARFromVTPResponse(BaseModel):
    insertion_depths_mm: list[float]
    bands: dict[str, BandCSARSeries]
    n_facets: int
    vtp_source: str
    run_id_used: Optional[str] = None
