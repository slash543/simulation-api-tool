"""
Pydantic request / response schemas for the simulation API endpoints.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Request body for a single simulation run."""

    speed_mm_s: float = Field(
        5.0,
        gt=0,
        description=(
            "Catheter insertion speed in mm/s (e.g. 5.0). "
            "Used for single-step templates (sample_catheterization). "
            "Ignored when speeds_mm_s is provided."
        ),
        examples=[5.0],
    )
    template: str = Field(
        "sample_catheterization",
        description=(
            "Template name to use for this simulation. "
            "Use GET /templates to list available templates."
        ),
        examples=["sample_catheterization", "DT_BT_14Fr_FO_10E_IR12"],
    )
    speeds_mm_s: Optional[list[float]] = Field(
        None,
        description=(
            "Per-step insertion speeds in mm/s for multi-step templates. "
            "Length must equal the template's n_steps (e.g. 10). "
            "If omitted for a multi-step template, speed_mm_s is broadcast "
            "to all steps."
        ),
    )
    dwell_time_s: float = Field(
        1.0,
        gt=0,
        description="Dwell time appended after each insertion ramp, in seconds.",
    )
    run_id: Optional[str] = Field(
        None,
        description="Optional explicit run identifier. Auto-generated if omitted.",
        examples=["run_20260310_143000_a1b2"],
    )
    extract: bool = Field(
        True,
        description="Whether to extract contact pressure after the simulation.",
    )
    log_mlflow: bool = Field(
        False,
        description="Whether to log the result to MLflow.",
    )


class DOERequest(BaseModel):
    """Request body for a Design-of-Experiments campaign."""

    n_samples: int = Field(
        10,
        ge=1,
        le=500,
        description="Number of simulation samples.",
    )
    speed_min: float = Field(
        10.0,
        gt=0,
        description="Minimum insertion speed in mm/s.",
    )
    speed_max: float = Field(
        25.0,
        gt=0,
        description="Maximum insertion speed in mm/s.",
    )
    sampler: str = Field(
        "lhs",
        description="Sampling strategy: 'lhs', 'sobol', or 'uniform'.",
        pattern="^(lhs|sobol|uniform)$",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional random seed for reproducibility.",
    )
    extract: bool = Field(
        True,
        description="Run xplt extraction for each simulation.",
    )
    log_mlflow: bool = Field(
        False,
        description="Log each run to MLflow.",
    )
    template: str = Field(
        "DT_BT_14Fr_FO_10E_IR12",
        description=(
            "Template name to use for all simulations in this campaign. "
            "Use GET /templates to list available templates."
        ),
    )
    max_perturbation: float = Field(
        0.20,
        ge=0.0,
        le=0.5,
        description=(
            "Maximum fractional perturbation applied to per-step speeds "
            "relative to the mean speed (used by CorrelatedSpeedSampler). "
            "E.g. 0.20 means ±20%."
        ),
    )
    dwell_time_s: float = Field(
        1.0,
        gt=0,
        description="Dwell time appended after each insertion ramp, in seconds.",
    )


class ExtractionRequest(BaseModel):
    """Request body for post-hoc xplt extraction."""

    xplt_path: str = Field(
        ...,
        description="Absolute path to the .xplt result file.",
    )
    variable_name: str = Field(
        "contact pressure",
        description="Name of the surface variable to extract.",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TaskResponse(BaseModel):
    """Returned immediately when an async task is accepted."""

    task_id: str = Field(..., description="Celery task identifier.")
    status: str = Field("PENDING", description="Initial task status.")
    message: str = Field("", description="Human-readable status message.")
    run_id: Optional[str] = Field(
        None,
        description="Run identifier — also the name of the result folder.",
    )
    run_dir: Optional[str] = Field(
        None,
        description="Absolute path to the run directory inside the container.",
    )
    xplt_path: Optional[str] = Field(
        None,
        description="Expected path to the .xplt results file (written by solver).",
    )


class TaskStatusResponse(BaseModel):
    """Current state of a Celery task."""

    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


class SimulationResultResponse(BaseModel):
    """Result of a completed simulation run."""

    run_id: str
    status: str
    speed_mm_s: float
    duration_s: Optional[float] = None
    error_message: Optional[str] = None
    extraction: Optional[dict[str, Any]] = None
    mlflow_run_id: Optional[str] = None
    run_dir: Optional[str] = Field(
        None,
        description="Absolute path to the run directory inside the container.",
    )
    xplt_path: Optional[str] = Field(
        None,
        description="Absolute path to the .xplt results file inside the container.",
    )


class PressureResultResponse(BaseModel):
    """Extracted contact pressure data."""

    variable_name: str
    times: list[float]
    max_pressure: float
    mean_pressure: list[float]
    pressures: Optional[list[list[float]]] = None
    n_faces: int
    source_path: Optional[str] = None


class DOEResultResponse(BaseModel):
    """Summary of a completed DOE campaign."""

    n_samples: int
    n_completed: int
    samples: list[dict[str, Any]]


class MLPredictRequest(BaseModel):
    """Request body for ML prediction."""

    speed_mm_s: float = Field(
        ...,
        gt=0,
        description="Catheter insertion speed in mm/s.",
    )


class MLPredictResponse(BaseModel):
    """ML prediction response."""

    speed_mm_s: float
    predicted_max_pressure: float


class MLPredictBatchRequest(BaseModel):
    """Batch ML prediction request."""

    speeds_mm_s: list[float] = Field(
        ...,
        min_length=1,
        description="List of insertion speeds in mm/s.",
    )


class MLPredictBatchResponse(BaseModel):
    """Batch ML prediction response."""

    predictions: list[MLPredictResponse]


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str = "0.1.0"


class TemplateInfo(BaseModel):
    """Summary information about one simulation template."""

    name: str = Field(..., description="Unique template identifier.")
    label: str = Field(..., description="Human-readable display name.")
    n_steps: int = Field(..., description="Number of insertion steps.")
    speed_range_min: float = Field(..., description="Minimum valid speed in mm/s.")
    speed_range_max: float = Field(..., description="Maximum valid speed in mm/s.")
    displacements_mm: list[float] = Field(
        ...,
        description="Prescribed displacement for each step in mm.",
    )


class TemplateListResponse(BaseModel):
    """Response containing all available simulation templates."""

    templates: list[TemplateInfo]
