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


class CancelRequest(BaseModel):
    """Request body to cancel a running simulation."""

    run_id: str = Field(
        ...,
        description="Run identifier returned when the simulation was submitted.",
        examples=["run_20260310_143000_a1b2"],
    )
    task_id: Optional[str] = Field(
        None,
        description=(
            "Celery task ID returned when the simulation was submitted. "
            "Used to also prevent the task from starting if still queued."
        ),
    )


class CancelResponse(BaseModel):
    """Response for a simulation cancellation request."""

    run_id: str
    task_id: Optional[str] = None
    status: str = "CANCELLATION_REQUESTED"
    message: str


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


# ---------------------------------------------------------------------------
# Catheter catalogue schemas  (design → configuration → speeds)
# ---------------------------------------------------------------------------

class CatalogueConfigEntry(BaseModel):
    """One size × urethra-model configuration for a catheter design."""

    key: str = Field(..., description="Config identifier, e.g. '14Fr_IR12'.")
    label: str = Field(..., description="Human-readable label shown to the user.")
    feb_file: str = Field(..., description="FEB filename in base_configuration/.")


class CatalogueDesignEntry(BaseModel):
    """One catheter tip design with all its available configurations."""

    name: str = Field(..., description="Design key, e.g. 'ball_tip'.")
    label: str = Field(..., description="Human-readable name, e.g. 'Ball Tip'.")
    configurations: list[CatalogueConfigEntry] = Field(
        ...,
        description="Available size × urethra-model combinations.",
    )


class CatalogueListResponse(BaseModel):
    """All catheter designs with their configurations and shared sim params."""

    designs: list[CatalogueDesignEntry]
    n_steps: int = Field(..., description="Number of insertion steps (same for all designs).")
    displacements_mm: list[float] = Field(
        ..., description="Per-step displacement magnitudes in mm."
    )
    speed_range_min: float = Field(..., description="Minimum valid speed in mm/s.")
    speed_range_max: float = Field(..., description="Maximum valid speed in mm/s.")
    default_uniform_speed_mm_s: float = Field(
        ..., description="Default speed for uniform profiles."
    )
    default_dwell_time_s: float = Field(
        ..., description="Default dwell time appended after each ramp, in seconds."
    )


class CatheterSimRequest(BaseModel):
    """Request body for a catheter simulation (design + configuration + per-step speeds)."""

    design: str = Field(
        ...,
        description=(
            "Catheter tip design key from GET /catheter-designs "
            "(e.g. 'ball_tip', 'nelaton_tip', 'vapro_introducer')."
        ),
        examples=["ball_tip"],
    )
    configuration: str = Field(
        ...,
        description=(
            "Size × urethra-model configuration key from the chosen design "
            "(e.g. '14Fr_IR12', '14Fr_IR25', '16Fr_IR12')."
        ),
        examples=["14Fr_IR12"],
    )
    speeds_mm_s: list[float] = Field(
        ...,
        min_length=1,
        description=(
            "Per-step insertion speeds in mm/s. "
            "Length must equal n_steps (10 for all current designs). "
            "Each value controls the ramp duration: ramp_i = displacement_mm[i] / speeds_mm_s[i]. "
            "If the user wants a uniform profile, provide the same value for all steps."
        ),
        examples=[[15.0, 15.0, 14.0, 14.0, 13.0, 13.0, 12.0, 12.0, 11.0, 11.0]],
    )
    dwell_time_s: float = Field(
        1.0,
        gt=0,
        description="Dwell time in seconds appended after each ramp (default 1.0 s).",
    )
    run_id: Optional[str] = Field(
        None,
        description="Optional explicit run identifier. Auto-generated if omitted.",
    )


class DOESpeedPreviewRequest(BaseModel):
    """Request body for previewing DOE speed samples without running simulations."""

    n_samples: int = Field(
        10,
        ge=1,
        le=200,
        description="Number of speed-array samples to generate.",
    )
    speed_min: float = Field(
        10.0,
        gt=0,
        description="Minimum mean speed in mm/s.",
    )
    speed_max: float = Field(
        25.0,
        gt=0,
        description="Maximum mean speed in mm/s.",
    )
    n_steps: int = Field(
        10,
        ge=1,
        description="Number of steps per sample (length of each speed array).",
    )
    max_perturbation: float = Field(
        0.20,
        ge=0.0,
        le=0.5,
        description="Maximum fractional per-step perturbation (0.20 = ±20%).",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional RNG seed for reproducibility.",
    )


class DOESpeedPreviewResponse(BaseModel):
    """Preview of DOE speed arrays (no simulations run)."""

    n_samples: int
    n_steps: int
    speed_min: float
    speed_max: float
    samples: list[list[float]] = Field(
        ...,
        description=(
            "List of n_samples speed arrays, each of length n_steps. "
            "Each array is sorted ascending within the allowed range."
        ),
    )
