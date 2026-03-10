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
        ...,
        gt=0,
        description="Catheter insertion speed in mm/s (e.g. 5.0).",
        examples=[5.0],
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
        4.0,
        gt=0,
        description="Minimum insertion speed in mm/s.",
    )
    speed_max: float = Field(
        6.0,
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
