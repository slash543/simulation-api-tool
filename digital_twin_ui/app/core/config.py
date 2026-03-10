"""
Centralised configuration for the Digital Twin UI platform.

Loads config/simulation.yaml and exposes a validated Settings object.
Environment variables with the prefix DTUI__ override YAML values,
allowing Docker / CI overrides without changing the YAML file.

Example override:
    DTUI__SIMULATION__SIMULATOR_EXECUTABLE=febio4-avx2 python -m ...
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models (one per top-level YAML section)
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    base_feb_path: Path = Path("templates/sample_catheterization.feb")
    simulator_executable: str = "febio4"   # invoked as: febio4 -i <input.feb>
    simulator_args: list[str] = Field(default_factory=lambda: ["-i"])
    runs_dir: Path = Path("runs")
    timeout_seconds: int = 3600

    displacement_mm: float = 10.0
    num_steps: int = 2
    step1_duration_s: float = 2.0
    default_step_size: float = 0.05
    insertion_step_id: int = 2

    loadcurve_start_time: float = 2.0
    loadcurve_id: int = 1

    @field_validator("base_feb_path", "runs_dir", mode="before")
    @classmethod
    def _to_path(cls, v: Any) -> Path:
        return Path(v)


class DOEConfig(BaseModel):
    speed_min_mm_s: float = 10.0
    speed_max_mm_s: float = 25.0
    default_sampler: str = "lhs"
    default_num_samples: int = 10
    max_perturbation: float = 0.20
    default_dwell_time_s: float = 1.0
    default_template: str = "DT_BT_14Fr_FO_10E_IR12"


class SQLConfig(BaseModel):
    connection_string: str = "sqlite:///data/synthetic_db.sqlite"


class MLflowConfig(BaseModel):
    tracking_uri: str = "mlruns"
    experiment_name: str = "catheter_insertion"


class MLConfig(BaseModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 128, 256])
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 500
    patience: int = 20
    val_fraction: float = 0.2
    checkpoint_dir: Path = Path("models")
    dataset_path: Path = Path("data/datasets/catheter_dataset.parquet")

    @field_validator("checkpoint_dir", "dataset_path", mode="before")
    @classmethod
    def _to_path(cls, v: Any) -> Path:
        return Path(v)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


class CeleryConfig(BaseModel):
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"


class LoggingConfig(BaseModel):
    level: str = "DEBUG"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{name}:{function}:{line} - {message}"
    )
    rotation: str = "10 MB"
    retention: str = "1 week"
    log_dir: Path = Path("logs")

    @field_validator("log_dir", mode="before")
    @classmethod
    def _to_path(cls, v: Any) -> Path:
        return Path(v)


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    """Root settings object populated from YAML + env overrides."""

    project_root: Path = Field(default_factory=lambda: _find_project_root())

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    doe: DOEConfig = Field(default_factory=DOEConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sql: SQLConfig = Field(default_factory=SQLConfig)

    # ------------------------------------------------------------------
    # Derived absolute paths (computed properties)
    # ------------------------------------------------------------------

    @property
    def base_feb_path_abs(self) -> Path:
        """Absolute path to the base FEB template."""
        p = self.simulation.base_feb_path
        return p if p.is_absolute() else self.project_root / p

    @property
    def runs_dir_abs(self) -> Path:
        """Absolute path to the simulation runs directory."""
        p = self.simulation.runs_dir
        return p if p.is_absolute() else self.project_root / p

    @property
    def log_dir_abs(self) -> Path:
        p = self.logging.log_dir
        return p if p.is_absolute() else self.project_root / p

    @property
    def dataset_path_abs(self) -> Path:
        p = self.ml.dataset_path
        return p if p.is_absolute() else self.project_root / p

    @property
    def checkpoint_dir_abs(self) -> Path:
        p = self.ml.checkpoint_dir
        return p if p.is_absolute() else self.project_root / p

    @property
    def mlflow_tracking_uri_abs(self) -> str:
        uri = self.mlflow.tracking_uri
        if uri.startswith(("http://", "https://", "file://")):
            return uri
        p = Path(uri)
        return str(p if p.is_absolute() else self.project_root / p)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml or config/."""
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "pyproject.toml").exists() or (parent / "config").is_dir():
            return parent
    return Path.cwd()


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """
    Apply DTUI__SECTION__KEY=value environment variable overrides.

    Example:
        DTUI__SIMULATION__SIMULATOR_EXECUTABLE=febio4-avx2
        DTUI__API__PORT=9000
    """
    prefix = "DTUI__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__", 1)
        if len(parts) != 2:
            continue
        section, field = parts
        if section in data and isinstance(data[section], dict):
            data[section][field] = value
    return data


def load_settings(config_path: Path | None = None) -> Settings:
    """
    Load settings from YAML file with optional env-var overrides.

    Args:
        config_path: Explicit path to YAML config. Defaults to
                     <project_root>/config/simulation.yaml.

    Returns:
        Validated Settings instance.
    """
    root = _find_project_root()

    if config_path is None:
        config_path = root / "config" / "simulation.yaml"

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
    else:
        raw = {}

    raw = _apply_env_overrides(raw)
    return Settings(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Module-level cached settings singleton.

    Use this throughout the application:
        from digital_twin_ui.app.core.config import get_settings
        cfg = get_settings()
    """
    return load_settings()
