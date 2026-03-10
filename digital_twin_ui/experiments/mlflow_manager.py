"""
MLflow experiment and run helpers for the Digital Twin UI platform.

Provides a thin, testable wrapper around MLflow that:
- Creates or retrieves an experiment by name
- Starts / ends runs with structured parameter and metric logging
- Logs artefacts (model checkpoints, CSV extractions, config snapshots)
- Offers a context-manager interface for RAII-style run management

Usage::

    from digital_twin_ui.experiments.mlflow_manager import MLflowManager

    mgr = MLflowManager()                          # uses settings from config
    with mgr.start_run(run_name="speed_5.0") as run:
        run.log_params({"speed_mm_s": 5.0})
        run.log_metrics({"max_pressure": 1.23}, step=0)
        run.log_artifact(Path("runs/run_001/extraction.csv"))
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Generator, Optional

import mlflow
from mlflow.entities import Run

from digital_twin_ui.app.core.config import get_settings
from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Run context object
# ---------------------------------------------------------------------------

class ActiveRun:
    """Thin wrapper around an active MLflow run with convenience helpers."""

    def __init__(self, mlflow_run: Run, active_run_ctx: Any) -> None:
        self._run = mlflow_run
        self._ctx = active_run_ctx  # mlflow.start_run context handle

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    @property
    def run_name(self) -> str:
        return self._run.info.run_name or ""

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None) -> None:
        """Log a single file as an MLflow artefact."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_artifacts(self, local_dir: Path, artifact_path: Optional[str] = None) -> None:
        """Log all files in a directory as MLflow artefacts."""
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def end(self, status: str = "FINISHED") -> None:
        """Explicitly end the run.  Prefer the context manager instead."""
        mlflow.end_run(status=status)

    def __repr__(self) -> str:
        return f"ActiveRun(run_id={self.run_id!r}, name={self.run_name!r})"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MLflowManager:
    """
    Facade over MLflow for the Digital Twin UI platform.

    Args:
        tracking_uri: MLflow tracking server URI.  Defaults to value in
                      ``config/simulation.yaml`` (``mlflow.tracking_uri``).
        experiment_name: Experiment name to create/retrieve.  Defaults to
                         YAML config value (``mlflow.experiment_name``).
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        cfg = get_settings()
        self._tracking_uri = tracking_uri or cfg.mlflow_tracking_uri_abs
        self._experiment_name = experiment_name or cfg.mlflow.experiment_name

        mlflow.set_tracking_uri(self._tracking_uri)
        self._experiment_id = self._get_or_create_experiment()
        logger.debug(
            "MLflowManager initialised",
            tracking_uri=self._tracking_uri,
            experiment=self._experiment_name,
            experiment_id=self._experiment_id,
        )

    # ------------------------------------------------------------------
    # Experiment helpers
    # ------------------------------------------------------------------

    @property
    def tracking_uri(self) -> str:
        return self._tracking_uri

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    def _get_or_create_experiment(self) -> str:
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self._experiment_name)
            logger.info("Created MLflow experiment", name=self._experiment_name, id=experiment_id)
        else:
            experiment_id = experiment.experiment_id
            logger.debug("Using existing MLflow experiment", name=self._experiment_name, id=experiment_id)
        return experiment_id

    # ------------------------------------------------------------------
    # Run management — context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        nested: bool = False,
    ) -> Generator[ActiveRun, None, None]:
        """
        Context manager that starts an MLflow run and yields an :class:`ActiveRun`.

        On successful exit the run is marked FINISHED; on exception it is FAILED.

        Example::

            with mgr.start_run(run_name="run_001") as run:
                run.log_params({"speed_mm_s": 5.0})
                run.log_metric("max_pressure", 1.23)
        """
        logger.info("Starting MLflow run", run_name=run_name, experiment=self._experiment_name)
        ctx = mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            tags=tags,
            nested=nested,
        )
        mlflow_run = ctx.__enter__()
        active = ActiveRun(mlflow_run, ctx)
        try:
            yield active
        except Exception:
            mlflow.end_run(status="FAILED")
            logger.error("MLflow run failed", run_id=active.run_id)
            raise
        else:
            mlflow.end_run(status="FINISHED")
            logger.info("MLflow run finished", run_id=active.run_id)

    # ------------------------------------------------------------------
    # Simulation-specific convenience helpers
    # ------------------------------------------------------------------

    def log_simulation_run(
        self,
        run_name: str,
        speed_mm_s: float,
        max_pressure: float,
        mean_pressures: list[float],
        times: list[float],
        *,
        artefact_path: Optional[Path] = None,
        extra_params: Optional[dict[str, Any]] = None,
        extra_metrics: Optional[dict[str, float]] = None,
    ) -> str:
        """
        Log a single catheter simulation result as one MLflow run.

        Args:
            run_name: Human-readable run identifier.
            speed_mm_s: Insertion speed used in this simulation.
            max_pressure: Peak contact pressure across all time steps.
            mean_pressures: Per-time-step mean contact pressure list.
            times: Simulation time values (same length as ``mean_pressures``).
            artefact_path: Optional CSV / Parquet file to attach.
            extra_params: Additional parameters to log.
            extra_metrics: Additional scalar metrics to log.

        Returns:
            The MLflow run ID string.
        """
        with self.start_run(run_name=run_name) as run:
            params: dict[str, Any] = {"speed_mm_s": speed_mm_s}
            if extra_params:
                params.update(extra_params)
            run.log_params(params)

            run.log_metric("max_pressure_kpa", max_pressure)
            if extra_metrics:
                run.log_metrics(extra_metrics)

            # Log time series as steps
            for step, (t, mp) in enumerate(zip(times, mean_pressures)):
                run.log_metric("mean_pressure_kpa", mp, step=step)
                run.log_metric("time_s", t, step=step)

            if artefact_path is not None and artefact_path.exists():
                run.log_artifact(artefact_path)

            return run.run_id

    def log_training_run(
        self,
        run_name: str,
        hyperparams: dict[str, Any],
        train_loss: list[float],
        val_loss: list[float],
        *,
        model_path: Optional[Path] = None,
        extra_metrics: Optional[dict[str, float]] = None,
    ) -> str:
        """
        Log a model-training result as one MLflow run.

        Args:
            run_name: Human-readable run identifier.
            hyperparams: Hyper-parameter dict (learning_rate, hidden_dims, …).
            train_loss: Training loss per epoch.
            val_loss: Validation loss per epoch.
            model_path: Optional path to saved model checkpoint.
            extra_metrics: Additional final metrics to log.

        Returns:
            The MLflow run ID string.
        """
        with self.start_run(run_name=run_name) as run:
            run.log_params(hyperparams)

            for epoch, (tl, vl) in enumerate(zip(train_loss, val_loss)):
                run.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)

            if extra_metrics:
                run.log_metrics(extra_metrics)

            if model_path is not None and model_path.exists():
                run.log_artifact(model_path, artifact_path="model")

            return run.run_id

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> Run:
        """Fetch a completed run by ID."""
        client = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri)
        return client.get_run(run_id)

    def list_runs(self, max_results: int = 100) -> list[Run]:
        """Return recent runs for the current experiment, newest first."""
        client = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri)
        runs = client.search_runs(
            experiment_ids=[self._experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        return list(runs)


# ---------------------------------------------------------------------------
# Module-level singleton (optional convenience)
# ---------------------------------------------------------------------------

_manager: Optional[MLflowManager] = None


def get_mlflow_manager() -> MLflowManager:
    """Return (creating if necessary) a module-level singleton manager."""
    global _manager
    if _manager is None:
        _manager = MLflowManager()
    return _manager
