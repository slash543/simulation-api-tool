"""
Surrogate model predictor for contact-pressure estimation.

Loads a trained MLP model from the shared artifact directory
(``data/surrogate/models/latest/``) and runs batched inference.

The artifact layout expected on disk::

    data/surrogate/models/latest/
        best_model.pt      # PyTorch state-dict (CPU-safe)
        x_scaler.pkl       # joblib-serialised sklearn scaler for features
        y_scaler.pkl       # joblib-serialised sklearn scaler for target
        config.yaml        # surrogate-lab training config (for architecture)

These files are written by the full_pipeline notebook after training.

Public API
----------
SurrogatePredictor.load_latest(data_dir)   -> SurrogatePredictor
SurrogatePredictor.load_from_run(run_id)   -> SurrogatePredictor
predictor.predict(df)                       -> np.ndarray [MPa]
predictor.is_available(data_dir)            -> bool
list_mlflow_runs(tracking_uri, n)           -> list[dict]
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — deferred so the module loads even without torch/sklearn
# ---------------------------------------------------------------------------

def _import_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for surrogate inference. "
            "Install the surrogate-lab dependencies: pip install -e surrogate-lab/"
        ) from e


def _import_surrogatelab():
    """Import surrogate-lab modules; raises RuntimeError with clear message if missing."""
    try:
        from src.features.engineer import FeaturePipeline, build_xy  # noqa: F401
        from src.models.factory import build_model
        from src.utils.config import load_config
        return FeaturePipeline, build_model, load_config
    except ImportError as e:
        raise RuntimeError(
            "surrogate-lab package is not installed. "
            "Run: pip install -e surrogate-lab/ (from the simulation-api-tool root)."
        ) from e


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR_NAME = "latest"
SURROGATE_DATA_SUBDIR = "surrogate"
MODELS_SUBDIR = "models"

# Environment override for the data root (container path)
DATA_ROOT_ENV = "SURROGATE_DATA_PATH"
DEFAULT_DATA_ROOT = "/app/surrogate_data"


def _default_data_root() -> Path:
    return Path(os.getenv(DATA_ROOT_ENV, DEFAULT_DATA_ROOT))


def default_model_dir() -> Path:
    """Return path to the 'latest' model directory (container-internal)."""
    return _default_data_root() / MODELS_SUBDIR / DEFAULT_MODEL_DIR_NAME


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_model_available(model_dir: Path | None = None) -> bool:
    """Return True if all required model artifacts exist in *model_dir*."""
    d = model_dir or default_model_dir()
    required = ["best_model.pt", "x_scaler.pkl", "y_scaler.pkl", "config.yaml"]
    return all((d / f).exists() for f in required)


# ---------------------------------------------------------------------------
# SurrogatePredictor
# ---------------------------------------------------------------------------

class SurrogatePredictor:
    """
    Wraps a trained MLP model + scalers for contact-pressure prediction.

    Parameters
    ----------
    model_dir:
        Directory containing best_model.pt, x_scaler.pkl, y_scaler.pkl,
        and config.yaml.  Typically ``data/surrogate/models/latest/``.
    """

    def __init__(self, model_dir: Path) -> None:
        torch = _import_torch()
        FeaturePipeline, build_model, load_config = _import_surrogatelab()

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                "Train a surrogate model using the full_pipeline notebook first."
            )

        self._model_dir = model_dir
        self._cfg = load_config(str(model_dir / "config.yaml"))
        self._pipeline = FeaturePipeline.load(str(model_dir), self._cfg)

        input_dim = len(self._cfg["features"]["inputs"])
        self._model = build_model(input_dim=input_dim, cfg=self._cfg)
        state = torch.load(
            model_dir / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        self._model.load_state_dict(state)
        self._model.eval()

        self._feature_names: list[str] = self._cfg["features"]["inputs"]
        logger.info("SurrogatePredictor loaded from %s", model_dir)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict contact pressure for each row in *df*.

        Parameters
        ----------
        df:
            DataFrame that must contain all feature columns defined in
            the model config (``centroid_x``, ``centroid_y``, ``centroid_z``,
            ``facet_area``, ``insertion_depth`` by default).

        Returns
        -------
        np.ndarray of shape ``(n_rows,)`` — contact pressure in MPa.
        """
        torch = _import_torch()
        missing = [c for c in self._feature_names if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing feature columns: {missing}. "
                f"Required: {self._feature_names}"
            )

        X = df[self._feature_names].values.astype(np.float32)
        X_scaled, _ = self._pipeline.transform(X, np.zeros(len(X), dtype=np.float32))

        with torch.no_grad():
            y_scaled = self._model(torch.from_numpy(X_scaled)).numpy()

        return self._pipeline.inverse_transform_y(y_scaled)

    def predict_at_depth(
        self,
        facets_df: pd.DataFrame,
        insertion_depth_mm: float,
    ) -> np.ndarray:
        """
        Convenience: predict for all facets at a single insertion depth.

        Parameters
        ----------
        facets_df:
            DataFrame with facet geometry columns (centroid_x/y/z, facet_area).
            Must NOT contain ``insertion_depth`` (it will be added here).
        insertion_depth_mm:
            Catheter insertion depth in mm.

        Returns
        -------
        np.ndarray of shape ``(n_facets,)`` — predicted contact pressure [MPa].
        """
        df = facets_df.copy()
        df["insertion_depth"] = insertion_depth_mm
        return self.predict(df)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load_latest(cls, data_dir: Path | None = None) -> "SurrogatePredictor":
        """
        Load the 'latest' trained model from the shared data directory.

        Parameters
        ----------
        data_dir:
            Root of the surrogate data directory (the directory that contains
            ``models/latest/``).  Defaults to the container path
            ``/app/surrogate_data`` (set ``SURROGATE_DATA_PATH`` to override).
        """
        if data_dir is None:
            model_dir = default_model_dir()
        else:
            model_dir = Path(data_dir) / MODELS_SUBDIR / DEFAULT_MODEL_DIR_NAME
        return cls(model_dir)

    @classmethod
    def load_from_run(
        cls,
        run_id: str,
        tracking_uri: str | None = None,
        target_dir: Path | None = None,
    ) -> "SurrogatePredictor":
        """
        Download and load a specific MLflow run's artifacts.

        Parameters
        ----------
        run_id:
            MLflow run ID (hex string from list_mlflow_runs()).
        tracking_uri:
            MLflow tracking URI.  Defaults to ``MLFLOW_TRACKING_URI`` env var
            or ``http://mlflow:5000``.
        target_dir:
            Directory to download artifacts into.  Defaults to
            ``data/surrogate/models/{run_id}/``.
        """
        try:
            import mlflow
        except ImportError as e:
            raise RuntimeError("mlflow is required to load from a run") from e

        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(uri)

        if target_dir is None:
            target_dir = _default_data_root() / MODELS_SUBDIR / run_id
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Download all artifacts from this run
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        for art in artifacts:
            client.download_artifacts(run_id, art.path, str(target_dir))

        logger.info("Downloaded artifacts for run %s to %s", run_id, target_dir)
        return cls(target_dir)

    @classmethod
    def load_from_registry(
        cls,
        model_name: str,
        stage: str | None = None,
        tracking_uri: str | None = None,
        target_dir: Path | None = None,
    ) -> "SurrogatePredictor":
        """
        Load the latest registered model version from the MLflow Model Registry.

        Parameters
        ----------
        model_name:
            Registered model name in the MLflow registry
            (e.g. ``"CatheterCSARSurrogate"``).
        stage:
            Model stage to prefer: ``"Production"``, ``"Staging"``, or ``None``
            to load the latest version regardless of stage.
        tracking_uri:
            MLflow tracking URI.  Defaults to ``MLFLOW_TRACKING_URI`` env var
            or ``http://mlflow:5000``.
        target_dir:
            Directory to download artifacts into.  Defaults to
            ``data/surrogate/models/{model_name}/``.
        """
        try:
            import mlflow
        except ImportError as e:
            raise RuntimeError("mlflow is required to load from the registry") from e

        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()

        # Resolve to a run_id via the registry
        versions: list[Any] = []
        if stage and stage not in ("any", "latest"):
            versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            # Fall back to all stages — pick the highest version number
            versions = client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(
                f"No registered versions found for model '{model_name}'. "
                "Register a model first: mlflow.register_model(model_uri, model_name)."
            )

        version = max(versions, key=lambda v: int(v.version))
        run_id = version.run_id
        logger.info(
            "Loading registry model '%s' v%s stage='%s' (run_id=%s)",
            model_name, version.version, version.current_stage, run_id,
        )

        cache_key = f"registry__{model_name}__{version.version}"
        dl_dir = target_dir or (_default_data_root() / MODELS_SUBDIR / cache_key)
        return cls.load_from_run(run_id, tracking_uri=uri, target_dir=dl_dir)


# ---------------------------------------------------------------------------
# MLflow run listing
# ---------------------------------------------------------------------------

def list_registered_models(
    tracking_uri: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return all registered models and their latest versions from the MLflow Model Registry.

    Parameters
    ----------
    tracking_uri:
        MLflow URI.  Defaults to ``MLFLOW_TRACKING_URI`` env var or
        ``http://mlflow:5000``.

    Returns
    -------
    List of dicts with keys: name, description, tags, latest_versions.
    Each version has: version, stage, run_id, status.
    """
    try:
        import mlflow
    except ImportError:
        return [{"error": "mlflow not installed"}]

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    try:
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        registered = client.search_registered_models()
        result = []
        for rm in registered:
            versions = client.get_latest_versions(rm.name)
            result.append({
                "name": rm.name,
                "description": rm.description or "",
                "tags": dict(rm.tags),
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "status": v.status,
                    }
                    for v in versions
                ],
            })
        return result
    except Exception as exc:
        logger.warning("Could not list registered models: %s", exc)
        return [{"error": str(exc)}]


def list_mlflow_runs(
    tracking_uri: str | None = None,
    experiment_name: str = "surrogate-lab",
    n: int = 10,
) -> list[dict[str, Any]]:
    """
    Return recent MLflow runs from the surrogate-lab experiment.

    Parameters
    ----------
    tracking_uri:
        MLflow URI.  Defaults to ``MLFLOW_TRACKING_URI`` env var or
        ``http://mlflow:5000``.
    experiment_name:
        MLflow experiment name.
    n:
        Maximum number of runs to return (sorted by start time, newest first).

    Returns
    -------
    List of dicts with keys: run_id, status, start_time, metrics, params.
    """
    try:
        import mlflow
    except ImportError:
        return [{"error": "mlflow not installed"}]

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    try:
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return []
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=n,
        )
        result = []
        for run in runs:
            result.append(
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "artifact_uri": run.info.artifact_uri,
                }
            )
        return result
    except Exception as exc:
        logger.warning("Could not list MLflow runs: %s", exc)
        return [{"error": str(exc)}]
