"""
Inference for FacetPressureMLP — predict (contact_pressure, facet_area) from
(facet_id, speed_mm_s).

Usage::

    from pathlib import Path
    from digital_twin_ui.ml.facet_inference import FacetPredictor

    pred = FacetPredictor.from_checkpoint(Path("models/facet_pressure_mlp.pt"))
    pressure, area = pred.predict(facet_id=100, speed_mm_s=5.0)
    results = pred.predict_batch(facet_ids=[0, 1, 2], speeds=[4.0, 5.0, 6.0])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.ml.facet_model import FacetPressureMLP

logger = get_logger(__name__)


class FacetPredictor:
    """
    Load a saved FacetPressureMLP and run inference.

    Args:
        model: Loaded model in eval mode.
        feature_mean: Feature normalisation mean, shape (input_dim,).
        feature_std: Feature normalisation std, shape (input_dim,).
        target_mean: Target denormalisation mean, shape (output_dim,).
        target_std: Target denormalisation std, shape (output_dim,).
        device: Torch device.
    """

    def __init__(
        self,
        model: FacetPressureMLP,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> None:
        self._model = model
        self._feature_mean = feature_mean
        self._feature_std = feature_std
        self._target_mean = target_mean
        self._target_std = target_std
        self._device = device or torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: Optional[str] = None,
    ) -> "FacetPredictor":
        """Load from a checkpoint saved by FacetTrainer."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        _dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=_dev, weights_only=False)

        required = {"model_state_dict", "hidden_dims", "input_dim", "output_dim",
                    "feature_mean", "feature_std", "target_mean", "target_std"}
        missing = required - set(ckpt.keys())
        if missing:
            raise KeyError(f"Checkpoint missing keys: {missing}")

        model = FacetPressureMLP(
            input_dim=ckpt["input_dim"],
            hidden_dims=ckpt["hidden_dims"],
            output_dim=ckpt["output_dim"],
        )
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(
            model=model,
            feature_mean=np.array(ckpt["feature_mean"], dtype=np.float32),
            feature_std=np.array(ckpt["feature_std"], dtype=np.float32),
            target_mean=np.array(ckpt["target_mean"], dtype=np.float32),
            target_std=np.array(ckpt["target_std"], dtype=np.float32),
            device=_dev,
        )

    def predict(self, facet_id: int, speed_mm_s: float) -> tuple[float, float]:
        """
        Predict (contact_pressure, facet_area) for one facet and speed.

        Returns:
            Tuple (predicted_pressure, predicted_area).
        """
        results = self.predict_batch([facet_id], [speed_mm_s])
        return results[0][0], results[0][1]

    def predict_batch(
        self,
        facet_ids: list[int],
        speeds: list[float],
    ) -> list[tuple[float, float]]:
        """
        Predict (contact_pressure, facet_area) for batches of facet_id + speed.

        Args:
            facet_ids: List of 0-based facet IDs.
            speeds: List of insertion speeds in mm/s (same length as facet_ids).

        Returns:
            List of (pressure, area) tuples.
        """
        if not facet_ids:
            return []
        X = np.array([[fid, s] for fid, s in zip(facet_ids, speeds)], dtype=np.float32)
        X_norm = (X - self._feature_mean) / self._feature_std
        x_tensor = torch.from_numpy(X_norm).to(self._device)
        with torch.no_grad():
            y_norm = self._model(x_tensor).cpu().numpy()
        y = y_norm * self._target_std + self._target_mean
        return [(float(row[0]), float(row[1])) for row in y]

    @property
    def model(self) -> FacetPressureMLP:
        return self._model
