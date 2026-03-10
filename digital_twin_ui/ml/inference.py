"""
Inference module — load a saved PressureMLP checkpoint and run predictions.

Usage::

    from pathlib import Path
    from digital_twin_ui.ml.inference import PressurePredictor

    predictor = PressurePredictor.from_checkpoint(Path("models/pressure_mlp.pt"))

    pressure = predictor.predict(speed_mm_s=5.0)   # float
    pressures = predictor.predict_batch([4.0, 5.0, 6.0])  # list[float]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from digital_twin_ui.app.core.config import get_settings
from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.ml.model import PressureMLP

logger = get_logger(__name__)


class PressurePredictor:
    """
    Load a saved :class:`~digital_twin_ui.ml.model.PressureMLP` and predict.

    Args:
        model: Loaded ``PressureMLP`` in eval mode.
        device: Torch device to run inference on.
    """

    def __init__(self, model: PressureMLP, device: Optional[torch.device] = None) -> None:
        self._model = model
        self._device = device or torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: Optional[str] = None,
    ) -> "PressurePredictor":
        """
        Load a predictor from a ``torch.save`` checkpoint file.

        The checkpoint must contain:
          - ``model_state_dict``
          - ``hidden_dims``
          - ``input_dim``
          - ``output_dim``

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint.
            device: ``"cpu"`` or ``"cuda"`` (auto-detected if None).

        Returns:
            :class:`PressurePredictor` ready for inference.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            KeyError: If required keys are missing from the checkpoint.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=True)

        required = {"model_state_dict", "hidden_dims", "input_dim", "output_dim"}
        missing = required - set(checkpoint.keys())
        if missing:
            raise KeyError(f"Checkpoint missing keys: {missing}")

        model = PressureMLP(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            output_dim=checkpoint["output_dim"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(
            "Loaded checkpoint",
            path=str(checkpoint_path),
            hidden_dims=checkpoint["hidden_dims"],
        )
        return cls(model, device=_device)

    @classmethod
    def from_default_checkpoint(cls) -> "PressurePredictor":
        """
        Load the default checkpoint defined in ``config/simulation.yaml``.

        Returns:
            :class:`PressurePredictor`

        Raises:
            FileNotFoundError: If the default checkpoint does not exist.
        """
        cfg = get_settings()
        path = cfg.checkpoint_dir_abs / "pressure_mlp.pt"
        return cls.from_checkpoint(path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, speed_mm_s: float) -> float:
        """
        Predict peak contact pressure for one insertion speed.

        Args:
            speed_mm_s: Insertion speed in mm/s.

        Returns:
            Predicted peak contact pressure (same units as training target).
        """
        return self.predict_batch([speed_mm_s])[0]

    def predict_batch(self, speeds: list[float]) -> list[float]:
        """
        Predict peak contact pressure for a list of speeds.

        Args:
            speeds: List of insertion speeds in mm/s.

        Returns:
            List of predicted pressures in the same order.
        """
        if not speeds:
            return []
        x = torch.tensor([[s] for s in speeds], dtype=torch.float32).to(self._device)
        with torch.no_grad():
            preds = self._model(x)
        return preds.cpu().numpy().flatten().tolist()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> PressureMLP:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device
