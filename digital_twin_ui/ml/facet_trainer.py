"""
Trainer for FacetPressureMLP.

Trains to predict (contact_pressure, facet_area) from (facet_id_norm, speed_mm_s).

Input features (after normalization):
  - facet_id_norm: facet_id / max_facet_id
  - speed_mm_s: scaled to [0, 1] over training range

Output targets (after normalization):
  - contact_pressure (kPa)
  - facet_area (mm²)

Usage::

    import pandas as pd
    from digital_twin_ui.ml.facet_trainer import FacetTrainer

    df = pd.read_parquet("data/datasets/facet_dataset.parquet")
    trainer = FacetTrainer()
    result = trainer.train(df)
    print(result.best_val_loss)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from digital_twin_ui.app.core.config import get_settings
from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.ml.facet_dataset import COL_AREA, COL_FACET_ID, COL_PEAK_PRESSURE, COL_SPEED
from digital_twin_ui.ml.facet_model import FacetPressureMLP

logger = get_logger(__name__)


@dataclass
class FacetTrainingResult:
    """Summary of a completed facet model training run."""

    model: FacetPressureMLP
    model_path: Optional[Path]

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)

    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epochs_trained: int = 0

    n_train: int = 0
    n_val: int = 0

    # Normalisation statistics (needed at inference time)
    feature_mean: Optional[np.ndarray] = None   # shape (2,)
    feature_std: Optional[np.ndarray] = None    # shape (2,)
    target_mean: Optional[np.ndarray] = None    # shape (2,)
    target_std: Optional[np.ndarray] = None     # shape (2,)

    @property
    def converged(self) -> bool:
        return self.epochs_trained > 0 and not math.isnan(self.best_val_loss)


class FacetTrainer:
    """
    Train FacetPressureMLP from a facet-level dataset.

    Args:
        hidden_dims: Hidden layer widths.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        max_epochs: Maximum training epochs.
        patience: Early-stopping patience.
        val_fraction: Fraction of data held out for validation.
        checkpoint_dir: Where to save model checkpoints.
        device: Torch device ("cpu" or "cuda").
    """

    def __init__(
        self,
        hidden_dims: Optional[list[int]] = None,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        max_epochs: int = 500,
        patience: int = 20,
        val_fraction: float = 0.2,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        cfg = get_settings()
        self._hidden_dims = hidden_dims or [64, 128, 256, 128, 64]
        self._lr = learning_rate
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._val_fraction = val_fraction
        self._checkpoint_dir = checkpoint_dir or cfg.checkpoint_dir_abs
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        target_cols: Optional[list[str]] = None,
        run_name: str = "facet_pressure_mlp",
    ) -> FacetTrainingResult:
        """
        Train FacetPressureMLP on the provided DataFrame.

        Args:
            df: DataFrame with at least COL_FACET_ID, COL_SPEED,
                COL_AREA, COL_PEAK_PRESSURE columns.
            feature_cols: Input feature column names.
                         Defaults to [COL_FACET_ID, COL_SPEED].
            target_cols: Target column names.
                         Defaults to [COL_PEAK_PRESSURE, COL_AREA].
            run_name: Checkpoint file stem.

        Returns:
            FacetTrainingResult
        """
        feature_cols = feature_cols or [COL_FACET_ID, COL_SPEED]
        target_cols = target_cols or [COL_PEAK_PRESSURE, COL_AREA]

        if len(df) < 2:
            raise ValueError(f"Dataset too small: {len(df)} rows (need >= 2)")

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_cols].values.astype(np.float32)

        # Normalise
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0) + 1e-8

        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std

        n = len(X_norm)
        n_val = max(1, int(n * self._val_fraction))
        n_train = n - n_val
        idx = np.random.permutation(n)
        tr, va = idx[:n_train], idx[n_train:]

        X_train = torch.from_numpy(X_norm[tr])
        y_train = torch.from_numpy(y_norm[tr])
        X_val   = torch.from_numpy(X_norm[va])
        y_val   = torch.from_numpy(y_norm[va])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self._batch_size,
            shuffle=True,
        )

        model = FacetPressureMLP(
            input_dim=len(feature_cols),
            hidden_dims=self._hidden_dims,
            output_dim=len(target_cols),
        ).to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        no_improve = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        logger.info(
            "FacetTrainer started: n_train=%d n_val=%d max_epochs=%d",
            n_train, n_val, self._max_epochs
        )

        for epoch in range(1, self._max_epochs + 1):
            model.train()
            batch_losses: list[float] = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_loss = float(np.mean(batch_losses))

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(self._device))
                val_loss = criterion(val_pred, y_val.to(self._device)).item()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 50 == 0 or epoch == 1:
                logger.debug(
                    "Epoch %d/%d  train=%.6f  val=%.6f",
                    epoch, self._max_epochs, train_loss, val_loss
                )

            if no_improve >= self._patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model_path = self._save_checkpoint(
            model, run_name, X_mean, X_std, y_mean, y_std
        )

        return FacetTrainingResult(
            model=model,
            model_path=model_path,
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            epochs_trained=len(train_losses),
            n_train=n_train,
            n_val=n_val,
            feature_mean=X_mean,
            feature_std=X_std,
            target_mean=y_mean,
            target_std=y_std,
        )

    def _save_checkpoint(
        self,
        model: FacetPressureMLP,
        run_name: str,
        X_mean: np.ndarray,
        X_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> Optional[Path]:
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self._checkpoint_dir / f"{run_name}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dims": model.hidden_dims,
                    "input_dim": model.input_dim,
                    "output_dim": model.output_dim,
                    "feature_mean": X_mean,
                    "feature_std": X_std,
                    "target_mean": y_mean,
                    "target_std": y_std,
                },
                path,
            )
            logger.info("Facet model checkpoint saved: %s", path)
            return path
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)
            return None
