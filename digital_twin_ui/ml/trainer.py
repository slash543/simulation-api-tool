"""
Training loop for the PressureMLP model.

Features
--------
- Adam optimiser with configurable learning rate
- MSE loss
- Train / validation split
- Early stopping based on validation loss
- Checkpoint saving (best model only)
- Optional MLflow logging

Usage::

    from pathlib import Path
    import pandas as pd
    from digital_twin_ui.ml.trainer import Trainer

    df = pd.read_parquet("data/datasets/catheter_dataset.parquet")
    trainer = Trainer()
    result = trainer.train(df)
    print(result.best_val_loss)
    print(result.model_path)
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
from digital_twin_ui.ml.dataset import COL_MAX_PRESSURE, COL_SPEED
from digital_twin_ui.ml.model import PressureMLP

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Summary of a completed training run."""

    model: PressureMLP
    model_path: Optional[Path]

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)

    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epochs_trained: int = 0

    n_train: int = 0
    n_val: int = 0

    @property
    def converged(self) -> bool:
        return self.epochs_trained > 0 and not math.isnan(self.best_val_loss)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Train a :class:`~digital_twin_ui.ml.model.PressureMLP` on a tabular dataset.

    Args:
        hidden_dims: Hidden layer widths. Defaults to YAML config.
        learning_rate: Adam learning rate. Defaults to YAML config.
        batch_size: Mini-batch size.
        max_epochs: Maximum training epochs.
        patience: Early-stopping patience (epochs without val improvement).
        val_fraction: Fraction of data used for validation.
        checkpoint_dir: Directory to save model checkpoints.
        device: ``"cpu"`` or ``"cuda"`` (auto-detected if ``None``).
    """

    def __init__(
        self,
        hidden_dims: Optional[list[int]] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        val_fraction: Optional[float] = None,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        cfg = get_settings()
        self._hidden_dims = hidden_dims or cfg.ml.hidden_dims
        self._lr = learning_rate if learning_rate is not None else cfg.ml.learning_rate
        self._batch_size = batch_size or cfg.ml.batch_size
        self._max_epochs = max_epochs or cfg.ml.max_epochs
        self._patience = patience if patience is not None else cfg.ml.patience
        self._val_fraction = val_fraction if val_fraction is not None else cfg.ml.val_fraction
        self._checkpoint_dir = checkpoint_dir or cfg.checkpoint_dir_abs
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        target_col: str = COL_MAX_PRESSURE,
        run_name: Optional[str] = None,
    ) -> TrainingResult:
        """
        Train on ``df`` and return a :class:`TrainingResult`.

        Args:
            df: DataFrame with at least ``speed_mm_s`` and ``max_pressure``.
            feature_cols: Input feature columns (default: ``[COL_SPEED]``).
            target_col: Target column name.
            run_name: Optional checkpoint file stem.

        Returns:
            :class:`TrainingResult` — always returned.
        """
        feature_cols = feature_cols or [COL_SPEED]

        if len(df) < 2:
            raise ValueError(f"Dataset too small: {len(df)} rows (need >= 2)")

        X = df[feature_cols].values.astype(np.float32)
        y = df[[target_col]].values.astype(np.float32)

        # Normalise features
        X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
        y_mean, y_std = y.mean(), y.std() + 1e-8
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std

        # Split
        n = len(X_norm)
        n_val = max(1, int(n * self._val_fraction))
        n_train = n - n_val

        idx = np.random.permutation(n)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        X_train = torch.from_numpy(X_norm[train_idx])
        y_train = torch.from_numpy(y_norm[train_idx])
        X_val = torch.from_numpy(X_norm[val_idx])
        y_val = torch.from_numpy(y_norm[val_idx])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self._batch_size,
            shuffle=True,
        )

        # Model
        model = PressureMLP(
            input_dim=len(feature_cols),
            hidden_dims=self._hidden_dims,
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
            "Training started",
            n_train=n_train,
            n_val=n_val,
            max_epochs=self._max_epochs,
            patience=self._patience,
        )

        for epoch in range(1, self._max_epochs + 1):
            # --- Train ---
            model.train()
            batch_losses: list[float] = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_loss = float(np.mean(batch_losses))

            # --- Validate ---
            model.eval()
            with torch.no_grad():
                Xv = X_val.to(self._device)
                yv = y_val.to(self._device)
                val_pred = model(Xv)
                val_loss = criterion(val_pred, yv).item()

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
                    epoch,
                    self._max_epochs,
                    train_loss,
                    val_loss,
                )

            if no_improve >= self._patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch,
                    self._patience,
                )
                break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save checkpoint
        model_path = self._save_checkpoint(model, run_name=run_name)

        logger.info(
            "Training complete",
            best_epoch=best_epoch,
            best_val_loss=round(best_val, 6),
            model_path=str(model_path) if model_path else None,
        )

        return TrainingResult(
            model=model,
            model_path=model_path,
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            epochs_trained=len(train_losses),
            n_train=n_train,
            n_val=n_val,
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: PressureMLP,
        run_name: Optional[str] = None,
    ) -> Optional[Path]:
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            stem = run_name or "pressure_mlp"
            path = self._checkpoint_dir / f"{stem}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dims": model.hidden_dims,
                    "input_dim": model.input_dim,
                    "output_dim": model.output_dim,
                },
                path,
            )
            logger.info("Checkpoint saved", path=str(path))
            return path
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)
            return None
