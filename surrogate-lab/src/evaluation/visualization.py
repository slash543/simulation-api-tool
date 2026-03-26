"""Visualization utilities for surrogate model evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _maybe_save(fig: plt.Figure, save_path: str | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual — Contact Pressure",
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=8, label="Samples")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.35, s=8)
    axes[0].axhline(0, color="r", linestyle="--", lw=1.2)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual (pred − true)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=60, edgecolor="k", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train MSE")
    ax.plot(val_losses, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
