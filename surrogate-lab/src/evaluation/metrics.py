"""Regression metrics: RMSE, MAE, R²."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Return dict with rmse, mae, r2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    log.info("RMSE=%.6f  MAE=%.6f  R²=%.4f", rmse, mae, r2)
    return metrics


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    pipeline,
    device_str: str = "cpu",
) -> dict[str, float]:
    """
    Run inference, inverse-transform predictions and ground truth, then
    compute metrics in the original (physical) scale.
    """
    device = torch.device(device_str)
    model.eval().to(device)
    with torch.no_grad():
        y_pred_scaled = (
            model(torch.from_numpy(X_test).to(device)).cpu().numpy()
        )
    y_pred = pipeline.inverse_transform_y(y_pred_scaled)
    y_true = pipeline.inverse_transform_y(y_test_scaled)
    return compute_metrics(y_true, y_pred)
