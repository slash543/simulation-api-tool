"""Feature engineering: config-driven selection and normalization."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.config import get_feature_names, get_target_name
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

_SCALERS: dict[str, type] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


def build_xy(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y using config."""
    features = get_feature_names(cfg)
    target = get_target_name(cfg)
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    return X, y


class FeaturePipeline:
    """
    Wraps sklearn scalers for X and y.

    Fit only on training data; use transform() for val/test to avoid leakage.
    """

    def __init__(self, cfg: dict) -> None:
        method = cfg["features"]["normalization"]["method"]
        scaler_cls = _SCALERS.get(method, StandardScaler)
        self.x_scaler = scaler_cls()
        self.y_scaler = scaler_cls()
        self.cfg = cfg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeaturePipeline":
        self.x_scaler.fit(X)
        self.y_scaler.fit(y.reshape(-1, 1))
        log.info("Scalers fitted  X=%s  y=%s", X.shape, y.shape)
        return self

    def transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X_s = self.x_scaler.transform(X).astype(np.float32)
        y_s = self.y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
        return X_s, y_s

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X, y).transform(X, y)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.x_scaler, Path(directory) / "x_scaler.pkl")
        joblib.dump(self.y_scaler, Path(directory) / "y_scaler.pkl")
        log.info("Scalers saved → %s", directory)

    @classmethod
    def load(cls, directory: str, cfg: dict) -> "FeaturePipeline":
        fp = cls(cfg)
        fp.x_scaler = joblib.load(Path(directory) / "x_scaler.pkl")
        fp.y_scaler = joblib.load(Path(directory) / "y_scaler.pkl")
        log.info("Scalers loaded ← %s", directory)
        return fp
