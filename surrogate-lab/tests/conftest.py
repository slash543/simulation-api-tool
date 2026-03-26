"""Shared fixtures for all tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def minimal_cfg():
    """Minimal config dict that mirrors configs/config.yaml structure."""
    return {
        "data": {
            "source": "data/simulations/",
            "file_pattern": "*.csv",
            "time_to_depth": {"method": "linear", "scale": 2.0},
        },
        "features": {
            "inputs": ["centroid_x", "centroid_y", "centroid_z", "facet_area", "insertion_depth"],
            "target": "contact_pressure",
            "normalization": {"method": "standard"},
        },
        "split": {"train": 0.70, "val": 0.15, "test": 0.15, "random_seed": 42},
        "model": {
            "type": "MLP",
            "layers": [16, 8],
            "activation": "relu",
            "dropout": 0.0,
        },
        "training": {
            "optimizer": "adam",
            "lr": 1e-3,
            "loss": "mse",
            "epochs": 3,
            "batch_size": 32,
            "early_stopping": {"enabled": True, "patience": 2, "min_delta": 1e-6},
            "checkpoint": {"enabled": True, "dir": "checkpoints/", "save_best": True},
        },
        "evaluation": {
            "metrics": ["rmse", "mae", "r2"],
            "kfold": {"enabled": False, "n_splits": 5},
        },
        "mlflow": {
            "experiment_name": "test-experiment",
            "tracking_uri": "mlruns/",
            "log_artifacts": False,
            "register_model": False,
            "model_name": "test_model",
        },
    }


@pytest.fixture()
def sample_df():
    """DataFrame matching the expected parser output schema (200 rows)."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "facet_id": np.arange(n),
            "centroid_x": rng.uniform(-5, 5, n),
            "centroid_y": rng.uniform(-5, 5, n),
            "centroid_z": rng.uniform(0, 10, n),
            "facet_area": rng.uniform(0.1, 2.0, n),
            "time_step": rng.uniform(0, 1, n),
            "contact_pressure": rng.uniform(0, 100, n),
        }
    )


@pytest.fixture()
def sample_df_with_depth(sample_df):
    """DataFrame that already has insertion_depth (no time_step conversion needed)."""
    df = sample_df.copy()
    df["insertion_depth"] = df["time_step"] * 2.0
    return df


@pytest.fixture()
def small_xy():
    """Small scaled numpy arrays ready for model/training tests."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 5)).astype(np.float32)
    y = rng.standard_normal(120).astype(np.float32)
    return X, y
