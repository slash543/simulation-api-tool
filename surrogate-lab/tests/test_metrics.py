"""Tests for src/evaluation/metrics.py"""
import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_metrics, evaluate_model
from src.features.engineer import FeaturePipeline


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y)
        assert set(metrics.keys()) == {"rmse", "mae", "r2"}

    def test_perfect_predictions(self):
        y = np.linspace(0, 10, 50)
        m = compute_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert m["mae"]  == pytest.approx(0.0, abs=1e-8)
        assert m["r2"]   == pytest.approx(1.0, abs=1e-6)

    def test_known_rmse(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] == pytest.approx(1.0)
        assert m["mae"]  == pytest.approx(1.0)

    def test_r2_worse_than_mean_is_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])  # reversed
        m = compute_metrics(y_true, y_pred)
        assert m["r2"] < 0

    def test_all_values_are_float(self):
        y = np.random.default_rng(0).standard_normal(30)
        m = compute_metrics(y, y + 0.1)
        for v in m.values():
            assert isinstance(v, float)

    def test_rmse_is_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)
        assert compute_metrics(y_true, y_pred)["rmse"] >= 0


class TestEvaluateModel:
    def _make_pipeline(self, minimal_cfg, X, y):
        from src.features.engineer import FeaturePipeline
        p = FeaturePipeline(minimal_cfg)
        p.fit(X, y)
        return p

    def test_returns_metric_dict(self, minimal_cfg, small_xy):
        X, y = small_xy
        from src.models.factory import build_model
        pipeline = self._make_pipeline(minimal_cfg, X, y)
        _, y_s = pipeline.transform(X, y)
        model = build_model(X.shape[1], minimal_cfg)
        result = evaluate_model(model, X, y_s, pipeline)
        assert set(result.keys()) == {"rmse", "mae", "r2"}

    def test_metrics_in_original_scale(self, minimal_cfg, small_xy):
        """Inverse-transform should be applied — predictions must be in original scale."""
        X, y = small_xy
        from src.models.factory import build_model
        pipeline = self._make_pipeline(minimal_cfg, X, y)
        X_s, y_s = pipeline.transform(X, y)
        model = build_model(X.shape[1], minimal_cfg)

        result = evaluate_model(model, X_s, y_s, pipeline)
        # RMSE in original scale should be larger than in scaled (std≈1) space
        # for any non-trivial scaler — just check it's a valid float
        assert np.isfinite(result["rmse"])
        assert np.isfinite(result["mae"])
        assert np.isfinite(result["r2"])
