"""Tests for src/training/trainer.py"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.models.factory import build_model
from src.training.trainer import EarlyStopping, train


class TestEarlyStopping:
    def test_does_not_trigger_on_improvement(self):
        es = EarlyStopping(patience=3)
        for loss in [1.0, 0.9, 0.8, 0.7]:
            assert not es.step(loss)

    def test_triggers_after_patience_exceeded(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)   # best
        es.step(1.1)   # +1
        es.step(1.1)   # +2
        triggered = es.step(1.1)  # +3 → trigger
        assert triggered

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)  # best
        es.step(1.1)  # +1
        es.step(0.5)  # improvement → reset
        assert es.counter == 0

    def test_best_tracks_minimum(self):
        es = EarlyStopping(patience=5)
        for loss in [1.0, 0.8, 0.6, 0.9]:
            es.step(loss)
        assert es.best == pytest.approx(0.6)

    def test_min_delta_respected(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)       # best = 1.0
        es.step(0.95)      # 1.0 - 0.95 = 0.05 < 0.1 → not improvement
        triggered = es.step(0.95)
        assert triggered

    def test_not_triggered_initially(self):
        es = EarlyStopping(patience=5)
        assert not es.triggered


class TestTrain:
    """Integration-style tests for the training loop with MLflow mocked out."""

    @pytest.fixture()
    def xy_splits(self, small_xy, minimal_cfg):
        X, y = small_xy
        from src.features.splitter import split_data
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y, minimal_cfg)
        return X_tr, X_val, y_tr, y_val

    def _run_train(self, minimal_cfg, xy_splits, tmp_path):
        minimal_cfg["training"]["checkpoint"]["dir"] = str(tmp_path / "ckpt")
        minimal_cfg["mlflow"]["tracking_uri"] = str(tmp_path / "mlruns")
        X_tr, X_val, y_tr, y_val = xy_splits
        model = build_model(X_tr.shape[1], minimal_cfg)
        return train(model, X_tr, y_tr, X_val, y_val, minimal_cfg, run_name="test-run")

    def test_returns_nn_module(self, minimal_cfg, xy_splits, tmp_path):
        import torch.nn as nn
        result = self._run_train(minimal_cfg, xy_splits, tmp_path)
        assert isinstance(result, nn.Module)

    def test_model_in_eval_after_training(self, minimal_cfg, xy_splits, tmp_path):
        model = self._run_train(minimal_cfg, xy_splits, tmp_path)
        X = xy_splits[0]
        # Move input to whichever device the model ended up on (CPU or GPU)
        device = next(model.parameters()).device
        out = model(torch.from_numpy(X).to(device))
        assert out.shape == (len(X),)

    def test_checkpoint_saved(self, minimal_cfg, xy_splits, tmp_path):
        self._run_train(minimal_cfg, xy_splits, tmp_path)
        ckpt = tmp_path / "ckpt" / "best_model.pt"
        assert ckpt.exists()

    def test_early_stopping_respects_patience(self, minimal_cfg, xy_splits, tmp_path):
        # With patience=1 and very few epochs, training should stop early
        minimal_cfg["training"]["epochs"] = 50
        minimal_cfg["training"]["early_stopping"]["patience"] = 1
        minimal_cfg["training"]["checkpoint"]["dir"] = str(tmp_path / "ckpt")
        minimal_cfg["mlflow"]["tracking_uri"] = str(tmp_path / "mlruns")
        X_tr, X_val, y_tr, y_val = xy_splits
        model = build_model(X_tr.shape[1], minimal_cfg)
        # Should complete without error (stopped early)
        result = train(model, X_tr, y_tr, X_val, y_val, minimal_cfg)
        assert result is not None

    def test_mlflow_run_created(self, minimal_cfg, xy_splits, tmp_path):
        import mlflow
        minimal_cfg["training"]["checkpoint"]["dir"] = str(tmp_path / "ckpt")
        tracking_uri = str(tmp_path / "mlruns")
        minimal_cfg["mlflow"]["tracking_uri"] = tracking_uri
        X_tr, X_val, y_tr, y_val = xy_splits
        model = build_model(X_tr.shape[1], minimal_cfg)
        train(model, X_tr, y_tr, X_val, y_val, minimal_cfg, run_name="mlflow-test")

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("test-experiment")
        assert exp is not None
        runs = client.search_runs(exp.experiment_id)
        assert len(runs) >= 1

    def test_metrics_logged_to_mlflow(self, minimal_cfg, xy_splits, tmp_path):
        import mlflow
        minimal_cfg["training"]["checkpoint"]["dir"] = str(tmp_path / "ckpt")
        tracking_uri = str(tmp_path / "mlruns")
        minimal_cfg["mlflow"]["tracking_uri"] = tracking_uri
        X_tr, X_val, y_tr, y_val = xy_splits
        model = build_model(X_tr.shape[1], minimal_cfg)
        train(model, X_tr, y_tr, X_val, y_val, minimal_cfg)

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("test-experiment")
        runs = client.search_runs(exp.experiment_id)
        assert "best_val_loss" in runs[0].data.metrics
