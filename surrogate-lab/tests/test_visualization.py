"""Tests for src/evaluation/visualization.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import pytest

from src.evaluation.visualization import (
    plot_predicted_vs_actual,
    plot_residuals,
    plot_training_curves,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def yt_yp():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 100, 80)
    y_pred = y_true + rng.normal(0, 5, 80)
    return y_true, y_pred


class TestPlotPredictedVsActual:
    def test_returns_figure(self, yt_yp):
        y_true, y_pred = yt_yp
        fig = plot_predicted_vs_actual(y_true, y_pred)
        assert isinstance(fig, plt.Figure)

    def test_has_one_axes(self, yt_yp):
        y_true, y_pred = yt_yp
        fig = plot_predicted_vs_actual(y_true, y_pred)
        assert len(fig.axes) == 1

    def test_saves_to_file(self, tmp_path, yt_yp):
        y_true, y_pred = yt_yp
        out = str(tmp_path / "pred_vs_actual.png")
        plot_predicted_vs_actual(y_true, y_pred, save_path=out)
        assert (tmp_path / "pred_vs_actual.png").exists()

    def test_creates_parent_directory(self, tmp_path, yt_yp):
        y_true, y_pred = yt_yp
        out = str(tmp_path / "subdir" / "fig.png")
        plot_predicted_vs_actual(y_true, y_pred, save_path=out)
        assert (tmp_path / "subdir" / "fig.png").exists()

    def test_custom_title(self, yt_yp):
        y_true, y_pred = yt_yp
        fig = plot_predicted_vs_actual(y_true, y_pred, title="My Title")
        assert fig.axes[0].get_title() == "My Title"


class TestPlotResiduals:
    def test_returns_figure(self, yt_yp):
        y_true, y_pred = yt_yp
        fig = plot_residuals(y_true, y_pred)
        assert isinstance(fig, plt.Figure)

    def test_has_two_axes(self, yt_yp):
        y_true, y_pred = yt_yp
        fig = plot_residuals(y_true, y_pred)
        assert len(fig.axes) == 2

    def test_saves_to_file(self, tmp_path, yt_yp):
        y_true, y_pred = yt_yp
        out = str(tmp_path / "residuals.png")
        plot_residuals(y_true, y_pred, save_path=out)
        assert (tmp_path / "residuals.png").exists()


class TestPlotTrainingCurves:
    def test_returns_figure(self):
        fig = plot_training_curves(
            train_losses=[1.0, 0.8, 0.6],
            val_losses=[1.1, 0.9, 0.7],
        )
        assert isinstance(fig, plt.Figure)

    def test_has_one_axes(self):
        fig = plot_training_curves([1.0, 0.5], [1.1, 0.6])
        assert len(fig.axes) == 1

    def test_saves_to_file(self, tmp_path):
        out = str(tmp_path / "curves.png")
        plot_training_curves([1.0, 0.5], [1.2, 0.6], save_path=out)
        assert (tmp_path / "curves.png").exists()

    def test_single_epoch(self):
        fig = plot_training_curves([0.5], [0.6])
        assert isinstance(fig, plt.Figure)
