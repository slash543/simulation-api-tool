"""
Tests for digital_twin_ui.experiments.mlflow_manager.

All tests use a temporary MLflow tracking directory so they never touch any
real mlruns directory and do not require a running MLflow server.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest

from digital_twin_ui.experiments.mlflow_manager import (
    ActiveRun,
    MLflowManager,
    get_mlflow_manager,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_tracking_uri(tmp_path: Path) -> str:
    """Return a local file:// tracking URI backed by a temp directory."""
    uri = f"file://{tmp_path / 'mlruns'}"
    return uri


@pytest.fixture()
def mgr(tmp_tracking_uri: str) -> MLflowManager:
    """A fresh MLflowManager pointing at a temp directory."""
    return MLflowManager(
        tracking_uri=tmp_tracking_uri,
        experiment_name="test_experiment",
    )


# ---------------------------------------------------------------------------
# MLflowManager initialisation
# ---------------------------------------------------------------------------

class TestMLflowManagerInit:
    def test_tracking_uri_stored(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        assert mgr.tracking_uri == tmp_tracking_uri

    def test_experiment_name_stored(self, mgr: MLflowManager) -> None:
        assert mgr.experiment_name == "test_experiment"

    def test_experiment_id_is_string(self, mgr: MLflowManager) -> None:
        assert isinstance(mgr.experiment_id, str)

    def test_experiment_created_in_mlflow(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tmp_tracking_uri)
        exp = mlflow.get_experiment_by_name("test_experiment")
        assert exp is not None
        assert exp.experiment_id == mgr.experiment_id

    def test_second_init_reuses_experiment(self, tmp_tracking_uri: str) -> None:
        mgr1 = MLflowManager(tracking_uri=tmp_tracking_uri, experiment_name="reuse_test")
        mgr2 = MLflowManager(tracking_uri=tmp_tracking_uri, experiment_name="reuse_test")
        assert mgr1.experiment_id == mgr2.experiment_id


# ---------------------------------------------------------------------------
# start_run context manager
# ---------------------------------------------------------------------------

class TestStartRun:
    def test_yields_active_run(self, mgr: MLflowManager) -> None:
        with mgr.start_run(run_name="test_run") as run:
            assert isinstance(run, ActiveRun)

    def test_run_id_is_string(self, mgr: MLflowManager) -> None:
        with mgr.start_run() as run:
            assert isinstance(run.run_id, str)
            assert len(run.run_id) > 0

    def test_run_name_stored(self, mgr: MLflowManager) -> None:
        with mgr.start_run(run_name="my_run") as run:
            assert run.run_name == "my_run"

    def test_repr_contains_run_id(self, mgr: MLflowManager) -> None:
        with mgr.start_run(run_name="repr_test") as run:
            r = repr(run)
            assert run.run_id in r

    def test_run_finished_after_exit(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        with mgr.start_run(run_name="finished_run") as run:
            run_id = run.run_id
        mlflow.set_tracking_uri(tmp_tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.info.status == "FINISHED"

    def test_run_failed_on_exception(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        run_id = None
        with pytest.raises(ValueError):
            with mgr.start_run(run_name="fail_run") as run:
                run_id = run.run_id
                raise ValueError("deliberate error")
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.info.status == "FAILED"

    def test_nested_runs(self, mgr: MLflowManager) -> None:
        with mgr.start_run(run_name="parent") as parent:
            with mgr.start_run(run_name="child", nested=True) as child:
                assert child.run_id != parent.run_id


# ---------------------------------------------------------------------------
# ActiveRun logging helpers
# ---------------------------------------------------------------------------

class TestActiveRunLogging:
    def test_log_param(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        with mgr.start_run() as run:
            run.log_param("speed", 5.0)
            run_id = run.run_id
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.params["speed"] == "5.0"

    def test_log_params(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        with mgr.start_run() as run:
            run.log_params({"a": 1, "b": "hello"})
            run_id = run.run_id
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.params["a"] == "1"
        assert fetched.data.params["b"] == "hello"

    def test_log_metric(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        with mgr.start_run() as run:
            run.log_metric("loss", 0.42)
            run_id = run.run_id
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.metrics["loss"] == pytest.approx(0.42)

    def test_log_metrics(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        with mgr.start_run() as run:
            run.log_metrics({"acc": 0.9, "f1": 0.85})
            run_id = run.run_id
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.metrics["acc"] == pytest.approx(0.9)
        assert fetched.data.metrics["f1"] == pytest.approx(0.85)

    def test_log_metric_with_step(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        with mgr.start_run() as run:
            run.log_metric("loss", 1.0, step=0)
            run.log_metric("loss", 0.5, step=1)
            run_id = run.run_id
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        history = client.get_metric_history(run_id, "loss")
        steps = [h.step for h in history]
        assert 0 in steps
        assert 1 in steps

    def test_set_tag(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        with mgr.start_run() as run:
            run.set_tag("env", "test")
            run_id = run.run_id
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.tags["env"] == "test"

    def test_set_tags(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        with mgr.start_run() as run:
            run.set_tags({"a": "1", "b": "2"})
            run_id = run.run_id
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.tags["a"] == "1"

    def test_log_artifact_file(self, mgr: MLflowManager, tmp_path: Path) -> None:
        artifact = tmp_path / "result.csv"
        artifact.write_text("time,pressure\n0.0,0.0\n")
        with mgr.start_run() as run:
            run.log_artifact(artifact)
            # No exception = success

    def test_log_artifacts_dir(self, mgr: MLflowManager, tmp_path: Path) -> None:
        art_dir = tmp_path / "artefacts"
        art_dir.mkdir()
        (art_dir / "a.txt").write_text("hello")
        (art_dir / "b.txt").write_text("world")
        with mgr.start_run() as run:
            run.log_artifacts(art_dir)
            # No exception = success


# ---------------------------------------------------------------------------
# log_simulation_run convenience helper
# ---------------------------------------------------------------------------

class TestLogSimulationRun:
    def _make_data(self):
        times = [float(i) * 0.1 for i in range(41)]
        mean_pressures = [0.0] * 23 + [float(i) * 0.01 for i in range(18)]
        max_pressure = max(mean_pressures)
        return times, mean_pressures, max_pressure

    def test_returns_run_id(self, mgr: MLflowManager) -> None:
        times, mean_pressures, max_pressure = self._make_data()
        run_id = mgr.log_simulation_run(
            run_name="sim_001",
            speed_mm_s=5.0,
            max_pressure=max_pressure,
            mean_pressures=mean_pressures,
            times=times,
        )
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_speed_param_logged(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        times, mean_pressures, max_pressure = self._make_data()
        run_id = mgr.log_simulation_run(
            run_name="sim_002",
            speed_mm_s=4.5,
            max_pressure=max_pressure,
            mean_pressures=mean_pressures,
            times=times,
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.params["speed_mm_s"] == "4.5"

    def test_max_pressure_metric_logged(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        times, mean_pressures, max_pressure = self._make_data()
        run_id = mgr.log_simulation_run(
            run_name="sim_003",
            speed_mm_s=5.0,
            max_pressure=max_pressure,
            mean_pressures=mean_pressures,
            times=times,
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert "max_pressure_kpa" in fetched.data.metrics

    def test_mean_pressure_steps_logged(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        times, mean_pressures, max_pressure = self._make_data()
        run_id = mgr.log_simulation_run(
            run_name="sim_004",
            speed_mm_s=5.0,
            max_pressure=max_pressure,
            mean_pressures=mean_pressures,
            times=times,
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        history = client.get_metric_history(run_id, "mean_pressure_kpa")
        assert len(history) == len(times)

    def test_extra_params_logged(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        times, mean_pressures, max_pressure = self._make_data()
        run_id = mgr.log_simulation_run(
            run_name="sim_005",
            speed_mm_s=5.0,
            max_pressure=max_pressure,
            mean_pressures=mean_pressures,
            times=times,
            extra_params={"run_dir": "runs/001"},
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.params["run_dir"] == "runs/001"

    def test_artefact_logged(self, mgr: MLflowManager, tmp_path: Path) -> None:
        art = tmp_path / "extraction.csv"
        art.write_text("time,pressure\n")
        times, mean_pressures, max_pressure = self._make_data()
        run_id = mgr.log_simulation_run(
            run_name="sim_006",
            speed_mm_s=5.0,
            max_pressure=max_pressure,
            mean_pressures=mean_pressures,
            times=times,
            artefact_path=art,
        )
        assert isinstance(run_id, str)

    def test_missing_artefact_silently_ignored(self, mgr: MLflowManager, tmp_path: Path) -> None:
        times, mean_pressures, max_pressure = self._make_data()
        # Does not raise even though the file does not exist
        run_id = mgr.log_simulation_run(
            run_name="sim_007",
            speed_mm_s=5.0,
            max_pressure=max_pressure,
            mean_pressures=mean_pressures,
            times=times,
            artefact_path=tmp_path / "nonexistent.csv",
        )
        assert isinstance(run_id, str)

    def test_run_status_finished(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        times, mean_pressures, max_pressure = self._make_data()
        run_id = mgr.log_simulation_run(
            run_name="sim_008",
            speed_mm_s=5.0,
            max_pressure=max_pressure,
            mean_pressures=mean_pressures,
            times=times,
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.info.status == "FINISHED"


# ---------------------------------------------------------------------------
# log_training_run convenience helper
# ---------------------------------------------------------------------------

class TestLogTrainingRun:
    def test_returns_run_id(self, mgr: MLflowManager) -> None:
        run_id = mgr.log_training_run(
            run_name="train_001",
            hyperparams={"lr": 0.001, "hidden_dims": "[64,128]"},
            train_loss=[1.0, 0.8, 0.6],
            val_loss=[1.1, 0.9, 0.7],
        )
        assert isinstance(run_id, str)

    def test_hyperparams_logged(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        run_id = mgr.log_training_run(
            run_name="train_002",
            hyperparams={"lr": 0.001},
            train_loss=[1.0],
            val_loss=[1.1],
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.data.params["lr"] == "0.001"

    def test_loss_steps_logged(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        run_id = mgr.log_training_run(
            run_name="train_003",
            hyperparams={},
            train_loss=[1.0, 0.5, 0.2],
            val_loss=[1.1, 0.6, 0.3],
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        train_hist = client.get_metric_history(run_id, "train_loss")
        val_hist = client.get_metric_history(run_id, "val_loss")
        assert len(train_hist) == 3
        assert len(val_hist) == 3

    def test_extra_metrics_logged(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        run_id = mgr.log_training_run(
            run_name="train_004",
            hyperparams={},
            train_loss=[0.1],
            val_loss=[0.2],
            extra_metrics={"test_rmse": 0.05},
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert "test_rmse" in fetched.data.metrics

    def test_model_artefact_logged(self, mgr: MLflowManager, tmp_path: Path) -> None:
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"\x00" * 16)
        run_id = mgr.log_training_run(
            run_name="train_005",
            hyperparams={},
            train_loss=[0.1],
            val_loss=[0.2],
            model_path=ckpt,
        )
        assert isinstance(run_id, str)

    def test_missing_model_path_ignored(self, mgr: MLflowManager, tmp_path: Path) -> None:
        run_id = mgr.log_training_run(
            run_name="train_006",
            hyperparams={},
            train_loss=[0.1],
            val_loss=[0.2],
            model_path=tmp_path / "no_model.pt",
        )
        assert isinstance(run_id, str)

    def test_run_status_finished(self, mgr: MLflowManager, tmp_tracking_uri: str) -> None:
        run_id = mgr.log_training_run(
            run_name="train_007",
            hyperparams={},
            train_loss=[0.1],
            val_loss=[0.2],
        )
        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_tracking_uri)
        fetched = client.get_run(run_id)
        assert fetched.info.status == "FINISHED"


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestQueryHelpers:
    def test_get_run(self, mgr: MLflowManager) -> None:
        with mgr.start_run(run_name="query_test") as run:
            run_id = run.run_id
        fetched = mgr.get_run(run_id)
        assert fetched.info.run_id == run_id

    def test_list_runs_returns_list(self, mgr: MLflowManager) -> None:
        with mgr.start_run():
            pass
        runs = mgr.list_runs()
        assert isinstance(runs, list)
        assert len(runs) >= 1

    def test_list_runs_newest_first(self, mgr: MLflowManager) -> None:
        with mgr.start_run(run_name="first"):
            pass
        time.sleep(0.05)
        with mgr.start_run(run_name="second"):
            pass
        runs = mgr.list_runs()
        assert runs[0].info.run_name == "second"

    def test_list_runs_max_results(self, mgr: MLflowManager) -> None:
        for i in range(5):
            with mgr.start_run(run_name=f"r{i}"):
                pass
        runs = mgr.list_runs(max_results=2)
        assert len(runs) <= 2


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_mlflow_manager_returns_manager(self, tmp_tracking_uri: str) -> None:
        import digital_twin_ui.experiments.mlflow_manager as mm
        # Pre-seed the singleton so the factory does not call get_settings()
        mm._manager = MLflowManager(tracking_uri=tmp_tracking_uri, experiment_name="singleton_test")
        try:
            mgr_a = mm.get_mlflow_manager()
            mgr_b = mm.get_mlflow_manager()
            assert isinstance(mgr_a, MLflowManager)
            assert mgr_a is mgr_b
        finally:
            mm._manager = None  # cleanup
