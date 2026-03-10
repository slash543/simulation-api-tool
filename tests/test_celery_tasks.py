"""
Tests for digital_twin_ui.tasks.*

All Celery tasks are called with .apply() (synchronous eager mode) to avoid
requiring a live broker.  Heavy dependencies (SimulationRunner, xplt_parser,
MLflowManager) are mocked so no actual simulation or MLflow server is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Force Celery into eager (synchronous) mode for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def celery_eager(monkeypatch):
    """Run all Celery tasks synchronously in the current process."""
    from digital_twin_ui.tasks.celery_app import celery_app
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
    )
    yield
    celery_app.conf.update(
        task_always_eager=False,
        task_eager_propagates=False,
    )


# ---------------------------------------------------------------------------
# Shared fake objects
# ---------------------------------------------------------------------------

def _fake_run_result(
    status: str = "COMPLETED",
    run_id: str = "run_test_001",
    speed: float = 5.0,
    xplt_exists: bool = True,
    tmp_path: Path | None = None,
) -> MagicMock:
    from digital_twin_ui.simulation.simulation_runner import SimulationStatus
    mock = MagicMock()
    mock.run_id = run_id
    mock.status = SimulationStatus(status)
    mock.speed_mm_s = speed
    mock.duration_s = 1.23
    mock.succeeded = (status == "COMPLETED")
    xplt = (tmp_path / "results.xplt") if tmp_path else Path("/tmp/results.xplt")
    if xplt_exists and tmp_path:
        xplt.touch()
    mock.xplt_file = xplt
    mock.as_dict.return_value = {
        "run_id": run_id,
        "status": status,
        "speed_mm_s": speed,
        "duration_s": 1.23,
        "xplt_file": str(xplt),
    }
    return mock


def _fake_pressure_result(
    times: list[float] | None = None,
    max_pressure: float = 1.5,
    mean_pressure: list[float] | None = None,
    n_faces: int = 10,
    variable_name: str = "contact pressure",
) -> MagicMock:
    if times is None:
        times = [float(i) * 0.1 for i in range(41)]
    if mean_pressure is None:
        mean_pressure = [0.0] * 23 + [float(i) * 0.01 for i in range(18)]
    mock = MagicMock()
    mock.times = times
    mock.max_pressure = max_pressure
    mock.mean_pressure = mean_pressure
    mock.n_faces = n_faces
    mock.variable_name = variable_name
    mock.as_dict.return_value = {
        "times": times,
        "max_pressure": max_pressure,
        "mean_pressure": mean_pressure,
        "n_faces": n_faces,
        "variable_name": variable_name,
    }
    return mock


# ---------------------------------------------------------------------------
# celery_app module tests
# ---------------------------------------------------------------------------

class TestCeleryApp:
    def test_app_importable(self) -> None:
        from digital_twin_ui.tasks.celery_app import celery_app
        assert celery_app is not None

    def test_app_name(self) -> None:
        from digital_twin_ui.tasks.celery_app import celery_app
        assert celery_app.main == "digital_twin_ui"

    def test_task_serializer_json(self) -> None:
        from digital_twin_ui.tasks.celery_app import celery_app
        assert celery_app.conf.task_serializer == "json"

    def test_result_serializer_json(self) -> None:
        from digital_twin_ui.tasks.celery_app import celery_app
        assert celery_app.conf.result_serializer == "json"

    def test_timezone_utc(self) -> None:
        from digital_twin_ui.tasks.celery_app import celery_app
        assert celery_app.conf.timezone == "UTC"

    def test_include_simulation_tasks(self) -> None:
        from digital_twin_ui.tasks.celery_app import celery_app
        assert "digital_twin_ui.tasks.simulation_tasks" in celery_app.conf.include

    def test_create_celery_app_returns_celery(self) -> None:
        from celery import Celery
        from digital_twin_ui.tasks.celery_app import create_celery_app
        app = create_celery_app()
        assert isinstance(app, Celery)


# ---------------------------------------------------------------------------
# run_simulation_task
# ---------------------------------------------------------------------------

class TestRunSimulationTask:
    def test_returns_dict(self, tmp_path: Path) -> None:
        fake_result = _fake_run_result(tmp_path=tmp_path)
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_runner_fn:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_result
            mock_runner_fn.return_value = mock_runner

            from digital_twin_ui.tasks.simulation_tasks import run_simulation_task
            result = run_simulation_task.apply(kwargs={"speed_mm_s": 5.0}).get()

        assert isinstance(result, dict)

    def test_speed_passed_to_runner(self, tmp_path: Path) -> None:
        fake_result = _fake_run_result(tmp_path=tmp_path)
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_runner_fn:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_result
            mock_runner_fn.return_value = mock_runner

            from digital_twin_ui.tasks.simulation_tasks import run_simulation_task
            run_simulation_task.apply(kwargs={"speed_mm_s": 4.5}).get()

        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args
        assert call_kwargs.kwargs.get("speed_mm_s") == 4.5 or call_kwargs.args[0] == 4.5

    def test_run_id_forwarded(self, tmp_path: Path) -> None:
        fake_result = _fake_run_result(run_id="my_run", tmp_path=tmp_path)
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_runner_fn:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_result
            mock_runner_fn.return_value = mock_runner

            from digital_twin_ui.tasks.simulation_tasks import run_simulation_task
            result = run_simulation_task.apply(kwargs={"speed_mm_s": 5.0, "run_id": "my_run"}).get()

        assert result["run_id"] == "my_run"

    def test_result_contains_status(self, tmp_path: Path) -> None:
        fake_result = _fake_run_result(tmp_path=tmp_path)
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_runner_fn:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_result
            mock_runner_fn.return_value = mock_runner

            from digital_twin_ui.tasks.simulation_tasks import run_simulation_task
            result = run_simulation_task.apply(kwargs={"speed_mm_s": 5.0}).get()

        assert "status" in result

    def test_failed_simulation_returned(self, tmp_path: Path) -> None:
        fake_result = _fake_run_result(status="FAILED", tmp_path=tmp_path)
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_runner_fn:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_result
            mock_runner_fn.return_value = mock_runner

            from digital_twin_ui.tasks.simulation_tasks import run_simulation_task
            result = run_simulation_task.apply(kwargs={"speed_mm_s": 5.0}).get()

        assert result["status"] == "FAILED"


# ---------------------------------------------------------------------------
# extract_results_task
# ---------------------------------------------------------------------------

class TestExtractResultsTask:
    def test_returns_dict(self, tmp_path: Path) -> None:
        xplt = tmp_path / "results.xplt"
        xplt.touch()
        fake_pressure = _fake_pressure_result()

        with patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_ext_fn:
            mock_ext = MagicMock(return_value=fake_pressure)
            mock_ext_fn.return_value = mock_ext

            from digital_twin_ui.tasks.simulation_tasks import extract_results_task
            result = extract_results_task.apply(kwargs={"xplt_path": str(xplt)}).get()

        assert isinstance(result, dict)

    def test_max_pressure_in_result(self, tmp_path: Path) -> None:
        xplt = tmp_path / "results.xplt"
        xplt.touch()
        fake_pressure = _fake_pressure_result(max_pressure=2.5)

        with patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_ext_fn:
            mock_ext_fn.return_value = MagicMock(return_value=fake_pressure)

            from digital_twin_ui.tasks.simulation_tasks import extract_results_task
            result = extract_results_task.apply(kwargs={"xplt_path": str(xplt)}).get()

        assert result["max_pressure"] == pytest.approx(2.5)

    def test_variable_name_forwarded(self, tmp_path: Path) -> None:
        xplt = tmp_path / "results.xplt"
        xplt.touch()
        fake_pressure = _fake_pressure_result()

        with patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_ext_fn:
            mock_fn = MagicMock(return_value=fake_pressure)
            mock_ext_fn.return_value = mock_fn

            from digital_twin_ui.tasks.simulation_tasks import extract_results_task
            extract_results_task.apply(
                kwargs={"xplt_path": str(xplt), "variable_name": "gap distance"}
            ).get()

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args
        passed_name = call_kwargs.kwargs.get("variable_name") or call_kwargs.args[1]
        assert passed_name == "gap distance"

    def test_extraction_error_raises(self, tmp_path: Path) -> None:
        xplt = tmp_path / "results.xplt"
        xplt.touch()
        with patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_ext_fn:
            mock_ext_fn.return_value = MagicMock(side_effect=ValueError("bad file"))

            from digital_twin_ui.tasks.simulation_tasks import extract_results_task
            with pytest.raises(Exception):
                extract_results_task.apply(kwargs={"xplt_path": str(xplt)}).get()


# ---------------------------------------------------------------------------
# log_to_mlflow_task
# ---------------------------------------------------------------------------

class TestLogToMlflowTask:
    def _pressure_dict(self) -> dict[str, Any]:
        return {
            "times": [0.1 * i for i in range(41)],
            "max_pressure": 1.5,
            "mean_pressure": [0.0] * 23 + [0.01 * i for i in range(18)],
            "n_faces": 10,
        }

    def test_returns_string_run_id(self) -> None:
        with patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_fn:
            mock_mgr = MagicMock()
            mock_mgr.log_simulation_run.return_value = "abc123"
            mock_fn.return_value = mock_mgr

            from digital_twin_ui.tasks.simulation_tasks import log_to_mlflow_task
            result = log_to_mlflow_task.apply(kwargs={
                "run_name": "run_001",
                "speed_mm_s": 5.0,
                "pressure_result": self._pressure_dict(),
            }).get()

        assert result == "abc123"

    def test_speed_forwarded_to_manager(self) -> None:
        with patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_fn:
            mock_mgr = MagicMock()
            mock_mgr.log_simulation_run.return_value = "xyz"
            mock_fn.return_value = mock_mgr

            from digital_twin_ui.tasks.simulation_tasks import log_to_mlflow_task
            log_to_mlflow_task.apply(kwargs={
                "run_name": "run_002",
                "speed_mm_s": 4.2,
                "pressure_result": self._pressure_dict(),
            }).get()

        call_kwargs = mock_mgr.log_simulation_run.call_args.kwargs
        assert call_kwargs["speed_mm_s"] == pytest.approx(4.2)

    def test_extra_params_forwarded(self) -> None:
        with patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_fn:
            mock_mgr = MagicMock()
            mock_mgr.log_simulation_run.return_value = "xyz"
            mock_fn.return_value = mock_mgr

            from digital_twin_ui.tasks.simulation_tasks import log_to_mlflow_task
            log_to_mlflow_task.apply(kwargs={
                "run_name": "run_003",
                "speed_mm_s": 5.0,
                "pressure_result": self._pressure_dict(),
                "extra_params": {"solver": "febio4"},
            }).get()

        call_kwargs = mock_mgr.log_simulation_run.call_args.kwargs
        assert call_kwargs["extra_params"] == {"solver": "febio4"}

    def test_mlflow_failure_raises(self) -> None:
        with patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_fn:
            mock_mgr = MagicMock()
            mock_mgr.log_simulation_run.side_effect = RuntimeError("mlflow down")
            mock_fn.return_value = mock_mgr

            from digital_twin_ui.tasks.simulation_tasks import log_to_mlflow_task
            with pytest.raises(Exception):
                log_to_mlflow_task.apply(kwargs={
                    "run_name": "fail_run",
                    "speed_mm_s": 5.0,
                    "pressure_result": self._pressure_dict(),
                }).get()


# ---------------------------------------------------------------------------
# run_doe_campaign_task
# ---------------------------------------------------------------------------

class TestRunDOECampaignTask:
    def _make_runner_mock(self, tmp_path: Path) -> MagicMock:
        mock_runner = MagicMock()
        mock_runner.run.side_effect = lambda speed_mm_s, **kw: _fake_run_result(
            speed=speed_mm_s, tmp_path=tmp_path
        )
        return mock_runner

    def test_returns_dict(self, tmp_path: Path) -> None:
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager"):
            mock_r.return_value = self._make_runner_mock(tmp_path)
            fake_pressure = _fake_pressure_result()
            mock_e.return_value = MagicMock(return_value=fake_pressure)

            from digital_twin_ui.tasks.simulation_tasks import run_doe_campaign_task
            result = run_doe_campaign_task.apply(kwargs={
                "n_samples": 3,
                "speed_min": 4.0,
                "speed_max": 6.0,
                "sampler": "uniform",
                "extract": False,
            }).get()

        assert isinstance(result, dict)

    def test_n_samples_respected(self, tmp_path: Path) -> None:
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor"), \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager"):
            mock_r.return_value = self._make_runner_mock(tmp_path)

            from digital_twin_ui.tasks.simulation_tasks import run_doe_campaign_task
            result = run_doe_campaign_task.apply(kwargs={
                "n_samples": 4,
                "extract": False,
            }).get()

        assert result["n_samples"] == 4
        assert len(result["samples"]) == 4

    def test_samples_have_simulation_key(self, tmp_path: Path) -> None:
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor"), \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager"):
            mock_r.return_value = self._make_runner_mock(tmp_path)

            from digital_twin_ui.tasks.simulation_tasks import run_doe_campaign_task
            result = run_doe_campaign_task.apply(kwargs={"n_samples": 2, "extract": False}).get()

        for sample in result["samples"]:
            assert "simulation" in sample

    def test_extraction_result_included(self, tmp_path: Path) -> None:
        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager"):
            mock_r.return_value = self._make_runner_mock(tmp_path)
            fake_pressure = _fake_pressure_result()
            mock_e.return_value = MagicMock(return_value=fake_pressure)

            from digital_twin_ui.tasks.simulation_tasks import run_doe_campaign_task
            result = run_doe_campaign_task.apply(kwargs={
                "n_samples": 2,
                "extract": True,
                "log_mlflow": False,
            }).get()

        for sample in result["samples"]:
            assert "extraction" in sample

    def test_speeds_within_range(self, tmp_path: Path) -> None:
        captured_speeds: list[float] = []

        def side_effect(speed_mm_s, **kw):
            captured_speeds.append(speed_mm_s)
            return _fake_run_result(speed=speed_mm_s, tmp_path=tmp_path)

        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor"), \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager"):
            mock_runner = MagicMock()
            mock_runner.run.side_effect = side_effect
            mock_r.return_value = mock_runner

            from digital_twin_ui.tasks.simulation_tasks import run_doe_campaign_task
            run_doe_campaign_task.apply(kwargs={
                "n_samples": 5,
                "speed_min": 4.0,
                "speed_max": 6.0,
                "sampler": "uniform",
                "extract": False,
            }).get()

        assert len(captured_speeds) == 5
        for s in captured_speeds:
            assert 4.0 <= s <= 6.0


# ---------------------------------------------------------------------------
# run_full_pipeline_task
# ---------------------------------------------------------------------------

class TestRunFullPipelineTask:
    def test_returns_dict(self, tmp_path: Path) -> None:
        fake_run = _fake_run_result(tmp_path=tmp_path)
        fake_pressure = _fake_pressure_result()

        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_m:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_run
            mock_r.return_value = mock_runner
            mock_e.return_value = MagicMock(return_value=fake_pressure)
            mock_m.return_value = MagicMock()
            mock_m.return_value.log_simulation_run.return_value = "mlflow_id"

            from digital_twin_ui.tasks.simulation_tasks import run_full_pipeline_task
            result = run_full_pipeline_task.apply(kwargs={
                "speed_mm_s": 5.0,
                "log_mlflow": True,
            }).get()

        assert isinstance(result, dict)
        assert "simulation" in result

    def test_extraction_included_on_success(self, tmp_path: Path) -> None:
        fake_run = _fake_run_result(tmp_path=tmp_path)
        fake_pressure = _fake_pressure_result()

        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_m:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_run
            mock_r.return_value = mock_runner
            mock_e.return_value = MagicMock(return_value=fake_pressure)
            mock_m.return_value = MagicMock()
            mock_m.return_value.log_simulation_run.return_value = "id1"

            from digital_twin_ui.tasks.simulation_tasks import run_full_pipeline_task
            result = run_full_pipeline_task.apply(kwargs={"speed_mm_s": 5.0}).get()

        assert "extraction" in result

    def test_mlflow_run_id_included(self, tmp_path: Path) -> None:
        fake_run = _fake_run_result(tmp_path=tmp_path)
        fake_pressure = _fake_pressure_result()

        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_m:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_run
            mock_r.return_value = mock_runner
            mock_e.return_value = MagicMock(return_value=fake_pressure)
            mock_mgr = MagicMock()
            mock_mgr.log_simulation_run.return_value = "mlflow_id_xyz"
            mock_m.return_value = mock_mgr

            from digital_twin_ui.tasks.simulation_tasks import run_full_pipeline_task
            result = run_full_pipeline_task.apply(kwargs={
                "speed_mm_s": 5.0,
                "log_mlflow": True,
            }).get()

        assert result.get("mlflow_run_id") == "mlflow_id_xyz"

    def test_failed_simulation_skips_extraction(self, tmp_path: Path) -> None:
        fake_run = _fake_run_result(status="FAILED", tmp_path=tmp_path)

        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_run
            mock_r.return_value = mock_runner
            mock_e.return_value = MagicMock()

            from digital_twin_ui.tasks.simulation_tasks import run_full_pipeline_task
            result = run_full_pipeline_task.apply(kwargs={
                "speed_mm_s": 5.0,
                "log_mlflow": False,
            }).get()

        assert "extraction" not in result
        mock_e.return_value.assert_not_called()

    def test_no_mlflow_when_disabled(self, tmp_path: Path) -> None:
        fake_run = _fake_run_result(tmp_path=tmp_path)
        fake_pressure = _fake_pressure_result()

        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_m:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_run
            mock_r.return_value = mock_runner
            mock_e.return_value = MagicMock(return_value=fake_pressure)

            from digital_twin_ui.tasks.simulation_tasks import run_full_pipeline_task
            result = run_full_pipeline_task.apply(kwargs={
                "speed_mm_s": 5.0,
                "log_mlflow": False,
            }).get()

        assert "mlflow_run_id" not in result
        mock_m.assert_not_called()

    def test_extraction_error_handled_gracefully(self, tmp_path: Path) -> None:
        fake_run = _fake_run_result(tmp_path=tmp_path)

        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager"):
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_run
            mock_r.return_value = mock_runner
            mock_e.return_value = MagicMock(side_effect=ValueError("parse error"))

            from digital_twin_ui.tasks.simulation_tasks import run_full_pipeline_task
            result = run_full_pipeline_task.apply(kwargs={
                "speed_mm_s": 5.0,
                "log_mlflow": False,
            }).get()

        assert "extraction_error" in result

    def test_mlflow_error_handled_gracefully(self, tmp_path: Path) -> None:
        fake_run = _fake_run_result(tmp_path=tmp_path)
        fake_pressure = _fake_pressure_result()

        with patch("digital_twin_ui.tasks.simulation_tasks._import_runner") as mock_r, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_extractor") as mock_e, \
             patch("digital_twin_ui.tasks.simulation_tasks._import_mlflow_manager") as mock_m:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_run
            mock_r.return_value = mock_runner
            mock_e.return_value = MagicMock(return_value=fake_pressure)
            mock_mgr = MagicMock()
            mock_mgr.log_simulation_run.side_effect = RuntimeError("mlflow crash")
            mock_m.return_value = mock_mgr

            from digital_twin_ui.tasks.simulation_tasks import run_full_pipeline_task
            result = run_full_pipeline_task.apply(kwargs={
                "speed_mm_s": 5.0,
                "log_mlflow": True,
            }).get()

        assert "mlflow_error" in result
        assert "extraction" in result  # extraction still succeeded
