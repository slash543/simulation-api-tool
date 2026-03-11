"""
Tests for the FastAPI simulation routes.

Uses httpx AsyncClient + ASGITransport to call the app in-process.
All Celery tasks, SimulationRunner, and xplt extractor are mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture (synchronous TestClient — fast, no broker needed)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Return a synchronous TestClient for the app."""
    from digital_twin_ui.app.main import create_app
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Fake Celery AsyncResult
# ---------------------------------------------------------------------------

def _make_async_result(task_id: str = "task-abc", state: str = "PENDING", result: Any = None):
    mock = MagicMock()
    mock.id = task_id
    mock.state = state
    mock.result = result
    return mock


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.json()["status"] == "ok"

    def test_health_version_present(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# POST /simulations/run  (async submission)
# ---------------------------------------------------------------------------

class TestSubmitSimulation:
    def test_accepted_202(self, client: TestClient) -> None:
        fake_task = _make_async_result("t1")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_full_pipeline_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/simulations/run",
                json={"speed_mm_s": 5.0, "extract": True},
            )
        assert resp.status_code == 202

    def test_task_id_returned(self, client: TestClient) -> None:
        fake_task = _make_async_result("task-xyz")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_full_pipeline_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/simulations/run",
                json={"speed_mm_s": 5.0, "extract": True},
            )
        assert resp.json()["task_id"] == "task-xyz"

    def test_status_pending(self, client: TestClient) -> None:
        fake_task = _make_async_result("t2")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_full_pipeline_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/simulations/run",
                json={"speed_mm_s": 5.0},
            )
        assert resp.json()["status"] == "PENDING"

    def test_extract_false_uses_simulation_task(self, client: TestClient) -> None:
        fake_task = _make_async_result("t3")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as mock_sim, patch(
            "digital_twin_ui.app.api.routes.simulation.run_full_pipeline_task"
        ) as mock_pipe:
            mock_sim.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/simulations/run",
                json={"speed_mm_s": 5.0, "extract": False},
            )
        mock_sim.delay.assert_called_once()
        mock_pipe.delay.assert_not_called()

    def test_invalid_speed_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/simulations/run",
            json={"speed_mm_s": -1.0},
        )
        assert resp.status_code == 422

    def test_zero_speed_422(self, client: TestClient) -> None:
        """speed_mm_s must be > 0 (gt=0 constraint)."""
        fake_task = _make_async_result("t-zero")
        with patch("digital_twin_ui.app.api.routes.simulation.run_simulation_task") as mt:
            mt.delay.return_value = fake_task
            resp = client.post("/api/v1/simulations/run", json={"speed_mm_s": 0.0, "extract": False})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /simulations/{task_id}
# ---------------------------------------------------------------------------

class TestGetSimulationStatus:
    def test_pending_status(self, client: TestClient) -> None:
        from celery.result import AsyncResult
        with patch(
            "digital_twin_ui.app.api.routes.simulation.AsyncResult"
        ) as mock_res_cls:
            mock_res_cls.return_value = _make_async_result("t4", "PENDING")
            resp = client.get("/api/v1/simulations/t4")
        assert resp.status_code == 200
        assert resp.json()["status"] == "PENDING"

    def test_success_status_includes_result(self, client: TestClient) -> None:
        fake_result = {"run_id": "run_001", "status": "COMPLETED"}
        with patch(
            "digital_twin_ui.app.api.routes.simulation.AsyncResult"
        ) as mock_res_cls:
            mock_res_cls.return_value = _make_async_result("t5", "SUCCESS", fake_result)
            resp = client.get("/api/v1/simulations/t5")
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert data["result"] is not None

    def test_failure_status_includes_error(self, client: TestClient) -> None:
        with patch(
            "digital_twin_ui.app.api.routes.simulation.AsyncResult"
        ) as mock_res_cls:
            mock_res_cls.return_value = _make_async_result("t6", "FAILURE", RuntimeError("oops"))
            resp = client.get("/api/v1/simulations/t6")
        data = resp.json()
        assert data["status"] == "FAILURE"
        assert data["error"] is not None

    def test_task_id_in_response(self, client: TestClient) -> None:
        with patch(
            "digital_twin_ui.app.api.routes.simulation.AsyncResult"
        ) as mock_res_cls:
            mock_res_cls.return_value = _make_async_result("my-task", "PENDING")
            resp = client.get("/api/v1/simulations/my-task")
        assert resp.json()["task_id"] == "my-task"


# ---------------------------------------------------------------------------
# POST /simulations/run/sync
# ---------------------------------------------------------------------------

class TestRunSimulationSync:
    def _mock_run_result(self, tmp_path: Path, status: str = "COMPLETED") -> MagicMock:
        from digital_twin_ui.simulation.simulation_runner import SimulationStatus
        mock = MagicMock()
        mock.run_id = "run_sync_001"
        mock.status = SimulationStatus(status)
        mock.speed_mm_s = 5.0
        mock.duration_s = 2.0
        mock.error_message = None
        mock.succeeded = (status == "COMPLETED")
        xplt = tmp_path / "results.xplt"
        if status == "COMPLETED":
            xplt.touch()
        mock.xplt_file = xplt
        return mock

    def test_sync_200(self, client: TestClient, tmp_path: Path) -> None:
        mock_result = self._mock_run_result(tmp_path)
        with patch(
            "digital_twin_ui.app.api.routes.simulation.SimulationRunner"
        ) as mock_cls:
            mock_cls.return_value.run_async = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/api/v1/simulations/run/sync",
                json={"speed_mm_s": 5.0, "extract": False},
            )
        assert resp.status_code == 200

    def test_sync_run_id_in_response(self, client: TestClient, tmp_path: Path) -> None:
        mock_result = self._mock_run_result(tmp_path)
        with patch("digital_twin_ui.app.api.routes.simulation.SimulationRunner") as mock_cls:
            mock_cls.return_value.run_async = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/api/v1/simulations/run/sync",
                json={"speed_mm_s": 5.0, "extract": False},
            )
        assert resp.json()["run_id"] == "run_sync_001"

    def test_sync_status_in_response(self, client: TestClient, tmp_path: Path) -> None:
        mock_result = self._mock_run_result(tmp_path)
        with patch("digital_twin_ui.app.api.routes.simulation.SimulationRunner") as mock_cls:
            mock_cls.return_value.run_async = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/api/v1/simulations/run/sync",
                json={"speed_mm_s": 5.0, "extract": False},
            )
        assert resp.json()["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# POST /doe/run
# ---------------------------------------------------------------------------

class TestSubmitDOE:
    def test_accepted_202(self, client: TestClient) -> None:
        fake_task = _make_async_result("doe-t1")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_doe_campaign_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/doe/run",
                json={"n_samples": 5, "speed_min": 4.0, "speed_max": 6.0},
            )
        assert resp.status_code == 202

    def test_task_id_returned(self, client: TestClient) -> None:
        fake_task = _make_async_result("doe-t2")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_doe_campaign_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/doe/run",
                json={"n_samples": 5},
            )
        assert resp.json()["task_id"] == "doe-t2"

    def test_speed_min_ge_speed_max_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/doe/run",
            json={"n_samples": 5, "speed_min": 6.0, "speed_max": 4.0},
        )
        assert resp.status_code == 422

    def test_invalid_sampler_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/doe/run",
            json={"n_samples": 5, "sampler": "unknown"},
        )
        assert resp.status_code == 422

    def test_n_samples_zero_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/doe/run",
            json={"n_samples": 0},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /doe/{task_id}
# ---------------------------------------------------------------------------

class TestGetDOEStatus:
    def test_pending_200(self, client: TestClient) -> None:
        with patch(
            "digital_twin_ui.app.api.routes.simulation.AsyncResult"
        ) as mock_res_cls:
            mock_res_cls.return_value = _make_async_result("doe-poll-1", "PENDING")
            resp = client.get("/api/v1/doe/doe-poll-1")
        assert resp.status_code == 200

    def test_success_includes_result(self, client: TestClient) -> None:
        doe_result = {"n_samples": 5, "samples": []}
        with patch(
            "digital_twin_ui.app.api.routes.simulation.AsyncResult"
        ) as mock_res_cls:
            mock_res_cls.return_value = _make_async_result("doe-poll-2", "SUCCESS", doe_result)
            resp = client.get("/api/v1/doe/doe-poll-2")
        assert resp.json()["result"] == doe_result


# ---------------------------------------------------------------------------
# POST /extract/sync
# ---------------------------------------------------------------------------

class TestExtractSync:
    def test_200_with_valid_file(self, client: TestClient, tmp_path: Path) -> None:
        xplt = tmp_path / "results.xplt"
        xplt.touch()
        fake_pressure = MagicMock()
        fake_pressure.variable_name = "contact pressure"
        fake_pressure.times = [0.0, 0.1]
        fake_pressure.max_pressure = 1.5
        fake_pressure.mean_pressure = [0.0, 1.5]
        fake_pressure.n_faces = 10
        fake_pressure.source_path = xplt

        with patch(
            "digital_twin_ui.app.api.routes.simulation.extract_contact_pressure"
        ) as mock_ext:
            mock_ext.return_value = fake_pressure
            resp = client.post(
                "/api/v1/extract/sync",
                json={"xplt_path": str(xplt)},
            )
        assert resp.status_code == 200

    def test_404_missing_file(self, client: TestClient, tmp_path: Path) -> None:
        resp = client.post(
            "/api/v1/extract/sync",
            json={"xplt_path": str(tmp_path / "nonexistent.xplt")},
        )
        assert resp.status_code == 404

    def test_422_extraction_error(self, client: TestClient, tmp_path: Path) -> None:
        xplt = tmp_path / "bad.xplt"
        xplt.write_bytes(b"\x00" * 8)
        with patch(
            "digital_twin_ui.app.api.routes.simulation.extract_contact_pressure",
            side_effect=ValueError("bad file"),
        ):
            resp = client.post(
                "/api/v1/extract/sync",
                json={"xplt_path": str(xplt)},
            )
        assert resp.status_code == 422

    def test_response_contains_max_pressure(self, client: TestClient, tmp_path: Path) -> None:
        xplt = tmp_path / "results.xplt"
        xplt.touch()
        fake_pressure = MagicMock()
        fake_pressure.variable_name = "contact pressure"
        fake_pressure.times = [0.0]
        fake_pressure.max_pressure = 3.14
        fake_pressure.mean_pressure = [3.14]
        fake_pressure.n_faces = 5
        fake_pressure.source_path = xplt

        with patch(
            "digital_twin_ui.app.api.routes.simulation.extract_contact_pressure"
        ) as mock_ext:
            mock_ext.return_value = fake_pressure
            resp = client.post(
                "/api/v1/extract/sync",
                json={"xplt_path": str(xplt)},
            )
        assert resp.json()["max_pressure"] == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# POST /extract (async submission)
# ---------------------------------------------------------------------------

class TestSubmitExtraction:
    def test_accepted_202(self, client: TestClient, tmp_path: Path) -> None:
        xplt = tmp_path / "results.xplt"
        xplt.touch()
        fake_task = _make_async_result("ext-t1")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.extract_results_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/extract",
                json={"xplt_path": str(xplt)},
            )
        assert resp.status_code == 202

    def test_404_for_missing_file(self, client: TestClient, tmp_path: Path) -> None:
        resp = client.post(
            "/api/v1/extract",
            json={"xplt_path": str(tmp_path / "nonexistent.xplt")},
        )
        assert resp.status_code == 404

    def test_task_id_returned(self, client: TestClient, tmp_path: Path) -> None:
        xplt = tmp_path / "results.xplt"
        xplt.touch()
        fake_task = _make_async_result("ext-t2")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.extract_results_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/extract",
                json={"xplt_path": str(xplt)},
            )
        assert resp.json()["task_id"] == "ext-t2"
