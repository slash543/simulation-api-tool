"""
Tests for the new template API endpoints and multi-step simulation parameters.

GET  /api/v1/templates             — list all templates
GET  /api/v1/templates/{name}      — get one template config
POST /api/v1/simulations/run       — with template / speeds_mm_s / dwell_time_s
POST /api/v1/doe/run               — with template / max_perturbation / dwell_time_s

These tests use FastAPI's synchronous TestClient; all Celery tasks are mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    from digital_twin_ui.app.main import create_app
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _make_async_result(task_id: str = "task-abc") -> MagicMock:
    mock = MagicMock()
    mock.id = task_id
    mock.state = "PENDING"
    mock.result = None
    return mock


# ---------------------------------------------------------------------------
# GET /templates
# ---------------------------------------------------------------------------

class TestListTemplates:
    def test_200_ok(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        assert resp.status_code == 200

    def test_returns_templates_key(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        body = resp.json()
        assert "templates" in body

    def test_templates_is_list(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        assert isinstance(resp.json()["templates"], list)

    def test_each_template_has_required_fields(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        templates = resp.json()["templates"]
        assert len(templates) > 0
        for t in templates:
            for field in ("name", "label", "n_steps", "speed_range_min",
                          "speed_range_max", "displacements_mm"):
                assert field in t, f"Template missing field '{field}': {t}"

    def test_sample_catheterization_present(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        names = [t["name"] for t in resp.json()["templates"]]
        assert "sample_catheterization" in names

    def test_ir12_present(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        names = [t["name"] for t in resp.json()["templates"]]
        assert "DT_BT_14Fr_FO_10E_IR12" in names

    def test_ir25_present(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        names = [t["name"] for t in resp.json()["templates"]]
        assert "DT_BT_14Fr_FO_10E_IR25" in names

    def test_ir12_has_10_steps(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        ir12 = next(
            t for t in resp.json()["templates"]
            if t["name"] == "DT_BT_14Fr_FO_10E_IR12"
        )
        assert ir12["n_steps"] == 10

    def test_displacements_mm_is_list(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        for t in resp.json()["templates"]:
            assert isinstance(t["displacements_mm"], list)
            assert len(t["displacements_mm"]) == t["n_steps"]

    def test_speed_range_min_lt_max(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates")
        for t in resp.json()["templates"]:
            assert t["speed_range_min"] < t["speed_range_max"]


# ---------------------------------------------------------------------------
# GET /templates/{name}
# ---------------------------------------------------------------------------

class TestGetTemplate:
    def test_200_for_known_template(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates/sample_catheterization")
        assert resp.status_code == 200

    def test_returns_correct_name(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates/DT_BT_14Fr_FO_10E_IR12")
        assert resp.json()["name"] == "DT_BT_14Fr_FO_10E_IR12"

    def test_ir12_n_steps_10(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates/DT_BT_14Fr_FO_10E_IR12")
        assert resp.json()["n_steps"] == 10

    def test_404_for_unknown_template(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates/does_not_exist_xyz")
        assert resp.status_code == 404

    def test_404_has_detail(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates/does_not_exist_xyz")
        assert "detail" in resp.json()

    def test_sample_catheterization_has_1_step(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates/sample_catheterization")
        assert resp.json()["n_steps"] == 1

    def test_displacements_length_matches_n_steps(self, client: TestClient) -> None:
        resp = client.get("/api/v1/templates/DT_BT_14Fr_FO_10E_IR25")
        body = resp.json()
        assert len(body["displacements_mm"]) == body["n_steps"]


# ---------------------------------------------------------------------------
# POST /simulations/run — multi-step template fields
# ---------------------------------------------------------------------------

class TestSubmitSimulationWithTemplate:
    def test_202_with_template_field(self, client: TestClient) -> None:
        fake_task = _make_async_result("t-template-1")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/simulations/run",
                json={
                    "speed_mm_s": 15.0,
                    "template": "DT_BT_14Fr_FO_10E_IR12",
                    "extract": False,
                },
            )
        assert resp.status_code == 202

    def test_task_receives_template_kwarg(self, client: TestClient) -> None:
        fake_task = _make_async_result("t-template-2")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            client.post(
                "/api/v1/simulations/run",
                json={
                    "speed_mm_s": 15.0,
                    "template": "DT_BT_14Fr_FO_10E_IR12",
                    "extract": False,
                },
            )
        call_kwargs = mock_task.delay.call_args.kwargs
        assert call_kwargs.get("template") == "DT_BT_14Fr_FO_10E_IR12"

    def test_speeds_mm_s_passed_to_task(self, client: TestClient) -> None:
        fake_task = _make_async_result("t-speeds-1")
        speeds = [10.0 + i for i in range(10)]
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            client.post(
                "/api/v1/simulations/run",
                json={
                    "speed_mm_s": 15.0,
                    "template": "DT_BT_14Fr_FO_10E_IR12",
                    "speeds_mm_s": speeds,
                    "extract": False,
                },
            )
        call_kwargs = mock_task.delay.call_args.kwargs
        assert call_kwargs.get("speeds_mm_s") == speeds

    def test_dwell_time_s_passed_to_task(self, client: TestClient) -> None:
        fake_task = _make_async_result("t-dwell-1")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            client.post(
                "/api/v1/simulations/run",
                json={
                    "speed_mm_s": 15.0,
                    "template": "DT_BT_14Fr_FO_10E_IR12",
                    "dwell_time_s": 2.0,
                    "extract": False,
                },
            )
        call_kwargs = mock_task.delay.call_args.kwargs
        assert call_kwargs.get("dwell_time_s") == 2.0

    def test_response_includes_run_id(self, client: TestClient) -> None:
        fake_task = _make_async_result("t-run-id")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/simulations/run",
                json={"speed_mm_s": 15.0, "template": "DT_BT_14Fr_FO_10E_IR12", "extract": False},
            )
        body = resp.json()
        assert "run_id" in body
        assert body["run_id"] is not None

    def test_response_includes_run_dir(self, client: TestClient) -> None:
        fake_task = _make_async_result("t-run-dir")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/simulations/run",
                json={"speed_mm_s": 15.0, "template": "DT_BT_14Fr_FO_10E_IR12", "extract": False},
            )
        body = resp.json()
        assert "run_dir" in body
        assert body["run_dir"] is not None

    def test_extract_false_uses_run_simulation_task(self, client: TestClient) -> None:
        fake_task = _make_async_result("t-nosim")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as sim_task, patch(
            "digital_twin_ui.app.api.routes.simulation.run_full_pipeline_task"
        ) as pipeline_task:
            sim_task.delay.return_value = fake_task
            client.post(
                "/api/v1/simulations/run",
                json={"speed_mm_s": 5.0, "extract": False},
            )
        sim_task.delay.assert_called_once()
        pipeline_task.delay.assert_not_called()

    def test_extract_true_uses_full_pipeline_task(self, client: TestClient) -> None:
        fake_task = _make_async_result("t-pipeline")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_full_pipeline_task"
        ) as pipeline_task, patch(
            "digital_twin_ui.app.api.routes.simulation.run_simulation_task"
        ) as sim_task:
            pipeline_task.delay.return_value = fake_task
            client.post(
                "/api/v1/simulations/run",
                json={"speed_mm_s": 5.0, "extract": True},
            )
        pipeline_task.delay.assert_called_once()
        sim_task.delay.assert_not_called()


# ---------------------------------------------------------------------------
# POST /doe/run — template / max_perturbation / dwell_time_s
# ---------------------------------------------------------------------------

class TestSubmitDOEWithTemplate:
    def test_202_with_ir12_template(self, client: TestClient) -> None:
        fake_task = _make_async_result("doe-t-1")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_doe_campaign_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/doe/run",
                json={
                    "n_samples": 5,
                    "speed_min": 10.0,
                    "speed_max": 25.0,
                    "template": "DT_BT_14Fr_FO_10E_IR12",
                    "max_perturbation": 0.20,
                    "dwell_time_s": 1.0,
                },
            )
        assert resp.status_code == 202

    def test_template_passed_to_task(self, client: TestClient) -> None:
        fake_task = _make_async_result("doe-t-2")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_doe_campaign_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            client.post(
                "/api/v1/doe/run",
                json={
                    "n_samples": 5,
                    "speed_min": 10.0,
                    "speed_max": 25.0,
                    "template": "DT_BT_14Fr_FO_10E_IR12",
                },
            )
        call_kwargs = mock_task.delay.call_args.kwargs
        assert call_kwargs.get("template") == "DT_BT_14Fr_FO_10E_IR12"

    def test_max_perturbation_passed_to_task(self, client: TestClient) -> None:
        fake_task = _make_async_result("doe-t-3")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_doe_campaign_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            client.post(
                "/api/v1/doe/run",
                json={
                    "n_samples": 5,
                    "speed_min": 10.0,
                    "speed_max": 25.0,
                    "max_perturbation": 0.30,
                },
            )
        call_kwargs = mock_task.delay.call_args.kwargs
        assert call_kwargs.get("max_perturbation") == pytest.approx(0.30)

    def test_dwell_time_passed_to_task(self, client: TestClient) -> None:
        fake_task = _make_async_result("doe-t-4")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_doe_campaign_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            client.post(
                "/api/v1/doe/run",
                json={
                    "n_samples": 5,
                    "speed_min": 10.0,
                    "speed_max": 25.0,
                    "dwell_time_s": 2.5,
                },
            )
        call_kwargs = mock_task.delay.call_args.kwargs
        assert call_kwargs.get("dwell_time_s") == pytest.approx(2.5)

    def test_speed_min_ge_max_422(self, client: TestClient) -> None:
        fake_task = _make_async_result("doe-bad")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_doe_campaign_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/doe/run",
                json={"n_samples": 5, "speed_min": 25.0, "speed_max": 10.0},
            )
        assert resp.status_code == 422

    def test_max_perturbation_out_of_range_422(self, client: TestClient) -> None:
        """max_perturbation must be 0.0–0.5 per schema."""
        fake_task = _make_async_result("doe-mp-bad")
        with patch(
            "digital_twin_ui.app.api.routes.simulation.run_doe_campaign_task"
        ) as mock_task:
            mock_task.delay.return_value = fake_task
            resp = client.post(
                "/api/v1/doe/run",
                json={
                    "n_samples": 5,
                    "speed_min": 10.0,
                    "speed_max": 25.0,
                    "max_perturbation": 0.9,  # > 0.5 limit
                },
            )
        assert resp.status_code == 422
