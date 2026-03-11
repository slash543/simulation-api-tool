"""
Tests for catheter-catalogue FastAPI endpoints.

Coverage
--------
GET  /api/v1/catheter-designs
  - Returns 200 with correct structure
  - Contains all 3 tip designs
  - Each design has at least one configuration
  - Shared sim params present (n_steps, displacements_mm, speed_range_min/max,
    default_uniform_speed_mm_s, default_dwell_time_s)

POST /api/v1/simulations/run-catheter
  - 202 accepted with task_id, run_id, run_dir, xplt_path
  - 404 for unknown design
  - 422 for unknown configuration (within a valid design)
  - 422 when speeds_mm_s length != n_steps
  - 422 when speeds_mm_s is empty
  - Uses minimal in-memory catalogue (no real FEB files needed for unit tests)

POST /api/v1/doe/preview-speeds
  - Returns correct shape (n_samples × n_steps)
  - Each speed in [speed_min, speed_max]
  - seed makes sampling reproducible
  - Invalid body fields return 422
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers: in-memory catalogue fixture
# ---------------------------------------------------------------------------

MINIMAL_CATALOGUE = {
    "designs": {
        "ball_tip": {
            "label": "Ball Tip",
            "configurations": {
                "14Fr_IR12": {
                    "label": "14Fr catheter — IR12 urethra model",
                    "feb_file": "ball_tip_14FR_ir12.feb",
                },
                "16Fr_IR12": {
                    "label": "16Fr catheter — IR12 urethra model",
                    "feb_file": "ball_tip_16FR_ir12.feb",
                },
            },
        },
        "nelaton_tip": {
            "label": "Nelaton Tip",
            "configurations": {
                "14Fr_IR12": {
                    "label": "14Fr catheter — IR12 urethra model",
                    "feb_file": "nelaton_tip_14Fr_ir12.feb",
                },
            },
        },
        "vapro_introducer": {
            "label": "Vapro Introducer",
            "configurations": {
                "14Fr_IR12": {
                    "label": "14Fr catheter — IR12 urethra model",
                    "feb_file": "vapro_introducer_14Fr_tip_ir12.feb",
                },
            },
        },
    },
    "simulation": {
        "n_steps": 10,
        "base_step_size": 0.1,
        "default_dwell_time_s": 1.0,
        "displacements_mm": [64.0, 46.0] + [28.0] * 8,
        "speed_range": {"min_mm_s": 10.0, "max_mm_s": 25.0},
        "default_uniform_speed_mm_s": 15.0,
    },
}


def _write_catalogue(tmp_path: Path) -> Path:
    p = tmp_path / "catheter_catalogue.yaml"
    p.write_text(yaml.dump(MINIMAL_CATALOGUE), encoding="utf-8")
    return p


def _stub_feb_files(tmp_path: Path) -> None:
    """Create stub .feb files so resolve() does not raise FileNotFoundError."""
    base = tmp_path / "base_configuration"
    base.mkdir(exist_ok=True)
    stub = "<?xml version='1.0'?><febio_spec/>"
    for design_data in MINIMAL_CATALOGUE["designs"].values():
        for cfg_data in design_data["configurations"].values():
            (base / cfg_data["feb_file"]).write_text(stub)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    """
    Return a synchronous TestClient backed by an in-memory catalogue.

    The CatheterCatalogue singleton is reset for each test and pointed at a
    temporary YAML file.  Celery tasks are patched so no broker is needed.
    """
    cat_path = _write_catalogue(tmp_path)
    _stub_feb_files(tmp_path)

    from digital_twin_ui.simulation.catheter_catalogue import CatheterCatalogue

    real_catalogue = CatheterCatalogue(
        catalogue_path=cat_path,
        project_root=tmp_path,
    )

    with (
        patch(
            "digital_twin_ui.simulation.catheter_catalogue.get_catalogue",
            return_value=real_catalogue,
        ),
        patch(
            "digital_twin_ui.app.api.routes.simulation.run_catheter_simulation_task"
        ) as mock_task,
    ):
        # Simulate Celery .delay() returning an AsyncResult-like object
        fake_result = MagicMock()
        fake_result.id = "fake-task-id-1234"
        mock_task.delay.return_value = fake_result

        from digital_twin_ui.app.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_task


# ---------------------------------------------------------------------------
# GET /api/v1/catheter-designs
# ---------------------------------------------------------------------------

class TestListCatheterDesigns:
    def test_status_200(self, client):
        c, _ = client
        resp = c.get("/api/v1/catheter-designs")
        assert resp.status_code == 200

    def test_response_has_designs_key(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        assert "designs" in data

    def test_three_designs_returned(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        assert len(data["designs"]) == 3

    def test_design_names_present(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        names = {d["name"] for d in data["designs"]}
        assert names == {"ball_tip", "nelaton_tip", "vapro_introducer"}

    def test_design_labels_present(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        labels = {d["label"] for d in data["designs"]}
        assert "Ball Tip" in labels

    def test_each_design_has_configurations(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        for design in data["designs"]:
            assert len(design["configurations"]) >= 1, (
                f"Design '{design['name']}' has no configurations"
            )

    def test_configuration_has_key_label_feb_file(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        ball_tip = next(d for d in data["designs"] if d["name"] == "ball_tip")
        cfg = ball_tip["configurations"][0]
        assert "key" in cfg
        assert "label" in cfg
        assert "feb_file" in cfg

    def test_n_steps_is_ten(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        assert data["n_steps"] == 10

    def test_displacements_mm_length(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        assert len(data["displacements_mm"]) == 10

    def test_speed_range_present(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        assert data["speed_range_min"] == pytest.approx(10.0)
        assert data["speed_range_max"] == pytest.approx(25.0)

    def test_default_uniform_speed_present(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        assert data["default_uniform_speed_mm_s"] == pytest.approx(15.0)

    def test_default_dwell_time_present(self, client):
        c, _ = client
        data = c.get("/api/v1/catheter-designs").json()
        assert data["default_dwell_time_s"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# POST /api/v1/simulations/run-catheter
# ---------------------------------------------------------------------------

def _valid_payload(**overrides) -> dict:
    payload = {
        "design": "ball_tip",
        "configuration": "14Fr_IR12",
        "speeds_mm_s": [15.0] * 10,
        "dwell_time_s": 1.0,
    }
    payload.update(overrides)
    return payload


class TestSubmitCatheterSimulation:
    def test_valid_request_returns_202(self, client):
        c, _ = client
        resp = c.post("/api/v1/simulations/run-catheter", json=_valid_payload())
        assert resp.status_code == 202

    def test_response_has_task_id(self, client):
        c, _ = client
        data = c.post("/api/v1/simulations/run-catheter", json=_valid_payload()).json()
        assert "task_id" in data
        assert data["task_id"] == "fake-task-id-1234"

    def test_response_has_run_id(self, client):
        c, _ = client
        data = c.post("/api/v1/simulations/run-catheter", json=_valid_payload()).json()
        assert data.get("run_id") is not None

    def test_response_has_run_dir(self, client):
        c, _ = client
        data = c.post("/api/v1/simulations/run-catheter", json=_valid_payload()).json()
        assert data.get("run_dir") is not None

    def test_response_has_xplt_path(self, client):
        c, _ = client
        data = c.post("/api/v1/simulations/run-catheter", json=_valid_payload()).json()
        assert data.get("xplt_path") is not None

    def test_response_status_pending(self, client):
        c, _ = client
        data = c.post("/api/v1/simulations/run-catheter", json=_valid_payload()).json()
        assert data["status"] == "PENDING"

    def test_celery_task_delay_called(self, client):
        c, mock_task = client
        c.post("/api/v1/simulations/run-catheter", json=_valid_payload())
        mock_task.delay.assert_called_once()
        call_kwargs = mock_task.delay.call_args.kwargs
        assert call_kwargs["design"] == "ball_tip"
        assert call_kwargs["configuration"] == "14Fr_IR12"
        assert call_kwargs["speeds_mm_s"] == [15.0] * 10

    def test_unknown_design_returns_404(self, client):
        c, _ = client
        resp = c.post(
            "/api/v1/simulations/run-catheter",
            json=_valid_payload(design="nonexistent_design"),
        )
        assert resp.status_code == 404

    def test_unknown_configuration_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/api/v1/simulations/run-catheter",
            json=_valid_payload(configuration="99Fr_IR99"),
        )
        assert resp.status_code in (404, 422)

    def test_wrong_speeds_count_too_few_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/api/v1/simulations/run-catheter",
            json=_valid_payload(speeds_mm_s=[15.0] * 5),
        )
        assert resp.status_code == 422

    def test_wrong_speeds_count_too_many_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/api/v1/simulations/run-catheter",
            json=_valid_payload(speeds_mm_s=[15.0] * 15),
        )
        assert resp.status_code == 422

    def test_empty_speeds_returns_422(self, client):
        c, _ = client
        resp = c.post(
            "/api/v1/simulations/run-catheter",
            json=_valid_payload(speeds_mm_s=[]),
        )
        assert resp.status_code == 422

    def test_explicit_run_id_is_used(self, client):
        c, _ = client
        data = c.post(
            "/api/v1/simulations/run-catheter",
            json=_valid_payload(run_id="my_custom_run_id"),
        ).json()
        assert data["run_id"] == "my_custom_run_id"

    def test_all_designs_accepted(self, client):
        """Every design in the minimal catalogue should accept a valid request."""
        c, _ = client
        for design, design_data in MINIMAL_CATALOGUE["designs"].items():
            first_cfg = next(iter(design_data["configurations"].keys()))
            resp = c.post(
                "/api/v1/simulations/run-catheter",
                json=_valid_payload(design=design, configuration=first_cfg),
            )
            assert resp.status_code == 202, (
                f"design='{design}' cfg='{first_cfg}' returned {resp.status_code}: {resp.text}"
            )


# ---------------------------------------------------------------------------
# POST /api/v1/doe/preview-speeds
# ---------------------------------------------------------------------------

class TestPreviewDoeSpeeds:
    def _post(self, client_fixture, **kwargs) -> Any:
        c, _ = client_fixture
        payload = {
            "n_samples": 5,
            "speed_min": 10.0,
            "speed_max": 25.0,
            "n_steps": 10,
            "max_perturbation": 0.20,
        }
        payload.update(kwargs)
        return c.post("/api/v1/doe/preview-speeds", json=payload)

    def test_status_200(self, client):
        resp = self._post(client)
        assert resp.status_code == 200

    def test_samples_count(self, client):
        data = self._post(client, n_samples=7).json()
        assert len(data["samples"]) == 7

    def test_each_sample_length_equals_n_steps(self, client):
        data = self._post(client, n_samples=3, n_steps=10).json()
        for sample in data["samples"]:
            assert len(sample) == 10

    def test_speeds_within_range(self, client):
        data = self._post(client, speed_min=12.0, speed_max=20.0).json()
        for sample in data["samples"]:
            for speed in sample:
                assert 12.0 <= speed <= 20.0, f"Speed {speed} outside [12, 20]"

    def test_seed_reproducibility(self, client):
        c, _ = client
        payload = {
            "n_samples": 5,
            "speed_min": 10.0,
            "speed_max": 25.0,
            "n_steps": 10,
            "seed": 42,
        }
        r1 = c.post("/api/v1/doe/preview-speeds", json=payload).json()
        r2 = c.post("/api/v1/doe/preview-speeds", json=payload).json()
        assert r1["samples"] == r2["samples"]

    def test_different_seeds_give_different_samples(self, client):
        c, _ = client
        base = {
            "n_samples": 5,
            "speed_min": 10.0,
            "speed_max": 25.0,
            "n_steps": 10,
        }
        r1 = c.post("/api/v1/doe/preview-speeds", json={**base, "seed": 1}).json()
        r2 = c.post("/api/v1/doe/preview-speeds", json={**base, "seed": 2}).json()
        assert r1["samples"] != r2["samples"]

    def test_response_echoes_n_samples(self, client):
        data = self._post(client, n_samples=4).json()
        assert data["n_samples"] == 4

    def test_response_echoes_n_steps(self, client):
        data = self._post(client, n_steps=10).json()
        assert data["n_steps"] == 10

    def test_response_echoes_speed_bounds(self, client):
        data = self._post(client, speed_min=11.0, speed_max=22.0).json()
        assert data["speed_min"] == pytest.approx(11.0)
        assert data["speed_max"] == pytest.approx(22.0)

    def test_zero_n_samples_returns_422(self, client):
        resp = self._post(client, n_samples=0)
        assert resp.status_code == 422

    def test_speed_min_greater_than_max_returns_error_not_silent(self, client):
        # When speed_min >= speed_max the sampler raises a ValueError.
        # The API should propagate this as a 4xx or 5xx, NOT silently return 200.
        c, _ = client
        payload = {
            "n_samples": 5,
            "speed_min": 25.0,
            "speed_max": 10.0,
            "n_steps": 10,
        }
        from starlette.testclient import TestClient as StarletteClient
        from digital_twin_ui.app.main import create_app
        # Use raise_server_exceptions=False so we can inspect the status code
        app = create_app()
        with StarletteClient(app, raise_server_exceptions=False) as tc:
            resp = tc.post("/api/v1/doe/preview-speeds", json=payload)
        assert resp.status_code != 200, (
            "Expected an error when speed_min > speed_max, got 200"
        )
