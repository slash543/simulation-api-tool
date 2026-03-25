"""
Unit tests for digital_twin_ui.app.api.routes.surrogate

Tests the FastAPI endpoints using TestClient with mocked surrogate module.

Covered endpoints:
  GET  /api/v1/surrogate/models
  POST /api/v1/surrogate/predict
  POST /api/v1/surrogate/csar
  POST /api/v1/surrogate/predict-vtp
  POST /api/v1/surrogate/csar-from-vtp
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    from digital_twin_ui.app.main import create_app
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /surrogate/models
# ---------------------------------------------------------------------------

class TestListSurrogateModels:
    def test_returns_200(self, client):
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.list_mlflow_runs",
                  return_value=[]),
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=False),
        ):
            resp = client.get("/api/v1/surrogate/models")
        assert resp.status_code == 200

    def test_response_shape(self, client):
        fake_run = {
            "run_id": "abc123",
            "status": "FINISHED",
            "start_time": 1700000000000,
            "metrics": {"best_val_loss": 0.01},
            "params": {"model_type": "MLP"},
            "artifact_uri": "mlruns/1/abc123/artifacts",
        }
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.list_mlflow_runs",
                  return_value=[fake_run]),
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
        ):
            resp = client.get("/api/v1/surrogate/models")

        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert "latest_available" in body
        assert body["latest_available"] is True
        assert body["models"][0]["run_id"] == "abc123"

    def test_mlflow_error_returns_empty(self, client):
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.list_mlflow_runs",
                  return_value=[{"error": "connection refused"}]),
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=False),
        ):
            resp = client.get("/api/v1/surrogate/models")
        assert resp.status_code == 200
        # Error entries should be filtered out → empty models list
        assert resp.json()["models"] == []


# ---------------------------------------------------------------------------
# POST /surrogate/predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def _mock_predictor(self, cp_values):
        pred = MagicMock()
        pred.predict.return_value = np.array(cp_values, dtype=np.float32)
        pred.model_dir = Path("/fake/model/dir")
        return pred

    def test_predict_inline_facets(self, client):
        facets = [
            {"centroid_x": 0.0, "centroid_y": 0.0, "centroid_z": 10.0,
             "facet_area": 1.0, "insertion_depth": 50.0},
            {"centroid_x": 1.0, "centroid_y": 0.0, "centroid_z": 20.0,
             "facet_area": 1.0, "insertion_depth": 50.0},
        ]
        pred = self._mock_predictor([0.1, 0.2])
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
            patch("digital_twin_ui.app.api.routes.surrogate._get_predictor_cached",
                  return_value=pred),
        ):
            resp = client.post("/api/v1/surrogate/predict", json={"facets": facets})

        assert resp.status_code == 200
        body = resp.json()
        assert body["n_facets"] == 2
        assert len(body["contact_pressure_MPa"]) == 2

    def test_predict_no_input_returns_422(self, client):
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
            patch("digital_twin_ui.app.api.routes.surrogate._get_predictor_cached",
                  return_value=MagicMock()),
        ):
            resp = client.post("/api/v1/surrogate/predict", json={})
        assert resp.status_code == 422

    def test_predict_model_unavailable_returns_503(self, client):
        with patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                   return_value=False):
            resp = client.post(
                "/api/v1/surrogate/predict",
                json={"facets": [
                    {"centroid_x": 0.0, "centroid_y": 0.0, "centroid_z": 0.0,
                     "facet_area": 1.0, "insertion_depth": 50.0}
                ]},
            )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /surrogate/csar
# ---------------------------------------------------------------------------

class TestCSAREndpoint:
    def _make_csar_df(self):
        return pd.DataFrame(
            {
                "insertion_depth_mm": [50.0, 100.0],
                "zmin_mm": [0.0, 0.0],
                "zmax_mm": [50.0, 50.0],
                "band_label": ["tip", "tip"],
                "csar": [0.3, 0.5],
                "contact_area_mm2": [15.0, 25.0],
                "total_area_mm2": [50.0, 50.0],
                "n_contact_facets": [30, 50],
                "n_total_facets": [100, 100],
                "mean_cp_MPa": [0.01, 0.02],
                "max_cp_MPa": [0.05, 0.08],
            }
        )

    def test_csar_returns_200(self, client, tmp_path):
        # Write reference facets CSV
        csv_path = tmp_path / "reference_facets.csv"
        df = pd.DataFrame(
            {
                "centroid_x": [0.0, 1.0],
                "centroid_y": [0.0, 0.0],
                "centroid_z": [10.0, 20.0],
                "facet_area": [1.0, 1.0],
            }
        )
        df.to_csv(csv_path, index=False)

        pred = MagicMock()
        pred.predict_at_depth.return_value = np.array([0.1, 0.2], dtype=np.float32)

        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
            patch("digital_twin_ui.app.api.routes.surrogate._get_predictor_cached",
                  return_value=pred),
            patch("digital_twin_ui.app.api.routes.surrogate._default_reference_facets_path",
                  return_value=csv_path),
        ):
            resp = client.post(
                "/api/v1/surrogate/csar",
                json={
                    "z_bands": [{"zmin": 0, "zmax": 30, "label": "tip"}],
                    "insertion_depths_mm": [50.0, 100.0],
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "bands" in body
        assert "insertion_depths_mm" in body

    def test_csar_no_reference_facets_returns_404(self, client, tmp_path):
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
            patch("digital_twin_ui.app.api.routes.surrogate._get_predictor_cached",
                  return_value=MagicMock()),
            patch("digital_twin_ui.app.api.routes.surrogate._default_reference_facets_path",
                  return_value=tmp_path / "nonexistent.csv"),
        ):
            resp = client.post(
                "/api/v1/surrogate/csar",
                json={"z_bands": [{"zmin": 0, "zmax": 50, "label": "tip"}]},
            )
        assert resp.status_code == 404

    def test_csar_empty_z_bands_returns_422(self, client):
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
        ):
            resp = client.post(
                "/api/v1/surrogate/csar",
                json={"z_bands": []},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /surrogate/predict-vtp
# ---------------------------------------------------------------------------

class TestPredictVTPEndpoint:
    def _make_vtp_data(self):
        from digital_twin_ui.surrogate.vtp_processor import VTPData
        return VTPData(
            points=np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32),
            connectivity=np.array([0,1,2], dtype=np.int32),
            offsets=np.array([3], dtype=np.int32),
            face_ids=np.array([1], dtype=np.int32),
            areas=np.array([0.5], dtype=np.float32),
            contact_pressure=np.array([0.0], dtype=np.float32),
        )

    def test_predict_vtp_missing_file_returns_404(self, client):
        pred = MagicMock()
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
            patch("digital_twin_ui.app.api.routes.surrogate._get_predictor_cached",
                  return_value=pred),
            patch("digital_twin_ui.app.api.routes.surrogate._resolve_path",
                  return_value=Path("/nonexistent/file.vtp")),
        ):
            resp = client.post(
                "/api/v1/surrogate/predict-vtp",
                json={"vtp_path": "/app/surrogate_data/results/bad.vtp",
                      "insertion_depth_mm": 100.0},
            )
        assert resp.status_code == 404

    def test_predict_vtp_model_unavailable_returns_503(self, client):
        with patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                   return_value=False):
            resp = client.post(
                "/api/v1/surrogate/predict-vtp",
                json={"vtp_path": "/app/surrogate_data/results/test.vtp",
                      "insertion_depth_mm": 100.0},
            )
        assert resp.status_code == 503

    def test_predict_vtp_success(self, client, tmp_path):
        from digital_twin_ui.surrogate.vtp_processor import VTPProcessor

        # Write a real VTP file
        vtp_data = self._make_vtp_data()
        vtp_path = tmp_path / "test.vtp"
        VTPProcessor.write(vtp_path, vtp_data)

        pred = MagicMock()
        pred.predict_at_depth.return_value = np.array([0.05], dtype=np.float32)

        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
            patch("digital_twin_ui.app.api.routes.surrogate._get_predictor_cached",
                  return_value=pred),
            patch("digital_twin_ui.app.api.routes.surrogate._resolve_path",
                  return_value=vtp_path),
        ):
            resp = client.post(
                "/api/v1/surrogate/predict-vtp",
                json={"vtp_path": str(vtp_path), "insertion_depth_mm": 50.0},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["n_faces"] == 1
        assert body["insertion_depth_mm"] == 50.0
        assert "output_vtp_path" in body


# ---------------------------------------------------------------------------
# POST /surrogate/csar-from-vtp
# ---------------------------------------------------------------------------

class TestCSARFromVTPEndpoint:
    def test_csar_from_vtp_missing_file_returns_404(self, client):
        pred = MagicMock()
        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
            patch("digital_twin_ui.app.api.routes.surrogate._get_predictor_cached",
                  return_value=pred),
            patch("digital_twin_ui.app.api.routes.surrogate._resolve_path",
                  return_value=Path("/nonexistent/file.vtp")),
        ):
            resp = client.post(
                "/api/v1/surrogate/csar-from-vtp",
                json={
                    "vtp_path": "/app/surrogate_data/results/bad.vtp",
                    "z_bands": [{"zmin": 0, "zmax": 50, "label": "tip"}],
                },
            )
        assert resp.status_code == 404

    def test_csar_from_vtp_success(self, client, tmp_path):
        from digital_twin_ui.surrogate.vtp_processor import VTPData, VTPProcessor

        # Write a real VTP file
        vtp_data = VTPData(
            points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,20],[1,0,20],[0,1,20]],
                            dtype=np.float32),
            connectivity=np.array([0,1,2, 3,4,5], dtype=np.int32),
            offsets=np.array([3,6], dtype=np.int32),
            face_ids=np.array([1,2], dtype=np.int32),
            areas=np.array([0.5, 0.5], dtype=np.float32),
            contact_pressure=np.array([0.0, 0.0], dtype=np.float32),
        )
        vtp_path = tmp_path / "geom.vtp"
        VTPProcessor.write(vtp_path, vtp_data)

        pred = MagicMock()
        pred.predict_at_depth.return_value = np.array([0.1, 0.0], dtype=np.float32)

        with (
            patch("digital_twin_ui.app.api.routes.surrogate.is_model_available",
                  return_value=True),
            patch("digital_twin_ui.app.api.routes.surrogate._get_predictor_cached",
                  return_value=pred),
            patch("digital_twin_ui.app.api.routes.surrogate._resolve_path",
                  return_value=vtp_path),
        ):
            resp = client.post(
                "/api/v1/surrogate/csar-from-vtp",
                json={
                    "vtp_path": str(vtp_path),
                    "z_bands": [{"zmin": -5, "zmax": 5, "label": "lower"},
                                {"zmin": 15, "zmax": 25, "label": "upper"}],
                    "insertion_depths_mm": [50.0, 100.0],
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "bands" in body
        assert body["n_facets"] == 2
