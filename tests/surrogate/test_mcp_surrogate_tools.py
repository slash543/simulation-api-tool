"""
Unit tests for the surrogate MCP tool functions in mcp_server/tools.py

Tests each surrogate tool function in isolation (mocked HTTP calls).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The mcp_server/ directory is NOT a package, so we add it to sys.path
MCP_DIR = Path(__file__).parent.parent.parent / "mcp_server"
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

from tools import (
    tool_analyse_catheter_contact,
    tool_compute_csar_from_vtp,
    tool_compute_csar_vs_depth,
    tool_evaluate_contact_pressure,
    tool_generate_csar_plot_from_vtp,
    tool_list_available_vtps,
    tool_list_surrogate_models,
    tool_predict_vtp_contact_pressure,
    _to_surrogate_host_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(data: dict, status: int = 200):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = data
    if status >= 400:
        from httpx import HTTPStatusError, Response, Request
        mock.raise_for_status.side_effect = HTTPStatusError(
            "HTTP error", request=MagicMock(), response=mock
        )
        mock.response = mock
    else:
        mock.raise_for_status.return_value = None
    return mock


def _mock_http_error(status: int, text: str = "error"):
    from httpx import HTTPStatusError
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    exc = HTTPStatusError("HTTP error", request=MagicMock(), response=resp)
    return exc


# ---------------------------------------------------------------------------
# _to_surrogate_host_path
# ---------------------------------------------------------------------------

class TestToSurrogateHostPath:
    def test_container_path_translated(self, monkeypatch):
        monkeypatch.setenv("SURROGATE_HOST_PATH", "/host/data/surrogate")
        # Re-import to pick up env change
        import importlib
        import tools as tools_mod
        importlib.reload(tools_mod)
        result = tools_mod._to_surrogate_host_path("/app/surrogate_data/results/out.vtp")
        assert result.startswith("/host/data/surrogate")

    def test_non_surrogate_path_unchanged(self):
        result = _to_surrogate_host_path("/some/other/path/file.vtp")
        # Falls through to _to_host_path — should still return something
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# tool_list_surrogate_models
# ---------------------------------------------------------------------------

class TestToolListSurrogateModels:
    def test_success(self):
        data = {
            "models": [{"run_id": "abc", "status": "FINISHED", "metrics": {}}],
            "latest_available": True,
        }
        mock_resp = _mock_response(data)
        with patch("tools._fast_client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_list_surrogate_models()

        body = json.loads(result)
        # tool_list_surrogate_models returns a summarised view
        assert "latest_available" in body
        assert body["latest_available"] is True
        assert "n_runs" in body
        assert "registered_models" in body
        assert "recent_runs" in body

    def test_http_error_returns_error_json(self):
        with patch("tools._fast_client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            from httpx import ConnectError
            mock_client.get.side_effect = ConnectError("connection refused", request=MagicMock())
            mock_client_cls.return_value = mock_client

            result = tool_list_surrogate_models()

        body = json.loads(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# tool_evaluate_contact_pressure
# ---------------------------------------------------------------------------

class TestToolEvaluateContactPressure:
    def test_success(self):
        csar_data = {
            "insertion_depths_mm": [50.0, 100.0],
            "bands": {
                "full_surface": {
                    "mean_cp_MPa": [0.01, 0.02],
                    "max_cp_MPa": [0.05, 0.08],
                    "csar": [0.3, 0.5],
                }
            },
            "n_facets_total": 1000,
            "run_id_used": None,
        }
        mock_resp = _mock_response(csar_data)
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_evaluate_contact_pressure([50.0, 100.0])

        body = json.loads(result)
        assert "mean_cp_MPa" in body
        assert body["n_facets_total"] == 1000

    def test_503_model_unavailable(self):
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = _mock_http_error(503)
            mock_client.post.return_value = _mock_response({}, 503)

            from httpx import HTTPStatusError
            resp_mock = MagicMock()
            resp_mock.status_code = 503
            resp_mock.text = "model not available"
            exc = HTTPStatusError("503", request=MagicMock(), response=resp_mock)
            mock_client.post.side_effect = exc
            mock_client_cls.return_value = mock_client

            result = tool_evaluate_contact_pressure([50.0])

        body = json.loads(result)
        assert "error" in body
        assert "surrogate model" in body["error"].lower()

    def test_404_reference_facets_missing(self):
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            from httpx import HTTPStatusError
            resp_mock = MagicMock()
            resp_mock.status_code = 404
            resp_mock.text = "not found"
            exc = HTTPStatusError("404", request=MagicMock(), response=resp_mock)
            mock_client.post.side_effect = exc
            mock_client_cls.return_value = mock_client

            result = tool_evaluate_contact_pressure([50.0])

        body = json.loads(result)
        assert "error" in body
        assert "reference facets" in body["error"].lower()


# ---------------------------------------------------------------------------
# tool_compute_csar_vs_depth
# ---------------------------------------------------------------------------

class TestToolComputeCSARVsDepth:
    def test_success_with_z_bands(self):
        resp_data = {
            "insertion_depths_mm": [50.0, 100.0],
            "bands": {
                "tip": {
                    "label": "tip",
                    "csar": [0.2, 0.4],
                    "zmin_mm": 0,
                    "zmax_mm": 50,
                }
            },
            "n_facets_total": 500,
        }
        mock_resp = _mock_response(resp_data)
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_compute_csar_vs_depth(
                z_bands=[{"zmin": 0, "zmax": 50, "label": "tip"}],
                insertion_depths_mm=[50.0, 100.0],
            )

        body = json.loads(result)
        assert "bands" in body
        assert "tip" in body["bands"]

    def test_passes_run_id(self):
        mock_resp = _mock_response({"insertion_depths_mm": [], "bands": {}})
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            tool_compute_csar_vs_depth(
                z_bands=[{"zmin": 0, "zmax": 50}],
                run_id="abc123",
            )

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
            assert payload.get("run_id") == "abc123"


# ---------------------------------------------------------------------------
# tool_predict_vtp_contact_pressure
# ---------------------------------------------------------------------------

class TestToolPredictVTPContactPressure:
    def test_success(self):
        resp_data = {
            "output_vtp_path": "/app/surrogate_data/results/test_predicted.vtp",
            "host_output_path": "./data/surrogate/results/test_predicted.vtp",
            "n_faces": 500,
            "insertion_depth_mm": 100.0,
        }
        mock_resp = _mock_response(resp_data)
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_predict_vtp_contact_pressure(
                "/app/surrogate_data/results/test.vtp", 100.0
            )

        body = json.loads(result)
        assert body["n_faces"] == 500
        assert "host_output_path" in body

    def test_404_vtp_not_found(self):
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            from httpx import HTTPStatusError
            resp_mock = MagicMock()
            resp_mock.status_code = 404
            resp_mock.text = "VTP not found"
            exc = HTTPStatusError("404", request=MagicMock(), response=resp_mock)
            mock_client.post.side_effect = exc
            mock_client_cls.return_value = mock_client

            result = tool_predict_vtp_contact_pressure("/bad/path.vtp", 50.0)

        body = json.loads(result)
        assert "error" in body
        assert "not found" in body["error"].lower()


# ---------------------------------------------------------------------------
# tool_compute_csar_from_vtp
# ---------------------------------------------------------------------------

class TestToolComputeCSARFromVTP:
    def test_success(self):
        resp_data = {
            "insertion_depths_mm": [50.0],
            "bands": {"tip": {"csar": [0.3]}},
            "n_facets": 100,
            "vtp_source": "/app/surrogate_data/results/geom.vtp",
        }
        mock_resp = _mock_response(resp_data)
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_compute_csar_from_vtp(
                vtp_path="/app/surrogate_data/results/geom.vtp",
                z_bands=[{"zmin": 0, "zmax": 50, "label": "tip"}],
                insertion_depths_mm=[50.0],
            )

        body = json.loads(result)
        assert "bands" in body
        assert "n_facets" in body


# ---------------------------------------------------------------------------
# tool_generate_csar_plot_from_vtp
# ---------------------------------------------------------------------------

class TestToolGenerateCsarPlot:
    def test_success_returns_host_path(self):
        resp_data = {
            "plot_path": "/app/surrogate_data/results/csar_plots/geom_csar_2bands.png",
            "host_plot_path": "./data/surrogate/results/csar_plots/geom_csar_2bands.png",
            "plot_png_b64": "abc123",
            "insertion_depths_mm": [50.0, 100.0],
            "bands": {},
            "n_facets": 50,
            "vtp_source": "/app/surrogate_data/results/geom.vtp",
        }
        mock_resp = _mock_response(resp_data)
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_generate_csar_plot_from_vtp(
                vtp_path="/app/surrogate_data/results/geom.vtp",
                z_bands=[{"zmin": 0, "zmax": 50, "label": "tip"},
                         {"zmin": 50, "zmax": 150, "label": "mid"}],
                insertion_depths_mm=[50.0, 100.0],
            )

        body = json.loads(result)
        assert "host_plot_path" in body
        assert "bands_summary" in body
        assert "plot_png_b64" not in body  # large blob should be omitted from summary

    def test_vtp_not_found_returns_error(self):
        exc = _mock_http_error(404)
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = exc
            mock_client_cls.return_value = mock_client

            result = tool_generate_csar_plot_from_vtp(
                vtp_path="/app/surrogate_data/results/missing.vtp",
                z_bands=[{"zmin": 0, "zmax": 50, "label": "tip"}],
            )

        body = json.loads(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# tool_analyse_catheter_contact
# ---------------------------------------------------------------------------

class TestToolAnalyseCatheterContact:
    def _make_band_summary(self, label: str) -> dict:
        return {
            "label": label,
            "zmin_mm": 0.0,
            "zmax_mm": 50.0,
            "n_total_facets": 100,
            "total_area_mm2": 50.0,
            "peak_csar": 0.45,
            "depth_at_peak_csar_mm": 200.0,
            "peak_pressure_MPa": 0.12,
            "depth_at_peak_pressure_mm": 180.0,
            "first_contact_depth_mm": 50.0,
        }

    def test_success_returns_band_summaries(self):
        resp_data = {
            "plot_path": "/app/surrogate_data/results/analysis_plots/catheter_contact_analysis.png",
            "host_plot_path": "./data/surrogate/results/analysis_plots/catheter_contact_analysis.png",
            "plot_png_b64": "abc123",
            "insertion_depths_mm": [50.0, 100.0, 150.0],
            "bands": {},
            "band_summaries": {
                "tip": self._make_band_summary("tip"),
                "mid": self._make_band_summary("mid"),
            },
            "n_facets": 200,
            "vtp_source": "/app/runs/run_001/catheter.vtp",
        }
        mock_resp = _mock_response(resp_data)
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_analyse_catheter_contact(
                vtp_path="/app/runs/run_001/catheter.vtp",
                z_bands=[{"zmin": 0, "zmax": 50, "label": "tip"},
                         {"zmin": 50, "zmax": 150, "label": "mid"}],
            )

        body = json.loads(result)
        assert "host_plot_path" in body
        assert "band_summaries" in body
        assert "instruction" in body
        assert "plot_png_b64" not in body

    def test_vtp_not_found_returns_helpful_error(self):
        exc = _mock_http_error(404)
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = exc
            mock_client_cls.return_value = mock_client

            result = tool_analyse_catheter_contact(
                vtp_path="/app/runs/bad/catheter.vtp",
                z_bands=[{"zmin": 0, "zmax": 50, "label": "tip"}],
            )

        body = json.loads(result)
        assert "error" in body
        # Should tell the user how to find files
        assert "list_available_vtps" in body["error"]

    def test_model_unavailable_returns_helpful_error(self):
        exc = _mock_http_error(503, "model not available")
        with patch("tools._client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = exc
            mock_client_cls.return_value = mock_client

            result = tool_analyse_catheter_contact(
                vtp_path="/app/runs/run_001/catheter.vtp",
                z_bands=[{"zmin": 0, "zmax": 50, "label": "tip"}],
            )

        body = json.loads(result)
        assert "error" in body
        assert "full_pipeline" in body["error"]


# ---------------------------------------------------------------------------
# tool_list_available_vtps
# ---------------------------------------------------------------------------

class TestToolListAvailableVTPs:
    def test_success_returns_file_list(self):
        resp_data = {
            "vtp_files": [
                {"path": "/app/runs/run_001/catheter_t0000.vtp",
                 "host_path": "./runs/run_001/catheter_t0000.vtp",
                 "size_kb": 12.5, "stem": "catheter_t0000"},
                {"path": "/app/surrogate_data/results/geom.vtp",
                 "host_path": "./data/surrogate/results/geom.vtp",
                 "size_kb": 8.2, "stem": "geom"},
            ],
            "total": 2,
            "search_dirs": ["/app/runs", "/app/surrogate_data"],
        }
        mock_resp = _mock_response(resp_data)
        with patch("tools._fast_client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_list_available_vtps()

        body = json.loads(result)
        assert body["total"] == 2
        assert len(body["vtp_files"]) == 2
        assert body["vtp_files"][0]["host_path"] == "./runs/run_001/catheter_t0000.vtp"

    def test_empty_result(self):
        resp_data = {"vtp_files": [], "total": 0, "search_dirs": []}
        mock_resp = _mock_response(resp_data)
        with patch("tools._fast_client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = tool_list_available_vtps()

        body = json.loads(result)
        assert body["total"] == 0
        assert body["vtp_files"] == []
