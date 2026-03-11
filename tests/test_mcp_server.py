"""
Tests for the MCP server tool implementations.

These tests mock httpx so no running API is required.
They cover happy paths, HTTP error codes, and network failures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Make mcp_server importable (it lives outside the main package)
# ---------------------------------------------------------------------------
MCP_SERVER_DIR = Path(__file__).parent.parent / "mcp_server"
if str(MCP_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER_DIR))

from tools import (  # noqa: E402
    tool_get_doe_status,
    tool_get_task_status,
    tool_health_check,
    tool_predict_pressure,
    tool_predict_pressure_batch,
    tool_run_doe_campaign,
    tool_run_simulation,
    tool_submit_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int, body: dict) -> MagicMock:
    """Build a fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _parse(result: str) -> dict:
    return json.loads(result)


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_healthy(self) -> None:
        body = {"status": "healthy", "version": "0.1.0"}
        mock = _mock_response(200, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_health_check())
        assert result["status"] == "healthy"

    def test_api_unreachable(self) -> None:
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get.side_effect = httpx.ConnectError("refused")
            result = _parse(tool_health_check())
        assert "error" in result
        assert "unreachable" in result["error"]


# ---------------------------------------------------------------------------
# TestRunSimulation
# ---------------------------------------------------------------------------

class TestRunSimulation:
    def test_success(self) -> None:
        body = {
            "run_id": "run_001",
            "speed_mm_s": 5.0,
            "peak_contact_pressure_pa": 1234.5,
            "status": "completed",
            "duration_s": 120.0,
        }
        mock = _mock_response(200, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_run_simulation(5.0))
        assert result["run_id"] == "run_001"
        assert result["peak_contact_pressure_pa"] == pytest.approx(1234.5)

    def test_sends_correct_payload(self) -> None:
        mock = _mock_response(200, {"run_id": "x", "peak_contact_pressure_pa": 0.0,
                                    "speed_mm_s": 3.5, "status": "completed", "duration_s": 1.0})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_simulation(3.5)
            client_inst.post.assert_called_once_with(
                "/simulations/run", json={"speed_mm_s": 3.5, "extract": False}
            )

    def test_http_error_returns_error_json(self) -> None:
        mock = _mock_response(500, {"detail": "server error"})
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_run_simulation(5.0))
        assert "error" in result
        assert "500" in result["error"]

    def test_network_error_returns_error_json(self) -> None:
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.side_effect = (
                httpx.ConnectError("timeout")
            )
            result = _parse(tool_run_simulation(5.0))
        assert "error" in result


# ---------------------------------------------------------------------------
# TestSubmitSimulation
# ---------------------------------------------------------------------------

class TestSubmitSimulation:
    def test_returns_task_id(self) -> None:
        body = {"task_id": "celery-abc-123", "status": "PENDING"}
        mock = _mock_response(202, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_submit_simulation(7.0))
        assert result["task_id"] == "celery-abc-123"
        assert result["status"] == "PENDING"

    def test_sends_to_async_endpoint(self) -> None:
        mock = _mock_response(202, {"task_id": "x", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_submit_simulation(6.0)
            client_inst.post.assert_called_once_with(
                "/simulations/run", json={"speed_mm_s": 6.0}
            )


# ---------------------------------------------------------------------------
# TestGetTaskStatus
# ---------------------------------------------------------------------------

class TestGetTaskStatus:
    def test_pending_status(self) -> None:
        body = {"task_id": "abc", "status": "PENDING", "result": None}
        mock = _mock_response(200, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_get_task_status("abc"))
        assert result["status"] == "PENDING"

    def test_success_status_with_result(self) -> None:
        body = {
            "task_id": "abc",
            "status": "SUCCESS",
            "result": {"peak_contact_pressure_pa": 999.0},
        }
        mock = _mock_response(200, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_get_task_status("abc"))
        assert result["status"] == "SUCCESS"
        assert result["result"]["peak_contact_pressure_pa"] == 999.0

    def test_uses_correct_url(self) -> None:
        mock = _mock_response(200, {"task_id": "xyz", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get.return_value = mock
            tool_get_task_status("xyz")
            client_inst.get.assert_called_once_with("/simulations/xyz")


# ---------------------------------------------------------------------------
# TestRunDoeCampaign
# ---------------------------------------------------------------------------

class TestRunDoeCampaign:
    def test_returns_task_id(self) -> None:
        body = {"task_id": "doe-task-001", "status": "PENDING"}
        mock = _mock_response(202, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_run_doe_campaign(10, 2.0, 8.0))
        assert result["task_id"] == "doe-task-001"

    def test_default_sampler_is_lhs(self) -> None:
        mock = _mock_response(202, {"task_id": "x", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(5, 3.0, 7.0)
            payload = client_inst.post.call_args[1]["json"]
        assert payload["sampler"] == "lhs"

    def test_seed_included_when_provided(self) -> None:
        mock = _mock_response(202, {"task_id": "x", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(5, 3.0, 7.0, seed=42)
            payload = client_inst.post.call_args[1]["json"]
        assert payload["seed"] == 42

    def test_seed_omitted_when_none(self) -> None:
        mock = _mock_response(202, {"task_id": "x", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(5, 3.0, 7.0, seed=None)
            payload = client_inst.post.call_args[1]["json"]
        assert "seed" not in payload

    @pytest.mark.parametrize("sampler", ["lhs", "sobol", "uniform"])
    def test_all_samplers_accepted(self, sampler: str) -> None:
        mock = _mock_response(202, {"task_id": "x", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            result = _parse(tool_run_doe_campaign(4, 2.0, 6.0, sampler=sampler))
        assert "task_id" in result

    def test_http_error_returns_error_json(self) -> None:
        mock = _mock_response(422, {"detail": "speed_min must be less than speed_max"})
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_run_doe_campaign(5, 8.0, 2.0))  # invalid range
        assert "error" in result


# ---------------------------------------------------------------------------
# TestGetDoeStatus
# ---------------------------------------------------------------------------

class TestGetDoeStatus:
    def test_returns_status(self) -> None:
        body = {"task_id": "doe-001", "status": "STARTED", "result": None}
        mock = _mock_response(200, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_get_doe_status("doe-001"))
        assert result["status"] == "STARTED"

    def test_completed_result(self) -> None:
        body = {
            "task_id": "doe-001",
            "status": "SUCCESS",
            "result": {"n_runs": 10, "successful": 10, "speeds": [2.0, 3.0]},
        }
        mock = _mock_response(200, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_get_doe_status("doe-001"))
        assert result["result"]["n_runs"] == 10


# ---------------------------------------------------------------------------
# TestPredictPressure
# ---------------------------------------------------------------------------

class TestPredictPressure:
    def test_success(self) -> None:
        body = {"speed_mm_s": 5.0, "predicted_pressure_pa": 876.3}
        mock = _mock_response(200, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_predict_pressure(5.0))
        assert result["predicted_pressure_pa"] == pytest.approx(876.3)

    def test_model_not_available_503(self) -> None:
        mock = _mock_response(503, {"detail": "model not found"})
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_predict_pressure(5.0))
        assert "error" in result
        assert "DOE" in result["error"] or "model" in result["error"].lower()

    def test_sends_correct_payload(self) -> None:
        mock = _mock_response(200, {"speed_mm_s": 4.0, "predicted_pressure_pa": 500.0})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_predict_pressure(4.0)
            client_inst.post.assert_called_once_with(
                "/ml/predict", json={"speed_mm_s": 4.0}
            )


# ---------------------------------------------------------------------------
# TestPredictPressureBatch
# ---------------------------------------------------------------------------

class TestPredictPressureBatch:
    def test_success(self) -> None:
        body = {
            "predictions": [
                {"speed_mm_s": 3.0, "predicted_pressure_pa": 500.0},
                {"speed_mm_s": 5.0, "predicted_pressure_pa": 800.0},
                {"speed_mm_s": 7.0, "predicted_pressure_pa": 1100.0},
            ]
        }
        mock = _mock_response(200, body)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_predict_pressure_batch([3.0, 5.0, 7.0]))
        assert len(result["predictions"]) == 3

    def test_model_not_available_503(self) -> None:
        mock = _mock_response(503, {"detail": "model not found"})
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_predict_pressure_batch([5.0]))
        assert "error" in result

    def test_sends_list_payload(self) -> None:
        mock = _mock_response(200, {"predictions": []})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_predict_pressure_batch([2.0, 4.0])
            client_inst.post.assert_called_once_with(
                "/ml/predict/batch", json={"speeds_mm_s": [2.0, 4.0]}
            )

    def test_empty_list(self) -> None:
        mock = _mock_response(200, {"predictions": []})
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(tool_predict_pressure_batch([]))
        assert result["predictions"] == []


# ---------------------------------------------------------------------------
# TestMcpServerRegistration
# ---------------------------------------------------------------------------

class TestMcpServerRegistration:
    """Verify the FastMCP app registers all expected tools."""

    def test_all_tools_registered(self) -> None:
        from server import mcp  # noqa: PLC0415

        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        expected = {
            "health_check",
            "run_simulation",
            "submit_simulation",
            "get_task_status",
            "run_doe_campaign",
            "get_doe_status",
            "predict_pressure",
            "predict_pressure_batch",
        }
        assert expected.issubset(tool_names), (
            f"Missing tools: {expected - tool_names}"
        )

    def test_server_name(self) -> None:
        from server import mcp  # noqa: PLC0415

        assert mcp.name == "digital-twin-simulation"

    def test_run_simulation_has_docstring(self) -> None:
        from server import mcp  # noqa: PLC0415

        tools = {t.name: t for t in mcp._tool_manager.list_tools()}
        desc = tools["run_simulation"].description
        assert desc is not None and len(desc) > 20

    def test_run_doe_campaign_has_docstring(self) -> None:
        from server import mcp  # noqa: PLC0415

        tools = {t.name: t for t in mcp._tool_manager.list_tools()}
        desc = tools["run_doe_campaign"].description
        assert desc is not None and "Design of Experiments" in desc
