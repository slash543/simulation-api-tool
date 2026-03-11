"""
Tests for the new MCP tool implementations:
  - tool_list_templates()   (tools.py)
  - tool_run_doe_campaign() with template/max_perturbation/dwell_time_s params

These tests mock httpx so no running API is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Make mcp_server importable
# ---------------------------------------------------------------------------
MCP_SERVER_DIR = Path(__file__).parent.parent / "mcp_server"
if str(MCP_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER_DIR))

from tools import (  # noqa: E402
    tool_list_templates,
    tool_run_doe_campaign,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int, body) -> MagicMock:
    """Build a fake httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body) if isinstance(body, (dict, list)) else str(body)
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _parse(result: str) -> dict | list:
    return json.loads(result)


SAMPLE_TEMPLATES_RESPONSE = {
    "templates": [
        {
            "name": "sample_catheterization",
            "label": "Sample Catheterization",
            "n_steps": 1,
            "speed_range_min": 4.0,
            "speed_range_max": 6.0,
            "displacements_mm": [10.0],
        },
        {
            "name": "DT_BT_14Fr_FO_10E_IR12",
            "label": "14Fr Foley — IR12",
            "n_steps": 10,
            "speed_range_min": 10.0,
            "speed_range_max": 25.0,
            "displacements_mm": [64.0, 46.0] + [28.0] * 8,
        },
    ]
}


# ---------------------------------------------------------------------------
# tool_list_templates
# ---------------------------------------------------------------------------

class TestToolListTemplates:
    def test_returns_templates_list(self) -> None:
        mock = _mock_response(200, SAMPLE_TEMPLATES_RESPONSE)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_list_templates())
        assert "templates" in result

    def test_templates_count(self) -> None:
        mock = _mock_response(200, SAMPLE_TEMPLATES_RESPONSE)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_list_templates())
        assert len(result["templates"]) == 2

    def test_template_name_field(self) -> None:
        mock = _mock_response(200, SAMPLE_TEMPLATES_RESPONSE)
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_list_templates())
        names = [t["name"] for t in result["templates"]]
        assert "DT_BT_14Fr_FO_10E_IR12" in names

    def test_calls_correct_endpoint(self) -> None:
        mock = _mock_response(200, SAMPLE_TEMPLATES_RESPONSE)
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get.return_value = mock
            tool_list_templates()
        client_inst.get.assert_called_once_with("/templates")

    def test_api_unreachable_returns_error(self) -> None:
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.get.side_effect = httpx.ConnectError("refused")
            result = _parse(tool_list_templates())
        assert "error" in result

    def test_http_error_returns_error(self) -> None:
        mock = _mock_response(500, {"detail": "internal server error"})
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock
            result = _parse(tool_list_templates())
        assert "error" in result


# ---------------------------------------------------------------------------
# tool_run_doe_campaign — extended parameters
# ---------------------------------------------------------------------------

class TestToolRunDoeCampaignExtended:
    def test_returns_task_id(self) -> None:
        mock = _mock_response(202, {"task_id": "doe-task-abc", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(
                tool_run_doe_campaign(
                    n_samples=5,
                    speed_min=10.0,
                    speed_max=25.0,
                    template="DT_BT_14Fr_FO_10E_IR12",
                )
            )
        assert result["task_id"] == "doe-task-abc"

    def test_template_sent_in_payload(self) -> None:
        mock = _mock_response(202, {"task_id": "t1", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(
                n_samples=5,
                speed_min=10.0,
                speed_max=25.0,
                template="DT_BT_14Fr_FO_10E_IR12",
            )
        call_kwargs = client_inst.post.call_args.kwargs
        payload = call_kwargs.get("json", {})
        assert payload.get("template") == "DT_BT_14Fr_FO_10E_IR12"

    def test_max_perturbation_sent_in_payload(self) -> None:
        mock = _mock_response(202, {"task_id": "t1", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(
                n_samples=5,
                speed_min=10.0,
                speed_max=25.0,
                max_perturbation=0.15,
            )
        payload = client_inst.post.call_args.kwargs.get("json", {})
        assert payload.get("max_perturbation") == pytest.approx(0.15)

    def test_dwell_time_sent_in_payload(self) -> None:
        mock = _mock_response(202, {"task_id": "t1", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(
                n_samples=5,
                speed_min=10.0,
                speed_max=25.0,
                dwell_time_s=2.0,
            )
        payload = client_inst.post.call_args.kwargs.get("json", {})
        assert payload.get("dwell_time_s") == pytest.approx(2.0)

    def test_seed_sent_when_provided(self) -> None:
        mock = _mock_response(202, {"task_id": "t1", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(
                n_samples=5,
                speed_min=10.0,
                speed_max=25.0,
                seed=42,
            )
        payload = client_inst.post.call_args.kwargs.get("json", {})
        assert payload.get("seed") == 42

    def test_seed_not_sent_when_none(self) -> None:
        mock = _mock_response(202, {"task_id": "t1", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(
                n_samples=5,
                speed_min=10.0,
                speed_max=25.0,
                seed=None,
            )
        payload = client_inst.post.call_args.kwargs.get("json", {})
        assert "seed" not in payload

    def test_sampler_lhs_sent_in_payload(self) -> None:
        mock = _mock_response(202, {"task_id": "t1", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(
                n_samples=5,
                speed_min=10.0,
                speed_max=25.0,
                sampler="lhs",
            )
        payload = client_inst.post.call_args.kwargs.get("json", {})
        assert payload.get("sampler") == "lhs"

    def test_http_error_returns_error_json(self) -> None:
        mock = _mock_response(422, {"detail": "speed_min must be less than speed_max"})
        with patch("tools.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = mock
            result = _parse(
                tool_run_doe_campaign(
                    n_samples=5,
                    speed_min=25.0,
                    speed_max=10.0,
                )
            )
        assert "error" in result

    def test_api_unreachable_returns_error(self) -> None:
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.side_effect = httpx.ConnectError("refused")
            result = _parse(
                tool_run_doe_campaign(
                    n_samples=5,
                    speed_min=10.0,
                    speed_max=25.0,
                )
            )
        assert "error" in result

    def test_calls_correct_endpoint(self) -> None:
        mock = _mock_response(202, {"task_id": "t1", "status": "PENDING"})
        with patch("tools.httpx.Client") as MockClient:
            client_inst = MockClient.return_value.__enter__.return_value
            client_inst.post.return_value = mock
            tool_run_doe_campaign(n_samples=5, speed_min=10.0, speed_max=25.0)
        client_inst.post.assert_called_once()
        call_args = client_inst.post.call_args.args
        assert "/doe/run" in call_args[0]
