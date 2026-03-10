"""
Tests for the logging subsystem.

Coverage:
- configure_logging() — console-only, with file sinks, log dir creation
- _configured flag lifecycle
- configure_from_settings() — idempotency, delegates to settings
- get_logger() — interface and context binding
- _json_sink_serialiser() — structure, required keys, exception serialisation,
                             no-extra context, private-key filtering
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — fake loguru record
# ---------------------------------------------------------------------------

class _Level:
    def __init__(self, name: str):
        self.name = name


class _Exc:
    def __init__(self, exc: Exception):
        self.type = type(exc)
        self.value = exc


def _make_record(
    level: str = "INFO",
    message: str = "test message",
    name: str = "test.module",
    extra: dict | None = None,
    exception=None,
) -> dict:
    return {
        "time": datetime.now(timezone.utc),
        "level": _Level(level),
        "name": name,
        "function": "test_fn",
        "line": 42,
        "message": message,
        "extra": extra or {},
        "exception": exception,
    }


# ---------------------------------------------------------------------------
# _json_sink_serialiser
# ---------------------------------------------------------------------------

class TestJsonSinkSerialiser:
    def test_returns_valid_json(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        result = _json_sink_serialiser(_make_record())
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_required_keys_present(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(_make_record()))
        for key in ("timestamp", "level", "logger", "function", "line", "message"):
            assert key in parsed, f"Missing key: {key}"

    def test_level_value(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(_make_record(level="WARNING")))
        assert parsed["level"] == "WARNING"

    def test_message_value(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(_make_record(message="hello")))
        assert parsed["message"] == "hello"

    def test_logger_name(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(_make_record(name="my.pkg")))
        assert parsed["logger"] == "my.pkg"

    def test_context_included_when_extra_present(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(
            _make_record(extra={"run_id": "run_0001", "speed": 4.5})
        ))
        assert "context" in parsed
        assert parsed["context"]["run_id"] == "run_0001"
        assert parsed["context"]["speed"] == pytest.approx(4.5)

    def test_no_context_key_when_extra_empty(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(_make_record(extra={})))
        assert "context" not in parsed

    def test_private_keys_filtered_from_context(self):
        """Keys starting with _ must not appear in context."""
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(
            _make_record(extra={"_internal": "secret", "visible": "yes"})
        ))
        ctx = parsed.get("context", {})
        assert "_internal" not in ctx
        assert ctx.get("visible") == "yes"

    def test_exception_included_when_present(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        exc = _Exc(ValueError("bad input"))
        parsed = json.loads(_json_sink_serialiser(_make_record(exception=exc)))
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert "bad input" in parsed["exception"]["value"]

    def test_no_exception_key_when_none(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(_make_record(exception=None)))
        assert "exception" not in parsed

    def test_timestamp_is_iso_format(self):
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        parsed = json.loads(_json_sink_serialiser(_make_record()))
        # Should parse without raising
        datetime.fromisoformat(parsed["timestamp"])

    def test_single_line_output(self):
        """Each record must be exactly one line (for log aggregators)."""
        from digital_twin_ui.app.core.logging import _json_sink_serialiser
        result = _json_sink_serialiser(_make_record())
        assert "\n" not in result


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_no_exception_without_log_dir(self, reset_logging_state):
        from digital_twin_ui.app.core.logging import configure_logging
        configure_logging(level="DEBUG", log_dir=None)  # must not raise

    def test_configured_flag_set_after_call(self, reset_logging_state):
        import digital_twin_ui.app.core.logging as log_module
        from digital_twin_ui.app.core.logging import configure_logging
        assert not log_module._configured
        configure_logging(level="INFO", log_dir=None)
        assert log_module._configured

    def test_creates_log_directory(self, tmp_path, reset_logging_state):
        from digital_twin_ui.app.core.logging import configure_logging
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()
        configure_logging(level="DEBUG", log_dir=log_dir)
        assert log_dir.exists()

    def test_log_files_created_on_write(self, tmp_path, reset_logging_state):
        from digital_twin_ui.app.core.logging import configure_logging, get_logger
        log_dir = tmp_path / "logs"
        configure_logging(level="DEBUG", log_dir=log_dir)
        get_logger("test").info("probe message")

        # Give loguru time to flush (it's synchronous by default)
        plain_log = log_dir / "digital_twin.log"
        jsonl_log = log_dir / "digital_twin.jsonl"
        assert plain_log.exists(), "Plain log file not created"
        assert jsonl_log.exists(), "JSONL log file not created"

    def test_jsonl_file_contains_valid_json(self, tmp_path, reset_logging_state):
        from digital_twin_ui.app.core.logging import configure_logging, get_logger
        log_dir = tmp_path / "logs"
        configure_logging(level="DEBUG", log_dir=log_dir)
        get_logger("test").info("json probe")

        jsonl_log = log_dir / "digital_twin.jsonl"
        lines = [l for l in jsonl_log.read_text().splitlines() if l.strip()]
        assert lines, "No log entries written to JSONL file"
        for line in lines:
            parsed = json.loads(line)
            assert "message" in parsed

    def test_plain_log_contains_message(self, tmp_path, reset_logging_state):
        from digital_twin_ui.app.core.logging import configure_logging, get_logger
        log_dir = tmp_path / "logs"
        configure_logging(level="DEBUG", log_dir=log_dir)
        get_logger("test").warning("unique_marker_xyz")

        plain_log = log_dir / "digital_twin.log"
        assert "unique_marker_xyz" in plain_log.read_text()


# ---------------------------------------------------------------------------
# configure_from_settings
# ---------------------------------------------------------------------------

class TestConfigureFromSettings:
    def test_is_noop_when_already_configured(self, reset_logging_state):
        import digital_twin_ui.app.core.logging as log_module
        from digital_twin_ui.app.core.logging import configure_from_settings
        log_module._configured = True
        configure_from_settings()   # must not raise or re-configure
        configure_from_settings()   # second call also fine

    def test_sets_configured_flag(self, tmp_path, reset_logging_state):
        import digital_twin_ui.app.core.logging as log_module
        from digital_twin_ui.app.core.logging import configure_from_settings
        assert not log_module._configured
        configure_from_settings()
        assert log_module._configured


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_object_with_log_methods(self):
        from digital_twin_ui.app.core.logging import get_logger
        log = get_logger("test.module")
        for method in ("debug", "info", "warning", "error", "critical", "exception"):
            assert callable(getattr(log, method)), f"Missing method: {method}"

    def test_supports_contextualize(self):
        from digital_twin_ui.app.core.logging import get_logger
        log = get_logger("test")
        assert hasattr(log, "contextualize")

    def test_supports_bind(self):
        from digital_twin_ui.app.core.logging import get_logger
        log = get_logger("test")
        bound = log.bind(run_id="run_0001")
        assert bound is not None

    def test_different_names_return_separate_loggers(self):
        from digital_twin_ui.app.core.logging import get_logger
        log_a = get_logger("module.a")
        log_b = get_logger("module.b")
        # They should be independent objects
        assert log_a is not log_b
