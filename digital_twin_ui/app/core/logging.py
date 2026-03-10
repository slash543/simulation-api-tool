"""
Structured logging for the Digital Twin UI platform.

Uses loguru as the backend with:
- Console sink (colourised, human-readable)
- Rotating file sink (JSON-structured for machine parsing)
- Context binding for simulation run IDs

Usage
-----
    from digital_twin_ui.app.core.logging import get_logger, configure_logging

    # Once at startup:
    configure_logging()

    # In every module:
    logger = get_logger(__name__)
    logger.info("Simulation started", run_id="run_0001", speed=5.0)

    # Bind context for a whole block:
    with logger.contextualize(run_id="run_0001"):
        logger.info("step 1 complete")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _root_logger

# Re-export the root logger so callers can import from one place.
logger = _root_logger

_configured = False


# ---------------------------------------------------------------------------
# JSON serialiser for the file sink
# ---------------------------------------------------------------------------

def _json_sink_serialiser(record: dict[str, Any]) -> str:  # type: ignore[type-arg]
    """Serialise a loguru record to a single JSON line."""
    payload = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    # Include any extra context fields bound via contextualize() or bind()
    if record["extra"]:
        payload["context"] = {
            k: v for k, v in record["extra"].items() if not k.startswith("_")
        }
    if record["exception"]:
        exc = record["exception"]
        payload["exception"] = {
            "type": exc.type.__name__ if exc.type else None,
            "value": str(exc.value) if exc.value else None,
        }
    return json.dumps(payload)


def _json_sink(message: Any) -> None:  # type: ignore[type-arg]
    """Write a JSON-formatted record to the open file sink."""
    # loguru file sinks handle the file handle; this formatter is used only
    # when we pass it as a `format` callable.
    print(_json_sink_serialiser(message.record), file=message)


# ---------------------------------------------------------------------------
# JSON file sink factory
# ---------------------------------------------------------------------------

def _make_json_file_sink(path: Path):
    """
    Return a callable sink that appends JSON lines to *path*.

    Loguru callable sinks receive a ``Message`` object whose ``.record``
    attribute is the standard record dict.  Writing directly avoids the
    ``format_map`` step that loguru applies to format-string results, which
    would break because JSON naturally contains ``{`` / ``}`` characters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def _sink(message: Any) -> None:  # type: ignore[type-arg]
        fh.write(_json_sink_serialiser(message.record) + "\n")
        fh.flush()

    return _sink


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    level: str = "DEBUG",
    console_format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    ),
    log_dir: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure loguru sinks.

    Should be called exactly once at application startup.

    Args:
        level:          Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console_format: Loguru format string for the colourised console sink.
        log_dir:        Directory for rotating log files. If None, file
                        logging is skipped.
        rotation:       When to rotate the log file (e.g. "10 MB", "1 day").
        retention:      How long to keep old log files (e.g. "1 week").
    """
    global _configured

    # Remove the default loguru handler so we don't get duplicate output.
    _root_logger.remove()

    # --- Console sink (human-readable, colourised) ---
    _root_logger.add(
        sys.stderr,
        level=level,
        format=console_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # --- File sinks ---
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Human-readable rotating file
        _root_logger.add(
            str(log_dir / "digital_twin.log"),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
                   "{name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            backtrace=True,
            diagnose=False,
        )

        # JSON-structured rotating file (for log aggregation pipelines).
        # loguru's serialize=True emits each record as a JSON object; we wrap it
        # with a custom sink factory so _json_sink_serialiser remains the
        # authoritative format and can be unit-tested independently.
        _root_logger.add(
            _make_json_file_sink(log_dir / "digital_twin.jsonl"),
            level=level,
            format="{message}",   # consumed by the sink, not the file path
        )

    _configured = True
    _root_logger.debug("Logging configured (level={}, log_dir={})", level, log_dir)


def configure_from_settings() -> None:
    """
    Convenience wrapper that reads config via get_settings() and configures
    logging accordingly.  Safe to call multiple times (no-op after first call).
    """
    if _configured:
        return

    from digital_twin_ui.app.core.config import get_settings  # local import to avoid circular

    cfg = get_settings()
    configure_logging(
        level=cfg.logging.level,
        log_dir=cfg.log_dir_abs,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
    )


def get_logger(name: str) -> Any:
    """
    Return a loguru logger bound to *name*.

    This mirrors the standard-library ``logging.getLogger(name)`` pattern
    so module-level usage is familiar:

        logger = get_logger(__name__)

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A loguru logger with the name bound as extra context.
    """
    return _root_logger.bind(name=name)
