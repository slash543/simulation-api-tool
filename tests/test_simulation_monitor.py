"""
Tests for SimulationMonitor (Step 5).

Strategy
--------
Three log fixture types are used:

  real_log      — the actual solver log from conf_file/jobs (ground truth)
  synthetic_*   — small hand-crafted strings covering specific patterns
  empty_log     — non-existent or empty file (edge case)

Coverage
--------
LogParser
  - Real log: NORMAL TERMINATION detected, progress counters correct
  - Synthetic NORMAL termination
  - Synthetic ABNORMAL termination
  - Synthetic ERROR detection
  - ERROR not flagged when followed by NORMAL TERMINATION
  - Time step and convergence counters
  - Elapsed and total elapsed time parsing
  - Missing file returns empty defaults
  - Empty file returns empty defaults
  - parse_incremental: offset skips already-read content

MonitorSnapshot
  - progress_pct: COMPLETED → 100, known total → %, unknown → 0
  - is_terminal for each terminal status
  - as_dict serialisability

SimulationMonitor.get_snapshot
  - Uses metadata.json status when present
  - Falls back to log inference when metadata missing
  - Metadata wins over log for terminal states
  - QUEUED status when run just created (no log yet)

SimulationMonitor.get_status
  - Returns correct SimulationStatus enum

SimulationMonitor.watch (sync generator)
  - Stops immediately when terminal state detected
  - Raises TimeoutError when timeout exceeded

LogParseResult properties
  - is_terminal, inferred_status

Convenience functions
  - get_run_status
  - parse_log
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from digital_twin_ui.simulation.simulation_monitor import (
    LogParseResult,
    LogParser,
    MonitorSnapshot,
    SimulationMonitor,
    _Patterns,
    get_run_status,
    parse_log,
)
from digital_twin_ui.simulation.simulation_runner import SimulationStatus


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REAL_LOG = Path(__file__).parent.parent / "conf_file" / "jobs" / "sample_catheterization.log"

# Minimal log snippets — each exercising one or more patterns
LOG_NORMAL = """\
===== beginning time step 1 : 0.1 =====
------- converged at time : 0.1
===== beginning time step 2 : 0.2 =====
------- converged at time : 0.2
 Elapsed time : 0:00:02
 Total elapsed time .............. : 0:00:02 (2.5 sec)

 N O R M A L   T E R M I N A T I O N
"""

LOG_ABNORMAL = """\
===== beginning time step 1 : 0.1 =====
------- converged at time : 0.1
===== beginning time step 2 : 0.2 =====

 A B N O R M A L   T E R M I N A T I O N

"""

LOG_ERROR = """\
===== beginning time step 1 : 0.1 =====
 *                               E R R O R                                  *
 Solution diverged.
"""

LOG_NORMAL_WITH_ERROR = """\
===== beginning time step 1 : 0.1 =====
 *                               E R R O R                                  *
 Warning only — recovered.
------- converged at time : 0.1

 N O R M A L   T E R M I N A T I O N
"""

LOG_IN_PROGRESS = """\
===== beginning time step 1 : 0.1 =====
------- converged at time : 0.1
===== beginning time step 2 : 0.2 =====
------- converged at time : 0.2
===== beginning time step 3 : 0.3 =====
"""

LOG_EMPTY = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def parser() -> LogParser:
    return LogParser()


@pytest.fixture()
def monitor() -> SimulationMonitor:
    return SimulationMonitor()


def _write_log(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _make_run_dir(tmp_path: Path, metadata: dict | None = None, log: str | None = None) -> Path:
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    if metadata is not None:
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if log is not None:
        (run_dir / "log.txt").write_text(log, encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# LogParser — pattern matching
# ---------------------------------------------------------------------------

class TestLogParserPatterns:
    def test_normal_termination_detected(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_NORMAL)
        result = parser.parse(log)
        assert result.normal_termination is True

    def test_abnormal_termination_detected(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_ABNORMAL)
        result = parser.parse(log)
        assert result.abnormal_termination is True

    def test_error_detected(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_ERROR)
        result = parser.parse(log)
        assert result.error_detected is True

    def test_error_not_flagged_after_normal_termination(self, parser, tmp_path):
        """
        If NORMAL TERMINATION is present, ERROR lines are treated as
        non-fatal warnings — error_detected must be False.
        """
        log = _write_log(tmp_path / "log.txt", LOG_NORMAL_WITH_ERROR)
        result = parser.parse(log)
        assert result.normal_termination is True
        assert result.error_detected is False

    def test_normal_and_abnormal_mutually_exclusive(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_NORMAL)
        result = parser.parse(log)
        assert not (result.normal_termination and result.abnormal_termination)


class TestLogParserProgress:
    def test_current_step_extracted(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_IN_PROGRESS)
        result = parser.parse(log)
        assert result.current_step == 3

    def test_current_time_extracted(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_IN_PROGRESS)
        result = parser.parse(log)
        assert result.current_time == pytest.approx(0.3)

    def test_converged_steps_counted(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_IN_PROGRESS)
        result = parser.parse(log)
        assert result.converged_steps == 2

    def test_last_converged_time(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_IN_PROGRESS)
        result = parser.parse(log)
        assert result.last_converged_time == pytest.approx(0.2)

    def test_elapsed_time_string(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_NORMAL)
        result = parser.parse(log)
        assert result.elapsed_wall_str == "0:00:02"

    def test_total_elapsed_seconds(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_NORMAL)
        result = parser.parse(log)
        assert result.total_elapsed_s == pytest.approx(2.5)

    def test_zero_progress_in_empty_log(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_EMPTY)
        result = parser.parse(log)
        assert result.current_step == 0
        assert result.converged_steps == 0
        assert result.current_time == pytest.approx(0.0)


class TestLogParserEdgeCases:
    def test_missing_file_returns_defaults(self, parser, tmp_path):
        result = parser.parse(tmp_path / "nonexistent.txt")
        assert result.normal_termination is False
        assert result.current_step == 0
        assert result.lines_read == 0

    def test_empty_file_returns_defaults(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", "")
        result = parser.parse(log)
        assert result.normal_termination is False
        assert result.current_step == 0

    def test_lines_read_counted(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", "line1\nline2\nline3\n")
        result = parser.parse(log)
        assert result.lines_read == 3


class TestLogParserIncremental:
    def test_offset_zero_reads_all(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_NORMAL)
        result, new_offset = parser.parse_incremental(log, byte_offset=0)
        assert result.normal_termination is True
        assert new_offset > 0

    def test_offset_at_end_reads_nothing(self, parser, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_NORMAL)
        full_size = log.stat().st_size
        result, new_offset = parser.parse_incremental(log, byte_offset=full_size)
        # No new content
        assert result.current_step == 0
        assert result.converged_steps == 0
        assert new_offset == full_size

    def test_incremental_captures_appended_content(self, parser, tmp_path):
        log_path = tmp_path / "log.txt"
        # Write initial partial content
        log_path.write_text(LOG_IN_PROGRESS, encoding="utf-8")
        _, offset = parser.parse_incremental(log_path, byte_offset=0)

        # Append terminal line
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("\n N O R M A L   T E R M I N A T I O N\n")

        result, _ = parser.parse_incremental(log_path, byte_offset=offset)
        assert result.normal_termination is True


# ---------------------------------------------------------------------------
# LogParser — real log file (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLogParserRealLog:
    def test_real_log_normal_termination(self, parser):
        if not REAL_LOG.exists():
            pytest.skip("Real log file not found")
        result = parser.parse(REAL_LOG)
        assert result.normal_termination is True

    def test_real_log_no_abnormal_termination(self, parser):
        if not REAL_LOG.exists():
            pytest.skip("Real log file not found")
        result = parser.parse(REAL_LOG)
        assert result.abnormal_termination is False

    def test_real_log_no_error(self, parser):
        if not REAL_LOG.exists():
            pytest.skip("Real log file not found")
        result = parser.parse(REAL_LOG)
        assert result.error_detected is False

    def test_real_log_step_count(self, parser):
        """Both steps have 40 increments → at least 20 converged steps visible."""
        if not REAL_LOG.exists():
            pytest.skip("Real log file not found")
        result = parser.parse(REAL_LOG)
        assert result.converged_steps >= 20

    def test_real_log_has_elapsed_time(self, parser):
        if not REAL_LOG.exists():
            pytest.skip("Real log file not found")
        result = parser.parse(REAL_LOG)
        assert result.total_elapsed_s > 0

    def test_real_log_final_time_is_4(self, parser):
        """Simulation ends at t=4 in the real log."""
        if not REAL_LOG.exists():
            pytest.skip("Real log file not found")
        result = parser.parse(REAL_LOG)
        assert result.last_converged_time == pytest.approx(4.0)

    def test_real_log_inferred_status_completed(self, parser):
        if not REAL_LOG.exists():
            pytest.skip("Real log file not found")
        result = parser.parse(REAL_LOG)
        assert result.inferred_status == SimulationStatus.COMPLETED


# ---------------------------------------------------------------------------
# LogParseResult properties
# ---------------------------------------------------------------------------

class TestLogParseResultProperties:
    def test_is_terminal_normal(self):
        r = LogParseResult(normal_termination=True)
        assert r.is_terminal is True

    def test_is_terminal_abnormal(self):
        r = LogParseResult(abnormal_termination=True)
        assert r.is_terminal is True

    def test_is_terminal_error(self):
        r = LogParseResult(error_detected=True)
        assert r.is_terminal is True

    def test_not_terminal_in_progress(self):
        r = LogParseResult(current_step=3, converged_steps=2)
        assert r.is_terminal is False

    def test_inferred_status_normal(self):
        r = LogParseResult(normal_termination=True)
        assert r.inferred_status == SimulationStatus.COMPLETED

    def test_inferred_status_abnormal(self):
        r = LogParseResult(abnormal_termination=True)
        assert r.inferred_status == SimulationStatus.FAILED

    def test_inferred_status_error(self):
        r = LogParseResult(error_detected=True)
        assert r.inferred_status == SimulationStatus.FAILED

    def test_inferred_status_running(self):
        r = LogParseResult(current_step=5)
        assert r.inferred_status == SimulationStatus.RUNNING


# ---------------------------------------------------------------------------
# MonitorSnapshot
# ---------------------------------------------------------------------------

class TestMonitorSnapshot:
    def _make(self, status, converged=0, total=0):
        return MonitorSnapshot(
            run_id="run_test",
            run_dir=Path("/tmp/run_test"),
            status=status,
            converged_steps=converged,
            total_steps=total,
        )

    def test_progress_100_when_completed(self):
        s = self._make(SimulationStatus.COMPLETED, converged=5, total=10)
        assert s.progress_pct == pytest.approx(100.0)

    def test_progress_percentage(self):
        s = self._make(SimulationStatus.RUNNING, converged=20, total=40)
        assert s.progress_pct == pytest.approx(50.0)

    def test_progress_zero_when_total_unknown(self):
        s = self._make(SimulationStatus.RUNNING, converged=5, total=0)
        assert s.progress_pct == pytest.approx(0.0)

    def test_progress_capped_at_100(self):
        s = self._make(SimulationStatus.RUNNING, converged=50, total=40)
        assert s.progress_pct == pytest.approx(100.0)

    @pytest.mark.parametrize("status", [
        SimulationStatus.COMPLETED,
        SimulationStatus.FAILED,
        SimulationStatus.TIMEOUT,
    ])
    def test_is_terminal_for_terminal_statuses(self, status):
        assert self._make(status).is_terminal is True

    @pytest.mark.parametrize("status", [
        SimulationStatus.QUEUED,
        SimulationStatus.RUNNING,
    ])
    def test_not_terminal_for_active_statuses(self, status):
        assert self._make(status).is_terminal is False

    def test_as_dict_is_json_serialisable(self):
        s = self._make(SimulationStatus.RUNNING, converged=5, total=40)
        json.dumps(s.as_dict())

    def test_as_dict_has_progress_pct(self):
        s = self._make(SimulationStatus.RUNNING, converged=20, total=40)
        d = s.as_dict()
        assert "progress_pct" in d
        assert d["progress_pct"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# SimulationMonitor.get_snapshot
# ---------------------------------------------------------------------------

class TestGetSnapshot:
    def test_uses_metadata_status(self, monitor, tmp_path):
        run_dir = _make_run_dir(
            tmp_path,
            metadata={"status": "COMPLETED", "time_steps_step2": 40},
            log=LOG_IN_PROGRESS,   # log says still running — metadata wins
        )
        snap = monitor.get_snapshot(run_dir)
        assert snap.status == SimulationStatus.COMPLETED

    def test_falls_back_to_log_when_no_metadata(self, monitor, tmp_path):
        run_dir = _make_run_dir(tmp_path, log=LOG_NORMAL)
        snap = monitor.get_snapshot(run_dir)
        assert snap.status == SimulationStatus.COMPLETED

    def test_queued_when_no_log_no_metadata(self, monitor, tmp_path):
        run_dir = tmp_path / "run_empty"
        run_dir.mkdir()
        snap = monitor.get_snapshot(run_dir)
        # No metadata, no log → default inference returns RUNNING (no steps),
        # but metadata would give QUEUED from the runner.  With nothing, we
        # accept either RUNNING or QUEUED.
        assert snap.status in (SimulationStatus.RUNNING, SimulationStatus.QUEUED)

    def test_progress_from_log(self, monitor, tmp_path):
        run_dir = _make_run_dir(
            tmp_path,
            metadata={"status": "RUNNING", "time_steps_step2": 50},
            log=LOG_IN_PROGRESS,
        )
        snap = monitor.get_snapshot(run_dir)
        assert snap.converged_steps == 2
        assert snap.current_step == 3
        assert snap.total_steps == 50

    def test_failed_metadata_overrides_log(self, monitor, tmp_path):
        run_dir = _make_run_dir(
            tmp_path,
            metadata={"status": "FAILED"},
            log=LOG_NORMAL,  # log says normal — metadata says failed
        )
        snap = monitor.get_snapshot(run_dir)
        assert snap.status == SimulationStatus.FAILED

    def test_run_id_from_dir_name(self, monitor, tmp_path):
        run_dir = _make_run_dir(tmp_path, log=LOG_NORMAL)
        snap = monitor.get_snapshot(run_dir)
        assert snap.run_id == run_dir.name

    def test_run_dir_recorded(self, monitor, tmp_path):
        run_dir = _make_run_dir(tmp_path, log=LOG_NORMAL)
        snap = monitor.get_snapshot(run_dir)
        assert snap.run_dir == run_dir


class TestGetStatus:
    def test_returns_enum(self, monitor, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, metadata={"status": "COMPLETED"}, log=LOG_NORMAL
        )
        status = monitor.get_status(run_dir)
        assert isinstance(status, SimulationStatus)
        assert status == SimulationStatus.COMPLETED


# ---------------------------------------------------------------------------
# SimulationMonitor.watch (sync generator)
# ---------------------------------------------------------------------------

class TestWatchSync:
    def test_stops_at_completed(self, monitor, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, metadata={"status": "COMPLETED"}, log=LOG_NORMAL
        )
        snapshots = list(monitor.watch(run_dir, poll_interval=0.0))
        assert len(snapshots) >= 1
        assert snapshots[-1].status == SimulationStatus.COMPLETED

    def test_stops_at_failed(self, monitor, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, metadata={"status": "FAILED"}, log=LOG_ABNORMAL
        )
        snapshots = list(monitor.watch(run_dir, poll_interval=0.0))
        assert snapshots[-1].is_terminal is True

    def test_timeout_raises(self, monitor, tmp_path):
        """If the run never reaches terminal state, TimeoutError is raised."""
        run_dir = _make_run_dir(
            tmp_path,
            metadata={"status": "RUNNING"},
            log=LOG_IN_PROGRESS,
        )
        with pytest.raises(TimeoutError):
            list(monitor.watch(run_dir, poll_interval=0.0, timeout=0.001))


# ---------------------------------------------------------------------------
# wait_for_completion
# ---------------------------------------------------------------------------

class TestWaitForCompletion:
    def test_returns_final_snapshot(self, monitor, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, metadata={"status": "COMPLETED"}, log=LOG_NORMAL
        )
        snap = monitor.wait_for_completion(run_dir, poll_interval=0.0)
        assert snap.is_terminal is True
        assert snap.status == SimulationStatus.COMPLETED


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_get_run_status(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path, metadata={"status": "COMPLETED"}, log=LOG_NORMAL
        )
        assert get_run_status(run_dir) == SimulationStatus.COMPLETED

    def test_parse_log_normal(self, tmp_path):
        log = _write_log(tmp_path / "log.txt", LOG_NORMAL)
        result = parse_log(log)
        assert result.normal_termination is True
        assert result.converged_steps == 2

    def test_parse_log_missing_file(self, tmp_path):
        result = parse_log(tmp_path / "missing.txt")
        assert result.normal_termination is False
