"""
Simulation Monitor
==================
Reads the solver log file to determine the current state of a simulation
run and track its progress.

The monitor is **purely log-based** — it reads what the solver has written
to disk without communicating with the subprocess.  This design means:

  - It works for currently-running AND already-completed runs.
  - It can recover state after a process restart (restartable simulations).
  - It is fully decoupled from SimulationRunner.

Log patterns recognised (from real solver output)
--------------------------------------------------
  NORMAL TERMINATION  : ``N O R M A L   T E R M I N A T I O N``
  Abnormal            : ``A B N O R M A L   T E R M I N A T I O N``
  Error               : ``*   E R R O R``  or standalone ``E R R O R``
  Time step start     : ``===== beginning time step N : T =====``
  Convergence         : ``------- converged at time : T``
  Elapsed             : ``Elapsed time : HH:MM:SS``

Usage
-----
    from digital_twin_ui.simulation.simulation_monitor import SimulationMonitor

    monitor = SimulationMonitor()

    # One-shot snapshot
    snapshot = monitor.get_snapshot(run_dir=Path("runs/run_001"))
    print(snapshot.status)          # SimulationStatus.RUNNING
    print(snapshot.current_time)    # 2.4  (simulation time, seconds)
    print(snapshot.converged_steps) # 8

    # Async polling until completion
    async for snapshot in monitor.watch_async(run_dir, poll_interval=2.0):
        print(snapshot.progress_pct, snapshot.status)

    # Sync blocking wait
    final = monitor.wait_for_completion(run_dir, timeout=3600)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterator

from digital_twin_ui.app.core.config import Settings, get_settings
from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.simulation.simulation_runner import SimulationStatus

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Log patterns
# ---------------------------------------------------------------------------

class _Patterns:
    """Compiled regular expressions for solver log parsing."""

    NORMAL_TERMINATION = re.compile(
        r"N\s+O\s+R\s+M\s+A\s+L\s+T\s+E\s+R\s+M\s+I\s+N\s+A\s+T\s+I\s+O\s+N"
    )
    ABNORMAL_TERMINATION = re.compile(
        r"A\s+B\s+N\s+O\s+R\s+M\s+A\s+L\s+T\s+E\s+R\s+M\s+I\s+N\s+A\s+T\s+I\s+O\s+N"
    )
    # FEBio formats errors as "* E R R O R *" or standalone "E R R O R"
    ERROR_LINE = re.compile(r"E\s+R\s+R\s+O\s+R")

    # "===== beginning time step 12 : 3.2 ====="
    TIME_STEP_BEGIN = re.compile(
        r"={5}\s+beginning\s+time\s+step\s+(\d+)\s*:\s*([\d.]+)\s*={5}"
    )
    # "------- converged at time : 3.2"
    CONVERGED_AT = re.compile(
        r"-{3,}\s+converged\s+at\s+time\s*:\s*([\d.]+)"
    )
    # "Elapsed time : 0:00:55"
    ELAPSED_TIME = re.compile(
        r"Elapsed\s+time\s*:\s*(\d+:\d+:\d+)"
    )
    # "Total elapsed time ......... : 0:00:55 (54.874 sec)"
    TOTAL_ELAPSED = re.compile(
        r"Total\s+elapsed\s+time\s*[.]*\s*:\s*(\d+:\d+:\d+)\s*\(([\d.]+)\s*sec\)"
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogParseResult:
    """
    Structured information extracted from a single pass over a log file.

    All fields are optional because the log may be empty or partially written
    (simulation still in progress).
    """

    # Terminal state detected
    normal_termination: bool = False
    abnormal_termination: bool = False
    error_detected: bool = False

    # Progress counters
    current_step: int = 0          # last "beginning time step N" seen
    current_time: float = 0.0      # simulation time of last step start (s)
    converged_steps: int = 0       # number of "converged at time" lines
    last_converged_time: float = 0.0

    # Timing
    elapsed_wall_str: str = ""     # HH:MM:SS string from log
    total_elapsed_s: float = 0.0   # seconds from "Total elapsed time" line

    # Raw line count (useful for detecting if file grew)
    lines_read: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.normal_termination or self.abnormal_termination or self.error_detected

    @property
    def inferred_status(self) -> SimulationStatus:
        """
        Derive SimulationStatus purely from log content.

        Priority: NORMAL > ABNORMAL = ERROR > RUNNING (in-progress)
        """
        if self.normal_termination:
            return SimulationStatus.COMPLETED
        if self.abnormal_termination or self.error_detected:
            return SimulationStatus.FAILED
        return SimulationStatus.RUNNING


@dataclass
class MonitorSnapshot:
    """
    Combined status snapshot for a run at a point in time.

    Merges the authoritative metadata.json status with log-derived progress
    data so callers get one unified view.
    """

    run_id: str
    run_dir: Path

    # Status — from metadata.json when available, log otherwise
    status: SimulationStatus

    # Progress (from log)
    current_step: int = 0
    current_time: float = 0.0
    converged_steps: int = 0
    last_converged_time: float = 0.0
    elapsed_wall_str: str = ""
    total_elapsed_s: float = 0.0

    # Total expected steps (from metadata, used for progress %)
    total_steps: int = 0

    # Raw log parse (for callers that need full detail)
    log_parse: LogParseResult | None = None

    @property
    def progress_pct(self) -> float:
        """
        Estimated completion percentage based on converged steps vs total.

        Returns 100.0 if completed, 0.0 if total_steps unknown.
        """
        if self.status == SimulationStatus.COMPLETED:
            return 100.0
        if self.total_steps > 0:
            return min(100.0, self.converged_steps / self.total_steps * 100)
        return 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            SimulationStatus.COMPLETED,
            SimulationStatus.FAILED,
            SimulationStatus.TIMEOUT,
        )

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "status": self.status.value,
            "current_step": self.current_step,
            "current_time": self.current_time,
            "converged_steps": self.converged_steps,
            "last_converged_time": self.last_converged_time,
            "elapsed_wall_str": self.elapsed_wall_str,
            "total_elapsed_s": self.total_elapsed_s,
            "total_steps": self.total_steps,
            "progress_pct": round(self.progress_pct, 1),
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class LogParser:
    """
    Parses a solver log file and extracts structured progress information.

    Stateless — each call to ``parse()`` reads the file from scratch.
    For incremental polling, use ``parse_incremental()`` which accepts a
    byte offset to avoid re-reading the whole file on each poll.
    """

    def parse(self, log_path: Path) -> LogParseResult:
        """
        Read the entire log file and return a LogParseResult.

        Args:
            log_path: Path to the solver log file.

        Returns:
            LogParseResult — empty defaults if file does not exist yet.
        """
        if not log_path.exists():
            return LogParseResult()

        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return LogParseResult()

        return self._parse_text(text)

    def parse_incremental(
        self,
        log_path: Path,
        byte_offset: int = 0,
    ) -> tuple[LogParseResult, int]:
        """
        Parse from *byte_offset* to end of file.

        Useful for polling: caller stores the returned offset and passes it
        back on the next call.  The returned LogParseResult reflects only
        the **new** content since the offset, so callers should merge with
        previous results if they need cumulative state.

        Returns:
            (LogParseResult for new content, new byte offset)
        """
        if not log_path.exists():
            return LogParseResult(), byte_offset

        try:
            with open(log_path, "rb") as fh:
                fh.seek(byte_offset)
                new_bytes = fh.read()
                new_offset = fh.tell()
        except OSError:
            return LogParseResult(), byte_offset

        text = new_bytes.decode("utf-8", errors="replace")
        return self._parse_text(text), new_offset

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_text(self, text: str) -> LogParseResult:
        """Apply all patterns to *text* and build a LogParseResult."""
        lines = text.splitlines()

        normal_termination = bool(_Patterns.NORMAL_TERMINATION.search(text))
        abnormal_termination = bool(_Patterns.ABNORMAL_TERMINATION.search(text))

        # Only flag error if not already a normal termination (FEBio can
        # log "E R R O R" in non-fatal warning blocks)
        error_detected = (
            not normal_termination
            and bool(_Patterns.ERROR_LINE.search(text))
            and not abnormal_termination
        )

        # Time step progress
        step_matches = _Patterns.TIME_STEP_BEGIN.findall(text)
        current_step = int(step_matches[-1][0]) if step_matches else 0
        current_time = float(step_matches[-1][1]) if step_matches else 0.0

        # Convergence
        converge_matches = _Patterns.CONVERGED_AT.findall(text)
        converged_steps = len(converge_matches)
        last_converged_time = float(converge_matches[-1]) if converge_matches else 0.0

        # Elapsed time
        elapsed_match = _Patterns.ELAPSED_TIME.search(text)
        elapsed_wall_str = elapsed_match.group(1) if elapsed_match else ""

        total_match = _Patterns.TOTAL_ELAPSED.search(text)
        total_elapsed_s = float(total_match.group(2)) if total_match else 0.0

        return LogParseResult(
            normal_termination=normal_termination,
            abnormal_termination=abnormal_termination,
            error_detected=error_detected,
            current_step=current_step,
            current_time=current_time,
            converged_steps=converged_steps,
            last_converged_time=last_converged_time,
            elapsed_wall_str=elapsed_wall_str,
            total_elapsed_s=total_elapsed_s,
            lines_read=len(lines),
        )


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class SimulationMonitor:
    """
    Observes a simulation run directory and reports its current state.

    Args:
        settings: Application settings (optional).
    """

    _METADATA_FILENAME = "metadata.json"
    _LOG_FILENAME = "log.txt"

    def __init__(self, settings: Settings | None = None) -> None:
        self._cfg = settings or get_settings()
        self._parser = LogParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_snapshot(self, run_dir: Path) -> MonitorSnapshot:
        """
        Return a one-shot MonitorSnapshot for the given run directory.

        Reads metadata.json for the authoritative status, then enriches
        with log-derived progress information.

        Args:
            run_dir: Path to the run directory (e.g. ``runs/run_001``).

        Returns:
            MonitorSnapshot
        """
        run_id = run_dir.name
        metadata = self._read_metadata(run_dir)
        log_parse = self._parser.parse(run_dir / self._LOG_FILENAME)

        # Authoritative status: metadata wins; fall back to log inference
        if metadata:
            status_str = metadata.get("status", "QUEUED")
            try:
                status = SimulationStatus(status_str)
            except ValueError:
                status = log_parse.inferred_status
        else:
            status = log_parse.inferred_status

        # If metadata says COMPLETED/FAILED but log says otherwise, trust metadata
        # (runner already validated exit code)

        total_steps = metadata.get("time_steps_step2", 0) if metadata else 0

        snapshot = MonitorSnapshot(
            run_id=run_id,
            run_dir=run_dir,
            status=status,
            current_step=log_parse.current_step,
            current_time=log_parse.current_time,
            converged_steps=log_parse.converged_steps,
            last_converged_time=log_parse.last_converged_time,
            elapsed_wall_str=log_parse.elapsed_wall_str,
            total_elapsed_s=log_parse.total_elapsed_s,
            total_steps=total_steps,
            log_parse=log_parse,
        )

        logger.debug(
            "Monitor snapshot",
            run_id=run_id,
            status=status.value,
            converged={log_parse.converged_steps},
            total=total_steps,
            pct=round(snapshot.progress_pct, 1),
        )
        return snapshot

    def get_status(self, run_dir: Path) -> SimulationStatus:
        """Convenience: return only the SimulationStatus for *run_dir*."""
        return self.get_snapshot(run_dir).status

    async def watch_async(
        self,
        run_dir: Path,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> AsyncIterator[MonitorSnapshot]:
        """
        Async generator that yields a MonitorSnapshot on each poll interval
        until the simulation reaches a terminal state.

        Args:
            run_dir:       Run directory to monitor.
            poll_interval: Seconds between polls (default 2.0).
            timeout:       Optional maximum total wait time in seconds.
                           Raises asyncio.TimeoutError if exceeded.

        Yields:
            MonitorSnapshot at each poll.

        Raises:
            asyncio.TimeoutError: If *timeout* is set and exceeded.
        """
        import time as _time

        deadline = (_time.monotonic() + timeout) if timeout is not None else None
        while True:
            snapshot = self.get_snapshot(run_dir)
            yield snapshot

            if snapshot.is_terminal:
                logger.info(
                    "Monitor: run reached terminal state",
                    run_id=snapshot.run_id,
                    status=snapshot.status.value,
                )
                return

            if deadline is not None and _time.monotonic() >= deadline:
                raise asyncio.TimeoutError(
                    f"Monitor timed out after {timeout}s waiting for {run_dir.name}"
                )

            await asyncio.sleep(poll_interval)

    def watch(
        self,
        run_dir: Path,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> Iterator[MonitorSnapshot]:
        """
        Synchronous generator version of watch_async().

        Uses time.sleep internally.  Safe to call from scripts and tests.
        """
        import time

        deadline = (time.monotonic() + timeout) if timeout is not None else None
        while True:
            snapshot = self.get_snapshot(run_dir)
            yield snapshot

            if snapshot.is_terminal:
                return

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Monitor timed out after {timeout}s waiting for {run_dir.name}"
                )

            time.sleep(poll_interval)

    def wait_for_completion(
        self,
        run_dir: Path,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> MonitorSnapshot:
        """
        Block until the simulation reaches a terminal state and return the
        final MonitorSnapshot.

        Args:
            run_dir:       Run directory.
            poll_interval: Seconds between polls.
            timeout:       Max wait time in seconds.

        Returns:
            Final MonitorSnapshot (status COMPLETED, FAILED, or TIMEOUT).

        Raises:
            TimeoutError: If *timeout* exceeded.
        """
        last: MonitorSnapshot | None = None
        for snapshot in self.watch(run_dir, poll_interval=poll_interval, timeout=timeout):
            last = snapshot
        assert last is not None
        return last

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_metadata(self, run_dir: Path) -> dict | None:
        """Read and parse metadata.json; return None on any error."""
        meta_path = run_dir / self._METADATA_FILENAME
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read metadata.json", path=str(meta_path), error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_run_status(run_dir: Path, settings: Settings | None = None) -> SimulationStatus:
    """Return the current SimulationStatus for a run directory."""
    return SimulationMonitor(settings=settings).get_status(run_dir)


def parse_log(log_path: Path) -> LogParseResult:
    """Parse a solver log file and return structured results."""
    return LogParser().parse(log_path)
