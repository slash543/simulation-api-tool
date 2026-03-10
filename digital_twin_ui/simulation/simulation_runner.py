"""
Simulation Runner
=================
Executes the simulation solver in an isolated run directory, streams
output to a log file, and returns a structured result.

Run lifecycle
-------------
Each call creates a self-contained directory under ``runs/``::

    runs/run_20260310_143000_a1b2/
    ├── input.feb        ← speed-modified simulation file (from configurator)
    ├── log.txt          ← solver stdout + stderr (streamed in real time)
    ├── results.xplt     ← binary results written by the solver
    └── metadata.json    ← run info and status, updated at each transition

Status transitions::

    QUEUED → RUNNING → COMPLETED
                     ↘ FAILED
                     ↘ TIMEOUT

Usage
-----
    from digital_twin_ui.simulation.simulation_runner import SimulationRunner

    runner = SimulationRunner()

    # Async (preferred in FastAPI / Celery async tasks)
    result = await runner.run_async(speed_mm_s=4.5)

    # Synchronous (scripts, tests)
    result = runner.run(speed_mm_s=4.5)

    print(result.status)       # SimulationStatus.COMPLETED
    print(result.duration_s)   # wall-clock seconds
    print(result.xplt_file)    # Path to .xplt
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from digital_twin_ui.app.core.config import Settings, get_settings
from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.simulation.simulation_configurator import (
    ConfigurationResult,
    SimulationConfigurator,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class SimulationStatus(str, Enum):
    """Lifecycle states of a single simulation run."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Immutable record of a completed (or failed) simulation run.

    All path fields are absolute.
    """

    run_id: str
    status: SimulationStatus
    speed_mm_s: float

    run_dir: Path
    input_feb: Path
    log_file: Path
    xplt_file: Path
    metadata_file: Path

    command: list[str]
    exit_code: int | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_s: float | None = None

    error_message: str | None = None

    # Configuration details (from SimulationConfigurator)
    lc_end_time: float | None = None
    time_steps_step2: int | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == SimulationStatus.COMPLETED

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert non-serialisable types
        d["status"] = self.status.value
        d["run_dir"] = str(self.run_dir)
        d["input_feb"] = str(self.input_feb)
        d["log_file"] = str(self.log_file)
        d["xplt_file"] = str(self.xplt_file)
        d["metadata_file"] = str(self.metadata_file)
        d["started_at"] = self.started_at.isoformat() if self.started_at else None
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return d


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Orchestrates a single simulation run end-to-end.

    Responsibilities:
      1. Generate a unique run ID and create an isolated run directory.
      2. Call SimulationConfigurator to produce a speed-modified input.feb.
      3. Execute the solver asynchronously, streaming output to log.txt.
      4. Write and update metadata.json at each status transition.
      5. Return a RunResult regardless of success or failure.

    The runner is stateless — one instance may handle many concurrent runs.

    Args:
        settings: Application settings. Defaults to the cached singleton.
    """

    _LOG_FILENAME = "log.txt"
    _INPUT_FILENAME = "input.feb"
    _METADATA_FILENAME = "metadata.json"

    def __init__(self, settings: Settings | None = None) -> None:
        self._cfg = settings or get_settings()
        self._sim = self._cfg.simulation
        self._configurator = SimulationConfigurator(settings=self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_async(
        self,
        speed_mm_s: float,
        run_id: str | None = None,
    ) -> RunResult:
        """
        Configure and execute a simulation for *speed_mm_s*.

        Args:
            speed_mm_s: Target insertion speed in mm/s.
            run_id:     Optional explicit run ID.  Auto-generated if None.

        Returns:
            RunResult — always returned, even on failure or timeout.
        """
        run_id = run_id or self._generate_run_id()
        run_dir = self._make_run_dir(run_id)

        logger.info(
            "Simulation queued",
            run_id=run_id,
            speed_mm_s=speed_mm_s,
            run_dir=str(run_dir),
        )

        # --- Step 1: configure input file ---
        input_feb = run_dir / self._INPUT_FILENAME
        try:
            cfg_result = self._configurator.configure(
                speed_mm_s=speed_mm_s,
                output_path=input_feb,
            )
        except Exception as exc:
            return self._fail(
                run_id=run_id,
                speed_mm_s=speed_mm_s,
                run_dir=run_dir,
                error=f"Configuration failed: {exc}",
                status=SimulationStatus.FAILED,
            )

        # --- Step 2: write initial metadata (QUEUED) ---
        # FEBio names the output file after the input file stem (input.feb → input.xplt)
        xplt_file = run_dir / (input_feb.stem + ".xplt")
        command = self._build_command(input_feb)
        result = RunResult(
            run_id=run_id,
            status=SimulationStatus.QUEUED,
            speed_mm_s=speed_mm_s,
            run_dir=run_dir,
            input_feb=input_feb,
            log_file=run_dir / self._LOG_FILENAME,
            xplt_file=xplt_file,
            metadata_file=run_dir / self._METADATA_FILENAME,
            command=command,
            lc_end_time=cfg_result.lc_end_time,
            time_steps_step2=cfg_result.time_steps_step2,
        )
        self._write_metadata(result)

        # --- Step 3: run the solver ---
        result = await self._execute(result)

        return result

    def run(
        self,
        speed_mm_s: float,
        run_id: str | None = None,
    ) -> RunResult:
        """
        Synchronous wrapper around run_async().

        Safe to call from scripts and unit tests.  Uses asyncio.run() so
        it must NOT be called from inside an already-running event loop.
        """
        return asyncio.run(self.run_async(speed_mm_s=speed_mm_s, run_id=run_id))

    # ------------------------------------------------------------------
    # Internal: execution
    # ------------------------------------------------------------------

    async def _execute(self, result: RunResult) -> RunResult:
        """Launch the solver subprocess and wait for completion."""
        started_at = datetime.now(timezone.utc)
        result = self._update(result, status=SimulationStatus.RUNNING, started_at=started_at)
        self._write_metadata(result)

        logger.info(
            "Simulation started",
            run_id=result.run_id,
            command=" ".join(result.command),
        )

        try:
            result = await asyncio.wait_for(
                self._run_subprocess(result, started_at),
                timeout=self._sim.timeout_seconds,
            )
        except asyncio.TimeoutError:
            completed_at = datetime.now(timezone.utc)
            duration = (completed_at - started_at).total_seconds()
            result = self._update(
                result,
                status=SimulationStatus.TIMEOUT,
                completed_at=completed_at,
                duration_s=duration,
                error_message=f"Timeout after {self._sim.timeout_seconds}s",
            )
            self._write_metadata(result)
            logger.error(
                "Simulation timed out",
                run_id=result.run_id,
                timeout_s=self._sim.timeout_seconds,
            )

        return result

    async def _run_subprocess(
        self,
        result: RunResult,
        started_at: datetime,
    ) -> RunResult:
        """
        Spawn the solver process, stream stdout+stderr to log.txt,
        and return an updated RunResult.
        """
        proc = await asyncio.create_subprocess_exec(
            *result.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(result.run_dir),
        )

        # Stream output line-by-line to log file
        await self._stream_to_log(proc, result.log_file)

        exit_code = await proc.wait()
        completed_at = datetime.now(timezone.utc)
        duration = (completed_at - started_at).total_seconds()

        if exit_code == 0:
            status = SimulationStatus.COMPLETED
            error_message = None
            logger.info(
                "Simulation completed",
                run_id=result.run_id,
                duration_s=round(duration, 2),
                exit_code=exit_code,
            )
        else:
            status = SimulationStatus.FAILED
            error_message = f"Solver exited with code {exit_code}"
            logger.error(
                "Simulation failed",
                run_id=result.run_id,
                exit_code=exit_code,
                duration_s=round(duration, 2),
            )

        final = self._update(
            result,
            status=status,
            exit_code=exit_code,
            completed_at=completed_at,
            duration_s=duration,
            error_message=error_message,
        )
        self._write_metadata(final)
        return final

    @staticmethod
    async def _stream_to_log(
        proc: asyncio.subprocess.Process,
        log_path: Path,
    ) -> None:
        """
        Read lines from proc.stdout and append them to log_path.

        Using synchronous file I/O inside an async function is intentional:
        disk writes are fast and blocking is brief enough to be acceptable
        without adding an aiofiles dependency.
        """
        with open(log_path, "wb") as fh:
            assert proc.stdout is not None
            async for line in proc.stdout:
                fh.write(line)

    # ------------------------------------------------------------------
    # Internal: directory and metadata helpers
    # ------------------------------------------------------------------

    def _make_run_dir(self, run_id: str) -> Path:
        """Create and return the isolated run directory."""
        run_dir = self._cfg.runs_dir_abs / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_command(self, input_feb: Path) -> list[str]:
        """
        Assemble the solver command.

        The solver is always invoked from within the run directory, so
        only the filename (not the full path) is passed.
        """
        return [
            self._sim.simulator_executable,
            *self._sim.simulator_args,
            input_feb.name,   # relative: solver CWD = run_dir
        ]

    def _write_metadata(self, result: RunResult) -> None:
        """Serialise RunResult to metadata.json (atomic overwrite)."""
        tmp = result.metadata_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(result.as_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(result.metadata_file)

    @staticmethod
    def _update(result: RunResult, **kwargs: Any) -> RunResult:
        """Return a new RunResult with selected fields replaced."""
        from dataclasses import replace
        return replace(result, **kwargs)

    def _fail(
        self,
        run_id: str,
        speed_mm_s: float,
        run_dir: Path,
        error: str,
        status: SimulationStatus = SimulationStatus.FAILED,
    ) -> RunResult:
        """Build a terminal RunResult for a pre-execution failure."""
        dummy = run_dir / self._METADATA_FILENAME
        input_feb = run_dir / self._INPUT_FILENAME
        result = RunResult(
            run_id=run_id,
            status=status,
            speed_mm_s=speed_mm_s,
            run_dir=run_dir,
            input_feb=input_feb,
            log_file=run_dir / self._LOG_FILENAME,
            xplt_file=run_dir / (input_feb.stem + ".xplt"),
            metadata_file=dummy,
            command=[],
            error_message=error,
        )
        self._write_metadata(result)
        logger.error("Pre-execution failure", run_id=run_id, error=error)
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_run_id() -> str:
        """
        Generate a human-readable, collision-resistant run ID.

        Format: ``run_YYYYMMDD_HHMMSS_<4-char-hex>``

        Example: ``run_20260310_143000_a1b2``
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:4]
        return f"run_{ts}_{suffix}"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_simulation(
    speed_mm_s: float,
    run_id: str | None = None,
    settings: Settings | None = None,
) -> RunResult:
    """
    Convenience wrapper: configure + run a simulation synchronously.

    Args:
        speed_mm_s: Insertion speed in mm/s.
        run_id:     Optional explicit run ID.
        settings:   Application settings (optional).

    Returns:
        RunResult
    """
    runner = SimulationRunner(settings=settings)
    return runner.run(speed_mm_s=speed_mm_s, run_id=run_id)
