"""
Tests for SimulationRunner (Step 4).

Strategy
--------
The solver binary (febio4) is not available in the test environment, so all
subprocess calls are replaced with AsyncMock objects that simulate the three
outcomes we care about:

  - Successful run   (exit code 0)
  - Failed run       (exit code 1)
  - Timeout          (asyncio.TimeoutError raised)

The SimulationConfigurator is NOT mocked: it reads the real minimal_feb
fixture and produces a real output file, so we validate the full
configure → run pipeline without needing the solver.

Coverage
--------
- Run directory creation and naming
- All file paths created / reported correctly
- metadata.json written at QUEUED and updated at RUNNING / terminal state
- Status transitions: QUEUED → RUNNING → COMPLETED / FAILED / TIMEOUT
- log.txt written from subprocess stdout
- Command assembly: ["febio4", "-i", "input.feb"]
- exit_code, duration_s, error_message fields
- RunResult.succeeded property
- RunResult.as_dict() serialisation
- SimulationRunner._generate_run_id() uniqueness and format
- SimulationRunner._build_command() uses filename only (not full path)
- run_simulation() convenience function
- Pre-execution failure path (configurator raises)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from digital_twin_ui.simulation.simulation_runner import (
    RunResult,
    SimulationRunner,
    SimulationStatus,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Helpers — fake subprocess
# ---------------------------------------------------------------------------

def _make_mock_proc(
    stdout_lines: list[bytes],
    returncode: int = 0,
) -> MagicMock:
    """
    Build a mock asyncio.subprocess.Process.

    stdout is an async iterator that yields each line then raises StopAsyncIteration.
    wait() is a coroutine that returns returncode.
    """
    proc = MagicMock()
    proc.returncode = returncode

    async def _aiter():
        for line in stdout_lines:
            yield line

    proc.stdout = _aiter()

    async def _wait():
        return returncode

    proc.wait = _wait
    return proc


def _patch_subprocess(proc: MagicMock):
    """Context manager that replaces asyncio.create_subprocess_exec."""
    return patch(
        "digital_twin_ui.simulation.simulation_runner.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def runner() -> SimulationRunner:
    return SimulationRunner()


@pytest.fixture()
def success_proc():
    return _make_mock_proc(
        stdout_lines=[
            b"FEBio version 4.0\n",
            b"Reading file...\n",
            b"NORMAL TERMINATION\n",
        ],
        returncode=0,
    )


@pytest.fixture()
def failure_proc():
    return _make_mock_proc(
        stdout_lines=[
            b"FEBio version 4.0\n",
            b"ERROR: convergence failure\n",
        ],
        returncode=1,
    )


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

class TestRunIdGeneration:
    def test_format(self):
        run_id = SimulationRunner._generate_run_id()
        # e.g. "run_20260310_143000_a1b2"
        parts = run_id.split("_")
        assert parts[0] == "run"
        assert len(parts[1]) == 8   # YYYYMMDD
        assert len(parts[2]) == 6   # HHMMSS
        assert len(parts[3]) == 4   # hex suffix

    def test_uniqueness(self):
        ids = {SimulationRunner._generate_run_id() for _ in range(20)}
        assert len(ids) == 20

    def test_starts_with_run_prefix(self):
        assert SimulationRunner._generate_run_id().startswith("run_")


# ---------------------------------------------------------------------------
# Command assembly
# ---------------------------------------------------------------------------

class TestBuildCommand:
    def test_command_structure(self, runner):
        feb = Path("/some/dir/run_001/input.feb")
        cmd = runner._build_command(feb)
        assert cmd == ["febio4", "-i", "input.feb"]

    def test_uses_filename_not_full_path(self, runner):
        feb = Path("/long/absolute/path/to/run_dir/input.feb")
        cmd = runner._build_command(feb)
        # Only the filename goes into the command — CWD handles the rest
        assert "input.feb" in cmd
        assert str(feb) not in cmd

    def test_executable_is_first(self, runner):
        cmd = runner._build_command(Path("input.feb"))
        assert cmd[0] == "febio4"


# ---------------------------------------------------------------------------
# Successful run
# ---------------------------------------------------------------------------

class TestSuccessfulRun:
    @pytest.mark.asyncio
    async def test_status_completed(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_test_success")
        assert result.status == SimulationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_succeeded_property(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_test_succ2")
        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_exit_code_zero(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_test_ec0")
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_no_error_message(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_test_noerr")
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_duration_recorded(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_test_dur")
        assert result.duration_s is not None
        assert result.duration_s >= 0.0

    @pytest.mark.asyncio
    async def test_timestamps_set(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_test_ts")
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at


# ---------------------------------------------------------------------------
# Run directory and file layout
# ---------------------------------------------------------------------------

class TestRunDirectoryLayout:
    @pytest.mark.asyncio
    async def test_run_dir_created(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_dir_test")
        assert result.run_dir.exists()
        assert result.run_dir.is_dir()

    @pytest.mark.asyncio
    async def test_run_id_in_dir_name(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_dirid_check")
        assert "run_dirid_check" in result.run_dir.name

    @pytest.mark.asyncio
    async def test_input_feb_created(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_input_feb")
        assert result.input_feb.exists()
        assert result.input_feb.name == "input.feb"

    @pytest.mark.asyncio
    async def test_log_file_created(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_log_check")
        assert result.log_file.exists()
        assert result.log_file.name == "log.txt"

    @pytest.mark.asyncio
    async def test_log_file_contains_output(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_log_content")
        log_text = result.log_file.read_text(encoding="utf-8", errors="replace")
        assert "NORMAL TERMINATION" in log_text

    @pytest.mark.asyncio
    async def test_metadata_json_created(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_meta_check")
        assert result.metadata_file.exists()
        assert result.metadata_file.name == "metadata.json"

    @pytest.mark.asyncio
    async def test_xplt_path_reported(self, runner, success_proc, minimal_feb):
        """xplt_file path is always reported even though the solver mock won't create it."""
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_xplt_path")
        assert result.xplt_file.name == "input.xplt"
        assert result.xplt_file.parent == result.run_dir

    @pytest.mark.asyncio
    async def test_all_paths_inside_run_dir(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_paths_check")
        for p in (result.input_feb, result.log_file, result.xplt_file, result.metadata_file):
            assert p.parent == result.run_dir


# ---------------------------------------------------------------------------
# Metadata JSON content
# ---------------------------------------------------------------------------

class TestMetadataJson:
    @pytest.mark.asyncio
    async def test_metadata_is_valid_json(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_meta_json")
        data = json.loads(result.metadata_file.read_text())
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_metadata_has_required_keys(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_meta_keys")
        data = json.loads(result.metadata_file.read_text())
        for key in ("run_id", "status", "speed_mm_s", "command",
                    "started_at", "completed_at", "duration_s", "exit_code"):
            assert key in data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_metadata_status_completed(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=4.0, run_id="run_meta_status")
        data = json.loads(result.metadata_file.read_text())
        assert data["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_metadata_speed_recorded(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=4.5, run_id="run_meta_speed")
        data = json.loads(result.metadata_file.read_text())
        assert data["speed_mm_s"] == pytest.approx(4.5)

    @pytest.mark.asyncio
    async def test_metadata_command_recorded(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_meta_cmd")
        data = json.loads(result.metadata_file.read_text())
        assert data["command"] == ["febio4", "-i", "input.feb"]

    @pytest.mark.asyncio
    async def test_metadata_cfg_details_recorded(self, runner, success_proc, minimal_feb):
        """lc_end_time and time_steps_step2 from configurator must be in metadata."""
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=4.0, run_id="run_meta_cfg")
        data = json.loads(result.metadata_file.read_text())
        assert data["lc_end_time"] == pytest.approx(4.5)
        assert data["time_steps_step2"] == 50


# ---------------------------------------------------------------------------
# Failed run
# ---------------------------------------------------------------------------

class TestFailedRun:
    @pytest.mark.asyncio
    async def test_status_failed(self, runner, failure_proc, minimal_feb):
        with _patch_subprocess(failure_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_fail_status")
        assert result.status == SimulationStatus.FAILED

    @pytest.mark.asyncio
    async def test_succeeded_false(self, runner, failure_proc, minimal_feb):
        with _patch_subprocess(failure_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_fail_succ")
        assert result.succeeded is False

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, runner, failure_proc, minimal_feb):
        with _patch_subprocess(failure_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_fail_ec")
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_error_message_set(self, runner, failure_proc, minimal_feb):
        with _patch_subprocess(failure_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_fail_errmsg")
        assert result.error_message is not None
        assert "1" in result.error_message   # exit code in message

    @pytest.mark.asyncio
    async def test_metadata_status_failed(self, runner, failure_proc, minimal_feb):
        with _patch_subprocess(failure_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_fail_meta")
        data = json.loads(result.metadata_file.read_text())
        assert data["status"] == "FAILED"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def _timeout_wait_for(coro, timeout):
    """
    Replacement for asyncio.wait_for that closes the coroutine cleanly
    (avoiding 'coroutine was never awaited' ResourceWarning) then raises
    TimeoutError to simulate a real timeout.
    """
    coro.close()
    raise asyncio.TimeoutError


class TestTimeout:
    @pytest.mark.asyncio
    async def test_status_timeout(self, runner, minimal_feb):
        with patch(
            "digital_twin_ui.simulation.simulation_runner.asyncio.wait_for",
            side_effect=_timeout_wait_for,
        ):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_timeout")
        assert result.status == SimulationStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_timeout_error_message(self, runner, minimal_feb):
        with patch(
            "digital_twin_ui.simulation.simulation_runner.asyncio.wait_for",
            side_effect=_timeout_wait_for,
        ):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_timeout_msg")
        assert result.error_message is not None
        assert "Timeout" in result.error_message or "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_timeout_metadata_written(self, runner, minimal_feb):
        with patch(
            "digital_twin_ui.simulation.simulation_runner.asyncio.wait_for",
            side_effect=_timeout_wait_for,
        ):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_timeout_meta")
        assert result.metadata_file.exists()
        data = json.loads(result.metadata_file.read_text())
        assert data["status"] == "TIMEOUT"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def _make_cancel_proc(returncode: int = -15) -> MagicMock:
    """
    Mock proc that simulates a process killed by the cancel watcher.

    terminate() and kill() are no-ops.
    wait() returns returncode.
    stdout yields nothing (proc was killed before writing output).
    """
    proc = MagicMock()
    proc.returncode = None  # starts as None — not yet finished

    terminate_called = []
    kill_called = []

    def _terminate():
        proc.returncode = returncode
        terminate_called.append(True)

    proc.terminate = _terminate

    def _kill():
        proc.returncode = returncode
        kill_called.append(True)

    proc.kill = _kill

    async def _aiter():
        return
        yield  # make it an async generator

    proc.stdout = _aiter()

    async def _wait():
        return proc.returncode if proc.returncode is not None else returncode

    proc.wait = _wait
    return proc


class TestCancellation:
    @pytest.mark.asyncio
    async def test_status_cancelled(self, runner, minimal_feb):
        """Writing a CANCEL file while the simulation is running marks it CANCELLED."""
        proc = _make_cancel_proc()

        async def _slow_stream_to_log(p, log_path):
            """Simulate a long-running process — pause until cancelled."""
            try:
                await asyncio.sleep(10)  # would run for 10s normally
            except asyncio.CancelledError:
                pass
            # Write an empty log file
            log_path.write_bytes(b"")

        with _patch_subprocess(proc):
            with patch.object(
                runner,
                "_stream_to_log",
                side_effect=_slow_stream_to_log,
            ):
                # Schedule writing the CANCEL file after a short delay
                async def _write_cancel():
                    await asyncio.sleep(0.05)
                    cfg = runner._cfg
                    cancel_file = cfg.runs_dir_abs / "run_cancel_test" / "CANCEL"
                    cancel_file.touch()

                cancel_writer = asyncio.create_task(_write_cancel())
                result = await runner.run_async(speed_mm_s=5.0, run_id="run_cancel_test")
                await cancel_writer

        assert result.status == SimulationStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancelled_error_message(self, runner, minimal_feb):
        proc = _make_cancel_proc()

        async def _slow_stream(p, log_path):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass
            log_path.write_bytes(b"")

        with _patch_subprocess(proc):
            with patch.object(runner, "_stream_to_log", side_effect=_slow_stream):
                async def _write_cancel():
                    await asyncio.sleep(0.05)
                    (runner._cfg.runs_dir_abs / "run_cancel_errmsg" / "CANCEL").touch()

                t = asyncio.create_task(_write_cancel())
                result = await runner.run_async(speed_mm_s=5.0, run_id="run_cancel_errmsg")
                await t

        assert result.error_message is not None
        assert "Cancelled" in result.error_message or "cancelled" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_cancelled_metadata_written(self, runner, minimal_feb):
        proc = _make_cancel_proc()

        async def _slow_stream(p, log_path):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass
            log_path.write_bytes(b"")

        with _patch_subprocess(proc):
            with patch.object(runner, "_stream_to_log", side_effect=_slow_stream):
                async def _write_cancel():
                    await asyncio.sleep(0.05)
                    (runner._cfg.runs_dir_abs / "run_cancel_meta" / "CANCEL").touch()

                t = asyncio.create_task(_write_cancel())
                result = await runner.run_async(speed_mm_s=5.0, run_id="run_cancel_meta")
                await t

        data = json.loads(result.metadata_file.read_text())
        assert data["status"] == "CANCELLED"

    @pytest.mark.asyncio
    async def test_no_cancel_file_completes_normally(self, runner, success_proc, minimal_feb):
        """Without a CANCEL file, the simulation completes normally."""
        with _patch_subprocess(success_proc):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_no_cancel")
        assert result.status == SimulationStatus.COMPLETED

    def test_cancelled_status_in_enum(self):
        assert SimulationStatus.CANCELLED == "CANCELLED"


# ---------------------------------------------------------------------------
# Pre-execution failure (configurator raises)
# ---------------------------------------------------------------------------

class TestPreExecutionFailure:
    @pytest.mark.asyncio
    async def test_status_failed_on_config_error(self, runner):
        with patch.object(
            runner._configurator,
            "configure",
            side_effect=FileNotFoundError("template missing"),
        ):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_pre_fail")
        assert result.status == SimulationStatus.FAILED

    @pytest.mark.asyncio
    async def test_error_message_on_config_error(self, runner):
        with patch.object(
            runner._configurator,
            "configure",
            side_effect=ValueError("bad speed"),
        ):
            result = await runner.run_async(speed_mm_s=-1.0, run_id="run_pre_err_msg")
        assert result.error_message is not None
        assert "Configuration failed" in result.error_message

    @pytest.mark.asyncio
    async def test_metadata_written_on_config_error(self, runner):
        with patch.object(
            runner._configurator,
            "configure",
            side_effect=RuntimeError("xml broken"),
        ):
            result = await runner.run_async(speed_mm_s=5.0, run_id="run_pre_meta")
        assert result.metadata_file.exists()


# ---------------------------------------------------------------------------
# RunResult dataclass
# ---------------------------------------------------------------------------

class TestRunResult:
    def _make_result(self, tmp_path) -> RunResult:
        return RunResult(
            run_id="run_test",
            status=SimulationStatus.COMPLETED,
            speed_mm_s=5.0,
            run_dir=tmp_path,
            input_feb=tmp_path / "input.feb",
            log_file=tmp_path / "log.txt",
            xplt_file=tmp_path / "results.xplt",
            metadata_file=tmp_path / "metadata.json",
            command=["febio4", "-i", "input.feb"],
            exit_code=0,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2026, 1, 1, second=30, tzinfo=timezone.utc),
            duration_s=30.0,
        )

    def test_succeeded_true_when_completed(self, tmp_path):
        r = self._make_result(tmp_path)
        assert r.succeeded is True

    def test_succeeded_false_when_failed(self, tmp_path):
        r = self._make_result(tmp_path)
        from dataclasses import replace
        r = replace(r, status=SimulationStatus.FAILED)
        assert r.succeeded is False

    def test_as_dict_contains_all_keys(self, tmp_path):
        r = self._make_result(tmp_path)
        d = r.as_dict()
        for key in ("run_id", "status", "speed_mm_s", "run_dir", "input_feb",
                    "log_file", "xplt_file", "metadata_file", "command",
                    "exit_code", "started_at", "completed_at", "duration_s"):
            assert key in d

    def test_as_dict_status_is_string(self, tmp_path):
        r = self._make_result(tmp_path)
        assert isinstance(r.as_dict()["status"], str)
        assert r.as_dict()["status"] == "COMPLETED"

    def test_as_dict_paths_are_strings(self, tmp_path):
        r = self._make_result(tmp_path)
        d = r.as_dict()
        for key in ("run_dir", "input_feb", "log_file", "xplt_file", "metadata_file"):
            assert isinstance(d[key], str)

    def test_as_dict_is_json_serialisable(self, tmp_path):
        r = self._make_result(tmp_path)
        json.dumps(r.as_dict())   # must not raise


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

class TestSyncWrapper:
    def test_run_returns_run_result(self, runner, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = runner.run(speed_mm_s=5.0, run_id="run_sync_test")
        assert isinstance(result, RunResult)
        assert result.status == SimulationStatus.COMPLETED


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

class TestRunSimulationFunction:
    def test_returns_run_result(self, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = run_simulation(speed_mm_s=5.0, run_id="run_convenience")
        assert isinstance(result, RunResult)

    def test_speed_recorded(self, success_proc, minimal_feb):
        with _patch_subprocess(success_proc):
            result = run_simulation(speed_mm_s=4.0, run_id="run_conv_speed")
        assert result.speed_mm_s == pytest.approx(4.0)
