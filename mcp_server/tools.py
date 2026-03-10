"""
Pure-function implementations for MCP simulation tools.

Each function makes one HTTP call to the FastAPI simulation service and returns
a JSON string.  They are kept separate from the MCP decorators so they can be
unit-tested without a running MCP server.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

API_BASE: str = os.getenv("SIMULATION_API_URL", "http://api:8000/api/v1")
HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "300"))

# Host-side path to the runs directory (Docker volume mount point on the host).
# Used to translate container-internal /app/runs/... paths to host paths so
# users can open result files directly in FEBio Studio.
RUNS_HOST_PATH: str = os.getenv(
    "RUNS_HOST_PATH",
    "/home/anukaran/simulation-api-tool/runs",
)
_CONTAINER_RUNS_PREFIX = "/app/runs"


def _to_host_path(container_path: str) -> str:
    """Translate a container-internal path under /app/runs to the host path."""
    if container_path.startswith(_CONTAINER_RUNS_PREFIX):
        relative = container_path[len(_CONTAINER_RUNS_PREFIX):]
        return RUNS_HOST_PATH.rstrip("/") + relative
    return container_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=HTTP_TIMEOUT)


def _ok(data: Any) -> str:
    """Serialise a response payload to a JSON string."""
    return json.dumps(data, default=str)


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_health_check() -> str:
    """Return the simulation API health status."""
    try:
        with _client() as c:
            r = c.get("/health")
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPError as exc:
        return _err(f"API unreachable: {exc}")


def tool_run_simulation(speed_mm_s: float) -> str:
    """
    Submit a FEBio catheter-insertion simulation and return IMMEDIATELY.

    Does NOT wait for the simulation to finish. The solver runs in the
    background; results are written to a dedicated folder on the host machine.

    Always tell the user:
      1. The simulation has been submitted and is running in the background.
      2. The exact host folder path (host_run_dir) where they can watch for files.
      3. That results.xplt (host_xplt_path) will appear in that folder when done.
      4. They can open host_xplt_path in FEBio Studio via File > Open once it appears.
      5. The log file (host_run_dir/log.txt) shows solver progress in real time.

    Returns: task_id, run_id, host_run_dir, host_xplt_path, status=PENDING.
    """
    try:
        with _client() as c:
            r = c.post("/simulations/run", json={"speed_mm_s": speed_mm_s, "extract": False})
            r.raise_for_status()
            data = r.json()

        # Translate container paths to host-accessible paths
        if data.get("xplt_path"):
            data["host_xplt_path"] = _to_host_path(data["xplt_path"])
        if data.get("run_dir"):
            data["host_run_dir"] = _to_host_path(data["run_dir"])

        return _ok(data)
    except httpx.HTTPStatusError as exc:
        return _err(f"Submit failed ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        return _err(f"Request error: {exc}")


def tool_submit_simulation(speed_mm_s: float) -> str:
    """
    Submit a simulation asynchronously and return a task_id.

    Poll the task with tool_get_task_status(task_id) to retrieve the result.
    """
    try:
        with _client() as c:
            r = c.post("/simulations/run", json={"speed_mm_s": speed_mm_s})
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPError as exc:
        return _err(f"Submit error: {exc}")


def tool_get_task_status(task_id: str) -> str:
    """
    Poll the status of an async simulation task.

    Status values: PENDING | STARTED | SUCCESS | FAILURE
    """
    try:
        with _client() as c:
            r = c.get(f"/simulations/{task_id}")
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPError as exc:
        return _err(f"Status check error: {exc}")


def tool_run_doe_campaign(
    n_samples: int,
    speed_min: float,
    speed_max: float,
    sampler: str = "lhs",
    seed: int | None = None,
    template: str = "DT_BT_14Fr_FO_10E_IR12",
    max_perturbation: float = 0.20,
    dwell_time_s: float = 1.0,
) -> str:
    """
    Submit a Design of Experiments (DOE) campaign asynchronously.

    For multi-step templates (e.g. DT_BT_14Fr_FO_10E_IR12), generates
    correlated per-step speed vectors using CorrelatedSpeedSampler.
    For sample_catheterization, uses the standard 1-D scalar sampler.

    Generates `n_samples` simulations across [speed_min, speed_max] using the
    chosen template.  Returns a task_id; poll with tool_get_doe_status().
    """
    payload: dict[str, Any] = {
        "n_samples": n_samples,
        "speed_min": speed_min,
        "speed_max": speed_max,
        "sampler": sampler,
        "template": template,
        "max_perturbation": max_perturbation,
        "dwell_time_s": dwell_time_s,
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        with _client() as c:
            r = c.post("/doe/run", json=payload)
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPStatusError as exc:
        return _err(f"DOE submit failed ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        return _err(f"Request error: {exc}")


def tool_list_templates() -> str:
    """
    Return the list of available simulation templates with their configurations.

    Each template describes a FEB file, the number of insertion steps,
    the valid speed range, and per-step displacement magnitudes.
    Use the template name when calling tool_run_doe_campaign() or
    tool_run_simulation().
    """
    try:
        with _client() as c:
            r = c.get("/templates")
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPError as exc:
        return _err(f"List templates error: {exc}")


def tool_get_doe_status(task_id: str) -> str:
    """Poll the status of a DOE campaign task."""
    try:
        with _client() as c:
            r = c.get(f"/doe/{task_id}")
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPError as exc:
        return _err(f"DOE status error: {exc}")


def tool_predict_pressure(speed_mm_s: float) -> str:
    """
    Predict catheter-tissue contact pressure using the trained ML model.

    Orders of magnitude faster than running a FEM simulation.
    The model must have been trained first (via a DOE campaign + training pipeline).
    """
    try:
        with _client() as c:
            r = c.post("/ml/predict", json={"speed_mm_s": speed_mm_s})
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            return _err("ML model not available — run a DOE campaign and train the model first.")
        return _err(f"Prediction failed ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        return _err(f"Request error: {exc}")


def tool_predict_pressure_batch(speeds_mm_s: list[float]) -> str:
    """
    Predict contact pressures for multiple insertion speeds in one call.

    Returns a list of {speed_mm_s, predicted_pressure_pa} objects.
    """
    try:
        with _client() as c:
            r = c.post("/ml/predict/batch", json={"speeds_mm_s": speeds_mm_s})
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            return _err("ML model not available — run a DOE campaign and train the model first.")
        return _err(f"Batch prediction failed ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        return _err(f"Request error: {exc}")
