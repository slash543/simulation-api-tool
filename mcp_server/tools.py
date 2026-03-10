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
    Run a FEBio catheter-insertion simulation synchronously.

    Blocks until the simulation completes (typically 1–5 min).
    Returns run_id, speed_mm_s, peak_contact_pressure_pa, duration_s and status.
    """
    try:
        with _client() as c:
            r = c.post("/simulations/run/sync", json={"speed_mm_s": speed_mm_s})
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPStatusError as exc:
        return _err(f"Simulation failed ({exc.response.status_code}): {exc.response.text}")
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
) -> str:
    """
    Submit a Design of Experiments (DOE) campaign asynchronously.

    Generates `n_samples` simulations across [speed_min, speed_max] using the
    chosen sampling strategy.  Returns a task_id; poll with tool_get_doe_status().
    """
    payload: dict[str, Any] = {
        "n_samples": n_samples,
        "speed_min": speed_min,
        "speed_max": speed_max,
        "sampler": sampler,
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
