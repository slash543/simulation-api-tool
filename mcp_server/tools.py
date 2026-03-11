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



def tool_list_catheter_designs() -> str:
    """
    Return all catheter designs with their available configurations.

    The response contains:
      - designs: list of { name, label, configurations: [{key, label, feb_file}] }
      - n_steps:                   number of insertion steps (10 for all current designs)
      - displacements_mm:          per-step displacement magnitudes in mm
      - speed_range_min/max:       valid insertion speed bounds in mm/s
      - default_uniform_speed_mm_s: default speed for uniform profiles (15 mm/s)
      - default_dwell_time_s:      default dwell time per step (1.0 s)

    Call this FIRST whenever the user asks to run a simulation.
    Present the three designs by label, then show configurations for the chosen design.
    """
    try:
        with _client() as c:
            r = c.get("/catheter-designs")
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPError as exc:
        return _err(f"List catheter designs error: {exc}")


def tool_run_catheter_simulation(
    design: str,
    configuration: str,
    speeds_mm_s: list[float],
    dwell_time_s: float = 1.0,
) -> str:
    """
    Submit a FEBio simulation for a catheter design + configuration with per-step speeds.

    Reads base_configuration/<feb_file> and modifies ONLY the load curve time
    intervals and time_steps counts.  All geometry/material/contact is preserved.

    For each step i:  ramp_i = displacement_mm[i] / speeds_mm_s[i]

    Returns IMMEDIATELY.  ALWAYS tell the user:
      1. The simulation is running in the background.
      2. host_run_dir — folder on their machine where files will appear.
      3. host_xplt_path — the .xplt to open in FEBio Studio (File > Open).
      4. log.txt inside that folder shows live solver progress.

    Args:
        design:        Tip design key (e.g. "ball_tip", "nelaton_tip", "vapro_introducer").
        configuration: Size x urethra-model key (e.g. "14Fr_IR12", "14Fr_IR25", "16Fr_IR12").
        speeds_mm_s:   Per-step insertion speeds in mm/s (length = n_steps = 10).
                       For uniform speed, repeat the same value 10 times.
        dwell_time_s:  Dwell time in seconds after each ramp (default 1.0).

    Returns:
        JSON with task_id, run_id, host_run_dir, host_xplt_path, status=PENDING.
    """
    payload = {
        "design": design,
        "configuration": configuration,
        "speeds_mm_s": speeds_mm_s,
        "dwell_time_s": dwell_time_s,
    }
    try:
        with _client() as c:
            r = c.post("/simulations/run-catheter", json=payload)
            r.raise_for_status()
            data = r.json()

        if data.get("xplt_path"):
            data["host_xplt_path"] = _to_host_path(data["xplt_path"])
        if data.get("run_dir"):
            data["host_run_dir"] = _to_host_path(data["run_dir"])

        return _ok(data)
    except httpx.HTTPStatusError as exc:
        return _err(
            f"Submit failed ({exc.response.status_code}): {exc.response.text}"
        )
    except httpx.HTTPError as exc:
        return _err(f"Request error: {exc}")


# ---------------------------------------------------------------------------
# RAG document tools
# ---------------------------------------------------------------------------

def tool_list_research_documents() -> str:
    """
    Return all PDF filenames currently indexed in the research document store,
    plus the total chunk count.

    Call this to check what has been ingested before searching.
    """
    try:
        with _client() as c:
            r = c.get("/documents/list")
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPError as exc:
        return _err(f"List documents error: {exc}")


def tool_ingest_research_documents(force: bool = False) -> str:
    """
    Scan the research_documents/ folder for new PDFs and index them.

    Each PDF is parsed (including scanned/OCR PDFs via docling), split into
    chunks, embedded with a local sentence-transformers model, and stored in
    a ChromaDB vector index.  Already-indexed PDFs are skipped unless
    force=True.

    Call this:
      - After adding new PDFs to research_documents/
      - With force=True after replacing a PDF with an updated version

    Args:
        force: Re-ingest all PDFs even if already indexed (default False).

    Returns:
        JSON with n_ingested, n_skipped, n_failed, and per-file results.
    """
    try:
        with _client() as c:
            r = c.post("/documents/ingest", params={"force": str(force).lower()})
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPStatusError as exc:
        return _err(f"Ingest failed ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        return _err(f"Request error: {exc}")


def tool_search_research_documents(query: str, n_results: int = 5) -> str:
    """
    Semantically search the indexed research documents and return relevant excerpts.

    Uses a local sentence-transformers embedding model (BAAI/bge-small-en-v1.5)
    and ChromaDB cosine-similarity search — no external API required.

    WORKFLOW:
      1. If the store might be empty, call list_research_documents() first.
      2. If empty, call ingest_research_documents() to index the PDFs.
      3. Then call this tool with the user's question.

    After receiving results, synthesise a clear answer from the returned chunks,
    always citing the source PDF filename and chunk index for each piece of
    information used.

    Args:
        query:     Natural-language question (e.g. "What is the Young's modulus
                   of the catheter material?").
        n_results: Number of chunks to retrieve (default 5, max 20).

    Returns:
        JSON with query, total_hits, and hits list (text, source, chunk_index, score).
        Score is cosine similarity in [0, 1] — higher means more relevant.
    """
    try:
        with _client() as c:
            r = c.post(
                "/documents/search",
                json={"query": query, "n_results": n_results},
            )
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            return _err(
                "No documents indexed. "
                "Call ingest_research_documents() first to index the research_documents/ folder."
            )
        return _err(f"Search failed ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        return _err(f"Request error: {exc}")


def tool_preview_doe_speeds(
    n_samples: int = 10,
    speed_min: float = 10.0,
    speed_max: float = 25.0,
    n_steps: int = 10,
    max_perturbation: float = 0.20,
    seed: int | None = None,
) -> str:
    """
    Generate and return DOE speed arrays without running any simulations.

    Uses CorrelatedSpeedSampler to produce n_samples arrays of n_steps correlated
    per-step speeds in mm/s, each row sorted ascending within [speed_min, speed_max].

    Use this to show the user what speed profiles a DOE campaign would use
    before committing to the full run, or to help the user choose their 10 speeds.

    Args:
        n_samples:        Number of speed-array samples to generate (default 10).
        speed_min:        Minimum mean speed in mm/s (default 10.0).
        speed_max:        Maximum mean speed in mm/s (default 25.0).
        n_steps:          Steps per sample (default 10, must match the design).
        max_perturbation: Max fractional per-step perturbation (0.20 = +/-20%).
        seed:             Optional RNG seed for reproducibility.

    Returns:
        JSON with samples: list of n_samples speed arrays, each of length n_steps.
    """
    payload: dict[str, Any] = {
        "n_samples": n_samples,
        "speed_min": speed_min,
        "speed_max": speed_max,
        "n_steps": n_steps,
        "max_perturbation": max_perturbation,
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        with _client() as c:
            r = c.post("/doe/preview-speeds", json=payload)
            r.raise_for_status()
            return _ok(r.json())
    except httpx.HTTPStatusError as exc:
        return _err(
            f"Preview failed ({exc.response.status_code}): {exc.response.text}"
        )
    except httpx.HTTPError as exc:
        return _err(f"Request error: {exc}")
