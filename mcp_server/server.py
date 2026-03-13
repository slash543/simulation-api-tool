"""
MCP Server — Digital Twin Simulation Tools
==========================================

Exposes the FastAPI simulation endpoints as MCP tools so that any
MCP-compatible LLM client (LibreChat + Ollama, Claude Desktop, VS Code Copilot
Chat, etc.) can trigger simulations, DOE campaigns, and ML predictions through
natural-language conversation.

Transport: SSE  (Server-Sent Events — HTTP-based, container-friendly)
Default:   http://0.0.0.0:8001/sse

Usage:
    python server.py                # start SSE server on :8001
    MCP_PORT=9001 python server.py  # custom port
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools import (
    tool_cancel_simulation,
    tool_get_doe_status,
    tool_get_task_status,
    tool_health_check,
    tool_ingest_research_documents,
    tool_list_catheter_designs,
    tool_list_research_documents,
    tool_list_simulation_jobs,
    tool_list_templates,
    tool_predict_pressure,
    tool_predict_pressure_batch,
    tool_preview_doe_speeds,
    tool_refresh_catalogue,
    tool_run_catheter_simulation,
    tool_run_doe_campaign,
    tool_run_simulation,
    tool_search_research_documents,
    tool_submit_simulation,
)

# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "digital-twin-simulation",
    instructions=(
        "You are an expert digital twin simulation assistant for catheter insertion "
        "biomechanics. You have tools to:\n"
        "  • list and refresh available catheter designs (tip geometries)\n"
        "  • run FEBio FEM simulations for a chosen catheter design with per-step speeds\n"
        "  • list, inspect, and cancel simulation jobs\n"
        "  • run DOE campaigns to build a synthetic simulation database\n"
        "  • predict contact pressure instantly using the trained ML model\n"
        "  • poll async task results\n\n"
        "WORKFLOW WHEN USER ASKS TO RUN A SIMULATION:\n"
        "  1. Call list_catheter_designs() → returns ALL designs auto-detected from\n"
        "     base_configuration/ at startup.  If the user says they just added a new\n"
        "     .feb file, call refresh_catalogue() FIRST to pick it up without restart.\n"
        "     Present every tip design by label.\n"
        "  2. Ask: 'Which catheter tip design?' (e.g. Ball Tip / Nelaton Tip / Vapro Introducer)\n"
        "  3. Show configurations for the chosen design → ask: 'Which size / urethra model?'\n"
        "     (e.g. 14Fr IR12, 14Fr IR25, 16Fr IR12)\n"
        "  4. Ask for insertion speeds. The design has 10 steps; you may ask for:\n"
        "       a) A UNIFORM speed — one value (10–25 mm/s) applied to all 10 steps.\n"
        "       b) PER-STEP speeds — 10 individual values (one per step).\n"
        "       c) Call preview_doe_speeds() to show example correlated speed profiles.\n"
        "     If the user says e.g. '15 mm/s uniform', repeat that value 10 times.\n"
        "  5. Confirm the full 10-element speed array with the user, then call\n"
        "     run_catheter_simulation(design, configuration, speeds_mm_s).\n\n"
        "AFTER run_catheter_simulation() RETURNS:\n"
        "  Always tell the user ALL of:\n"
        "  • The simulation is running in the background.\n"
        "  • host_run_dir — folder on their machine where result files will appear.\n"
        "  • host_xplt_path — the .xplt to open in FEBio Studio once the run finishes.\n"
        "  • log.txt inside that folder shows live solver progress.\n"
        "  • They can cancel at any time — just ask.\n\n"
        "LISTING SIMULATION JOBS:\n"
        "  If the user asks 'what simulations are running?', 'show my jobs', or wants\n"
        "  to find a result file:\n"
        "  1. Call list_simulation_jobs() → returns all run directories, newest first.\n"
        "  2. For each job, show: run_id, status, host_run_dir, host_xplt_path.\n"
        "  3. For completed jobs, host_xplt_path is the file to open in FEBio Studio.\n\n"
        "CANCELLING A SIMULATION:\n"
        "  If the user asks to stop, cancel, abort, or kill a simulation:\n"
        "  1. If you don't have the task_id and run_id, call list_simulation_jobs()\n"
        "     to find the running job, then ask the user to confirm which one.\n"
        "     NOTE: list_simulation_jobs() does NOT return task_id — if you need it,\n"
        "     ask the user (it was shown when the job was submitted).\n"
        "  2. Call cancel_simulation(task_id, run_id).\n"
        "  3. Tell the user the simulation will stop within ~1 second.\n\n"
        "ADDING NEW .FEB FILES (after git clone on a new VM):\n"
        "  New .feb files placed in base_configuration/ are picked up automatically.\n"
        "  If the stack is already running: call refresh_catalogue() once to rescan\n"
        "  without restarting.  Then call list_catheter_designs() to confirm.\n\n"
        "Use predict_pressure() for instant ML estimates (requires prior training). "
        "For building a database, prefer run_doe_campaign() with 10–50 samples.\n\n"
        "RESEARCH DOCUMENT WORKFLOW:\n"
        "  When a user asks about catheter biomechanics, FEBio, or simulation methods:\n"
        "  1. Call search_research_documents(query) to find relevant excerpts.\n"
        "  2. If the store is empty, call ingest_research_documents() first.\n"
        "  3. Synthesise a clear answer and cite the source PDF filenames."
    ),
)


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
def health_check() -> str:
    """Check that the simulation API is up and healthy."""
    return tool_health_check()


@mcp.tool()
def run_simulation(speed_mm_s: float) -> str:
    """
    Submit a catheter insertion FEBio simulation and return IMMEDIATELY.

    The simulation runs in the background — do NOT wait for it to finish.
    After calling this tool, ALWAYS tell the user:
      1. The simulation is running in the background.
      2. The result folder on their machine: host_run_dir
      3. The xplt file they can open in FEBio Studio: host_xplt_path
         (File > Open in FEBio Studio once the file appears)
      4. Progress can be watched in: host_run_dir/log.txt

    Args:
        speed_mm_s: Catheter insertion speed in mm/s (valid range: 0.5 – 20.0).

    Returns:
        JSON with task_id, run_id, host_run_dir, host_xplt_path, status=PENDING.
    """
    return tool_run_simulation(speed_mm_s)


@mcp.tool()
def cancel_simulation(task_id: str, run_id: str) -> str:
    """
    Cancel a running or queued simulation immediately.

    Call this when the user asks to stop, abort, cancel, or kill a simulation.

    Mechanism: writes a CANCEL sentinel in the run directory (worker picks it
    up within ~1 second and kills the FEBio subprocess) and revokes the Celery
    task.  Returns immediately — cancellation is asynchronous.

    Prerequisites: you need the task_id and run_id from the submission response.
    Both are returned by run_catheter_simulation() and run_simulation().

    Args:
        task_id: Celery task ID from the submission response.
        run_id:  Run identifier from the submission response.

    Returns:
        JSON with status='CANCELLATION_REQUESTED' and a confirmation message.
    """
    return tool_cancel_simulation(task_id, run_id)


@mcp.tool()
def submit_simulation(speed_mm_s: float) -> str:
    """
    Submit a simulation job **asynchronously** and return immediately.

    The simulation runs in the background.  Use get_task_status(task_id) to
    poll for completion and retrieve the result.

    Args:
        speed_mm_s: Catheter insertion speed in mm/s.

    Returns:
        JSON with task_id (use to poll) and initial status='PENDING'.
    """
    return tool_submit_simulation(speed_mm_s)


@mcp.tool()
def get_task_status(task_id: str) -> str:
    """
    Poll the status of an async simulation task.

    Args:
        task_id: The task ID returned by submit_simulation().

    Returns:
        JSON with status (PENDING | STARTED | SUCCESS | FAILURE) and,
        when complete, the full simulation result.
    """
    return tool_get_task_status(task_id)


@mcp.tool()
def list_templates() -> str:
    """
    List all available simulation templates with their configurations.

    Returns a JSON object with a ``templates`` array.  Each entry contains:
      - name:               Template identifier (use in run_doe_campaign / run_simulation)
      - label:              Human-readable display name
      - n_steps:            Number of sequential insertion steps (1 or 10)
      - speed_range_min/max: Valid insertion speed bounds in mm/s
      - displacements_mm:   Per-step prescribed displacement magnitudes

    Available templates:
      - sample_catheterization  — toy 2-step case (backwards compatible)
      - DT_BT_14Fr_FO_10E_IR12  — 14Fr catheter, 10-step, urethra IR12
      - DT_BT_14Fr_FO_10E_IR25  — 14Fr catheter, 10-step, urethra IR25

    Returns:
        JSON with ``templates`` list.
    """
    return tool_list_templates()


@mcp.tool()
def run_doe_campaign(
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
    Submit a Design of Experiments campaign to build a synthetic simulation database.

    For multi-step templates (DT_BT_14Fr_FO_10E_IR12, DT_BT_14Fr_FO_10E_IR25),
    uses CorrelatedSpeedSampler to generate physiologically plausible per-step
    speed vectors — each sample has 10 correlated speeds with a random mean and
    small per-step perturbations sorted ascending to simulate speed ramp-up.

    For sample_catheterization (single-step), uses the standard 1-D scalar sampler.

    Runs `n_samples` simulations and stores results for ML training.

    Args:
        n_samples:        Number of simulation points (recommended: 10 – 50).
        speed_min:        Minimum mean insertion speed in mm/s (e.g. 10.0).
        speed_max:        Maximum mean insertion speed in mm/s (e.g. 25.0).
        sampler:          Sampling strategy for single-step templates — 'lhs'
                          (Latin Hypercube, default), 'sobol', or 'uniform'.
        seed:             Optional integer seed for reproducibility.
        template:         Template name — use list_templates() to see options.
                          Default: 'DT_BT_14Fr_FO_10E_IR12'.
        max_perturbation: Maximum fractional perturbation per step (0.0–0.5).
                          0.20 means each step speed varies by up to ±20% of
                          the mean. Only applies to multi-step templates.
        dwell_time_s:     Dwell time (seconds) appended after each insertion
                          ramp.  Longer dwell gives the tissue more time to
                          relax between steps.

    Returns:
        JSON with task_id.  Poll with get_doe_status(task_id).
    """
    return tool_run_doe_campaign(
        n_samples,
        speed_min,
        speed_max,
        sampler,
        seed,
        template,
        max_perturbation,
        dwell_time_s,
    )


@mcp.tool()
def get_doe_status(task_id: str) -> str:
    """
    Poll the status of an async DOE campaign task.

    Args:
        task_id: The task ID returned by run_doe_campaign().

    Returns:
        JSON with status and, when complete, a summary of runs executed.
    """
    return tool_get_doe_status(task_id)


@mcp.tool()
def predict_pressure(speed_mm_s: float) -> str:
    """
    Predict catheter-tissue contact pressure **instantly** via the trained ML model.

    Much faster than a FEM simulation.  Requires that the ML model has been
    trained first (run a DOE campaign, then trigger the training pipeline).

    Args:
        speed_mm_s: Insertion speed in mm/s.

    Returns:
        JSON with speed_mm_s and predicted_pressure_pa.
    """
    return tool_predict_pressure(speed_mm_s)


@mcp.tool()
def predict_pressure_batch(speeds_mm_s: list[float]) -> str:
    """
    Predict contact pressures for **multiple** insertion speeds in one call.

    Args:
        speeds_mm_s: List of insertion speeds in mm/s (e.g. [2.0, 4.0, 6.0, 8.0]).

    Returns:
        JSON list of {speed_mm_s, predicted_pressure_pa} entries.
    """
    return tool_predict_pressure_batch(speeds_mm_s)


@mcp.tool()
def list_catheter_designs() -> str:
    """
    Return all catheter designs with their available configurations.

    ALWAYS call this first when a user asks to run a simulation.

    The list is built dynamically: all .feb files in the base_configuration/
    folder are detected automatically at startup using the filename convention
    <design>_<size>Fr[_extra]_ir<ir>.feb  (e.g. ball_tip_14Fr_ir12.feb).
    New designs appear here as soon as the stack is restarted after adding a file.

    Conversation flow:
      1. Call this tool → present all detected tip designs by label.
      2. Ask: "Which catheter tip design would you like?"
      3. Show configurations for the chosen design → ask: "Which size / urethra model?"
      4. Ask for insertion speeds:
           - Uniform: one value repeated 10 times (e.g. 15 mm/s uniform)
           - Per-step: 10 individual values
      5. Call run_catheter_simulation() with design, configuration, speeds_mm_s.

    Returns:
        JSON with designs list + shared simulation params (n_steps, displacements_mm,
        speed_range_min/max, default_uniform_speed_mm_s, default_dwell_time_s).
    """
    return tool_list_catheter_designs()


@mcp.tool()
def refresh_catalogue() -> str:
    """
    Rescan base_configuration/ for new .feb files and refresh the design catalogue.

    Call this after the user has dropped a new .feb file into base_configuration/
    on the host (e.g. after 'git clone' on a new VM and adding files) WITHOUT
    restarting the containers.  Returns the same JSON as list_catheter_designs()
    but with the freshly discovered designs included.

    Typical usage:
      1. User: 'I added a new .feb file, can you see it?'
      2. Call refresh_catalogue() → confirm the new design appears in the list.
      3. Proceed with the normal simulation workflow.

    Returns:
        JSON with designs list + shared simulation params (same as list_catheter_designs).
    """
    return tool_refresh_catalogue()


@mcp.tool()
def list_simulation_jobs() -> str:
    """
    Return all simulation runs found in the runs/ directory, newest first.

    Use this when the user asks:
      • 'What simulations are running?'
      • 'Show me my recent jobs'
      • 'Where is my result file?'
      • 'I want to cancel a simulation but I don't have the run_id'

    For each job the response includes:
      - run_id:         folder name (use in cancel_simulation)
      - status:         'completed' | 'cancelled' | 'running' | 'unknown'
      - host_run_dir:   path on the HOST machine to the run folder
      - host_xplt_path: path on the HOST machine to the .xplt result file
      - xplt_exists:    true when the solver has finished writing the file
      - log_path:       path to the live solver log inside the container

    Note: task_id is NOT returned here — it was shown at job submission time.
    If needed for cancel_simulation(), ask the user to provide it.

    Returns:
        JSON with jobs list (newest first) and total count.
    """
    return tool_list_simulation_jobs()


@mcp.tool()
def run_catheter_simulation(
    design: str,
    configuration: str,
    speeds_mm_s: list[float],
    dwell_time_s: float = 1.0,
) -> str:
    """
    Submit a FEBio simulation for a catheter design + configuration with per-step speeds.

    PREREQUISITE: call list_catheter_designs() and confirm design, configuration,
    and all 10 speeds with the user before calling this tool.

    Only the load curve time intervals and time_steps counts are modified —
    all geometry, material, and contact definitions are preserved from the base file.

    If the user specifies a single uniform speed (e.g. "15 mm/s"), repeat it
    10 times automatically. If they specify per-step speeds, use them directly.
    If they are unsure, call preview_doe_speeds() to show example profiles.

    Returns IMMEDIATELY.  ALWAYS tell the user host_run_dir and host_xplt_path.

    Args:
        design:        Tip design key (e.g. "ball_tip", "nelaton_tip", "vapro_introducer").
        configuration: Config key (e.g. "14Fr_IR12", "14Fr_IR25", "16Fr_IR12").
        speeds_mm_s:   10 insertion speeds in mm/s (one per step).
        dwell_time_s:  Dwell time in seconds after each ramp (default 1.0).

    Returns:
        JSON with task_id, run_id, host_run_dir, host_xplt_path, status=PENDING.
    """
    return tool_run_catheter_simulation(design, configuration, speeds_mm_s, dwell_time_s)


@mcp.tool()
def list_research_documents() -> str:
    """
    Return all PDF filenames currently indexed in the research document store.

    IMPORTANT: If total_chunks is 0 (store is empty), immediately call
    ingest_research_documents() WITHOUT asking the user first — PDFs are
    already present in the research_documents/ folder and just need indexing.
    Do NOT tell the user the store is empty or ask them to upload files.
    Just ingest silently and then answer their original question.

    Returns:
        JSON with sources (list of filenames) and total_chunks count.
    """
    return tool_list_research_documents()


@mcp.tool()
def ingest_research_documents(force: bool = False) -> str:
    """
    Scan research_documents/ for new PDFs and index them for semantic search.

    Already-indexed PDFs are skipped unless force=True.  Call this:
      - After adding new PDFs to the research_documents/ folder.
      - With force=True after replacing a PDF with an updated version.

    Ingestion pipeline (fully local, no external APIs):
      • docling  — parses text-layer PDFs and scanned/OCR PDFs
      • sentence-transformers (BAAI/bge-small-en-v1.5, MIT) — local embeddings
      • ChromaDB — persistent vector index on disk

    Args:
        force: Re-ingest all PDFs even if already indexed (default False).

    Returns:
        JSON with n_ingested, n_skipped, n_failed, and per-file results.
    """
    return tool_ingest_research_documents(force)


@mcp.tool()
def search_research_documents(query: str, n_results: int = 5) -> str:
    """
    Semantically search the indexed research documents and return relevant excerpts.

    Uses local embeddings and ChromaDB — no external API calls.

    WORKFLOW:
      1. If you haven't searched before, call list_research_documents() first.
      2. If the store is empty, call ingest_research_documents() to index PDFs.
      3. Then call this tool with the user's question.
      4. Synthesise an answer from the returned chunks and cite the source PDFs.

    Args:
        query:     Natural-language question or keyword string.
        n_results: Number of chunks to retrieve (default 5, max 20).

    Returns:
        JSON with query, total_hits, and hits: [{text, source, chunk_index, score}].
        Score is cosine similarity in [0, 1] — higher = more relevant.
    """
    return tool_search_research_documents(query, n_results)


@mcp.tool()
def preview_doe_speeds(
    n_samples: int = 10,
    speed_min: float = 10.0,
    speed_max: float = 25.0,
    n_steps: int = 10,
    max_perturbation: float = 0.20,
    seed: int | None = None,
) -> str:
    """
    Generate and return DOE speed arrays WITHOUT running any simulations.

    Use this to:
      • Show the user example 10-element speed profiles before they choose their own.
      • Preview what speed arrays a DOE campaign would generate.
      • Help a user who doesn't know what 10 speeds to use.

    Uses CorrelatedSpeedSampler: each sample is a 10-element array sorted
    ascending with small per-step perturbations around a random mean speed.

    Args:
        n_samples:        Number of speed arrays to generate (default 10).
        speed_min:        Minimum speed in mm/s (default 10.0).
        speed_max:        Maximum speed in mm/s (default 25.0).
        n_steps:          Steps per array — must match the design (default 10).
        max_perturbation: Max fractional per-step variation (0.20 = ±20%).
        seed:             Optional RNG seed for reproducibility.

    Returns:
        JSON with samples: list of n_samples speed arrays, each of length n_steps.
    """
    return tool_preview_doe_speeds(
        n_samples, speed_min, speed_max, n_steps, max_perturbation, seed
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    host = os.getenv("FASTMCP_HOST", "0.0.0.0")
    port = int(os.getenv("FASTMCP_PORT", "8001"))

    # Build the SSE ASGI app directly so uvicorn gets exact host/port.
    # This bypasses FastMCP.run() which ignores env vars in some versions.
    _sse = SseServerTransport("/messages/")
    _srv = mcp._mcp_server  # underlying mcp.server.Server instance

    async def _handle_sse(request):
        async with _sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (recv, send):
            await _srv.run(recv, send, _srv.create_initialization_options())

    _app = Starlette(
        routes=[
            Route("/sse", endpoint=_handle_sse),
            Mount("/messages/", app=_sse.handle_post_message),
        ]
    )

    uvicorn.run(_app, host=host, port=port)
