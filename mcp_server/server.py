"""
MCP Server — Digital Twin Simulation Tools

Transport: SSE (Server-Sent Events)
Default:   http://0.0.0.0:8001/sse
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
    tool_search_research_documents,
)

mcp = FastMCP(
    "digital-twin-simulation",
    instructions=(
        "You run catheter insertion FEM simulations.\n\n"
        "CRITICAL: ALWAYS call list_catheter_designs() FIRST before any simulation.\n"
        "Never assume which designs or configurations are available — the set of\n"
        "available .feb files changes when users add new files to base_configuration/.\n"
        "Only use design names and configuration keys returned by list_catheter_designs().\n\n"
        "SPEED: 10–25 mm/s (check speed_range_min/max in list_catheter_designs response).\n"
        "All current designs have 10 steps; repeat the same value for uniform speed.\n\n"
        "WORKFLOW:\n"
        "  1. Call list_catheter_designs() — MANDATORY, every time.\n"
        "  2. Present available designs and configurations to the user.\n"
        "  3. Ask the user to choose design, configuration, and speed(s).\n"
        "  4. Call run_catheter_simulation(design, configuration, speeds_mm_s).\n"
        "  5. Tell user: host_run_dir and host_xplt_path.\n\n"
        "NEW .FEB FILES: If the user says they added a file and it is not in the list,\n"
        "call refresh_catalogue() to rescan base_configuration/ without restarting.\n\n"
        "TO CHECK STATUS: call list_simulation_jobs().\n"
        "TO CANCEL: get run_id from list_simulation_jobs(), ask user for task_id,\n"
        "then call cancel_simulation().\n\n"
        "DOE CAMPAIGNS: call run_doe_campaign(); poll with get_doe_status().\n"
        "ML PREDICTIONS: call predict_pressure() or predict_pressure_batch().\n"
        "RAG DOCUMENTS: list → ingest → search_research_documents().\n"
    ),
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@mcp.tool()
def health_check() -> str:
    """Return the simulation API health status."""
    return tool_health_check()


# ---------------------------------------------------------------------------
# Catheter designs & templates
# ---------------------------------------------------------------------------

@mcp.tool()
def list_catheter_designs() -> str:
    """
    Return all catheter designs with available configurations.

    Call this FIRST when the user asks to run a simulation.
    Returns designs, n_steps, speed_range, displacements_mm, default speeds.
    """
    return tool_list_catheter_designs()


@mcp.tool()
def list_templates() -> str:
    """
    Return all simulation templates with their FEB file, step count, and speed range.

    Use the template name when calling run_doe_campaign().
    """
    return tool_list_templates()


@mcp.tool()
def refresh_catalogue() -> str:
    """
    Rescan base_configuration/ for new .feb files and refresh the catalogue.

    Call this when the user says they added (or copied) a .feb file and it
    does not appear in the list returned by list_catheter_designs().
    The catalogue is updated in-place — no container restart needed.
    Returns the updated list (same format as list_catheter_designs()).
    """
    return tool_refresh_catalogue()


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

@mcp.tool()
def run_catheter_simulation(
    design: str,
    configuration: str,
    speeds_mm_s: list[float],
    dwell_time_s: float = 1.0,
) -> str:
    """
    Submit a FEBio catheter-insertion simulation and return IMMEDIATELY.

    Call list_catheter_designs() first to get valid design and configuration values.

    Args:
        design:        tip design key from list_catheter_designs() (e.g. "ball_tip")
        configuration: size+urethra key from list_catheter_designs() (e.g. "14Fr_IR12")
        speeds_mm_s:   per-step speeds in mm/s; length must equal n_steps from
                       list_catheter_designs() (repeat same value for uniform speed)
        dwell_time_s:  dwell time per step in seconds (default 1.0)

    Returns: task_id, run_id, host_run_dir, host_xplt_path, status=PENDING.
    After submitting ALWAYS tell the user host_run_dir and host_xplt_path.
    """
    return tool_run_catheter_simulation(design, configuration, speeds_mm_s, dwell_time_s)


@mcp.tool()
def list_simulation_jobs() -> str:
    """
    List recent simulation runs (newest first, up to 20).

    Returns: run_id, status (completed|cancelled|running|unknown),
             host_run_dir, host_xplt_path, xplt_exists, created_at.
    """
    return tool_list_simulation_jobs()


@mcp.tool()
def get_task_status(task_id: str) -> str:
    """
    Poll the status of an async simulation task.

    Status values: PENDING | STARTED | SUCCESS | FAILURE
    """
    return tool_get_task_status(task_id)


@mcp.tool()
def cancel_simulation(task_id: str, run_id: str) -> str:
    """
    Cancel a running or queued simulation.

    task_id: from the submission response.
    run_id:  from list_simulation_jobs() or the submission response.
    """
    return tool_cancel_simulation(task_id, run_id)


# ---------------------------------------------------------------------------
# DOE campaigns
# ---------------------------------------------------------------------------

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
    Submit a Design of Experiments (DOE) campaign asynchronously.

    Generates n_samples simulations across [speed_min, speed_max] using
    CorrelatedSpeedSampler for multi-step templates.
    Poll the result with get_doe_status(task_id).

    Args:
        n_samples:        Number of simulations to run.
        speed_min:        Minimum insertion speed in mm/s.
        speed_max:        Maximum insertion speed in mm/s.
        sampler:          Sampling strategy: "lhs" (default) or "random".
        seed:             Optional RNG seed for reproducibility.
        template:         FEB template key (default DT_BT_14Fr_FO_10E_IR12).
        max_perturbation: Max fractional per-step speed variation (default 0.20).
        dwell_time_s:     Dwell time per step in seconds (default 1.0).

    Returns: task_id to poll with get_doe_status().
    """
    return tool_run_doe_campaign(
        n_samples, speed_min, speed_max, sampler, seed, template, max_perturbation, dwell_time_s
    )


@mcp.tool()
def get_doe_status(task_id: str) -> str:
    """Poll the status of a DOE campaign task submitted with run_doe_campaign()."""
    return tool_get_doe_status(task_id)


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
    Preview DOE speed arrays without running any simulations.

    Use this to show the user what per-step speed profiles a DOE campaign
    would generate before committing to the full run.

    Returns: JSON with samples — list of n_samples arrays of n_steps speeds in mm/s.
    """
    return tool_preview_doe_speeds(n_samples, speed_min, speed_max, n_steps, max_perturbation, seed)


# ---------------------------------------------------------------------------
# ML predictions
# ---------------------------------------------------------------------------

@mcp.tool()
def predict_pressure(speed_mm_s: float) -> str:
    """
    Predict catheter-tissue contact pressure for a single insertion speed.

    Orders of magnitude faster than FEM. Requires a trained model
    (run a DOE campaign first, then train the ML pipeline).

    Returns: predicted_pressure_pa.
    """
    return tool_predict_pressure(speed_mm_s)


@mcp.tool()
def predict_pressure_batch(speeds_mm_s: list[float]) -> str:
    """
    Predict contact pressures for multiple insertion speeds in one call.

    Returns: list of {speed_mm_s, predicted_pressure_pa}.
    """
    return tool_predict_pressure_batch(speeds_mm_s)


# ---------------------------------------------------------------------------
# RAG — research documents
# ---------------------------------------------------------------------------

@mcp.tool()
def list_research_documents() -> str:
    """
    Return all PDF filenames currently indexed in the research document store
    plus the total chunk count.

    Call this before searching to check what has been ingested.
    """
    return tool_list_research_documents()


@mcp.tool()
def ingest_research_documents(force: bool = False) -> str:
    """
    Scan research_documents/ for new PDFs and index them into ChromaDB.

    Already-indexed PDFs are skipped unless force=True.
    Call this after adding new PDFs, or with force=True to re-index.

    Returns: n_ingested, n_skipped, n_failed, per-file results.
    """
    return tool_ingest_research_documents(force)


@mcp.tool()
def search_research_documents(query: str, n_results: int = 5) -> str:
    """
    Semantically search indexed research documents and return relevant excerpts.

    WORKFLOW: list_research_documents() → ingest_research_documents() if empty
              → search_research_documents(query).

    After receiving results synthesise an answer citing source PDF and chunk_index.

    Args:
        query:     Natural-language question.
        n_results: Number of chunks to return (default 5, max 20).

    Returns: query, total_hits, hits (text, source, chunk_index, score).
    """
    return tool_search_research_documents(query, n_results)


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

    _sse = SseServerTransport("/messages/")
    _srv = mcp._mcp_server

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
