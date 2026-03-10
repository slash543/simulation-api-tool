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
    tool_get_doe_status,
    tool_get_task_status,
    tool_health_check,
    tool_predict_pressure,
    tool_predict_pressure_batch,
    tool_run_doe_campaign,
    tool_run_simulation,
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
        "  • run FEBio finite-element simulations at specific insertion speeds\n"
        "  • run DOE campaigns to build a synthetic simulation database\n"
        "  • predict contact pressure instantly using the trained ML model\n"
        "  • poll async task results\n\n"
        "Typical insertion speed range: 2–10 mm/s. "
        "Use run_simulation() for accurate FEM results — it fires in the background "
        "and returns immediately with host_run_dir and host_xplt_path. "
        "ALWAYS show these paths to the user after calling run_simulation(). "
        "Use predict_pressure() for instant ML estimates (requires prior training). "
        "For building a database, prefer run_doe_campaign() with 10–50 samples."
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
def run_doe_campaign(
    n_samples: int,
    speed_min: float,
    speed_max: float,
    sampler: str = "lhs",
    seed: int | None = None,
) -> str:
    """
    Submit a Design of Experiments campaign to build a synthetic simulation database.

    Runs `n_samples` simulations sampled across [speed_min, speed_max] using the
    chosen strategy.  Results are stored for subsequent ML training.

    Args:
        n_samples: Number of simulation points (recommended: 10 – 50).
        speed_min: Minimum insertion speed in mm/s (e.g. 2.0).
        speed_max: Maximum insertion speed in mm/s (e.g. 10.0).
        sampler:   Sampling strategy — 'lhs' (Latin Hypercube, default),
                   'sobol', or 'uniform'.
        seed:      Optional integer seed for reproducibility.

    Returns:
        JSON with task_id.  Poll with get_doe_status(task_id).
    """
    return tool_run_doe_campaign(n_samples, speed_min, speed_max, sampler, seed)


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
