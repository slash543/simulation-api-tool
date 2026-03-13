"""
MCP Server — Digital Twin Simulation Tools

Transport: SSE (Server-Sent Events)
Default:   http://0.0.0.0:8001/sse
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools import (
    tool_cancel_simulation,
    tool_list_simulation_jobs,
    tool_run_catheter_simulation,
)

mcp = FastMCP(
    "digital-twin-simulation",
    instructions=(
        "You run catheter insertion FEM simulations.\n\n"
        "AVAILABLE DESIGNS (hardcoded — no lookup needed):\n"
        "  ball_tip:          14Fr_IR12, 14Fr_IR25, 16Fr_IR12\n"
        "  nelaton_tip:       14Fr_IR12, 14Fr_IR25, 16Fr_IR12\n"
        "  vapro_introducer:  14Fr_IR12, 16Fr_IR12\n\n"
        "SPEED: 10–25 mm/s. All designs have 10 steps.\n"
        "For uniform speed repeat the value 10 times (e.g. 15 mm/s → [15]*10).\n\n"
        "TO RUN: ask design + config + speed, then call run_catheter_simulation().\n"
        "AFTER SUBMIT: tell user host_run_dir and host_xplt_path.\n"
        "TO CHECK STATUS: call list_simulation_jobs().\n"
        "TO CANCEL: get run_id from list_simulation_jobs(), ask user for task_id, "
        "then call cancel_simulation()."
    ),
)


@mcp.tool()
def run_catheter_simulation(
    design: str,
    configuration: str,
    speeds_mm_s: list[float],
    dwell_time_s: float = 1.0,
) -> str:
    """
    Submit a catheter FEM simulation.
    design: ball_tip | nelaton_tip | vapro_introducer
    configuration: 14Fr_IR12 | 14Fr_IR25 | 16Fr_IR12
    speeds_mm_s: exactly 10 float values in mm/s
    Returns: task_id, run_id, host_run_dir, host_xplt_path
    """
    return tool_run_catheter_simulation(design, configuration, speeds_mm_s, dwell_time_s)


@mcp.tool()
def list_simulation_jobs() -> str:
    """
    List recent simulation runs (newest first).
    Returns: run_id, status, host_run_dir, host_xplt_path, xplt_exists.
    """
    return tool_list_simulation_jobs()


@mcp.tool()
def cancel_simulation(task_id: str, run_id: str) -> str:
    """
    Cancel a running simulation.
    task_id: from submission response. run_id: from list_simulation_jobs().
    """
    return tool_cancel_simulation(task_id, run_id)


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
