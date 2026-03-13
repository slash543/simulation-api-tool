#!/usr/bin/env python3
"""
setup-agent.py — Create the "Simulation Assistant" Agent in LibreChat via API.

Run this AFTER the full stack is up and you have registered a LibreChat account:

    python scripts/setup-agent.py \
        --url http://localhost:3080 \
        --username admin@example.com \
        --password your_password

The script:
  1. Logs in and obtains a JWT token
  2. Creates a "Simulation Assistant" agent pre-configured with:
       - Endpoint: Ollama (local)
       - Model:    qwen2.5:7b
       - System prompt: expert simulation assistant
       - All simulation MCP tools enabled (fetched from the server)

Requirements:  pip install httpx
"""
from __future__ import annotations

import argparse
import json
import sys

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install it with:  pip install httpx")
    sys.exit(1)


SYSTEM_PROMPT = """\
You are a digital twin simulation assistant for catheter insertion biomechanics.

CRITICAL: ALWAYS call list_catheter_designs() FIRST before any simulation.
Never assume which designs or configurations are available — new .feb files may have
been added to base_configuration/ since the last conversation. Only use design names
and configuration keys returned by list_catheter_designs().

CRITICAL: After calling list_catheter_designs(), check the "warning" field in the response.
If "warning" is not null, STOP and tell the user exactly what the warning says BEFORE listing
designs or asking for simulation parameters. Do not proceed with a simulation if a warning is present.

SPEED: check speed_range_min/max in the list_catheter_designs() response.
For a uniform speed (e.g. "15 mm/s") repeat it once per step (n_steps from response).

=== SIMULATION TOOLS ===
TO RUN A SIMULATION:
1. Call list_catheter_designs() — MANDATORY every time, even if user names a design.
2. Present the available designs and configurations to the user.
3. Ask the user which design, configuration, and speed they want (if not already stated).
4. Call run_catheter_simulation(design, configuration, speeds_mm_s).
5. Tell the user: run_id, host_run_dir (results folder), host_xplt_path (open in FEBio Studio).

NEW .FEB FILES: If the user says they added a file and it is not in the list,
call refresh_catalogue() to rescan base_configuration/ without restarting containers.

TO CHECK STATUS: call list_simulation_jobs().
TO POLL A SPECIFIC TASK: call get_task_status(task_id).
TO CANCEL: call list_simulation_jobs() to get run_id, ask user for task_id, then call cancel_simulation().

=== DOE CAMPAIGNS ===
TO RUN A DOE SWEEP: call run_doe_campaign(n_samples, speed_min, speed_max, template).
TO CHECK DOE PROGRESS: call get_doe_status(task_id).
TO PREVIEW SPEED PROFILES: call preview_doe_speeds() before committing to a full run.

=== ML PREDICTIONS ===
TO PREDICT PRESSURE (single): call predict_pressure(speed_mm_s).
TO PREDICT PRESSURE (batch):  call predict_pressure_batch(speeds_mm_s=[...]).
NOTE: ML model must be trained first (requires a completed DOE campaign).

=== RESEARCH DOCUMENTS (RAG) ===
TO LIST INDEXED DOCS: call list_research_documents().
TO INDEX NEW PDFs:    call ingest_research_documents() after adding PDFs to research_documents/.
TO SEARCH DOCS:       call search_research_documents(query). Always cite source + chunk_index.

=== CATALOGUE ===
TO LIST TEMPLATES: call list_templates().
TO REFRESH AFTER ADDING .FEB FILES: call refresh_catalogue() (no container restart needed).

=== API HEALTH ===
TO CHECK API IS UP: call health_check().

Keep responses concise. Do not make up paths or IDs.
"""


def login(client: httpx.Client, base_url: str, username: str, password: str) -> str:
    resp = client.post(
        f"{base_url}/api/auth/login",
        json={"email": username, "password": password},
    )
    if resp.status_code != 200:
        print(f"ERROR: Login failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    token = resp.json().get("token")
    if not token:
        print("ERROR: No token in login response.")
        sys.exit(1)
    print(f"  Logged in as {username}")
    return token


def get_mcp_tools(client: httpx.Client, base_url: str, token: str) -> list[str]:
    """Return the list of pluginKeys from the simulation-tools MCP server."""
    resp = client.get(
        f"{base_url}/api/mcp/tools",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        print(f"  WARNING: Could not fetch MCP tools ({resp.status_code}). Skipping tool assignment.")
        return []
    data = resp.json()
    # Response format: {"servers": {"simulation-tools": {"tools": [{pluginKey, name, ...}]}}}
    servers = data.get("servers", {}) if isinstance(data, dict) else {}
    sim_server = servers.get("simulation-tools", {})
    tools = sim_server.get("tools", [])
    plugin_keys = [t["pluginKey"] for t in tools if t.get("pluginKey")]
    print(f"  Found {len(plugin_keys)} simulation MCP tool(s).")
    return plugin_keys


def find_existing_agent(client: httpx.Client, base_url: str, token: str) -> str | None:
    """Return the agent_id of the existing 'Simulation Assistant', or None.

    LibreChat's GET /api/agents returns an SSE stream, not JSON — so we use
    the user endpoint to derive the agent URL and try a direct PATCH approach.
    Returns None if the agent is not found or the API is not accessible.
    """
    # LibreChat exposes individual agents at GET /api/agents/:id but there's
    # no JSON list endpoint in most versions.  Return None to fall through to
    # the POST path; the --update flag will use PATCH on a known ID instead.
    return None


def create_or_update_agent(
    client: httpx.Client,
    base_url: str,
    token: str,
    tool_ids: list[str],
    force_update: bool = False,
    agent_id: str | None = None,
) -> None:
    payload = {
        "name": "Simulation Assistant",
        "description": (
            "Runs catheter insertion FEM simulations. "
            "Tell it a design, size, and speed to start."
        ),
        "instructions": SYSTEM_PROMPT,
        "model": "qwen2.5:7b",
        "endpoint": "Ollama (local)",
        "tools": tool_ids,
        "model_parameters": {
            "temperature": 0.2,
            "max_tokens": 4096,
        },
    }

    if force_update and agent_id:
        resp = client.patch(
            f"{base_url}/api/agents/{agent_id}",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        if resp.status_code in (200, 201):
            print(f"  Agent updated: id={agent_id}")
        else:
            print(f"  WARNING: Update returned {resp.status_code}: {resp.text[:200]}")
        return

    resp = client.post(
        f"{base_url}/api/agents",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )

    if resp.status_code in (200, 201):
        try:
            agent = resp.json()
            print(f"  Agent created: id={agent.get('id')}  name={agent.get('name')}")
        except Exception:
            print(f"  Agent created (status {resp.status_code}; no JSON body returned).")
    elif resp.status_code == 409:
        print("  Agent already exists — use --update with --agent-id to overwrite.")
    else:
        print(f"  WARNING: Agent creation returned {resp.status_code}: {resp.text[:200]}")
        print("  You can create the agent manually in the LibreChat UI → Agents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Simulation Assistant agent in LibreChat.")
    parser.add_argument("--url", default="http://localhost:3080", help="LibreChat base URL")
    parser.add_argument("--username", required=True, help="LibreChat account email")
    parser.add_argument("--password", required=True, help="LibreChat account password")
    parser.add_argument("--update", action="store_true", help="Update existing agent's system prompt instead of creating a new one")
    parser.add_argument("--agent-id", default=None, help="Agent ID to update (required with --update, e.g. agent_t_i9u644tIs83cXDSTTph)")
    args = parser.parse_args()

    print(f"\nConnecting to LibreChat at {args.url} …")
    with httpx.Client(timeout=30) as client:
        token = login(client, args.url, args.username, args.password)
        tool_ids = get_mcp_tools(client, args.url, token)
        create_or_update_agent(client, args.url, token, tool_ids, force_update=args.update, agent_id=args.agent_id)

    print("\nDone!  Open LibreChat → Agents to find 'Simulation Assistant'.")
    print("Or create it manually in the UI and enable the simulation-tools MCP server.\n")


if __name__ == "__main__":
    main()
