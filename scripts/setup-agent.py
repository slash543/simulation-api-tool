#!/usr/bin/env python3
"""
setup-agent.py — Create or update the "Simulation Assistant" Agent in LibreChat.

Run via the wrapper script (handles waiting + credential lookup):
    bash scripts/create-agent.sh

Or directly:
    python scripts/setup-agent.py --url http://localhost:3080 \
        --username you@example.com --password yourpassword

Flags:
    --list            List all agents visible to this user account (diagnostic)
    --verbose         Print full HTTP response bodies
    --agent-id <id>   Force update of a specific agent ID

Requirements:  pip install httpx
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install it with:  pip install httpx")
    sys.exit(1)


SYSTEM_PROMPT = """\
You are a digital twin simulation assistant for catheter insertion biomechanics. \
Always call list_catheter_designs_mcp_simulation_tools first to see what designs \
and configurations are available before running anything. If the response contains \
a "warning" field, tell the user that warning before proceeding. \
To run a simulation call run_catheter_simulation_mcp_simulation_tools with the \
design name, configuration key, and a list of speeds in mm/s (one per step; \
use n_steps from the list response). Report back the run_id, host_run_dir, and \
host_xplt_path. To check jobs call list_simulation_jobs_mcp_simulation_tools or \
get_task_status_mcp_simulation_tools. To cancel call cancel_simulation_mcp_simulation_tools. \
If the user adds a new .feb file call refresh_catalogue_mcp_simulation_tools. \
Keep replies concise; do not invent paths or IDs.
"""

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _headers(token: str, base_url: str) -> dict:
    """Build headers that pass LibreChat's origin/CSRF checks."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Origin": base_url,
        "Referer": f"{base_url}/",
    }


def login(client: httpx.Client, base_url: str, username: str, password: str) -> str:
    resp = client.post(
        f"{base_url}/api/auth/login",
        headers={"Origin": base_url, "Referer": f"{base_url}/"},
        json={"email": username, "password": password},
    )
    if resp.status_code != 200:
        print(f"ERROR: Login failed ({resp.status_code}): {resp.text[:300]}")
        sys.exit(1)
    token = resp.json().get("token")
    if not token:
        print("ERROR: No token in login response.")
        sys.exit(1)
    print(f"  Logged in as {username}")
    return token


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

def get_mcp_tools(
    client: httpx.Client, base_url: str, token: str, verbose: bool = False
) -> list[str]:
    """Return pluginKeys from the simulation-tools MCP server."""
    resp = client.get(
        f"{base_url}/api/mcp/tools",
        headers=_headers(token, base_url),
    )
    if verbose:
        print(f"  GET /api/mcp/tools → {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")

    if resp.status_code != 200:
        print(f"  WARNING: Could not fetch MCP tools ({resp.status_code}). "
              "Agent will be created without tools — re-run the script once "
              "the MCP server is healthy.")
        return []

    data = resp.json()
    servers = data.get("servers", {}) if isinstance(data, dict) else {}
    sim_server = servers.get("simulation_tools", {})
    tools = sim_server.get("tools", [])
    plugin_keys = [t["pluginKey"] for t in tools if t.get("pluginKey")]
    print(f"  Found {len(plugin_keys)} simulation MCP tool(s).")
    if not plugin_keys:
        print("  WARNING: 0 tools found. The MCP server may still be starting up.")
        print("  The agent will be created anyway — re-run this script in ~30 s")
        print("  to attach tools:  bash scripts/create-agent.sh")
    return plugin_keys


# ---------------------------------------------------------------------------
# Agent listing (diagnostic)
# ---------------------------------------------------------------------------

def list_agents(
    client: httpx.Client, base_url: str, token: str, verbose: bool = False
) -> list[dict]:
    """Return all agents visible to this user.

    LibreChat's GET /api/agents returns an SSE stream in some versions.
    We parse SSE data lines manually; fall back to an empty list on any error.
    """
    resp = client.get(
        f"{base_url}/api/agents",
        headers=_headers(token, base_url),
        params={"limit": 100},
    )
    if verbose:
        print(f"  GET /api/agents → {resp.status_code}")
        print(f"  Response: {resp.text[:1000]}")

    if resp.status_code != 200:
        print(f"  NOTE: GET /api/agents returned {resp.status_code} — skipping duplicate check.")
        return []

    # Try plain JSON first
    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = resp.json()
            return data if isinstance(data, list) else data.get("data", [])
        except Exception:
            pass

    # Parse SSE stream: look for "data: {...}" lines that are valid JSON arrays/objects
    agents: list[dict] = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
            if isinstance(obj, list):
                agents.extend(obj)
            elif isinstance(obj, dict) and obj.get("name"):
                agents.append(obj)
        except json.JSONDecodeError:
            pass

    return agents


def find_existing_agent(
    client: httpx.Client, base_url: str, token: str, verbose: bool = False
) -> str | None:
    """Return the agent_id of 'Simulation Assistant' if it already exists."""
    agents = list_agents(client, base_url, token, verbose=verbose)
    for agent in agents:
        if agent.get("name") == "Simulation Assistant":
            agent_id = agent.get("id") or agent.get("_id")
            if agent_id:
                print(f"  Found existing 'Simulation Assistant' agent: id={agent_id}")
                return str(agent_id)
    return None


# ---------------------------------------------------------------------------
# Agent creation / update
# ---------------------------------------------------------------------------

CORE_TOOL_NAMES = {
    "list_catheter_designs",
    "run_catheter_simulation",
    "list_simulation_jobs",
    "get_task_status",
    "cancel_simulation",
    "refresh_catalogue",
}


def _build_payload(tool_ids: list[str]) -> dict[str, Any]:
    # LibreChat stores both "provider" (the display-name key from librechat.yaml)
    # and "endpoint" (same value) on the agent document.  "mcpServerNames" is
    # required for the MCP tools sidebar to render in the UI.
    # "sys__server__sys_mcp_simulation-tools" must be the first tool entry so
    # LibreChat recognises this as an MCP-enabled agent.
    mcp_sys = "sys__server__sys_mcp_simulation_tools"
    # Only attach core simulation tools — keeps context window lean for qwen2.5.
    core_tools = [
        t for t in tool_ids
        if t != mcp_sys and any(t.startswith(name) for name in CORE_TOOL_NAMES)
    ]
    all_tools = ([mcp_sys] + core_tools) if tool_ids else []
    return {
        "name": "Simulation Assistant",
        "description": (
            "Runs catheter insertion FEM simulations. "
            "Tell it a design, size, and speed to start."
        ),
        "instructions": SYSTEM_PROMPT,
        "provider": "Ollama (local)",
        "endpoint": "Ollama (local)",
        "model": "qwen2.5:7b",
        "tools": all_tools,
        "mcpServerNames": ["simulation_tools"],
        "model_parameters": {
            "temperature": 0.2,
            "max_tokens": 4096,
        },
        "artifacts": "",
        "tool_kwargs": [],
        "agent_ids": [],
        "edges": [],
        "conversation_starters": [],
        "projectIds": [],
        "category": "general",
        "support_contact": {"name": "", "email": ""},
        "is_promoted": False,
        "end_after_tools": False,
        "hide_sequential_outputs": False,
        "tool_options": {},
    }


def _verify_agent(
    client: httpx.Client, base_url: str, token: str, agent_id: str, verbose: bool
) -> bool:
    """Confirm the agent is fetchable after creation/update."""
    resp = client.get(
        f"{base_url}/api/agents/{agent_id}",
        headers=_headers(token, base_url),
    )
    if verbose:
        print(f"  GET /api/agents/{agent_id} → {resp.status_code}")
        print(f"  Response: {resp.text[:400]}")
    return resp.status_code == 200


def create_or_update_agent(
    client: httpx.Client,
    base_url: str,
    token: str,
    tool_ids: list[str],
    force_update: bool = False,
    agent_id: str | None = None,
    verbose: bool = False,
) -> None:
    payload = _build_payload(tool_ids)

    # Auto-detect an existing agent by name if no ID was explicitly supplied.
    if not agent_id:
        agent_id = find_existing_agent(client, base_url, token, verbose=verbose)

    if agent_id:
        if verbose:
            print(f"  PATCH /api/agents/{agent_id}")
            print(f"  Payload keys: {list(payload.keys())}")

        resp = client.patch(
            f"{base_url}/api/agents/{agent_id}",
            headers=_headers(token, base_url),
            json=payload,
        )
        if verbose:
            print(f"  Response ({resp.status_code}): {resp.text[:400]}")

        if resp.status_code in (200, 201):
            print(f"  Agent updated successfully: id={agent_id}")
        else:
            print(f"  WARNING: Update returned {resp.status_code}: {resp.text[:300]}")
            print("  Falling back to creating a new agent...")
            agent_id = None  # fall through to POST

    if not agent_id:
        if verbose:
            print("  POST /api/agents")
            print(f"  Payload: {json.dumps({k: v for k, v in payload.items() if k != 'instructions'}, indent=2)}")

        resp = client.post(
            f"{base_url}/api/agents",
            headers=_headers(token, base_url),
            json=payload,
        )
        if verbose:
            print(f"  Response ({resp.status_code}): {resp.text[:600]}")

        if resp.status_code in (200, 201):
            try:
                agent = resp.json()
                agent_id = agent.get("id") or agent.get("_id", "unknown")
                print(f"  Agent created: id={agent_id}  name={agent.get('name')}")
            except Exception:
                print(f"  Agent created (status {resp.status_code}; could not parse response).")
                return
        elif resp.status_code == 409:
            print("  Agent already exists (conflict). Re-run this script to find and update it.")
            return
        else:
            print(f"\n  ERROR: Agent creation returned {resp.status_code}:")
            print(f"  {resp.text[:500]}")
            print("\n  MANUAL FALLBACK:")
            print("    1. Open LibreChat → http://localhost:3080")
            print("    2. Left sidebar → Agents → '+ New Agent'")
            print("    3. Name: 'Simulation Assistant'")
            print("    4. Model: Ollama (local) → qwen2.5:7b")
            print("    5. Enable ALL tools from 'simulation-tools' MCP server")
            return

    if agent_id and agent_id != "unknown":
        ok = _verify_agent(client, base_url, token, agent_id, verbose=verbose)
        if ok:
            print(f"\n  Verified: agent is accessible at /api/agents/{agent_id}")
        else:
            print(f"\n  WARNING: Agent was created/updated but GET /api/agents/{agent_id} failed.")
            print("  Try refreshing the Agents page in LibreChat.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or update the Simulation Assistant agent in LibreChat."
    )
    parser.add_argument("--url", default="http://localhost:3080", help="LibreChat base URL")
    parser.add_argument("--username", required=True, help="LibreChat account email")
    parser.add_argument("--password", required=True, help="LibreChat account password")
    parser.add_argument("--update", action="store_true",
                        help="(deprecated) Force update — now happens automatically")
    parser.add_argument("--agent-id", default=None,
                        help="Explicitly target this agent ID for update")
    parser.add_argument("--list", action="store_true",
                        help="List all agents visible to this account and exit (diagnostic)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full HTTP request/response bodies")
    args = parser.parse_args()

    print(f"\nConnecting to LibreChat at {args.url} …")
    with httpx.Client(timeout=30) as client:
        token = login(client, args.url, args.username, args.password)

        if args.list:
            agents = list_agents(client, args.url, token, verbose=args.verbose)
            if not agents:
                print("\n  No agents found for this account.")
            else:
                print(f"\n  {len(agents)} agent(s) found:")
                for a in agents:
                    aid = a.get("id") or a.get("_id", "?")
                    print(f"    id={aid}  name={a.get('name')}  "
                          f"model={a.get('model')}  endpoint={a.get('endpoint')}")
            return

        tool_ids = get_mcp_tools(client, args.url, token, verbose=args.verbose)
        create_or_update_agent(
            client, args.url, token, tool_ids,
            force_update=args.update,
            agent_id=args.agent_id,
            verbose=args.verbose,
        )

    print("\nDone!")
    print("\nHow to find the agent in LibreChat:")
    print("  1. Open http://localhost:3080")
    print("  2. In the LEFT SIDEBAR click the robot icon or 'Agents'")
    print("  3. Look for 'Simulation Assistant' in the list")
    print("  4. Click it — then click 'New Conversation' to start chatting")
    print("\n  OR: click '+ New Chat' → in the model selector → switch tab to 'Agents'")
    print("      → select 'Simulation Assistant'\n")


if __name__ == "__main__":
    main()
