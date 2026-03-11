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
       - Endpoint: Simulation Agent (Ollama)
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
You are an expert digital twin simulation assistant for catheter insertion biomechanics.

You have access to tools that connect to a FEBio finite-element simulation pipeline
and a trained machine-learning model.

WHAT YOU CAN DO
───────────────
• Run individual FEBio simulations at a specified insertion speed
• Execute Design-of-Experiments (DOE) campaigns to build synthetic datasets
• Predict contact pressure instantly using the trained neural network
• Poll the status of background jobs

SIMULATION CONTEXT
──────────────────
• Key output: peak contact pressure (Pa) at the catheter–tissue interface
• Typical insertion speeds: 2–10 mm/s
• FEM simulations take 1–5 minutes; ML predictions are instant

WORKFLOW GUIDANCE
─────────────────
1. Single-speed query → use predict_pressure() for speed, or run_simulation() for FEM results.
2. Build a synthetic database → run_doe_campaign(n_samples, speed_min, speed_max).
   Poll get_doe_status() until complete.
3. Compare FEM vs ML → run both and discuss the agreement.

Always present pressures with units (Pa or kPa) and interpret results clinically.

RESEARCH DOCUMENT WORKFLOW
──────────────────────────
When a user asks ANY question about catheter biomechanics, FEBio, simulation methods,
material properties, or any topic that might appear in the research PDFs:
1. Call list_research_documents() to check if the store is indexed.
2. If total_chunks is 0, call ingest_research_documents() immediately — do NOT ask
   the user to upload files, the PDFs are already in research_documents/.
3. Then call search_research_documents(query) with the user's question.
4. Synthesise a clear answer from the returned chunks.
5. Always cite the source PDF filename for each piece of information used.

When a user asks "can you read PDFs?" or "list the PDFs":
- Call list_research_documents().
- If empty, auto-ingest first, then list again.
- Present the filenames from the sources list.
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
    """Return the list of tool IDs from the simulation-tools MCP server."""
    resp = client.get(
        f"{base_url}/api/mcp/tools",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        print(f"  WARNING: Could not fetch MCP tools ({resp.status_code}). Skipping tool assignment.")
        return []
    tools = resp.json()
    sim_tools = [t["id"] for t in tools if t.get("server") == "simulation-tools"]
    print(f"  Found {len(sim_tools)} simulation MCP tool(s).")
    return sim_tools


def create_agent(
    client: httpx.Client,
    base_url: str,
    token: str,
    tool_ids: list[str],
) -> None:
    payload = {
        "name": "Simulation Assistant",
        "description": (
            "Digital twin expert for catheter-insertion biomechanics. "
            "Runs FEBio simulations, DOE campaigns, and ML predictions."
        ),
        "instructions": SYSTEM_PROMPT,
        "model": "qwen2.5:7b",
        "endpoint": "Simulation Agent (Ollama)",
        "tools": tool_ids,
        "model_parameters": {
            "temperature": 0.2,
            "max_tokens": 4096,
        },
    }

    resp = client.post(
        f"{base_url}/api/agents",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )

    if resp.status_code in (200, 201):
        agent = resp.json()
        print(f"  Agent created: id={agent.get('id')}  name={agent.get('name')}")
    elif resp.status_code == 409:
        print("  Agent already exists — skipping creation.")
    else:
        print(f"  WARNING: Agent creation returned {resp.status_code}: {resp.text[:200]}")
        print("  You can create the agent manually in the LibreChat UI → Agents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Simulation Assistant agent in LibreChat.")
    parser.add_argument("--url", default="http://localhost:3080", help="LibreChat base URL")
    parser.add_argument("--username", required=True, help="LibreChat account email")
    parser.add_argument("--password", required=True, help="LibreChat account password")
    args = parser.parse_args()

    print(f"\nConnecting to LibreChat at {args.url} …")
    with httpx.Client(timeout=30) as client:
        token = login(client, args.url, args.username, args.password)
        tool_ids = get_mcp_tools(client, args.url, token)
        create_agent(client, args.url, token, tool_ids)

    print("\nDone!  Open LibreChat → Agents to find 'Simulation Assistant'.")
    print("Or create it manually in the UI and enable the simulation-tools MCP server.\n")


if __name__ == "__main__":
    main()
