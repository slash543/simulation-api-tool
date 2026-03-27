#!/usr/bin/env python3
"""
setup-csar-agent.py — Create or update the "Coating Coverage Analyst" Agent in LibreChat.

This agent specialises in CSAR (Contact Surface Area Ratio) analysis for catheter
coating coverage.  It picks up the trained surrogate model from the MLflow registry
automatically — including models trained on another VM.

Usage:
    python scripts/setup-csar-agent.py --url http://localhost:3080 \\
        --username you@example.com --password yourpassword

    # To list existing agents (diagnostic):
    python scripts/setup-csar-agent.py ... --list

    # To force-update a specific agent by ID:
    python scripts/setup-csar-agent.py ... --agent-id <id>

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


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Urethral Catheter Coverage Profiler — a specialist in characterising \
how a catheter's surface contacts the urethral wall as it is inserted, and \
therefore how effectively a surface coating (drug-eluting, hydrophilic, \
antimicrobial) is delivered across each axial region of the catheter.

PHYSICAL CONTEXT:
  The surrogate model was trained on a quasi-static FEM simulation of catheter \
insertion into a hyperelastic urethra model. Because the urethra is hyperelastic \
(rate-independent), insertion speed does not change the contact state — only \
insertion depth determines how the urethra deforms and where contact occurs. \
This means the surrogate is a depth-to-coverage profiler for one specific \
catheter-urethra geometry pair. Do not compare across speeds or conditions; \
there are none. All variation is along the insertion depth axis.

YOUR PRIMARY METRIC: CSAR (Contact Surface Area Ratio)
  CSAR(depth, band) = catheter facets in band with contact pressure > 0
                      ─────────────────────────────────────────────────
                      total catheter facets in that band

  CSAR = 0.0 → no contact — coating cannot be delivered at this depth/band
  CSAR = 1.0 → full contact — coating covers the entire band surface

Coverage interpretation guide:
  < 10 %  → no meaningful coating delivery
  10–30 % → partial delivery — early-contact or distal-only bands
  30–60 % → moderate delivery — typical mid-insertion values
  60–80 % → good delivery — effective for most coating types
  > 80 %  → excellent delivery — near-complete wall contact

THE THREE CLINICAL QUESTIONS THIS AGENT ANSWERS:
  1. THRESHOLD — At what insertion depth does coating delivery begin for each band?
     → first_contact_depth_mm per z-band
  2. OPTIMUM   — At what depth is coverage maximised? Does it plateau?
     → depth_at_peak_csar_mm; inspect whether the CSAR curve flattens
  3. REGIONAL  — Which catheter region contacts first / most?
     → compare first_contact_depth_mm and peak_csar across bands

CATHETER Z-BANDS (Z = 0 at the catheter tip, increasing proximally):
  Z-bands select CATHETER SURFACE FACES whose centroid Z-coordinate falls within
  [zmin, zmax]. They are axial divisions of the catheter geometry — not urethra
  anatomy. The surrogate model predicts contact pressure on catheter faces, and
  CSAR is the fraction of those faces (within the band) that have cp > 0.

  Typical catheter region breakdown:
    [{"zmin":   0, "zmax":  50, "label": "distal_tip"},
     {"zmin":  50, "zmax": 150, "label": "mid_shaft"},
     {"zmin": 150, "zmax": 300, "label": "proximal"}]
  If the user does not specify regions, use the WHOLE catheter as one band:
    [{"zmin": 0, "zmax": 9999, "label": "whole_catheter"}]
  If the user gives custom mm values, use them exactly. If the catheter length is
  unknown, call list_available_vtps() to find the VTP and infer the length.

DEPTH GRID:
  Use depth_step_mm=2 by default (not the default 5). Because depth is the only \
axis, finer resolution captures the exact onset of contact and shape of the \
plateau more accurately. Use depth_step_mm=1 if the user wants high-resolution \
curves.

WORKFLOW:
  STEP 0 — call list_surrogate_models_mcp_simulation_tools
    • latest_available=true  → proceed, no extra argument needed
    • registered_models non-empty → pass registered_model_name=<name> to all calls
    • neither → ask user to train via notebooks/full_pipeline.ipynb

  STEP 1 — find the VTP file
    • User does not know path → call list_available_vtps_mcp_simulation_tools
    • User provides path → use it directly

  STEP 2 — clarify catheter regions
    • No regions specified → whole_catheter default
    • User names a region (e.g. "tip", "shaft") → map to mm ranges; confirm if uncertain

  STEP 3 — run the depth profile
    Call analyse_catheter_contact_mcp_simulation_tools(
        vtp_path, z_bands,
        depth_step_mm=2,
        registered_model_name=<if needed>
    )

  STEP 4 — answer the three clinical questions
    For each band report:
      • "Coating first contacts the <band> region at <first_contact_depth_mm> mm insertion."
      • "Maximum coverage in <band> is <peak_csar as %>% reached at <depth_at_peak_csar_mm> mm."
      • If peak_csar < 30% → flag this band as under-covered for the clinical use case.
      • If CSAR plateaus before full insertion → state the effective coverage depth range.
    Tell the user: "Open <host_plot_path> to view the coverage vs depth profile."

  STEP 5 (optional) — 3-D snapshot at a specific depth
    Call predict_vtp_contact_pressure_mcp_simulation_tools(
        vtp_path, insertion_depth_mm, registered_model_name=<if needed>
    )
    Open the output VTP in ParaView to visualise which facets are in contact.

INTERPRETATION RULES:
  • Always state coverage as a percentage (e.g. "62% coverage"), never as a decimal.
  • Refer to insertion depth in millimetres.
  • Do not discuss speed — it is irrelevant for this hyperelastic model.
  • Do not compare across different catheters or anatomies unless the user \
explicitly provides multiple VTP files from different designs.
  • If the CSAR curve does not reach >10% anywhere, tell the user the catheter \
does not make meaningful contact with the urethra in that zone at any depth.

Do not invent file paths or model names. Keep answers concise and clinical.\
"""

# ---------------------------------------------------------------------------
# Auth helpers (shared with setup-agent.py)
# ---------------------------------------------------------------------------

def _headers(token: str, base_url: str) -> dict:
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
# MCP tool discovery
# ---------------------------------------------------------------------------

def get_mcp_tools(
    client: httpx.Client, base_url: str, token: str, verbose: bool = False
) -> list[str]:
    """Return pluginKeys from the simulation-tools MCP server."""
    resp = client.get(f"{base_url}/api/mcp/tools", headers=_headers(token, base_url))
    if verbose:
        print(f"  GET /api/mcp/tools → {resp.status_code}")
        print(f"  Response: {resp.text[:600]}")
    if resp.status_code != 200:
        print(f"  WARNING: Could not fetch MCP tools ({resp.status_code}). "
              "Agent will be created without tools — re-run once the MCP server is healthy.")
        return []

    data = resp.json()
    servers = data.get("servers", {}) if isinstance(data, dict) else {}
    sim_server = servers.get("simulation_tools", {})
    tools = sim_server.get("tools", [])
    plugin_keys = [t["pluginKey"] for t in tools if t.get("pluginKey")]
    print(f"  Found {len(plugin_keys)} simulation MCP tool(s).")
    return plugin_keys


# ---------------------------------------------------------------------------
# Agent listing (diagnostic)
# ---------------------------------------------------------------------------

def list_agents(
    client: httpx.Client, base_url: str, token: str, verbose: bool = False
) -> list[dict]:
    resp = client.get(
        f"{base_url}/api/agents",
        headers=_headers(token, base_url),
        params={"limit": 100},
    )
    if verbose:
        print(f"  GET /api/agents → {resp.status_code}")
        print(f"  Response: {resp.text[:1000]}")
    if resp.status_code != 200:
        return []

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = resp.json()
            return data if isinstance(data, list) else data.get("data", [])
        except Exception:
            pass

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
    client: httpx.Client, base_url: str, token: str,
    agent_name: str = "Coating Coverage Analyst",
    verbose: bool = False,
) -> str | None:
    agents = list_agents(client, base_url, token, verbose=verbose)
    for agent in agents:
        if agent.get("name") == agent_name:
            agent_id = agent.get("id") or agent.get("_id")
            if agent_id:
                print(f"  Found existing '{agent_name}' agent: id={agent_id}")
                return str(agent_id)
    return None


# ---------------------------------------------------------------------------
# Agent payload
# ---------------------------------------------------------------------------

# Only attach surrogate + listing tools — keeps context lean for qwen2.5.
CSAR_TOOL_NAMES = {
    "list_surrogate_models",
    "list_available_vtps",
    "analyse_catheter_contact",
    "generate_csar_plot_from_vtp",
    "compute_csar_from_vtp",
    "compute_csar_vs_depth",
    "evaluate_contact_pressure",
    "predict_vtp_contact_pressure",
    "health_check",
}


def _build_payload(tool_ids: list[str]) -> dict[str, Any]:
    mcp_sys = "sys__server__sys_mcp_simulation_tools"
    csar_tools = [
        t for t in tool_ids
        if t != mcp_sys and any(t.startswith(name) for name in CSAR_TOOL_NAMES)
    ]
    all_tools = ([mcp_sys] + csar_tools) if tool_ids else []
    return {
        "name": "Coating Coverage Analyst",
        "description": (
            "Analyses catheter coating coverage using CSAR (Contact Surface Area Ratio). "
            "Picks up surrogate models from MLflow registry automatically — "
            "works across VMs. Ask it about coverage %, contact depth, or pressure."
        ),
        "instructions": SYSTEM_PROMPT,
        "provider": "Ollama (local)",
        "endpoint": "Ollama (local)",
        "model": "qwen2.5:7b",
        "tools": all_tools,
        "mcpServerNames": ["simulation_tools"],
        "model_parameters": {
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "artifacts": "",
        "tool_kwargs": [],
        "agent_ids": [],
        "edges": [],
        "conversation_starters": [
            "At what insertion depth does coating delivery begin?",
            "What is the optimal insertion depth for maximum urethral coverage?",
            "Compare coverage across the penile, membranous, and prostatic zones.",
            "Does the catheter coating ever reach the prostatic urethra?",
        ],
        "projectIds": [],
        "category": "general",
        "support_contact": {"name": "", "email": ""},
        "is_promoted": False,
        "end_after_tools": False,
        "hide_sequential_outputs": False,
        "tool_options": {},
    }


# ---------------------------------------------------------------------------
# Agent creation / update
# ---------------------------------------------------------------------------

def _verify_agent(
    client: httpx.Client, base_url: str, token: str, agent_id: str, verbose: bool
) -> bool:
    resp = client.get(
        f"{base_url}/api/agents/{agent_id}",
        headers=_headers(token, base_url),
    )
    if verbose:
        print(f"  GET /api/agents/{agent_id} → {resp.status_code}")
    return resp.status_code == 200


def create_or_update_agent(
    client: httpx.Client,
    base_url: str,
    token: str,
    tool_ids: list[str],
    agent_id: str | None = None,
    verbose: bool = False,
) -> None:
    payload = _build_payload(tool_ids)

    if not agent_id:
        agent_id = find_existing_agent(client, base_url, token, verbose=verbose)

    if agent_id:
        if verbose:
            print(f"  PATCH /api/agents/{agent_id}")
        resp = client.patch(
            f"{base_url}/api/agents/{agent_id}",
            headers=_headers(token, base_url),
            json=payload,
        )
        if verbose:
            print(f"  Response ({resp.status_code}): {resp.text[:400]}")
        if resp.status_code in (200, 201):
            print(f"  Agent updated successfully: id={agent_id}")
            _verify_agent(client, base_url, token, agent_id, verbose=verbose)
            return
        print(f"  WARNING: Update returned {resp.status_code}. Falling back to POST …")
        agent_id = None

    if not agent_id:
        if verbose:
            print("  POST /api/agents")
        resp = client.post(
            f"{base_url}/api/agents",
            headers=_headers(token, base_url),
            json=payload,
        )
        if verbose:
            print(f"  Response ({resp.status_code}): {resp.text[:600]}")
        if resp.status_code in (200, 201):
            try:
                created = resp.json()
                agent_id = created.get("id") or created.get("_id", "unknown")
                print(f"  Agent created: id={agent_id}  name={created.get('name')}")
            except Exception:
                print(f"  Agent created (status {resp.status_code}).")
                return
        elif resp.status_code == 409:
            print("  Agent already exists. Re-run to update it.")
            return
        else:
            print(f"\n  ERROR: Agent creation returned {resp.status_code}:")
            print(f"  {resp.text[:500]}")
            print("\n  MANUAL FALLBACK:")
            print("    1. Open LibreChat → http://localhost:3080")
            print("    2. Left sidebar → Agents → '+ New Agent'")
            print("    3. Name: 'Coating Coverage Analyst'")
            print("    4. Model: Ollama (local) → qwen2.5:7b")
            print("    5. Paste the SYSTEM_PROMPT from this script")
            print("    6. Enable surrogate tools from the 'simulation-tools' MCP server")
            return

    if agent_id and agent_id != "unknown":
        ok = _verify_agent(client, base_url, token, agent_id, verbose=verbose)
        if ok:
            print(f"\n  Verified: agent is accessible at /api/agents/{agent_id}")
        else:
            print(f"\n  WARNING: Created but GET /api/agents/{agent_id} failed.")
            print("  Try refreshing the Agents page in LibreChat.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or update the Coating Coverage Analyst agent in LibreChat."
    )
    parser.add_argument("--url", default="http://localhost:3080", help="LibreChat base URL")
    parser.add_argument("--username", required=True, help="LibreChat account email")
    parser.add_argument("--password", required=True, help="LibreChat account password")
    parser.add_argument("--agent-id", default=None,
                        help="Explicitly target this agent ID for update")
    parser.add_argument("--list", action="store_true",
                        help="List all agents and exit (diagnostic)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full HTTP response bodies")
    args = parser.parse_args()

    print(f"\nConnecting to LibreChat at {args.url} …")
    with httpx.Client(timeout=30) as client:
        token = login(client, args.url, args.username, args.password)

        if args.list:
            agents = list_agents(client, args.url, token, verbose=args.verbose)
            if not agents:
                print("\n  No agents found.")
            else:
                print(f"\n  {len(agents)} agent(s):")
                for a in agents:
                    aid = a.get("id") or a.get("_id", "?")
                    print(f"    id={aid}  name={a.get('name')}  model={a.get('model')}")
            return

        tool_ids = get_mcp_tools(client, args.url, token, verbose=args.verbose)
        create_or_update_agent(
            client, args.url, token, tool_ids,
            agent_id=args.agent_id,
            verbose=args.verbose,
        )

    print("\nDone!")
    print("\nHow to use the Coating Coverage Analyst in LibreChat:")
    print("  1. Open http://localhost:3080")
    print("  2. Left sidebar → Agents → 'Coating Coverage Analyst'")
    print("  3. Start a conversation — the agent picks up models from MLflow registry")
    print("\n  Try: 'What is the coating coverage for my latest simulation?'")
    print("  Or:  'Show CSAR vs depth for the distal tip and mid-shaft'\n")


if __name__ == "__main__":
    main()
