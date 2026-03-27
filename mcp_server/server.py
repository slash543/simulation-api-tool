"""
MCP Server — Digital Twin Simulation Tools

Transport: SSE (Server-Sent Events)
Default:   http://0.0.0.0:8001/sse
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools import (
    tool_analyse_catheter_contact,
    tool_cancel_simulation,
    tool_compute_csar_from_vtp,
    tool_compute_csar_vs_depth,
    tool_evaluate_contact_pressure,
    tool_generate_csar_plot_from_vtp,
    tool_get_doe_status,
    tool_get_task_status,
    tool_health_check,
    tool_ingest_research_documents,
    tool_list_available_vtps,
    tool_list_catheter_designs,
    tool_list_research_documents,
    tool_list_simulation_jobs,
    tool_list_surrogate_models,
    tool_list_templates,
    tool_predict_pressure,
    tool_predict_pressure_batch,
    tool_predict_vtp_contact_pressure,
    tool_preview_doe_speeds,
    tool_refresh_catalogue,
    tool_run_catheter_simulation,
    tool_run_doe_campaign,
    tool_search_research_documents,
)

mcp = FastMCP(
    "digital-twin-simulation",
    instructions=(
        "You are the Urethral Catheter Coverage Profiler — a specialist in\n"
        "characterising how a catheter's surface contacts the urethral wall as it\n"
        "is inserted, and how effectively a surface coating is delivered across\n"
        "each axial region of the catheter.\n\n"

        "═══ PHYSICAL CONTEXT (READ THIS FIRST) ═══\n"
        "The surrogate model was trained on a quasi-static FEM simulation of\n"
        "catheter insertion into a hyperelastic urethra. Hyperelastic materials\n"
        "are RATE-INDEPENDENT: the contact state at a given depth is identical\n"
        "regardless of insertion speed. Therefore:\n"
        "  • Insertion DEPTH is the one meaningful variable.\n"
        "  • Insertion SPEED is irrelevant — never discuss it.\n"
        "  • The model profiles one specific catheter-urethra geometry pair.\n"
        "  • Do NOT compare across speeds, conditions, or multiple anatomies\n"
        "    unless the user provides separate VTP files for each.\n\n"

        "═══ YOUR PRIMARY METRIC: CSAR ═══\n"
        "CSAR(depth, band) = catheter facets in band with contact pressure > 0\n"
        "                    ─────────────────────────────────────────────────\n"
        "                    total catheter facets in that band\n\n"
        "Z-bands select CATHETER SURFACE FACES by the Z-coordinate of their\n"
        "centroid (Z=0 = distal tip, increasing toward the handle). The surrogate\n"
        "predicts contact pressure on catheter faces; CSAR is the fraction of\n"
        "faces within a band that have cp > 0 at a given insertion depth.\n\n"
        "  0%%   → no coating delivery (no contact in this catheter region)\n"
        "  100%% → full coverage of that catheter region\n"
        "  < 10%%  → no meaningful delivery\n"
        "  10–30%% → partial — early contact or distal-only\n"
        "  30–60%% → moderate — typical mid-insertion\n"
        "  60–80%% → good — effective for most coating types\n"
        "  > 80%%  → excellent — near-complete wall contact\n\n"

        "═══ THE THREE CLINICAL QUESTIONS ═══\n"
        "Every analysis must answer these three questions per band:\n"
        "  1. THRESHOLD  At what depth does coating delivery begin?\n"
        "                → first_contact_depth_mm\n"
        "  2. OPTIMUM    At what depth is coverage maximised? Does it plateau?\n"
        "                → depth_at_peak_csar_mm; note if CSAR flattens before\n"
        "                   full insertion (plateau = further insertion adds nothing)\n"
        "  3. REGIONAL   Which catheter region contacts first / most?\n"
        "                → compare first_contact_depth_mm and peak_csar across bands\n\n"

        "═══ CATHETER Z-BANDS ═══\n"
        "Z = 0 at the catheter tip (first to enter); increases proximally.\n"
        "Z-bands are divisions of CATHETER GEOMETRY, not urethra anatomy.\n"
        "Typical catheter regions:\n"
        "  distal tip:  zmin=0,   zmax=50,  label='distal_tip'\n"
        "  mid-shaft:   zmin=50,  zmax=150, label='mid_shaft'\n"
        "  proximal:    zmin=150, zmax=300, label='proximal'\n"
        "DEFAULT (no regions specified):\n"
        "  [{\"zmin\": 0, \"zmax\": 9999, \"label\": \"whole_catheter\"}]\n"
        "Use custom mm values exactly as the user provides them.\n\n"

        "═══ DEPTH GRID ═══\n"
        "Always use depth_step_mm=2 (not the default 5).\n"
        "Depth is the only axis — finer steps capture the exact contact onset\n"
        "and plateau shape. Use depth_step_mm=1 for high-resolution curves.\n\n"

        "═══ WORKFLOW ═══\n"
        "STEP 0 — list_surrogate_models()\n"
        "  latest_available=true  → proceed\n"
        "  registered_models non-empty → pass registered_model_name=<name> to all calls\n"
        "  neither → ask user to run full_pipeline.ipynb\n"
        "STEP 1 — find VTP: list_available_vtps() or use user-provided path\n"
        "STEP 2 — clarify bands: default whole_catheter; use user-specified mm ranges\n"
        "STEP 3 — analyse_catheter_contact(vtp_path, z_bands,\n"
        "                                  depth_step_mm=2,\n"
        "                                  registered_model_name=<if needed>)\n"
        "STEP 4 — answer the three clinical questions per zone:\n"
        "  'Coating first contacts the <band> region at <depth> mm insertion.'\n"
        "  'Maximum coverage in <band> is <%%> reached at <depth> mm.'\n"
        "  Flag bands where peak_csar < 30%% as under-covered.\n"
        "  Note if the CSAR curve plateaus (further insertion gains nothing).\n"
        "  Tell user: Open <host_plot_path> to view the coverage profile.\n"
        "STEP 5 (optional) — 3-D snapshot:\n"
        "  predict_vtp_contact_pressure(vtp_path, depth) → open in ParaView\n\n"

        "═══ TOOLS ═══\n"
        "  list_surrogate_models()           — ★ ALWAYS FIRST\n"
        "  list_available_vtps()             — discover VTP files\n"
        "  analyse_catheter_contact(...)     — ★ PRIMARY: depth profile + plot\n"
        "  generate_csar_plot_from_vtp(...)  — CSAR-only plot\n"
        "  compute_csar_from_vtp(...)        — raw CSAR data (no plot)\n"
        "  evaluate_contact_pressure(...)    — mean/max pressure summary\n"
        "  predict_vtp_contact_pressure(...) — annotated VTP for ParaView\n\n"
        "  FEM SIMULATIONS: list_catheter_designs() → run_catheter_simulation()\n"
        "  RESEARCH DOCS: list_research_documents() → search_research_documents()\n\n"

        "Always state CSAR as a percentage. Never mention speed.\n"
        "VTP paths: container paths under /app/surrogate_data/ or /app/runs/.\n"
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
# Surrogate model tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_surrogate_models() -> str:
    """
    List trained surrogate models from MLflow.

    ★ ALWAYS call this FIRST before any surrogate/CSAR tool.

    Returns:
      - latest_available: True if a local model is ready
      - registered_models: models from the MLflow Registry (may be trained on another VM)
      - recent_runs: last 5 training run IDs and metrics

    If registered_models is non-empty, pass registered_model_name=<name> to all
    prediction calls so the system downloads and uses that cross-VM model.
    """
    return tool_list_surrogate_models()


@mcp.tool()
def evaluate_contact_pressure(
    insertion_depths_mm: list[float],
    facets_csv_path: str | None = None,
    run_id: str | None = None,
    registered_model_name: str | None = None,
) -> str:
    """
    Compute mean/max contact pressure (MPa) at given insertion depths.

    Args:
        insertion_depths_mm: Depths [mm] to evaluate.  Example: [0,50,100,200,300]
        facets_csv_path: Container path to reference facets CSV (optional).
        run_id: Specific MLflow run. None = latest model.
        registered_model_name: Registry model name for cross-VM loading.

    Returns: insertion_depths_mm, mean_cp_MPa, max_cp_MPa, csar per depth.
    """
    return tool_evaluate_contact_pressure(
        insertion_depths_mm, facets_csv_path, run_id, registered_model_name
    )


@mcp.tool()
def compute_csar_vs_depth(
    z_bands: list[dict],
    insertion_depths_mm: list[float] | None = None,
    max_depth_mm: float = 300.0,
    depth_step_mm: float = 5.0,
    facets_csv_path: str | None = None,
    run_id: str | None = None,
    registered_model_name: str | None = None,
) -> str:
    """
    Compute CSAR vs insertion depth for given Z bands (no plot, raw data).

    CSAR = fraction of catheter surface in contact with tissue at each depth.

    Args:
        z_bands: [{"zmin": 0.0, "zmax": 50.0, "label": "tip"}, ...]
        insertion_depths_mm: Specific depths [mm]. None = auto grid.
        max_depth_mm: Auto-grid upper bound (default 300).
        depth_step_mm: Auto-grid step (default 5 mm).
        facets_csv_path: Container path to facets CSV (optional).
        run_id: MLflow run ID. None = latest model.
        registered_model_name: Registry model name for cross-VM loading.

    Returns: per-band CSAR series with csar, contact_area_mm2, n_contact_facets.
    """
    return tool_compute_csar_vs_depth(
        z_bands, insertion_depths_mm, max_depth_mm, depth_step_mm,
        facets_csv_path, run_id, registered_model_name
    )


@mcp.tool()
def predict_vtp_contact_pressure(
    vtp_path: str,
    insertion_depth_mm: float,
    output_path: str | None = None,
    run_id: str | None = None,
    registered_model_name: str | None = None,
) -> str:
    """
    Annotate a VTP mesh with predicted contact pressures at a given insertion depth.

    Outputs a new VTP file with contact_pressure_MPa cell data — open in ParaView
    for a 3D colour-mapped view of which areas are under pressure.

    Args:
        vtp_path: Container path to input VTP file.
        insertion_depth_mm: Catheter insertion depth [mm].
        output_path: Optional output path (defaults to input_stem + '_predicted.vtp').
        run_id: MLflow run ID. None = latest model.
        registered_model_name: Registry model name for cross-VM loading.

    Returns: output_vtp_path, host_output_path, n_faces.
    """
    return tool_predict_vtp_contact_pressure(
        vtp_path, insertion_depth_mm, output_path, run_id, registered_model_name
    )


@mcp.tool()
def compute_csar_from_vtp(
    vtp_path: str,
    z_bands: list[dict],
    insertion_depths_mm: list[float] | None = None,
    max_depth_mm: float = 300.0,
    depth_step_mm: float = 5.0,
    run_id: str | None = None,
    registered_model_name: str | None = None,
) -> str:
    """
    Compute CSAR vs insertion depth from a VTP file (raw data, no plot).

    Args:
        vtp_path: Container path to VTP file.
        z_bands: [{"zmin": 0.0, "zmax": 50.0, "label": "tip"}, ...]
        insertion_depths_mm: Specific depths [mm]. None = auto grid.
        max_depth_mm: Auto-grid upper bound (default 300 mm).
        depth_step_mm: Auto-grid step (default 5 mm).
        run_id: MLflow run ID. None = latest model.
        registered_model_name: Registry model name for cross-VM loading.

    Returns: per-band CSAR series vs insertion depth.
    """
    return tool_compute_csar_from_vtp(
        vtp_path, z_bands, insertion_depths_mm, max_depth_mm, depth_step_mm,
        run_id, registered_model_name
    )


@mcp.tool()
def list_available_vtps(max_files: int = 30) -> str:
    """
    List VTP files available in runs/ and surrogate_data/ directories.

    Call this when the user doesn't know the VTP path, or before calling
    analyse_catheter_contact().

    Returns: list of VTP files with host_path, stem, size_kb (newest first).
    """
    return tool_list_available_vtps(max_files)


@mcp.tool()
def analyse_catheter_contact(
    vtp_path: str,
    z_bands: list[dict],
    insertion_depths_mm: list[float] | None = None,
    max_depth_mm: float = 300.0,
    depth_step_mm: float = 5.0,
    run_id: str | None = None,
    registered_model_name: str | None = None,
    title: str | None = None,
) -> str:
    """
    ★ PRIMARY COATING COVERAGE ANALYSIS TOOL.

    Generates a two-panel plot:
      TOP:    CSAR (coating coverage fraction) vs insertion depth per Z band
      BOTTOM: Peak contact pressure [MPa] vs insertion depth per Z band

    No mesh knowledge needed — just specify which Z regions you care about.
    DEFAULT: If user doesn't specify bands, use whole catheter:
             [{"zmin": 0, "zmax": 9999, "label": "whole_catheter"}]

    Args:
        vtp_path: Container path to the VTP file (use list_available_vtps() to find).
        z_bands: Axial catheter regions. Z=0 is the distal tip (mm from tip).
                 Default whole-catheter: [{"zmin": 0, "zmax": 9999, "label": "whole_catheter"}]
                 3-zone example:
                   [{"zmin":   0, "zmax":  50, "label": "distal_tip"},
                    {"zmin":  50, "zmax": 150, "label": "mid_shaft"},
                    {"zmin": 150, "zmax": 300, "label": "proximal"}]
        insertion_depths_mm: Depths [mm] to evaluate. None = auto grid (recommended).
        max_depth_mm: Auto-grid upper bound (default 300 mm).
        depth_step_mm: Auto-grid resolution (default 5 mm).
        run_id: MLflow run ID. None = latest model.
        registered_model_name: Registry model name for cross-VM loading.
                                Get from list_surrogate_models() → registered_models[].name
        title: Optional plot title.

    Returns:
        host_plot_path: Open this PNG to view the coverage plots.
        band_summaries: peak_csar, depth_at_peak_csar_mm, peak_pressure_MPa,
                        first_contact_depth_mm per band.
    """
    return tool_analyse_catheter_contact(
        vtp_path, z_bands, insertion_depths_mm, max_depth_mm, depth_step_mm,
        run_id, registered_model_name, title
    )


@mcp.tool()
def generate_csar_plot_from_vtp(
    vtp_path: str,
    z_bands: list[dict],
    insertion_depths_mm: list[float] | None = None,
    max_depth_mm: float = 300.0,
    depth_step_mm: float = 5.0,
    run_id: str | None = None,
    registered_model_name: str | None = None,
    title: str | None = None,
    output_path: str | None = None,
) -> str:
    """
    Compute CSAR vs insertion depth from a VTP file and generate a CSAR-only PNG plot.

    Use this for a CSAR curve plot without the pressure panel.
    For the full combined plot (CSAR + pressure), use analyse_catheter_contact().

    Args:
        vtp_path: Container path to the VTP file.
        z_bands: Z-band definitions. Each becomes a separate curve in the plot.
        insertion_depths_mm: Specific depths [mm] to evaluate. None = auto grid.
        max_depth_mm: Auto-grid upper bound (default 300 mm).
        depth_step_mm: Auto-grid step size (default 5 mm).
        run_id: MLflow run ID. None = latest model.
        registered_model_name: Registry model name for cross-VM loading.
        title: Optional plot title.
        output_path: Optional container path for the PNG output.

    Returns: host_plot_path (open this PNG), bands_summary with peak CSAR values.
    """
    return tool_generate_csar_plot_from_vtp(
        vtp_path, z_bands, insertion_depths_mm, max_depth_mm, depth_step_mm,
        run_id, registered_model_name, title, output_path
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
