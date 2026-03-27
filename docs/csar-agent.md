# Urethral Catheter Coverage Profiler — CSAR Agent

A specialist LibreChat agent for characterising how a catheter's surface contacts the urethral wall as it is inserted, and how effective coating delivery (drug-eluting, hydrophilic, antimicrobial, etc.) will be across each axial region of the catheter at each insertion depth.

---

## Physical Context

The surrogate model was trained on a **quasi-static FEM simulation** of catheter insertion into a **hyperelastic urethra model**.

> **Key principle**: Hyperelastic materials are **rate-independent** — the deformation state (and therefore the contact pressure field) at a given insertion depth is identical regardless of how quickly the catheter was advanced.
>
> **Insertion depth** is the one meaningful variable.
> **Insertion speed** is physically irrelevant and is never discussed.

This means a single FEM simulation sweeping through a range of insertion depths provides all the training data needed for the surrogate. The surrogate then evaluates any depth in milliseconds — no re-simulation required.

---

## What is CSAR?

**Contact Surface Area Ratio** is the primary metric for coating coverage:

```
CSAR = faces with contact pressure > 0
       ────────────────────────────────
       total catheter surface faces in the region
```

| CSAR | Coverage | Interpretation |
|---|---|---|
| 0.0 | 0 % | No contact — coating cannot be delivered |
| < 0.1 | < 10 % | Very low — coating barely contacts the wall |
| 0.1 – 0.3 | 10 – 30 % | Low — may be insufficient for drug delivery |
| 0.3 – 0.6 | 30 – 60 % | Moderate — typical for standard catheter designs |
| 0.6 – 0.8 | 60 – 80 % | Good — effective coating contact |
| > 0.8 | > 80 % | Excellent — near-full surface contact |
| 1.0 | 100 % | Full contact |

CSAR is computed using the **surrogate model**, which predicts per-facet contact pressure at any insertion depth in milliseconds — no FEM simulation required at query time.

---

## Why CSAR Matters for Coating

For any surface-active catheter coating:

- **Drug-eluting catheters**: coating only transfers where the surface contacts the wall. CSAR directly predicts the fraction of drug that reaches tissue.
- **Hydrophilic coatings**: friction reduction only occurs at contact zones. CSAR tells you how much of the catheter benefits from lubrication.
- **Antimicrobial coatings**: coverage fraction determines the effective antimicrobial area.

---

## The Three Clinical Questions

The agent is designed to answer three specific questions about coating delivery:

| # | Question | Key metric |
|---|---|---|
| 1 | **Threshold** — At what depth does coating delivery first begin? | `first_contact_depth_mm` |
| 2 | **Optimum** — At what depth is coverage maximised? Does it plateau? | `depth_at_peak_csar_mm`, `peak_csar` |
| 3 | **Regional** — How does coverage differ across urethral zones? | per-band CSAR comparison |

---

## Z-Band Analysis

Z-bands select **catheter surface faces** by the Z-coordinate of their centroid. They are axial divisions of the **catheter geometry** — not urethra anatomy labels.

- **Z = 0**: distal tip of the catheter (first to enter)
- **Z increases**: toward the handle (proximal end)

The surrogate model predicts contact pressure on catheter faces. For a given band, CSAR counts the fraction of faces in that band with `cp > 0`.

```
Handle ◄────────────────────────────────────────► Tip
          proximal          mid-shaft         distal
        [150 → 300 mm]    [50 → 150 mm]     [0 → 50 mm]
          Z = 150–300       Z = 50–150         Z = 0–50
```

### Default behaviour (whole catheter)

If no Z-bands are specified, the agent analyses the **entire catheter as one region**:

```json
[{"zmin": 0, "zmax": 9999, "label": "whole_catheter"}]
```

### Custom Z-bands

| User request | Agent constructs |
|---|---|
| "whole catheter" | `[{zmin:0, zmax:9999}]` |
| "tip only" | `[{zmin:0, zmax:50}]` |
| "tip and shaft" | `[{zmin:0,zmax:50}, {zmin:50,zmax:150}]` |
| "3 regions" | `[{zmin:0,zmax:50}, {zmin:50,zmax:150}, {zmin:150,zmax:300}]` |
| Custom mm values | Exactly as specified |

Z-band format:

```json
[
  {"zmin": 0.0,   "zmax": 50.0,  "label": "distal_tip"},
  {"zmin": 50.0,  "zmax": 150.0, "label": "mid_shaft"},
  {"zmin": 150.0, "zmax": 300.0, "label": "proximal"}
]
```

---

## Default Analysis Parameters

| Parameter | Default | Reason |
|---|---|---|
| `depth_step_mm` | **2** | Finer resolution: depth is the only axis of variation |
| `max_depth_mm` | 300 | Covers full catheter insertion range |
| `z_bands` | whole catheter | Conservative default; user can narrow |

Always use `depth_step_mm=2` (not the API default of 5) to capture sharp transitions in the coverage curve.

---

## What the Agent Produces

### Two-panel PNG plot

Every `analyse_catheter_contact` call produces a two-panel figure:

```
┌─────────────────────────────────────────────────┐
│  TOP:  CSAR (%) vs Insertion Depth               │
│  ──── penile  ──── membranous  ──── prostatic    │
│                                                   │
│  0.8 │       ╭─────────────────                  │
│  0.6 │   ╭───╯                                   │
│  0.4 │───╯                                       │
│  0.0 └────────────────────────────────────────   │
│        0      100      200      300   depth [mm] │
├─────────────────────────────────────────────────┤
│  BOTTOM: Peak Contact Pressure [MPa] vs Depth    │
│  (same bands, separate curves)                   │
└─────────────────────────────────────────────────┘
```

The PNG is saved locally and the host path is returned so you can open it directly.

### Band summaries (per Z-band statistics)

| Field | Meaning |
|---|---|
| `first_contact_depth_mm` | Shallowest insertion depth where any contact occurs (CSAR > 0) |
| `peak_csar` | Maximum CSAR across all depths (0.0 – 1.0) |
| `depth_at_peak_csar_mm` | Insertion depth that achieves peak coverage |
| `peak_pressure_MPa` | Maximum predicted contact pressure |
| `depth_at_peak_pressure_mm` | Insertion depth that produces peak pressure |
| `total_area_mm2` | Total surface area of the band |
| `n_total_facets` | Number of mesh facets in the band |

---

## Conversation Examples

### Whole-catheter coverage

```
You:   What is the coating coverage for my latest simulation?

Agent: Calls list_surrogate_models()    → model ready
       Calls list_available_vtps()      → finds latest VTP
       Calls analyse_catheter_contact(
               vtp_path="...catheter_t0000.vtp",
               z_bands=[{zmin:0, zmax:9999, label:"whole_catheter"}],
               depth_step_mm=2
             )

       "The catheter achieves 58% contact coverage (CSAR = 0.58) at
        maximum insertion. Coating delivery first begins at 32 mm
        insertion depth, and peak coverage is reached at 185 mm.
        Open ./data/surrogate/results/plots/whole_catheter_csar.png
        to view the coverage curve."
```

### Three-region analysis

```
You:   Compare coating coverage for the tip, mid-shaft, and proximal regions.

Agent: Calls analyse_catheter_contact(
               vtp_path=...,
               z_bands=[
                 {zmin:0,   zmax:50,  label:"distal_tip"},
                 {zmin:50,  zmax:150, label:"mid_shaft"},
                 {zmin:150, zmax:300, label:"proximal"}
               ],
               depth_step_mm=2
             )

       "Distal tip: 81% peak coverage at 95 mm insertion.
        First contact at 18 mm.

        Mid-shaft: 51% peak coverage at 160 mm insertion.
        First contact at 72 mm.

        Proximal region: 23% peak coverage at 240 mm insertion.
        First contact at 140 mm — coating barely reaches this region.

        Recommendation: the tip and mid-shaft are well served;
        the proximal region may require deeper insertion to achieve
        meaningful coating delivery."
```

### Threshold question

```
You:   At what insertion depth does coating delivery begin?

Agent: Calls list_surrogate_models() → model ready
       Calls list_available_vtps()   → latest VTP
       Calls analyse_catheter_contact(
               vtp_path=...,
               z_bands=[{zmin:0, zmax:9999, label:"whole_catheter"}],
               depth_step_mm=2
             )

       "Coating delivery begins at 32 mm insertion depth —
        the point where any part of the catheter surface first
        contacts the urethral wall (CSAR transitions from 0 to > 0)."
```

### 3D pressure map at a specific depth

```
You:   Show me the 3D pressure map at 150 mm insertion depth.

Agent: Calls predict_vtp_contact_pressure(
               vtp_path=...,
               insertion_depth_mm=150
             )

       "Annotated VTP saved to:
        ./data/surrogate/results/catheter_t0000_predicted.vtp
        Open this file in ParaView to see contact pressure
        colour-mapped on the catheter surface."
```

---

## Cross-VM Model Loading (MLflow Registry)

The agent supports models trained on a **different VM** through the MLflow Model Registry. No manual file copying is needed.

### Workflow

**On the training VM:**

```python
# After running full_pipeline.ipynb, register the model:
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
run_id = "<the run_id from your training>"
mlflow.register_model(f"runs:/{run_id}/.", "CatheterCSARSurrogate")

# Optionally promote to Production stage:
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="CatheterCSARSurrogate",
    version="1",
    stage="Production"
)
```

**On the inference VM:**

The agent detects the registered model automatically:

```
You:   What is the coating coverage for my simulation?

Agent: Calls list_surrogate_models()
       → sees: registered_models: [{name: "CatheterCSARSurrogate", stage: "Production"}]
       → automatically passes registered_model_name="CatheterCSARSurrogate" to all calls
       → downloads artifacts from MLflow on first use, caches locally
       → proceeds with analysis using the registry-trained model
```

### Manual override

```
You:   Use the model named "CatheterCSARSurrogate" for this analysis.

Agent: Calls analyse_catheter_contact(
               vtp_path=...,
               z_bands=...,
               registered_model_name="CatheterCSARSurrogate",
               depth_step_mm=2
             )
```

### Model loading priority

When a prediction is requested, the system resolves the model in this order:

1. `registered_model_name` (explicit registry lookup) — **cross-VM**
2. `run_id` (specific MLflow run artifacts) — **advanced use**
3. `SURROGATE_REGISTRY_MODEL_NAME` env var (auto-fallback) — **deployment default**
4. `data/surrogate/models/latest/` (locally deployed artifacts) — **standard**

Set the env var in `.env` to always use a specific registered model:

```ini
# .env
SURROGATE_REGISTRY_MODEL_NAME=CatheterCSARSurrogate
```

---

## Setting Up the Agent

### Prerequisites

The full stack must be running:

```bash
docker compose -f docker-compose.librechat.yml up -d
```

A surrogate model must be available — either locally trained or registered in MLflow (see [Cross-VM Model Loading](#cross-vm-model-loading-mlflow-registry) above).

### Create the agent

```bash
python scripts/setup-csar-agent.py \
  --url http://localhost:3080 \
  --username you@example.com \
  --password yourpassword
```

This creates (or updates) the **"Urethral Catheter Coverage Profiler"** agent in LibreChat with:

- Only the 8 surrogate analysis tools attached (lean context window)
- Temperature 0.1 for deterministic, reproducible analysis
- Pre-loaded conversation starters focused on the three clinical questions

Re-run any time after pulling new code to update the agent.

### Verify the agent

Open **http://localhost:3080 → Agents → Urethral Catheter Coverage Profiler**

Try a starter conversation:

```
"At what insertion depth does coating delivery begin?"
```

### Diagnostic commands

```bash
# List all existing agents
python scripts/setup-csar-agent.py --url http://localhost:3080 \
  --username you@example.com --password yourpassword --list

# Update a specific agent by ID
python scripts/setup-csar-agent.py ... --agent-id <agent_id>
```

---

## Available Tools

The agent has access to these MCP tools. All accept an optional `registered_model_name` parameter for cross-VM registry loading.

| Tool | Description |
|---|---|
| `list_surrogate_models` | Check model availability; list MLflow registry models. **Always called first.** |
| `list_available_vtps` | Discover VTP files in `runs/` and `surrogate_data/`. Called when the user doesn't know their VTP path. |
| `analyse_catheter_contact` | **Primary tool.** Generates 2-panel CSAR + pressure plot with per-band summaries. |
| `generate_csar_plot_from_vtp` | CSAR-only plot (no pressure panel). |
| `compute_csar_from_vtp` | Raw CSAR data without generating a plot. |
| `compute_csar_vs_depth` | CSAR from reference_facets.csv (not VTP-specific geometry). |
| `evaluate_contact_pressure` | Quick mean/max pressure summary at given depths. |
| `predict_vtp_contact_pressure` | Annotate a VTP mesh with predicted pressures → open in ParaView. |

---

## REST API Endpoints

The same functionality is available directly without the agent:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/surrogate/models` | List MLflow runs and registered models |
| `POST` | `/api/v1/surrogate/analyse-from-vtp` | **Primary endpoint** — 2-panel plot + band summaries |
| `POST` | `/api/v1/surrogate/csar-plot-from-vtp` | CSAR-only plot |
| `POST` | `/api/v1/surrogate/csar-from-vtp` | CSAR data from VTP geometry |
| `POST` | `/api/v1/surrogate/csar` | CSAR from reference_facets.csv |
| `POST` | `/api/v1/surrogate/predict-vtp` | Annotate VTP with predicted pressures |
| `POST` | `/api/v1/surrogate/predict` | Batch facet-level pressure prediction |
| `GET` | `/api/v1/surrogate/list-vtps` | Discover available VTP files |

All `POST` endpoints accept `registered_model_name` and `model_stage` fields in the request body.

### Example: analyse-from-vtp (three anatomical zones)

```bash
curl -s -X POST http://localhost:8000/api/v1/surrogate/analyse-from-vtp \
  -H "Content-Type: application/json" \
  -d '{
    "vtp_path": "/app/runs/run_20240315_120000_abcd/results_vtp/catheter_t0000.vtp",
    "z_bands": [
      {"zmin": 0,   "zmax": 80,  "label": "penile"},
      {"zmin": 80,  "zmax": 100, "label": "membranous"},
      {"zmin": 100, "zmax": 150, "label": "prostatic"}
    ],
    "max_depth_mm": 300,
    "depth_step_mm": 2
  }' | python -m json.tool
```

To use a registry model:

```bash
curl -s -X POST http://localhost:8000/api/v1/surrogate/analyse-from-vtp \
  -H "Content-Type: application/json" \
  -d '{
    "vtp_path": "/app/runs/.../catheter_t0000.vtp",
    "z_bands": [{"zmin": 0, "zmax": 9999, "label": "whole_catheter"}],
    "depth_step_mm": 2,
    "registered_model_name": "CatheterCSARSurrogate",
    "model_stage": "Production"
  }'
```

---

## Data Flow

```
VTP file (mesh geometry)
    │
    │  centroid_x/y/z, facet_area  ──► SurrogatePredictor
    │                                        │
    │                              predict_at_depth(depth_mm)
    │                                        │
    ├── per-facet contact_pressure_MPa ◄─────┘
    │
    │  for each depth in [0 → max_depth_mm] at 2 mm steps:
    │    for each Z-band:
    │      CSAR = count(cp > threshold) / total_facets_in_band
    │      peak_cp = max(cp_in_band)
    │
    ▼
  CSAR DataFrame  ──►  2-panel PNG plot
                  ──►  band_summaries (peak_csar, first_contact_depth, etc.)
```

---

## Troubleshooting

**"Surrogate model not available"**

- Local: run `notebooks/full_pipeline.ipynb` to train and deploy to `data/surrogate/models/latest/`
- Cross-VM: register the model (`mlflow.register_model(...)`) and pass `registered_model_name` in the request

**"No registered models found"**

- Check MLflow is reachable: `curl http://localhost:5000/health`
- Verify the training VM registered the model: open `http://localhost:5000` → Models tab

**"VTP file not found"**

- Call `list_available_vtps()` to see what files exist
- VTP files are in `runs/<run_id>/results_vtp/` (from simulations) or `data/surrogate/results/` (from xplt export)

**Agent gives no response or hangs**

- Check the MCP server is healthy: `curl http://localhost:8001/sse`
- Check the API is healthy: `curl http://localhost:8000/api/v1/health`

**Registry model download is slow**

- First call downloads artifacts from MLflow — subsequent calls use the local cache in `data/surrogate/models/registry__<name>__<version>/`
- Set `SURROGATE_REGISTRY_MODEL_NAME` in `.env` so the cache is populated at startup

---

> Research prototype. Not for clinical use.
