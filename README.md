# Digital Twin UI

A production-quality Python platform that automates catheter insertion simulations,
performs Design of Experiments (DOE), builds a machine-learning dataset, and serves
predictions through a FastAPI interface.

---

## Overview

The platform wraps a catheter-in-urethra finite element simulation.  It modifies
simulation input files to sweep insertion speed, runs the solver, extracts contact
pressure from the results, feeds the data to an MLP, and exposes everything through
a REST API and a Celery task queue.

```
API request
     │
     ▼
Task queue (Celery + Redis)
     │
     ▼
DOE sampler  ──► Simulation Configurator  ──► Solver (febio4 -i input.feb)
                                                       │
                                                       ▼
                                               Result extraction (.xplt)
                                                       │
                                                       ▼
                                               CSV / Parquet dataset
                                                       │
                                                       ├──► MLflow experiment log
                                                       │
                                                       ▼
                                               PyTorch MLP training
                                                       │
                                                       ▼
                                               FastAPI  /ml/predict
```

---

## Simulation Physics

| Parameter | Value |
|---|---|
| Solver | `febio4` (single processor) |
| Run command | `febio4 -i input.feb` |
| Analysis | 2-step static |
| Step 1 (clamping) | t = 0 → 2 s, 40 increments |
| Step 2 (insertion) | t = 2 s → `2 + D/v`, increments = `D/(v·dt)` |
| Prescribed displacement D | 10 mm per step |
| Default step size dt | 0.05 s |
| DOE speed range | 4 – 6 mm/s |
| Default speed | 5 mm/s |

Speed → time mapping:

```
speed = 4 mm/s  →  duration = 2.50 s  →  50 increments  →  LC end = 4.500
speed = 5 mm/s  →  duration = 2.00 s  →  40 increments  →  LC end = 4.000  (default)
speed = 6 mm/s  →  duration = 1.67 s  →  33 increments  →  LC end = 3.667
```

---

## Project Structure

```
simulation-api-tool/
├── digital_twin_ui/            # Main Python package
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/         # FastAPI route handlers
│   │   │   └── schemas/        # Pydantic request/response models
│   │   └── core/
│   │       ├── config.py       # Settings (YAML + env overrides)
│   │       └── logging.py      # Structured logging (loguru)
│   ├── simulation/
│   │   ├── simulation_configurator.py  # Modifies .feb for target speed
│   │   ├── simulation_runner.py        # Runs febio4, captures output
│   │   └── simulation_monitor.py       # Polls log for NORMAL TERMINATION
│   ├── extraction/
│   │   └── xplt_parser.py      # Reads .xplt, extracts contact pressure
│   ├── doe/
│   │   ├── sampler.py          # LHS / Sobol / uniform samplers
│   │   └── doe_pipeline.py     # Orchestrates DOE campaign
│   ├── experiments/
│   │   └── mlflow_manager.py   # MLflow experiment & run helpers
│   ├── ml/
│   │   ├── dataset.py          # Dataset builder (Parquet)
│   │   ├── model.py            # MLP architecture
│   │   ├── trainer.py          # Train loop with early stopping
│   │   └── inference.py        # Load model, run predictions
│   ├── tasks/
│   │   ├── celery_app.py       # Celery application factory
│   │   └── simulation_tasks.py # Async simulation & training tasks
│   ├── services/
│   │   ├── simulation_service.py
│   │   ├── dataset_service.py
│   │   └── training_service.py
│   └── utils/
├── tests/
│   ├── conftest.py                         # Shared fixtures
│   ├── test_config.py                      # Configuration tests
│   ├── test_logging.py                     # Logging tests
│   └── test_simulation_configurator.py     # Simulation configurator tests
├── config/
│   └── simulation.yaml         # All tunable parameters
├── templates/
│   └── sample_catheterization.feb          # Base simulation template
├── runs/                        # Auto-created per-run directories
├── data/
│   ├── raw/                     # Per-run CSV extractions
│   └── datasets/                # Merged Parquet dataset
├── models/                      # Saved PyTorch checkpoints
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Add the package to the Python path

```bash
echo "$PWD" > .venv/lib/python3.12/site-packages/digital_twin_ui.pth
```

### 4. Verify installation

```bash
python -c "import digital_twin_ui; print(digital_twin_ui.__version__)"
# → 0.1.0
```

---

## Configuration

All parameters live in `config/simulation.yaml`.

```yaml
simulation:
  simulator_executable: "febio4"   # command name on PATH
  simulator_args: ["-i"]           # assembled as: febio4 -i input.feb
  displacement_mm: 10.0
  default_step_size: 0.05
  loadcurve_start_time: 2.0

doe:
  speed_min_mm_s: 4.0
  speed_max_mm_s: 6.0
  default_sampler: "lhs"           # lhs | sobol | uniform
  default_num_samples: 10

mlflow:
  tracking_uri: "mlruns"
  experiment_name: "catheter_insertion"

ml:
  hidden_dims: [64, 128, 256]
  learning_rate: 0.001
  max_epochs: 500
  patience: 20
```

### Environment variable overrides

Any YAML value can be overridden without editing the file:

```bash
DTUI__SIMULATION__SIMULATOR_EXECUTABLE=febio4-avx2  # use different binary
DTUI__API__PORT=9000                                 # change API port
DTUI__ML__LEARNING_RATE=0.0005
```

Pattern: `DTUI__<SECTION>__<FIELD>=<value>`

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only (skip slow integration tests)
pytest tests/ -v -m "not integration"

# Integration tests only (require real template)
pytest tests/ -v -m integration

# With coverage report
pytest tests/ --cov=digital_twin_ui --cov-report=term-missing
```

### Test organisation

| File | What it covers |
|---|---|
| `tests/conftest.py` | Shared fixtures: `project_root`, `template_feb`, `minimal_feb`, `patch_env`, `clean_settings_cache`, `reset_logging_state` |
| `tests/test_config.py` | All `Settings` sub-models, derived paths, YAML loading, env overrides, singleton cache (52 tests) |
| `tests/test_logging.py` | JSON serialiser, file sinks, `_configured` flag lifecycle, `get_logger` interface (22 tests) |
| `tests/test_simulation_configurator.py` | Physics formulas, XML helpers, XML modification, content preservation, error paths, idempotency (62 tests) |

### Test markers

| Marker | Usage |
|---|---|
| `integration` | Tests using the full 17 000-line simulation template. Run slower but validate the real file. |

---

## Usage Examples

### Modify a simulation file for a target speed

```python
from pathlib import Path
from digital_twin_ui.simulation.simulation_configurator import configure_simulation

result = configure_simulation(
    speed_mm_s=4.5,
    output_path=Path("runs/run_0001/input.feb"),
)

print(f"LC end time : {result.lc_end_time:.4f} s")
print(f"Time steps  : {result.time_steps_step2}")
print(f"Output      : {result.output_path}")
```

```
LC end time : 4.2222 s
Time steps  : 44
Output      : runs/run_0001/input.feb
```

### Load configuration

```python
from digital_twin_ui.app.core.config import get_settings

cfg = get_settings()
print(cfg.simulation.simulator_executable)  # febio4
print(cfg.simulation.simulator_args)        # ['-i']
print(cfg.doe.speed_min_mm_s)               # 4.0
```

### Initialise logging

```python
from digital_twin_ui.app.core.logging import configure_from_settings, get_logger

configure_from_settings()   # call once at startup

logger = get_logger(__name__)
logger.info("Platform ready")

with logger.contextualize(run_id="run_0001", speed=4.5):
    logger.debug("Simulation step started")
```

---

## Implementation Progress

| Step | Module | Status |
|---|---|---|
| 1 | Project structure | Done |
| 2 | Configuration system | Done |
| 3 | Logging | Done |
| 3 | Simulation Configurator | Done |
| 4 | Simulation Runner | Done |
| 5 | Simulation Monitor | Done |
| 6 | DOE Sampler | Done |
| 7 | Result Extraction | Done |
| 8 | MLflow Integration | Done |
| 9 | Task Queue (Celery) | Done |
| 10 | FastAPI Endpoints | Done |
| 11 | Dataset Builder | Done |
| 12 | PyTorch Training | Done |
| 13 | Full Test Suite | Ongoing |
| 13b | Per-facet tracking pipeline | Done |
| 14 | Docker | Done |
| 15 | LibreChat + Ollama + MCP Integration | Done |

---

## Step 15 — LibreChat + Ollama + MCP Integration

### Architecture

```
User  ──►  LibreChat UI (port 3080)
                │
                │  OpenAI-compat API (Ollama qwen2.5:7b)
                ▼
           Ollama (port 11434)
                │
                │  tool calls via MCP (SSE)
                ▼
         MCP Server (port 8001)   ← mcp_server/server.py
                │
                │  HTTP calls
                ▼
         FastAPI (port 8000)  ← existing simulation API
                │
        ┌───────┴────────┐
        Celery Worker   MLflow (port 5000)
        (FEBio sims)    (experiment tracking)
```

The **MCP server** is the portability layer — any MCP-compatible client
(LibreChat, Claude Desktop, VS Code Copilot Chat, …) can connect to it at
`http://<host>:8001/sse` without changes to the simulation back-end.

### Quick Start

```bash
# 1. Create LibreChat secrets
cp .env.librechat.example .env.librechat

# Fill in the four secrets (each command prints the value to use):
openssl rand -hex 32   # → JWT_SECRET
openssl rand -hex 32   # → JWT_REFRESH_SECRET
openssl rand -hex 32   # → CREDS_KEY
openssl rand -hex 16   # → CREDS_IV
openssl rand -hex 24   # → MEILI_MASTER_KEY

# 2. Start the full stack (first run pulls qwen2.5:7b — ~4.7 GB)
docker compose -f docker-compose.librechat.yml up --build

# 3. Open http://localhost:3080 and register an account

# 4. Create the Simulation Assistant agent (optional — can be done in UI)
python scripts/setup-agent.py \
    --url http://localhost:3080 \
    --username you@example.com \
    --password your_password
```

### LLM Choices (Ollama)

| Model | RAM | Quality | Tool calling |
|---|---|---|---|
| `qwen2.5:7b` | 4.7 GB | ★★★★ | Excellent (default) |
| `qwen2.5:14b` | 9 GB | ★★★★★ | Excellent |
| `llama3.1:8b` | 4.7 GB | ★★★★ | Very good |
| `qwen2.5:3b` | 2 GB | ★★★ | Good |

To switch, edit the model name in `librechat.yaml` and in the
`ollama-init` command in `docker-compose.librechat.yml`.

### Creating the Agent in the LibreChat UI

1. Click **New Agent** (left sidebar → Agent icon)
2. Name: `Simulation Assistant`
3. Endpoint: `Simulation Agent (Ollama)` · Model: `qwen2.5:7b`
4. Paste the system prompt from `librechat.yaml` → `modelSpecs.list[0].preset.system`
5. Under **Tools** → enable all tools from `simulation-tools`
6. Save and start chatting

### Example Conversations

> *"Run a simulation at 5 mm/s and tell me the peak contact pressure."*

> *"Generate a DOE database with 20 samples between 2 and 10 mm/s using Latin Hypercube sampling."*

> *"Predict the pressure at 3, 5, 7, and 9 mm/s — which speed is safest?"*

### Connecting a Different Chat Interface

The MCP server exposes a standard SSE endpoint at `http://<host>:8001/sse`.
Add it to any MCP-compatible client:

| Client | Config location |
|---|---|
| Claude Desktop | `claude_desktop_config.json` → `mcpServers` |
| VS Code (GitHub Copilot) | `.vscode/mcp.json` |
| Any LibreChat instance | `librechat.yaml` → `mcpServers` |

---

## Requirements

- Python 3.11+
- `febio4` on `$PATH`
- Redis (for Celery task queue)
- CUDA-capable GPU (optional — PyTorch falls back to CPU)

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE).

All dependencies are free for commercial use (MIT, Apache 2.0, BSD-3-Clause, PSF).
See [`LICENSES.md`](LICENSES.md) for the full audit, including version-lock rationale
for Redis (stay on v7, not v8) and Meilisearch (stay on v1.7.3, not v1.19+).

> Research prototype. Not for clinical use.
