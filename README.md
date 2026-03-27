# Digital Twin UI

An integrated platform for catheter insertion FEM simulations (FEBio) combined with a surrogate neural-network model for fast contact-pressure and coating coverage prediction — all accessible through a conversational AI interface.

Drop in FEB files, chat with the agent, submit simulation jobs, train a surrogate model from results, and query contact-pressure and CSAR (coating coverage) profiles — no scripting required.

**Two specialist agents are included:**
- **Simulation Assistant** — runs FEM jobs, manages results, runs DOE campaigns
- **Coating Coverage Analyst** — CSAR analysis, z-band coverage, cross-VM registry model loading

---

## Features

| Feature | Description |
|---|---|
| **Conversational simulation** | Submit, monitor, and cancel FEBio jobs through natural language (LibreChat + Ollama or Azure OpenAI) |
| **Auto-discovery of .feb files** | Drop a `.feb` file into `base_configuration/` and the agent finds it automatically |
| **Per-step speed control** | Each of the 10 insertion steps can have a different speed; load-curve timings updated automatically |
| **Background execution** | Simulations run in the background via Celery; the chat session ends immediately after submission |
| **Job management** | List running/completed/cancelled jobs, get result file paths, cancel a specific simulation |
| **DOE campaigns** | Latin-Hypercube or random speed sweeps over many designs, tracked in MLflow |
| **Surrogate model inference** | Predict contact pressure from facet geometry without running FEBio — instant results |
| **CSAR — coating coverage** | Compute Contact Surface Area Ratio curves for user-defined axial Z-bands; default covers whole catheter |
| **CSAR plot generation** | Two-panel PNG output: CSAR (%) and peak contact pressure vs insertion depth per Z-band |
| **VTP annotation** | Upload a VTP file; get back a new VTP with per-facet contact pressure predicted by the surrogate |
| **Cross-VM model registry** | Register a model in MLflow from any VM; all agents pick it up automatically via `registered_model_name` |
| **JupyterLab training environment** | Full training pipeline in `notebooks/full_pipeline.ipynb`; retrain with new data using `notebooks/retraining.ipynb` |
| **ML pressure prediction (DOE)** | Classic PyTorch surrogate trained on DOE speed sweeps for pressure prediction |
| **Research document RAG** | Index PDFs from `research_documents/` and query them semantically from the same chat interface |
| **REST API** | All capabilities exposed as documented FastAPI endpoints at `/api/v1/` |
| **Portable** | Clone on any VM, add FEB files, run `setup.sh`, and the full stack is ready |

---

## Prerequisites

| Requirement | How to check |
|---|---|
| Docker + Docker Compose v2 | `docker compose version` |
| FEBio 4 | Install from [febio.org/downloads](https://febio.org/downloads/) |
| `openssl` | `openssl version` (pre-installed on most Linux/macOS) |
| ~12 GB free disk | Ollama model (4.7 GB) + Docker images + run outputs |

> FEBio is proprietary and is **not bundled**. `setup.sh` finds the binary automatically if installed to a standard location, or set `FEBIO_BINARY_PATH` in `.env` manually.

---

## Quick Start

### 1. Clone and add your FEB files

```bash
git clone --recurse-submodules <your-repo-url>
cd simulation-api-tool
cp /path/to/your_files/*.feb base_configuration/
```

> If you already cloned without `--recurse-submodules`, run:
> ```bash
> git submodule update --init --recursive
> ```

FEB files must follow this naming convention:

```
<design_key>_<size>Fr[_<extra>]_ir<ir_value>.feb

Examples:
  ball_tip_14Fr_IR12.feb               → "Ball Tip"         14Fr  IR12
  ball_tip_16Fr_IR12.feb               → "Ball Tip"         16Fr  IR12
  nelaton_tip_14Fr_IR12.feb            → "Nelaton Tip"       14Fr  IR12
  vapro_introducer_tip_14Fr_IR12.feb   → "Vapro Introducer"  14Fr  IR12
```

- Filename matching is **case-insensitive**
- The design label is auto-generated from the filename (`ball_tip` → "Ball Tip")
- Override labels in `config/catheter_catalogue.yaml` — see [Adding a new design](#adding-a-new-catheter-design)

> FEB files are gitignored and never committed. Each VM maintains its own `base_configuration/` files.

### 2. Set up the environment (run once)

```bash
chmod +x setup.sh && ./setup.sh
```

This does the following automatically:
- Checks Docker, Docker Compose, and openssl
- Scans for port conflicts (3080, 8000, 8001, 5000, 6379, 8888, 11434, 7700, 27017)
- Finds the FEBio binary and writes paths to `.env`
- Sets `RUNS_HOST_PATH` and `BASE_CONFIG_HOST_PATH` to absolute paths in `.env`
- Generates fresh cryptographic secrets in `.env.librechat`

For local Python development (outside Docker), also run:

```bash
bash scripts/setup-common-env.sh
source .venv/bin/activate
```

This creates a single `.venv/` shared by the main tool, xplt-parser, surrogate-lab, and JupyterLab.

### 3. Start the stack

```bash
docker compose -f docker-compose.librechat.yml up --build -d
```

**First run only:** Ollama downloads `qwen2.5:7b` (~4.7 GB). Wait for it:

```bash
docker compose -f docker-compose.librechat.yml logs -f ollama-init
# Wait for: "Model pull complete."
```

Check all services are up:

```bash
docker compose -f docker-compose.librechat.yml ps
```

### 4. Register a LibreChat account

Open **http://localhost:3080** and click **Sign up**. Use any email and password.

### 5. Create the agents

**To skip the credential prompt**, add your account to `.env.librechat` first:

```ini
LIBRECHAT_ADMIN_EMAIL=you@example.com
LIBRECHAT_ADMIN_PASSWORD=yourpassword
```

**Simulation Assistant** (FEM jobs, DOE, research documents):

```bash
bash scripts/create-agent.sh
```

**Coating Coverage Analyst** (CSAR analysis, cross-VM model loading):

```bash
python scripts/setup-csar-agent.py \
  --url http://localhost:3080 \
  --username you@example.com \
  --password yourpassword
```

Both scripts create or update the respective agent and are safe to re-run after pulling new code.

> **Do not create agents manually via the Agent Builder UI.** The scripts are the only way to attach all MCP tools correctly.

### 6. Start chatting

- **http://localhost:3080 → Agents → Simulation Assistant** — for running FEM simulations
- **http://localhost:3080 → Agents → Coating Coverage Analyst** — for CSAR / coating coverage analysis

---

## Using the Agents

### Coating Coverage Analyst — CSAR analysis

> Full documentation: [docs/csar-agent.md](docs/csar-agent.md)

The Coating Coverage Analyst specialises in **CSAR (Contact Surface Area Ratio)** — the fraction of the catheter's surface that is in contact with the vessel wall at a given insertion depth. This is the key metric for evaluating how effectively a coating (drug-eluting, hydrophilic, antimicrobial) will be delivered.

```
CSAR = faces with contact pressure > 0
       ────────────────────────────────
       total catheter surface faces
```

**Example conversation:**

```
You:   What is the coating coverage for my latest simulation?
Agent: Finds VTP file, runs analysis → returns coverage %, depth at peak,
       first-contact depth, and a 2-panel PNG plot.

You:   Compare tip (0–50 mm) and mid-shaft (50–150 mm) coverage separately.
Agent: Runs multi-band analysis, reports each region independently.

You:   Show me the 3D pressure map at 120 mm insertion.
Agent: Generates an annotated VTP → open in ParaView for heat-map view.
```

**Z-band analysis:** Specify axial regions (Z = 0 at the tip). If you don't specify, the agent analyses the whole catheter by default.

**Cross-VM model loading:** If the surrogate model was trained on another VM and registered in MLflow, the agent picks it up automatically — no manual file copying needed.

---

### Simulation Assistant — FEM jobs

### Run a simulation

```
You:   What catheter designs are available?
Agent: Lists designs discovered in base_configuration/

You:   Run a ball tip 14Fr simulation at 12 mm/s
Agent: Submits the job, returns the result folder and xplt path

You:   (conversation ends — simulation runs in the background)
```

Speed range: **10–25 mm/s**. All 10 insertion steps can share one speed or have different speeds.

### Check and cancel jobs

```
You:   What simulations are running?
Agent: Lists all recent jobs with status, run_id, and result paths

You:   Cancel the ball tip simulation
Agent: Calls cancel_simulation() — takes effect within ~1 second
```

### Surrogate model — evaluate contact pressure and CSAR

Before using the surrogate tools, a model must be trained. See [Training the surrogate model](#training-the-surrogate-model) below, or use a model from the MLflow registry trained on another VM.

```
You:   Check if the surrogate model is available
Agent: Calls list_surrogate_models() → shows latest_available, registered_models

You:   What is the coating coverage for my latest simulation?
Agent: Calls analyse_catheter_contact(vtp_path, z_bands=[whole_catheter])
       → 2-panel PNG (CSAR % + pressure vs depth) + peak coverage statistics

You:   Compare tip and mid-shaft coverage
Agent: Calls analyse_catheter_contact with two z_bands
       → separate curves per region in the same plot

You:   What is the contact pressure at 50, 100, 150 mm?
Agent: Calls evaluate_contact_pressure([50, 100, 150]) → mean/max CP per depth

You:   Annotate the VTP at 100 mm insertion
Agent: Calls predict_vtp_contact_pressure(vtp_path=..., insertion_depth_mm=100)
       → saves annotated VTP for ParaView; returns host path
```

> For full CSAR documentation including Z-band format, cross-VM registry loading, and API examples, see [docs/csar-agent.md](docs/csar-agent.md).

### Research document Q&A

```bash
cp my_paper.pdf research_documents/
```

```
You:   What is the Young's modulus used for the urethra?
Agent: Returns relevant excerpts with source + page references
```

---

## Training the Surrogate Model

Training is done through **JupyterLab** using the notebooks in `notebooks/`. The trained model is deployed to `data/surrogate/models/latest/` where the API and LibreChat agent can use it immediately.

### Access JupyterLab

When running the Docker stack:

```
http://localhost:8888?token=dtui-jupyter
```

Or set a custom token in `.env.librechat`:

```ini
JUPYTER_TOKEN=your-secure-token
```

### Training notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/full_pipeline.ipynb` | **Start here.** Discover `.xplt` results → extract surrogate data → build CSV → train → evaluate → deploy |
| `notebooks/retraining.ipynb` | **Incremental retraining.** Append new simulation results and retrain; only deploys if the new model is better |

### Using pre-run simulation results (manual / external sims)

Drop your `.feb` and `.xplt` files into the `solved_simulations/` folder at the repo root. The Jupyter container mounts this folder at `/workspace/manual_sims` and the `full_pipeline` notebook reads it automatically — no notebook edits required.

```
simulation-api-tool/
└── solved_simulations/      ← copy your .feb + .xplt files here
    ├── my_run.feb
    └── my_run.xplt
```

Files can be placed **flat** (directly in `solved_simulations/`) or in **sub-folders** (one folder per run — batch mode).

> **To change this path later**, update `MANUAL_SIMS_HOST_PATH` in **two places**:
> 1. `.env` — used by Docker Compose for bind-mount path resolution (volume substitution)
> 2. `.env.librechat` — used as container runtime env vars in the LibreChat stack
>
> Both files must match, otherwise the mount silently falls back to an empty directory.
> After changing the path, recreate the Jupyter container:
> ```bash
> docker compose -f docker-compose.librechat.yml up -d --force-recreate jupyter
> ```

### Training workflow (full pipeline)

1. Run FEBio simulations (via the agent) to generate `.xplt` result files in `runs/`, **or** copy external results into `solved_simulations/`
2. Open JupyterLab at `http://localhost:8888?token=dtui-jupyter`
3. Open `notebooks/full_pipeline.ipynb` and run all cells top to bottom
4. The notebook:
   - Discovers `.xplt` files in `runs/` or `solved_simulations/`
   - Extracts per-facet contact pressure data using **xplt-parser**
   - Builds and saves `data/surrogate/training/combined.csv`
   - Exports `data/surrogate/training/reference_facets.csv` (geometry for CSAR endpoint)
   - Trains the MLP model using **surrogate-lab**, tracked in MLflow
   - Copies artifacts to `data/surrogate/models/latest/`
5. Tell the agent: *"Check if the surrogate model is available"* — it will confirm `latest_available: true`

**Optional — register the model for cross-VM use:**

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.register_model(f"runs:/{run_id}/.", "CatheterCSARSurrogate")
```

Once registered, any VM pointing at the same MLflow server can load this model via `registered_model_name="CatheterCSARSurrogate"` — no file copying needed. See [docs/csar-agent.md](docs/csar-agent.md) for the full cross-VM workflow.

### Retraining with new data

After adding new simulation runs:

```
Open notebooks/retraining.ipynb → Run All
```

The retraining notebook appends new cases to `combined.csv`, retrains, compares RMSE against the previously deployed model, and only overwrites `models/latest/` if the new model is better. Set `FORCE_DEPLOY = True` to override the comparison.

### Shared data directory

All surrogate data is stored in `./data/surrogate/` on the host and bind-mounted into containers:

```
data/surrogate/
├── training/
│   ├── combined.csv          ← merged dataset from all .xplt extractions
│   ├── reference_facets.csv  ← catheter geometry (used by /surrogate/csar endpoint)
│   └── <run_id>.csv          ← per-case CSVs for inspection
├── models/
│   └── latest/
│       ├── best_model.pt     ← trained PyTorch weights
│       ├── config.yaml       ← model architecture and feature config
│       ├── x_scaler.pkl      ← input feature normaliser
│       └── y_scaler.pkl      ← output normaliser
└── results/
    ├── *.png                 ← CSAR plots, training scatter plots
    └── *_predicted.vtp       ← annotated VTP files
```

---

## Services and Ports

| Port | Service | Purpose |
|---|---|---|
| **3080** | LibreChat UI | Main conversational interface |
| **8000** | Simulation API | FastAPI REST (`/api/v1/`) — docs at `/docs` |
| **8001** | MCP Server | Tool bridge between agent and API (SSE) |
| **8888** | JupyterLab | Surrogate training notebooks |
| **5000** | MLflow | Experiment tracking and model registry |
| **6379** | Redis | Celery task queue |
| **11434** | Ollama | Local LLM (qwen2.5:7b) |
| **7700** | Meilisearch | LibreChat conversation search |
| **27017** | MongoDB | LibreChat history and agent config |

**Useful commands:**

```bash
# Check all services
docker compose -f docker-compose.librechat.yml ps

# API health
curl http://localhost:8000/api/v1/health

# Live logs for any service
docker compose -f docker-compose.librechat.yml logs <service> --tail=50
# Services: librechat  api  worker  mcp-server  jupyter  ollama  mlflow  redis

# Stop everything (volumes are preserved)
docker compose -f docker-compose.librechat.yml down

# Full teardown including volumes (all data deleted)
docker compose -f docker-compose.librechat.yml down -v
```

---

## Simulation Results

Every run creates a folder at `runs/run_YYYYMMDD_HHMMSS_xxxx/`:

| File | Description |
|---|---|
| `input.xplt` | FEBio binary result — open with FEBio Studio or parse with xplt-parser |
| `log.txt` | Live solver output |
| `input.feb` | The configured FEB file submitted to the solver |
| `CANCEL` | Present only if the run was cancelled |

The `runs/` directory is gitignored.

---

## REST API Reference

Interactive docs at **http://localhost:8000/docs**

### Simulation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness check |
| `GET` | `/api/v1/catheter-designs` | List all designs |
| `POST` | `/api/v1/catheter-designs/refresh` | Rescan `base_configuration/` |
| `POST` | `/api/v1/simulations/run-catheter` | Submit a simulation (async) |
| `GET` | `/api/v1/simulations` | List all simulation runs |
| `GET` | `/api/v1/simulations/{task_id}` | Poll job status |
| `POST` | `/api/v1/simulations/cancel` | Cancel a running job |
| `POST` | `/api/v1/doe/run` | Submit a DOE speed sweep |
| `GET` | `/api/v1/doe/{task_id}` | Poll DOE status |
| `POST` | `/api/v1/doe/preview-speeds` | Preview DOE speed arrays |

### Surrogate Model & CSAR

All `POST` endpoints accept optional `registered_model_name` and `model_stage` fields for cross-VM registry loading.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/surrogate/models` | List MLflow runs, registered models, and local availability |
| `GET` | `/api/v1/surrogate/list-vtps` | Discover VTP files in `runs/` and `surrogate_data/` |
| `POST` | `/api/v1/surrogate/analyse-from-vtp` | **Primary.** 2-panel CSAR + pressure plot with band summaries |
| `POST` | `/api/v1/surrogate/csar-plot-from-vtp` | CSAR-only plot (PNG) from VTP geometry |
| `POST` | `/api/v1/surrogate/csar-from-vtp` | CSAR data from VTP geometry (no plot) |
| `POST` | `/api/v1/surrogate/csar` | CSAR from reference_facets.csv (no VTP required) |
| `POST` | `/api/v1/surrogate/predict-vtp` | Annotate a VTP file with predicted per-facet contact pressure |
| `POST` | `/api/v1/surrogate/predict` | Batch contact pressure prediction from inline facet list |

### ML (DOE surrogate)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ml/predict` | Instant pressure prediction (DOE-trained model) |
| `POST` | `/api/v1/ml/predict/batch` | Batch predictions |

### Research Documents

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/documents/list` | List indexed PDFs |
| `POST` | `/api/v1/documents/ingest` | Index new PDFs from `research_documents/` |
| `POST` | `/api/v1/documents/search` | Semantic search over indexed PDFs |

---

## Adding a New Catheter Design

**Drop-in (recommended):**

```bash
cp tiemann_tip_14Fr_IR12.feb base_configuration/

# Tell the agent to refresh, or via API:
curl -s -X POST http://localhost:8000/api/v1/catheter-designs/refresh
```

**Override the display label:** add an entry in `config/catheter_catalogue.yaml`:

```yaml
designs:
  tiemann_tip:
    label: "Tiemann Tip (Custom Label)"
    configurations:
      14Fr_IR12:
        label: "14Fr catheter — IR12 urethra model"
        feb_file: "tiemann_tip_14Fr_IR12.feb"
```

Then restart the API:

```bash
docker compose -f docker-compose.librechat.yml restart api
```

> Only load-curve timings and `time_steps` counts are modified at runtime — geometry, materials, and contact definitions are always preserved from the base FEB file.

---

## LLM Options

The default LLM is **qwen2.5:7b via Ollama** (runs locally, no GPU required, 4.7 GB).

**CPU-only VM:**

```ini
# .env.librechat
OLLAMA_MODEL=qwen2.5:3b
```

**GPU VM (NVIDIA):** uncomment the `deploy` block under the `ollama` service in `docker-compose.librechat.yml`.

**Azure OpenAI (recommended for production):** fill in the `AZURE_OPENAI_*` section in `.env.librechat`. A gpt-4o deployment provides better tool-calling accuracy and removes the need for a local GPU.

---

## Architecture

```
User  ──►  LibreChat UI  (port 3080)
                │
                │  MCP tools via SSE
                ▼
          MCP Server  (port 8001)
          ┌──────────────────────────┐
          │ Simulation Assistant     │  ← FEM jobs, DOE, research docs
          │ Coating Coverage Analyst │  ← CSAR, z-bands, registry models
          └──────────────────────────┘
                │
                │  HTTP  /api/v1/
                ▼
          Simulation API  (port 8000, FastAPI)
                │
       ┌────────┼────────────┬──────────────┬───────────────┐
       ▼        ▼            ▼              ▼               ▼
  Celery     SQLite       MLflow        ChromaDB      Surrogate
  Worker    job store   (port 5000)   (RAG index)   Predictor
  (FEBio)               Registry                    (SurrogatePredictor)
       │                    │                              │
       ▼                    ▼                    ┌─────────┴──────────┐
  runs/run_*/          registered             local latest/    cross-VM registry
  input.xplt           models                data/surrogate/  models:/Name/stage
                          │
                          ▼
                  JupyterLab (port 8888)
                  full_pipeline.ipynb
                  retraining.ipynb
```

- **LibreChat** — chat frontend; routes tool calls through the MCP SSE server
- **MCP Server** — thin HTTP bridge; exposes two agent personalities via the same SSE endpoint
- **Simulation API** — FastAPI; loads catheter catalogue at startup, validates requests, enqueues Celery tasks
- **Celery Worker** — runs FEBio subprocesses; polls for cancellation every ~1 second
- **MLflow** — experiment tracking + **Model Registry** for cross-VM surrogate model sharing
- **SurrogatePredictor** — loads from local `latest/`, specific `run_id`, or MLflow registry by name
- **ChromaDB** — local vector store for research document semantic search
- **JupyterLab** — interactive training environment; bind-mounted access to xplt-parser, surrogate-lab, runs/, and data/surrogate/

---

## Portability — Using on a New VM

Agent definitions live in LibreChat's MongoDB (not in git). On every fresh machine:

```bash
# 1. Clone and add FEB files
git clone <repo> && cd simulation-api-tool
cp /path/to/*.feb base_configuration/

# 2. Run setup (finds FEBio, writes .env)
./setup.sh

# 3. (Optional) set credentials for non-interactive agent creation
echo "LIBRECHAT_ADMIN_EMAIL=you@example.com" >> .env.librechat
echo "LIBRECHAT_ADMIN_PASSWORD=yourpassword" >> .env.librechat

# 4. (Optional) auto-load a registry model trained on another VM
echo "SURROGATE_REGISTRY_MODEL_NAME=CatheterCSARSurrogate" >> .env

# 5. Start the stack
docker compose -f docker-compose.librechat.yml up --build -d

# 6. Wait for Ollama model download (first run only)
docker compose -f docker-compose.librechat.yml logs -f ollama-init

# 7. Register account at http://localhost:3080, then create both agents
bash scripts/create-agent.sh
python scripts/setup-csar-agent.py \
  --url http://localhost:3080 \
  --username you@example.com \
  --password yourpassword

# 8. Simulation Assistant: http://localhost:3080 → Agents → Simulation Assistant
# 9. Coverage analysis:   http://localhost:3080 → Agents → Coating Coverage Analyst
# 10. Train surrogate:    http://localhost:8888?token=dtui-jupyter
```

Both setup scripts find and update existing agents rather than creating duplicates — safe to run multiple times.

**Using a model trained on another VM:** if a model has been registered in the shared MLflow instance (`mlflow.register_model(...)`), set `SURROGATE_REGISTRY_MODEL_NAME` in `.env` before starting the stack. The Coating Coverage Analyst will automatically use it without requiring local training.

---

## Troubleshooting

**Agent can't list catheter designs / no tools available**
> Re-run the setup script — do not create agents manually via the UI:
```bash
bash scripts/create-agent.sh
```

**Ollama not ready — agent gets no response**
```bash
docker compose -f docker-compose.librechat.yml logs -f ollama-init
# Wait for: "Model pull complete."
```

**Port already in use**
```bash
sudo lsof -i TCP:<port> -sTCP:LISTEN
sudo kill -9 <PID>
./setup.sh   # confirm all ports free
```

**FEBio binary not found**
```bash
cat .env | grep FEBIO_BINARY_PATH
which febio4
# Edit .env → FEBIO_BINARY_PATH=/correct/path/to/febio4
docker compose -f docker-compose.librechat.yml restart worker
```

**Surrogate model not available — `latest_available: false`**

Option A — train locally: open `http://localhost:8888?token=dtui-jupyter` and run `notebooks/full_pipeline.ipynb`.

Option B — use a registry model from another VM:
```bash
echo "SURROGATE_REGISTRY_MODEL_NAME=CatheterCSARSurrogate" >> .env
docker compose -f docker-compose.librechat.yml restart api
```
See [docs/csar-agent.md](docs/csar-agent.md) for the full registry workflow.

**Coating Coverage Analyst agent missing**
```bash
python scripts/setup-csar-agent.py \
  --url http://localhost:3080 \
  --username you@example.com \
  --password yourpassword
```

**New FEB file not visible after dropping it in**
```
Tell the agent: "refresh catalogue"
```
Or via API:
```bash
curl -s -X POST http://localhost:8000/api/v1/catheter-designs/refresh
```

**LibreChat shows unhealthy in `docker ps`**
> False alarm — `curl` is not installed in the LibreChat image so the health check reports unhealthy. The UI is fully functional at http://localhost:3080.

**Running tests (no Docker required)**
```bash
.venv/bin/pytest tests/ -v
```

---

## License

MIT — free for commercial use. All Python dependencies are Apache 2.0, MIT, BSD-3-Clause, or PSF licensed.

> Research prototype. Not for clinical use.
