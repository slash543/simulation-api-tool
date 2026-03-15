# Digital Twin UI

A platform for running catheter insertion FEM simulations (FEBio) through a conversational AI interface. Drop in FEB files, chat with the agent, submit jobs, monitor progress, and retrieve results — no scripting required.

---

## Features

| Feature | Description |
|---|---|
| **Conversational simulation** | Submit, monitor, and cancel FEBio jobs through natural language (LibreChat + Ollama or Azure OpenAI) |
| **Auto-discovery of .feb files** | Drop a `.feb` file into `base_configuration/` and the agent finds it automatically — no YAML or code changes |
| **Per-step speed control** | Each of the 10 insertion steps can have a different speed; the agent modifies load-curve timings automatically |
| **Background execution** | Simulations run in the background via Celery; the chat session ends immediately after submission |
| **Job management** | List running/completed/cancelled jobs, get result file paths, kill a specific simulation |
| **DOE campaigns** | Latin-Hypercube or random speed sweeps over many designs, tracked in MLflow |
| **ML pressure prediction** | PyTorch surrogate model trained on DOE results; instant predictions without running FEBio |
| **Research document RAG** | Index PDFs from `research_documents/` and query them semantically from the same chat interface |
| **REST API** | All capabilities exposed as documented FastAPI endpoints at `/api/v1/` |
| **Portable** | Clone the repo on any VM, add FEB files, run `setup.sh`, and the full stack is ready |

---

## Prerequisites

| Requirement | How to check |
|---|---|
| Docker + Docker Compose v2 | `docker compose version` |
| FEBio 4 | Install from [febio.org/downloads](https://febio.org/downloads/) |
| `openssl` | `openssl version` (pre-installed on most Linux/macOS) |
| ~12 GB free disk | Ollama model (4.7 GB) + Docker images + run outputs |

> FEBio is proprietary and is **not bundled**. `setup.sh` finds the binary automatically if installed to a standard location, or you can set `FEBIO_BINARY_PATH` in `.env` manually.

---

## Quick start

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd simulation-api-tool
```

### 2. Add your FEB files

```bash
cp /path/to/your_files/*.feb base_configuration/
```

FEB files must follow this naming convention so the API can parse them automatically:

```
<design_key>_<size>Fr[_<extra>]_ir<ir_value>.feb

Examples:
  ball_tip_14Fr_IR12.feb               → "Ball Tip"          14Fr  IR12
  ball_tip_16Fr_IR12.feb               → "Ball Tip"          16Fr  IR12
  nelaton_tip_14Fr_IR12.feb            → "Nelaton Tip"        14Fr  IR12
  nelaton_tip_12Fr_IR12.feb            → "Nelaton Tip"        12Fr  IR12
  vapro_introducer_tip_14Fr_IR12.feb   → "Vapro Introducer"   14Fr  IR12
  tiemann_tip_14Fr_IR12.feb            → "Tiemann Tip"        14Fr  IR12  ← auto-detected
```

- Filename matching is **case-insensitive** (`Ball_tip_14Fr_IR12.feb` and `ball_tip_14fr_ir12.feb` both work)
- The design label is auto-generated from the filename (`ball_tip` → "Ball Tip")
- A custom label can be set in `config/catheter_catalogue.yaml` — see [Adding a new design](#adding-a-new-catheter-design)

> FEB files are gitignored and never committed. Each VM maintains its own `base_configuration/` files.

### 3. Run setup

```bash
chmod +x setup.sh && ./setup.sh
```

This runs once and does the following automatically:

- Checks Docker, Docker Compose, and openssl
- Scans for port conflicts (3080, 8000, 8001, 5000, 6379, 11434, 7700, 27017)
- Finds the FEBio binary and writes paths to `.env`
- Sets `RUNS_HOST_PATH` and `BASE_CONFIG_HOST_PATH` to absolute paths in `.env`
- Generates fresh cryptographic secrets in `.env.librechat`

### 4. Start the stack

```bash
docker compose -f docker-compose.librechat.yml up --build -d
```

**First run only:** Ollama downloads `qwen2.5:7b` (~4.7 GB). Wait for it:

```bash
docker compose -f docker-compose.librechat.yml logs -f ollama-init
# Wait for: "Model pull complete."
```

Check everything is healthy:

```bash
docker compose -f docker-compose.librechat.yml ps
```

### 5. Register a LibreChat account

Open **http://localhost:3080** and click **Sign up**. Use any email and password — this account will be used in the next step.

### 6. Create the Simulation Assistant agent

Run this after registering. It creates (or updates) the "Simulation Assistant" agent with the correct system prompt and all simulation tools:

```bash
bash scripts/create-agent.sh
```

The script prompts for your LibreChat credentials, waits for LibreChat to be ready, then creates or updates the agent automatically. Re-run it any time after pulling new code.

**To skip the credential prompt**, add your account to `.env.librechat`:

```ini
LIBRECHAT_ADMIN_EMAIL=you@example.com
LIBRECHAT_ADMIN_PASSWORD=yourpassword
```

> **Do not create the agent manually via the Agent Builder UI.** The script is the only way to attach all simulation MCP tools correctly.

### 7. Start chatting

Open **http://localhost:3080 → Agents → Simulation Assistant** and start a conversation.

---

## Using the agent

### Run a simulation

```
You:   What catheter designs are available?
Agent: Lists designs discovered in base_configuration/ (e.g. Ball Tip 14Fr, Nelaton Tip 12Fr)

You:   Run a ball tip 14Fr simulation at 12 mm/s
Agent: Submits the job, returns the result folder path and xplt path
       "Simulation running in background — results at: /home/user/.../runs/run_20250315_143022_a1b2/"

You:   (conversation ends — simulation runs for ~5 hours in the background)
```

The agent always:
1. Calls `list_catheter_designs()` first to get the actual files on disk
2. Asks you to confirm design, size, and speed if not fully specified
3. Submits via `run_catheter_simulation()` and immediately reports the results folder
4. The conversation can be closed — the simulation keeps running

### Uniform vs per-step speed

All 10 insertion steps can share the same speed or use different speeds:

```
You:   Run ball tip 14Fr at 12 mm/s for all steps
→ speeds_mm_s = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

You:   Run ball tip 14Fr starting at 10 mm/s and increasing to 20 mm/s
→ speeds_mm_s = [10, 11, 12, 13, 14, 15, 16, 17, 18, 20]
```

Speed range: **10–25 mm/s**. Outside this range the API will reject the request.

### Check and cancel jobs

```
You:   What simulations are running?
Agent: Lists all recent jobs with status, run_id, and result paths

You:   Cancel the ball tip simulation
Agent: Calls cancel_simulation() — takes effect within ~1 second
```

Cancellation writes a sentinel file to the run directory. The Celery worker picks it up and stops FEBio cleanly.

### Add new FEB files without restarting

```
You:   I just added nelaton_tip_12Fr_IR12.feb
Agent: Calls refresh_catalogue() — no container restart needed
       "Found 1 new file: nelaton_tip_12Fr_IR12.feb. Now available: ..."
```

Or trigger it manually:

```bash
curl -s -X POST http://localhost:8000/api/v1/catheter-designs/refresh
```

### Research document Q&A (RAG)

Drop PDF files into `research_documents/` and ask questions:

```bash
cp my_paper.pdf research_documents/
```

```
You:   What is the Young's modulus used for the urethra?
Agent: Calls ingest_research_documents() if needed, then search_research_documents()
       Returns relevant excerpts with source + page references
```

---

## Simulation results

Every run creates a folder at `runs/run_YYYYMMDD_HHMMSS_xxxx/` on your host machine:

| File | Description |
|---|---|
| `input.xplt` | FEBio binary result — open with **File → Open** in FEBio Studio |
| `log.txt` | Live solver output — tail this to watch progress |
| `input.feb` | The configured FEB file submitted to the solver |
| `CANCEL` | Present only if the run was cancelled |

The agent reports the exact path to `input.xplt` after every submission. The `runs/` directory is gitignored.

---

## Services and ports

| Port | Service | Purpose |
|---|---|---|
| **3080** | LibreChat UI | Main chat interface |
| **8000** | Simulation API | FastAPI REST (`/api/v1/`) — docs at `/docs` |
| **8001** | MCP Server | Tool bridge between agent and API |
| **5000** | MLflow | Experiment tracking for DOE campaigns |
| **6379** | Redis | Celery task queue |
| **11434** | Ollama | Local LLM (qwen2.5:7b) |
| **7700** | Meilisearch | LibreChat conversation search |
| **27017** | MongoDB | LibreChat history + agent config |

**Useful commands:**

```bash
# Check all services
docker compose -f docker-compose.librechat.yml ps

# API health
curl http://localhost:8000/api/v1/health

# Live logs for any service
docker compose -f docker-compose.librechat.yml logs <service> --tail=50
# Services: librechat  api  worker  mcp-server  ollama  mlflow  redis

# Stop everything (volumes are preserved)
docker compose -f docker-compose.librechat.yml down

# Full teardown including volumes (all data deleted)
docker compose -f docker-compose.librechat.yml down -v
```

---

## Adding a new catheter design

**Drop-in (recommended):** just name the file correctly:

```bash
cp tiemann_tip_14Fr_IR12.feb base_configuration/

# No restart needed — tell the agent to refresh:
# "refresh catalogue"
# or via API:
curl -s -X POST http://localhost:8000/api/v1/catheter-designs/refresh
```

The design is auto-detected from the filename. No YAML or Python changes needed.

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

## Portability — using on a new VM

The agent definition lives in LibreChat's MongoDB (not in git). On every fresh machine:

```bash
# 1. Clone and add FEB files
git clone <repo> && cd simulation-api-tool
cp /path/to/*.feb base_configuration/

# 2. Run setup (finds FEBio, writes .env)
./setup.sh

# 3. (Optional) set credentials for non-interactive agent creation
echo "LIBRECHAT_ADMIN_EMAIL=you@example.com" >> .env.librechat
echo "LIBRECHAT_ADMIN_PASSWORD=yourpassword" >> .env.librechat

# 4. Start the stack
docker compose -f docker-compose.librechat.yml up --build -d

# 5. Wait for Ollama model download (first run only)
docker compose -f docker-compose.librechat.yml logs -f ollama-init

# 6. Register account at http://localhost:3080, then create the agent
bash scripts/create-agent.sh

# 7. Chat at http://localhost:3080 → Agents → Simulation Assistant
```

`create-agent.sh` automatically finds and updates an existing "Simulation Assistant" agent rather than creating duplicates — safe to run multiple times.

---

## REST API reference

Interactive docs at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | API liveness check |
| `GET` | `/api/v1/catheter-designs` | List all designs with available configs |
| `POST` | `/api/v1/catheter-designs/refresh` | Rescan `base_configuration/` for new files |
| `POST` | `/api/v1/simulations/run-catheter` | Submit a simulation (async) |
| `GET` | `/api/v1/simulations` | List all simulation runs |
| `GET` | `/api/v1/simulations/{task_id}` | Poll job status |
| `POST` | `/api/v1/simulations/cancel` | Cancel a running job |
| `POST` | `/api/v1/doe/run` | Submit a DOE speed sweep |
| `GET` | `/api/v1/doe/{task_id}` | Poll DOE campaign status |
| `POST` | `/api/v1/doe/preview-speeds` | Preview DOE speed arrays without running |
| `POST` | `/api/v1/ml/predict` | Instant pressure prediction (ML surrogate) |
| `POST` | `/api/v1/ml/predict/batch` | Batch pressure predictions |
| `GET` | `/api/v1/documents/list` | List indexed research PDFs |
| `POST` | `/api/v1/documents/ingest` | Index new PDFs from `research_documents/` |
| `POST` | `/api/v1/documents/search` | Semantic search over indexed PDFs |

---

## LLM options

The default LLM is **qwen2.5:7b via Ollama** (runs locally, no GPU required, 4.7 GB).

**CPU-only VM (no GPU):** use the smaller model:

```ini
# .env.librechat
OLLAMA_MODEL=qwen2.5:3b
```

Rebuild to pull the new model:
```bash
docker compose -f docker-compose.librechat.yml up --build -d
```

**GPU VM (NVIDIA):** uncomment the `deploy` block in `docker-compose.librechat.yml` under the `ollama` service.

**Azure OpenAI (recommended for production):** fill in the `AZURE_OPENAI_*` section in `.env.librechat`. A gpt-4o deployment provides better tool-calling accuracy and runs without a local GPU.

---

## Architecture

```
User  ──►  LibreChat UI  (port 3080)
                │
                │  MCP tools via SSE
                ▼
          MCP Server  (port 8001)
                │
                │  HTTP  /api/v1/
                ▼
          Simulation API  (port 8000, FastAPI)
                │
       ┌────────┼────────────┬──────────────┐
       ▼        ▼            ▼              ▼
  Celery     SQLite       MLflow        ChromaDB
  Worker    job store   (port 5000)   (RAG index)
  (FEBio)
       │
       ▼
  base_configuration/*.feb  →  runs/run_*/input.xplt
```

- **LibreChat** — chat frontend; routes tool calls through the MCP SSE server
- **MCP Server** — thin HTTP bridge; translates LLM tool calls to FastAPI requests
- **Simulation API** — FastAPI; loads catheter catalogue at startup, validates requests, enqueues Celery tasks
- **Celery Worker** — runs FEBio subprocesses; polls for cancellation every ~1 second; auto-tunes thread count
- **MLflow** — experiment tracking for DOE campaigns and training runs
- **ChromaDB** — local vector store for research document semantic search

---

## Troubleshooting

**Agent can't list catheter designs / session exits immediately**
> The agent was likely created via the Agent Builder UI without MCP tools. Re-run the setup script:
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
# Re-run setup.sh to confirm all ports are free
./setup.sh
```

**FEBio binary not found — simulation fails immediately**
```bash
cat .env | grep FEBIO_BINARY_PATH
which febio4
# Update the path:
# Edit .env → FEBIO_BINARY_PATH=/correct/path/to/febio4
docker compose -f docker-compose.librechat.yml restart worker
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

**Agent duplicates appearing in LibreChat**
> Re-run `create-agent.sh` — it finds the existing agent by name and updates it in-place.

**Running tests (no Docker required)**
```bash
.venv/bin/pytest tests/ -v
```

---

## License

MIT — free for commercial use. All Python dependencies are Apache 2.0, MIT, BSD-3-Clause, or PSF licensed.

> Research prototype. Not for clinical use.
