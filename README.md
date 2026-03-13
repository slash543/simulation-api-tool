# Digital Twin UI

A platform for running catheter insertion FEM simulations (FEBio) through a conversational AI interface. Drop in a FEB file, chat with the agent, submit jobs, monitor progress, and retrieve results — no scripting required.

---

## What it does

- **AI chat interface** (LibreChat + Ollama) — submit simulations, list jobs, cancel runs, search research papers, all through natural language
- **FEBio simulation runner** — modifies load-curve timings and step counts from chat parameters, runs FEBio in the background via a task queue
- **Dynamic file discovery** — place `.feb` files in `base_configuration/` and they appear in the agent's menu automatically
- **Job management** — list running/completed/cancelled jobs, get result file paths, kill long-running simulations
- **Research document search** — drop PDFs in `research_documents/`, the agent indexes and searches them with source citations
- **REST API** — every capability is also exposed as a documented FastAPI endpoint

---

## Prerequisites

| Requirement | Check |
|---|---|
| Docker + Docker Compose v2 | `docker compose version` |
| FEBio 4 | Download from [febio.org/downloads](https://febio.org/downloads/) and install |
| `openssl` | `openssl version` (pre-installed on most Linux/macOS) |
| ~10 GB free disk | Ollama model (~4.7 GB) + Docker images + run outputs |

> FEBio is proprietary and not bundled. The `setup.sh` script finds it automatically if installed to a standard location, or you can specify the path manually.

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/slash543/simulation-api-tool.git
cd simulation-api-tool
```

### 2. Add your FEB files

Copy your `.feb` simulation templates into `base_configuration/`:

```bash
cp /path/to/your_design.feb base_configuration/
```

FEB files follow this naming convention so the agent can parse them automatically:

```
<design_key>_<size>Fr[_<extra>]_ir<ir_value>.feb

Examples:
  ball_tip_14Fr_ir12.feb           → "Ball Tip", 14Fr, IR12
  nelaton_tip_16Fr_ir25.feb        → "Nelaton Tip", 16Fr, IR25
  vapro_introducer_tip_14Fr_ir12.feb → "Vapro Introducer Tip", 14Fr, IR12
  tiemann_tip_14Fr_ir12.feb        → "Tiemann Tip", 14Fr, IR12  ← auto-detected
```

Any `.feb` file that does not follow this pattern is silently skipped by auto-discovery. You can still register it manually in `config/catheter_catalogue.yaml`.

> **FEB files are gitignored** — they are never committed. Each VM/deployment maintains its own `base_configuration/` files.

### 3. Run setup

```bash
chmod +x setup.sh && ./setup.sh
```

This runs once and does the following automatically:

- Checks Docker, Docker Compose, and openssl are present
- Scans ports 3080, 8000, 8001, 5000, 6379, 11434, 7700, 27017 for conflicts
- Finds the FEBio binary and writes `FEBIO_BINARY_PATH` to `.env`
- Sets `RUNS_HOST_PATH` to the project's `runs/` directory
- Generates fresh cryptographic secrets in `.env.librechat` (JWT, credential encryption, Meilisearch key)
- Prints a startup summary

If `setup.sh` reports a port conflict, free that port and re-run before continuing.

### 4. Start the stack

```bash
docker compose -f docker-compose.librechat.yml up --build -d
```

**First run only:** Ollama downloads `qwen2.5:7b` (~4.7 GB). Wait for it before using the chat interface:

```bash
docker compose -f docker-compose.librechat.yml logs -f ollama-init
# Wait for: "Model pull complete."
```

### 5. Open the chat interface

Go to **http://localhost:3080**, register an account, and start chatting.

---

## Using the chatbot

The agent has the following capabilities. All are triggered through natural language.

### Running a simulation

```
You:   I want to run a simulation
Agent: Lists available catheter designs from base_configuration/
You:   Ball Tip, 14Fr IR12, 15 mm/s uniform
Agent: Confirms 10-step speed array, submits the job, reports the result folder
```

The agent returns immediately. The simulation runs in the background. You will be told:
- `host_run_dir` — the folder on your machine where files appear (e.g. `runs/run_20260313_143000_a1b2/`)
- `host_xplt_path` — the result file to open in FEBio Studio once the job finishes
- Live solver output is written to `log.txt` in that folder

### Listing jobs

```
You:   What simulations are running?
You:   Show me my recent jobs
You:   Where is the result file from my last run?
```

The agent calls `list_simulation_jobs` and shows all runs with their status (`running`, `completed`, `cancelled`), result paths, and timestamps.

### Cancelling a simulation

```
You:   Cancel the simulation
You:   Kill that job
You:   Stop the current run
```

If you don't provide a `run_id`, the agent lists your jobs first so you can identify which one to stop. Cancellation takes effect within ~1 second.

### Adding a new FEB file while the stack is running

```bash
cp /path/to/new_design.feb base_configuration/
```

Then tell the agent:

```
You:   I added a new FEB file, can you see it?
Agent: Calls refresh_catalogue, confirms the new design appears
```

No container restart needed.

### Searching research documents

Drop PDFs into `research_documents/` and ask questions:

```
You:   What material model is used for the urethra tissue?
You:   What Young's modulus is assigned to the catheter body?
```

The agent searches indexed documents and cites the source PDF for every answer. New PDFs are indexed automatically at startup and periodically while running.

---

## Services and ports

| Port | Service | Purpose |
|---|---|---|
| **3080** | LibreChat UI | Main chat interface |
| **8000** | Simulation API | FastAPI REST endpoints (`/api/v1/`) |
| **8001** | MCP Server | Tool bridge between agent and API |
| **5000** | MLflow | Experiment tracking |
| **6379** | Redis | Celery task queue |
| **11434** | Ollama | Local LLM |
| **7700** | Meilisearch | LibreChat search index |
| **27017** | MongoDB | LibreChat conversation database |

**Check all services are healthy:**

```bash
docker compose -f docker-compose.librechat.yml ps
curl http://localhost:8000/api/v1/health
```

**Stop everything:**

```bash
docker compose -f docker-compose.librechat.yml down
```

---

## Simulation results

Every run creates a folder at `runs/run_YYYYMMDD_HHMMSS_xxxx/` on your host machine:

| File | Description |
|---|---|
| `input.xplt` | FEBio result binary — open with **File → Open** in FEBio Studio |
| `log.txt` | Live solver output — tail this to watch progress |
| `input.feb` | The configured FEB file actually submitted to the solver |
| `CANCEL` | Present only if the run was cancelled |

The `runs/` directory is gitignored. Results accumulate locally and are never pushed to the repo.

---

## Catheter designs

The agent presents all designs discovered from `base_configuration/`. Each simulation requires:

1. **Tip design** — e.g. Ball Tip, Nelaton Tip, Vapro Introducer Tip
2. **Configuration** — catheter size × urethra model — e.g. 14Fr IR12, 16Fr IR12
3. **10 insertion speeds** — one per step, in mm/s (valid range: 10–25 mm/s)
   - Or give a single uniform speed: `"15 mm/s uniform"` → repeated across all 10 steps

### Adding a new design

**Simple way (recommended):** just name the file correctly and drop it in:

```bash
cp tiemann_tip_14Fr_ir12.feb base_configuration/
```

The file is auto-detected. If the stack is running, tell the agent to refresh the catalogue. No YAML or code changes needed.

**Override the label:** add an entry in `config/catheter_catalogue.yaml`:

```yaml
designs:
  tiemann_tip:
    label: "Tiemann Tip"
    configurations:
      14Fr_IR12:
        label: "14Fr catheter — IR12 urethra model"
        feb_file: "tiemann_tip_14Fr_ir12.feb"
```

Only the load-curve timings and `time_steps` counts are modified at runtime — geometry, materials, and contact definitions are always preserved from the base file.

---

## LLM options

The stack uses **Ollama with `qwen2.5:7b`** by default — no credentials required.

### Optional: Azure OpenAI

To add Azure OpenAI as a second option alongside Ollama:

**Step 1 — fill in `.env.librechat`:**

```dotenv
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_INSTANCE_NAME=<resource-name>   # subdomain only, e.g. my-openai-eastus
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_MINI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

**Step 2 — uncomment the Azure blocks in `librechat.yaml`:**
- Uncomment the `azureOpenAI:` endpoint block (~line 32)
- Uncomment the `DTUI-Azure` modelSpec block (~line 74)

**Step 3 — restart LibreChat:**

```bash
docker compose -f docker-compose.librechat.yml up -d librechat
```

Both **"Digital Twin User Interface (Ollama)"** and **"Digital Twin User Interface (Azure OpenAI)"** will appear as selectable presets. You choose per conversation from the LibreChat UI.

> If you set `AZURE_OPENAI_API_KEY` but leave the YAML block commented out (or vice versa), LibreChat will crash at startup. Either configure both or configure neither.

---

## REST API

All chatbot capabilities are also available as direct HTTP calls:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | API health check |
| `GET` | `/api/v1/catheter-designs` | List all designs |
| `POST` | `/api/v1/catheter-designs/refresh` | Rescan `base_configuration/` |
| `POST` | `/api/v1/simulations/run-catheter` | Submit a simulation job |
| `GET` | `/api/v1/simulations` | List all simulation runs |
| `GET` | `/api/v1/simulations/{task_id}` | Poll job status |
| `POST` | `/api/v1/simulations/cancel` | Cancel a running job |
| `GET` | `/api/v1/templates` | List YAML-defined templates |
| `POST` | `/api/v1/doe/run` | Submit a DOE campaign |

Interactive docs: **http://localhost:8000/docs**

---

## Standalone scripts

Scripts can be run directly without Docker. Requires the project venv:

```bash
# One-time: setup.sh creates the venv and installs dependencies
chmod +x setup.sh && ./setup.sh

# Always run from the project root using the venv Python
cd simulation-api-tool
.venv/bin/python scripts/extract_pressure.py ...
```

### extract_pressure.py

Parses a `.xplt` result file and produces CSV output and optional contour plots.

```bash
# Inspect surfaces and variables in a file
.venv/bin/python scripts/extract_pressure.py path/to/result.xplt --list

# Extract to CSV + contour PNG
.venv/bin/python scripts/extract_pressure.py path/to/result.xplt \
    --surface "catheter_slidePrimary" \
    --speed 15.0 \
    --output-dir results/

# CSV only (skip plots)
.venv/bin/python scripts/extract_pressure.py path/to/result.xplt \
    --surface "catheter_slidePrimary" --no-plot

# With animated GIF
.venv/bin/python scripts/extract_pressure.py path/to/result.xplt \
    --surface "catheter_slidePrimary" --animate --output-dir results/
```

Output files in `--output-dir`:

| File | Contents |
|---|---|
| `<stem>_<surface>_pressure.csv` | `facet_id, surface_name, time_s, contact_pressure, ...` |
| `<stem>_<surface>_contour.png` | 12-panel contour snapshot grid |
| `<stem>_<surface>_animation.gif` | Animated contour (`--animate` only) |

---

## Running tests

```bash
.venv/bin/pytest tests/ -v
```

The test suite does not require Docker, FEBio, or a running API server. All external calls are mocked.

---

## Architecture

```
User  ──►  LibreChat UI  (port 3080)
                │
                │  tool calls via MCP/SSE
                ▼
          MCP Server  (port 8001)
                │
                │  HTTP
                ▼
          Simulation API  (port 8000, FastAPI)
                │
       ┌────────┼──────────────────┐
       ▼        ▼                  ▼
  Celery     MLflow             ChromaDB
  Worker    (port 5000)         (local disk)
  (FEBio)
       │
       ▼
  base_configuration/*.feb  →  runs/run_*/input.xplt
```

- **LibreChat** — chat frontend; routes tool calls through the MCP server
- **MCP Server** — thin bridge; translates agent tool calls into HTTP requests to the API
- **Simulation API** — FastAPI; handles requests, enqueues Celery tasks, manages run directories
- **Celery Worker** — runs FEBio subprocesses; polls for cancellation every ~1 second
- **MLflow** — optional experiment tracking for DOE campaigns
- **ChromaDB** — local vector store for research document search

---

## Troubleshooting

**Ollama not ready — "model not found" in LibreChat**
```bash
docker compose -f docker-compose.librechat.yml logs -f ollama-init
# Wait for: "Model pull complete."
```

**Port already in use**
```bash
sudo lsof -i TCP:<port> -sTCP:LISTEN    # find the process
sudo kill -9 <PID>
```
Re-run `setup.sh` to confirm all ports are free before starting.

**FEBio not found — simulation fails with exit 127**
```bash
cat .env | grep FEBIO_BINARY_PATH        # check the path
which febio4                             # find the actual binary
```
Update `FEBIO_BINARY_PATH` in `.env` and restart the worker:
```bash
docker compose -f docker-compose.librechat.yml restart worker
```

**New FEB file not appearing in the agent menu**

If you added a file while the stack was running:
```
You:  refresh the catalogue
```
Or from the terminal:
```bash
curl -X POST http://localhost:8000/api/v1/catheter-designs/refresh
```
If it still does not appear, check that the filename matches the naming convention (`<design>_<size>Fr[_extra]_ir<ir>.feb`).

**RAG returns 503 / store is empty**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest"
```
Or ask the agent: `"ingest the research documents"`.

**LibreChat won't start after setting Azure credentials**
Either fill in all `AZURE_OPENAI_*` vars in `.env.librechat` and uncomment the Azure blocks in `librechat.yaml`, or leave both untouched (Ollama-only mode). A partial configuration causes a crash.

**Logs for any service**
```bash
docker compose -f docker-compose.librechat.yml logs <service> --tail=50
# Services: librechat, api, worker, mcp-server, ollama, ollama-init, mlflow, redis
```

---

## License

MIT — free for commercial use. All dependencies are Apache 2.0, MIT, BSD-3-Clause, or PSF licensed.

> Research prototype. Not for clinical use.
