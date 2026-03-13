# Digital Twin UI

A platform for running catheter insertion FEM simulations (FEBio) through a conversational AI interface. Drop in FEB files, chat with the agent, submit jobs, monitor progress, and retrieve results — no scripting required.

---

## What it does

- **AI chat interface** (LibreChat + Ollama) — submit simulations, list jobs, cancel runs through natural language
- **FEBio simulation runner** — modifies load-curve timings and step counts from chat parameters, runs FEBio in the background via a task queue
- **Job management** — list running/completed/cancelled jobs, get result file paths, cancel long-running simulations
- **REST API** — every capability is also exposed as a documented FastAPI endpoint

---

## Prerequisites

| Requirement | Check |
|---|---|
| Docker + Docker Compose v2 | `docker compose version` |
| FEBio 4 | Download from [febio.org/downloads](https://febio.org/downloads/) and install |
| `openssl` | `openssl version` (pre-installed on most Linux/macOS) |
| ~10 GB free disk | Ollama model (~4.7 GB) + Docker images + run outputs |

> FEBio is proprietary and not bundled. `setup.sh` finds it automatically if installed to a standard location, or you can specify the path manually.

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/slash543/simulation-api-tool.git
cd simulation-api-tool
```

### 2. Add your FEB files

The `base_configuration/` folder is already in the repo (empty). Copy your `.feb` files into it:

```bash
cp /path/to/your_files/*.feb base_configuration/
```

FEB files must follow this naming convention so the API can parse them automatically:

```
<design_key>_<size>Fr[_<extra>]_ir<ir_value>.feb

Examples:
  ball_tip_14Fr_ir12.feb               → "Ball Tip", 14Fr, IR12
  nelaton_tip_16Fr_ir12.feb            → "Nelaton Tip", 16Fr, IR12
  vapro_introducer_tip_14Fr_ir12.feb   → "Vapro Introducer Tip", 14Fr, IR12
  tiemann_tip_14Fr_ir12.feb            → "Tiemann Tip", 14Fr, IR12  ← auto-detected
```

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

### 5. Create the agent

The agent configuration is **not** persisted in the repo — it lives in LibreChat's MongoDB. Run this once after the stack is healthy to create the "Simulation Assistant" agent:

```bash
python scripts/setup-agent.py \
    --url http://localhost:3080 \
    --username your@email.com \
    --password yourpassword
```

> You must register a LibreChat account at http://localhost:3080 first, then use those credentials here.

To update an existing agent's system prompt (e.g. after pulling new code):

```bash
python scripts/setup-agent.py \
    --url http://localhost:3080 \
    --username your@email.com \
    --password yourpassword \
    --update --agent-id <agent_id>
```

### 6. Select MCP tools for the agent

In **LibreChat → Agents → Edit "Simulation Assistant"**, enable exactly these 3 MCP tools and disable all others:

| Tool | Purpose |
|---|---|
| `run_catheter_simulation` | Submit a simulation job |
| `list_simulation_jobs` | Check status / find result files |
| `cancel_simulation` | Stop a running job |

> Keeping only 3 tools reduces context window usage, which is important for smaller LLMs like qwen2.5:7b.

### 7. Open the chat interface

Go to **http://localhost:3080**, select the **Simulation Assistant** agent, and start chatting.

---

## Using the chatbot

### Running a simulation

The agent already knows all available designs from its system prompt — it does **not** need to look them up. Just tell it what you want:

```
You:   Run a ball tip 14Fr IR12 simulation at 15 mm/s
Agent: Calls run_catheter_simulation immediately, returns the result folder path
```

Available designs and configurations:

| Design key | Configurations |
|---|---|
| `ball_tip` | 14Fr_IR12, 14Fr_IR25, 16Fr_IR12 |
| `nelaton_tip` | 14Fr_IR12, 14Fr_IR25, 16Fr_IR12 |
| `vapro_introducer` | 14Fr_IR12, 16Fr_IR12 |

Speed range: **10–25 mm/s**. Every design has 10 insertion steps. A single uniform speed is automatically repeated 10 times.

After submission the agent tells you:
- `host_run_dir` — the folder on your machine where result files appear
- `host_xplt_path` — the result file to open in FEBio Studio when the job finishes
- Live solver output is written to `log.txt` in that folder

### Listing jobs

```
You:   What simulations are running?
You:   Show me my recent jobs
You:   Where is the result file from my last run?
```

### Cancelling a simulation

```
You:   Cancel the simulation
You:   Stop the current run
```

The agent calls `list_simulation_jobs` to find the `run_id`, then asks you for the `task_id` (shown at submission time), and cancels. Cancellation takes effect within ~1 second.

### Adding new FEB files

New `.feb` files are detected **only at Docker startup**. To add new designs:

```bash
cp /path/to/new_design.feb base_configuration/
docker compose -f docker-compose.librechat.yml restart api worker
```

The agent will see the new design automatically on the next conversation — no agent update needed. The filename must follow the naming convention above.

---

## Services and ports

| Port | Service | Purpose |
|---|---|---|
| **3080** | LibreChat UI | Main chat interface |
| **8000** | Simulation API | FastAPI REST endpoints (`/api/v1/`) |
| **8001** | MCP Server | Tool bridge between agent and API |
| **5000** | MLflow | Experiment tracking |
| **6379** | Redis | Celery task queue |
| **11434** | Ollama | Local LLM (qwen2.5:7b) |
| **7700** | Meilisearch | LibreChat search index |
| **27017** | MongoDB | LibreChat conversation history + agent config |

**Check all services are healthy:**

```bash
docker compose -f docker-compose.librechat.yml ps
curl http://localhost:8000/api/v1/health
```

> LibreChat may show as `unhealthy` — this is a false alarm caused by `curl` not being installed inside that image. The UI is accessible at port 3080 regardless.

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
| `input.feb` | The configured FEB file submitted to the solver |
| `CANCEL` | Present only if the run was cancelled |

The `runs/` directory is gitignored. Results accumulate locally and are never pushed to the repo.

---

## Adding a new catheter design

**Drop-in method (recommended):** name the file correctly and copy it in:

```bash
cp tiemann_tip_14Fr_ir12.feb base_configuration/
docker compose -f docker-compose.librechat.yml restart api worker
```

The design is auto-detected from the filename. No YAML or code changes needed. The agent's system prompt is updated by re-running `setup-agent.py --update`.

**Override the display label:** add an entry in `config/catheter_catalogue.yaml`:

```yaml
designs:
  tiemann_tip:
    label: "Tiemann Tip"
    configurations:
      14Fr_IR12:
        label: "14Fr catheter — IR12 urethra model"
        feb_file: "tiemann_tip_14Fr_ir12.feb"
```

Then restart the API and worker as above.

> Only load-curve timings and `time_steps` counts are modified at runtime — geometry, materials, and contact definitions are always preserved from the base FEB file.

---

## Agent information across VMs

The agent definition (system prompt, tool list) is stored in LibreChat's **MongoDB container** — it is not in the git repo. On every fresh VM:

1. Clone the repo and add FEB files
2. Run `setup.sh` and start the stack
3. Register a LibreChat account at http://localhost:3080
4. Run `setup-agent.py` to create the agent (takes ~5 seconds)

The agent will be identical to any other VM because `setup-agent.py` is the single source of truth for the prompt and tool configuration.

---

## REST API

All capabilities are also available as direct HTTP calls:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | API health check |
| `GET` | `/api/v1/catheter-designs` | List all loaded designs |
| `POST` | `/api/v1/catheter-designs/refresh` | Rescan `base_configuration/` (admin use) |
| `POST` | `/api/v1/simulations/run-catheter` | Submit a simulation job |
| `GET` | `/api/v1/simulations` | List all simulation runs |
| `GET` | `/api/v1/simulations/{task_id}` | Poll job status |
| `POST` | `/api/v1/simulations/cancel` | Cancel a running job |
| `POST` | `/api/v1/doe/run` | Submit a DOE campaign |

Interactive docs: **http://localhost:8000/docs**

---

## Standalone scripts

Scripts can be run directly without Docker. Requires the project venv (created by `setup.sh`):

```bash
cd simulation-api-tool

# Inspect surfaces and variables in a result file
.venv/bin/python scripts/extract_pressure.py path/to/result.xplt --list

# Extract contact pressure to CSV + contour PNG
.venv/bin/python scripts/extract_pressure.py path/to/result.xplt \
    --surface "catheter_slidePrimary" \
    --speed 15.0 \
    --output-dir results/

# CSV only (no plots)
.venv/bin/python scripts/extract_pressure.py path/to/result.xplt \
    --surface "catheter_slidePrimary" --no-plot

# With animated GIF
.venv/bin/python scripts/extract_pressure.py path/to/result.xplt \
    --surface "catheter_slidePrimary" --animate --output-dir results/
```

---

## Running tests

```bash
.venv/bin/pytest tests/ -v
```

Does not require Docker, FEBio, or a running API. All external calls are mocked.

---

## Architecture

```
User  ──►  LibreChat UI  (port 3080)
                │
                │  3 MCP tools via SSE
                ▼
          MCP Server  (port 8001)
                │
                │  HTTP
                ▼
          Simulation API  (port 8000, FastAPI)
                │
       ┌────────┼──────────┐
       ▼        ▼          ▼
  Celery     SQLite      MLflow
  Worker    job store   (port 5000)
  (FEBio)
       │
       ▼
  base_configuration/*.feb  →  runs/run_*/input.xplt
```

- **LibreChat** — chat frontend; routes tool calls through the MCP server
- **MCP Server** — thin bridge; 3 tools only (run / list / cancel); designs hardcoded in system prompt
- **Simulation API** — FastAPI; loads catalogue at startup, enqueues Celery tasks, manages run directories
- **Celery Worker** — runs FEBio subprocesses; polls for cancellation every ~1 second
- **MLflow** — experiment tracking for DOE campaigns

---

## Troubleshooting

**Ollama not ready — agent gets no response**
```bash
docker compose -f docker-compose.librechat.yml logs -f ollama-init
# Wait for: "Model pull complete."
```

**Port already in use**
```bash
sudo lsof -i TCP:<port> -sTCP:LISTEN
sudo kill -9 <PID>
```
Re-run `setup.sh` to confirm all ports are free before starting.

**FEBio not found — simulation fails immediately**
```bash
cat .env | grep FEBIO_BINARY_PATH
which febio4
```
Update `FEBIO_BINARY_PATH` in `.env` and restart the worker:
```bash
docker compose -f docker-compose.librechat.yml restart worker
```

**New FEB file not visible to the agent**

The catalogue is loaded once at startup. Restart the API and worker:
```bash
docker compose -f docker-compose.librechat.yml restart api worker
```
Also check the filename matches the naming convention: `<design>_<size>Fr[_extra]_ir<ir>.feb`

**LibreChat shows as unhealthy in `docker ps`**

This is a false alarm — the health check uses `curl` which is not installed in the LibreChat image. The UI is fully functional at http://localhost:3080.

**Agent not appearing in LibreChat after clone**

The agent lives in MongoDB, not in git. Re-run `setup-agent.py` to recreate it:
```bash
python scripts/setup-agent.py --url http://localhost:3080 --username ... --password ...
```

**Logs for any service**
```bash
docker compose -f docker-compose.librechat.yml logs <service> --tail=50
# Services: librechat, api, worker, mcp-server, ollama, ollama-init, mlflow, redis
```

---

## License

MIT — free for commercial use. All dependencies are Apache 2.0, MIT, BSD-3-Clause, or PSF licensed.

> Research prototype. Not for clinical use.
