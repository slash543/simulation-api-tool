# Digital Twin UI

A platform that automates catheter insertion FEM simulations, runs Design of Experiments campaigns, builds an ML dataset, and serves predictions through a REST API — with an AI agent interface via LibreChat.

---

## Quick Start

```bash
git clone https://github.com/slash543/simulation-api-tool.git
cd simulation-api-tool
chmod +x setup.sh && ./setup.sh
docker compose -f docker-compose.librechat.yml up --build -d
```

Open **http://localhost:3080**, register an account, and start chatting with the Simulation Assistant.

> **First run:** Ollama downloads `qwen2.5:7b` (~4.7 GB). Watch progress with:
> ```bash
> docker compose -f docker-compose.librechat.yml logs -f ollama-init
> # Wait for: "Model pull complete."
> ```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker + Docker Compose v2 | `docker compose version` must work |
| FEBio 4 | Install from [febio.org/downloads](https://febio.org/downloads/) — `setup.sh` auto-detects it |
| `openssl` | Usually pre-installed; used by `setup.sh` to generate secrets |

---

## What `setup.sh` does

Run once after cloning. It handles everything so you don't need to:

1. Checks Docker, Docker Compose, and openssl are available
2. Scans all required ports for conflicts and tells you exactly how to free them
3. Finds the FEBio binary and writes its path to `.env`
4. Creates the `runs/` directory and sets `RUNS_HOST_PATH` in `.env`
5. Generates `.env.librechat` with fresh cryptographic secrets (JWT, credentials encryption, Meilisearch)
6. Prints the complete startup guide

---

## Starting the stack

```bash
docker compose -f docker-compose.librechat.yml up --build -d
```

**Check all services are running:**

```bash
docker compose -f docker-compose.librechat.yml ps
```

**Stop the stack:**

```bash
docker compose -f docker-compose.librechat.yml down
```

---

## Services and ports

| Port | Service | Purpose |
|---|---|---|
| 3080 | LibreChat UI | Main chat interface |
| 8000 | Simulation API | FastAPI REST endpoints |
| 8001 | MCP Server | Tool bridge between agent and API |
| 5000 | MLflow | Experiment tracking |
| 6379 | Redis | Celery task queue |
| 11434 | Ollama | Local LLM |
| 7700 | Meilisearch | LibreChat search index |
| 27017 | MongoDB | LibreChat database |

---

## LLM Strategy — Ollama (default) + Azure OpenAI (optional)

The stack runs with **Ollama out of the box** — no credentials needed.  Azure OpenAI can be enabled later for faster, higher-quality responses.  When both are active, you select the model from the LibreChat UI per conversation.

| Endpoint | Model spec label | Requires |
|---|---|---|
| Ollama (local) | Digital Twin User Interface (Ollama) | Nothing — always active |
| Azure OpenAI | Digital Twin User Interface (Azure OpenAI) | Azure API key + uncommenting config |

### Enabling Azure OpenAI (both endpoints side-by-side)

**Step 1 — fill in `.env.librechat`:**

```dotenv
AZURE_OPENAI_API_KEY=<your-key-from-Azure-portal>
AZURE_OPENAI_INSTANCE_NAME=<resource-name>   # subdomain only, e.g. my-openai-eastus
                                              # NOT the full URL
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

Both **"Digital Twin User Interface (Ollama)"** and **"Digital Twin User Interface (Azure OpenAI)"** will appear as selectable presets in the LibreChat UI.

> **Important:** LibreChat validates every endpoint in `librechat.yaml` at startup. If `AZURE_OPENAI_API_KEY` is set in `.env.librechat` but the `azureOpenAI:` block is still commented out (or vice versa), LibreChat will start fine — Azure is only active when **both** the env vars and the YAML block are configured. Never set the key without also uncommenting the YAML block, or leave the block uncommented without a valid key — either mismatch will cause a startup crash.

---

## Simulation results

Every run writes files to `runs/run_YYYYMMDD_HHMMSS_xxxx/` on your host machine:

| File | Purpose |
|---|---|
| `input.xplt` | FEBio result — open in FEBio Studio: **File → Open** |
| `log.txt` | Live solver progress |
| `input.feb` | Configured input file used for this run |

---

## Running Python scripts independently

All analysis scripts can be run directly from a terminal — no Docker, no FastAPI server, no Celery required.

### Prerequisites

```bash
# One-time: create the virtual environment and install dependencies
chmod +x setup.sh && ./setup.sh
```

### Rule: always run from the project root with the venv Python

```bash
cd /path/to/simulation-api-tool

# Good — uses the venv Python which has all packages installed
.venv/bin/python scripts/my_script.py

# Bad — system Python lacks the project packages
python scripts/my_script.py
```

Alternatively, activate the venv once for your session:

```bash
source .venv/bin/activate
python scripts/my_script.py   # now works
deactivate                     # when done
```

### Quick sanity check

```bash
cd simulation-api-tool
.venv/bin/python -c "from digital_twin_ui.extraction.xplt_parser import XpltParser; print('OK')"
```

---

### extract_pressure.py — Extract contact pressure from an xplt file

Parses a FEBio `.xplt` result file and produces:
- **CSV** — `facet_id, surface_name, surface_id, speed_mm_s, facet_area, time_step, time_s, contact_pressure`
- **PNG** — grid of contact-pressure contour snapshots at evenly-spaced timesteps (cylindrical unroll)
- **GIF** — animated contour cycling through every timestep (optional)

**Step 1 — inspect the file to see available surfaces:**

```bash
.venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt --list
```

```
Surfaces:
  id=3  faces= 23734  name='catheter_slidePrimary'
  id=4  faces= 10571  name='catheter_slideSecondary'
  ...
Surface variables : ['contact pressure', 'contact gap', ...]
```

**Step 2 — extract to CSV + contour plot:**

```bash
.venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt \
    --surface "catheter_slidePrimary" \
    --speed 5.0 \
    --output-dir results/
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--surface NAME` | first surface | Surface name (use `--list` to see choices) |
| `--speed N` | `5.0` | Insertion speed in mm/s (stored as metadata in CSV) |
| `--variable NAME` | `contact pressure` | Surface variable to extract |
| `--output-dir DIR` | `results/` | Directory for output files |
| `--list` | — | Print surfaces and variables, then exit |
| `--no-plot` | — | CSV only — skip PNG and GIF generation |
| `--animate` | — | Also save an animated GIF (requires Pillow) |

**Output files** (in `--output-dir`):

| File | Contents |
|---|---|
| `<stem>_<surface>_pressure.csv` | Full time-series table |
| `<stem>_<surface>_contour.png` | 12-panel contour snapshot grid |
| `<stem>_<surface>_animation.gif` | Animated contour (`--animate` only) |

**Examples:**

```bash
# CSV only (skip plots — fastest):
.venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt \
    --surface "catheter_slidePrimary" --no-plot

# Both contact surfaces, separate output files:
.venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt \
    --surface "catheter_slidePrimary" --output-dir results/

.venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt \
    --surface "catheter_slideSecondary" --output-dir results/

# With animation:
.venv/bin/python scripts/extract_pressure.py conf_file/DT_BT_14Fr_FO_10E_IR12.xplt \
    --surface "catheter_slidePrimary" --animate --output-dir results/
```

---

## Research Documents (RAG)

Add PDFs to `research_documents/` and the agent can answer questions about them with source citations.

### Adding documents

Drop PDFs into `research_documents/` — the API detects them automatically:

```bash
cp my_paper.pdf research_documents/
```

The API scans `research_documents/` at startup and again every 60 seconds.  New PDFs are ingested automatically; already-indexed PDFs are skipped.  You do not need to call any endpoint manually.

To force re-ingest after replacing a PDF with an updated version:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest?force=true"
```

### Asking questions

> *"What material model is used for the urethra tissue?"*
> *"What is the Young's modulus of the catheter body?"*
> *"Explain the contact algorithm used in FEBio."*

The agent retrieves relevant chunks and cites the source PDF for every piece of information.

### RAG stack — all open-source, commercial use permitted

| Component | Library | License |
|---|---|---|
| PDF parsing + OCR | `docling` | MIT |
| Embeddings | `sentence-transformers` (BAAI/bge-small-en-v1.5) | Apache 2.0 / MIT |
| Vector store | `chromadb` | Apache 2.0 |

Everything runs locally — no external API calls required.

---

## Catheter Designs

The agent guides the user through design selection before running a simulation:

1. **Choose tip design** — Ball Tip, Nelaton Tip, or Vapro Introducer
2. **Choose configuration** — catheter size × urethra model (e.g. 14Fr IR12, 16Fr IR12)
3. **Provide 10 insertion speeds** (one per step), or ask for a uniform value between 10–25 mm/s

Base FEB files live in `base_configuration/`.  Replace any file with updated geometry and the system continues to work — only load curve timings and step counts are modified, never geometry or materials.

---

## Troubleshooting

**Ollama "model not found" in LibreChat**
The model is still downloading. Check progress:
```bash
docker compose -f docker-compose.librechat.yml logs ollama-init
```
Wait for `"Model pull complete."` then refresh LibreChat.

**Port already in use**
```bash
sudo lsof -i TCP:<port> -sTCP:LISTEN   # find the PID
sudo kill -9 <PID>
# or: sudo fuser -k <port>/tcp
```
Re-run `setup.sh` to verify all ports are free.

**Simulation fails (exit 127)**
FEBio binary path is wrong. Check:
```bash
cat .env | grep FEBIO_BINARY_PATH
```
Update the path and restart the worker:
```bash
docker compose -f docker-compose.librechat.yml restart worker
```

**RAG endpoints return 500 / "RAG dependencies not installed"**
The Docker image was built before `chromadb`, `sentence-transformers`, and `docling` were added to `requirements.txt`.  Rebuild the image:
```bash
docker compose -f docker-compose.librechat.yml build api worker
docker compose -f docker-compose.librechat.yml up -d api worker
```

**RAG endpoints return 404 "research_documents/ directory not found"**
The bind-mount was missing.  After pulling the latest code, rebuild and restart:
```bash
git pull
docker compose -f docker-compose.librechat.yml up -d --build api worker
```

**LibreChat won't start after setting Azure OpenAI key**
LibreChat crashes if the Azure key is set in `.env.librechat` but the `azureOpenAI:` block in `librechat.yaml` is still commented out, or if the block is uncommented but `AZURE_OPENAI_INSTANCE_NAME` is empty.
Two options:
- **Use Ollama only:** leave the `azureOpenAI:` block commented out and clear `AZURE_OPENAI_API_KEY=` in `.env.librechat`.
- **Use both:** fill in all `AZURE_OPENAI_*` vars **and** uncomment both the endpoint block and the `DTUI-Azure` modelSpec in `librechat.yaml` (see [Enabling Azure OpenAI](#enabling-azure-openai-both-endpoints-side-by-side) above).

**LibreChat won't start (other reasons)**
```bash
docker compose -f docker-compose.librechat.yml logs librechat --tail=50
```

---

## Architecture

```
User  ──►  LibreChat UI  (port 3080)
                │
                │  tool calls via MCP (SSE)
                ▼
          MCP Server  (port 8001)
                │
                │  HTTP
                ▼
          Simulation API  (port 8000, FastAPI)
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
  Celery     MLflow    ChromaDB
  Worker    (port 5000)  (local)
  (FEBio)
```

---

## License

MIT — free for commercial use.

All dependencies are Apache 2.0, MIT, BSD-3-Clause, or PSF licensed.

> Research prototype. Not for clinical use.
