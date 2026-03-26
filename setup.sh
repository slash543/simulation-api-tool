#!/usr/bin/env bash
# =============================================================================
# setup.sh — First-time setup for the Digital Twin Simulation stack
#
# Run once after cloning:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   1.  Checks prerequisites (Docker, Docker Compose, openssl)
#   2.  Initialises git submodules (xplt-parser, surrogate-lab)
#   3.  Checks for port conflicts
#   4.  Detects the FEBio binary and writes all paths to .env
#   5.  Generates LibreChat secrets (.env.librechat) if not present
#   6.  Checks for Azure OpenAI configuration
#   7.  Verifies Python 3.10+
#   8.  Creates a .venv and installs all Python packages
#   9.  Installs surrogate-lab and xplt-parser (editable)
#   10. Installs JupyterLab + dev tools and registers the Jupyter kernel
#   11. Creates shared data directories
#   12. Writes a PYTHONPATH/env-vars helper script
#   13. Prints a clear startup guide
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${CYAN}${BOLD}==> $*${NC}"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 0. Prerequisites check
# ---------------------------------------------------------------------------
section "Checking prerequisites"

if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Install it from https://docs.docker.com/engine/install/"
fi
if ! docker info &>/dev/null; then
    error "Docker daemon is not running. Start it with: sudo systemctl start docker"
fi
info "Docker: $(docker --version)"

if docker compose version &>/dev/null 2>&1; then
    info "Docker Compose: $(docker compose version)"
else
    error "Docker Compose v2 plugin not found. Install it: https://docs.docker.com/compose/install/"
fi

if ! command -v openssl &>/dev/null; then
    error "openssl not found. Install it: sudo apt-get install -y openssl"
fi
info "openssl: OK"

# ---------------------------------------------------------------------------
# 0b. Initialise git submodules
# ---------------------------------------------------------------------------
section "Initialising git submodules"
if git -C "${REPO_ROOT}" submodule update --init --recursive; then
    info "Submodules ready (surrogate-lab, xplt-parser)"
else
    warn "git submodule init failed — surrogate notebooks may not work."
    warn "Run manually: git submodule update --init --recursive"
fi

# ---------------------------------------------------------------------------
# 1. Port conflict check
# ---------------------------------------------------------------------------
section "Checking for port conflicts"

declare -A PORTS=(
    [3080]="LibreChat UI"
    [8000]="Simulation API"
    [8001]="MCP Server"
    [5000]="MLflow"
    [6379]="Redis"
    [11434]="Ollama"
    [7700]="Meilisearch"
    [27017]="MongoDB"
)

CONFLICTS=()
for PORT in "${!PORTS[@]}"; do
    SERVICE="${PORTS[$PORT]}"
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || \
       lsof -iTCP:"${PORT}" -sTCP:LISTEN &>/dev/null 2>&1; then
        CONFLICTS+=("$PORT ($SERVICE)")
        PID=$(lsof -ti TCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | head -1 || true)
        if [[ -n "$PID" ]]; then
            PROCNAME=$(ps -p "$PID" -o comm= 2>/dev/null || echo "unknown")
            warn "Port ${PORT} (${SERVICE}) is in use by '${PROCNAME}' (PID ${PID})"
            warn "  → To free it: sudo kill -9 ${PID}"
        else
            warn "Port ${PORT} (${SERVICE}) appears to be in use"
            warn "  → To free it: sudo fuser -k ${PORT}/tcp"
        fi
    else
        info "Port ${PORT} (${SERVICE}): free"
    fi
done

if [[ ${#CONFLICTS[@]} -gt 0 ]]; then
    echo ""
    warn "Conflicting ports: ${CONFLICTS[*]}"
    warn "Stop the processes above before starting the stack, otherwise services"
    warn "will fail to bind. Re-run setup.sh after freeing ports to verify."
    echo ""
fi

# ---------------------------------------------------------------------------
# 2. Detect FEBio binary
# ---------------------------------------------------------------------------
section "Detecting FEBio installation"

SEARCH_PATHS=(
    "$(which febio4 2>/dev/null || true)"
    "$(which febio3 2>/dev/null || true)"
    "/usr/local/bin/febio4"
    "/usr/bin/febio4"
    "/opt/febio/bin/febio4"
    "$HOME/FEBioStudio/bin/febio4"
    "$HOME/febio/bin/febio4"
    "/opt/FEBioStudio/bin/febio4"
)

FEBIO_PATH=""
for p in "${SEARCH_PATHS[@]}"; do
    if [[ -n "$p" && -x "$p" ]]; then
        FEBIO_PATH="$p"
        break
    fi
done

if [[ -z "$FEBIO_PATH" ]]; then
    error "FEBio binary not found. Install FEBio Studio from https://febio.org/downloads/ then re-run this script."
fi
info "Found FEBio at: $FEBIO_PATH"

# ---------------------------------------------------------------------------
# 3. Write / update .env
# ---------------------------------------------------------------------------
section "Writing .env"

ENV_FILE="${REPO_ROOT}/.env"

upsert_env() {
    local KEY="$1" VAL="$2"
    if [[ -f "$ENV_FILE" ]]; then
        grep -v "^${KEY}=" "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
    fi
    echo "${KEY}=${VAL}" >> "$ENV_FILE"
    info "  ${KEY}=${VAL}"
}

upsert_env "FEBIO_BINARY_PATH" "$FEBIO_PATH"

FEBIO_LIB_PATH="$(dirname "$(dirname "$FEBIO_PATH")")/lib"
if [[ -d "$FEBIO_LIB_PATH" ]]; then
    upsert_env "FEBIO_LIB_PATH" "$FEBIO_LIB_PATH"
else
    warn "FEBio lib dir not found at $FEBIO_LIB_PATH — set FEBIO_LIB_PATH manually in $ENV_FILE"
fi

FEBIO_XML_PATH="$(dirname "$FEBIO_PATH")/febio.xml"
if [[ -f "$FEBIO_XML_PATH" ]]; then
    upsert_env "FEBIO_XML_PATH" "$FEBIO_XML_PATH"
else
    warn "febio.xml not found at $FEBIO_XML_PATH — set FEBIO_XML_PATH manually in $ENV_FILE"
fi

RUNS_HOST_PATH="${REPO_ROOT}/runs"
mkdir -p "$RUNS_HOST_PATH"
upsert_env "RUNS_HOST_PATH" "$RUNS_HOST_PATH"
info "  Runs folder created at: $RUNS_HOST_PATH"

BASE_CONFIG_HOST_PATH="${REPO_ROOT}/base_configuration"
mkdir -p "$BASE_CONFIG_HOST_PATH"
upsert_env "BASE_CONFIG_HOST_PATH" "$BASE_CONFIG_HOST_PATH"
info "  base_configuration folder ensured at: $BASE_CONFIG_HOST_PATH"

FEB_COUNT=$(find "$BASE_CONFIG_HOST_PATH" -maxdepth 1 -name "*.feb" 2>/dev/null | wc -l)
if [[ "$FEB_COUNT" -eq 0 ]]; then
    warn "No .feb files found in $BASE_CONFIG_HOST_PATH"
    warn "  Copy your .feb files there before starting the agent:"
    warn "    cp /path/to/your/ball_tip_14Fr_IR12.feb $BASE_CONFIG_HOST_PATH/"
    warn "  Naming convention:  <design>_<size>Fr_ir<speed>.feb"
    warn "    e.g.  ball_tip_14Fr_IR12.feb"
    warn "          nelaton_tip_12Fr_IR12.feb"
else
    info "  Found $FEB_COUNT .feb file(s) in base_configuration/"
fi

# ---------------------------------------------------------------------------
# 4. Generate LibreChat secrets (.env.librechat)
# ---------------------------------------------------------------------------
section "Generating LibreChat secrets"

LIBRECHAT_ENV="${REPO_ROOT}/.env.librechat"

if [[ -f "$LIBRECHAT_ENV" ]]; then
    info "$LIBRECHAT_ENV already exists — skipping (delete it to regenerate)"
    JUPYTER_TOKEN=$(grep '^JUPYTER_TOKEN=' "$LIBRECHAT_ENV" 2>/dev/null | cut -d= -f2 || openssl rand -hex 12)
else
    if [[ ! -f "${REPO_ROOT}/.env.librechat.example" ]]; then
        error ".env.librechat.example not found — cannot generate $LIBRECHAT_ENV"
    fi

    JWT_SECRET=$(openssl rand -hex 32)
    JWT_REFRESH_SECRET=$(openssl rand -hex 32)
    CREDS_KEY=$(openssl rand -hex 32)
    CREDS_IV=$(openssl rand -hex 16)
    MEILI_MASTER_KEY=$(openssl rand -hex 24)
    JUPYTER_TOKEN=$(openssl rand -hex 12)

    sed \
        -e "s/CHANGE_ME_JWT_SECRET/$JWT_SECRET/" \
        -e "s/CHANGE_ME_JWT_REFRESH_SECRET/$JWT_REFRESH_SECRET/" \
        -e "s/CHANGE_ME_CREDS_KEY/$CREDS_KEY/" \
        -e "s/CHANGE_ME_CREDS_IV/$CREDS_IV/" \
        -e "s/CHANGE_ME_MEILI_MASTER_KEY/$MEILI_MASTER_KEY/" \
        "${REPO_ROOT}/.env.librechat.example" > "$LIBRECHAT_ENV"

    # Upsert JUPYTER_TOKEN (add or replace)
    if grep -q '^JUPYTER_TOKEN=' "$LIBRECHAT_ENV"; then
        sed -i "s|^JUPYTER_TOKEN=.*|JUPYTER_TOKEN=${JUPYTER_TOKEN}|" "$LIBRECHAT_ENV"
    else
        echo "JUPYTER_TOKEN=${JUPYTER_TOKEN}" >> "$LIBRECHAT_ENV"
    fi

    info "Generated $LIBRECHAT_ENV with fresh secrets"
fi

# Configure JupyterLab to use the token automatically (no browser prompt)
JUPYTER_CONFIG_DIR="${HOME}/.jupyter"
mkdir -p "$JUPYTER_CONFIG_DIR"
cat > "${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py" <<JCFG
# Auto-generated by setup.sh — re-run setup.sh to rotate the token
c.IdentityProvider.token = '${JUPYTER_TOKEN}'
c.ServerApp.open_browser = False
JCFG
info "  Jupyter token pre-configured in ${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py"
info "  Token: ${JUPYTER_TOKEN}"
info "  Direct URL: http://localhost:8888/lab?token=${JUPYTER_TOKEN}"
upsert_env "JUPYTER_TOKEN" "${JUPYTER_TOKEN}"

# ---------------------------------------------------------------------------
# 5. Azure OpenAI configuration (optional)
# ---------------------------------------------------------------------------
section "Azure OpenAI configuration"

if grep -q "^AZURE_OPENAI_API_KEY=." "$LIBRECHAT_ENV" 2>/dev/null; then
    info "Azure OpenAI key already set in $LIBRECHAT_ENV"
else
    warn "Azure OpenAI is not configured. Ollama (local) will be used as the LLM."
    warn "To enable Azure OpenAI (recommended for production):"
    warn "  Edit $LIBRECHAT_ENV and fill in:"
    warn "    AZURE_OPENAI_API_KEY=<your key>"
    warn "    AZURE_OPENAI_INSTANCE_NAME=<resource name, e.g. my-openai-resource>"
    warn "    AZURE_OPENAI_DEPLOYMENT_NAME=<deployment name, e.g. gpt-4o>"
    warn "    AZURE_OPENAI_MINI_DEPLOYMENT_NAME=<mini deployment, e.g. gpt-4o-mini>"
    warn "    AZURE_OPENAI_API_VERSION=2024-02-15-preview"
fi

# ---------------------------------------------------------------------------
# 6. Python version check
# ---------------------------------------------------------------------------
section "Setting up Python environment"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    error "Python 3.10+ is required but not found. Install it with: sudo apt-get install python3.12"
fi
info "Python: $("$PYTHON_CMD" --version) ($PYTHON_CMD)"

# ---------------------------------------------------------------------------
# 7. Create virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="${REPO_ROOT}/.venv"

if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at ${VENV_DIR}"
    info "  To recreate: rm -rf ${VENV_DIR} && ./setup.sh"
else
    info "Creating virtual environment at ${VENV_DIR} ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    info "Virtual environment created."
fi

PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

"$PIP" install --upgrade pip "setuptools<82" wheel -q

# ---------------------------------------------------------------------------
# 8. Install simulation-api-tool Python dependencies
# ---------------------------------------------------------------------------
info "Installing simulation-api-tool dependencies ..."
"$PIP" install -r "${REPO_ROOT}/requirements.txt" --quiet
info "  ✓ requirements.txt installed"

# Install digital_twin_ui as editable (falls back to .pth if pyproject.toml absent)
"$PIP" install -e "${REPO_ROOT}" --no-deps --quiet 2>/dev/null || true

# Guarantee digital_twin_ui is importable via .pth
if ! ls "${VENV_DIR}/lib/python"*/site-packages/digital_twin_ui.pth &>/dev/null 2>&1; then
    SITE_PKG=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])")
    echo "${REPO_ROOT}" > "${SITE_PKG}/digital_twin_ui.pth"
fi
info "  ✓ digital_twin_ui importable"

# ---------------------------------------------------------------------------
# 9. Install surrogate-lab and xplt-parser (editable)
# ---------------------------------------------------------------------------
SURROGATE_DIR="${REPO_ROOT}/surrogate-lab"
if [ -d "$SURROGATE_DIR" ] && [ -f "${SURROGATE_DIR}/pyproject.toml" ]; then
    info "Installing surrogate-lab ..."
    "$PIP" install -e "${SURROGATE_DIR}[notebook]" --quiet
    info "  ✓ surrogate-lab installed (editable)"
else
    warn "surrogate-lab not found or submodule not populated — skipping"
fi

XPLT_DIR="${REPO_ROOT}/xplt-parser"
if [ -d "$XPLT_DIR" ] && [ -f "${XPLT_DIR}/pyproject.toml" ]; then
    info "Installing xplt-parser ..."
    "$PIP" install -e "${XPLT_DIR}[notebook]" --quiet
    info "  ✓ xplt-parser installed (editable, import as 'import xplt_core')"
else
    warn "xplt-parser not found or submodule not populated — skipping"
fi

# ---------------------------------------------------------------------------
# 10. Install JupyterLab + dev tools; register kernel
# ---------------------------------------------------------------------------
info "Installing JupyterLab and dev tools ..."
"$PIP" install --quiet \
    "jupyterlab>=4.0.0" \
    "ipywidgets>=8.0.0" \
    "nbformat>=5.9.0" \
    "pytest>=8.0.0" \
    "pytest-cov>=5.0.0" \
    "httpx>=0.27.0"
info "  ✓ JupyterLab installed"

info "Registering Jupyter kernel 'dtui' ..."
"$PYTHON" -m ipykernel install --user \
    --name dtui \
    --display-name "DTUI (Digital Twin UI)" \
    2>/dev/null \
    && info "  ✓ Kernel registered as 'DTUI (Digital Twin UI)'" \
    || warn "  Kernel registration skipped (ipykernel not available or already registered)"

# ---------------------------------------------------------------------------
# 11. Create shared data directories
# ---------------------------------------------------------------------------
section "Creating shared data directories"

mkdir -p \
    "${REPO_ROOT}/data/surrogate/training" \
    "${REPO_ROOT}/data/surrogate/models/latest" \
    "${REPO_ROOT}/data/surrogate/results" \
    "${REPO_ROOT}/data/mlruns" \
    "${REPO_ROOT}/runs" \
    "${REPO_ROOT}/logs"
info "  ✓ data/surrogate/{training,models/latest,results}"
info "  ✓ data/mlruns  — MLflow local tracking store"
info "  ✓ runs/        — simulation output files"
info "  ✓ logs/        — application logs"

# ---------------------------------------------------------------------------
# 12. Write PYTHONPATH / env-vars helper
# ---------------------------------------------------------------------------
ACTIVATE_EXTRA="${VENV_DIR}/bin/activate_dtui_extra.sh"
cat > "$ACTIVATE_EXTRA" <<EOF
# Extra environment variables for the DTUI virtual environment.
# Source alongside the venv activation:
#   source .venv/bin/activate
#   source .venv/bin/activate_dtui_extra.sh

export PYTHONPATH="${XPLT_DIR}:${SURROGATE_DIR}:\${PYTHONPATH:-}"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export SURROGATE_DATA_PATH="${REPO_ROOT}/data/surrogate"
EOF
info "  ✓ Wrote ${ACTIVATE_EXTRA}"

# ---------------------------------------------------------------------------
# Done — print startup guide
# ---------------------------------------------------------------------------
section "Setup complete"

cat <<EOF

${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BOLD}  STARTUP GUIDE${NC}
${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${BOLD}1. Add your .feb files${NC}
   Copy your FEBio geometry files to:
     ${BASE_CONFIG_HOST_PATH}/

   Naming convention (case-insensitive):
     <design>_<size>Fr_ir<speed>.feb
     e.g.  ball_tip_14Fr_IR12.feb
           nelaton_tip_12Fr_IR12.feb
           ball_tip_16Fr_IR25.feb

   The agent auto-discovers all .feb files at startup — no YAML edits needed.

${BOLD}2. Start the full stack${NC}
   docker compose -f docker-compose.librechat.yml up --build -d

${BOLD}3. Wait for services to be ready (~3–5 min on first run)${NC}
   docker compose -f docker-compose.librechat.yml logs -f ollama-init
   # Wait until you see: "Model pull complete."
   # Ollama must finish downloading qwen2.5:7b (4.7 GB) before you can chat.
   # Then verify all services are healthy:
   docker compose -f docker-compose.librechat.yml ps

${BOLD}4. Register a LibreChat account${NC}
   Open: http://localhost:3080
   Click "Sign up" and register with your email + password.

${BOLD}5. Create the Simulation Assistant agent  ← IMPORTANT${NC}
   Run this AFTER registering and AFTER the stack is fully healthy:

     bash scripts/create-agent.sh

   The script will prompt for your LibreChat credentials, then create or
   update the 'Simulation Assistant' agent with all simulation MCP tools.

   TIP: Skip the prompt by setting credentials in .env.librechat:
     LIBRECHAT_ADMIN_EMAIL=you@example.com
     LIBRECHAT_ADMIN_PASSWORD=yourpassword

${BOLD}6. Start chatting${NC}
   Open LibreChat → left sidebar → Agents → Simulation Assistant
   Ask: "What catheter designs are available to run?"

${BOLD}7. Simulation results${NC}
   All result files are written to: ${RUNS_HOST_PATH}
   Each run gets its own sub-folder: runs/run_YYYYMMDD_HHMMSS_xxxx/
     • input.xplt  — open in FEBio Studio: File > Open
     • log.txt     — solver progress (live)
     • input.feb   — configured input file

${BOLD}PYTHON ENVIRONMENT (for local development / notebooks)${NC}
   source .venv/bin/activate
   source .venv/bin/activate_dtui_extra.sh
   jupyter lab --notebook-dir=. --port=8888
   → Open http://localhost:8888/lab?token=${JUPYTER_TOKEN}
     (or copy the token below if prompted)

   Jupyter token: ${JUPYTER_TOKEN}
   Direct URL:    http://localhost:8888/lab?token=${JUPYTER_TOKEN}

   Run tests:
   .venv/bin/pytest tests/ -v

${BOLD}ADDING NEW .FEB FILES AFTER STARTUP${NC}
   Just copy the file to: ${BASE_CONFIG_HOST_PATH}/
   Then in the chat, say: "refresh catalogue"
   (no container restart needed)

${BOLD}PORTS USED${NC}
   3080   LibreChat UI      ← main browser interface
   8000   Simulation API
   8001   MCP Server (tools bridge)
   5000   MLflow tracking
   6379   Redis (task queue)
   11434  Ollama (LLM)
   7700   Meilisearch
   27017  MongoDB
   8888   JupyterLab (token pre-configured via ~/.jupyter/jupyter_lab_config.py)

${BOLD}IF A PORT IS BUSY${NC}
   Find and kill the process:
     sudo lsof -i TCP:<port> -sTCP:LISTEN
     sudo kill -9 <PID>
   Or kill by port directly:
     sudo fuser -k <port>/tcp

${BOLD}LLM OPTIONS${NC}
   Azure OpenAI (recommended for production/no-GPU VMs):
     Edit .env.librechat and set AZURE_OPENAI_API_KEY + AZURE_OPENAI_INSTANCE_NAME
     + AZURE_OPENAI_DEPLOYMENT_NAME.

   Ollama CPU-only (slower but free):
     Set OLLAMA_MODEL=qwen2.5:3b in .env.librechat for a smaller model (1.9 GB).
     Default is qwen2.5:7b (4.7 GB) — works well on GPU VMs.

${BOLD}TROUBLESHOOTING${NC}
   • Agent can't list designs → make sure you ran scripts/create-agent.sh

   • "Ollama not found" in LibreChat → model is still downloading.
     Run: docker compose -f docker-compose.librechat.yml logs ollama-init
     Wait for "Model pull complete." then refresh LibreChat.

   • LibreChat unhealthy → check logs:
     docker compose -f docker-compose.librechat.yml logs librechat --tail=50

   • Simulation fails (exit 127) → FEBio binary missing or wrong path.
     Check: cat .env  and verify FEBIO_BINARY_PATH points to your febio4 binary.

   • To restart a single service:
     docker compose -f docker-compose.librechat.yml restart <service>

   • base_configuration empty after clone → copy your .feb files:
     cp /path/to/your/design.feb ${BASE_CONFIG_HOST_PATH}/

${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
EOF
