#!/usr/bin/env bash
# =============================================================================
# setup.sh — First-time setup for the Digital Twin Simulation stack
#
# Run once before starting the stack:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   1. Checks prerequisites (Docker, Docker Compose, openssl)
#   2. Scans for port conflicts and warns with the fix command
#   3. Detects the FEBio binary and writes all paths to .env
#   4. Sets RUNS_HOST_PATH to the project ./runs/ directory
#   5. Generates LibreChat secrets (.env.librechat) if not present
#   6. Prints a clear startup guide
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

# ---------------------------------------------------------------------------
# 0. Prerequisites check
# ---------------------------------------------------------------------------
section "Checking prerequisites"

# Docker
if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Install it from https://docs.docker.com/engine/install/"
fi
if ! docker info &>/dev/null; then
    error "Docker daemon is not running. Start it with: sudo systemctl start docker"
fi
info "Docker: $(docker --version)"

# Docker Compose (v2 plugin)
if docker compose version &>/dev/null 2>&1; then
    info "Docker Compose: $(docker compose version)"
else
    error "Docker Compose v2 plugin not found. Install it: https://docs.docker.com/compose/install/"
fi

# openssl (for secret generation)
if ! command -v openssl &>/dev/null; then
    error "openssl not found. Install it: sudo apt-get install -y openssl"
fi
info "openssl: OK"

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

ENV_FILE=".env"

# Helper: upsert a key=value line in .env
upsert_env() {
    local KEY="$1" VAL="$2"
    if [[ -f "$ENV_FILE" ]]; then
        grep -v "^${KEY}=" "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
    fi
    echo "${KEY}=${VAL}" >> "$ENV_FILE"
    info "  ${KEY}=${VAL}"
}

# FEBio binary
upsert_env "FEBIO_BINARY_PATH" "$FEBIO_PATH"

# FEBio shared libraries (sibling lib/ directory)
FEBIO_LIB_PATH="$(dirname "$(dirname "$FEBIO_PATH")")/lib"
if [[ -d "$FEBIO_LIB_PATH" ]]; then
    upsert_env "FEBIO_LIB_PATH" "$FEBIO_LIB_PATH"
else
    warn "FEBio lib dir not found at $FEBIO_LIB_PATH — set FEBIO_LIB_PATH manually in $ENV_FILE"
fi

# FEBio config file (febio.xml alongside the binary)
FEBIO_XML_PATH="$(dirname "$FEBIO_PATH")/febio.xml"
if [[ -f "$FEBIO_XML_PATH" ]]; then
    upsert_env "FEBIO_XML_PATH" "$FEBIO_XML_PATH"
else
    warn "febio.xml not found at $FEBIO_XML_PATH — set FEBIO_XML_PATH manually in $ENV_FILE"
fi

# Runs directory — bind-mounted so result files are directly accessible
RUNS_HOST_PATH="$(pwd)/runs"
mkdir -p "$RUNS_HOST_PATH"
upsert_env "RUNS_HOST_PATH" "$RUNS_HOST_PATH"
info "  Runs folder created at: $RUNS_HOST_PATH"

# ---------------------------------------------------------------------------
# 4. Generate LibreChat secrets + Azure OpenAI config
# ---------------------------------------------------------------------------
section "Generating LibreChat secrets"

LIBRECHAT_ENV=".env.librechat"

if [[ -f "$LIBRECHAT_ENV" ]]; then
    info "$LIBRECHAT_ENV already exists — skipping (delete it to regenerate)"
else
    if [[ ! -f ".env.librechat.example" ]]; then
        error ".env.librechat.example not found — cannot generate $LIBRECHAT_ENV"
    fi

    JWT_SECRET=$(openssl rand -hex 32)
    JWT_REFRESH_SECRET=$(openssl rand -hex 32)
    CREDS_KEY=$(openssl rand -hex 32)
    CREDS_IV=$(openssl rand -hex 16)
    MEILI_MASTER_KEY=$(openssl rand -hex 24)

    sed \
        -e "s/CHANGE_ME_JWT_SECRET/$JWT_SECRET/" \
        -e "s/CHANGE_ME_JWT_REFRESH_SECRET/$JWT_REFRESH_SECRET/" \
        -e "s/CHANGE_ME_CREDS_KEY/$CREDS_KEY/" \
        -e "s/CHANGE_ME_CREDS_IV/$CREDS_IV/" \
        -e "s/CHANGE_ME_MEILI_MASTER_KEY/$MEILI_MASTER_KEY/" \
        ".env.librechat.example" > "$LIBRECHAT_ENV"

    info "Generated $LIBRECHAT_ENV with fresh secrets"
fi

# ---------------------------------------------------------------------------
# 5. Azure OpenAI configuration (optional but recommended)
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
    warn ""
    warn "  Find these values in:"
    warn "    Azure Portal → your OpenAI resource → Keys and Endpoint"
    warn "    Azure Portal → your OpenAI resource → Model deployments"
fi

# ---------------------------------------------------------------------------
# Done — print startup guide
# ---------------------------------------------------------------------------
section "Setup complete"

cat <<EOF

${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BOLD}  STARTUP GUIDE${NC}
${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${BOLD}1. Start the full stack${NC}
   docker compose -f docker-compose.librechat.yml up --build -d

${BOLD}2. Wait for services to be ready (~3–5 min on first run)${NC}
   docker compose -f docker-compose.librechat.yml logs -f ollama-init
   # Wait until you see: "Model pull complete."
   # Ollama must finish downloading qwen2.5:7b (4.7 GB) before you can chat.

${BOLD}3. Check all services are healthy${NC}
   docker compose -f docker-compose.librechat.yml ps

${BOLD}4. Open LibreChat${NC}
   http://localhost:3080
   Register an account → go to Agents → create the Simulation Assistant agent
   (select "Simulation Agent (Ollama)" as the model endpoint)

${BOLD}5. Simulation results${NC}
   All result files are written to: ${RUNS_HOST_PATH}
   Each run gets its own sub-folder: runs/run_YYYYMMDD_HHMMSS_xxxx/
     • input.xplt  — open in FEBio Studio: File > Open
     • log.txt     — solver progress (live)
     • input.feb   — configured input file

${BOLD}PORTS USED${NC}
   3080   LibreChat UI      ← main browser interface
   8000   Simulation API
   8001   MCP Server (tools bridge)
   5000   MLflow tracking
   6379   Redis (task queue)
   11434  Ollama (LLM)
   7700   Meilisearch
   27017  MongoDB

${BOLD}IF A PORT IS BUSY${NC}
   Find and kill the process:
     sudo lsof -i TCP:<port> -sTCP:LISTEN
     sudo kill -9 <PID>
   Or kill by port directly:
     sudo fuser -k <port>/tcp

${BOLD}LLM OPTIONS${NC}
   Azure OpenAI (recommended):
     Edit .env.librechat and set AZURE_OPENAI_API_KEY + AZURE_OPENAI_INSTANCE_NAME
     + AZURE_OPENAI_DEPLOYMENT_NAME. Use the "Simulation Assistant (Azure OpenAI)"
     model spec in LibreChat. Faster, no GPU needed.

   Ollama fallback (always available):
     No config needed. Use "Simulation Assistant (Ollama — fallback)".
     First run downloads qwen2.5:7b (4.7 GB) — wait for ollama-init to finish.

${BOLD}TROUBLESHOOTING${NC}
   • "Ollama not found" in LibreChat → model is still downloading.
     Run: docker compose -f docker-compose.librechat.yml logs ollama-init
     Wait for "Model pull complete." then refresh LibreChat.

   • LibreChat unhealthy → check logs:
     docker compose -f docker-compose.librechat.yml logs librechat --tail=50

   • Simulation fails (exit 127) → FEBio binary missing or wrong path.
     Check: cat .env  and verify FEBIO_BINARY_PATH points to your febio4 binary.

   • To restart a single service:
     docker compose -f docker-compose.librechat.yml restart <service>

${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
EOF
