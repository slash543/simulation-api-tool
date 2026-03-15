#!/usr/bin/env bash
# =============================================================================
# create-agent.sh — Create or update the Simulation Assistant agent in LibreChat
#
# Usage:
#   bash scripts/create-agent.sh
#   bash scripts/create-agent.sh --agent-id agent_XXXXXXXXXXXX
#
# The script:
#   1. Waits for LibreChat to be reachable
#   2. Ensures httpx is installed
#   3. Reads credentials from env vars or .env.librechat, then prompts if needed
#   4. Finds any existing 'Simulation Assistant' agent and updates it in-place,
#      or creates a new one if none exists (no duplicates)
#
# Environment variable overrides (skip the prompts):
#   LIBRECHAT_URL       LibreChat base URL       (default: http://localhost:3080)
#   LIBRECHAT_USERNAME  Your LibreChat email
#   LIBRECHAT_PASSWORD  Your LibreChat password
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_bold()   { printf '\033[1m%s\033[0m\n'   "$*"; }

# ---------------------------------------------------------------------------
# Detect Python — prefer the project venv so httpx is already available
# ---------------------------------------------------------------------------
detect_python() {
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        echo "${REPO_ROOT}/.venv/bin/python"
        return
    fi
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            echo "$cmd"
            return
        fi
    done
    _red "ERROR: Python 3 is not installed. Install it with:"
    _red "  sudo apt-get install -y python3 python3-pip   # Debian/Ubuntu"
    exit 1
}

PYTHON=$(detect_python)
_green "Using Python: $($PYTHON --version)"

# ---------------------------------------------------------------------------
# Ensure httpx is available
# ---------------------------------------------------------------------------
if ! "$PYTHON" -c "import httpx" 2>/dev/null; then
    _yellow "httpx not found — installing..."
    if ! "$PYTHON" -m pip install --user httpx 2>/dev/null; then
        _yellow "  --user install failed, retrying with --break-system-packages..."
        "$PYTHON" -m pip install --break-system-packages httpx
    fi
    _green "httpx installed."
fi

# ---------------------------------------------------------------------------
# Resolve LibreChat URL
# ---------------------------------------------------------------------------
LIBRECHAT_URL="${LIBRECHAT_URL:-http://localhost:3080}"

# ---------------------------------------------------------------------------
# Wait for LibreChat to be reachable (up to 3 minutes)
# ---------------------------------------------------------------------------
wait_for_librechat() {
    local url="$1"
    local max_wait=180
    local waited=0
    local check_url="${url}/api/health"

    printf 'Waiting for LibreChat at %s ' "$url"
    while true; do
        if curl -sf --max-time 5 "$check_url" > /dev/null 2>&1; then
            echo ""
            _green "LibreChat is ready."
            return 0
        fi
        if [[ $waited -ge $max_wait ]]; then
            echo ""
            _yellow "LibreChat not ready after ${max_wait}s — proceeding anyway (may fail)."
            _yellow "If it fails, wait for 'docker compose ... ps' to show all services healthy."
            return 1
        fi
        printf '.'
        sleep 5
        waited=$((waited + 5))
    done
}

# ---------------------------------------------------------------------------
# Read credentials from .env.librechat if available
# ---------------------------------------------------------------------------
LIBRECHAT_ENV_FILE="${REPO_ROOT}/.env.librechat"
if [[ -f "$LIBRECHAT_ENV_FILE" ]]; then
    # Source only the two credential vars safely (no eval)
    while IFS='=' read -r key val; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        case "$key" in
            LIBRECHAT_ADMIN_EMAIL)
                [[ -z "${LIBRECHAT_USERNAME:-}" ]] && LIBRECHAT_USERNAME="${val//\"/}"
                ;;
            LIBRECHAT_ADMIN_PASSWORD)
                [[ -z "${LIBRECHAT_PASSWORD:-}" ]] && LIBRECHAT_PASSWORD="${val//\"/}"
                ;;
        esac
    done < "$LIBRECHAT_ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Prompt for credentials if still missing
# ---------------------------------------------------------------------------
if [[ -z "${LIBRECHAT_USERNAME:-}" ]]; then
    printf 'LibreChat email   [your registered account]: '
    read -r LIBRECHAT_USERNAME
fi

if [[ -z "${LIBRECHAT_PASSWORD:-}" ]]; then
    printf 'LibreChat password: '
    read -rs LIBRECHAT_PASSWORD
    echo
fi

if [[ -z "$LIBRECHAT_USERNAME" || -z "$LIBRECHAT_PASSWORD" ]]; then
    _red "ERROR: Username and password are required."
    exit 1
fi

# ---------------------------------------------------------------------------
# Wait for LibreChat
# ---------------------------------------------------------------------------
wait_for_librechat "$LIBRECHAT_URL"

# ---------------------------------------------------------------------------
# Pass through any extra flags (e.g. --agent-id <id>)
# ---------------------------------------------------------------------------
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# Run the setup script
# ---------------------------------------------------------------------------
_bold ""
_bold "Creating / updating 'Simulation Assistant' agent in LibreChat at ${LIBRECHAT_URL} ..."

"$PYTHON" "${REPO_ROOT}/scripts/setup-agent.py" \
    --url      "$LIBRECHAT_URL"      \
    --username "$LIBRECHAT_USERNAME" \
    --password "$LIBRECHAT_PASSWORD" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    _green "========================================================"
    _green "  Simulation Assistant agent is ready in LibreChat!"
    _green "========================================================"
    echo ""
    echo "  Open LibreChat → http://localhost:3080"
    echo "  Go to:  Agents (left sidebar)  →  'Simulation Assistant'"
    echo "  Start a chat and ask:  'What catheter designs are available?'"
    echo ""
    _yellow "  TIP: If you add new .feb files to base_configuration/,"
    _yellow "  tell the agent: 'refresh catalogue'"
    _yellow "  (no container restart needed)"
    echo ""
else
    _red "Agent setup encountered errors (exit code $EXIT_CODE)."
    _red "Check that LibreChat is fully up and your credentials are correct."
    _red "Re-run this script once the stack is healthy:"
    _red "  docker compose -f docker-compose.librechat.yml ps"
fi
