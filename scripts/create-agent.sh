#!/usr/bin/env bash
# =============================================================================
# create-agent.sh — Create (or update) the Simulation Assistant agent in LibreChat
#
# Usage:
#   bash scripts/create-agent.sh
#   bash scripts/create-agent.sh --update --agent-id agent_XXXXXXXXXXXX
#
# The script:
#   1. Installs httpx if not present
#   2. Prompts for your LibreChat credentials (or reads from env vars)
#   3. Runs scripts/setup-agent.py with the correct arguments
#
# Environment variable overrides (skip the prompts):
#   LIBRECHAT_URL       LibreChat base URL       (default: http://localhost:3080)
#   LIBRECHAT_USERNAME  Your LibreChat email
#   LIBRECHAT_PASSWORD  Your LibreChat password
# =============================================================================

set -euo pipefail

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
    # Project venv first (has all dependencies already installed)
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
    _red "  sudo yum install -y python3                   # RHEL/CentOS"
    exit 1
}

PYTHON=$(detect_python)
_green "Using Python: $($PYTHON --version)"

# ---------------------------------------------------------------------------
# Ensure httpx is available
# ---------------------------------------------------------------------------
if ! "$PYTHON" -c "import httpx" 2>/dev/null; then
    _yellow "httpx not found — installing..."
    # On modern Debian/Ubuntu the system Python is externally-managed; try
    # --user first, then fall back to --break-system-packages.
    if ! "$PYTHON" -m pip install --user httpx 2>/dev/null; then
        _yellow "  --user install failed (externally-managed env), retrying with --break-system-packages..."
        "$PYTHON" -m pip install --break-system-packages httpx
    fi
    _green "httpx installed."
fi

# ---------------------------------------------------------------------------
# Resolve credentials (env vars → prompt)
# ---------------------------------------------------------------------------
LIBRECHAT_URL="${LIBRECHAT_URL:-http://localhost:3080}"

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
# Pass through any extra flags (e.g. --update --agent-id <id>)
# ---------------------------------------------------------------------------
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# Run the setup script
# ---------------------------------------------------------------------------
_bold ""
_bold "Connecting to LibreChat at ${LIBRECHAT_URL} ..."

"$PYTHON" "${REPO_ROOT}/scripts/setup-agent.py" \
    --url      "$LIBRECHAT_URL"      \
    --username "$LIBRECHAT_USERNAME" \
    --password "$LIBRECHAT_PASSWORD" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
