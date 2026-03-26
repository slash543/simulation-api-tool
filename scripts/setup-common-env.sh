#!/usr/bin/env bash
# =============================================================================
# setup-common-env.sh — Python-only environment setup for simulation-api-tool
#
# NOTE: If you have Docker installed, run ./setup.sh instead — it covers
# everything in this script plus Docker/FEBio/LibreChat configuration in one go.
#
# Use this script only when you need a Python environment without Docker
# (e.g. a headless compute node, CI, or local notebook-only usage).
#
# Creates a single .venv that supports:
#   • simulation-api-tool  (FastAPI, Celery, MLflow, RAG, etc.)
#   • xplt-parser          (FEBio .xplt binary parsing, VTP export)
#   • surrogate-lab        (neural network surrogate training and inference)
#   • JupyterLab           (to run integration notebooks locally)
#
# Usage:
#   bash scripts/setup-common-env.sh
#
# After running:
#   • Activate:        source .venv/bin/activate
#   • Run notebooks:   jupyter lab
#   • Run tests:       .venv/bin/pytest tests/ -v
#   • Run API locally: uvicorn digital_twin_ui.app.main:app --reload
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  Digital Twin UI — Common Environment Setup"
echo "  Repo root: ${REPO_ROOT}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 0. Initialise git submodules (xplt-parser, surrogate-lab)
# ---------------------------------------------------------------------------
echo "Initialising git submodules ..."
if git -C "${REPO_ROOT}" submodule update --init --recursive; then
    echo "  ✓ Submodules ready (surrogate-lab, xplt-parser)"
else
    echo "  WARNING: git submodule init failed."
    echo "           Run manually: git submodule update --init --recursive"
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Verify Python version
# ---------------------------------------------------------------------------
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
    echo "ERROR: Python 3.10+ is required but not found."
    echo "Install it with: sudo apt-get install python3.12"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version)
echo "Using Python: ${PYTHON_VERSION} (${PYTHON_CMD})"
echo ""

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="${REPO_ROOT}/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
    echo "To recreate it, run: rm -rf ${VENV_DIR} && bash scripts/setup-common-env.sh  (or ./setup.sh)"
    echo ""
else
    echo "Creating virtual environment at ${VENV_DIR} ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment created."
    echo ""
fi

PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

# Upgrade pip silently
"$PIP" install --upgrade pip "setuptools<82" wheel -q

# ---------------------------------------------------------------------------
# 3. Install simulation-api-tool dependencies
# ---------------------------------------------------------------------------
echo "Installing simulation-api-tool dependencies ..."
"$PIP" install -r "${REPO_ROOT}/requirements.txt" --quiet
echo "  ✓ simulation-api-tool requirements installed"

# Install digital_twin_ui package (editable)
"$PIP" install -e "${REPO_ROOT}" --no-deps --quiet 2>/dev/null || \
    echo "  (Note: digital_twin_ui.pth approach used — editable install not required)"

# Ensure digital_twin_ui is importable
if [ ! -f "${VENV_DIR}/lib/python"*/site-packages/digital_twin_ui.pth 2>/dev/null ]; then
    SITE_PKG=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])")
    echo "${REPO_ROOT}" > "${SITE_PKG}/digital_twin_ui.pth"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Install surrogate-lab
# ---------------------------------------------------------------------------
SURROGATE_DIR="${REPO_ROOT}/surrogate-lab"
if [ -d "$SURROGATE_DIR" ]; then
    echo "Installing surrogate-lab (with notebook extras) ..."
    "$PIP" install -e "${SURROGATE_DIR}[notebook]" --quiet
    echo "  ✓ surrogate-lab installed (editable)"
else
    echo "WARNING: surrogate-lab directory not found at ${SURROGATE_DIR}"
    echo "         Skipping — run this script from the simulation-api-tool root."
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Install xplt-parser
# ---------------------------------------------------------------------------
XPLT_DIR="${REPO_ROOT}/xplt-parser"
if [ -d "$XPLT_DIR" ]; then
    echo "Installing xplt-parser ..."
    "$PIP" install -e "${XPLT_DIR}[notebook]" --quiet
    echo "  ✓ xplt-parser installed (editable, importable as 'import xplt_core')"
else
    echo "WARNING: xplt-parser directory not found at ${XPLT_DIR}"
    echo "         Skipping."
fi
echo ""

# ---------------------------------------------------------------------------
# 6. Install JupyterLab and dev tools
# ---------------------------------------------------------------------------
echo "Installing JupyterLab and dev tools ..."
"$PIP" install --quiet \
    jupyterlab>=4.0.0 \
    ipywidgets>=8.0.0 \
    nbformat>=5.9.0 \
    pytest>=8.0.0 \
    pytest-cov>=5.0.0 \
    httpx>=0.27.0
echo "  ✓ JupyterLab installed"
echo ""

# ---------------------------------------------------------------------------
# 7. Register Jupyter kernel
# ---------------------------------------------------------------------------
echo "Registering Jupyter kernel 'dtui' ..."
"$PYTHON" -m ipykernel install --user \
    --name dtui \
    --display-name "DTUI (Digital Twin UI)" \
    2>/dev/null && echo "  ✓ Kernel registered as 'DTUI (Digital Twin UI)'" \
    || echo "  (Kernel registration skipped — ipykernel already registered or not available)"
echo ""

# ---------------------------------------------------------------------------
# 8. Generate .env.librechat (auto-fill secrets on first clone)
# ---------------------------------------------------------------------------
ENV_LIBRECHAT="${REPO_ROOT}/.env.librechat"
ENV_EXAMPLE="${REPO_ROOT}/.env.librechat.example"

if [ -f "$ENV_LIBRECHAT" ]; then
    echo ".env.librechat already exists — skipping secret generation."
    JUPYTER_TOKEN=$(grep '^JUPYTER_TOKEN=' "$ENV_LIBRECHAT" | cut -d= -f2)
else
    echo "Generating .env.librechat with auto-generated secrets ..."
    cp "$ENV_EXAMPLE" "$ENV_LIBRECHAT"

    JWT_SECRET=$(openssl rand -hex 32)
    JWT_REFRESH_SECRET=$(openssl rand -hex 32)
    CREDS_KEY=$(openssl rand -hex 32)
    CREDS_IV=$(openssl rand -hex 16)
    MEILI_MASTER_KEY=$(openssl rand -hex 24)
    JUPYTER_TOKEN=$(openssl rand -hex 12)

    sed -i \
        -e "s|^JWT_SECRET=.*|JWT_SECRET=${JWT_SECRET}|" \
        -e "s|^JWT_REFRESH_SECRET=.*|JWT_REFRESH_SECRET=${JWT_REFRESH_SECRET}|" \
        -e "s|^CREDS_KEY=.*|CREDS_KEY=${CREDS_KEY}|" \
        -e "s|^CREDS_IV=.*|CREDS_IV=${CREDS_IV}|" \
        -e "s|^MEILI_MASTER_KEY=.*|MEILI_MASTER_KEY=${MEILI_MASTER_KEY}|" \
        -e "s|^JUPYTER_TOKEN=.*|JUPYTER_TOKEN=${JUPYTER_TOKEN}|" \
        "$ENV_LIBRECHAT"

    echo "  ✓ .env.librechat created with generated secrets"
fi
echo ""

# ---------------------------------------------------------------------------
# 9. Create shared data directories (was 8)
# ---------------------------------------------------------------------------
echo "Creating shared data directories ..."
mkdir -p \
    "${REPO_ROOT}/data/surrogate/training" \
    "${REPO_ROOT}/data/surrogate/models/latest" \
    "${REPO_ROOT}/data/surrogate/results" \
    "${REPO_ROOT}/data/mlruns" \
    "${REPO_ROOT}/runs" \
    "${REPO_ROOT}/logs"
echo "  ✓ data/surrogate/training/    — CSV exports from xplt-parser"
echo "  ✓ data/surrogate/models/latest/ — trained model artifacts"
echo "  ✓ data/surrogate/results/     — CSAR plots, predicted VTP files"
echo "  ✓ data/mlruns/                — MLflow local tracking"
echo ""

# ---------------------------------------------------------------------------
# 10. Set up PYTHONPATH helper file
# ---------------------------------------------------------------------------
ACTIVATE_EXTRA="${VENV_DIR}/bin/activate_dtui_extra.sh"
cat > "$ACTIVATE_EXTRA" <<EOF
# Extra environment variables for the DTUI virtual environment.
# Source this file in addition to activating the venv:
#   source .venv/bin/activate && source .venv/bin/activate_dtui_extra.sh

export PYTHONPATH="${XPLT_DIR}:${SURROGATE_DIR}:\${PYTHONPATH:-}"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export SURROGATE_DATA_PATH="${REPO_ROOT}/data/surrogate"
EOF
echo "  ✓ Created ${ACTIVATE_EXTRA}"
echo ""

# ---------------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "For full environment variables (MLflow, PYTHONPATH, etc.):"
echo "  source .venv/bin/activate"
echo "  source .venv/bin/activate_dtui_extra.sh"
echo ""
echo "Start JupyterLab (locally):"
echo "  source .venv/bin/activate"
echo "  jupyter lab --notebook-dir=. --port=8888"
echo "  → Open http://localhost:8888"
echo ""
echo "Start the Docker stack (recommended for full integration):"
echo "  docker compose -f docker-compose.librechat.yml up --build"
echo "  → LibreChat:  http://localhost:3080"
echo "  → JupyterLab: http://localhost:8888?token=${JUPYTER_TOKEN}"
echo "  → MLflow:     http://localhost:5000"
echo ""
echo "Run tests:"
echo "  .venv/bin/pytest tests/ -v"
echo ""
