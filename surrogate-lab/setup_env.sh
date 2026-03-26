#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SurrogateLab — isolated environment setup
#
# Usage:
#   bash setup_env.sh            # create .venv and install all deps
#   bash setup_env.sh --dev      # also install dev/test deps
#   bash setup_env.sh --notebook # also install Jupyter
#   bash setup_env.sh --full     # dev + notebook
# ---------------------------------------------------------------------------
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# ── Parse flags ──────────────────────────────────────────────────────────────
INSTALL_DEV=false
INSTALL_NOTEBOOK=false

for arg in "$@"; do
  case $arg in
    --dev)      INSTALL_DEV=true ;;
    --notebook) INSTALL_NOTEBOOK=true ;;
    --full)     INSTALL_DEV=true; INSTALL_NOTEBOOK=true ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# ── Check Python version ─────────────────────────────────────────────────────
PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.11"

if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)"; then
  echo "Python $PYTHON_VERSION — OK"
else
  echo "ERROR: Python >= $REQUIRED required (found $PYTHON_VERSION)"
  exit 1
fi

# ── Create virtual environment ───────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  echo "Virtual environment already exists at $VENV_DIR — skipping creation."
  echo "  To recreate: rm -rf $VENV_DIR && bash setup_env.sh"
else
  echo "Creating virtual environment in $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
  echo "Virtual environment created."
fi

# ── Upgrade pip inside the venv ───────────────────────────────────────────────
echo "Upgrading pip ..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip

# ── Install the project in editable mode ────────────────────────────────────
echo "Installing surrogatelab (editable) ..."
"$VENV_DIR/bin/pip" install --quiet -e .

# ── Optional extras ──────────────────────────────────────────────────────────
if [ "$INSTALL_DEV" = true ]; then
  echo "Installing dev/test extras ..."
  "$VENV_DIR/bin/pip" install --quiet -e ".[dev]"
fi

if [ "$INSTALL_NOTEBOOK" = true ]; then
  echo "Installing notebook extras ..."
  "$VENV_DIR/bin/pip" install --quiet -e ".[notebook]"
  # Register the venv as a Jupyter kernel so notebooks find the right packages
  "$VENV_DIR/bin/python" -m ipykernel install --user --name surrogatelab --display-name "SurrogateLab (.venv)"
  echo "Jupyter kernel 'surrogatelab' registered."
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Setup complete!"
echo ""
echo " Activate the environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " Then run training:"
echo "   python scripts/train.py --data path/to/simulation.csv"
echo ""
if [ "$INSTALL_NOTEBOOK" = true ]; then
  echo " Launch notebook:"
  echo "   jupyter notebook notebooks/training.ipynb"
  echo ""
fi
if [ "$INSTALL_DEV" = true ]; then
  echo " Run tests:"
  echo "   pytest"
  echo ""
fi
echo " View MLflow dashboard:"
echo "   mlflow ui --backend-store-uri mlruns/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
