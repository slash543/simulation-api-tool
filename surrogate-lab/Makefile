.PHONY: env env-dev env-full train test clean mlflow help

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

help:
	@echo "SurrogateLab — available targets:"
	@echo "  make env          Create .venv and install core deps"
	@echo "  make env-dev      + dev/test deps (pytest)"
	@echo "  make env-full     + dev + notebook (Jupyter)"
	@echo "  make train        Run training with configs/config.yaml"
	@echo "  make test         Run unit tests"
	@echo "  make mlflow       Open MLflow UI"
	@echo "  make clean        Remove .venv, checkpoints, artifacts, mlruns"

env:
	bash setup_env.sh

env-dev:
	bash setup_env.sh --dev

env-full:
	bash setup_env.sh --full

train:
	$(PYTHON) scripts/train.py

test:
	$(VENV)/bin/pytest

mlflow:
	$(VENV)/bin/mlflow ui --backend-store-uri mlruns/

clean:
	rm -rf $(VENV) checkpoints/ artifacts/ mlruns/ __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
