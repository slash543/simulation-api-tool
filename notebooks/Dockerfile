# =============================================================================
# Jupyter Lab — Digital Twin UI notebook environment
#
# Installs all dependencies for:
#   - xplt-parser (FEBio .xplt binary parsing)
#   - surrogate-lab (neural network surrogate training)
#   - JupyterLab with DTUI kernel
#
# Bind-mounts at runtime (from docker-compose.librechat.yml):
#   ./xplt-parser     → /workspace/xplt-parser
#   ./surrogate-lab   → /workspace/surrogate-lab
#   ./notebooks       → /workspace/notebooks
#   ./data/surrogate  → /workspace/data/surrogate
#   ./runs            → /workspace/runs (read-only)
# =============================================================================

FROM python:3.12-slim

# System dependencies for numpy, matplotlib, PyTorch CPU builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---------------------------------------------------------------------------
# 1. Copy source packages into the image (for pip install --no-deps fallback)
#    The actual source directories are bind-mounted at runtime, but we also
#    install them during build so the image works standalone.
# ---------------------------------------------------------------------------
COPY xplt-parser/ /tmp/xplt-parser/
COPY surrogate-lab/ /tmp/surrogate-lab/

# ---------------------------------------------------------------------------
# 2. Install all requirements in one environment
# ---------------------------------------------------------------------------

# Install surrogate-lab + its dependencies (torch, sklearn, mlflow, etc.)
RUN pip install --no-cache-dir /tmp/surrogate-lab/[notebook]

# Install xplt-parser as a package (numpy, pandas, matplotlib already satisfied)
RUN pip install --no-cache-dir /tmp/xplt-parser/

# Install JupyterLab and additional utilities
RUN pip install --no-cache-dir \
        jupyterlab>=4.0.0 \
        ipywidgets>=8.0.0 \
        nbformat>=5.9.0 \
        matplotlib>=3.8.0 \
        seaborn>=0.13.0 \
        httpx>=0.27.0

# ---------------------------------------------------------------------------
# 3. Register Jupyter kernel
# ---------------------------------------------------------------------------
RUN python -m ipykernel install --sys-prefix --name dtui --display-name "DTUI (Python 3.12)"

# ---------------------------------------------------------------------------
# 4. Create non-root user to avoid permission issues with bind-mounts
#    UID/GID 1000 matches the default user on most Linux hosts.
# ---------------------------------------------------------------------------
RUN groupadd -g 1000 nbuser && useradd -u 1000 -g nbuser -ms /bin/bash nbuser

# Create workspace directories owned by nbuser
RUN mkdir -p /workspace/data/surrogate/training \
             /workspace/data/surrogate/models/latest \
             /workspace/data/surrogate/results \
             /workspace/data/mlruns \
             /workspace/notebooks \
    && chown -R nbuser:nbuser /workspace

USER nbuser

# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
EXPOSE 8888

# Start JupyterLab — shell form so $JUPYTER_TOKEN is expanded at runtime.
# Uses IdentityProvider.token (Jupyter Server 2.x API; ServerApp.token is deprecated).
CMD ["sh", "-c", "python -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/workspace --IdentityProvider.token=${JUPYTER_TOKEN} --ServerApp.password= --ServerApp.allow_root=False"]
