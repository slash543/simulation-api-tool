# =============================================================================
# Digital Twin UI — multi-stage Dockerfile
#
# Stages:
#   base      — Python 3.12 slim with system deps
#   builder   — install Python packages into /install
#   api       — production FastAPI / Uvicorn image
#   worker    — production Celery worker image (same deps, different CMD)
#
# Build:
#   docker build --target api    -t digital-twin-ui:api    .
#   docker build --target worker -t digital-twin-ui:worker .
#
# The default target is "api".
# =============================================================================

# --------------------------------------------------------------------------- #
# Stage 1: base — minimal OS layer shared by all stages                        #
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS base

# Prevent .pyc files and enable unbuffered stdout for log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages needed at runtime:
#   libgomp1  — OpenMP (required by PyTorch CPU builds)
#   libglib2-0 — transitive dep for several packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# --------------------------------------------------------------------------- #
# Stage 2: builder — install Python dependencies                               #
# --------------------------------------------------------------------------- #
FROM base AS builder

# Build tools only needed during install (gcc for some C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer-cache efficiency
COPY requirements.txt .

# Install into a dedicated prefix so we can copy it cleanly to later stages
RUN pip install --prefix=/install -r requirements.txt


# --------------------------------------------------------------------------- #
# Stage 3: api — FastAPI / Uvicorn production image                            #
# --------------------------------------------------------------------------- #
FROM base AS api

LABEL org.opencontainers.image.title="Digital Twin UI — API"
LABEL org.opencontainers.image.description="FastAPI service for catheter simulation, DOE, and ML prediction"
LABEL org.opencontainers.image.version="0.1.0"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY digital_twin_ui/ ./digital_twin_ui/
COPY config/           ./config/
COPY templates/        ./templates/

# Create runtime directories that are expected to exist
RUN mkdir -p runs logs data/raw data/datasets models mlruns

# Add the package to Python path
RUN echo "/app" > /usr/local/lib/python3.12/site-packages/digital_twin_ui.pth

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck: call the /api/v1/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "digital_twin_ui.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]


# --------------------------------------------------------------------------- #
# Stage 4: worker — Celery worker image                                        #
# --------------------------------------------------------------------------- #
FROM base AS worker

LABEL org.opencontainers.image.title="Digital Twin UI — Celery Worker"
LABEL org.opencontainers.image.description="Celery worker for async simulation and training tasks"
LABEL org.opencontainers.image.version="0.1.0"

COPY --from=builder /install /usr/local

COPY digital_twin_ui/ ./digital_twin_ui/
COPY config/           ./config/
COPY templates/        ./templates/

RUN mkdir -p runs logs data/raw data/datasets models mlruns

RUN echo "/app" > /usr/local/lib/python3.12/site-packages/digital_twin_ui.pth

# Worker needs access to the solver binary; it is expected to be on PATH
# when the container is run (either baked into a derived image or bind-mounted).
# Example derived image:
#   FROM digital-twin-ui:worker
#   COPY --chown=root:root febio4 /usr/local/bin/febio4
#   RUN chmod +x /usr/local/bin/febio4

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Healthcheck: verify the Celery app imports cleanly
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD python -c "from digital_twin_ui.tasks.celery_app import celery_app; print('ok')" || exit 1

CMD ["celery", \
     "--app", "digital_twin_ui.tasks.celery_app:celery_app", \
     "worker", \
     "--loglevel=info", \
     "--concurrency=1", \
     "--queues=celery"]
