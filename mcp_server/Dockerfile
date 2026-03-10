# =============================================================================
# MCP Server — Digital Twin Simulation Tools
#
# Lightweight Python 3.12 image that runs the FastMCP SSE server.
# Exposes simulation tools to any MCP-compatible LLM client.
#
# Port: 8001 (SSE transport at /sse)
# =============================================================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Non-root user
RUN useradd --create-home --shell /bin/bash mcpuser \
    && chown -R mcpuser:mcpuser /app
USER mcpuser

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); r=s.connect_ex(('localhost',8001)); s.close(); exit(r)"

CMD ["python", "server.py"]
