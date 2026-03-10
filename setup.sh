#!/usr/bin/env bash
# =============================================================================
# setup.sh — First-time setup for the Digital Twin Simulation stack
#
# Run this once before starting the stack:
#   chmod +x setup.sh
#   ./setup.sh
#
# What it does:
#   1. Detects the FEBio binary path and writes FEBIO_BINARY_PATH to .env
#   2. Generates LibreChat secrets (.env.librechat) if not already present
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
error() { echo -e "${RED}[setup]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Detect FEBio binary
# ---------------------------------------------------------------------------
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
    error "FEBio binary not found. Install FEBio first (https://febio.org/downloads/) then re-run this script."
fi

info "Found FEBio at: $FEBIO_PATH"

# ---------------------------------------------------------------------------
# 2. Write / update .env
# ---------------------------------------------------------------------------
ENV_FILE=".env"

# Remove existing FEBIO_BINARY_PATH line if present, then append new value
if [[ -f "$ENV_FILE" ]]; then
    grep -v "^FEBIO_BINARY_PATH=" "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
fi

echo "FEBIO_BINARY_PATH=$FEBIO_PATH" >> "$ENV_FILE"
info "Written FEBIO_BINARY_PATH to $ENV_FILE"

# Derive and write FEBIO_LIB_PATH (sibling lib/ dir next to bin/)
FEBIO_LIB_PATH="$(dirname "$(dirname "$FEBIO_PATH")")/lib"
if [[ -d "$FEBIO_LIB_PATH" ]]; then
    grep -v "^FEBIO_LIB_PATH=" "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
    echo "FEBIO_LIB_PATH=$FEBIO_LIB_PATH" >> "$ENV_FILE"
    info "Written FEBIO_LIB_PATH to $ENV_FILE"
else
    warn "Could not find FEBio lib dir at $FEBIO_LIB_PATH — set FEBIO_LIB_PATH manually in $ENV_FILE"
fi

# Derive and write FEBIO_XML_PATH (febio.xml alongside the binary)
FEBIO_XML_PATH="$(dirname "$FEBIO_PATH")/febio.xml"
if [[ -f "$FEBIO_XML_PATH" ]]; then
    grep -v "^FEBIO_XML_PATH=" "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
    echo "FEBIO_XML_PATH=$FEBIO_XML_PATH" >> "$ENV_FILE"
    info "Written FEBIO_XML_PATH to $ENV_FILE"
else
    warn "Could not find febio.xml at $FEBIO_XML_PATH — set FEBIO_XML_PATH manually in $ENV_FILE"
fi

# ---------------------------------------------------------------------------
# 3. Generate LibreChat secrets if .env.librechat doesn't exist
# ---------------------------------------------------------------------------
LIBRECHAT_ENV=".env.librechat"

if [[ -f "$LIBRECHAT_ENV" ]]; then
    info "$LIBRECHAT_ENV already exists — skipping secret generation"
else
    if [[ ! -f ".env.librechat.example" ]]; then
        error ".env.librechat.example not found — cannot generate $LIBRECHAT_ENV"
    fi

    info "Generating LibreChat secrets..."

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
# Done
# ---------------------------------------------------------------------------
echo ""
info "Setup complete. Start the full stack with:"
echo ""
echo "    docker compose -f docker-compose.librechat.yml up --build"
echo ""
