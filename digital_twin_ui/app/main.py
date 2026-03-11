"""
FastAPI application factory for the Digital Twin UI platform.

Usage::

    # Development server
    uvicorn digital_twin_ui.app.main:app --reload --port 8000

    # Production
    uvicorn digital_twin_ui.app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from digital_twin_ui.app.core.logging import configure_from_settings, get_logger
from digital_twin_ui.app.api.routes.simulation import router as simulation_router
from digital_twin_ui.app.api.routes.documents import router as documents_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging and any startup/shutdown resources."""
    configure_from_settings()
    logger.info("Digital Twin UI API starting up")
    yield
    logger.info("Digital Twin UI API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from digital_twin_ui.app.core.config import get_settings
    cfg = get_settings()

    app = FastAPI(
        title="Digital Twin UI",
        description=(
            "REST API for catheter insertion simulation, DOE campaigns, "
            "contact pressure extraction, and ML training."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(simulation_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")

    return app


#: Module-level singleton used by uvicorn.
app: FastAPI = create_app()
