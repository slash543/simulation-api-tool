"""
FastAPI application factory for the Digital Twin UI platform.

Usage::

    # Development server
    uvicorn digital_twin_ui.app.main:app --reload --port 8000

    # Production
    uvicorn digital_twin_ui.app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from digital_twin_ui.app.core.logging import configure_from_settings, get_logger
from digital_twin_ui.app.api.routes.simulation import router as simulation_router
from digital_twin_ui.app.api.routes.documents import router as documents_router

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PDF auto-ingest helpers
# ---------------------------------------------------------------------------

_PDF_POLL_INTERVAL_S = 60  # how often to check research_documents/ for new PDFs


def _auto_ingest() -> None:
    """
    Scan research_documents/ and ingest any PDFs not yet indexed.

    Runs synchronously so it can be dispatched to a thread executor.
    Already-indexed PDFs are skipped automatically (ingestor is idempotent).
    """
    try:
        from digital_twin_ui.app.core.config import get_settings
        from digital_twin_ui.rag.ingestor import ingest_directory

        cfg = get_settings()
        docs_dir = cfg.rag_documents_dir_abs
        docs_dir.mkdir(parents=True, exist_ok=True)

        results = ingest_directory(docs_dir)
        new = [r for r in results if not r.skipped and r.error is None]
        failed = [r for r in results if r.error is not None]

        if new:
            logger.info(
                "Auto-ingested {n} new PDF(s): {names}",
                n=len(new),
                names=[r.source for r in new],
            )
        if failed:
            logger.warning(
                "Failed to ingest {n} PDF(s): {names}",
                n=len(failed),
                names=[r.source for r in failed],
            )
    except Exception as exc:
        logger.warning("Auto-ingest skipped (RAG unavailable): {exc}", exc=exc)


async def _pdf_poll_loop() -> None:
    """
    Background asyncio task: check research_documents/ every 60 s.

    Runs _auto_ingest() in a thread executor so it does not block the event loop.
    Fires once immediately at startup, then repeats on the interval.
    """
    loop = asyncio.get_running_loop()
    # First run: ingest anything present at startup
    await loop.run_in_executor(None, _auto_ingest)
    while True:
        await asyncio.sleep(_PDF_POLL_INTERVAL_S)
        await loop.run_in_executor(None, _auto_ingest)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging, auto-ingest PDFs on startup, and poll for new ones."""
    configure_from_settings()
    logger.info("Digital Twin UI API starting up")

    # Start background PDF watcher (ingest immediately, then every 60 s)
    poll_task = asyncio.create_task(_pdf_poll_loop())

    yield

    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass
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
