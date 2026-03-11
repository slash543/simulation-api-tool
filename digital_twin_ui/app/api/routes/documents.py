"""
FastAPI route handlers for RAG document management.

Endpoints
---------
GET  /documents/list    — list all indexed PDF filenames + chunk count
POST /documents/ingest  — scan research_documents/ and ingest new PDFs
POST /documents/search  — semantic search over indexed documents
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from digital_twin_ui.app.api.schemas.documents import (
    DocumentListResponse,
    IngestResponse,
    IngestResultItem,
    SearchRequest,
    SearchResponse,
    SearchHitSchema,
)
from digital_twin_ui.app.core.config import get_settings
from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# List indexed documents
# ---------------------------------------------------------------------------

@router.get(
    "/documents/list",
    response_model=DocumentListResponse,
    tags=["documents"],
)
async def list_documents() -> DocumentListResponse:
    """
    Return all PDF filenames currently indexed in the vector store,
    plus the total chunk count.

    Call this to check what has been ingested before running a search.
    """
    try:
        from digital_twin_ui.rag.document_store import get_document_store
        store = get_document_store()
        return DocumentListResponse(
            sources=store.list_sources(),
            total_chunks=store.count(),
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG dependencies not installed: {exc}. Rebuild the Docker image.",
        ) from exc
    except Exception as exc:
        logger.error("list_documents error: {exc}", exc=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document store error: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Ingest documents
# ---------------------------------------------------------------------------

@router.post(
    "/documents/ingest",
    response_model=IngestResponse,
    tags=["documents"],
)
async def ingest_documents(
    force: bool = Query(
        False,
        description=(
            "Re-ingest all PDFs, replacing existing chunks. "
            "Use after updating or replacing a PDF file."
        ),
    ),
) -> IngestResponse:
    """
    Scan ``research_documents/`` for ``.pdf`` files and ingest any that are new.

    Already-indexed PDFs are skipped unless ``force=true``.

    Ingestion pipeline per PDF:
      1. Parse with **docling** (text-layer PDFs + scanned/OCR PDFs).
      2. Export to clean Markdown preserving headings and tables.
      3. Split into 512-character overlapping chunks.
      4. Embed with **sentence-transformers** (BAAI/bge-small-en-v1.5, MIT).
      5. Upsert into **ChromaDB** persistent vector store.
    """
    from digital_twin_ui.rag.ingestor import ingest_directory

    cfg = get_settings()
    docs_dir = cfg.rag_documents_dir_abs

    if not docs_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"research_documents/ directory not found at '{docs_dir}'. "
                "Create the directory and add PDF files to it."
            ),
        )

    results = ingest_directory(docs_dir, force=force)

    ingested = [r for r in results if not r.skipped and r.error is None]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if r.error is not None]

    logger.info(
        "Ingest complete: {i} ingested, {s} skipped, {f} failed",
        i=len(ingested),
        s=len(skipped),
        f=len(failed),
    )

    return IngestResponse(
        n_ingested=len(ingested),
        n_skipped=len(skipped),
        n_failed=len(failed),
        results=[
            IngestResultItem(
                source=r.source,
                n_chunks=r.n_chunks,
                skipped=r.skipped,
                error=r.error,
            )
            for r in results
        ],
    )


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

@router.post(
    "/documents/search",
    response_model=SearchResponse,
    tags=["documents"],
)
async def search_documents(body: SearchRequest) -> SearchResponse:
    """
    Perform a semantic search over all indexed research documents.

    Returns the *n_results* most relevant text chunks along with:
    - The source PDF filename
    - The chunk's position within that document
    - A cosine-similarity score in [0, 1] (higher = more relevant)

    **Prerequisite**: call ``POST /documents/ingest`` at least once.
    """
    from digital_twin_ui.rag.document_store import get_document_store
    from digital_twin_ui.rag.searcher import search

    store = get_document_store()
    if store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No documents are indexed. "
                "Call POST /documents/ingest to index the research_documents/ folder first."
            ),
        )

    hits = search(body.query, n_results=body.n_results)

    return SearchResponse(
        query=body.query,
        hits=[
            SearchHitSchema(
                text=h.text,
                source=h.source,
                chunk_index=h.chunk_index,
                score=h.score,
            )
            for h in hits
        ],
        total_hits=len(hits),
    )
