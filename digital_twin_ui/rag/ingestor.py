"""
PDF ingestor: docling → chunks → embeddings → ChromaDB.

Pipeline per PDF
----------------
1. Parse PDF with ``docling`` (handles text-layer PDFs and scanned/OCR PDFs).
2. Export to clean Markdown (preserves headings, tables, lists).
3. Split into overlapping character chunks (~512 chars, 64-char overlap),
   preferring paragraph boundaries.
4. Embed all chunks with ``sentence-transformers`` (BAAI/bge-small-en-v1.5).
5. Upsert into ChromaDB with ``{"source": filename, "chunk_index": i}`` metadata.

Already-ingested PDFs are skipped unless ``force=True``.

Usage
-----
    from digital_twin_ui.rag.ingestor import ingest_directory, ingest_pdf
    results = ingest_directory(Path("research_documents/"))
    for r in results:
        print(r.source, r.n_chunks, r.skipped, r.error)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 512     # target characters per chunk
_CHUNK_OVERLAP = 64   # overlap characters between consecutive chunks


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    """Outcome of ingesting one PDF file."""
    source: str
    n_chunks: int
    skipped: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping character chunks.

    Tries to break on newline boundaries to keep sentences intact.
    Returns an empty list for blank input.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Prefer breaking on a newline near the end of the window
        if end < length:
            break_pos = text.rfind("\n", start + chunk_size // 2, end)
            if break_pos != -1:
                end = break_pos + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(end - overlap, start + 1)  # guard against infinite loop

    return chunks


def _parse_pdf(pdf_path: Path) -> str:
    """
    Parse a PDF with docling and return clean Markdown text.

    docling handles:
    - Digitally-created PDFs (text layer extraction)
    - Scanned PDFs (OCR via Tesseract, built-in)
    - Complex layouts: multi-column, tables, captions
    """
    from docling.document_converter import DocumentConverter

    logger.debug("docling: converting '{path}'", path=pdf_path.name)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, force: bool = False) -> IngestResult:
    """
    Parse, chunk, embed, and store a single PDF.

    Args:
        pdf_path: Absolute path to the .pdf file.
        force:    Re-ingest even if already stored (replaces existing chunks).

    Returns:
        :class:`IngestResult`
    """
    from digital_twin_ui.rag.document_store import get_document_store
    from digital_twin_ui.rag.embedder import get_embedder

    source = pdf_path.name
    store = get_document_store()

    if not force and store.source_exists(source):
        logger.info("Skipping already-ingested PDF: '{source}'", source=source)
        return IngestResult(source=source, n_chunks=0, skipped=True)

    logger.info("Ingesting PDF: '{source}'", source=source)

    # --- Parse ---
    try:
        text = _parse_pdf(pdf_path)
    except Exception as exc:
        logger.error("Parse failed for '{source}': {exc}", source=source, exc=exc)
        return IngestResult(source=source, n_chunks=0, error=f"Parse error: {exc}")

    if not text.strip():
        logger.warning("No text extracted from '{source}'", source=source)
        return IngestResult(source=source, n_chunks=0, error="No text extracted")

    # --- Chunk ---
    chunks = _chunk_text(text)
    if not chunks:
        return IngestResult(source=source, n_chunks=0, error="No chunks produced")

    # --- Remove old chunks on force re-ingest ---
    if force:
        removed = store.delete_source(source)
        if removed:
            logger.info("Removed {n} old chunks for '{source}'", n=removed, source=source)

    # --- Embed ---
    embedder = get_embedder()
    embeddings = embedder.embed(chunks)

    # --- Build IDs and metadata ---
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        digest = hashlib.md5(chunk.encode()).hexdigest()[:8]
        ids.append(f"{source}__chunk_{i}__{digest}")
        metadatas.append({
            "source": source,
            "chunk_index": i,
            "chunk_total": len(chunks),
        })

    # --- Store ---
    store.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    logger.info(
        "Ingested '{source}': {n} chunks stored",
        source=source,
        n=len(chunks),
    )
    return IngestResult(source=source, n_chunks=len(chunks))


def ingest_directory(documents_dir: Path, force: bool = False) -> list[IngestResult]:
    """
    Scan *documents_dir* for ``.pdf`` files and ingest any that are new.

    Existing PDFs are skipped unless ``force=True``.

    Args:
        documents_dir: Directory containing .pdf files.
        force:         Re-ingest all PDFs, replacing existing chunks.

    Returns:
        List of :class:`IngestResult`, one per PDF found (sorted by name).
    """
    pdfs = sorted(documents_dir.glob("*.pdf"))
    if not pdfs:
        logger.info("No PDFs found in '{dir}'", dir=documents_dir)
        return []

    logger.info("Found {n} PDF(s) in '{dir}'", n=len(pdfs), dir=documents_dir)
    return [ingest_pdf(pdf, force=force) for pdf in pdfs]
