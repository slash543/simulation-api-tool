"""
Semantic search over ingested research documents.

Embeds the query with the same sentence-transformers model used during
ingestion, then retrieves the top-N nearest chunks from ChromaDB using
cosine similarity.

Usage
-----
    from digital_twin_ui.rag.searcher import search

    hits = search("What is the Young's modulus of the catheter material?", n_results=5)
    for h in hits:
        print(h.score, h.source, h.text[:120])
"""
from __future__ import annotations

from dataclasses import dataclass

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchHit:
    """One retrieved chunk with its provenance and relevance score."""
    text: str
    source: str         # PDF filename (e.g. "FEBio Users Manual.pdf")
    chunk_index: int    # position of this chunk within its source document
    score: float        # cosine similarity in [0, 1] (higher = more relevant)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, n_results: int = 5) -> list[SearchHit]:
    """
    Retrieve the most relevant chunks for *query*.

    Args:
        query:     Natural-language question or keyword string.
        n_results: Maximum number of chunks to return (default 5).

    Returns:
        List of :class:`SearchHit` sorted by relevance (best first).
        Returns an empty list if the store is empty.
    """
    from digital_twin_ui.rag.document_store import get_document_store
    from digital_twin_ui.rag.embedder import get_embedder

    store = get_document_store()
    if store.count() == 0:
        logger.warning("Document store is empty — ingest PDFs first")
        return []

    embedder = get_embedder()
    query_embedding = embedder.embed([query])

    raw = store.query(query_embeddings=query_embedding, n_results=n_results)

    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    hits: list[SearchHit] = []
    for text, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance: 0 = identical, 2 = maximally different
        # Convert to similarity in [0, 1]
        score = round(max(0.0, 1.0 - dist / 2.0), 4)
        hits.append(SearchHit(
            text=text,
            source=(meta or {}).get("source", "unknown"),
            chunk_index=int((meta or {}).get("chunk_index", 0)),
            score=score,
        ))

    logger.debug(
        "Search '{q}' → {n} hits (top score: {s})",
        q=query[:60],
        n=len(hits),
        s=hits[0].score if hits else 0,
    )
    return hits
