"""
ChromaDB persistent vector store wrapper.

Uses a cosine-similarity HNSW index.  All chunks from a given source PDF
are stored with ``{"source": pdf_filename}`` metadata so they can be
listed, queried, and deleted by source.

Singleton access via ``get_document_store()``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "research_documents"

_store_singleton: DocumentStore | None = None


class DocumentStore:
    """
    Thin wrapper around a persistent ChromaDB collection.

    Args:
        persist_dir: Directory where ChromaDB stores its on-disk index.
    """

    def __init__(self, persist_dir: Path) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "DocumentStore ready — {n} chunks in collection '{col}'",
            n=self._collection.count(),
            col=COLLECTION_NAME,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of chunks stored."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Return sorted list of unique source PDF filenames."""
        if self.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        seen: set[str] = set()
        sources: list[str] = []
        for meta in result.get("metadatas") or []:
            src = (meta or {}).get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        return sorted(sources)

    def source_exists(self, source: str) -> bool:
        """Return True if at least one chunk from *source* is stored."""
        result = self._collection.get(
            where={"source": source},
            limit=1,
            include=["metadatas"],
        )
        return bool(result.get("ids"))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add chunks to the collection (upsert semantics on id collision)."""
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_source(self, source: str) -> int:
        """Delete all chunks for *source*. Returns the number deleted."""
        result = self._collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """
        Return the *n_results* nearest chunks for each query embedding.

        Result keys: ``documents``, ``metadatas``, ``distances``.
        """
        safe_n = min(n_results, max(self.count(), 1))
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=safe_n,
            include=["documents", "metadatas", "distances"],
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_document_store() -> DocumentStore:
    """Return the module-level DocumentStore singleton."""
    global _store_singleton
    if _store_singleton is None:
        from digital_twin_ui.app.core.config import get_settings
        cfg = get_settings()
        _store_singleton = DocumentStore(persist_dir=cfg.rag_chroma_dir_abs)
    return _store_singleton
