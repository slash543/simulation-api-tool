"""
Sentence-transformers embedding wrapper.

Model: BAAI/bge-small-en-v1.5
  - License : MIT (commercial use OK)
  - Size    : ~130 MB
  - Runs on CPU (no GPU required)
  - Dim     : 384

The model is downloaded on first use to the HuggingFace cache
(~/.cache/huggingface/hub/).  Subsequent starts load from cache instantly.

Singleton access via ``get_embedder()``.
"""
from __future__ import annotations

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)

_embedder_singleton: Embedder | None = None


class Embedder:
    """
    Wraps a SentenceTransformer model for text → vector conversion.

    Args:
        model_name: HuggingFace model id (default ``BAAI/bge-small-en-v1.5``).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '{model}' …", model=model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Embedding model ready")

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Encode *texts* and return L2-normalised float vectors.

        Normalisation ensures cosine similarity == dot product, which is
        what ChromaDB's ``hnsw:space=cosine`` collection expects.

        Args:
            texts: List of strings to embed (may be 1 item for a query).

        Returns:
            List of float lists, one per input text.
        """
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_embedder() -> Embedder:
    """Return the module-level Embedder singleton."""
    global _embedder_singleton
    if _embedder_singleton is None:
        from digital_twin_ui.app.core.config import get_settings
        cfg = get_settings()
        _embedder_singleton = Embedder(model_name=cfg.rag.embedding_model)
    return _embedder_singleton
