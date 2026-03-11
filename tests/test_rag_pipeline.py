"""
Unit tests for the RAG pipeline (ingestor, searcher, document_store).

All external dependencies (docling, sentence-transformers, chromadb) are
mocked so tests run fast without GPU, large models, or a real vector store.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_store(count: int = 0, sources: list[str] | None = None):
    """Return a MagicMock that mimics DocumentStore."""
    store = MagicMock()
    store.count.return_value = count
    store.list_sources.return_value = sources or []
    store.source_exists.return_value = False
    store.delete_source.return_value = 0
    store.add.return_value = None
    store.query.return_value = {
        "documents": [["chunk text one", "chunk text two"]],
        "metadatas": [
            [
                {"source": "doc.pdf", "chunk_index": 0},
                {"source": "doc.pdf", "chunk_index": 1},
            ]
        ],
        "distances": [[0.1, 0.3]],
    }
    return store


def _make_mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    return embedder


# ===========================================================================
# _chunk_text (pure function — no mocks needed)
# ===========================================================================

class TestChunkText:
    from digital_twin_ui.rag.ingestor import _chunk_text

    def test_empty_string_returns_empty(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        assert _chunk_text("") == []

    def test_blank_string_returns_empty(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        assert _chunk_text("   \n  ") == []

    def test_short_text_returns_one_chunk(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        chunks = _chunk_text("Hello world.", chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_long_text_splits_into_multiple_chunks(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        text = "A" * 2000
        chunks = _chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) > 1

    def test_all_chunks_non_empty(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        text = "word " * 1000
        chunks = _chunk_text(text, chunk_size=200, overlap=20)
        assert all(c.strip() for c in chunks)

    def test_chunk_size_respected(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        text = "x" * 10000
        chunks = _chunk_text(text, chunk_size=300, overlap=30)
        # Allow some flexibility for the overlap
        for c in chunks[:-1]:
            assert len(c) <= 350, f"Chunk too long: {len(c)}"

    def test_overlap_carries_content(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        # With large overlap, consecutive chunks should share some content
        text = "word " * 500
        chunks = _chunk_text(text, chunk_size=100, overlap=50)
        if len(chunks) >= 2:
            # Check there's content overlap (not zero-length)
            assert len(chunks[1]) > 0

    def test_exact_chunk_size_text_is_one_chunk(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        text = "a" * 512
        chunks = _chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1

    def test_preserves_newline_breaks(self):
        from digital_twin_ui.rag.ingestor import _chunk_text
        # Text with newlines: chunker should prefer breaking there
        text = "paragraph one\n\n" * 100
        chunks = _chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 0


# ===========================================================================
# ingest_pdf
# ===========================================================================

class TestIngestPdf:
    # Patch at the source module so local imports pick up the mock
    _STORE_PATCH = "digital_twin_ui.rag.document_store.get_document_store"
    _EMBED_PATCH = "digital_twin_ui.rag.embedder.get_embedder"
    _PARSE_PATCH = "digital_twin_ui.rag.ingestor._parse_pdf"

    def test_skip_already_ingested(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_pdf

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_store = _make_mock_store()
        mock_store.source_exists.return_value = True

        with patch(self._STORE_PATCH, return_value=mock_store):
            result = ingest_pdf(pdf, force=False)

        assert result.skipped is True
        assert result.n_chunks == 0
        mock_store.add.assert_not_called()

    def test_force_reingest_deletes_old_chunks(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_pdf

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_store = _make_mock_store()
        mock_store.source_exists.return_value = True
        mock_store.delete_source.return_value = 5
        mock_embedder = _make_mock_embedder()
        mock_embedder.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._EMBED_PATCH, return_value=mock_embedder),
            patch(self._PARSE_PATCH, return_value="chunk a\n\nchunk b\n\nchunk c"),
        ):
            result = ingest_pdf(pdf, force=True)

        mock_store.delete_source.assert_called_once_with("doc.pdf")
        assert result.skipped is False

    def test_parse_error_returns_error_result(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_pdf

        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"not a real pdf")

        mock_store = _make_mock_store()

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._PARSE_PATCH, side_effect=RuntimeError("corrupt")),
        ):
            result = ingest_pdf(pdf)

        assert result.error is not None
        assert "corrupt" in result.error
        assert result.n_chunks == 0

    def test_empty_text_returns_error_result(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_pdf

        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_store = _make_mock_store()

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._PARSE_PATCH, return_value="   "),
        ):
            result = ingest_pdf(pdf)

        assert result.error is not None
        assert result.n_chunks == 0

    def test_successful_ingest_stores_chunks(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_pdf

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        text = "Section one content.\n\nSection two content.\n\nSection three content."
        mock_store = _make_mock_store()
        mock_embedder = _make_mock_embedder()
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 3] * len(texts)

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._EMBED_PATCH, return_value=mock_embedder),
            patch(self._PARSE_PATCH, return_value=text),
        ):
            result = ingest_pdf(pdf)

        assert result.error is None
        assert result.skipped is False
        assert result.n_chunks > 0
        mock_store.add.assert_called_once()
        call_kwargs = mock_store.add.call_args.kwargs
        assert len(call_kwargs["ids"]) == result.n_chunks
        assert len(call_kwargs["embeddings"]) == result.n_chunks
        assert len(call_kwargs["documents"]) == result.n_chunks

    def test_chunk_ids_are_unique(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_pdf

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        text = "\n\n".join([f"paragraph {i} with distinct content" for i in range(20)])

        mock_store = _make_mock_store()
        mock_embedder = _make_mock_embedder()
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 3] * len(texts)

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._EMBED_PATCH, return_value=mock_embedder),
            patch(self._PARSE_PATCH, return_value=text),
        ):
            ingest_pdf(pdf)

        ids = mock_store.add.call_args.kwargs["ids"]
        assert len(ids) == len(set(ids)), "Chunk IDs are not unique"

    def test_chunk_metadata_has_required_keys(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_pdf

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        text = "Content here.\n\nMore content here."

        mock_store = _make_mock_store()
        mock_embedder = _make_mock_embedder()
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 3] * len(texts)

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._EMBED_PATCH, return_value=mock_embedder),
            patch(self._PARSE_PATCH, return_value=text),
        ):
            ingest_pdf(pdf)

        metadatas = mock_store.add.call_args.kwargs["metadatas"]
        for meta in metadatas:
            assert "source" in meta
            assert "chunk_index" in meta
            assert "chunk_total" in meta
            assert meta["source"] == "paper.pdf"


# ===========================================================================
# ingest_directory
# ===========================================================================

class TestIngestDirectory:
    def test_empty_directory_returns_empty_list(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_directory
        results = ingest_directory(tmp_path)
        assert results == []

    def test_processes_all_pdfs(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_directory

        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (tmp_path / name).write_bytes(b"%PDF-1.4")

        with patch("digital_twin_ui.rag.ingestor.ingest_pdf") as mock_ingest:
            mock_ingest.side_effect = lambda p, force=False: MagicMock(
                source=p.name, n_chunks=5, skipped=False, error=None
            )
            results = ingest_directory(tmp_path)

        assert len(results) == 3
        assert mock_ingest.call_count == 3

    def test_ignores_non_pdf_files(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_directory

        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "notes.txt").write_text("text")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        with patch("digital_twin_ui.rag.ingestor.ingest_pdf") as mock_ingest:
            mock_ingest.return_value = MagicMock(source="doc.pdf", n_chunks=3)
            results = ingest_directory(tmp_path)

        assert len(results) == 1
        assert mock_ingest.call_count == 1

    def test_passes_force_flag(self, tmp_path):
        from digital_twin_ui.rag.ingestor import ingest_directory

        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")

        with patch("digital_twin_ui.rag.ingestor.ingest_pdf") as mock_ingest:
            mock_ingest.return_value = MagicMock(source="doc.pdf", n_chunks=3)
            ingest_directory(tmp_path, force=True)

        _, kwargs = mock_ingest.call_args
        assert kwargs.get("force") is True or mock_ingest.call_args[0][1] is True


# ===========================================================================
# search
# ===========================================================================

class TestSearch:
    # Patch at the source so local imports inside search() pick up mocks
    _STORE_PATCH = "digital_twin_ui.rag.document_store.get_document_store"
    _EMBED_PATCH = "digital_twin_ui.rag.embedder.get_embedder"

    def test_empty_store_returns_empty_list(self):
        from digital_twin_ui.rag.searcher import search

        mock_store = _make_mock_store(count=0)

        with patch(self._STORE_PATCH, return_value=mock_store):
            hits = search("what is Young's modulus?")

        assert hits == []

    def test_returns_search_hits(self):
        from digital_twin_ui.rag.searcher import search

        mock_store = _make_mock_store(count=10)
        mock_embedder = _make_mock_embedder()

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._EMBED_PATCH, return_value=mock_embedder),
        ):
            hits = search("catheter material properties", n_results=2)

        assert len(hits) == 2
        assert hits[0].text == "chunk text one"
        assert hits[0].source == "doc.pdf"
        assert hits[0].chunk_index == 0

    def test_score_is_cosine_similarity(self):
        from digital_twin_ui.rag.searcher import search

        mock_store = _make_mock_store(count=5)
        mock_embedder = _make_mock_embedder()

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._EMBED_PATCH, return_value=mock_embedder),
        ):
            hits = search("test query")

        for h in hits:
            assert 0.0 <= h.score <= 1.0

    def test_distance_zero_gives_score_one(self):
        from digital_twin_ui.rag.searcher import search

        mock_store = _make_mock_store(count=1)
        mock_store.query.return_value = {
            "documents": [["perfect match"]],
            "metadatas": [[{"source": "doc.pdf", "chunk_index": 0}]],
            "distances": [[0.0]],
        }
        mock_embedder = _make_mock_embedder()

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._EMBED_PATCH, return_value=mock_embedder),
        ):
            hits = search("test")

        assert hits[0].score == pytest.approx(1.0)

    def test_query_is_embedded(self):
        from digital_twin_ui.rag.searcher import search

        mock_store = _make_mock_store(count=5)
        mock_embedder = _make_mock_embedder()

        with (
            patch(self._STORE_PATCH, return_value=mock_store),
            patch(self._EMBED_PATCH, return_value=mock_embedder),
        ):
            search("my specific query")

        mock_embedder.embed.assert_called_once_with(["my specific query"])


# ===========================================================================
# DocumentStore (unit — uses in-memory chromadb)
# ===========================================================================

class TestDocumentStore:
    @pytest.fixture()
    def store(self, tmp_path):
        from digital_twin_ui.rag.document_store import DocumentStore
        return DocumentStore(persist_dir=tmp_path / "chroma")

    def test_initially_empty(self, store):
        assert store.count() == 0

    def test_list_sources_empty(self, store):
        assert store.list_sources() == []

    def test_source_exists_false_when_empty(self, store):
        assert store.source_exists("doc.pdf") is False

    def test_add_and_count(self, store):
        store.add(
            ids=["id1", "id2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["chunk one", "chunk two"],
            metadatas=[
                {"source": "doc.pdf", "chunk_index": 0, "chunk_total": 2},
                {"source": "doc.pdf", "chunk_index": 1, "chunk_total": 2},
            ],
        )
        assert store.count() == 2

    def test_add_and_list_sources(self, store):
        store.add(
            ids=["id1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["chunk one"],
            metadatas=[{"source": "paper_a.pdf", "chunk_index": 0, "chunk_total": 1}],
        )
        assert store.list_sources() == ["paper_a.pdf"]

    def test_source_exists_after_add(self, store):
        store.add(
            ids=["id1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["chunk"],
            metadatas=[{"source": "doc.pdf", "chunk_index": 0, "chunk_total": 1}],
        )
        assert store.source_exists("doc.pdf") is True
        assert store.source_exists("other.pdf") is False

    def test_delete_source(self, store):
        store.add(
            ids=["id1", "id2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["chunk a", "chunk b"],
            metadatas=[
                {"source": "to_delete.pdf", "chunk_index": 0, "chunk_total": 2},
                {"source": "to_delete.pdf", "chunk_index": 1, "chunk_total": 2},
            ],
        )
        assert store.count() == 2
        deleted = store.delete_source("to_delete.pdf")
        assert deleted == 2
        assert store.count() == 0

    def test_query_returns_results(self, store):
        store.add(
            ids=["id1"],
            embeddings=[[1.0, 0.0, 0.0]],
            documents=["catheter material"],
            metadatas=[{"source": "doc.pdf", "chunk_index": 0, "chunk_total": 1}],
        )
        result = store.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
        assert result["documents"][0][0] == "catheter material"

    def test_multiple_sources_listed(self, store):
        store.add(
            ids=["id1", "id2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["a", "b"],
            metadatas=[
                {"source": "paper_a.pdf", "chunk_index": 0, "chunk_total": 1},
                {"source": "paper_b.pdf", "chunk_index": 0, "chunk_total": 1},
            ],
        )
        sources = store.list_sources()
        assert "paper_a.pdf" in sources
        assert "paper_b.pdf" in sources
