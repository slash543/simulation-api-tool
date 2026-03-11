"""
Tests for the RAG FastAPI document endpoints.

GET  /api/v1/documents/list
POST /api/v1/documents/ingest
POST /api/v1/documents/search

All RAG components (document_store, ingestor, searcher) are mocked so tests
run without chromadb, sentence-transformers, or docling installed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    """
    TestClient with all RAG singletons mocked.

    The documents directory is set to tmp_path so we control what PDFs exist.
    """
    from digital_twin_ui.app.main import create_app

    # Patch the RAG singletons used by the routes
    mock_store = MagicMock()
    mock_store.count.return_value = 5
    mock_store.list_sources.return_value = ["FEBio Users Manual.pdf"]
    mock_store.source_exists.return_value = False

    # Patch settings so documents dir points to tmp_path
    mock_settings = MagicMock()
    mock_settings.rag_documents_dir_abs = tmp_path
    mock_settings.rag_chroma_dir_abs = tmp_path / "chroma"

    # Create a stub PDF so the ingest route finds files
    (tmp_path / "FEBio Users Manual.pdf").write_bytes(b"%PDF-1.4")

    app = create_app()
    # All RAG functions are locally imported inside route handlers, so we
    # patch at the source module so that local imports pick up the mocks.
    with (
        patch(
            "digital_twin_ui.app.api.routes.documents.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "digital_twin_ui.rag.document_store.get_document_store",
            return_value=mock_store,
        ),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_store, mock_settings, tmp_path


# ---------------------------------------------------------------------------
# GET /api/v1/documents/list
# ---------------------------------------------------------------------------

class TestListDocuments:
    def test_status_200(self, client):
        c, store, *_ = client
        resp = c.get("/api/v1/documents/list")
        assert resp.status_code == 200

    def test_response_has_sources(self, client):
        c, store, *_ = client
        data = c.get("/api/v1/documents/list").json()
        assert "sources" in data

    def test_response_has_total_chunks(self, client):
        c, store, *_ = client
        data = c.get("/api/v1/documents/list").json()
        assert "total_chunks" in data

    def test_sources_match_store(self, client):
        c, store, *_ = client
        store.list_sources.return_value = ["paper_a.pdf", "paper_b.pdf"]
        data = c.get("/api/v1/documents/list").json()
        assert "paper_a.pdf" in data["sources"]
        assert "paper_b.pdf" in data["sources"]

    def test_total_chunks_matches_store_count(self, client):
        c, store, *_ = client
        store.count.return_value = 42
        data = c.get("/api/v1/documents/list").json()
        assert data["total_chunks"] == 42

    def test_empty_store(self, client):
        c, store, *_ = client
        store.count.return_value = 0
        store.list_sources.return_value = []
        data = c.get("/api/v1/documents/list").json()
        assert data["sources"] == []
        assert data["total_chunks"] == 0


# ---------------------------------------------------------------------------
# POST /api/v1/documents/ingest
# ---------------------------------------------------------------------------

class TestIngestDocuments:
    def test_status_200(self, client):
        c, store, settings, tmp_path = client
        from digital_twin_ui.rag.ingestor import IngestResult
        # Patch at source since ingest_directory is locally imported in the route
        with patch(
            "digital_twin_ui.rag.ingestor.ingest_directory",
            return_value=[IngestResult(source="FEBio Users Manual.pdf", n_chunks=120)],
        ):
            resp = c.post("/api/v1/documents/ingest")
        assert resp.status_code == 200

    def test_response_has_counts(self, client):
        c, store, settings, tmp_path = client
        from digital_twin_ui.rag.ingestor import IngestResult
        with patch(
            "digital_twin_ui.rag.ingestor.ingest_directory",
            return_value=[IngestResult(source="FEBio Users Manual.pdf", n_chunks=120)],
        ):
            data = c.post("/api/v1/documents/ingest").json()
        assert "n_ingested" in data
        assert "n_skipped" in data
        assert "n_failed" in data

    def test_ingested_count_correct(self, client):
        c, store, settings, tmp_path = client
        from digital_twin_ui.rag.ingestor import IngestResult
        with patch(
            "digital_twin_ui.rag.ingestor.ingest_directory",
            return_value=[
                IngestResult(source="a.pdf", n_chunks=50),
                IngestResult(source="b.pdf", n_chunks=0, skipped=True),
            ],
        ):
            data = c.post("/api/v1/documents/ingest").json()
        assert data["n_ingested"] == 1
        assert data["n_skipped"] == 1
        assert data["n_failed"] == 0

    def test_failed_count_correct(self, client):
        c, store, settings, tmp_path = client
        from digital_twin_ui.rag.ingestor import IngestResult
        with patch(
            "digital_twin_ui.rag.ingestor.ingest_directory",
            return_value=[
                IngestResult(source="bad.pdf", n_chunks=0, error="corrupt PDF"),
            ],
        ):
            data = c.post("/api/v1/documents/ingest").json()
        assert data["n_failed"] == 1

    def test_results_list_present(self, client):
        c, store, settings, tmp_path = client
        from digital_twin_ui.rag.ingestor import IngestResult
        with patch(
            "digital_twin_ui.rag.ingestor.ingest_directory",
            return_value=[IngestResult(source="doc.pdf", n_chunks=10)],
        ):
            data = c.post("/api/v1/documents/ingest").json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["source"] == "doc.pdf"
        assert data["results"][0]["n_chunks"] == 10

    def test_missing_documents_dir_returns_404(self, client):
        c, store, settings, tmp_path = client
        # Point documents dir to a path that doesn't exist
        settings.rag_documents_dir_abs = tmp_path / "nonexistent_dir"
        resp = c.post("/api/v1/documents/ingest")
        assert resp.status_code == 404

    def test_force_param_passed_through(self, client):
        c, store, settings, tmp_path = client
        with patch(
            "digital_twin_ui.rag.ingestor.ingest_directory"
        ) as mock_ingest:
            mock_ingest.return_value = []
            c.post("/api/v1/documents/ingest?force=true")
            call_args = mock_ingest.call_args
            force_val = call_args.kwargs.get("force") or (
                call_args.args[1] if len(call_args.args) > 1 else None
            )
            assert force_val is True


# ---------------------------------------------------------------------------
# POST /api/v1/documents/search
# ---------------------------------------------------------------------------

class TestSearchDocuments:
    def _post_search(self, client_fixture, query: str = "test query", n_results: int = 5):
        c, *_ = client_fixture
        return c.post(
            "/api/v1/documents/search",
            json={"query": query, "n_results": n_results},
        )

    def _make_hits(self):
        from digital_twin_ui.rag.searcher import SearchHit
        return [
            SearchHit(text="FEBio uses neo-Hookean model", source="FEBio Users Manual.pdf", chunk_index=0, score=0.92),
            SearchHit(text="Young's modulus is defined as...", source="FEBio Users Manual.pdf", chunk_index=5, score=0.87),
        ]

    # search is locally imported in the route handler, so patch at source
    _SEARCH_PATCH = "digital_twin_ui.rag.searcher.search"

    def test_status_200(self, client):
        with patch(self._SEARCH_PATCH, return_value=self._make_hits()):
            resp = self._post_search(client)
        assert resp.status_code == 200

    def test_response_has_hits(self, client):
        with patch(self._SEARCH_PATCH, return_value=self._make_hits()):
            data = self._post_search(client).json()
        assert "hits" in data

    def test_response_echoes_query(self, client):
        with patch(self._SEARCH_PATCH, return_value=self._make_hits()):
            data = self._post_search(client, query="Young's modulus").json()
        assert data["query"] == "Young's modulus"

    def test_hit_fields_present(self, client):
        with patch(self._SEARCH_PATCH, return_value=self._make_hits()):
            data = self._post_search(client).json()
        for hit in data["hits"]:
            assert "text" in hit
            assert "source" in hit
            assert "chunk_index" in hit
            assert "score" in hit

    def test_score_in_range(self, client):
        with patch(self._SEARCH_PATCH, return_value=self._make_hits()):
            data = self._post_search(client).json()
        for hit in data["hits"]:
            assert 0.0 <= hit["score"] <= 1.0

    def test_total_hits_matches_hits_length(self, client):
        with patch(self._SEARCH_PATCH, return_value=self._make_hits()):
            data = self._post_search(client).json()
        assert data["total_hits"] == len(data["hits"])

    def test_empty_store_returns_503(self, client):
        c, store, *_ = client
        store.count.return_value = 0
        resp = self._post_search(client)
        assert resp.status_code == 503

    def test_empty_query_returns_422(self, client):
        c, *_ = client
        resp = c.post("/api/v1/documents/search", json={"query": "", "n_results": 5})
        assert resp.status_code == 422

    def test_n_results_too_large_returns_422(self, client):
        c, *_ = client
        resp = c.post("/api/v1/documents/search", json={"query": "test", "n_results": 100})
        assert resp.status_code == 422

    def test_n_results_zero_returns_422(self, client):
        c, *_ = client
        resp = c.post("/api/v1/documents/search", json={"query": "test", "n_results": 0})
        assert resp.status_code == 422

    def test_search_called_with_correct_args(self, client):
        with patch(self._SEARCH_PATCH) as mock_search:
            mock_search.return_value = self._make_hits()
            self._post_search(client, query="what is neo-Hookean?", n_results=3)
        mock_search.assert_called_once_with("what is neo-Hookean?", n_results=3)
