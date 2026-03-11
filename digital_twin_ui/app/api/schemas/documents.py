"""
Pydantic request / response schemas for the RAG document endpoints.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Request body for semantic document search."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question or keyword string.",
        examples=["What is the Young's modulus of the catheter material?"],
    )
    n_results: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of document chunks to return.",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class IngestResultItem(BaseModel):
    """Outcome of ingesting one PDF."""

    source: str = Field(..., description="PDF filename.")
    n_chunks: int = Field(..., description="Number of chunks stored.")
    skipped: bool = Field(False, description="True if file was already indexed.")
    error: Optional[str] = Field(None, description="Error message if ingestion failed.")


class IngestResponse(BaseModel):
    """Summary of a bulk ingest operation."""

    n_ingested: int = Field(..., description="PDFs newly ingested.")
    n_skipped: int = Field(..., description="PDFs skipped (already indexed).")
    n_failed: int = Field(..., description="PDFs that failed to ingest.")
    results: list[IngestResultItem] = Field(..., description="Per-file breakdown.")


class DocumentListResponse(BaseModel):
    """List of all indexed source documents."""

    sources: list[str] = Field(
        ...,
        description="Sorted list of PDF filenames currently in the index.",
    )
    total_chunks: int = Field(
        ...,
        description="Total number of text chunks stored across all documents.",
    )


class SearchHitSchema(BaseModel):
    """One retrieved text chunk with its provenance and relevance score."""

    text: str = Field(..., description="Raw chunk text.")
    source: str = Field(..., description="Source PDF filename.")
    chunk_index: int = Field(..., description="Position of this chunk within its source.")
    score: float = Field(..., description="Cosine similarity score in [0, 1].")


class SearchResponse(BaseModel):
    """Semantic search results."""

    query: str = Field(..., description="The original query string.")
    hits: list[SearchHitSchema] = Field(..., description="Retrieved chunks, best first.")
    total_hits: int = Field(..., description="Number of hits returned.")
