"""
RAG (Retrieval-Augmented Generation) pipeline for research documents.

Stack — all Apache 2.0 / MIT, commercial use permitted:
  - docling        (MIT)  : PDF parsing + OCR
  - sentence-transformers (Apache 2.0) : local embeddings (BAAI/bge-small-en-v1.5, MIT)
  - chromadb       (Apache 2.0) : persistent vector store

Entry points:
  from digital_twin_ui.rag.ingestor  import ingest_directory
  from digital_twin_ui.rag.searcher  import search
  from digital_twin_ui.rag.document_store import get_document_store
  from digital_twin_ui.rag.embedder  import get_embedder
"""
