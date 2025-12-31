"""Vector store backends."""

from open_rag_pipeline.vector_stores.base import VectorStoreInterface
from open_rag_pipeline.vector_stores.factory import create_vector_store

__all__ = ["VectorStoreInterface", "create_vector_store"]

