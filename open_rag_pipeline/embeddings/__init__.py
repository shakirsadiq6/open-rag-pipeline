"""Embedding providers."""

from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.embeddings.factory import create_embedding

__all__ = ["EmbeddingInterface", "create_embedding"]

