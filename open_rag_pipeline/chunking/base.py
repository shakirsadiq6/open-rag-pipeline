"""
Base chunker interface.
"""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

from open_rag_pipeline.exceptions import ChunkingError


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents

        Raises:
            ChunkingError: If chunking fails
        """
        pass

