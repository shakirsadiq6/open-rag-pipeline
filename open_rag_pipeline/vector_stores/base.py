"""
Base vector store interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from open_rag_pipeline.exceptions import VectorStoreError


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings

        Returns:
            List of document IDs

        Raises:
            VectorStoreError: If adding documents fails
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar documents

        Raises:
            VectorStoreError: If search fails
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful

        Raises:
            VectorStoreError: If deletion fails
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary with statistics (count, etc.)
        """
        pass

