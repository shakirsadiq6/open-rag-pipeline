"""
Chroma vector store implementation.
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from open_rag_pipeline.config import ChromaConfig, VectorStoreConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.exceptions import VectorStoreError
from open_rag_pipeline.utils.logging_utils import get_logger
from open_rag_pipeline.vector_stores.base import VectorStoreInterface

logger = get_logger(__name__)


class ChromaStore(VectorStoreInterface):
    """Chroma vector store implementation."""

    def __init__(
        self,
        config: VectorStoreConfig,
        embedding: EmbeddingInterface,
        collection_name: str,
    ):
        """
        Initialize Chroma vector store.

        Args:
            config: Vector store configuration
            embedding: Embedding provider
            collection_name: Collection name
        """
        if not config.chroma:
            raise VectorStoreError("Chroma configuration is required")

        self.config = config.chroma
        self.embedding = embedding
        self.collection_name = collection_name

        # Create embedding function wrapper
        from langchain_core.embeddings import Embeddings

        class EmbeddingWrapper(Embeddings):
            """Wrapper to convert our embedding interface to LangChain Embeddings."""

            def __init__(self, embedding_interface: EmbeddingInterface):
                self.embedding_interface = embedding_interface

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.embedding_interface.embed_documents(texts)

            def embed_query(self, text: str) -> List[float]:
                return self.embedding_interface.embed_query(text)

        try:
            self.vectorstore = Chroma(
                embedding_function=EmbeddingWrapper(embedding),
                persist_directory=self.config.persist_directory,
                collection_name=collection_name,
            )
            logger.info(f"Initialized Chroma vector store: {collection_name}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Chroma: {str(e)}") from e

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to Chroma.

        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings

        Returns:
            List of document IDs
        """
        try:
            ids = self.vectorstore.add_documents(documents)
            return ids
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents to Chroma: {str(e)}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search in Chroma.

        Args:
            query: Query text
            k: Number of results
            filter: Optional metadata filter

        Returns:
            List of similar documents
        """
        try:
            search_kwargs = {"k": k}
            if filter:
                search_kwargs["where"] = filter

            results = self.vectorstore.similarity_search(query, **search_kwargs)
            return results
        except Exception as e:
            raise VectorStoreError(f"Failed to search Chroma: {str(e)}") from e

    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents from Chroma.

        Args:
            ids: List of document IDs

        Returns:
            True if successful
        """
        try:
            self.vectorstore.delete(ids)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Chroma: {str(e)}") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Chroma collection statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "collection_name": self.collection_name,
                "num_entities": count,
            }
        except Exception as e:
            logger.warning(f"Failed to get Chroma stats: {str(e)}")
            return {"collection_name": self.collection_name, "num_entities": "unknown"}

