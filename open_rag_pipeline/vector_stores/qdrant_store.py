"""
Qdrant vector store implementation.
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_qdrant import Qdrant

from open_rag_pipeline.config import QdrantConfig, VectorStoreConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.exceptions import VectorStoreError
from open_rag_pipeline.utils.logging_utils import get_logger
from open_rag_pipeline.vector_stores.base import VectorStoreInterface

logger = get_logger(__name__)


class QdrantStore(VectorStoreInterface):
    """Qdrant vector store implementation."""

    def __init__(
        self,
        config: VectorStoreConfig,
        embedding: EmbeddingInterface,
        collection_name: str,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            config: Vector store configuration
            embedding: Embedding provider
            collection_name: Collection name
        """
        if not config.qdrant:
            raise VectorStoreError("Qdrant configuration is required")

        self.config = config.qdrant
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
            self.vectorstore = Qdrant(
                embedding_function=EmbeddingWrapper(embedding),
                url=self.config.url,
                api_key=self.config.api_key,
                collection_name=collection_name,
            )
            logger.info(f"Initialized Qdrant vector store: {collection_name}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Qdrant: {str(e)}") from e

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to Qdrant.

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
            raise VectorStoreError(f"Failed to add documents to Qdrant: {str(e)}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search in Qdrant.

        Args:
            query: Query text
            k: Number of results
            filter: Optional metadata filter (Qdrant filter format)

        Returns:
            List of similar documents
        """
        try:
            search_kwargs = {"k": k}
            if filter:
                # Qdrant uses a specific filter format
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                conditions = []
                for key, value in filter.items():
                    if isinstance(value, str):
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value)),
                        )
                    elif isinstance(value, (int, float)):
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value)),
                        )

                if conditions:
                    search_kwargs["filter"] = Filter(must=conditions)

            results = self.vectorstore.similarity_search(query, **search_kwargs)
            return results
        except Exception as e:
            raise VectorStoreError(f"Failed to search Qdrant: {str(e)}") from e

    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents from Qdrant.

        Args:
            ids: List of document IDs

        Returns:
            True if successful
        """
        try:
            self.vectorstore.delete(ids)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Qdrant: {str(e)}") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Qdrant collection statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            # Get collection info from Qdrant client
            client = self.vectorstore.client
            collection_info = client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "num_entities": collection_info.points_count,
            }
        except Exception as e:
            logger.warning(f"Failed to get Qdrant stats: {str(e)}")
            return {"collection_name": self.collection_name, "num_entities": "unknown"}

    def list_collections(self) -> List[str]:
        """
        List all collections in Qdrant.

        Returns:
            List of collection names
        """
        try:
            client = self.vectorstore.client
            collections = client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            raise VectorStoreError(f"Failed to list Qdrant collections: {str(e)}") from e

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful
        """
        try:
            client = self.vectorstore.client
            client.delete_collection(collection_name)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete Qdrant collection {collection_name}: {str(e)}") from e

