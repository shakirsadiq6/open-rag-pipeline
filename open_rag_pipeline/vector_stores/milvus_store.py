"""
Milvus vector store implementation.
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus

from open_rag_pipeline.config import MilvusConfig, VectorStoreConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.exceptions import VectorStoreError
from open_rag_pipeline.utils.logging_utils import get_logger
from open_rag_pipeline.vector_stores.base import VectorStoreInterface

logger = get_logger(__name__)


class MilvusStore(VectorStoreInterface):
    """Milvus vector store implementation."""

    def __init__(
        self,
        config: VectorStoreConfig,
        embedding: EmbeddingInterface,
        collection_name: str,
    ):
        """
        Initialize Milvus vector store.

        Args:
            config: Vector store configuration
            embedding: Embedding provider
            collection_name: Collection name
        """
        if not config.milvus:
            raise VectorStoreError("Milvus configuration is required")

        self.config = config.milvus
        self.embedding = embedding
        self.collection_name = collection_name

        # Build connection args
        connection_args = {"uri": self.config.uri}
        if self.config.token:
            connection_args["token"] = self.config.token

        # Use collection name from config if provided, otherwise use passed name
        final_collection_name = self.config.collection_name or collection_name

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
            self.vectorstore = Milvus(
                embedding_function=EmbeddingWrapper(embedding),
                connection_args=connection_args,
                collection_name=final_collection_name,
            )
            logger.info(f"Initialized Milvus vector store: {final_collection_name}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Milvus: {str(e)}") from e

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to Milvus.

        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings

        Returns:
            List of document IDs
        """
        try:
            # If embeddings are provided, we need to use a different approach
            # For now, let Milvus handle embedding generation
            ids = self.vectorstore.add_documents(documents)
            return ids
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents to Milvus: {str(e)}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search in Milvus.

        Args:
            query: Query text
            k: Number of results
            filter: Optional metadata filter (as Milvus expression string)

        Returns:
            List of similar documents
        """
        try:
            search_kwargs = {"k": k}
            if filter:
                # Convert filter dict to Milvus expression string
                expr = self._build_milvus_expression(filter)
                if expr:
                    search_kwargs["expr"] = expr

            results = self.vectorstore.similarity_search(query, **search_kwargs)
            return results
        except Exception as e:
            raise VectorStoreError(f"Failed to search Milvus: {str(e)}") from e

    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents from Milvus.

        Args:
            ids: List of document IDs

        Returns:
            True if successful
        """
        try:
            self.vectorstore.delete(ids)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Milvus: {str(e)}") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Milvus collection statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            # Get collection info
            collection = self.vectorstore._collection
            stats = {
                "collection_name": self.collection_name,
                "num_entities": collection.num_entities,
            }
            return stats
        except Exception as e:
            logger.warning(f"Failed to get Milvus stats: {str(e)}")
            return {"collection_name": self.collection_name, "num_entities": "unknown"}

    def list_collections(self) -> List[str]:
        """
        List all collections in Milvus.

        Returns:
            List of collection names
        """
        try:
            from pymilvus import utility
            return utility.list_collections()
        except Exception as e:
            raise VectorStoreError(f"Failed to list Milvus collections: {str(e)}") from e

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Milvus.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful
        """
        try:
            from pymilvus import utility
            utility.drop_collection(collection_name)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete Milvus collection {collection_name}: {str(e)}") from e

    def _build_milvus_expression(self, filter: Dict[str, Any]) -> Optional[str]:
        """
        Build Milvus expression string from filter dict.

        Args:
            filter: Filter dictionary

        Returns:
            Milvus expression string or None
        """
        if not filter:
            return None

        expressions = []
        for key, value in filter.items():
            if isinstance(value, str):
                expressions.append(f'{key} == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f"{key} == {value}")
            elif isinstance(value, list):
                # Handle IN operator
                if all(isinstance(v, str) for v in value):
                    values_str = ", ".join(f'"{v}"' for v in value)
                    expressions.append(f"{key} in [{values_str}]")
                else:
                    values_str = ", ".join(str(v) for v in value)
                    expressions.append(f"{key} in [{values_str}]")

        return " and ".join(expressions) if expressions else None

