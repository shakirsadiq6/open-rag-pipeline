"""
Query pipeline for similarity search and retrieval.
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from open_rag_pipeline.config import PipelineConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.embeddings.factory import create_embedding
from open_rag_pipeline.exceptions import PipelineError
from open_rag_pipeline.utils.logging_utils import get_logger
from open_rag_pipeline.vector_stores.base import VectorStoreInterface
from open_rag_pipeline.vector_stores.factory import create_vector_store

logger = get_logger(__name__)


class QueryPipeline:
    """
    Pipeline for querying the vector store.

    Handles query embedding and similarity search.
    """

    def __init__(self, config: PipelineConfig, collection_name: str):
        """
        Initialize the query pipeline.

        Args:
            config: Pipeline configuration
            collection_name: Name of the vector store collection
        """
        self.config = config
        self.collection_name = collection_name

        # Initialize embedding provider
        self.embedding = create_embedding(config.embedding)
        logger.info(f"Initialized embedding provider: {config.embedding.provider}")

        # Initialize vector store
        self.vector_store = create_vector_store(
            config.vector_store,
            self.embedding,
            collection_name,
        )
        logger.info(f"Initialized vector store: {config.vector_store.store_type}")

    def search(
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
            PipelineError: If search fails
        """
        try:
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
            logger.debug(f"Found {len(results)} result(s) for query")
            return results
        except Exception as e:
            raise PipelineError(f"Failed to search: {str(e)}") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            raise PipelineError(f"Failed to get stats: {str(e)}") from e

