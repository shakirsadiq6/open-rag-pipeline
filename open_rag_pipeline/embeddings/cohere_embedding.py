"""
Cohere embedding provider.
"""

from typing import List

import cohere

from open_rag_pipeline.config import EmbeddingConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.exceptions import EmbeddingError
from open_rag_pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CohereEmbedding(EmbeddingInterface):
    """Cohere embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize Cohere embedding provider.

        Args:
            config: Embedding configuration with Cohere API key
        """
        if not config.cohere_api_key:
            raise EmbeddingError("Cohere API key is required")

        self.config = config
        self.client = cohere.Client(api_key=config.cohere_api_key)
        self.model = "embed-english-v3.0"  # Default model
        logger.info("Initialized Cohere embedding provider")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents using Cohere.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document",
            )
            return response.embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate Cohere embeddings: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query using Cohere.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query",
            )
            return response.embeddings[0]
        except Exception as e:
            raise EmbeddingError(f"Failed to generate Cohere query embedding: {str(e)}") from e

    def get_dimension(self) -> int:
        """
        Get the dimension of Cohere embeddings.

        Returns:
            Embedding dimension (1024 for embed-english-v3.0)
        """
        # embed-english-v3.0 has 1024 dimensions
        return 1024

