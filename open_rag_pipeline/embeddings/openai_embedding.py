"""
OpenAI embedding provider.
"""

from typing import List

from langchain_openai import OpenAIEmbeddings

from open_rag_pipeline.config import EmbeddingConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.exceptions import EmbeddingError
from open_rag_pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OpenAIEmbedding(EmbeddingInterface):
    """OpenAI embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize OpenAI embedding provider.

        Args:
            config: Embedding configuration with OpenAI API key
        """
        if not config.openai_api_key:
            raise EmbeddingError("OpenAI API key is required")

        self.config = config
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.openai_api_key,
            model="text-embedding-3-small",  # Default model
        )
        logger.info("Initialized OpenAI embedding provider")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents using OpenAI.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate OpenAI embeddings: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query using OpenAI.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate OpenAI query embedding: {str(e)}") from e

    def get_dimension(self) -> int:
        """
        Get the dimension of OpenAI embeddings.

        Returns:
            Embedding dimension (1536 for text-embedding-3-small)
        """
        # text-embedding-3-small has 1536 dimensions
        return 1536

