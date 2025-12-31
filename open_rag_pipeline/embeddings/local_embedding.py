"""
Local embedding provider using sentence-transformers.
"""

from typing import List

from sentence_transformers import SentenceTransformer

from open_rag_pipeline.config import EmbeddingConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.exceptions import EmbeddingError
from open_rag_pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LocalEmbedding(EmbeddingInterface):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize local embedding provider.

        Args:
            config: Embedding configuration with model name
        """
        self.config = config
        try:
            self.model = SentenceTransformer(config.local_model_name)
            logger.info(f"Initialized local embedding provider with model: {config.local_model_name}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load sentence-transformers model: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents using local model.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate local embeddings: {str(e)}") from e

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query using local model.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            embedding = self.model.encode([text], show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate local query embedding: {str(e)}") from e

    def get_dimension(self) -> int:
        """
        Get the dimension of local embeddings.

        Returns:
            Embedding dimension (varies by model)
        """
        return self.model.get_sentence_embedding_dimension()

