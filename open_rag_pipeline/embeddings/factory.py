"""
Factory for creating embedding providers.
"""

from open_rag_pipeline.config import EmbeddingConfig, EmbeddingProvider
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.embeddings.cohere_embedding import CohereEmbedding
from open_rag_pipeline.embeddings.openai_embedding import OpenAIEmbedding
from open_rag_pipeline.exceptions import ConfigurationError


def create_embedding(config: EmbeddingConfig) -> EmbeddingInterface:
    """
    Create an embedding provider based on configuration.

    Args:
        config: Embedding configuration

    Returns:
        Embedding provider instance

    Raises:
        ConfigurationError: If provider type is not supported
    """
    if config.provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbedding(config)
    elif config.provider == EmbeddingProvider.COHERE:
        return CohereEmbedding(config)
    elif config.provider == EmbeddingProvider.LOCAL:
        # Lazy import to avoid requiring sentence-transformers if not using local embeddings
        try:
            from open_rag_pipeline.embeddings.local_embedding import LocalEmbedding
            return LocalEmbedding(config)
        except ImportError:
            raise ConfigurationError(
                "Local embedding provider requires sentence-transformers. "
                "Install it with: pip install -e '.[local-embeddings]'"
            )
    else:
        raise ConfigurationError(f"Unsupported embedding provider: {config.provider}")

