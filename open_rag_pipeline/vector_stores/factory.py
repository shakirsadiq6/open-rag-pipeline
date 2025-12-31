"""
Factory for creating vector stores.
"""

from open_rag_pipeline.config import VectorStoreConfig, VectorStoreType
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.exceptions import ConfigurationError
from open_rag_pipeline.vector_stores.base import VectorStoreInterface


def create_vector_store(
    config: VectorStoreConfig,
    embedding: EmbeddingInterface,
    collection_name: str,
) -> VectorStoreInterface:
    """
    Create a vector store based on configuration.

    Args:
        config: Vector store configuration
        embedding: Embedding provider
        collection_name: Collection name

    Returns:
        Vector store instance

    Raises:
        ConfigurationError: If vector store type is not supported
    """
    if config.store_type == VectorStoreType.MILVUS:
        try:
            from open_rag_pipeline.vector_stores.milvus_store import MilvusStore
            return MilvusStore(config, embedding, collection_name)
        except ImportError:
            raise ConfigurationError(
                "Milvus vector store requires langchain-milvus and pymilvus. "
                "Install with: pip install -e '.[milvus]'"
            )
    elif config.store_type == VectorStoreType.CHROMA:
        try:
            from open_rag_pipeline.vector_stores.chroma_store import ChromaStore
            return ChromaStore(config, embedding, collection_name)
        except ImportError:
            raise ConfigurationError(
                "Chroma vector store requires langchain-chroma and chromadb. "
                "Install with: pip install -e '.[chroma]'"
            )
    elif config.store_type == VectorStoreType.QDRANT:
        try:
            from open_rag_pipeline.vector_stores.qdrant_store import QdrantStore
            return QdrantStore(config, embedding, collection_name)
        except ImportError:
            raise ConfigurationError(
                "Qdrant vector store requires langchain-qdrant and qdrant-client. "
                "Install with: pip install -e '.[qdrant]'"
            )
    else:
        raise ConfigurationError(f"Unsupported vector store type: {config.store_type}")

