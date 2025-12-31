"""
Unit tests for vector stores.
"""

from unittest import TestCase
from unittest.mock import MagicMock, patch

from open_rag_pipeline.config import (
    ChromaConfig,
    MilvusConfig,
    VectorStoreConfig,
    VectorStoreType,
)
from open_rag_pipeline.exceptions import VectorStoreError


class TestVectorStores(TestCase):
    """Test cases for vector stores."""

    def test_milvus_store_missing_config(self):
        """Test Milvus store without configuration."""
        config = VectorStoreConfig(store_type=VectorStoreType.MILVUS)
        mock_embedding = MagicMock()

        from open_rag_pipeline.vector_stores.factory import create_vector_store
        from open_rag_pipeline.exceptions import ConfigurationError

        # Milvus will fail with ImportError if not installed, which gets converted to ConfigurationError
        with self.assertRaises(ConfigurationError):
            create_vector_store(config, mock_embedding, "test_collection")

    def test_chroma_store_missing_config(self):
        """Test Chroma store without configuration."""
        config = VectorStoreConfig(store_type=VectorStoreType.CHROMA)
        mock_embedding = MagicMock()

        from open_rag_pipeline.vector_stores.factory import create_vector_store
        from open_rag_pipeline.exceptions import ConfigurationError

        # Chroma will fail with ImportError if not installed, which gets converted to ConfigurationError
        with self.assertRaises((VectorStoreError, ConfigurationError, ImportError)):
            create_vector_store(config, mock_embedding, "test_collection")

    def test_qdrant_store_missing_config(self):
        """Test Qdrant store without configuration."""
        config = VectorStoreConfig(store_type=VectorStoreType.QDRANT)
        mock_embedding = MagicMock()

        from open_rag_pipeline.vector_stores.factory import create_vector_store
        from open_rag_pipeline.exceptions import ConfigurationError

        # Qdrant will fail with ImportError if not installed, which gets converted to ConfigurationError
        with self.assertRaises((VectorStoreError, ConfigurationError, ImportError)):
            create_vector_store(config, mock_embedding, "test_collection")

