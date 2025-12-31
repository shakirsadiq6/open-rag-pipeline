"""
Unit tests for embedding providers.
"""

from unittest import TestCase
from unittest.mock import MagicMock, patch

from open_rag_pipeline.config import EmbeddingConfig, EmbeddingProvider
from open_rag_pipeline.embeddings.factory import create_embedding
from open_rag_pipeline.exceptions import ConfigurationError, EmbeddingError


class TestEmbeddings(TestCase):
    """Test cases for embedding providers."""

    def test_create_embedding_invalid_provider(self):
        """Test creating embedding with invalid provider."""
        # Pydantic will validate the enum, so we need to bypass that
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            EmbeddingConfig(provider="invalid")

    @patch("open_rag_pipeline.embeddings.openai_embedding.OpenAIEmbeddings")
    def test_openai_embedding(self, mock_openai):
        """Test OpenAI embedding provider."""
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1] * 1536]
        mock_instance.embed_query.return_value = [0.1] * 1536
        mock_openai.return_value = mock_instance

        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            openai_api_key="test-key",
        )

        embedding = create_embedding(config)
        self.assertEqual(embedding.get_dimension(), 1536)

    def test_openai_embedding_missing_key(self):
        """Test OpenAI embedding without API key."""
        # The validation happens at config creation, not at embedding creation
        from open_rag_pipeline.exceptions import ConfigurationError
        with self.assertRaises(ConfigurationError):
            EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                openai_api_key=None,
            )

