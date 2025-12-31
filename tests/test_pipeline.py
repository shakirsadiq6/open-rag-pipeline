"""
Integration tests for pipelines.
"""

from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from open_rag_pipeline.config import PipelineConfig


class TestPipeline(TestCase):
    """Test cases for pipelines."""

    @patch("open_rag_pipeline.pipeline.ingestion_pipeline.create_embedding")
    @patch("open_rag_pipeline.pipeline.ingestion_pipeline.create_vector_store")
    def test_ingestion_pipeline_init(self, mock_vector_store, mock_embedding):
        """Test ingestion pipeline initialization."""
        mock_embedding.return_value = MagicMock()
        mock_vector_store.return_value = MagicMock()

        # Create minimal config for testing
        from open_rag_pipeline.config import (
            BatchProcessingConfig,
            ChunkingConfig,
            EmbeddingConfig,
            EmbeddingProvider,
            PipelineConfig,
            VectorStoreConfig,
            VectorStoreType,
        )

        config = PipelineConfig(
            embedding=EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                openai_api_key="test-key",
            ),
            vector_store=VectorStoreConfig(store_type=VectorStoreType.MILVUS),
            chunking=ChunkingConfig(),
            batch_processing=BatchProcessingConfig(),
        )

        from open_rag_pipeline.pipeline.ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline(config, "test_collection")
        self.assertIsNotNone(pipeline)

