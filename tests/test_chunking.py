"""
Unit tests for text chunking.
"""

from unittest import TestCase

from langchain_core.documents import Document

from open_rag_pipeline.chunking.config import ChunkingConfig
from open_rag_pipeline.chunking.recursive_splitter import RecursiveTextSplitter


class TestChunking(TestCase):
    """Test cases for text chunking."""

    def test_chunk_documents(self):
        """Test chunking documents."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = RecursiveTextSplitter(config)

        # Create a long document
        long_text = " ".join(["This is a test sentence."] * 20)
        doc = Document(page_content=long_text, metadata={"source": "test.txt"})

        chunks = chunker.chunk_documents([doc])

        self.assertGreater(len(chunks), 1)
        self.assertIn("source", chunks[0].metadata)
        self.assertEqual(chunks[0].metadata["source"], "test.txt")

    def test_preserve_metadata(self):
        """Test that metadata is preserved in chunks."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10, preserve_metadata=True)
        chunker = RecursiveTextSplitter(config)

        doc = Document(
            page_content="This is a test document with some content.",
            metadata={"source": "test.txt", "author": "Test Author"},
        )

        chunks = chunker.chunk_documents([doc])

        for chunk in chunks:
            self.assertIn("source", chunk.metadata)
            self.assertIn("author", chunk.metadata)
            self.assertEqual(chunk.metadata["author"], "Test Author")

