"""
Unit tests for document loaders.
"""

import tempfile
from pathlib import Path
from unittest import TestCase

from open_rag_pipeline.exceptions import DocumentLoaderError
from open_rag_pipeline.loaders.factory import get_loader, get_supported_extensions
from open_rag_pipeline.loaders.txt_loader import TXTLoader


class TestLoaders(TestCase):
    """Test cases for document loaders."""

    def test_get_supported_extensions(self):
        """Test getting supported file extensions."""
        extensions = get_supported_extensions()
        self.assertIsInstance(extensions, list)
        self.assertIn(".pdf", extensions)
        self.assertIn(".txt", extensions)
        self.assertIn(".docx", extensions)
        self.assertIn(".html", extensions)

    def test_get_loader_unsupported_file(self):
        """Test getting loader for unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(DocumentLoaderError):
                get_loader(temp_path)
        finally:
            temp_path.unlink()

    def test_txt_loader(self):
        """Test text loader."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("This is a test document.\nIt has multiple lines.")
            temp_path = Path(f.name)

        try:
            loader = TXTLoader(temp_path)
            documents = loader.load()

            self.assertGreater(len(documents), 0)
            self.assertIn("source", documents[0].metadata)
            self.assertEqual(documents[0].metadata["file_type"], "txt")
        finally:
            temp_path.unlink()

    def test_loader_file_not_found(self):
        """Test loader with non-existent file."""
        non_existent = Path("/nonexistent/file.txt")
        with self.assertRaises(DocumentLoaderError):
            TXTLoader(non_existent)

