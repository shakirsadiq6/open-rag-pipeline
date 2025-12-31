"""
HTML document loader using BeautifulSoup.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document

from open_rag_pipeline.exceptions import DocumentLoaderError
from open_rag_pipeline.loaders.base import BaseLoader


class HTMLLoader(BaseLoader):
    """Loader for HTML files."""

    def load(self) -> List[Document]:
        """
        Load HTML documents.

        Returns:
            List of Document objects

        Raises:
            DocumentLoaderError: If loading fails
        """
        try:
            loader = BSHTMLLoader(str(self.file_path))
            documents = loader.load()

            # Add file metadata to each document
            for doc in documents:
                doc.metadata["source"] = str(self.file_path)
                doc.metadata["file_type"] = "html"
                doc.metadata["file_name"] = self.file_path.name

            return documents
        except Exception as e:
            raise DocumentLoaderError(f"Failed to load HTML file {self.file_path}: {str(e)}") from e

