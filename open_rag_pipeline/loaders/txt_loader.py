"""
Text document loader using TextLoader.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from open_rag_pipeline.exceptions import DocumentLoaderError
from open_rag_pipeline.loaders.base import BaseLoader


class TXTLoader(BaseLoader):
    """Loader for plain text files."""

    def load(self) -> List[Document]:
        """
        Load text documents.

        Returns:
            List of Document objects

        Raises:
            DocumentLoaderError: If loading fails
        """
        try:
            loader = TextLoader(str(self.file_path), encoding="utf-8")
            documents = loader.load()

            # Add file metadata to each document
            for doc in documents:
                doc.metadata["source"] = str(self.file_path)
                doc.metadata["file_type"] = "txt"
                doc.metadata["file_name"] = self.file_path.name

            return documents
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                loader = TextLoader(str(self.file_path), encoding="latin-1")
                documents = loader.load()
                for doc in documents:
                    doc.metadata["source"] = str(self.file_path)
                    doc.metadata["file_type"] = "txt"
                    doc.metadata["file_name"] = self.file_path.name
                return documents
            except Exception as e:
                raise DocumentLoaderError(f"Failed to load text file {self.file_path}: {str(e)}") from e
        except Exception as e:
            raise DocumentLoaderError(f"Failed to load text file {self.file_path}: {str(e)}") from e

