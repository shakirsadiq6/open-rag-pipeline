"""
Base loader interface for document loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from open_rag_pipeline.exceptions import DocumentLoaderError
from open_rag_pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    All document loaders must implement the load method which returns
    a list of LangChain Document objects.
    """

    def __init__(self, file_path: Path):
        """
        Initialize the loader.

        Args:
            file_path: Path to the file to load

        Raises:
            DocumentLoaderError: If file does not exist
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise DocumentLoaderError(f"File not found: {file_path}")
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load documents from the file.

        Returns:
            List of Document objects

        Raises:
            DocumentLoaderError: If loading fails
        """
        pass

    def get_file_extension(self) -> str:
        """
        Get the file extension.

        Returns:
            File extension (e.g., '.pdf', '.txt')
        """
        return self.file_path.suffix.lower()

