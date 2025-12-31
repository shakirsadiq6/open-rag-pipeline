"""
Factory for creating document loaders based on file type.
"""

from pathlib import Path
from typing import Dict, List, Type

from open_rag_pipeline.exceptions import DocumentLoaderError
from open_rag_pipeline.loaders.base import BaseLoader
from open_rag_pipeline.loaders.docx_loader import DOCXLoader
from open_rag_pipeline.loaders.html_loader import HTMLLoader
from open_rag_pipeline.loaders.pdf_loader import PDFLoader
from open_rag_pipeline.loaders.txt_loader import TXTLoader

# Mapping of file extensions to loader classes
LOADER_REGISTRY: Dict[str, Type[BaseLoader]] = {
    ".pdf": PDFLoader,
    ".txt": TXTLoader,
    ".text": TXTLoader,
    ".docx": DOCXLoader,
    ".doc": DOCXLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
}


def get_loader(file_path: Path) -> BaseLoader:
    """
    Get the appropriate loader for a file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Loader instance for the file type

    Raises:
        DocumentLoaderError: If file type is not supported
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension not in LOADER_REGISTRY:
        supported = ", ".join(LOADER_REGISTRY.keys())
        raise DocumentLoaderError(
            f"Unsupported file type: {extension}. Supported types: {supported}",
        )

    loader_class = LOADER_REGISTRY[extension]
    return loader_class(file_path)


def get_supported_extensions() -> list[str]:
    """
    Get list of supported file extensions.

    Returns:
        List of supported file extensions
    """
    return list(LOADER_REGISTRY.keys())

