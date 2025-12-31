"""
Metadata extraction utilities.
"""

from typing import Any, Dict
from pathlib import Path


def extract_file_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file metadata
    """
    metadata = {
        "source": str(file_path),
        "file_name": file_path.name,
        "file_extension": file_path.suffix.lower(),
        "file_size": file_path.stat().st_size if file_path.exists() else 0,
    }

    return metadata

