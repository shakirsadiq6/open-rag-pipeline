"""
File handling utilities.
"""

from pathlib import Path
from typing import List


def get_files_from_directory(
    directory: Path,
    recursive: bool = True,
    supported_extensions: List[str] = None,
) -> List[Path]:
    """
    Get all files from a directory.

    Args:
        directory: Directory path
        recursive: Whether to search recursively
        supported_extensions: List of supported file extensions (e.g., ['.pdf', '.txt'])

    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    files = []
    pattern = "**/*" if recursive else "*"

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if supported_extensions is None:
                files.append(file_path)
            elif file_path.suffix.lower() in supported_extensions:
                files.append(file_path)

    return sorted(files)

