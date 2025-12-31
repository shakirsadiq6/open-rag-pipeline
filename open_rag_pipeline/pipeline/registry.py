"""
Registry for tracking file hashes to support incremental ingestion.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from open_rag_pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)

class HashRegistry:
    """
    Tracks MD5 hashes of processed files to avoid re-processing unchanged files.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the hash registry.

        Args:
            registry_path: Path to the registry JSON file.
        """
        if registry_path is None:
            registry_path = Path.cwd() / ".open-rag-registry.json"
        
        self.registry_path = registry_path
        self.hashes: Dict[str, str] = self._load_registry()

    def _load_registry(self) -> Dict[str, str]:
        """Load the registry from disk."""
        if not self.registry_path.exists():
            return {}
        
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")
            return {}

    def save(self):
        """Save the registry to disk."""
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate instructions MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def has_changed(self, file_path: Path) -> bool:
        """
        Check if a file has changed since it was last registered.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file is new or has a different hash.
        """
        if not file_path.exists():
            return True
        
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.hashes.get(str(file_path.absolute()))
        
        return current_hash != stored_hash

    def update(self, file_path: Path):
        """
        Update the registered hash for a file.

        Args:
            file_path: Path to the file.
        """
        if file_path.exists():
            self.hashes[str(file_path.absolute())] = self.get_file_hash(file_path)
            self.save()

    def clear(self):
        """Clear the registry."""
        self.hashes = {}
        if self.registry_path.exists():
            self.registry_path.unlink()
