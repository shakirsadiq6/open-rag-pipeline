"""
Chunking configuration.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    separators: Optional[List[str]] = Field(
        default=None,
        description="Custom separators for splitting (defaults to LangChain defaults)",
    )
    preserve_metadata: bool = Field(
        default=True,
        description="Whether to preserve document metadata in chunks",
    )
