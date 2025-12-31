"""
Custom exceptions for the Open RAG Pipeline.
"""


class OpenRAGPipelineError(Exception):
    """Base exception for all Open RAG Pipeline errors."""

    pass


class ConfigurationError(OpenRAGPipelineError):
    """Raised when there is a configuration error."""

    pass


class DocumentLoaderError(OpenRAGPipelineError):
    """Raised when document loading fails."""

    pass


class EmbeddingError(OpenRAGPipelineError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(OpenRAGPipelineError):
    """Raised when vector store operations fail."""

    pass


class ChunkingError(OpenRAGPipelineError):
    """Raised when text chunking fails."""

    pass


class PipelineError(OpenRAGPipelineError):
    """Raised when pipeline execution fails."""

    pass

