"""
Configuration management for Open RAG Pipeline.

Uses Pydantic models for validation and environment variables for configuration.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from environs import Env
from pydantic import BaseModel, ConfigDict, Field, field_validator

from open_rag_pipeline.exceptions import ConfigurationError


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    COHERE = "cohere"
    LOCAL = "local"


class VectorStoreType(str, Enum):
    """Supported vector store types."""

    MILVUS = "milvus"
    CHROMA = "chroma"
    QDRANT = "qdrant"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding provider."""

    provider: EmbeddingProvider = Field(default=EmbeddingProvider.OPENAI)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, alias="COHERE_API_KEY")
    local_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="LOCAL_EMBEDDING_MODEL",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate OpenAI API key is provided when using OpenAI provider."""
        if info.data.get("provider") == EmbeddingProvider.OPENAI and not v:
            raise ConfigurationError("OPENAI_API_KEY is required when using OpenAI provider")
        return v

    @field_validator("cohere_api_key")
    @classmethod
    def validate_cohere_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate Cohere API key is provided when using Cohere provider."""
        if info.data.get("provider") == EmbeddingProvider.COHERE and not v:
            raise ConfigurationError("COHERE_API_KEY is required when using Cohere provider")
        return v

    model_config = ConfigDict(populate_by_name=True)


class MilvusConfig(BaseModel):
    """Configuration for Milvus vector store."""

    uri: str = Field(default="./milvus_demo.db", alias="MILVUS_URI")
    token: Optional[str] = Field(default=None, alias="MILVUS_TOKEN")
    collection_name: Optional[str] = Field(default=None, alias="MILVUS_COLLECTION_NAME")

    model_config = ConfigDict(populate_by_name=True)


class ChromaConfig(BaseModel):
    """Configuration for Chroma vector store."""

    persist_directory: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    collection_name: Optional[str] = Field(default=None)

    model_config = ConfigDict(populate_by_name=True)


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store."""

    url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    collection_name: Optional[str] = Field(default=None)

    model_config = ConfigDict(populate_by_name=True)


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    store_type: VectorStoreType = Field(default=VectorStoreType.MILVUS, alias="VECTOR_STORE")
    milvus: Optional[MilvusConfig] = None
    chroma: Optional[ChromaConfig] = None
    qdrant: Optional[QdrantConfig] = None

    model_config = ConfigDict(populate_by_name=True)


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = Field(default=1000, alias="DEFAULT_CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="DEFAULT_CHUNK_OVERLAP")
    separators: Optional[list[str]] = None
    preserve_metadata: bool = True

    model_config = ConfigDict(populate_by_name=True)


class BatchProcessingConfig(BaseModel):
    """Configuration for batch processing."""

    batch_size: int = Field(default=100, alias="DEFAULT_BATCH_SIZE")
    max_workers: Optional[int] = None
    enable_parallel: bool = False

    model_config = ConfigDict(populate_by_name=True)


class PipelineConfig(BaseModel):
    """Main configuration for the Open RAG Pipeline."""

    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    batch_processing: BatchProcessingConfig

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "PipelineConfig":
        """
        Create configuration from environment variables.

        Args:
            env_file: Optional path to .env file

        Returns:
            PipelineConfig instance
        """
        env = Env()
        if env_file and env_file.exists():
            env.read_env(env_file)
        else:
            # Try to find .env file in current directory or parent
            current_dir = Path.cwd()
            env_files = [
                current_dir / ".env",
                current_dir.parent / ".env",
                Path(__file__).parent.parent / ".env",
            ]
            for env_path in env_files:
                if env_path.exists():
                    env.read_env(env_path)
                    break

        # Load embedding config
        embedding_provider = env.str("EMBEDDING_PROVIDER", "openai")
        embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider(embedding_provider.lower()),
            OPENAI_API_KEY=env.str("OPENAI_API_KEY", None),
            COHERE_API_KEY=env.str("COHERE_API_KEY", None),
            LOCAL_EMBEDDING_MODEL=env.str("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        )

        # Load vector store config
        vector_store_type = env.str("VECTOR_STORE", "milvus")
        vector_store_config = VectorStoreConfig(
            store_type=VectorStoreType(vector_store_type.lower()),
        )

        # Load Milvus config
        if vector_store_type.lower() == "milvus":
            vector_store_config.milvus = MilvusConfig(
                MILVUS_URI=env.str("MILVUS_URI", "./milvus_demo.db"),
                MILVUS_TOKEN=env.str("MILVUS_TOKEN", None),
                MILVUS_COLLECTION_NAME=env.str("MILVUS_COLLECTION_NAME", None),
            )

        # Load Chroma config
        if vector_store_type.lower() == "chroma":
            vector_store_config.chroma = ChromaConfig(
                CHROMA_PERSIST_DIRECTORY=env.str("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
            )

        # Load Qdrant config
        if vector_store_type.lower() == "qdrant":
            vector_store_config.qdrant = QdrantConfig(
                QDRANT_URL=env.str("QDRANT_URL", "http://localhost:6333"),
                QDRANT_API_KEY=env.str("QDRANT_API_KEY", None),
            )

        # Load chunking config
        chunking_config = ChunkingConfig(
            DEFAULT_CHUNK_SIZE=env.int("DEFAULT_CHUNK_SIZE", 1000),
            DEFAULT_CHUNK_OVERLAP=env.int("DEFAULT_CHUNK_OVERLAP", 200),
        )

        # Load batch processing config
        batch_config = BatchProcessingConfig(
            DEFAULT_BATCH_SIZE=env.int("DEFAULT_BATCH_SIZE", 100),
        )

        return cls(
            embedding=embedding_config,
            vector_store=vector_store_config,
            chunking=chunking_config,
            batch_processing=batch_config,
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "embedding": self.embedding.model_dump(),
            "vector_store": self.vector_store.model_dump(),
            "chunking": self.chunking.model_dump(),
            "batch_processing": self.batch_processing.model_dump(),
        }

