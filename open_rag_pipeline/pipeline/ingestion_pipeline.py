"""
Document ingestion pipeline.

Orchestrates loading, chunking, embedding, and storage of documents.
"""

from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from tqdm import tqdm

from open_rag_pipeline.chunking.recursive_splitter import RecursiveTextSplitter
from open_rag_pipeline.chunking.config import ChunkingConfig
from open_rag_pipeline.config import PipelineConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.embeddings.factory import create_embedding
from open_rag_pipeline.exceptions import PipelineError
from open_rag_pipeline.loaders.factory import get_loader, get_supported_extensions
from open_rag_pipeline.utils.file_utils import get_files_from_directory
from open_rag_pipeline.utils.logging_utils import get_logger
from open_rag_pipeline.vector_stores.base import VectorStoreInterface
from open_rag_pipeline.vector_stores.factory import create_vector_store
from open_rag_pipeline.pipeline.registry import HashRegistry

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Pipeline for ingesting documents into a vector store.

    Handles the full workflow: loading → chunking → embedding → storage
    """

    def __init__(
        self,
        config: PipelineConfig,
        collection_name: str,
        chunking_config: Optional[ChunkingConfig] = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            config: Pipeline configuration
            collection_name: Name of the vector store collection
            chunking_config: Optional custom chunking configuration
        """
        self.config = config
        self.collection_name = collection_name

        # Initialize embedding provider
        self.embedding = create_embedding(config.embedding)
        logger.info(f"Initialized embedding provider: {config.embedding.provider}")

        # Initialize vector store
        self.vector_store = create_vector_store(
            config.vector_store,
            self.embedding,
            collection_name,
        )
        logger.info(f"Initialized vector store: {config.vector_store.store_type}")

        # Initialize chunker
        chunk_config = chunking_config or ChunkingConfig(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            separators=config.chunking.separators,
            preserve_metadata=config.chunking.preserve_metadata,
        )
        self.chunker = RecursiveTextSplitter(chunk_config)
        logger.info("Initialized text chunker")

        # Initialize hash registry for incremental ingestion
        self.registry = HashRegistry()

    def ingest_file(self, file_path: Path, force: bool = False) -> Dict[str, int]:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file
            force: Whether to force ingestion even if the file hasn't changed

        Returns:
            Dictionary with ingestion statistics

        Raises:
            PipelineError: If ingestion fails
        """
        try:
            # Check if file has changed
            if not force and not self.registry.has_changed(file_path):
                logger.info(f"Skipping unchanged file: {file_path}")
                return {
                    "documents_loaded": 0,
                    "chunks_created": 0,
                    "chunks_stored": 0,
                    "skipped": True,
                }

            # Load document
            loader = get_loader(file_path)
            documents = loader.load()
            logger.debug(f"Loaded {len(documents)} document(s) from {file_path}")

            # Chunk documents
            chunked_docs = self.chunker.chunk_documents(documents)
            logger.debug(f"Created {len(chunked_docs)} chunk(s) from {file_path}")

            # Add to vector store
            ids = self.vector_store.add_documents(chunked_docs)
            logger.debug(f"Added {len(ids)} chunk(s) to vector store")

            # Update registry
            self.registry.update(file_path)

            return {
                "documents_loaded": len(documents),
                "chunks_created": len(chunked_docs),
                "chunks_stored": len(ids),
                "skipped": False,
            }
        except Exception as e:
            raise PipelineError(f"Failed to ingest file {file_path}: {str(e)}") from e

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        show_progress: bool = True,
        force: bool = False,
    ) -> Dict[str, any]:
        """
        Ingest all files from a directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively
            show_progress: Whether to show progress bar
            force: Whether to force ingestion for all files

        Returns:
            Dictionary with ingestion statistics

        Raises:
            PipelineError: If ingestion fails
        """
        # Get all supported files
        supported_extensions = get_supported_extensions()
        files = get_files_from_directory(
            directory,
            recursive=recursive,
            supported_extensions=supported_extensions,
        )

        if not files:
            logger.warning(f"No supported files found in {directory}")
            return {
                "files_processed": 0,
                "documents_loaded": 0,
                "chunks_created": 0,
                "chunks_stored": 0,
                "skipped": 0,
                "errors": [],
            }

        logger.info(f"Found {len(files)} file(s) to process")

        # Process files
        total_stats = {
            "files_processed": 0,
            "documents_loaded": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "skipped": 0,
            "errors": [],
        }

        iterator = tqdm(files, desc="Ingesting files") if show_progress else files

        for file_path in iterator:
            try:
                stats = self.ingest_file(file_path, force=force)
                if stats.get("skipped"):
                    total_stats["skipped"] += 1
                else:
                    total_stats["files_processed"] += 1
                    total_stats["documents_loaded"] += stats["documents_loaded"]
                    total_stats["chunks_created"] += stats["chunks_created"]
                    total_stats["chunks_stored"] += stats["chunks_stored"]
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                total_stats["errors"].append(error_msg)

        logger.info(f"Ingestion complete: {total_stats}")
        return total_stats

    def ingest_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """
        Ingest a list of documents directly.

        Args:
            documents: List of documents to ingest
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with ingestion statistics
        """
        try:
            # Chunk documents
            chunked_docs = self.chunker.chunk_documents(documents)
            logger.debug(f"Created {len(chunked_docs)} chunk(s) from {len(documents)} document(s)")

            # Add to vector store
            ids = self.vector_store.add_documents(chunked_docs)
            logger.debug(f"Added {len(ids)} chunk(s) to vector store")

            return {
                "documents_loaded": len(documents),
                "chunks_created": len(chunked_docs),
                "chunks_stored": len(ids),
            }
        except Exception as e:
            raise PipelineError(f"Failed to ingest documents: {str(e)}") from e

