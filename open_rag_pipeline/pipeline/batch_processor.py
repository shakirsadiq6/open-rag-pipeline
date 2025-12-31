"""
Batch processor for scalable processing of large datasets.

Handles batch processing with memory-efficient streaming and progress tracking.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional

from langchain_core.documents import Document
from tqdm import tqdm

from open_rag_pipeline.chunking.recursive_splitter import RecursiveTextSplitter
from open_rag_pipeline.chunking.config import ChunkingConfig
from open_rag_pipeline.config import BatchProcessingConfig, PipelineConfig
from open_rag_pipeline.embeddings.base import EmbeddingInterface
from open_rag_pipeline.exceptions import PipelineError
from open_rag_pipeline.loaders.factory import get_loader, get_supported_extensions
from open_rag_pipeline.pipeline.ingestion_pipeline import IngestionPipeline
from open_rag_pipeline.utils.file_utils import get_files_from_directory
from open_rag_pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """
    Batch processor for handling large datasets efficiently.

    Processes files in batches to manage memory and provide progress tracking.
    """

    def __init__(
        self,
        pipeline: IngestionPipeline,
        batch_config: Optional[BatchProcessingConfig] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            pipeline: Ingestion pipeline instance
            batch_config: Optional batch processing configuration
        """
        self.pipeline = pipeline
        self.batch_config = batch_config or pipeline.config.batch_processing
        logger.info(f"Initialized batch processor with batch size: {self.batch_config.batch_size}")

    def process_files_in_batches(
        self,
        files: List[Path],
        show_progress: bool = True,
        force: bool = False,
    ) -> Dict[str, any]:
        """
        Process files in batches.

        Args:
            files: List of file paths to process
            show_progress: Whether to show progress bar
            force: Whether to force ingestion even if files haven't changed

        Returns:
            Dictionary with processing statistics
        """
        total_stats = {
            "files_processed": 0,
            "documents_loaded": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "skipped": 0,
            "errors": [],
        }

        # Process files in batches
        batch_size = self.batch_config.batch_size
        num_batches = (len(files) + batch_size - 1) // batch_size

        iterator = (
            tqdm(range(num_batches), desc="Processing batches")
            if show_progress
            else range(num_batches)
        )

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(files))
            batch_files = files[start_idx:end_idx]

            logger.debug(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)")

            for file_path in batch_files:
                try:
                    stats = self.pipeline.ingest_file(file_path, force=force)
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

        return total_stats

    def process_directory_in_batches(
        self,
        directory: Path,
        recursive: bool = True,
        show_progress: bool = True,
        force: bool = False,
    ) -> Dict[str, any]:
        """
        Process directory files in batches.

        Args:
            directory: Directory path
            recursive: Whether to search recursively
            show_progress: Whether to show progress bar
            force: Whether to force ingestion even if files haven't changed

        Returns:
            Dictionary with processing statistics
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

        logger.info(f"Found {len(files)} file(s) to process in batches")
        return self.process_files_in_batches(files, show_progress=show_progress, force=force)

    def process_documents_in_batches(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """
        Process documents in batches.

        Args:
            documents: List of documents to process
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with processing statistics
        """
        total_stats = {
            "documents_loaded": len(documents),
            "chunks_created": 0,
            "chunks_stored": 0,
        }

        # Process documents in batches
        batch_size = self.batch_config.batch_size
        num_batches = (len(documents) + batch_size - 1) // batch_size

        iterator = (
            tqdm(range(num_batches), desc="Processing document batches")
            if show_progress
            else range(num_batches)
        )

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]

            try:
                stats = self.pipeline.ingest_documents(batch_docs, show_progress=False)
                total_stats["chunks_created"] += stats["chunks_created"]
                total_stats["chunks_stored"] += stats["chunks_stored"]
            except Exception as e:
                logger.error(f"Error processing document batch {batch_idx + 1}: {str(e)}")
                raise PipelineError(f"Failed to process document batch: {str(e)}") from e

        return total_stats

