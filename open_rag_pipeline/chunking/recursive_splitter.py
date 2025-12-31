"""
Recursive text splitter wrapper using LangChain's RecursiveCharacterTextSplitter.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from open_rag_pipeline.chunking.base import BaseChunker
from open_rag_pipeline.chunking.config import ChunkingConfig
from open_rag_pipeline.exceptions import ChunkingError


class RecursiveTextSplitter(BaseChunker):
    """
    Wrapper around LangChain's RecursiveCharacterTextSplitter.

    Splits text recursively by trying different separators in order of preference.
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize the recursive text splitter.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents with preserved metadata

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            chunked_docs = []
            for doc in documents:
                chunks = self.splitter.split_documents([doc])

                # Preserve metadata if configured
                if self.config.preserve_metadata:
                    for chunk in chunks:
                        # Copy original metadata
                        chunk.metadata.update(doc.metadata)
                        # Add chunk-specific metadata
                        chunk.metadata["chunk_index"] = len(chunked_docs)
                        chunk.metadata["chunk_size"] = len(chunk.page_content)

                chunked_docs.extend(chunks)

            return chunked_docs
        except Exception as e:
            raise ChunkingError(f"Failed to chunk documents: {str(e)}") from e

