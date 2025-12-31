"""
CLI commands for Open RAG Pipeline.

Following the existing codebase patterns using Click.
"""

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from open_rag_pipeline.chunking.config import ChunkingConfig
from open_rag_pipeline.config import (
    EmbeddingConfig,
    EmbeddingProvider,
    PipelineConfig,
    VectorStoreConfig,
    VectorStoreType,
)
from open_rag_pipeline.exceptions import ConfigurationError, PipelineError
from open_rag_pipeline.pipeline.batch_processor import BatchProcessor
from open_rag_pipeline.pipeline.ingestion_pipeline import IngestionPipeline
from open_rag_pipeline.pipeline.query_pipeline import QueryPipeline
from open_rag_pipeline.utils.logging_utils import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """Open RAG Pipeline CLI - Scalable document ingestion and vectorization."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)


def _create_config_from_options(
    embedding_provider: str,
    vector_store: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_config_override: Optional[dict] = None,
    vector_store_config_override: Optional[dict] = None,
) -> PipelineConfig:
    """
    Create pipeline configuration from CLI options.

    Args:
        embedding_provider: Embedding provider name
        vector_store: Vector store name
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        embedding_config_override: Optional embedding config overrides
        vector_store_config_override: Optional vector store config overrides

    Returns:
        PipelineConfig instance
    """
    # Start with environment-based config
    try:
        config = PipelineConfig.from_env()
    except Exception:
        # Create minimal config if env loading fails
        from open_rag_pipeline.config import (
            BatchProcessingConfig,
            ChunkingConfig as ConfigChunkingConfig,
            EmbeddingConfig,
            VectorStoreConfig,
        )

        config = PipelineConfig(
            embedding=EmbeddingConfig(provider=EmbeddingProvider.OPENAI),
            vector_store=VectorStoreConfig(store_type=VectorStoreType.MILVUS),
            chunking=ConfigChunkingConfig(),
            batch_processing=BatchProcessingConfig(),
        )

    # Override embedding provider
    try:
        provider = EmbeddingProvider(embedding_provider.lower())
        config.embedding.provider = provider
    except ValueError:
        raise ConfigurationError(f"Invalid embedding provider: {embedding_provider}")

    # Override vector store
    try:
        store_type = VectorStoreType(vector_store.lower())
        config.vector_store.store_type = store_type
    except ValueError:
        raise ConfigurationError(f"Invalid vector store: {vector_store}")

    # Override chunking config
    config.chunking.chunk_size = chunk_size
    config.chunking.chunk_overlap = chunk_overlap

    # Apply overrides
    if embedding_config_override:
        for key, value in embedding_config_override.items():
            setattr(config.embedding, key, value)

    if vector_store_config_override:
        if config.vector_store.store_type == VectorStoreType.MILVUS:
            if not config.vector_store.milvus:
                from open_rag_pipeline.config import MilvusConfig

                config.vector_store.milvus = MilvusConfig()
            for key, value in vector_store_config_override.items():
                setattr(config.vector_store.milvus, key, value)
        elif config.vector_store.store_type == VectorStoreType.CHROMA:
            if not config.vector_store.chroma:
                from open_rag_pipeline.config import ChromaConfig

                config.vector_store.chroma = ChromaConfig()
            for key, value in vector_store_config_override.items():
                setattr(config.vector_store.chroma, key, value)
        elif config.vector_store.store_type == VectorStoreType.QDRANT:
            if not config.vector_store.qdrant:
                from open_rag_pipeline.config import QdrantConfig

                config.vector_store.qdrant = QdrantConfig()
            for key, value in vector_store_config_override.items():
                setattr(config.vector_store.qdrant, key, value)

    return config


@cli.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input file or directory",
)
@click.option("--collection", "-c", required=True, help="Collection name")
@click.option(
    "--embedding-provider",
    default="openai",
    type=click.Choice(["openai", "cohere", "local"]),
    help="Embedding provider",
)
@click.option(
    "--vector-store",
    default="milvus",
    type=click.Choice(["milvus", "chroma", "qdrant"]),
    help="Vector store backend",
)
@click.option("--chunk-size", default=1000, type=int, help="Chunk size")
@click.option("--chunk-overlap", default=200, type=int, help="Chunk overlap")
@click.option("--batch-size", default=100, type=int, help="Batch size for processing")
@click.option("--recursive", is_flag=True, default=True, help="Recursively process directories")
@click.pass_context
def ingest(
    ctx,
    input: Path,
    collection: str,
    embedding_provider: str,
    vector_store: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    recursive: bool,
):
    """Ingest documents into vector store."""
    try:
        console.print(f"[bold green]Starting ingestion...[/bold green]")
        console.print(f"Input: {input}")
        console.print(f"Collection: {collection}")
        console.print(f"Embedding Provider: {embedding_provider}")
        console.print(f"Vector Store: {vector_store}")

        # Create configuration
        config = _create_config_from_options(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        config.batch_processing.batch_size = batch_size

        # Create chunking config
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Initialize pipeline
        pipeline = IngestionPipeline(config, collection, chunking_config)

        # Process input
        if input.is_file():
            console.print(f"[yellow]Processing file: {input}[/yellow]")
            stats = pipeline.ingest_file(input)
            console.print(f"[green]✓ Processed file successfully[/green]")
            console.print(f"  Documents loaded: {stats['documents_loaded']}")
            console.print(f"  Chunks created: {stats['chunks_created']}")
            console.print(f"  Chunks stored: {stats['chunks_stored']}")
        elif input.is_dir():
            console.print(f"[yellow]Processing directory: {input}[/yellow]")
            batch_processor = BatchProcessor(pipeline, config.batch_processing)
            stats = batch_processor.process_directory_in_batches(
                input,
                recursive=recursive,
                show_progress=True,
            )
            console.print(f"[green]✓ Processing complete[/green]")
            console.print(f"  Files processed: {stats['files_processed']}")
            console.print(f"  Documents loaded: {stats['documents_loaded']}")
            console.print(f"  Chunks created: {stats['chunks_created']}")
            console.print(f"  Chunks stored: {stats['chunks_stored']}")
            if stats["errors"]:
                console.print(f"[red]  Errors: {len(stats['errors'])}[/red]")
                for error in stats["errors"][:5]:  # Show first 5 errors
                    console.print(f"    - {error}")
        else:
            console.print(f"[red]Error: {input} is not a valid file or directory[/red]")
            raise click.Abort()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("Ingestion failed")
        raise click.Abort()


@cli.command()
@click.option("--query", "-q", required=True, help="Query string")
@click.option("--collection", "-c", required=True, help="Collection name")
@click.option("--k", default=5, type=int, help="Number of results")
@click.option(
    "--embedding-provider",
    default="openai",
    type=click.Choice(["openai", "cohere", "local"]),
    help="Embedding provider",
)
@click.option(
    "--vector-store",
    default="milvus",
    type=click.Choice(["milvus", "chroma", "qdrant"]),
    help="Vector store backend",
)
@click.option("--filter", type=str, help="Metadata filter (JSON string)")
def query(
    query: str,
    collection: str,
    k: int,
    embedding_provider: str,
    vector_store: str,
    filter: Optional[str],
):
    """Query the vector store."""
    try:
        console.print(f"[bold green]Searching...[/bold green]")
        console.print(f"Query: {query}")
        console.print(f"Collection: {collection}")

        # Create configuration
        config = _create_config_from_options(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            chunk_size=1000,  # Not used for querying
            chunk_overlap=200,  # Not used for querying
        )

        # Initialize query pipeline
        query_pipeline = QueryPipeline(config, collection)

        # Parse filter if provided
        filter_dict = None
        if filter:
            try:
                filter_dict = json.loads(filter)
            except json.JSONDecodeError:
                console.print(f"[red]Error: Invalid filter JSON: {filter}[/red]")
                raise click.Abort()

        # Perform search
        results = query_pipeline.search(query, k=k, filter=filter_dict)

        # Display results
        console.print(f"\n[bold green]Found {len(results)} result(s):[/bold green]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Source", width=40)
        table.add_column("Content Preview", width=60)

        for idx, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            table.add_row(str(idx), source, content_preview)

        console.print(table)

        # Show full content if verbose
        if len(results) > 0:
            console.print("\n[bold]Full Results:[/bold]")
            for idx, doc in enumerate(results, 1):
                console.print(f"\n[bold cyan]Result {idx}:[/bold cyan]")
                console.print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                console.print(f"Content:\n{doc.page_content}\n")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("Query failed")
        raise click.Abort()


@cli.command()
@click.option("--collection", "-c", required=True, help="Collection name")
@click.option(
    "--embedding-provider",
    default="openai",
    type=click.Choice(["openai", "cohere", "local"]),
    help="Embedding provider",
)
@click.option(
    "--vector-store",
    default="milvus",
    type=click.Choice(["milvus", "chroma", "qdrant"]),
    help="Vector store backend",
)
def status(collection: str, embedding_provider: str, vector_store: str):
    """Get collection status and statistics."""
    try:
        console.print(f"[bold green]Getting collection status...[/bold green]")
        console.print(f"Collection: {collection}")

        # Create configuration
        config = _create_config_from_options(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            chunk_size=1000,  # Not used for status
            chunk_overlap=200,  # Not used for status
        )

        # Initialize query pipeline
        query_pipeline = QueryPipeline(config, collection)

        # Get stats
        stats = query_pipeline.get_stats()

        # Display stats
        console.print(f"\n[bold green]Collection Statistics:[/bold green]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Property", width=30)
        table.add_column("Value", width=30)

        for key, value in stats.items():
            table.add_row(key, str(value))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("Status check failed")
        raise click.Abort()


if __name__ == "__main__":
    cli()

