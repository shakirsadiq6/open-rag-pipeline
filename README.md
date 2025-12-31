# Open RAG Pipeline

A scalable, open-source document ingestion and vectorization system that accepts any file type, converts documents into vector embeddings, and stores them in configurable vector databases.

## Features

- **Multi-format Support**: PDF, DOCX, TXT, HTML, and more
- **Configurable Embeddings**: OpenAI, Cohere, or local models (sentence-transformers)
- **Multiple Vector Stores**: Milvus, Chroma, Qdrant
- **Scalable**: Handles datasets from 5GB to 100GB+ with batch processing
- **Vendor Agnostic**: Switch providers without code changes
- **CLI Interface**: Easy-to-use command-line tools
- **Error Handling**: Robust error handling with retry logic

## Installation

### Basic Installation

```bash
cd open-rag-pipeline
pip install -e .
```

### Installation with Vector Store Backends

Install with specific vector store backends:

```bash
# Milvus only
pip install -e ".[milvus]"

# Chroma only
pip install -e ".[chroma]"

# Qdrant only
pip install -e ".[qdrant]"

# Local embeddings (sentence-transformers)
pip install -e ".[local-embeddings]"

# All backends
pip install -e ".[all]"
```

### Development Installation

```bash
pip install -e ".[dev,all]"
```

## Quick Start

### 1. Set up environment variables

Create a `.env` file in the project root:

```bash
# Embedding Provider
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

# Vector Store
VECTOR_STORE=milvus
MILVUS_URI=./milvus_demo.db

# Chunking
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
```

### 2. Ingest documents

```bash
# Ingest a single file
open-rag ingest --input document.pdf --collection my_collection

# Ingest a directory
open-rag ingest --input ./documents --collection my_collection --recursive

# With custom settings
open-rag ingest \
  --input ./documents \
  --collection my_collection \
  --embedding-provider openai \
  --vector-store milvus \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --batch-size 100
```

### 3. Query the vector store

```bash
# Basic query
open-rag query --query "What is machine learning?" --collection my_collection --k 5

# With metadata filter
open-rag query \
  --query "What is machine learning?" \
  --collection my_collection \
  --k 5 \
  --filter '{"file_type": "pdf"}'
```

### 4. Check status

```bash
open-rag status --collection my_collection
```

## Configuration

### Embedding Providers

**OpenAI** (default):
```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

**Cohere**:
```bash
EMBEDDING_PROVIDER=cohere
COHERE_API_KEY=your_key_here
```

**Local (sentence-transformers)**:
```bash
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Vector Stores

**Milvus** (default):
```bash
VECTOR_STORE=milvus
MILVUS_URI=./milvus_demo.db  # For Milvus Lite
# Or for server: MILVUS_URI=http://localhost:19530
```

**Chroma**:
```bash
VECTOR_STORE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

**Qdrant**:
```bash
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional
```

## Python API

You can also use the pipeline programmatically:

```python
from pathlib import Path
from open_rag_pipeline.config import PipelineConfig
from open_rag_pipeline.pipeline.ingestion_pipeline import IngestionPipeline
from open_rag_pipeline.pipeline.query_pipeline import QueryPipeline

# Load configuration
config = PipelineConfig.from_env()

# Ingest documents
pipeline = IngestionPipeline(config, collection_name="my_collection")
stats = pipeline.ingest_file(Path("document.pdf"))
print(f"Processed {stats['chunks_stored']} chunks")

# Query
query_pipeline = QueryPipeline(config, collection_name="my_collection")
results = query_pipeline.search("What is machine learning?", k=5)
for doc in results:
    print(doc.page_content)
```

## Architecture

The pipeline follows a modular architecture:

1. **Document Loaders**: Extract text from various file formats
2. **Text Chunking**: Split documents into manageable chunks
3. **Embedding Generation**: Convert text to vector embeddings
4. **Vector Storage**: Store embeddings in vector databases
5. **Query Interface**: Search and retrieve relevant documents

## Supported File Types

- PDF (`.pdf`)
- Text (`.txt`, `.text`)
- Word Documents (`.docx`, `.doc`)
- HTML (`.html`, `.htm`)

## Error Handling

The pipeline includes comprehensive error handling:
- Automatic retries with exponential backoff
- Detailed error logging
- Graceful failure handling
- Progress tracking for long-running operations

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## License

MIT License

## Contributing

When adding new features:
1. Follow the existing code structure
2. Add type hints and docstrings
3. Write tests for new functionality
4. Update documentation
