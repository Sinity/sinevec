# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Sinevec is a semantic search system that embeds documents using Voyage AI's contextualized embeddings and stores them in Qdrant for similarity search. The system is designed to process personal knowledge bases, code repositories, and conversations with intelligent chunking strategies.

## Development Commands

### Environment Setup

```bash
# Enter Nix development shell (always run first)
nix develop

# The shell automatically:
# - Creates Python virtual environment (.venv)
# - Installs dependencies (voyageai, qdrant-client, langchain-text-splitters, etc.)
# - Loads VOYAGE_API_KEY from .env file
```

### Primary Embedding Operations

```bash
# Main embedding pipeline (v3 - latest version)
python embed_v3.py                  # Start fresh embedding
python embed_v3.py --resume         # Resume from last state
python embed_v3.py --reset          # Clear state and restart
python embed_v3.py --stats          # Show embedding statistics
python embed_v3.py --force          # Force re-embed all files

# Check embedding status
python check_status.py              # View token usage and collection counts
```

### Search Operations

```bash
# Interactive search interface
sinevec search "query text"         # Direct search from command line

# Specialized searches (in interactive mode)
code: [query]                       # Search code collection only
chat: [query]                       # Search conversations only
git: [query]                        # Search git history
timeline: [query]                   # Search with date filtering
```

## Architecture

### Core Components

**Embedding Pipeline (`embed_v3.py`)**

- Resumable state management with graceful interruption handling
- Token-aware chunking using tiktoken for accurate counting
- Contextualized embeddings via `voyage-context-3` model
- Automatic rate limit handling with retry logic
- Progress tracking with token usage statistics

**Storage Layer (Qdrant)**

- Collections: `knowledgebase`, `code`, `conversations` (default: `unified`)
- Persistent storage under `/realm/data/qdrant`
- Metadata tracking: source files, timestamps, chunk indices
- Support for filtered queries and timeline searches

**Chunking Strategies**

- **Simple chunking** (`embed_v3.py`): Fast, recursion-safe paragraph/sentence splitting
- **Smart chunking** (`smart_utils.py`): Token-accurate splitting with tiktoken
- **Hierarchical chunking** (`hierarchical_chunker.py`): Content-aware parsing for journals, MOCs, code

### Key Technical Patterns

**Token Management**

- Uses `cl100k_base` tokenizer (GPT-4 compatible) for accurate counting
- Groups chunks to maximize context window usage (30K token limit)
- Automatically splits oversized content with semantic preservation

**Content Type Detection**

- Identifies document types: journals, MOCs, AI conversations, code
- Applies type-specific chunking strategies
- Preserves structural information in metadata

**State Persistence**

- JSON-based state tracking in `embedding_state_v3.json`
- Tracks processed files, failures, and token usage
- Atomic file writes for crash safety
- Saves state every 10-25 files and on interruption

### File Organization

```
sinevec/
├── embed_v3.py              # Main embedding pipeline (latest)
├── embed_v2.py              # Previous version (deprecated)
├── embed_everything.py      # Original version (deprecated)
├── (use CLI: sinevec search)    # Search interface
├── utils.py                 # Basic utilities
├── smart_utils.py          # Token-aware utilities
├── hierarchical_chunker.py # Content-aware chunking
├── check_status.py         # Status monitoring
├── embedding_state_v3.json # Persistent state
├── (Qdrant service)        # Vector database storage lives under /realm/data/qdrant
└── logs/                   # Token usage logs
```

## Common Development Tasks

### Adding New Document Sources

```python
# In embed_v3.py main(), add to sources list:
sources = [
    (Path("/path/to/documents"), "collection_name"),
    # Add new source here
]
```

### Adjusting Chunk Sizes

```python
# In embed_v3.py:
simple_chunk_document(content, max_chunk_size=8000)  # Character limit
group_chunks_for_voyage(chunks, max_tokens=30000)    # Token limit per group
```

### Implementing Custom Chunking

```python
# In hierarchical_chunker.py, add content type detection:
def detect_content_type(path: str, content: str) -> str:
    # Add custom patterns

# Add corresponding parser:
def parse_custom_hierarchy(content: str) -> ChunkNode:
    # Implement hierarchical parsing
```

## Debugging

### Common Issues and Solutions

**Rate Limiting**

- Automatic 20-second delay and retry on rate limit errors
- Payment method added should prevent most rate limiting

**Memory/Recursion Errors**

- Use simple_chunk_document() instead of recursive splitters
- Verify chunk sizes with count_tokens() not estimates

**Collection Errors**

- Collections auto-create if missing
- Use collection name (string) not ID with ChromaDB client

**Token Count Verification**

```python
# Always use tiktoken for accurate counts:
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = len(tokenizer.encode(text))
```

### Monitoring Commands

```bash
# Check embedding progress
tail -f embedding_v3.log

# View state file
jq . embedding_state_v3.json

# Check ChromaDB collections
python -c "from qdrant_client import QdrantClient; client = QdrantClient(host='127.0.0.1', port=6333); print([c.name for c in client.get_collections().collections])"
```

## API Integration

### Voyage AI Configuration

- Model: `voyage-context-3` for documents, queries
- Input types: `document` for indexing, `query` for search
- Contextualized embedding for better semantic understanding

### Environment Variables

```bash
# Required in .env file:
VOYAGE_API_KEY=your_api_key_here
```
