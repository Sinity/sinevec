"""Compatibility layer re-exporting helpers from embed_utils to avoid duplication."""

from pathlib import Path
from typing import List
from embed_utils import (
    extract_timestamp,
    get_file_type,
    should_skip_file,
)

# Retain old names for compatibility
def smart_chunk_document(content: str, source_path: str, chunk_size: int = 3000) -> List[str]:
    from embed_utils import simple_chunk_document
    return simple_chunk_document(content, max_chunk_size=chunk_size)

def group_chunks_for_context(chunks: List[str], max_tokens: int = 10000) -> List[List[str]]:
    from embed_utils import group_chunks_for_voyage
    return group_chunks_for_voyage(chunks, max_tokens=max_tokens)
