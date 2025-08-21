"""Smart utility functions with proper token counting."""

import tiktoken
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Initialize tokenizer for accurate counting
# voyage-context-3 likely uses cl100k_base encoding (same as GPT-4)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count actual tokens using tiktoken."""
    return len(tokenizer.encode(text))

def smart_split_text(text: str, max_tokens: int = 30000) -> List[str]:
    """
    Intelligently split text to fit within token limits.
    Tries to preserve semantic boundaries.
    """
    # If text fits, return as is
    if count_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    
    # Try splitting by major boundaries first
    # Order matters - from most semantic to least
    separators = [
        "\n\n---\n\n",  # Major section breaks
        "\n\n## ",      # Markdown sections
        "\n\n# ",       # Markdown headers
        "\n\n",         # Paragraphs
        "\n",           # Lines
        ". ",           # Sentences
        " ",            # Words
    ]
    
    remaining = text
    
    for separator in separators:
        if separator in remaining:
            parts = remaining.split(separator)
            current_chunk = ""
            current_tokens = 0
            
            for i, part in enumerate(parts):
                # Add separator back except for first part
                if i > 0 and separator != "\n\n---\n\n":
                    part = separator + part
                
                part_tokens = count_tokens(part)
                
                # If single part is too large, need to split it further
                if part_tokens > max_tokens:
                    # Save current chunk if any
                    if current_chunk:
                        chunks.append(current_chunk)
                    # Recursively split the large part
                    sub_chunks = smart_split_text(part, max_tokens)
                    chunks.extend(sub_chunks[:-1])  # Add all but last
                    current_chunk = sub_chunks[-1]  # Last becomes current
                    current_tokens = count_tokens(current_chunk)
                # If adding would exceed limit, start new chunk
                elif current_tokens + part_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
                    current_tokens = part_tokens
                else:
                    current_chunk += part
                    current_tokens += part_tokens
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
    
    # If no separators work, do hard split by token count
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    
    return chunks

def adaptive_chunk_document(content: str, source_path: str) -> List[str]:
    """
    Adaptively chunk document based on type and actual token counts.
    Returns chunks that will fit in Voyage's context window.
    """
    file_type = get_file_type(str(source_path))
    
    # First pass: split by semantic boundaries based on file type
    if file_type == 'ai_conversation':
        # Split conversations by turns
        pattern = r'(?:Human:|Assistant:|User:|Model:|<\|.*?\|>|\n\n---\n\n)'
        parts = re.split(f'({pattern})', content)
        
        chunks = []
        current = ""
        for part in parts:
            if count_tokens(current + part) > 30000:
                if current:
                    chunks.append(current)
                current = part
            else:
                current += part
        if current:
            chunks.append(current)
    
    elif 'code' in file_type:
        # For code, try to keep functions/classes together
        if 'python' in file_type:
            pattern = r'(?:^|\n)(class\s+\w+|def\s+\w+|async\s+def\s+\w+)'
        elif 'rust' in file_type:
            pattern = r'(?:^|\n)(fn\s+\w+|impl\s+|struct\s+\w+|enum\s+\w+)'
        else:
            pattern = r'(?:^|\n)(function\s+\w+|class\s+\w+|const\s+\w+)'
        
        parts = re.split(f'({pattern})', content, flags=re.MULTILINE)
        chunks = combine_parts_by_tokens(parts, 30000)
    
    else:
        # For other content, use smart splitting
        chunks = smart_split_text(content, 30000)
    
    # Verify all chunks fit
    verified_chunks = []
    for chunk in chunks:
        if count_tokens(chunk) > 31000:  # Leave 1K buffer
            # Split oversized chunks
            verified_chunks.extend(smart_split_text(chunk, 30000))
        else:
            verified_chunks.append(chunk)
    
    return verified_chunks

def combine_parts_by_tokens(parts: List[str], max_tokens: int) -> List[str]:
    """Combine parts into chunks based on token count."""
    chunks = []
    current = ""
    current_tokens = 0
    
    for part in parts:
        part_tokens = count_tokens(part)
        if current_tokens + part_tokens > max_tokens:
            if current:
                chunks.append(current)
            current = part
            current_tokens = part_tokens
        else:
            current += part
            current_tokens += part_tokens
    
    if current:
        chunks.append(current)
    
    return chunks

def group_chunks_for_context(chunks: List[str], overlap_ratio: float = 0.1) -> List[List[str]]:
    """
    Group chunks for contextualized embedding with small overlap.
    Uses actual token counting to maximize context usage.
    
    Args:
        chunks: List of text chunks
        overlap_ratio: How much to overlap between groups (0.1 = 10% overlap)
    """
    MAX_CONTEXT = 31000  # Leave 1K buffer for safety
    
    groups = []
    i = 0
    
    while i < len(chunks):
        current_group = []
        current_tokens = 0
        start_idx = i
        
        # Add chunks until we hit the limit
        while i < len(chunks):
            chunk_tokens = count_tokens(chunks[i])
            
            # Verify chunk isn't too large on its own
            if chunk_tokens > MAX_CONTEXT:
                print(f"WARNING: Chunk has {chunk_tokens} tokens, splitting further")
                sub_chunks = smart_split_text(chunks[i], MAX_CONTEXT - 1000)
                # Replace the oversized chunk with sub-chunks
                chunks = chunks[:i] + sub_chunks + chunks[i+1:]
                chunk_tokens = count_tokens(chunks[i])
            
            # Check if adding this chunk would exceed limit
            if current_tokens + chunk_tokens > MAX_CONTEXT:
                break
            
            current_group.append(chunks[i])
            current_tokens += chunk_tokens
            i += 1
        
        if current_group:
            groups.append(current_group)
            
            # Calculate overlap for next group
            if i < len(chunks) and len(current_group) > 1:
                # Include last 10% of chunks in next group for context
                overlap_chunks = max(1, int(len(current_group) * overlap_ratio))
                i = max(start_idx + 1, i - overlap_chunks)
    
    # Log group statistics
    for idx, group in enumerate(groups):
        total_tokens = sum(count_tokens(chunk) for chunk in group)
        print(f"  Group {idx}: {len(group)} chunks, {total_tokens} tokens")
    
    return groups

def get_file_type(path: str) -> str:
    """Determine file type for metadata."""
    path = str(path).lower()
    
    if 'claude' in path or 'chatgpt' in path or 'gemini' in path:
        return 'ai_conversation'
    elif path.endswith('.rs'):
        return 'rust_code'
    elif path.endswith('.py'):
        return 'python_code'
    elif path.endswith('.js') or path.endswith('.ts'):
        return 'javascript_code'
    elif path.endswith('.sql'):
        return 'sql'
    elif path.endswith('.md'):
        if 'log' in path:
            return 'log'
        elif 'moc' in path:
            return 'moc'
        else:
            return 'markdown'
    elif path.endswith('.toml') or path.endswith('.yaml') or path.endswith('.json'):
        return 'config'
    else:
        return 'text'