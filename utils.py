"""Utility functions for embedding pipeline."""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

def estimate_tokens(text: str) -> int:
    """Rough estimate of token count."""
    return int(len(text.split()) * 1.3)

def extract_timestamp(content: str, path: str) -> str:
    """Extract timestamp from content or filename."""
    patterns = [
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        r"(\d{4}-\d{2}-\d{2})",
        r"Date: (\d{4}-\d{2}-\d{2})",
        r"\*\*(\d{4}-\d{2}-\d{2})",
    ]
    
    # Try content first
    for pattern in patterns:
        match = re.search(pattern, content[:2000] if len(content) > 2000 else content)
        if match:
            return match.group(1)
    
    # Try filename
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    if match:
        return match.group(1)
    
    return "unknown"

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

def smart_chunk_document(content: str, source_path: str, chunk_size: int = 3000) -> List[str]:
    """
    Smart chunking based on document type.
    Default to 3000 CHARACTERS (not tokens!) to ensure we stay well under 32K token limit.
    """
    
    file_type = get_file_type(source_path)
    
    # Use CHARACTER counts, not token estimates
    # Roughly 4 chars = 1 token, so 3000 chars = ~750 tokens
    
    # For conversations, split by turn
    if file_type == 'ai_conversation':
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n---\n\n", "\nHuman:", "\nAssistant:", "\nUser:", "\nModel:", "\n\n"],
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=len  # Use character count
        )
    
    # For code, split by function/class
    elif 'code' in file_type:
        if 'rust' in file_type:
            separators = ["\n\nfn ", "\n\nimpl ", "\n\nstruct ", "\n\nenum ", "\n\n"]
        elif 'python' in file_type:
            separators = ["\n\nclass ", "\n\ndef ", "\n\nasync def ", "\n\n"]
        else:
            separators = ["\n\nfunction ", "\n\nclass ", "\n\nconst ", "\n\n"]
        
        splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=len
        )
    
    # For markdown, split by headers
    elif file_type in ['markdown', 'moc', 'log']:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n# ", "\n## ", "\n### ", "\n---", "\n\n"],
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=len
        )
    
    # For SQL, split by statements
    elif file_type == 'sql':
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\nCREATE ", "\n\nALTER ", "\n\nINSERT ", "\n\n--", "\n\n"],
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=len
        )
    
    else:
        # Default splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=len
        )
    
    return splitter.split_text(content)

def group_chunks_for_context(chunks: List[str], max_tokens: int = 10000) -> List[List[str]]:
    """
    Group chunks into context windows for Voyage.
    Each group will share context during embedding.
    Max 10K tokens to ensure we stay WELL under 32K limit even with bad estimates.
    """
    groups = []
    current_group = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk)
        
        # If single chunk is too large, split it aggressively
        if chunk_tokens > max_tokens:
            # Split by character count to ensure small pieces
            max_chars = 8000  # ~2000 tokens max per piece
            for i in range(0, len(chunk), max_chars):
                sub_chunk = chunk[i:i+max_chars]
                sub_tokens = estimate_tokens(sub_chunk)
                
                # Start new group if needed
                if current_tokens + sub_tokens > max_tokens:
                    if current_group:
                        groups.append(current_group)
                    current_group = [sub_chunk]
                    current_tokens = sub_tokens
                else:
                    current_group.append(sub_chunk)
                    current_tokens += sub_tokens
        elif current_tokens + chunk_tokens > max_tokens:
            if current_group:
                groups.append(current_group)
            current_group = [chunk]
            current_tokens = chunk_tokens
        else:
            current_group.append(chunk)
            current_tokens += chunk_tokens
    
    if current_group:
        groups.append(current_group)
    
    return groups

def should_skip_file(file_path: Path) -> bool:
    """Determine if a file should be skipped."""
    skip_patterns = [
        # Version control
        '.git/', '.gitignore',
        # Build artifacts  
        '__pycache__/', '.venv/', 'venv/', 'env/',
        'node_modules/', '/target/', 'dist/', 'build/',
        '.next/', '.nuxt/', 'out/',
        # Dependencies
        'vendor/', 'packages/', '.cargo/',
        # Binary files
        '.pyc', '.pyo', '.so', '.dylib', '.dll', '.o', '.a',
        '.wasm', '.exe',
        # Media files
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
        '.mp4', '.avi', '.mov', '.mkv', '.webm',
        '.mp3', '.wav', '.flac', '.ogg',
        # Archives
        '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z',
        # System files
        '.DS_Store', 'Thumbs.db', '.directory',
        # Lock files
        'package-lock.json', 'yarn.lock', 'Cargo.lock', 'flake.lock',
        'poetry.lock', 'composer.lock',
        # Cache
        '.cache/', '.pytest_cache/', '.mypy_cache/',
        # Databases
        '.db', '.sqlite', '.sqlite3',
        # Logs (unless in knowledgebase)
        '.log',
        # Large data files
        '.csv', '.parquet', '.feather', '.h5', '.hdf5',
        # Other
        '.min.js', '.min.css', '.map',
        'LICENSE', 'COPYING',
    ]
    
    path_str = str(file_path)
    
    # Skip if matches any pattern
    for pattern in skip_patterns:
        if pattern in path_str:
            # Exception: don't skip logs in knowledgebase
            if pattern == '.log' and '/knowledgebase/' in path_str:
                continue
            return True
    
    # Skip hidden files/directories (starting with .)
    for part in file_path.parts:
        if part.startswith('.') and part not in ['.', '..']:
            return True
    
    # Only process text-like files in code directories
    if '/project/' in path_str:
        allowed_extensions = {
            '.rs', '.py', '.js', '.ts', '.jsx', '.tsx',
            '.go', '.c', '.cpp', '.h', '.hpp',
            '.md', '.txt', '.toml', '.yaml', '.yml', '.json',
            '.sql', '.sh', '.bash', '.fish', '.nix',
            '.html', '.css', '.scss',
        }
        
        # Check if has allowed extension
        if file_path.suffix and file_path.suffix not in allowed_extensions:
            return True
    
    return False