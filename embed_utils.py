"""
Shared utilities for embedding scripts: clients, tokenization, context windows,
and common helpers.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Dict
from pathlib import Path
import re as _re

import chromadb
import tiktoken
import voyageai
from dotenv import load_dotenv


# Load env once per process
load_dotenv()

# Defaults
DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db_v3")
UNIFIED = os.environ.get("CHROMA_COLLECTION", "unified")
EMBED_DIM = int(os.environ.get("EMBED_OUTPUT_DIMENSION", "1024"))
CONTEXT_MODEL = os.environ.get("VOYAGE_CONTEXT_MODEL", "voyage-context-3")
DEFAULT_MODEL = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-3")
MAX_DOC_TOKENS = int(os.environ.get("CONTEXT_DOC_TOKEN_LIMIT", "30000"))

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text or ""))


def get_clients() -> tuple[voyageai.Client, chromadb.PersistentClient]:
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    client = chromadb.PersistentClient(path=DB_PATH)
    return vo, client


def ensure_collection(client: chromadb.PersistentClient, name: str = UNIFIED):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)


def detect_code(text: str) -> bool:
    indicators = [
        "```",
        "def ", "class ", "function ", "const ", "let ", "var ",
        "import ", "from ", "require(",
        "if __name__", "pub fn", "fn main",
        "#!/usr/bin",
        "SELECT ", "CREATE TABLE", "INSERT INTO",
    ]
    t = text or ""
    return any(i in t for i in indicators)


def domain_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1).lower() if m else ""


def split_long_text(text: str, max_tokens: int = 8000) -> List[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    parts: List[str] = []
    paras = re.split(r"\n\n+", text or "")
    cur: List[str] = []
    cur_t = 0
    for p in paras:
        t = count_tokens(p)
        if cur_t + t + 10 > max_tokens:
            if cur:
                parts.append("\n\n".join(cur))
            cur = [p]
            cur_t = t
        else:
            cur.append(p)
            cur_t += t
    if cur:
        parts.append("\n\n".join(cur))
    final: List[str] = []
    for part in parts:
        while count_tokens(part) > max_tokens:
            mid = len(part) // 2
            final.append(part[:mid])
            part = part[mid:]
        final.append(part)
    return final


def contextual_windows(
    texts: List[str],
    max_tokens: int = MAX_DOC_TOKENS,
    always_include_first: bool = True,
) -> List[Tuple[int, int]]:
    """Compute windows [start,end) so total tokens fit limit.
    If always_include_first, each window includes index 0 (e.g., a summary).
    Returns list of (start,end) relative to current list.
    """
    toks = [count_tokens(t) for t in texts]
    windows: List[Tuple[int, int]] = []
    if not texts:
        return windows
    if always_include_first:
        # Greedy windows that always include index 0
        base_text = texts[0]
        base_t = toks[0]
        remaining_idx = 1
        while True:
            total = base_t
            end = 1
            while end < len(texts) and total + toks[end] <= max_tokens:
                total += toks[end]
                end += 1
            windows.append((0, end))
            if end >= len(texts):
                break
            # Slide by removing the included highlights and keep summary at 0
            texts = [base_text] + texts[end:]
            toks = [base_t] + toks[end:]
    else:
        start = 0
        while start < len(texts):
            total = 0
            end = start
            while end < len(texts) and total + toks[end] <= max_tokens:
                total += toks[end]
                end += 1
            if end == start:
                end += 1
            windows.append((start, end))
            start = end
    return windows


# -------------------------
# File utilities (moved from utils.py)
# -------------------------

def extract_timestamp(content: str, path: str) -> str:
    patterns = [
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        r"(\d{4}-\d{2}-\d{2})",
        r"Date: (\d{4}-\d{2}-\d{2})",
        r"\*\*(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in patterns:
        match = _re.search(pattern, content[:2000] if len(content) > 2000 else content)
        if match:
            return match.group(1)
    match = _re.search(r"(\d{4}-\d{2}-\d{2})", path)
    if match:
        return match.group(1)
    return "unknown"


def get_file_type(path: str) -> str:
    p = str(path).lower()
    if 'claude' in p or 'chatgpt' in p or 'gemini' in p:
        return 'ai_conversation'
    if p.endswith('.rs'):
        return 'rust_code'
    if p.endswith('.py'):
        return 'python_code'
    if p.endswith('.js') or p.endswith('.ts'):
        return 'javascript_code'
    if p.endswith('.sql'):
        return 'sql'
    if p.endswith('.md'):
        if 'log' in p:
            return 'log'
        if 'moc' in p:
            return 'moc'
        return 'markdown'
    if p.endswith('.toml') or p.endswith('.yaml') or p.endswith('.json'):
        return 'config'
    return 'text'


def should_skip_file(file_path: Path) -> bool:
    skip_patterns = [
        '.git/', '.gitignore',
        '__pycache__/', '.venv/', 'venv/', 'env/',
        'node_modules/', '/target/', 'dist/', 'build/',
        '.next/', '.nuxt/', 'out/',
        'vendor/', 'packages/', '.cargo/',
        '.pyc', '.pyo', '.so', '.dylib', '.dll', '.o', '.a',
        '.wasm', '.exe',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
        '.mp4', '.avi', '.mov', '.mkv', '.webm',
        '.mp3', '.wav', '.flac', '.ogg',
        '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z',
        '.DS_Store', 'Thumbs.db', '.directory',
        'package-lock.json', 'yarn.lock', 'Cargo.lock', 'flake.lock',
        'poetry.lock', 'composer.lock',
        '.cache/', '.pytest_cache/', '.mypy_cache/',
        '.db', '.sqlite', '.sqlite3',
        '.log',
        '.csv', '.parquet', '.feather', '.h5', '.hdf5',
        '.min.js', '.min.css', '.map',
        'LICENSE', 'COPYING',
    ]
    path_str = str(file_path)
    for pattern in skip_patterns:
        if pattern in path_str:
            if pattern == '.log' and '/knowledgebase/' in path_str:
                continue
            return True
    for part in file_path.parts:
        if part.startswith('.') and part not in ['.', '..']:
            return True
    if '/project/' in path_str:
        allowed_extensions = {
            '.rs', '.py', '.js', '.ts', '.jsx', '.tsx',
            '.go', '.c', '.cpp', '.h', '.hpp',
            '.md', '.txt', '.toml', '.yaml', '.yml', '.json',
            '.sql', '.sh', '.bash', '.fish', '.nix',
            '.html', '.css', '.scss',
        }
        if file_path.suffix and file_path.suffix not in allowed_extensions:
            return True
    return False


# -------------------------
# Chunking helpers (generalized from embed_v3)
# -------------------------

def simple_chunk_document(content: str, max_chunk_size: int = 8000) -> List[str]:
    if len(content) < max_chunk_size:
        return [content]
    chunks: List[str] = []
    paragraphs = content.split('\n\n')
    current_chunk = ""
    for para in paragraphs:
        if len(para) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            if '. ' in para:
                sentences = para.split('. ')
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 2 > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent + '. '
                    else:
                        current_chunk += sent + '. '
            else:
                for i in range(0, len(para), max_chunk_size):
                    chunks.append(para[i:i+max_chunk_size])
        elif len(current_chunk) + len(para) + 2 > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def group_chunks_for_voyage(chunks: List[str], max_tokens: int = MAX_DOC_TOKENS) -> List[List[str]]:
    groups: List[List[str]] = []
    current_group: List[str] = []
    current_tokens = 0
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens > max_tokens:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
            # Split via simple_chunk_document to reduce size further
            for sub in simple_chunk_document(chunk, max_chunk_size=4000):
                if count_tokens(sub) > max_tokens:
                    groups.append([sub[:8000]])
                else:
                    groups.append([sub])
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
