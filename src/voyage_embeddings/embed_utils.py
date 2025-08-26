from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import chromadb
import tiktoken
import voyageai
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
UNIFIED = os.environ.get("CHROMA_COLLECTION", "unified")
EMBED_DIM = int(os.environ.get("EMBED_OUTPUT_DIMENSION", "1024"))
# Defaults: contextualized uses 'voyage-context-3' unless overridden; standard uses 'voyage-2'
CONTEXT_MODEL = os.environ.get("VOYAGE_CONTEXT_MODEL", "voyage-context-3")
DEFAULT_MODEL = os.environ.get("VOYAGE_EMBED_MODEL", "voyage-2")
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


def contextual_windows(texts: List[str], max_tokens: int = MAX_DOC_TOKENS, always_include_first: bool = True) -> List[Tuple[int, int]]:
    toks = [count_tokens(t) for t in texts]
    windows: List[Tuple[int, int]] = []
    if not texts:
        return windows
    if always_include_first:
        base_text = texts[0]
        base_t = toks[0]
        while True:
            total = base_t
            end = 1
            while end < len(texts) and total + toks[end] <= max_tokens:
                total += toks[end]
                end += 1
            windows.append((0, end))
            if end >= len(texts):
                break
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
