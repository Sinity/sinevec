from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import json
import signal

from tqdm import tqdm
import tiktoken

from voyage_embeddings.embed_utils import (
    get_clients, ensure_collection, count_tokens, simple_chunk_document, group_chunks_for_voyage, should_skip_file
)


class EmbeddingState:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self.load_state()
        self.token_usage = self.state.get('token_usage', {})
        self.processed_files = set(self.state.get('processed_files', []))
        self.failed_files = self.state.get('failed_files', {})
        self.current_file = None
        self.start_time = datetime.now().isoformat()
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def load_state(self):
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {'token_usage': {}, 'processed_files': [], 'failed_files': {}}

    def save_state(self):
        self.state['token_usage'] = self.token_usage
        self.state['processed_files'] = list(self.processed_files)
        self.state['failed_files'] = self.failed_files
        self.state['last_updated'] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.rename(self.state_file)

    def handle_interrupt(self, signum, frame):
        if self.current_file and self.current_file in self.processed_files:
            self.processed_files.remove(self.current_file)
        self.save_state()
        raise SystemExit(0)

    def mark_processed(self, file_path: str, tokens_used: int = 0):
        self.processed_files.add(file_path)
        if tokens_used:
            self.token_usage['total'] = self.token_usage.get('total', 0) + tokens_used
            self.token_usage[file_path] = tokens_used
        if len(self.processed_files) % 10 == 0:
            self.save_state()

    def mark_failed(self, file_path: str, error: str):
        self.failed_files[file_path] = {'error': error[:500], 'timestamp': datetime.now().isoformat()}


def embed_file(vo, client, file_path: Path, collection_name: str, state: EmbeddingState, force: bool = False) -> int:
    file_str = str(file_path)
    if not force and file_str in state.processed_files:
        return 0
    state.current_file = file_str
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        if not content.strip():
            return 0
        try:
            collection = ensure_collection(client, collection_name)
        except Exception:
            collection = ensure_collection(client, collection_name)
        chunks = simple_chunk_document(content)
        groups = group_chunks_for_voyage(chunks)
        if not groups:
            state.mark_processed(file_str, 0)
            return 0
        total_tokens = 0
        for group_idx, chunk_group in enumerate(groups):
            if not chunk_group or not any(c.strip() for c in chunk_group):
                continue
            embeds = vo.contextualized_embed(inputs=[chunk_group], model='voyage-context-3', input_type='document')
            for chunk_idx, (chunk, embedding) in enumerate(zip(chunk_group, embeds.results[0].embeddings)):
                chunk_id = f"{file_str}#g{group_idx}#c{chunk_idx}"
                try:
                    collection.delete(ids=[chunk_id])
                except Exception:
                    pass
                collection.add(embeddings=[embedding], documents=[chunk[:65536]], metadatas=[{
                    'source': file_str,
                    'file_name': file_path.name,
                    'group_index': group_idx,
                    'chunk_index': chunk_idx,
                    'total_groups': len(groups),
                    'embedded_at': datetime.now().isoformat()
                }], ids=[chunk_id])
            total_tokens += embeds.total_tokens
        if total_tokens > 0:
            state.mark_processed(file_str, total_tokens)
        else:
            state.mark_failed(file_str, 'No groups successfully embedded')
        state.current_file = None
        return total_tokens
    except Exception as e:
        state.mark_failed(file_str, str(e))
        state.current_file = None
        return 0


def scan_files(directory: Path, state: EmbeddingState) -> List[Path]:
    files: List[Path] = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and not should_skip_file(file_path):
            if file_str := str(file_path) and file_str not in state.processed_files:
                files.append(file_path)
    files.sort(key=lambda f: f.stat().st_size)
    return files

