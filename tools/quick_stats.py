#!/usr/bin/env python3
"""Quick stats about embeddings."""

import json
from pathlib import Path

from qdrant_client import QdrantClient

from sinevec.embed_utils import (
    DATA_ROOT,
    STATE_DIR,
    QDRANT_API_KEY,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    QDRANT_HTTPS,
)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"⚠️  Unable to read {path}: {exc}")
        return None


def summarise_bookmarks() -> None:
    state = load_json(STATE_DIR / "raindrop_embed_state.json")
    print("📚 Bookmarks:")
    if not state:
        print("  (no state file)")
        return
    processed = len(state.get("processed_ids", []))
    failed = len(state.get("failed", {}))
    tokens = int(state.get("token_usage", 0))
    print(f"  Processed IDs: {processed}")
    print(f"  Failed IDs: {failed}")
    print(f"  Token usage: {tokens:,}")


def summarise_chats() -> None:
    state = load_json(STATE_DIR / "chat_embed_state.json")
    print("\n💬 Chats:")
    if not state:
        print("  (no state file)")
        return
    processed = sum(len(p or {}) for p in state.get("processed", {}).values())
    failed = len(state.get("failed", {}))
    tokens = int(state.get("token_usage", 0))
    print(f"  Conversations embedded: {processed}")
    print(f"  Failed conversations: {failed}")
    print(f"  Token usage: {tokens:,}")


def summarise_knowledge() -> None:
    state = load_json(STATE_DIR / "knowledge_code_state.json")
    print("\n🧠 Knowledge / Code:")
    if not state:
        print("  (no state file)")
        return
    processed_files = state.get("processed_files", [])
    failed_files = state.get("failed_files", {})
    token_usage = state.get("token_usage", {})
    total_tokens = token_usage.get("total", token_usage if isinstance(token_usage, int) else 0)
    created = state.get("created_at")
    updated = state.get("last_updated")
    print(f"  Files processed: {len(processed_files)}")
    print(f"  Failed files: {len(failed_files)}")
    if created:
        print(f"  Created: {created}")
    if updated:
        print(f"  Last updated: {updated}")
    print(f"  Token usage: {total_tokens:,}")
    if processed_files:
        sample = list(processed_files)[:5]
        print("  Sample:")
        for entry in sample:
            print(f"    - {Path(entry).name}")


def summarise_qdrant() -> None:
    print("\n📦 Qdrant Collections:")
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        api_key=QDRANT_API_KEY,
        https=QDRANT_HTTPS,
    )
    try:
        collections = client.get_collections().collections or []
        if not collections:
            print("  (none)")
            return
        for col in collections:
            name = col.name
            try:
                count = client.count(name, exact=True).count
            except Exception as exc:  # pragma: no cover - diagnostic helper
                print(f"  {name}: error ({exc})")
            else:
                print(f"  {name}: {count} embeddings")
    except Exception as exc:
        print(f"  Unable to query Qdrant: {exc}")


if __name__ == "__main__":  # pragma: no cover
    print(f"📁 Data root: {DATA_ROOT}")
    print(f"🗂  State dir: {STATE_DIR}")
    summarise_bookmarks()
    summarise_chats()
    summarise_knowledge()
    summarise_qdrant()
