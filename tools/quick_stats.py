#!/usr/bin/env python3
"""Quick stats about embeddings."""

import json
import os
from pathlib import Path

from qdrant_client import QdrantClient

from sinevec.embed_utils import (
    QDRANT_API_KEY,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    QDRANT_HTTPS,
)

# Check state
state_file = Path("var/state/knowledge_code_state.json")
if state_file.exists():
    with open(state_file) as f:
        state = json.load(f)
    
    print("📊 V3 Embedding State:")
    print(f"  Created: {state['created_at'][:19]}")
    print(f"  Last updated: {state['last_updated'][:19]}")
    print(f"  Total tokens used: {state['token_usage']['total']:,}")
    print(f"  Files processed: {len(state['processed_files'])}")
    print(f"  Failed files: {len(state['failed_files'])}")
    
    # Sample of processed files
    processed = list(state['processed_files'])[:10]
    print("\n  Sample processed files:")
    for f in processed:
        print(f"    - {Path(f).name}")
    
    # Failed files
    if state['failed_files']:
        print("\n  Failed files:")
        for f, info in state['failed_files'].items():
            print(f"    - {Path(f).name}: {info['error'][:100]}")

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

# Check what directories were targeted
print("\n🎯 Target directories for knowledge/code:")
print(f"  - {os.environ.get('KB_DIR','data/knowledgebase')} -> collection: knowledgebase")
print(f"  - {os.environ.get('CODE_DIR','data/code')} -> collection: code")

# Analyze processed paths
if state_file.exists():
    kb_files = [f for f in state['processed_files'] if '/knowledgebase/' in f]
    code_files = [f for f in state['processed_files'] if '/project/sinex' in f]
    
    print(f"\n📁 Files by source:")
    print(f"  Knowledgebase: {len(kb_files)} files")
    print(f"  Sinex code: {len(code_files)} files")
    
    # File types
    extensions = {}
    for f in state['processed_files']:
        ext = Path(f).suffix or 'no_extension'
        extensions[ext] = extensions.get(ext, 0) + 1
    
    print(f"\n📄 File types processed:")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ext}: {count} files")
