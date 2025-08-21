#!/usr/bin/env python3
"""Quick stats about embeddings."""

import json
from pathlib import Path
import sqlite3

# Check v3 state
state_file = Path("embedding_state_v3.json")
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

# Check ChromaDB directly via SQLite
print("\n📦 ChromaDB Collections (v3):")
db_path = Path("chroma_db_v3/chroma.sqlite3")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get collections
    cursor.execute("SELECT name, id FROM collections")
    collections = cursor.fetchall()
    
    for name, coll_id in collections:
        # Count embeddings in each collection - check table structure first
        try:
            cursor.execute("SELECT COUNT(*) FROM embeddings_queue WHERE collection_id = ?", (coll_id,))
            count = cursor.fetchone()[0]
            print(f"  {name}: {count} embeddings")
        except sqlite3.OperationalError:
            # Try alternate table structure
            cursor.execute("SELECT COUNT(*) FROM embedding_metadata WHERE collection_id = ?", (coll_id,))
            count = cursor.fetchone()[0]
            print(f"  {name}: {count} embeddings")
    
    conn.close()

# Check what directories were targeted
print("\n🎯 Target directories from embed_v3.py:")
print("  - /realm/knowledgebase -> collection: knowledgebase")
print("  - /realm/project/sinex -> collection: code")

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