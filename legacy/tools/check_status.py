#!/usr/bin/env python3
"""Check the status of embeddings."""

import chromadb
import json
from pathlib import Path

# Load token usage
token_log = Path("logs/token_usage.json")
if token_log.exists():
    with open(token_log) as f:
        token_data = json.load(f)
    print(f"📊 Token Usage Status:")
    print(f"  Total used: {token_data['total_tokens']:,}")
    print(f"  Budget: {token_data['budget']:,}")
    print(f"  Remaining: {token_data['remaining']:,}")
    print(f"  Usage: {(token_data['total_tokens'] / token_data['budget']) * 100:.4f}%")
    print()

# Check collections
client = chromadb.PersistentClient(path="./chroma_db")

print("📦 Collection Status:")
collections_info = []

for name in ["knowledgebase", "code", "conversations", "git_history", "documents"]:
    try:
        collection = client.get_collection(name)
        count = collection.count()
        collections_info.append((name, count))
    except:
        collections_info.append((name, 0))

for name, count in collections_info:
    print(f"  {name}: {count} chunks embedded")

print(f"\n✅ Total chunks: {sum(c for _, c in collections_info)}")