#!/usr/bin/env python3
"""
Attempt to recover data from corrupted ChromaDB.
"""

import os
import json
import shutil
from pathlib import Path
import sqlite3
import pickle

def attempt_direct_sqlite_recovery():
    """Try to recover data directly from SQLite database."""
    print("\n🔍 Attempting direct SQLite recovery...")
    
    db_path = Path("chroma_db_v3/chroma.sqlite3")
    if not db_path.exists():
        print("❌ SQLite database not found")
        return None
    
    recovered_data = {
        'collections': [],
        'embeddings': [],
        'documents': [],
        'metadatas': []
    }
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"  Found tables: {[t[0] for t in tables]}")
        
        # Try to get collections
        try:
            cursor.execute("SELECT * FROM collections")
            collections = cursor.fetchall()
            print(f"  Found {len(collections)} collections")
            recovered_data['collections'] = collections
        except Exception as e:
            print(f"  ⚠️ Could not read collections: {e}")
        
        # Try to get embeddings
        try:
            cursor.execute("SELECT * FROM embeddings")
            embeddings = cursor.fetchall()
            print(f"  Found {len(embeddings)} embeddings")
            recovered_data['embeddings'] = embeddings
        except Exception as e:
            print(f"  ⚠️ Could not read embeddings: {e}")
        
        # Try to get documents
        try:
            cursor.execute("SELECT * FROM documents")
            documents = cursor.fetchall()
            print(f"  Found {len(documents)} documents")
            recovered_data['documents'] = documents
        except Exception as e:
            print(f"  ⚠️ Could not read documents: {e}")
        
        conn.close()
        return recovered_data
        
    except Exception as e:
        print(f"  ❌ SQLite recovery failed: {e}")
        return None

def check_parquet_files():
    """Check for Parquet files that might contain embeddings."""
    print("\n🔍 Checking for Parquet files...")
    
    parquet_files = []
    for root, dirs, files in os.walk("chroma_db_v3"):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    if parquet_files:
        print(f"  Found {len(parquet_files)} Parquet files:")
        for pf in parquet_files[:5]:  # Show first 5
            size_mb = os.path.getsize(pf) / (1024 * 1024)
            print(f"    {pf}: {size_mb:.2f} MB")
        
        # Try to read with pandas/pyarrow
        try:
            import pandas as pd
            import pyarrow.parquet as pq
            
            for pf in parquet_files[:1]:  # Try first file
                try:
                    df = pd.read_parquet(pf)
                    print(f"\n  ✅ Successfully read {pf}")
                    print(f"     Shape: {df.shape}")
                    print(f"     Columns: {list(df.columns)}")
                    return parquet_files
                except Exception as e:
                    print(f"  ⚠️ Could not read {pf}: {e}")
        except ImportError:
            print("  ⚠️ pandas/pyarrow not installed, cannot read Parquet files")
    else:
        print("  No Parquet files found")
    
    return parquet_files

def check_binary_files():
    """Check for binary/pickle files."""
    print("\n🔍 Checking for binary data files...")
    
    binary_files = []
    for root, dirs, files in os.walk("chroma_db_v3"):
        for file in files:
            if file.endswith(('.pkl', '.pickle', '.bin', '.npy')):
                binary_files.append(os.path.join(root, file))
    
    if binary_files:
        print(f"  Found {len(binary_files)} binary files")
        for bf in binary_files[:5]:
            size_mb = os.path.getsize(bf) / (1024 * 1024)
            print(f"    {bf}: {size_mb:.2f} MB")
    else:
        print("  No binary files found")
    
    return binary_files

def analyze_directory_structure():
    """Analyze the ChromaDB directory structure."""
    print("\n📁 Analyzing ChromaDB directory structure...")
    
    total_size = 0
    file_types = {}
    
    for root, dirs, files in os.walk("chroma_db_v3"):
        # Skip deep nesting
        depth = root.replace("chroma_db_v3", "").count(os.sep)
        if depth < 3:
            print(f"  {' ' * depth}{os.path.basename(root)}/ ({len(files)} files)")
        
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            total_size += size
            
            ext = os.path.splitext(file)[1]
            if ext not in file_types:
                file_types[ext] = {'count': 0, 'size': 0}
            file_types[ext]['count'] += 1
            file_types[ext]['size'] += size
    
    print(f"\n📊 Summary:")
    print(f"  Total size: {total_size / (1024**2):.2f} MB")
    print(f"  File types:")
    for ext, info in sorted(file_types.items(), key=lambda x: x[1]['size'], reverse=True)[:10]:
        print(f"    {ext or '(no ext)'}: {info['count']} files, {info['size'] / (1024**2):.2f} MB")

def try_chromadb_migration():
    """Try to use ChromaDB's migration tools if available."""
    print("\n🔧 Checking ChromaDB version and migration options...")
    
    # Check if there's a version file
    version_file = Path("chroma_db_v3/.chroma_version")
    if version_file.exists():
        with open(version_file, 'r') as f:
            version = f.read().strip()
            print(f"  ChromaDB version: {version}")
    else:
        print("  No version file found")
    
    # Check for metadata
    metadata_file = Path("chroma_db_v3/metadata.json")
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            print(f"  Metadata: {json.dumps(metadata, indent=2)}")

def export_embedding_state():
    """Try to export the embedding state files."""
    print("\n💾 Checking embedding state files...")
    
    state_files = [
        "embedding_state_v3.json",
        "embedding_state.json",
        "ai_embed_state.json",
        "reddit_embed_state.json"
    ]
    
    for sf in state_files:
        if Path(sf).exists():
            with open(sf, 'r') as f:
                data = json.load(f)
                processed = len(data.get('processed_files', []))
                failed = len(data.get('failed_files', []))
                tokens = data.get('total_tokens', 0)
                print(f"  {sf}:")
                print(f"    Processed: {processed} files")
                print(f"    Failed: {failed} files")
                print(f"    Tokens used: {tokens:,}")

def main():
    """Main recovery process."""
    print("🚑 ChromaDB Recovery Tool")
    print("=" * 60)
    
    if not Path("chroma_db_v3").exists():
        print("❌ chroma_db_v3 directory not found!")
        return
    
    # Analyze structure
    analyze_directory_structure()
    
    # Check different recovery methods
    export_embedding_state()
    
    # Try SQLite recovery
    sqlite_data = attempt_direct_sqlite_recovery()
    
    # Check for Parquet files
    parquet_files = check_parquet_files()
    
    # Check for binary files
    binary_files = check_binary_files()
    
    # Check migration options
    try_chromadb_migration()
    
    print("\n" + "=" * 60)
    print("📋 Recovery Report:")
    
    if sqlite_data and any(sqlite_data.values()):
        print("✅ SQLite data partially recoverable")
    else:
        print("❌ SQLite data not recoverable")
    
    if parquet_files:
        print(f"✅ Found {len(parquet_files)} Parquet files (may contain embeddings)")
    else:
        print("⚠️ No Parquet files found")
    
    if binary_files:
        print(f"✅ Found {len(binary_files)} binary files")
    
    print("\n💡 Recommendations:")
    print("1. The embedding state files show what was already processed")
    print("2. We can resume from where we left off using state files")
    print("3. Consider starting fresh and re-embedding only missing data")

if __name__ == "__main__":
    main()