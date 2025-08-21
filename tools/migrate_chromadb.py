#!/usr/bin/env python3
"""
Migrate data from corrupted ChromaDB to fresh database.
"""

import os
import json
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import struct

def export_from_sqlite(db_path: str) -> Dict:
    """Export all data from SQLite database."""
    print("\n📤 Exporting data from SQLite...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    exported_data = {
        'collections': {},
        'embeddings': [],
        'total_count': 0
    }
    
    try:
        # Get collection information
        cursor.execute("""
            SELECT id, name
            FROM collections
        """)
        collections = cursor.fetchall()
        
        for coll_id, coll_name in collections:
            print(f"  Collection: {coll_name} (ID: {coll_id})")
            exported_data['collections'][coll_id] = {
                'name': coll_name,
                'embeddings': []
            }
        
        # First, check the embeddings table structure
        cursor.execute("PRAGMA table_info(embeddings)")
        columns = cursor.fetchall()
        print(f"  Embeddings table columns: {[c[1] for c in columns]}")
        
        # Get embeddings with metadata
        cursor.execute("""
            SELECT 
                e.id,
                e.segment_id,
                e.embedding,
                e.seq_id,
                em.key,
                em.string_value,
                em.int_value,
                em.float_value
            FROM embeddings e
            LEFT JOIN embedding_metadata em ON e.id = em.id
            ORDER BY e.id, em.key
        """)
        
        rows = cursor.fetchall()
        
        # Group metadata by embedding ID
        embedding_data = {}
        for row in rows:
            emb_id = row[0]
            if emb_id not in embedding_data:
                embedding_data[emb_id] = {
                    'id': emb_id,
                    'segment_id': row[1],
                    'embedding': row[2],
                    'seq_id': row[3],
                    'metadata': {}
                }
            
            # Add metadata if present
            if row[4]:  # if key exists
                key = row[4]
                # Use whichever value is not None
                value = row[5] or row[6] or row[7]
                embedding_data[emb_id]['metadata'][key] = value
        
        # Get document content from fulltext search table
        cursor.execute("""
            SELECT id, document
            FROM embedding_fulltext_search
        """)
        
        documents = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Map segments to collections
        cursor.execute("""
            SELECT s.id, s.collection
            FROM segments s
        """)
        segment_to_collection = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Combine data
        for emb_id, data in embedding_data.items():
            if emb_id in documents:
                data['document'] = documents[emb_id]
            
            # Map segment to collection
            segment_id = data['segment_id']
            coll_id = segment_to_collection.get(segment_id)
            
            if coll_id and coll_id in exported_data['collections']:
                exported_data['collections'][coll_id]['embeddings'].append(data)
        
        # Summary
        for coll_id, coll_data in exported_data['collections'].items():
            count = len(coll_data['embeddings'])
            exported_data['total_count'] += count
            print(f"    → Exported {count} embeddings from {coll_data['name']}")
        
    except Exception as e:
        print(f"  ❌ Export error: {e}")
    finally:
        conn.close()
    
    return exported_data

def read_binary_embeddings(binary_dir: Path) -> np.ndarray:
    """Read embedding vectors from binary files."""
    print(f"\n📖 Reading binary embeddings from {binary_dir}...")
    
    try:
        # Read the data file
        data_file = binary_dir / "data_level0.bin"
        if not data_file.exists():
            print(f"  ❌ Data file not found: {data_file}")
            return None
        
        # Read header to understand format
        header_file = binary_dir / "header.bin"
        if header_file.exists():
            with open(header_file, 'rb') as f:
                header_data = f.read()
                print(f"  Header size: {len(header_data)} bytes")
        
        # Read index metadata
        metadata_file = binary_dir / "index_metadata.pickle"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                print(f"  Index metadata: dims={metadata.get('dimensionality', 'unknown')}")
                dims = metadata.get('dimensionality', 1024)
        else:
            dims = 1024  # Default Voyage-3 dimensions
        
        # Read binary data
        with open(data_file, 'rb') as f:
            data = f.read()
            
        # Calculate number of vectors
        bytes_per_float = 4
        bytes_per_vector = dims * bytes_per_float
        num_vectors = len(data) // bytes_per_vector
        
        print(f"  Found {num_vectors} vectors of {dims} dimensions")
        
        # Parse vectors
        vectors = []
        for i in range(num_vectors):
            start = i * bytes_per_vector
            end = start + bytes_per_vector
            vector_bytes = data[start:end]
            
            # Unpack floats
            vector = struct.unpack(f'{dims}f', vector_bytes)
            vectors.append(vector)
        
        return np.array(vectors)
        
    except Exception as e:
        print(f"  ❌ Error reading binary: {e}")
        return None

def save_export(exported_data: Dict, vectors: Dict[str, np.ndarray], output_file: str = "chromadb_export.json"):
    """Save exported data to file."""
    print(f"\n💾 Saving export to {output_file}...")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {
        'collections': exported_data['collections'],
        'total_count': exported_data['total_count'],
        'vectors': {}
    }
    
    for coll_id, vecs in vectors.items():
        if vecs is not None:
            serializable_data['vectors'][coll_id] = vecs.tolist()
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  ✅ Saved {file_size_mb:.2f} MB")

def main():
    """Main migration process."""
    print("🔄 ChromaDB Migration Tool")
    print("=" * 60)
    
    # Check if database exists
    db_path = "chroma_db_v3/chroma.sqlite3"
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    # Export from SQLite
    exported_data = export_from_sqlite(db_path)
    
    if exported_data['total_count'] == 0:
        print("\n❌ No data to export")
        return
    
    # Read binary embeddings for each collection
    vectors = {}
    
    # Find binary directories (UUIDs)
    binary_dirs = []
    for item in Path("chroma_db_v3").iterdir():
        if item.is_dir() and len(item.name) == 36 and '-' in item.name:  # UUID format
            binary_dirs.append(item)
    
    print(f"\n🔍 Found {len(binary_dirs)} binary directories")
    
    for i, binary_dir in enumerate(binary_dirs):
        print(f"\n[{i+1}/{len(binary_dirs)}] Processing {binary_dir.name}")
        vecs = read_binary_embeddings(binary_dir)
        if vecs is not None:
            # Match to collection (this is approximate - may need adjustment)
            # In practice, we'd need to map these properly
            vectors[binary_dir.name] = vecs
    
    # Save export
    save_export(exported_data, vectors)
    
    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print(f"📊 Summary:")
    print(f"  Collections: {len(exported_data['collections'])}")
    print(f"  Total embeddings: {exported_data['total_count']}")
    print(f"  Binary vectors: {sum(len(v) for v in vectors.values() if v is not None)}")
    
    print("\n💡 Next steps:")
    print("1. Review chromadb_export.json")
    print("2. Create fresh ChromaDB")
    print("3. Import data with import_chromadb.py")

if __name__ == "__main__":
    main()