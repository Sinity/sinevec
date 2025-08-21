#!/usr/bin/env python3
"""
Recover embeddings from corrupted ChromaDB and create fresh database.
Uses direct binary reading to avoid segfaults.
"""

import os
import json
import sqlite3
import pickle
import struct
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import voyageai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_metadata_from_sqlite(db_path: str) -> Dict:
    """Extract metadata from SQLite without using ChromaDB."""
    print("\n📤 Extracting metadata from SQLite...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    metadata_map = {}
    
    try:
        # First check what columns exist in embedding_fulltext_search
        cursor.execute("PRAGMA table_info(embedding_fulltext_search)")
        cols = cursor.fetchall()
        doc_col = 'embedding_fulltext_search' if any('embedding_fulltext_search' in str(c) for c in cols) else None
        
        # Get all embeddings with their IDs and documents
        if doc_col:
            cursor.execute(f"""
                SELECT 
                    e.id,
                    e.embedding_id,
                    e.segment_id,
                    efs.{doc_col},
                    s.collection,
                    c.name as collection_name
                FROM embeddings e
                LEFT JOIN embedding_fulltext_search efs ON e.id = efs.id
                JOIN segments s ON e.segment_id = s.id
                JOIN collections c ON s.collection = c.id
                ORDER BY e.id
            """)
        else:
            # Try without document column
            cursor.execute("""
                SELECT 
                    e.id,
                    e.embedding_id,
                    e.segment_id,
                    NULL as document,
                    s.collection,
                    c.name as collection_name
                FROM embeddings e
                JOIN segments s ON e.segment_id = s.id
                JOIN collections c ON s.collection = c.id
                ORDER BY e.id
            """)
        
        for row in cursor.fetchall():
            emb_id, embedding_id, segment_id, document, collection_id, collection_name = row
            
            # Parse the embedding_id to get source file
            source = embedding_id.split('#')[0] if '#' in embedding_id else embedding_id
            
            metadata_map[emb_id] = {
                'embedding_id': embedding_id,
                'document': document or '',
                'source': source,
                'collection_name': collection_name,
                'collection_id': collection_id,
                'segment_id': segment_id
            }
        
        # Get additional metadata
        cursor.execute("""
            SELECT 
                em.id,
                em.key,
                em.string_value,
                em.int_value,
                em.float_value
            FROM embedding_metadata em
        """)
        
        for row in cursor.fetchall():
            emb_id, key, str_val, int_val, float_val = row
            if emb_id in metadata_map:
                if 'metadata' not in metadata_map[emb_id]:
                    metadata_map[emb_id]['metadata'] = {}
                # Use whichever value is not None
                value = str_val or int_val or float_val
                metadata_map[emb_id]['metadata'][key] = value
        
        print(f"  ✅ Extracted metadata for {len(metadata_map)} embeddings")
        
    except Exception as e:
        print(f"  ❌ Error extracting metadata: {e}")
    finally:
        conn.close()
    
    return metadata_map

def read_hnsw_vectors(binary_dir: Path) -> Optional[List]:
    """Read vectors from HNSW binary files."""
    print(f"\n📖 Reading vectors from {binary_dir.name}...")
    
    try:
        # Read index metadata to get dimensions
        metadata_file = binary_dir / "index_metadata.pickle"
        dims = None
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    dims = metadata.get('dimensionality') or metadata.get('dim') or metadata.get('dimensions')
                    print(f"  Vector dimensions from metadata: {dims}")
            except Exception as e:
                print(f"  Warning: Could not read metadata: {e}")
        
        if dims is None:
            # Try to infer from file size
            data_file = binary_dir / "data_level0.bin"
            if data_file.exists():
                file_size = data_file.stat().st_size
                # Assume 1024 dimensions (Voyage-3 default)
                dims = 1024
                bytes_per_vector = dims * 4
                estimated_vectors = file_size // bytes_per_vector
                print(f"  Inferred dimensions: {dims} (file has ~{estimated_vectors} vectors)")
        
        # Read data file
        data_file = binary_dir / "data_level0.bin"
        if not data_file.exists():
            print(f"  ❌ Data file not found")
            return None
        
        with open(data_file, 'rb') as f:
            data = f.read()
        
        # Parse vectors (4 bytes per float)
        bytes_per_vector = dims * 4
        num_vectors = len(data) // bytes_per_vector
        
        vectors = []
        for i in range(num_vectors):
            start = i * bytes_per_vector
            end = start + bytes_per_vector
            vector_bytes = data[start:end]
            
            # Unpack as floats
            vector = list(struct.unpack(f'{dims}f', vector_bytes))
            vectors.append(vector)
        
        print(f"  ✅ Read {len(vectors)} vectors")
        return vectors
        
    except Exception as e:
        print(f"  ❌ Error reading vectors: {e}")
        return None

def map_vectors_to_metadata(metadata_map: Dict, segment_vectors: Dict) -> List[Dict]:
    """Map vectors to their metadata."""
    print("\n🔗 Mapping vectors to metadata...")
    
    complete_data = []
    
    # Group metadata by segment
    segment_metadata = {}
    for emb_id, meta in metadata_map.items():
        segment_id = meta['segment_id']
        if segment_id not in segment_metadata:
            segment_metadata[segment_id] = []
        segment_metadata[segment_id].append((emb_id, meta))
    
    # Match vectors to metadata
    for segment_id, vectors in segment_vectors.items():
        if segment_id in segment_metadata:
            meta_list = segment_metadata[segment_id]
            
            # Match by order (embeddings are stored sequentially)
            for i, (emb_id, meta) in enumerate(meta_list):
                if i < len(vectors):
                    complete_data.append({
                        'id': meta['embedding_id'],
                        'vector': vectors[i],
                        'document': meta['document'],
                        'metadata': meta.get('metadata', {}),
                        'collection': meta['collection_name'],
                        'source': meta['source']
                    })
    
    print(f"  ✅ Mapped {len(complete_data)} complete embeddings")
    return complete_data

def save_recovery_data(data: List[Dict], output_file: str = "recovered_embeddings.json"):
    """Save recovered data to file."""
    print(f"\n💾 Saving recovered data to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  ✅ Saved {file_size_mb:.2f} MB")

def recreate_in_fresh_chromadb(data: List[Dict]):
    """Recreate embeddings in a fresh ChromaDB."""
    print("\n🔄 Recreating in fresh ChromaDB...")
    
    # Create new database
    import chromadb
    
    # Backup old database
    if Path("chroma_db_v3").exists():
        backup_name = "chroma_db_v3_corrupted"
        if Path(backup_name).exists():
            shutil.rmtree(backup_name)
        shutil.move("chroma_db_v3", backup_name)
        print(f"  ✅ Backed up old database to {backup_name}")
    
    # Create fresh client
    client = chromadb.PersistentClient(path="./chroma_db_v3")
    
    # Group by collection
    collections_data = {}
    for item in data:
        coll_name = item['collection']
        if coll_name not in collections_data:
            collections_data[coll_name] = []
        collections_data[coll_name].append(item)
    
    # Recreate collections and add data
    for coll_name, items in collections_data.items():
        print(f"\n  Creating collection: {coll_name}")
        
        try:
            collection = client.create_collection(name=coll_name)
            
            # Add in batches
            batch_size = 100
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                
                ids = [item['id'] for item in batch]
                embeddings = [item['vector'] for item in batch]
                documents = [item['document'][:65536] if item['document'] else '' for item in batch]
                metadatas = []
                
                for item in batch:
                    metadata = {
                        'source': item['source'] or 'unknown',
                        'file_type': 'recovered'
                    }
                    # Add any additional metadata, filtering out reserved keys and None values
                    if item.get('metadata'):
                        for key, value in item['metadata'].items():
                            # Skip ChromaDB reserved keys and None values
                            if not key.startswith('chroma:') and value is not None:
                                # Convert to appropriate type
                                if isinstance(value, bool):
                                    metadata[key] = value
                                elif isinstance(value, (int, float)):
                                    metadata[key] = value
                                else:
                                    metadata[key] = str(value)
                    metadatas.append(metadata)
                
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                print(f"    Added batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
            
            print(f"  ✅ Recreated {coll_name} with {len(items)} embeddings")
            
        except Exception as e:
            print(f"  ❌ Error recreating {coll_name}: {e}")

def main():
    """Main recovery process."""
    print("🚑 ChromaDB Recovery Tool")
    print("=" * 60)
    
    # Check if corrupted database exists
    if not Path("chroma_db_v3").exists():
        print("❌ chroma_db_v3 not found!")
        return
    
    # Extract metadata
    metadata_map = extract_metadata_from_sqlite("chroma_db_v3/chroma.sqlite3")
    
    if not metadata_map:
        print("\n❌ No metadata to recover")
        return
    
    # Find and read vector files
    segment_vectors = {}
    
    # Get the mapping between metadata segments and vector segments
    conn = sqlite3.connect("chroma_db_v3/chroma.sqlite3")
    cursor = conn.cursor()
    
    # Get collection to vector segment mapping
    cursor.execute("""
        SELECT s1.collection, s1.id as metadata_segment, s2.id as vector_segment
        FROM segments s1
        JOIN segments s2 ON s1.collection = s2.collection
        WHERE s1.type = 'urn:chroma:segment/metadata/sqlite'
        AND s2.type = 'urn:chroma:segment/vector/hnsw-local-persisted'
    """)
    
    metadata_to_vector = {}
    for row in cursor.fetchall():
        collection_id, meta_seg, vec_seg = row
        metadata_to_vector[meta_seg] = vec_seg
    
    conn.close()
    
    print("\n🗺️ Segment mapping:")
    for meta, vec in metadata_to_vector.items():
        print(f"  {meta[:8]}... → {vec[:8]}...")
    
    # Read vectors from vector segments
    for meta_seg, vec_seg in metadata_to_vector.items():
        binary_dir = Path(f"chroma_db_v3/{vec_seg}")
        if binary_dir.exists():
            vectors = read_hnsw_vectors(binary_dir)
            if vectors:
                # Store under metadata segment ID for mapping
                segment_vectors[meta_seg] = vectors
    
    # Map vectors to metadata
    complete_data = map_vectors_to_metadata(metadata_map, segment_vectors)
    
    if not complete_data:
        print("\n❌ Could not recover complete embeddings")
        return
    
    # Save recovered data
    save_recovery_data(complete_data)
    
    # Ask before recreating
    print("\n" + "=" * 60)
    print(f"✅ Recovery complete!")
    print(f"📊 Recovered {len(complete_data)} embeddings")
    print("\nData saved to recovered_embeddings.json")
    print("\nTo recreate in fresh ChromaDB, run:")
    print("  python recover_embeddings.py --recreate")

if __name__ == "__main__":
    import sys
    
    if "--recreate" in sys.argv:
        # Load recovered data and recreate
        if Path("recovered_embeddings.json").exists():
            with open("recovered_embeddings.json", 'r') as f:
                data = json.load(f)
            recreate_in_fresh_chromadb(data)
        else:
            print("❌ recovered_embeddings.json not found. Run without --recreate first.")
    else:
        main()