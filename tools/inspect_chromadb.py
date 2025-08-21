#!/usr/bin/env python3
"""
Inspect ChromaDB structure in detail.
"""

import sqlite3
from pathlib import Path

def inspect_database():
    """Inspect the database structure."""
    db_path = "chroma_db_v3/chroma.sqlite3"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🔍 Database Structure Analysis")
    print("=" * 60)
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table_name in [t[0] for t in tables]:
        print(f"\n📋 Table: {table_name}")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        print("  Columns:")
        for col in columns:
            print(f"    {col[1]:<20} {col[2]:<15} {'NOT NULL' if col[3] else 'NULL':<10} {'PK' if col[5] else ''}")
        
        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  Row count: {count}")
            
            # Show sample data for important tables
            if table_name in ['embeddings', 'collections', 'segments'] and count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                samples = cursor.fetchall()
                print(f"  Sample data:")
                for sample in samples:
                    print(f"    {sample}")
        except Exception as e:
            print(f"  Error reading table: {e}")
    
    # Check relationships
    print("\n🔗 Key Relationships:")
    
    # Segments to collections
    cursor.execute("""
        SELECT s.id, s.collection, c.name
        FROM segments s
        JOIN collections c ON s.collection = c.id
        LIMIT 5
    """)
    results = cursor.fetchall()
    print("  Segments → Collections:")
    for r in results:
        print(f"    Segment {r[0]} → Collection {r[2]}")
    
    # Embeddings to segments
    cursor.execute("""
        SELECT e.id, e.segment_id, e.embedding_id, s.collection
        FROM embeddings e
        JOIN segments s ON e.segment_id = s.id
        LIMIT 5
    """)
    results = cursor.fetchall()
    print("\n  Embeddings → Segments:")
    for r in results:
        print(f"    Embedding {r[0]} (ID: {r[2]}) → Segment {r[1]} → Collection {r[3]}")
    
    conn.close()

if __name__ == "__main__":
    inspect_database()