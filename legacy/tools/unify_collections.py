#!/usr/bin/env python3
"""
Unify all collections into a single 'unified' collection with proper metadata.
"""

import chromadb
from datetime import datetime

def migrate_to_unified():
    """Migrate all collections to unified with proper metadata."""
    
    print("🔄 Unifying ChromaDB Collections")
    print("=" * 60)
    
    client = chromadb.PersistentClient(path="./chroma_db_v3")
    
    # Get or create unified collection
    try:
        unified = client.get_collection("unified")
        print("✅ Using existing unified collection")
    except:
        unified = client.create_collection("unified")
        print("✅ Created unified collection")
    
    # Collections to migrate
    collections_to_migrate = {
        'knowledgebase': {
            'category': 'knowledgebase',
            'subcategory_extract': lambda src: src.split('/')[-2] if '/' in src else 'general',
            'file_type': 'markdown'
        },
        'code': {
            'category': 'code', 
            'subcategory_extract': lambda src: src.split('/')[2] if src.startswith('/realm/project/') and len(src.split('/')) > 2 else 'general',
            'file_type': 'code'
        }
    }
    
    total_migrated = 0
    
    for col_name, config in collections_to_migrate.items():
        try:
            source_col = client.get_collection(col_name)
            count = source_col.count()
            
            if count == 0:
                print(f"\n⚠️ {col_name}: Empty, skipping")
                continue
            
            print(f"\n📦 Migrating {col_name}: {count} items")
            
            # Get all data from source collection
            batch_size = 1000
            for offset in range(0, count, batch_size):
                limit = min(batch_size, count - offset)
                data = source_col.get(
                    limit=limit,
                    offset=offset,
                    include=['embeddings', 'documents', 'metadatas']
                )
                
                if not data['ids']:
                    continue
                
                # Update metadata for unified structure
                new_metadatas = []
                new_ids = []
                
                for i, (id_val, meta) in enumerate(zip(data['ids'], data['metadatas'])):
                    # Extract source info
                    source = meta.get('source', '')
                    
                    # Create unified metadata
                    new_meta = {
                        'category': config['category'],
                        'subcategory': config['subcategory_extract'](source),
                        'source': source,
                        'file_type': meta.get('file_type') or config['file_type'],
                        'original_collection': col_name
                    }
                    
                    # Preserve any additional metadata
                    for key, value in meta.items():
                        if key not in ['category', 'subcategory', 'source', 'file_type'] and value is not None:
                            # Convert to safe types
                            if isinstance(value, bool):
                                new_meta[key] = value
                            elif isinstance(value, (int, float)):
                                new_meta[key] = value
                            else:
                                new_meta[key] = str(value)
                    
                    new_metadatas.append(new_meta)
                    
                    # Create unique ID for unified collection
                    new_ids.append(f"{col_name}#{id_val}")
                
                # Add to unified collection
                try:
                    # First try to delete if exists (in case of re-run)
                    try:
                        unified.delete(ids=new_ids)
                    except:
                        pass
                    
                    # Add the batch
                    unified.add(
                        ids=new_ids,
                        embeddings=data['embeddings'],
                        documents=data['documents'],
                        metadatas=new_metadatas
                    )
                    
                    total_migrated += len(new_ids)
                    print(f"  ✅ Migrated batch {offset//batch_size + 1}/{(count-1)//batch_size + 1}")
                    
                except Exception as e:
                    print(f"  ❌ Error migrating batch: {e}")
            
        except Exception as e:
            print(f"\n❌ Error with {col_name}: {e}")
    
    # Final report
    print("\n" + "=" * 60)
    print("📊 MIGRATION COMPLETE")
    print("=" * 60)
    
    final_count = unified.count()
    print(f"Total items in unified collection: {final_count}")
    
    # Analyze final structure
    print("\n📋 Content breakdown:")
    sample = unified.get(limit=1000, include=['metadatas'])
    
    categories = {}
    for meta in sample['metadatas']:
        cat = meta.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, cnt in sorted(categories.items()):
        estimated_total = int((cnt / len(sample['metadatas'])) * final_count)
        print(f"  {cat}: ~{estimated_total} items")
    
    print("\n✅ All collections unified!")
    print("\n💡 You can now:")
    print("  - Search across all content with: collection.query()")
    print("  - Filter by category: where={'category': 'code'}")
    print("  - Filter by subcategory: where={'subcategory': 'lesswrong'}")
    print("  - Combine filters: where={'$and': [{'category': 'reddit'}, {'subcategory': 'r/programming'}]}")
    
    # Optionally delete old collections
    print("\n🗑️ Old collections can be deleted with:")
    print("  client.delete_collection('knowledgebase')")
    print("  client.delete_collection('code')")

if __name__ == "__main__":
    migrate_to_unified()