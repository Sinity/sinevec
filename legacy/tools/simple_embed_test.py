#!/usr/bin/env python3
"""
Simple test of embedding functionality without tqdm.
"""

import os
import voyageai
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
client = chromadb.PersistentClient(path="./chroma_db_v3")

def test_embedding():
    """Test basic embedding functionality."""
    
    print("Testing Voyage AI embedding...")
    
    # Test text
    test_text = "This is a test of the Voyage embedding system. It should work without issues now that we have fixed the library dependencies."
    
    try:
        # Create embedding
        embeds_obj = vo.embed(
            [test_text],
            model="voyage-3",
            input_type="document"
        )
        
        print(f"✅ Embedding successful!")
        print(f"  Embedding dimensions: {len(embeds_obj.embeddings[0])}")
        print(f"  Tokens used: {embeds_obj.total_tokens}")
        
        # Try to get or create collection
        try:
            collection = client.get_collection("test_collection")
            print("✅ Got existing test collection")
        except:
            collection = client.create_collection("test_collection")
            print("✅ Created new test collection")
        
        # Add to collection
        collection.add(
            embeddings=embeds_obj.embeddings,
            documents=[test_text],
            metadatas=[{"type": "test"}],
            ids=["test_1"]
        )
        
        print("✅ Added to ChromaDB successfully!")
        
        # Query test
        query_embeds = vo.embed(
            ["test query"],
            model="voyage-3",
            input_type="query"
        )
        
        results = collection.query(
            query_embeddings=query_embeds.embeddings,
            n_results=1
        )
        
        print("✅ Query successful!")
        print(f"  Found {len(results['documents'][0])} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing embedding system...")
    print("-" * 40)
    
    if test_embedding():
        print("\n🎉 All tests passed! System is working.")
    else:
        print("\n⚠️ Tests failed. Check errors above.")