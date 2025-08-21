#!/usr/bin/env python3
"""
Search interface for querying embedded content.
"""

import os
import sys
from pathlib import Path
import voyageai
import chromadb
from dotenv import load_dotenv
from typing import List, Dict
import json

# Load environment variables
load_dotenv()

# Initialize clients
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
client = chromadb.PersistentClient(path="./chroma_db")

def search(
    query: str,
    collection_name: str = None,
    n_results: int = 10,
    filter_dict: Dict = None
) -> List[Dict]:
    """
    Search embedded content.
    
    Args:
        query: Search query text
        collection_name: Specific collection to search (None = all)
        n_results: Number of results to return
        filter_dict: Metadata filters
    
    Returns:
        List of search results with documents, metadata, and distances
    """
    
    # Embed the query
    query_embedding = vo.embed(
        [query],
        model="voyage-context-3",
        input_type="query"
    ).embeddings[0]
    
    results = []
    
    if collection_name:
        collections = [collection_name]
    else:
        # Search all collections
        collections = ["knowledgebase", "code", "conversations", "git_history", "documents"]
    
    for coll_name in collections:
        try:
            collection = client.get_collection(coll_name)
            
            # Query the collection
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict
            )
            
            # Format results
            for i in range(len(result['ids'][0])):
                results.append({
                    'collection': coll_name,
                    'id': result['ids'][0][i],
                    'document': result['documents'][0][i][:500],  # First 500 chars
                    'metadata': result['metadatas'][0][i],
                    'distance': result['distances'][0][i]
                })
        except Exception as e:
            if "does not exist" not in str(e):
                print(f"Error searching {coll_name}: {e}")
    
    # Sort by distance (lower is better)
    results.sort(key=lambda x: x['distance'])
    
    return results[:n_results]

def search_timeline(query: str, start_date: str = None, end_date: str = None) -> List[Dict]:
    """
    Search with timeline filtering.
    """
    filter_dict = {}
    
    if start_date or end_date:
        filter_dict['timestamp'] = {}
        if start_date:
            filter_dict['timestamp']['$gte'] = start_date
        if end_date:
            filter_dict['timestamp']['$lte'] = end_date
    
    return search(query, filter_dict=filter_dict if filter_dict else None)

def search_code(query: str) -> List[Dict]:
    """Search specifically in code."""
    return search(query, collection_name="code")

def search_conversations(query: str) -> List[Dict]:
    """Search specifically in conversations."""
    return search(query, collection_name="conversations")

def search_git(query: str) -> List[Dict]:
    """Search specifically in git history."""
    return search(query, collection_name="git_history")

def print_results(results: List[Dict]):
    """Pretty print search results."""
    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"Result {i} | Collection: {result['collection']} | Distance: {result['distance']:.4f}")
        print(f"Source: {result['metadata'].get('source', 'Unknown')}")
        if 'timestamp' in result['metadata']:
            print(f"Timestamp: {result['metadata']['timestamp']}")
        print(f"\nContent preview:")
        print(result['document'])
        print("...")

def interactive_search():
    """Interactive search interface."""
    print("🔍 Voyage Embedding Search Interface")
    print("Commands: 'quit' to exit, 'help' for options")
    print("-" * 80)
    
    while True:
        query = input("\nSearch query: ").strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'help':
            print("""
Commands:
  [query]                - Search all collections
  code: [query]         - Search code only
  chat: [query]         - Search conversations only
  git: [query]          - Search git history only
  timeline: [query]     - Search with date filtering
  quit                  - Exit
            """)
            continue
        elif query.startswith("code:"):
            results = search_code(query[5:].strip())
        elif query.startswith("chat:"):
            results = search_conversations(query[5:].strip())
        elif query.startswith("git:"):
            results = search_git(query[4:].strip())
        elif query.startswith("timeline:"):
            # Simple timeline search
            results = search_timeline(query[9:].strip())
        else:
            results = search(query)
        
        if results:
            print_results(results)
        else:
            print("No results found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line search
        query = " ".join(sys.argv[1:])
        results = search(query)
        print_results(results)
    else:
        # Interactive mode
        interactive_search()