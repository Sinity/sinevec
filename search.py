#!/usr/bin/env python3
"""
Enhanced search interface with metadata filtering for unified collection.
"""

import os
import sys
from pathlib import Path
import voyageai
import chromadb
from dotenv import load_dotenv
from typing import List, Dict, Optional
import json
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize clients
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
client = chromadb.PersistentClient(path="./chroma_db")

def search_unified(
    query: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    channel: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    has_code: Optional[bool] = None,
    has_urls: Optional[bool] = None,
    participants: Optional[List[str]] = None,
    n_results: int = 10
) -> List[Dict]:
    """
    Search unified collection with rich metadata filtering.
    
    Args:
        query: Search query text
        category: Filter by category (irc, code, knowledge, conversations)
        subcategory: Filter by subcategory
        channel: Filter by IRC channel
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        has_code: Filter for code discussions
        has_urls: Filter for messages with URLs
        participants: Filter by participants
        n_results: Number of results to return
    """
    
    # Embed the query (contextualized model for better alignment)
    query_embedding = vo.embed(
        [query],
        model=os.environ.get("VOYAGE_QUERY_MODEL", "voyage-context-3"),
        input_type="query"
    ).embeddings[0]
    
    # Build filter
    where_filter = {}
    conditions = []
    
    if category:
        conditions.append({"category": category})
    if subcategory:
        conditions.append({"subcategory": subcategory})
    if channel:
        conditions.append({"channel": channel})
    if has_code is not None:
        conditions.append({"has_code": has_code})
    if has_urls is not None:
        conditions.append({"has_urls": has_urls})
    
    # Date range filter
    if date_from or date_to:
        date_filter = {}
        if date_from:
            date_filter["$gte"] = date_from
        if date_to:
            date_filter["$lte"] = date_to
        conditions.append({"date": date_filter})
    
    # Participant filter (any of the specified participants)
    if participants:
        # This would need contains operator which ChromaDB might not support directly
        # Would need to check ChromaDB's where clause capabilities
        pass
    
    # Combine conditions
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}
    
    # Try unified collection first
    try:
        collection = client.get_collection("unified")
    except:
        # Fall back to old collections
        print("⚠️ Unified collection not found, searching legacy collections...")
        return search_legacy(query, n_results)
    
    # Query the collection
    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        # Format results
        results = []
        for i in range(len(result['ids'][0])):
            results.append({
                'id': result['ids'][0][i],
                'document': result['documents'][0][i][:500],
                'metadata': result['metadatas'][0][i],
                'distance': result['distances'][0][i]
            })
        
        return results
        
    except Exception as e:
        print(f"Error querying unified collection: {e}")
        return []

def search_legacy(query: str, n_results: int = 10) -> List[Dict]:
    """Fallback to search old collections."""
    # Embed the query (contextualized by default)
    query_embedding = vo.embed(
        [query],
        model=os.environ.get("VOYAGE_QUERY_MODEL", "voyage-context-3"),
        input_type="query"
    ).embeddings[0]
    
    results = []
    
    for coll_name in ["knowledgebase", "code", "conversations"]:
        try:
            collection = client.get_collection(coll_name)
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            for i in range(len(result['ids'][0])):
                results.append({
                    'collection': coll_name,
                    'id': result['ids'][0][i],
                    'document': result['documents'][0][i][:500],
                    'metadata': result['metadatas'][0][i],
                    'distance': result['distances'][0][i]
                })
        except:
            pass
    
    # Sort by distance
    results.sort(key=lambda x: x['distance'])
    return results[:n_results]

def search_irc(
    query: str,
    channel: Optional[str] = None,
    date_from: Optional[str] = None,
    days_back: Optional[int] = None,
    has_code: Optional[bool] = None,
    n_results: int = 10
) -> List[Dict]:
    """Specialized IRC search."""
    
    # Calculate date range if days_back specified
    if days_back and not date_from:
        date_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    return search_unified(
        query=query,
        category="irc",
        channel=channel,
        date_from=date_from,
        has_code=has_code,
        n_results=n_results
    )

def print_results(results: List[Dict]):
    """Pretty print search results."""
    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"Result {i} | Distance: {result['distance']:.4f}")
        
        # Display metadata
        metadata = result.get('metadata', {})
        if 'category' in metadata:
            print(f"Category: {metadata['category']}")
        if 'channel' in metadata:
            print(f"Channel: {metadata['channel']}")
        if 'date' in metadata:
            print(f"Date: {metadata['date']}")
        if 'participants' in metadata and metadata['participants']:
            print(f"Participants: {', '.join(metadata['participants'][:5])}")
        if 'has_code' in metadata and metadata['has_code']:
            print("🔧 Contains code discussion")
        if 'urls' in metadata and metadata['urls']:
            print(f"🔗 URLs: {metadata['urls'][0]}")
        
        print(f"\nContent preview:")
        print(result['document'])
        print("...")

def interactive_search():
    """Interactive search interface with enhanced filtering."""
    print("🔍 Enhanced Search Interface")
    print("=" * 80)
    print("Commands:")
    print("  [query]                     - Search all content")
    print("  irc: [query]               - Search IRC logs")
    print("  irc #channel: [query]      - Search specific channel")
    print("  code: [query]              - Search code")
    print("  recent: [query]            - Search last 7 days")
    print("  today: [query]             - Search today's content")
    print("  with-code: [query]         - Search code discussions")
    print("  help                       - Show this help")
    print("  quit                       - Exit")
    print("-" * 80)
    
    while True:
        query_input = input("\nSearch: ").strip()
        
        if query_input.lower() == 'quit':
            break
        elif query_input.lower() == 'help':
            print(__doc__)
            continue
        
        # Parse command prefixes
        results = []
        
        if query_input.startswith("irc:"):
            query = query_input[4:].strip()
            # Check for channel specification
            if query.startswith("#"):
                parts = query.split(":", 1)
                if len(parts) == 2:
                    channel = parts[0].strip()
                    query = parts[1].strip()
                    results = search_irc(query, channel=channel)
                else:
                    results = search_irc(query)
            else:
                results = search_irc(query)
        
        elif query_input.startswith("code:"):
            query = query_input[5:].strip()
            results = search_unified(query, category="code")
        
        elif query_input.startswith("recent:"):
            query = query_input[7:].strip()
            results = search_irc(query, days_back=7)
        
        elif query_input.startswith("today:"):
            query = query_input[6:].strip()
            today = datetime.now().strftime('%Y-%m-%d')
            results = search_unified(query, date_from=today)
        
        elif query_input.startswith("with-code:"):
            query = query_input[10:].strip()
            results = search_unified(query, has_code=True)
        
        else:
            # General search
            results = search_unified(query_input)
        
        if results:
            print_results(results)
            print(f"\n📊 Found {len(results)} results")
        else:
            print("❌ No results found")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line search
        query = " ".join(sys.argv[1:])
        results = search_unified(query)
        print_results(results)
    else:
        # Interactive mode
        interactive_search()
