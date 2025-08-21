#!/usr/bin/env python3
"""
Simple embedding script without tqdm to avoid display issues.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import voyageai
import chromadb
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

# Initialize clients
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
client = chromadb.PersistentClient(path="./chroma_db_v3")

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count actual tokens using tiktoken."""
    return len(tokenizer.encode(text))

def load_state(state_file: str = "embedding_state_simple.json") -> Dict:
    """Load embedding state."""
    if Path(state_file).exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {
        "processed_files": [],
        "failed_files": [],
        "total_tokens": 0,
        "last_updated": None
    }

def save_state(state: Dict, state_file: str = "embedding_state_simple.json"):
    """Save embedding state."""
    state["last_updated"] = datetime.now().isoformat()
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def embed_ai_conversations():
    """Embed AI conversations."""
    print("\n🤖 Embedding AI Conversations")
    print("-" * 40)
    
    state = load_state("ai_embed_state.json")
    
    # Parse conversations
    claude_path = Path("/realm/data/chatlog/claude/claude_conversations_20241007.json")
    chatgpt_path = Path("/realm/data/chatlog/chatgpt/conversations.json")
    
    conversations = []
    
    # Load Claude conversations
    if claude_path.exists():
        with open(claude_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for conv in data:
                    messages = []
                    for msg in conv.get('chat_messages', []):
                        messages.append({
                            'role': msg.get('sender', 'unknown'),
                            'content': msg.get('text', '')
                        })
                    if messages:
                        conversations.append({
                            'source': 'claude',
                            'title': conv.get('name', 'Untitled'),
                            'messages': messages
                        })
    
    # Load ChatGPT conversations
    if chatgpt_path.exists():
        with open(chatgpt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for conv_id, conv in data.items():
                messages = []
                for msg_id, msg in conv.get('mapping', {}).items():
                    msg_data = msg.get('message', {})
                    if msg_data and msg_data.get('content'):
                        content = msg_data['content']
                        if content.get('content_type') == 'text':
                            parts = content.get('parts', [])
                            if parts and isinstance(parts[0], str):
                                messages.append({
                                    'role': msg_data.get('author', {}).get('role', 'unknown'),
                                    'content': parts[0]
                                })
                if messages:
                    conversations.append({
                        'source': 'chatgpt',
                        'title': conv.get('title', 'Untitled'),
                        'messages': messages
                    })
    
    print(f"Found {len(conversations)} conversations to embed")
    
    # Get or create collection
    try:
        collection = client.get_collection("unified")
    except:
        collection = client.create_collection("unified")
    
    total_embedded = 0
    total_tokens = state.get("total_tokens", 0)
    
    for i, conv in enumerate(conversations):
        conv_id = f"{conv['source']}#{i}"
        
        # Skip if already processed
        if conv_id in state.get("processed_files", []):
            continue
        
        print(f"\r[{i+1}/{len(conversations)}] Embedding: {conv['title'][:50]}...", end="", flush=True)
        
        # Build conversation text
        conv_text = f"Title: {conv['title']}\n\n"
        for msg in conv['messages']:
            role = msg['role'].upper()
            content = msg['content']
            conv_text += f"{role}:\n{content}\n\n"
        
        # Check token count
        tokens = count_tokens(conv_text)
        if tokens > 30000:
            conv_text = conv_text[:100000]  # Truncate if too long
        
        try:
            # Embed
            embeds_obj = vo.embed(
                [conv_text],
                model="voyage-3",
                input_type="document"
            )
            
            # Add to collection
            collection.add(
                embeddings=embeds_obj.embeddings,
                documents=[conv_text[:65536]],
                metadatas=[{
                    'source': f"/realm/data/chatlog/{conv['source']}",
                    'category': 'ai_conversation',
                    'subcategory': conv['source'],
                    'title': conv['title'],
                    'file_type': 'conversation'
                }],
                ids=[conv_id]
            )
            
            total_embedded += 1
            total_tokens += embeds_obj.total_tokens
            
            # Update state
            state["processed_files"].append(conv_id)
            state["total_tokens"] = total_tokens
            
            # Save state periodically
            if i % 10 == 0:
                save_state(state, "ai_embed_state.json")
            
        except Exception as e:
            print(f"\n  ⚠️ Error: {str(e)[:100]}")
            state["failed_files"].append(conv_id)
            time.sleep(20)  # Rate limit delay
    
    save_state(state, "ai_embed_state.json")
    print(f"\n✅ Embedded {total_embedded} conversations")
    print(f"   Total tokens used: {total_tokens:,}")

def embed_reddit_comments():
    """Embed Reddit comments."""
    print("\n🎯 Embedding Reddit Comments")
    print("-" * 40)
    
    state = load_state("reddit_embed_state.json")
    
    reddit_file = Path("/realm/data/reddit_comments/formatted.txt")
    if not reddit_file.exists():
        print("❌ Reddit file not found")
        return
    
    # Parse comments
    with open(reddit_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    import re
    comments_raw = re.split(r'🌀🌀🌀\n?', content)
    comments = [c.strip() for c in comments_raw if c.strip()]
    
    print(f"Found {len(comments)} comments to embed")
    
    # Get or create collection
    try:
        collection = client.get_collection("unified")
    except:
        collection = client.create_collection("unified")
    
    total_embedded = 0
    total_tokens = state.get("total_tokens", 0)
    
    for i, comment in enumerate(comments):
        comment_id = f"reddit#{i}"
        
        # Skip if already processed
        if comment_id in state.get("processed_files", []):
            continue
        
        print(f"\r[{i+1}/{len(comments)}] Embedding comment {i}...", end="", flush=True)
        
        # Truncate if too long
        if count_tokens(comment) > 8000:
            comment = comment[:30000]
        
        try:
            # Embed
            embeds_obj = vo.embed(
                [comment],
                model="voyage-3",
                input_type="document"
            )
            
            # Parse metadata from comment
            subreddit = "unknown"
            subreddit_match = re.search(r'/r/(\w+)/', comment)
            if subreddit_match:
                subreddit = f"r/{subreddit_match.group(1)}"
            
            # Add to collection
            collection.add(
                embeddings=embeds_obj.embeddings,
                documents=[comment[:65536]],
                metadatas=[{
                    'source': str(reddit_file),
                    'category': 'reddit',
                    'subcategory': subreddit,
                    'comment_index': i,
                    'file_type': 'reddit_comment'
                }],
                ids=[comment_id]
            )
            
            total_embedded += 1
            total_tokens += embeds_obj.total_tokens
            
            # Update state
            state["processed_files"].append(comment_id)
            state["total_tokens"] = total_tokens
            
            # Save state periodically
            if i % 25 == 0:
                save_state(state, "reddit_embed_state.json")
            
        except Exception as e:
            print(f"\n  ⚠️ Error: {str(e)[:100]}")
            state["failed_files"].append(comment_id)
            time.sleep(20)  # Rate limit delay
    
    save_state(state, "reddit_embed_state.json")
    print(f"\n✅ Embedded {total_embedded} comments")
    print(f"   Total tokens used: {total_tokens:,}")

def main():
    """Main embedding pipeline."""
    print("🚀 Simple Embedding Pipeline")
    print("=" * 60)
    
    # Check current status
    print("\n📊 Checking current status...")
    try:
        collections = client.list_collections()
        for col in collections:
            count = col.count()
            print(f"  {col.name}: {count} chunks")
    except:
        print("  No collections found")
    
    # Embed AI conversations
    embed_ai_conversations()
    
    # Embed Reddit comments
    embed_reddit_comments()
    
    print("\n✅ Embedding complete!")

if __name__ == "__main__":
    main()