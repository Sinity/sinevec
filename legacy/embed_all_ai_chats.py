#!/usr/bin/env python3
"""
Embed all AI conversations (Claude, ChatGPT, Cody) into unified collection.
"""

import os
import json
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

def load_claude_conversations(base_path: Path) -> List[Dict]:
    """Load Claude conversations from extracted files."""
    conversations = []
    
    # Claude stores conversations in individual folders
    for folder in base_path.iterdir():
        if folder.is_dir() and len(folder.name) == 36:  # UUID format
            chat_file = folder / "chat.json"
            if chat_file.exists():
                try:
                    with open(chat_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Extract messages
                    messages = []
                    for msg in data.get('messages', []):
                        role = msg.get('sender', 'unknown')
                        content = msg.get('content', '')
                        if isinstance(content, list) and content:
                            # Handle structured content
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get('type') == 'text':
                                    text_parts.append(part.get('text', ''))
                            content = '\n'.join(text_parts)
                        messages.append({'role': role, 'content': content})
                    
                    if messages:
                        conversations.append({
                            'id': folder.name,
                            'title': data.get('name', 'Untitled'),
                            'messages': messages,
                            'created': data.get('created_at', ''),
                            'updated': data.get('updated_at', '')
                        })
                except Exception as e:
                    print(f"  ⚠️ Error loading {chat_file}: {e}")
    
    return conversations

def load_chatgpt_conversations(file_path: Path) -> List[Dict]:
    """Load ChatGPT conversations from conversations.json."""
    conversations = []
    
    if not file_path.exists():
        return conversations
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for conv in data:
            messages = []
            
            # Extract messages from mapping
            mapping = conv.get('mapping', {})
            for msg_id, msg_data in mapping.items():
                msg = msg_data.get('message')
                if msg and msg.get('content'):
                    content = msg['content']
                    role = msg.get('author', {}).get('role', 'unknown')
                    
                    # Extract text from parts
                    if content.get('content_type') == 'text':
                        parts = content.get('parts', [])
                        if parts and isinstance(parts[0], str):
                            messages.append({
                                'role': role,
                                'content': parts[0]
                            })
            
            if messages:
                conversations.append({
                    'id': conv.get('id', ''),
                    'title': conv.get('title', 'Untitled'),
                    'messages': messages,
                    'created': conv.get('create_time', ''),
                    'updated': conv.get('update_time', '')
                })
    except Exception as e:
        print(f"  ⚠️ Error loading ChatGPT conversations: {e}")
    
    return conversations

def load_cody_conversations(file_path: Path) -> List[Dict]:
    """Load Cody conversations."""
    conversations = []
    
    if not file_path.exists():
        return conversations
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Cody format is simpler - it's a list of conversations
        for conv in data:
            messages = []
            for msg in conv.get('interactions', []):
                # Handle human messages
                if msg.get('humanMessage'):
                    messages.append({
                        'role': 'user',
                        'content': msg['humanMessage'].get('text', '')
                    })
                # Handle assistant messages
                if msg.get('assistantMessage'):
                    messages.append({
                        'role': 'assistant',
                        'content': msg['assistantMessage'].get('text', '')
                    })
            
            if messages:
                conversations.append({
                    'id': conv.get('id', ''),
                    'title': messages[0]['content'][:100] if messages else 'Untitled',
                    'messages': messages,
                    'created': conv.get('timestamp', ''),
                    'updated': conv.get('timestamp', '')
                })
    except Exception as e:
        print(f"  ⚠️ Error loading Cody conversations: {e}")
    
    return conversations

def embed_conversations(conversations: List[Dict], platform: str):
    """Embed conversations into unified collection."""
    
    if not conversations:
        print(f"  No {platform} conversations to embed")
        return 0, 0
    
    print(f"\n📝 Embedding {len(conversations)} {platform} conversations...")
    
    # Get or create collection
    try:
        collection = client.get_collection("unified")
    except:
        collection = client.create_collection("unified")
    
    total_tokens = 0
    total_embedded = 0
    
    for i, conv in enumerate(conversations):
        print(f"\r  [{i+1}/{len(conversations)}] {conv['title'][:50]}...", end="", flush=True)
        
        # Build conversation text
        conv_text = f"Title: {conv['title']}\n"
        if conv.get('created'):
            conv_text += f"Created: {conv['created']}\n"
        conv_text += "\n"
        
        for msg in conv['messages']:
            role = msg['role'].upper()
            content = msg['content']
            conv_text += f"{role}:\n{content}\n\n"
        
        # Check token count and truncate if needed
        tokens = count_tokens(conv_text)
        if tokens > 30000:
            # Truncate to fit
            conv_text = conv_text[:100000]
        
        try:
            # Create unique ID
            conv_id = f"conversation#{platform}#{conv['id']}"
            
            # Check if already exists
            existing = collection.get(ids=[conv_id])
            if existing['ids']:
                continue  # Skip if already embedded
            
            # Embed
            embeds_obj = vo.embed(
                [conv_text],
                model="voyage-3",
                input_type="document"
            )
            
            # Prepare metadata
            metadata = {
                'category': 'conversations',
                'subcategory': platform,
                'title': conv['title'][:500],
                'source': f"chatlog/{platform}",
                'file_type': 'ai_conversation',
                'created': str(conv.get('created', '')),
                'updated': str(conv.get('updated', ''))
            }
            
            # Add to collection
            collection.add(
                ids=[conv_id],
                embeddings=embeds_obj.embeddings,
                documents=[conv_text[:65536]],
                metadatas=[metadata]
            )
            
            total_embedded += 1
            total_tokens += embeds_obj.total_tokens
            
        except Exception as e:
            print(f"\n  ⚠️ Error embedding conversation: {str(e)[:100]}")
    
    print(f"\n  ✅ Embedded {total_embedded} {platform} conversations")
    return total_embedded, total_tokens

def main():
    """Main embedding pipeline for AI conversations."""
    print("🤖 AI Conversation Embedding Pipeline")
    print("=" * 60)
    
    base_path = Path("chatlog")
    
    # Load all conversations
    print("\n📚 Loading conversations...")
    
    # Claude
    print("  Loading Claude conversations...")
    claude_convs = load_claude_conversations(base_path)
    print(f"    Found {len(claude_convs)} conversations")
    
    # ChatGPT
    print("  Loading ChatGPT conversations...")
    chatgpt_convs = load_chatgpt_conversations(base_path / "conversations.json")
    print(f"    Found {len(chatgpt_convs)} conversations")
    
    # Cody
    print("  Loading Cody conversations...")
    cody_convs = load_cody_conversations(base_path / "cody-chat-history-2025-01-19T19-03-05.json")
    print(f"    Found {len(cody_convs)} conversations")
    
    # Embed all conversations
    total_embedded = 0
    total_tokens = 0
    
    embedded, tokens = embed_conversations(claude_convs, "claude")
    total_embedded += embedded
    total_tokens += tokens
    
    embedded, tokens = embed_conversations(chatgpt_convs, "chatgpt")
    total_embedded += embedded
    total_tokens += tokens
    
    embedded, tokens = embed_conversations(cody_convs, "cody")
    total_embedded += embedded
    total_tokens += tokens
    
    print("\n" + "=" * 60)
    print("✅ AI Conversation Embedding Complete!")
    print(f"📊 Stats:")
    print(f"  Total conversations embedded: {total_embedded}")
    print(f"  Total tokens used: {total_tokens:,}")
    if total_embedded > 0:
        print(f"  Avg tokens/conversation: {total_tokens/total_embedded:.0f}")

if __name__ == "__main__":
    main()