#!/usr/bin/env python3
"""
AI conversation embedding pipeline for Claude, ChatGPT, and Cody exports.
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import voyageai
import chromadb
from tqdm import tqdm
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

def extract_claude_conversations(archive_path: Path) -> List[Dict]:
    """Extract conversations from Claude export."""
    conversations = []
    
    with zipfile.ZipFile(archive_path, 'r') as z:
        # Extract conversations.json
        with z.open('conversations.json') as f:
            data = json.load(f)
            
            # Claude export is a list of conversations
            if isinstance(data, list):
                for conv in data:
                    messages = []
                    # Extract messages from chat_messages
                    for msg in conv.get('chat_messages', []):
                        messages.append({
                            'role': msg.get('sender', 'unknown'),
                            'content': msg.get('text', '')
                        })
                    
                    if messages:
                        conversations.append({
                            'id': conv.get('uuid', ''),
                            'title': conv.get('name', 'Untitled'),
                            'created_at': conv.get('created_at', ''),
                            'updated_at': conv.get('updated_at', ''),
                            'messages': messages,
                            'source': 'claude'
                        })
    
    return conversations

def extract_chatgpt_conversations(archive_path: Path) -> List[Dict]:
    """Extract conversations from ChatGPT export."""
    conversations = []
    
    with zipfile.ZipFile(archive_path, 'r') as z:
        # ChatGPT exports have conversations.json in root
        with z.open('conversations.json') as f:
            data = json.load(f)
            
            # ChatGPT format is usually a list of conversations
            for conv in data:
                if 'mapping' in conv:
                    # Extract messages from the mapping structure
                    messages = []
                    for msg_id, msg_data in conv['mapping'].items():
                        if 'message' in msg_data and msg_data['message']:
                            msg = msg_data['message']
                            if 'content' in msg and 'parts' in msg['content']:
                                role = msg.get('author', {}).get('role', 'unknown')
                                parts = msg['content']['parts']
                                # Join parts, handling both string and dict parts
                                content_parts = []
                                for part in parts:
                                    if isinstance(part, str):
                                        content_parts.append(part)
                                    elif isinstance(part, dict):
                                        content_parts.append(str(part.get('text', part)))
                                content = ' '.join(content_parts)
                                messages.append({
                                    'role': role,
                                    'content': content,
                                    'timestamp': msg.get('create_time', '')
                                })
                    
                    if messages:
                        conversations.append({
                            'id': conv.get('id', ''),
                            'title': conv.get('title', 'Untitled'),
                            'created_at': conv.get('create_time', ''),
                            'updated_at': conv.get('update_time', ''),
                            'messages': messages,
                            'source': 'chatgpt'
                        })
    
    return conversations

def extract_cody_conversations(json_path: Path) -> List[Dict]:
    """Extract conversations from Cody JSON export."""
    conversations = []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
        # Cody format varies, adapt based on structure
        if isinstance(data, list):
            for conv in data:
                messages = []
                if 'interactions' in conv:
                    for interaction in conv['interactions']:
                        if 'humanMessage' in interaction:
                            messages.append({
                                'role': 'user',
                                'content': interaction['humanMessage'].get('text', '')
                            })
                        if 'assistantMessage' in interaction:
                            messages.append({
                                'role': 'assistant', 
                                'content': interaction['assistantMessage'].get('text', '')
                            })
                
                if messages:
                    conversations.append({
                        'id': conv.get('id', ''),
                        'title': conv.get('title', messages[0]['content'][:50] if messages else 'Untitled'),
                        'created_at': conv.get('timestamp', ''),
                        'updated_at': conv.get('lastUpdated', ''),
                        'messages': messages,
                        'source': 'cody'
                    })
        
    return conversations

def chunk_conversation(conv: Dict, max_tokens: int = 25000) -> List[Dict]:
    """
    Chunk a conversation into embeddable segments.
    Preserves conversation flow with overlap.
    """
    chunks = []
    messages = conv['messages']
    
    if not messages:
        return []
    
    current_chunk = []
    current_tokens = 0
    chunk_start_idx = 0
    
    for i, msg in enumerate(messages):
        # Format message
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if isinstance(content, list):
            # Handle Claude's content format
            content = ' '.join([c.get('text', '') if isinstance(c, dict) else str(c) for c in content])
        
        msg_text = f"{role.upper()}: {content}\n\n"
        msg_tokens = count_tokens(msg_text)
        
        # Check if adding this message would exceed limit
        if current_tokens + msg_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunk_text = '\n'.join(current_chunk)
            
            # Add context from next few messages
            context_msgs = []
            for j in range(i, min(i + 3, len(messages))):
                next_msg = messages[j]
                next_role = next_msg.get('role', 'unknown')
                next_content = next_msg.get('content', '')
                if isinstance(next_content, list):
                    next_content = ' '.join([c.get('text', '') if isinstance(c, dict) else str(c) for c in next_content])
                context_msgs.append(f"{next_role.upper()}: {next_content[:200]}...")
            
            if context_msgs:
                chunk_text += "\n\n[... next messages ...]\n" + '\n'.join(context_msgs[:2])
            
            chunks.append({
                'content': chunk_text,
                'metadata': {
                    'source': str(conv['source']),
                    'category': 'conversations',
                    'subcategory': conv['source'],
                    'conversation_id': str(conv['id']),
                    'conversation_title': str(conv['title']),
                    'chunk_index': len(chunks),
                    'message_range': f"{chunk_start_idx}-{i-1}",
                    'created_at': str(conv.get('created_at', '')),
                    'has_code': detect_code(chunk_text),
                    'file_type': 'ai_conversation'
                }
            })
            
            # Start new chunk with overlap
            overlap_start = max(0, i - 2)
            current_chunk = []
            for j in range(overlap_start, i):
                prev_msg = messages[j]
                prev_role = prev_msg.get('role', 'unknown')
                prev_content = prev_msg.get('content', '')
                if isinstance(prev_content, list):
                    prev_content = ' '.join([c.get('text', '') if isinstance(c, dict) else str(c) for c in prev_content])
                current_chunk.append(f"{prev_role.upper()}: {prev_content[:500]}...")
            
            if current_chunk:
                current_chunk.insert(0, "[... previous context ...]")
            
            current_chunk.append(msg_text)
            current_tokens = count_tokens('\n'.join(current_chunk))
            chunk_start_idx = i
        else:
            current_chunk.append(msg_text)
            current_tokens += msg_tokens
    
    # Don't forget last chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append({
            'content': chunk_text,
            'metadata': {
                'source': str(conv['source']),
                'category': 'conversations',
                'subcategory': conv['source'],
                'conversation_id': str(conv['id']),
                'conversation_title': str(conv['title']),
                'chunk_index': len(chunks),
                'message_range': f"{chunk_start_idx}-{len(messages)-1}",
                'created_at': str(conv.get('created_at', '')),
                'has_code': detect_code(chunk_text),
                'file_type': 'ai_conversation'
            }
        })
    
    return chunks

def detect_code(text: str) -> bool:
    """Detect if text contains code."""
    code_indicators = [
        '```',
        'def ', 'class ', 'function ', 'const ', 'let ', 'var ',
        'import ', 'from ', 'require(',
        'if __name__', 'pub fn', 'fn main',
        '#!/usr/bin',
        'SELECT ', 'CREATE TABLE', 'INSERT INTO'
    ]
    return any(indicator in text for indicator in code_indicators)

def embed_conversations(conversations: List[Dict], collection_name: str = "unified") -> int:
    """Embed AI conversations."""
    
    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"created_at": datetime.now().isoformat()}
        )
    
    total_tokens = 0
    total_chunks = 0
    
    for conv in tqdm(conversations, desc="Processing conversations"):
        chunks = chunk_conversation(conv)
        
        for chunk in tqdm(chunks, desc=f"Embedding {conv['title'][:30]}", leave=False):
            try:
                # Embed the chunk
                embeds_obj = vo.embed(
                    [chunk['content']],
                    model="voyage-3",
                    input_type="document"
                )
                
                # Create unique ID
                chunk_id = f"{conv['source']}#{conv['id']}#chunk{chunk['metadata']['chunk_index']}"
                
                # Delete old version if exists
                try:
                    collection.delete(ids=[chunk_id])
                except:
                    pass
                
                # Add to collection
                collection.add(
                    embeddings=embeds_obj.embeddings,
                    documents=[chunk['content'][:65536]],
                    metadatas=[chunk['metadata']],
                    ids=[chunk_id]
                )
                
                total_tokens += embeds_obj.total_tokens
                total_chunks += 1
                
            except Exception as e:
                print(f"  ⚠️ Error embedding chunk: {str(e)[:100]}")
    
    return total_tokens, total_chunks

def main():
    """Main AI conversation embedding pipeline."""
    print("🤖 AI Conversation Embedding Pipeline")
    print("=" * 60)
    
    chatlog_dir = Path("/realm/data/chatlog")
    
    all_conversations = []
    
    # Process Claude export
    claude_archive = chatlog_dir / "claude-ai-data-2025-01-31-01-35-27.zip"
    if claude_archive.exists():
        print(f"\n📦 Processing Claude conversations...")
        try:
            claude_convs = extract_claude_conversations(claude_archive)
            print(f"  Found {len(claude_convs)} Claude conversations")
            all_conversations.extend(claude_convs)
        except Exception as e:
            print(f"  ❌ Error processing Claude: {e}")
    
    # Process ChatGPT export
    chatgpt_archive = chatlog_dir / "chatgpt-data-2025-01-31-02-51-50.zip"
    if chatgpt_archive.exists():
        print(f"\n📦 Processing ChatGPT conversations...")
        try:
            chatgpt_convs = extract_chatgpt_conversations(chatgpt_archive)
            print(f"  Found {len(chatgpt_convs)} ChatGPT conversations")
            all_conversations.extend(chatgpt_convs)
        except Exception as e:
            print(f"  ❌ Error processing ChatGPT: {e}")
    
    # Process Cody export
    cody_json = chatlog_dir / "cody-chat-history-2025-01-19T19-03-05.json"
    if cody_json.exists():
        print(f"\n📦 Processing Cody conversations...")
        try:
            cody_convs = extract_cody_conversations(cody_json)
            print(f"  Found {len(cody_convs)} Cody conversations")
            all_conversations.extend(cody_convs)
        except Exception as e:
            print(f"  ❌ Error processing Cody: {e}")
    
    if not all_conversations:
        print("❌ No conversations found to embed!")
        return
    
    print(f"\n📊 Total conversations to process: {len(all_conversations)}")
    
    # Sort by date (newest first) - handle mixed types
    all_conversations.sort(key=lambda x: str(x.get('updated_at', '')), reverse=True)
    
    # Embed conversations
    print("\n🚀 Starting embedding process...")
    total_tokens, total_chunks = embed_conversations(all_conversations)
    
    print(f"\n✅ Embedding complete!")
    print(f"📊 Stats:")
    print(f"  Conversations processed: {len(all_conversations)}")
    print(f"  Chunks created: {total_chunks}")
    print(f"  Tokens used: {total_tokens:,}")

if __name__ == "__main__":
    main()