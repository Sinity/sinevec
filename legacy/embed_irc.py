#!/usr/bin/env python3
"""
IRC log embedding pipeline with conversation-aware chunking.
Uses rich metadata for subdivision within a single collection.
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
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

def parse_irc_timestamp(line: str) -> Optional[datetime]:
    """Extract timestamp from IRC log line."""
    # Format: 2025-01-05 00:02:00	username	message
    match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
    return None

def extract_username(line: str) -> Optional[str]:
    """Extract username from IRC log line."""
    # Format: timestamp\tusername\tmessage or timestamp\t<--\taction
    parts = line.split('\t')
    if len(parts) >= 2:
        username = parts[1].strip()
        # Clean up special prefixes
        username = username.lstrip('+@')
        # Handle special IRC actions
        if username in ['<--', '-->', '---', '*']:
            return None
        # Clean up backslashes and special chars for ChromaDB
        username = username.replace('\\', '_').replace('/', '_')
        # Handle feepbot posts
        if username == 'feepbot' and len(parts) > 2:
            # Extract actual poster from feepbot message
            match = re.match(r'<(\w+)>', parts[2])
            if match:
                return f"feepbot:{match.group(1)}"
        return username
    return None

def extract_participants(chunk_lines: List[str]) -> List[str]:
    """Extract unique participants from chunk."""
    participants = set()
    for line in chunk_lines:
        username = extract_username(line)
        if username and username not in ['<--', '-->', '---', '*']:
            participants.add(username)
    return sorted(list(participants))

def detect_code_discussion(content: str) -> bool:
    """Detect if chunk contains code discussion."""
    code_indicators = [
        r'\bdef\s+\w+',
        r'\bclass\s+\w+',
        r'\bfunction\s+\w+',
        r'\bfn\s+\w+',
        r'\bimpl\s+',
        r'```',
        r'\bgit\s+(pull|push|commit|merge)',
        r'\b(python|rust|javascript|typescript|go|c\+\+)\b',
        r'\b(bug|debug|error|exception|stacktrace)\b',
        r'\b(api|endpoint|database|query|sql)\b',
    ]
    
    content_lower = content.lower()
    for pattern in code_indicators:
        if re.search(pattern, content_lower):
            return True
    return False

def detect_urls(content: str) -> List[str]:
    """Extract URLs from content."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, content)
    return urls[:10]  # Limit to 10 URLs for metadata

def chunk_irc_by_conversations(
    log_content: str,
    file_path: Path,
    gap_minutes: int = 30,
    context_lines: int = 20,  # Increased from 5-10 to 20 for more overlap
    max_chunk_tokens: int = 25000  # Leave room for context
) -> List[Dict]:
    """
    Chunk IRC logs by conversation with generous overlap.
    
    Args:
        log_content: Raw IRC log content
        file_path: Path to the log file
        gap_minutes: Minutes of silence to consider conversation break
        context_lines: Number of lines to include as context before/after
        max_chunk_tokens: Maximum tokens per chunk
    """
    
    # Extract metadata from filename
    filename = file_path.name
    channel_match = re.match(r'#?([^.]+)', filename)
    channel = f"#{channel_match.group(1)}" if channel_match else "unknown"
    
    # Parse date from filename or path
    date_match = re.search(r'(\d{4}).*?(\d{2})\.(\d{2})\.log', filename)
    if date_match:
        year = date_match.group(1)
        month = date_match.group(2)
        day = date_match.group(3)
        log_date = f"{year}-{month}-{day}"
    else:
        # Try to get from directory structure
        if '2024' in str(file_path):
            year = '2024'
        elif '2025' in str(file_path):
            year = '2025'
        else:
            year = str(datetime.now().year)
        log_date = f"{year}-01-01"  # Default
    
    lines = log_content.split('\n')
    chunks = []
    current_chunk = []
    current_tokens = 0
    last_timestamp = None
    chunk_start_time = None
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        timestamp = parse_irc_timestamp(line)
        line_tokens = count_tokens(line)
        
        # Check for conversation break
        should_break = False
        if timestamp and last_timestamp:
            time_gap = (timestamp - last_timestamp).total_seconds() / 60
            if time_gap > gap_minutes:
                should_break = True
        
        # Also break if chunk is getting too large
        if current_tokens + line_tokens > max_chunk_tokens:
            should_break = True
        
        if should_break and len(current_chunk) > 10:  # Min conversation size
            # Get context from next conversation (look ahead)
            next_context = []
            for j in range(i, min(i + context_lines, len(lines))):
                if j < len(lines) and lines[j].strip():
                    next_context.append(lines[j])
            
            # Create chunk with metadata
            chunk_content = '\n'.join(current_chunk)
            if next_context:
                chunk_content += '\n\n[... context from next conversation ...]\n'
                chunk_content += '\n'.join(next_context)
            
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'source': str(file_path),
                    'category': 'irc',
                    'subcategory': channel.replace('#', ''),
                    'channel': channel,
                    'date': log_date,
                    'conversation_start': chunk_start_time.isoformat() if chunk_start_time else log_date,
                    'conversation_end': last_timestamp.isoformat() if last_timestamp else log_date,
                    'participants': ', '.join(extract_participants(current_chunk)[:10]),  # Join as string, limit to 10
                    'line_count': len(current_chunk),
                    'has_code': detect_code_discussion(chunk_content),
                    'has_urls': len(detect_urls(chunk_content)) > 0,
                    'urls': ', '.join(detect_urls(chunk_content)[:3]),  # Join first 3 URLs as string
                    'file_type': 'irc_log'
                }
            })
            
            # Start new chunk with context from previous
            prev_context = []
            for j in range(max(0, i - context_lines), i):
                if lines[j].strip():
                    prev_context.append(lines[j])
            
            current_chunk = []
            if prev_context:
                current_chunk.append('[... context from previous conversation ...]')
                current_chunk.extend(prev_context)
                current_chunk.append('[... current conversation ...]')
            
            current_chunk.append(line)
            current_tokens = count_tokens('\n'.join(current_chunk))
            chunk_start_time = timestamp
        else:
            current_chunk.append(line)
            current_tokens += line_tokens
            if not chunk_start_time and timestamp:
                chunk_start_time = timestamp
        
        last_timestamp = timestamp
    
    # Don't forget the last chunk
    if len(current_chunk) > 10:
        chunk_content = '\n'.join(current_chunk)
        chunks.append({
            'content': chunk_content,
            'metadata': {
                'source': str(file_path),
                'category': 'irc',
                'subcategory': channel.replace('#', ''),
                'channel': channel,
                'date': log_date,
                'conversation_start': chunk_start_time.isoformat() if chunk_start_time else log_date,
                'conversation_end': last_timestamp.isoformat() if last_timestamp else log_date,
                'participants': extract_participants(current_chunk),
                'line_count': len(current_chunk),
                'has_code': detect_code_discussion(chunk_content),
                'has_urls': len(detect_urls(chunk_content)) > 0,
                'urls': detect_urls(chunk_content)[:5],
                'file_type': 'irc_log'
            }
        })
    
    return chunks

def embed_irc_file(file_path: Path, collection_name: str = "unified") -> int:
    """Embed a single IRC log file."""
    
    try:
        # Read file
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        if not content.strip() or len(content) < 100:
            return 0
        
        # Get or create collection
        try:
            collection = client.get_collection(collection_name)
        except:
            collection = client.create_collection(
                name=collection_name,
                metadata={"created_at": datetime.now().isoformat()}
            )
        
        # Chunk by conversations
        chunks = chunk_irc_by_conversations(content, file_path)
        
        if not chunks:
            return 0
        
        total_tokens = 0
        print(f"\n📄 Processing {file_path.name}: {len(chunks)} conversation chunks")
        
        # Process each chunk
        for i, chunk_data in enumerate(tqdm(chunks, desc=f"Embedding {file_path.name}")):
            try:
                # Embed the chunk
                embeds_obj = vo.embed(
                    [chunk_data['content']],
                    model="voyage-3",
                    input_type="document"
                )
                
                # Create unique ID
                chunk_id = f"{file_path}#chunk{i}"
                
                # Delete old version if exists
                try:
                    collection.delete(ids=[chunk_id])
                except:
                    pass
                
                # Add to collection with rich metadata
                collection.add(
                    embeddings=embeds_obj.embeddings,
                    documents=[chunk_data['content'][:65536]],  # ChromaDB limit
                    metadatas=[chunk_data['metadata']],
                    ids=[chunk_id]
                )
                
                total_tokens += embeds_obj.total_tokens
                
            except Exception as e:
                print(f"  ⚠️ Error embedding chunk {i}: {str(e)[:100]}")
                if "rate limit" in str(e).lower():
                    import time
                    time.sleep(20)
        
        return total_tokens
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return 0

def scan_irc_logs(base_path: Path = Path("/home/sinity/.local/share/weechat/logs")) -> List[Path]:
    """Scan for IRC log files to process."""
    log_files = []
    
    # Process 2024 and 2025 logs
    for year in ['2024', '2025']:
        year_path = base_path / year
        if year_path.exists():
            # Get all .log files that look like IRC channel logs
            for log_file in year_path.glob("*.log"):
                # Skip non-channel logs
                if any(skip in log_file.name for skip in ['all.log', 'core.', 'found.', 'irc_logs.']):
                    continue
                log_files.append(log_file)
    
    # Sort by date (newer first for testing)
    log_files.sort(reverse=True)
    
    return log_files

def main():
    """Main IRC embedding pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed IRC logs with conversation chunking")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--channel", help="Only process specific channel")
    parser.add_argument("--year", help="Only process specific year (2024 or 2025)")
    args = parser.parse_args()
    
    print("🚀 IRC Log Embedding Pipeline")
    print("💬 Using conversation-based chunking with 20-line overlap")
    
    # Scan for log files
    log_files = scan_irc_logs()
    
    # Apply filters
    if args.channel:
        log_files = [f for f in log_files if args.channel in f.name]
    if args.year:
        log_files = [f for f in log_files if args.year in str(f)]
    
    if args.limit:
        log_files = log_files[:args.limit]
    
    print(f"📁 Found {len(log_files)} IRC log files to process")
    
    if not log_files:
        print("No files to process!")
        return
    
    # Show sample files
    print("\nSample files:")
    for f in log_files[:5]:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    # Process files
    total_tokens = 0
    processed = 0
    
    for log_file in tqdm(log_files, desc="Processing IRC logs"):
        tokens = embed_irc_file(log_file)
        if tokens > 0:
            total_tokens += tokens
            processed += 1
            
            # Save progress periodically
            if processed % 10 == 0:
                print(f"\n📊 Progress: {processed} files, {total_tokens:,} tokens used")
    
    print(f"\n✅ IRC Embedding complete!")
    print(f"📊 Processed {processed} files")
    print(f"🎯 Total tokens used: {total_tokens:,}")

if __name__ == "__main__":
    main()