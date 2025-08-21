#!/usr/bin/env python3
"""
Reddit comment embedding pipeline with hierarchical chunking.
Handles blockquotes, point-by-point responses, and temporal context.
"""

import os
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

class RedditComment:
    """Represents a Reddit comment with its structure."""
    
    def __init__(self, raw_text: str, index: int):
        self.raw_text = raw_text
        self.index = index
        self.timestamp = None
        self.subreddit = None
        self.thread_id = None
        self.comment_id = None
        self.score = None
        self.body = ""
        self.blockquote_pairs = []  # List of (quote, response) tuples
        self.pure_response = ""  # Response without blockquotes
        
        self._parse()
    
    def _parse(self):
        """Parse the comment structure."""
        lines = self.raw_text.strip().split('\n')
        
        # First line has format: 🌀 #N YYYY-MM-DD HH:MM:SS UTC https://reddit.com/r/subreddit/...
        if lines and lines[0].startswith('🌀'):
            header = lines[0]
            
            # Parse timestamp
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', header)
            if timestamp_match:
                try:
                    self.timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            # Parse subreddit from URL
            subreddit_match = re.search(r'/r/(\w+)/', header)
            if subreddit_match:
                self.subreddit = f"r/{subreddit_match.group(1)}"
            
            # Parse comment ID
            id_match = re.search(r'#(\d+)', header)
            if id_match:
                self.comment_id = id_match.group(1)
            
            # Body starts from second line
            self.body = '\n'.join(lines[1:]).strip()
        else:
            # Fallback parsing
            self.body = self.raw_text.strip()
        
        # Parse blockquote structure
        self._parse_blockquotes()
    
    def _parse_blockquotes(self):
        """Parse blockquote-response pairs for point-by-point responses."""
        lines = self.body.split('\n')
        current_quote = []
        current_response = []
        in_quote = False
        
        for line in lines:
            if line.startswith('>'):
                if current_response:
                    # Save previous pair
                    if current_quote:
                        self.blockquote_pairs.append((
                            '\n'.join(current_quote),
                            '\n'.join(current_response)
                        ))
                    current_quote = [line]
                    current_response = []
                else:
                    current_quote.append(line)
                in_quote = True
            else:
                if in_quote and line.strip():
                    current_response.append(line)
                    in_quote = False
                elif current_response:
                    current_response.append(line)
                else:
                    # Pure response without quotes
                    self.pure_response += line + '\n'
        
        # Save last pair if exists
        if current_quote and current_response:
            self.blockquote_pairs.append((
                '\n'.join(current_quote),
                '\n'.join(current_response)
            ))
        elif current_quote and not current_response:
            # Quote without response - add to pure response
            self.pure_response += '\n'.join(current_quote) + '\n'
        
        self.pure_response = self.pure_response.strip()

def parse_reddit_file(file_path: Path) -> List[RedditComment]:
    """Parse the formatted.txt file into structured comments."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by the spiral separator pattern 🌀🌀🌀
    comments_raw = re.split(r'🌀🌀🌀\n?', content)
    
    comments = []
    for i, raw_comment in enumerate(comments_raw):
        if raw_comment.strip():
            comment = RedditComment(raw_comment, i)
            comments.append(comment)
    
    # Sort by timestamp if available
    comments.sort(key=lambda c: c.timestamp if c.timestamp else datetime.min)
    
    return comments

def create_temporal_groups(comments: List[RedditComment], 
                          time_window_hours: int = 24) -> List[List[RedditComment]]:
    """Group comments by temporal proximity."""
    groups = []
    current_group = []
    last_timestamp = None
    
    for comment in comments:
        if not comment.timestamp:
            # No timestamp, add to current group
            current_group.append(comment)
            continue
        
        if last_timestamp and comment.timestamp - last_timestamp > timedelta(hours=time_window_hours):
            # Time gap too large, start new group
            if current_group:
                groups.append(current_group)
            current_group = [comment]
        else:
            current_group.append(comment)
        
        last_timestamp = comment.timestamp
    
    if current_group:
        groups.append(current_group)
    
    return groups

def create_hierarchical_chunks(comment: RedditComment) -> List[Dict]:
    """
    Create hierarchical chunks for a comment:
    1. Full comment
    2. Blockquote-response pairs (for point-by-point discussions)
    3. Pure response (without blockquotes)
    """
    chunks = []
    
    # 1. Full comment chunk
    if comment.body:
        chunks.append({
            'content': comment.body,
            'metadata': {
                'type': 'full_comment',
                'level': 0,  # Top level
                'has_blockquotes': len(comment.blockquote_pairs) > 0,
                'comment_index': comment.index
            }
        })
    
    # 2. Individual blockquote-response pairs
    for i, (quote, response) in enumerate(comment.blockquote_pairs):
        # Combined pair
        pair_content = f"[Quote]:\n{quote}\n\n[Response]:\n{response}"
        chunks.append({
            'content': pair_content,
            'metadata': {
                'type': 'quote_response_pair',
                'level': 1,  # Sub-level
                'pair_index': i,
                'comment_index': comment.index
            }
        })
        
        # Just the response (for searching responses without context)
        if len(response) > 100:  # Only if substantial
            chunks.append({
                'content': response,
                'metadata': {
                    'type': 'response_only',
                    'level': 2,  # Deeper level
                    'pair_index': i,
                    'comment_index': comment.index
                }
            })
    
    # 3. Pure response (parts without blockquotes)
    if comment.pure_response and len(comment.pure_response) > 100:
        chunks.append({
            'content': comment.pure_response,
            'metadata': {
                'type': 'pure_response',
                'level': 1,
                'comment_index': comment.index
            }
        })
    
    return chunks

def create_contextualized_chunks(comment_group: List[RedditComment], 
                                max_tokens: int = 25000) -> List[Dict]:
    """
    Create chunks with temporal context from nearby comments.
    """
    chunks = []
    
    for i, comment in enumerate(comment_group):
        # Get hierarchical chunks for this comment
        comment_chunks = create_hierarchical_chunks(comment)
        
        # Add temporal context to each chunk
        for chunk in comment_chunks:
            # Add context from nearby comments (before and after)
            context_before = []
            context_after = []
            
            # Previous comments context (up to 2 comments)
            for j in range(max(0, i-2), i):
                prev_comment = comment_group[j]
                # Add brief summary of previous comment
                preview = prev_comment.body[:200] if prev_comment.body else ""
                if preview:
                    context_before.append(f"[Earlier comment {j+1}]: {preview}...")
            
            # Next comments context (up to 2 comments)
            for j in range(i+1, min(i+3, len(comment_group))):
                next_comment = comment_group[j]
                preview = next_comment.body[:200] if next_comment.body else ""
                if preview:
                    context_after.append(f"[Later comment {j+1}]: {preview}...")
            
            # Build contextualized content
            contextualized_content = ""
            if context_before:
                contextualized_content += "=== Temporal Context (Before) ===\n"
                contextualized_content += '\n'.join(context_before) + '\n\n'
            
            contextualized_content += "=== Current Content ===\n"
            contextualized_content += chunk['content']
            
            if context_after:
                contextualized_content += "\n\n=== Temporal Context (After) ===\n"
                contextualized_content += '\n'.join(context_after)
            
            # Check token limit
            if count_tokens(contextualized_content) <= max_tokens:
                chunk['content_with_context'] = contextualized_content
            else:
                # Too large, use just the chunk content
                chunk['content_with_context'] = chunk['content']
            
            # Enhance metadata
            chunk['metadata'].update({
                'timestamp': comment.timestamp.isoformat() if comment.timestamp else None,
                'subreddit': comment.subreddit,
                'score': comment.score,
                'group_size': len(comment_group),
                'position_in_group': i
            })
            
            chunks.append(chunk)
    
    return chunks

def embed_reddit_comments(file_path: Path, collection_name: str = "unified") -> Tuple[int, int]:
    """Main function to embed Reddit comments."""
    
    print(f"📖 Parsing Reddit comments from {file_path.name}...")
    comments = parse_reddit_file(file_path)
    print(f"  Found {len(comments)} comments")
    
    # Create temporal groups
    print("⏰ Creating temporal groups...")
    temporal_groups = create_temporal_groups(comments, time_window_hours=24)
    print(f"  Created {len(temporal_groups)} temporal groups")
    
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
    
    # Process each temporal group
    for group_idx, comment_group in enumerate(tqdm(temporal_groups, desc="Processing temporal groups")):
        # Create contextualized chunks for this group
        chunks = create_contextualized_chunks(comment_group)
        
        # Embed each chunk
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc=f"Group {group_idx}", leave=False)):
            try:
                # Use contextualized content for embedding
                content_to_embed = chunk.get('content_with_context', chunk['content'])
                
                # Embed
                embeds_obj = vo.embed(
                    [content_to_embed],
                    model="voyage-3",
                    input_type="document"
                )
                
                # Create unique ID
                chunk_id = f"reddit#g{group_idx}#c{chunk['metadata']['comment_index']}#chunk{chunk_idx}"
                
                # Delete old version if exists
                try:
                    collection.delete(ids=[chunk_id])
                except:
                    pass
                
                # Prepare metadata
                metadata = {
                    'source': str(file_path),
                    'category': 'reddit',
                    'subcategory': chunk['metadata'].get('subreddit', 'unknown'),
                    'chunk_type': chunk['metadata']['type'],
                    'hierarchy_level': chunk['metadata']['level'],
                    'timestamp': chunk['metadata'].get('timestamp', ''),
                    'score': chunk['metadata'].get('score', 0),
                    'has_blockquotes': chunk['metadata'].get('has_blockquotes', False),
                    'temporal_group': group_idx,
                    'comment_index': chunk['metadata']['comment_index'],
                    'file_type': 'reddit_comment'
                }
                
                # Add to collection
                collection.add(
                    embeddings=embeds_obj.embeddings,
                    documents=[chunk['content'][:65536]],  # Store original content
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
                
                total_tokens += embeds_obj.total_tokens
                total_chunks += 1
                
            except Exception as e:
                print(f"\n  ⚠️ Error embedding chunk: {str(e)[:100]}")
    
    return total_tokens, total_chunks

def main():
    """Main Reddit embedding pipeline."""
    print("🎯 Reddit Comment Embedding Pipeline")
    print("=" * 60)
    print("Features:")
    print("  • Hierarchical chunking (full → quote-pairs → responses)")
    print("  • Temporal contextualization (nearby comments)")
    print("  • Blockquote-aware parsing for point-by-point discussions")
    print("=" * 60)
    
    # Path to Reddit comments
    reddit_file = Path("/realm/data/reddit_comments/formatted.txt")
    
    if not reddit_file.exists():
        print(f"❌ File not found: {reddit_file}")
        return
    
    # Get file size
    file_size_mb = reddit_file.stat().st_size / (1024 * 1024)
    print(f"\n📄 File size: {file_size_mb:.1f} MB")
    
    # Embed comments
    tokens, chunks = embed_reddit_comments(reddit_file)
    
    print(f"\n✅ Reddit Embedding Complete!")
    print(f"📊 Stats:")
    print(f"  Total chunks: {chunks}")
    print(f"  Tokens used: {tokens:,}")
    print(f"  Avg tokens/chunk: {tokens/chunks:.1f}" if chunks > 0 else "")

if __name__ == "__main__":
    main()