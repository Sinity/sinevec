#!/usr/bin/env python3
"""
Version 2 of embedding pipeline with proper resumability and state management.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
import voyageai
import chromadb
from tqdm import tqdm
import git
from dotenv import load_dotenv
import pickle
import signal

from smart_utils import count_tokens, get_file_type
from hierarchical_chunker import hierarchical_chunk_document
from utils import extract_timestamp, should_skip_file

# Load environment variables
load_dotenv()

# Initialize clients
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
client = chromadb.PersistentClient(path="./chroma_db_v2")

class EmbeddingState:
    """Manages embedding state for resumability."""
    
    def __init__(self, state_file: str = "embedding_state.json"):
        self.state_file = Path(state_file)
        self.state = self.load_state()
        self.token_usage = self.state.get('token_usage', {})
        self.processed_files = set(self.state.get('processed_files', []))
        self.failed_files = self.state.get('failed_files', {})
        self.current_file = None
        self.start_time = datetime.now().isoformat()
        
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
    
    def load_state(self) -> Dict:
        """Load state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'version': 2,
            'created_at': datetime.now().isoformat(),
            'token_usage': {},
            'processed_files': [],
            'failed_files': {},
            'collections': {}
        }
    
    def save_state(self):
        """Save current state to disk."""
        self.state['token_usage'] = self.token_usage
        self.state['processed_files'] = list(self.processed_files)
        self.state['failed_files'] = self.failed_files
        self.state['last_updated'] = datetime.now().isoformat()
        
        # Write to temp file first then rename (atomic operation)
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        temp_file.rename(self.state_file)
    
    def handle_interrupt(self, signum, frame):
        """Handle graceful shutdown."""
        print(f"\n\n🛑 Interrupted! Saving state...")
        if self.current_file:
            print(f"   Was processing: {self.current_file}")
            # Don't mark as processed since we were interrupted
            if self.current_file in self.processed_files:
                self.processed_files.remove(self.current_file)
        self.save_state()
        print(f"✅ State saved. Resume with: python embed_v2.py --resume")
        sys.exit(0)
    
    def mark_processed(self, file_path: str, tokens_used: int = 0):
        """Mark a file as processed."""
        self.processed_files.add(file_path)
        if tokens_used:
            collection = self.token_usage.get('total', 0)
            self.token_usage['total'] = collection + tokens_used
            self.token_usage[file_path] = tokens_used
        self.save_state()
    
    def mark_failed(self, file_path: str, error: str):
        """Mark a file as failed."""
        self.failed_files[file_path] = {
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
        self.save_state()
    
    def should_process(self, file_path: str) -> bool:
        """Check if file should be processed."""
        return file_path not in self.processed_files
    
    def get_resume_stats(self) -> str:
        """Get statistics for resume."""
        total_files = len(self.processed_files)
        failed = len(self.failed_files)
        tokens = self.token_usage.get('total', 0)
        
        return f"""
📊 Resume Statistics:
   Processed: {total_files} files
   Failed: {failed} files
   Tokens used: {tokens:,}
   Last run: {self.state.get('last_updated', 'Never')}
        """

class RateLimiter:
    """Handle rate limiting with exponential backoff."""
    
    def __init__(self, rpm: int = 1000):
        self.rpm = rpm
        self.min_interval = 60.0 / rpm if rpm < 1000 else 0
        self.last_request = 0
        self.consecutive_errors = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        if self.min_interval > 0:
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request = time.time()
    
    def handle_error(self, error: Exception) -> bool:
        """Handle rate limit error with backoff. Returns True if should retry."""
        if "rate limit" in str(error).lower():
            self.consecutive_errors += 1
            wait_time = min(300, 20 * (2 ** self.consecutive_errors))  # Max 5 min
            print(f"   Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
            return True
        return False
    
    def reset_errors(self):
        """Reset error counter on success."""
        self.consecutive_errors = 0

def get_or_create_collection(name: str, description: str = "") -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    try:
        return client.get_collection(name)
    except:
        return client.create_collection(
            name=name,
            metadata={"description": description, "created_at": datetime.now().isoformat()}
        )

def compute_file_hash(file_path: Path) -> str:
    """Compute hash of file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def embed_file(
    file_path: Path,
    collection: chromadb.Collection,
    state: EmbeddingState,
    rate_limiter: RateLimiter,
    force: bool = False
) -> int:
    """Embed a single file with proper error handling and resumability."""
    
    file_str = str(file_path)
    
    # Check if already processed (unless forcing)
    if not force and not state.should_process(file_str):
        return 0
    
    # Mark as current file being processed
    state.current_file = file_str
    
    try:
        # Read file
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        if not content.strip() or len(content) < 100:
            state.mark_processed(file_str, 0)
            return 0
        
        # Extract metadata
        file_hash = compute_file_hash(file_path)
        timestamp = extract_timestamp(content, file_str)
        file_type = get_file_type(file_str)
        
        # Check if file changed since last embedding
        existing = collection.get(
            where={"source": file_str}
        )
        if existing and existing['metadatas']:
            old_hash = existing['metadatas'][0].get('file_hash')
            if old_hash == file_hash and not force:
                print(f"   Skipping unchanged: {file_path.name}")
                state.mark_processed(file_str, 0)
                return 0
        
        # Use hierarchical chunking
        chunk_groups = hierarchical_chunk_document(content, file_str)
        if not chunk_groups:
            state.mark_processed(file_str, 0)
            return 0
        
        total_tokens = 0
        
        # Process each group
        for group_idx, chunk_group in enumerate(chunk_groups):
            # Rate limiting
            rate_limiter.wait_if_needed()
            
            # Retry loop for rate limit errors
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Embed the group
                    embeds_obj = vo.contextualized_embed(
                        inputs=[chunk_group],
                        model="voyage-context-3",
                        input_type="document"
                    )
                    
                    # Store each chunk
                    for chunk_idx, (chunk, embedding) in enumerate(zip(chunk_group, embeds_obj.results[0].embeddings)):
                        chunk_id = f"{file_str}#g{group_idx}#c{chunk_idx}"
                        
                        # Delete old version if exists
                        try:
                            collection.delete(ids=[chunk_id])
                        except:
                            pass
                        
                        # Add new version
                        collection.add(
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{
                                "source": file_str,
                                "timestamp": timestamp,
                                "file_type": file_type,
                                "file_hash": file_hash,
                                "group_index": group_idx,
                                "chunk_index": chunk_idx,
                                "total_groups": len(chunk_groups),
                                "embedded_at": datetime.now().isoformat()
                            }],
                            ids=[chunk_id]
                        )
                    
                    total_tokens += embeds_obj.total_tokens
                    rate_limiter.reset_errors()
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if rate_limiter.handle_error(e) and retry < max_retries - 1:
                        continue  # Retry
                    else:
                        print(f"   Error embedding group {group_idx}: {e}")
                        if group_idx == 0:
                            # If first group fails, mark file as failed
                            raise
                        # Otherwise continue with next group
                        break
        
        # Mark as processed
        state.mark_processed(file_str, total_tokens)
        state.current_file = None
        return total_tokens
        
    except Exception as e:
        print(f"   Failed to process {file_path}: {e}")
        state.mark_failed(file_str, str(e))
        state.current_file = None
        return 0

def scan_directory(directory: Path, state: EmbeddingState) -> List[Path]:
    """Scan directory for files to process."""
    files_to_process = []
    
    for file_path in directory.rglob("*"):
        if file_path.is_file() and not should_skip_file(file_path):
            if state.should_process(str(file_path)):
                files_to_process.append(file_path)
    
    return files_to_process

def main():
    """Main embedding pipeline with resume support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed files with Voyage AI")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state")
    parser.add_argument("--reset", action="store_true", help="Reset and start fresh")
    parser.add_argument("--force", action="store_true", help="Force re-embedding of all files")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--collection", default="knowledgebase_v2", help="Collection name")
    parser.add_argument("--rpm", type=int, default=1000, help="Requests per minute (default: 1000 = no limit)")
    args = parser.parse_args()
    
    # Initialize state
    state = EmbeddingState()
    
    if args.stats:
        print(state.get_resume_stats())
        return
    
    if args.reset:
        print("🗑️ Resetting state...")
        state = EmbeddingState()
        state.state = {
            'version': 2,
            'created_at': datetime.now().isoformat(),
            'token_usage': {},
            'processed_files': [],
            'failed_files': {},
            'collections': {}
        }
        state.save_state()
        
        # Also reset ChromaDB
        try:
            client.delete_collection(args.collection)
            print(f"   Deleted collection: {args.collection}")
        except:
            pass
    
    elif args.resume:
        print(state.get_resume_stats())
    
    print(f"\n🚀 Voyage Embedding Pipeline v2")
    print(f"📦 Collection: {args.collection}")
    print(f"⚡ Rate limit: {args.rpm} RPM")
    
    # Initialize components
    rate_limiter = RateLimiter(rpm=args.rpm)
    collection = get_or_create_collection(
        args.collection,
        "Knowledge base with hierarchical chunking"
    )
    
    # Define sources to embed
    sources = [
        (Path("/realm/knowledgebase"), "knowledgebase"),
        (Path("/realm/project/sinex"), "code"),
        # Add more sources as needed
    ]
    
    # Scan all sources
    all_files = []
    for source_path, source_type in sources:
        if source_path.exists():
            files = scan_directory(source_path, state)
            all_files.extend(files)
            print(f"📁 {source_type}: {len(files)} files to process")
    
    if not all_files:
        print("✅ All files already processed!")
        return
    
    print(f"\n📊 Total files to process: {len(all_files)}")
    
    # Process files
    total_tokens = state.token_usage.get('total', 0)
    
    with tqdm(all_files, desc="Embedding files") as pbar:
        for file_path in pbar:
            pbar.set_description(f"Embedding {file_path.name[:30]}")
            
            tokens = embed_file(
                file_path,
                collection,
                state,
                rate_limiter,
                force=args.force
            )
            
            if tokens > 0:
                total_tokens += tokens
                pbar.set_postfix(tokens=f"{total_tokens:,}")
    
    print(f"\n✅ Embedding complete!")
    print(f"📊 Total tokens used: {total_tokens:,}")
    print(f"📁 Files processed: {len(state.processed_files)}")
    if state.failed_files:
        print(f"⚠️ Failed files: {len(state.failed_files)}")
        for file, info in list(state.failed_files.items())[:5]:
            print(f"   - {file}: {info['error'][:50]}")

if __name__ == "__main__":
    main()