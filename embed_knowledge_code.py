#!/usr/bin/env python3
"""
Embed knowledgebase and code trees with Voyage contextualized embeddings.
Resumable, simple chunking, and unified collection output.
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
import signal
import tiktoken
from voyage_embeddings.embed_utils import (
    get_clients,
    ensure_collection as _ensure_collection,
    count_tokens,
    simple_chunk_document,
    group_chunks_for_voyage,
    extract_timestamp,
    should_skip_file,
)

# Load environment variables
load_dotenv()

# Initialize clients
vo, client = get_clients()

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

class EmbeddingState:
    """Manages embedding state for resumability."""
    
    def __init__(self, state_file: str = "var/state/knowledge_code_state.json"):
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
            'version': 3,
            'created_at': datetime.now().isoformat(),
            'token_usage': {},
            'processed_files': [],
            'failed_files': {},
        }
    
    def save_state(self):
        """Save current state to disk."""
        self.state['token_usage'] = self.token_usage
        self.state['processed_files'] = list(self.processed_files)
        self.state['failed_files'] = self.failed_files
        self.state['last_updated'] = datetime.now().isoformat()
        
        # Write to temp file first then rename (atomic operation)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
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
        print(f"✅ State saved. Resume with: python embed_v3.py --resume")
        sys.exit(0)
    
    def mark_processed(self, file_path: str, tokens_used: int = 0):
        """Mark a file as processed."""
        self.processed_files.add(file_path)
        if tokens_used:
            collection = self.token_usage.get('total', 0)
            self.token_usage['total'] = collection + tokens_used
            self.token_usage[file_path] = tokens_used
        # Save state after each file
        if len(self.processed_files) % 10 == 0:  # Save every 10 files
            self.save_state()
    
    def mark_failed(self, file_path: str, error: str):
        """Mark a file as failed."""
        self.failed_files[file_path] = {
            'error': str(error)[:500],  # Limit error message length
            'timestamp': datetime.now().isoformat()
        }
        # Don't save after every failure to avoid slowdown
    
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

## Removed duplicate chunking helpers; using embed_utils.simple_chunk_document and group_chunks_for_voyage

def embed_file(
    file_path: Path,
    collection_name: str,
    state: EmbeddingState,
    force: bool = False
) -> int:
    """Embed a single file."""
    
    file_str = str(file_path)
    
    # Check if already processed (unless forcing)
    if not force and not state.should_process(file_str):
        return 0
    
    # Mark as current file being processed
    state.current_file = file_str
    
    try:
        # Read file
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        if not content.strip() or len(content) < 50:
            state.mark_processed(file_str, 0)
            return 0
        
        # Get or create collection (fixed: use name not ID)
        try:
            collection = client.get_collection(collection_name)
        except:
            collection = client.create_collection(
                name=collection_name,
                metadata={"created_at": datetime.now().isoformat()}
            )
        
        # Simple chunking to avoid recursion issues
        chunks = simple_chunk_document(content)
        chunk_groups = group_chunks_for_voyage(chunks)
        
        if not chunk_groups:
            state.mark_processed(file_str, 0)
            return 0
        
        total_tokens = 0
        successfully_embedded = 0
        
        # Process each group
        for group_idx, chunk_group in enumerate(chunk_groups):
            try:
                # Skip if group is empty
                if not chunk_group or not any(chunk.strip() for chunk in chunk_group):
                    continue
                
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
                        documents=[chunk[:65536]],  # ChromaDB has document size limit
                        metadatas=[{
                            "source": file_str,
                            "file_name": file_path.name,
                            "group_index": group_idx,
                            "chunk_index": chunk_idx,
                            "total_groups": len(chunk_groups),
                            "embedded_at": datetime.now().isoformat(),
                            "embedding_model": "voyage-context-3"
                        }],
                        ids=[chunk_id]
                    )
                
                total_tokens += embeds_obj.total_tokens
                successfully_embedded += 1
                
            except Exception as e:
                # Log error but continue with next group
                if "rate limit" in str(e).lower():
                    # Wait and retry once for rate limits
                    time.sleep(20)
                    try:
                        embeds_obj = vo.contextualized_embed(
                            inputs=[chunk_group],
                            model="voyage-context-3",
                            input_type="document"
                        )
                        total_tokens += embeds_obj.total_tokens
                        successfully_embedded += 1
                    except:
                        pass
                elif group_idx == 0:
                    # If first group fails, skip entire file
                    raise e
        
        # Mark as processed if we embedded at least something
        if successfully_embedded > 0:
            state.mark_processed(file_str, total_tokens)
        else:
            state.mark_failed(file_str, "No groups successfully embedded")
        
        state.current_file = None
        return total_tokens
        
    except Exception as e:
        error_msg = str(e)
        # Simplify common errors
        if "does not exist" in error_msg:
            error_msg = "Collection error"
        elif "recursion" in error_msg:
            error_msg = "File too complex"
        
        state.mark_failed(file_str, error_msg)
        state.current_file = None
        return 0

def scan_files(directory: Path, state: EmbeddingState) -> List[Path]:
    """Scan directory for files to process."""
    from embed_utils import should_skip_file
    
    files_to_process = []
    
    for file_path in directory.rglob("*"):
        if file_path.is_file() and not should_skip_file(file_path):
            if state.should_process(str(file_path)):
                files_to_process.append(file_path)
    
    # Sort by size (process smaller files first for quick wins)
    files_to_process.sort(key=lambda f: f.stat().st_size)
    
    return files_to_process

def main():
    """Main embedding pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed files with Voyage AI v3")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state")
    parser.add_argument("--reset", action="store_true", help="Reset and start fresh")
    parser.add_argument("--force", action="store_true", help="Force re-embedding of all files")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
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
            'version': 3,
            'created_at': datetime.now().isoformat(),
            'token_usage': {},
            'processed_files': [],
            'failed_files': {},
        }
        state.save_state()
        
        # Reset ChromaDB collections
        for collection_name in ["knowledgebase", "code", "conversations"]:
            try:
                client.delete_collection(collection_name)
                print(f"   Deleted collection: {collection_name}")
            except:
                pass
    
    elif args.resume:
        print(state.get_resume_stats())
    
    print(f"\n🚀 Knowledge/Code Embedding Pipeline")
    print(f"⚡ No rate limiting (payment method added)")
    
    # Define sources to embed (env-configurable; defaults to repo data/)
    kb_dir = Path(os.environ.get("KB_DIR", "data/knowledgebase"))
    code_dir = Path(os.environ.get("CODE_DIR", "data/code"))
    sources = [
        (kb_dir, "knowledgebase"),
        (code_dir, "code"),
    ]
    
    # Scan all sources
    all_files = []
    for source_path, collection_name in sources:
        if source_path.exists():
            files = scan_files(source_path, state)
            all_files.extend([(f, collection_name) for f in files])
            print(f"📁 {collection_name}: {len(files)} files to process")
    
    if not all_files:
        print("✅ All files already processed!")
        return
    
    print(f"\n📊 Total files to process: {len(all_files)}")
    print("   (Processing smallest files first for quick progress)")
    
    # Process files
    total_tokens = state.token_usage.get('total', 0)
    
    with tqdm(all_files, desc="Embedding files") as pbar:
        for file_path, collection_name in pbar:
            # Update description with current file
            display_name = file_path.name[:40]
            pbar.set_description(f"{collection_name}: {display_name}")
            
            tokens = embed_file(
                file_path,
                collection_name,
                state,
                force=args.force
            )
            
            if tokens > 0:
                total_tokens += tokens
                pbar.set_postfix(tokens=f"{total_tokens:,}")
            
            # Save state periodically
            if len(state.processed_files) % 25 == 0:
                state.save_state()
    
    # Final save
    state.save_state()
    
    print(f"\n✅ Embedding complete!")
    print(f"📊 Total tokens used: {total_tokens:,}")
    print(f"📁 Files processed: {len(state.processed_files)}")
    if state.failed_files:
        print(f"⚠️ Failed files: {len(state.failed_files)}")
        # Show a few examples
        for file, info in list(state.failed_files.items())[:3]:
            print(f"   - {Path(file).name}: {info['error']}")

if __name__ == "__main__":
    main()
