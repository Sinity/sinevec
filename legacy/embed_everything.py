#!/usr/bin/env python3
"""
Embed everything using Voyage AI contextualized embeddings.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import time
import voyageai
import chromadb
from tqdm import tqdm
import git
from dotenv import load_dotenv
from smart_utils import (
    count_tokens, get_file_type, smart_split_text
)
from hierarchical_chunker import hierarchical_chunk_document
from utils import extract_timestamp, should_skip_file

# Load environment variables
load_dotenv()

# Initialize Voyage client
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Token tracking
class TokenTracker:
    def __init__(self, budget=200_000_000):
        self.total_tokens = 0
        self.budget = budget
        self.log_file = Path("logs/token_usage.json")
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Load existing usage if any
        if self.log_file.exists():
            with open(self.log_file) as f:
                data = json.load(f)
                self.total_tokens = data.get("total_tokens", 0)
    
    def add(self, tokens):
        self.total_tokens += tokens
        self.save()
        
    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump({
                "total_tokens": self.total_tokens,
                "budget": self.budget,
                "remaining": self.budget - self.total_tokens,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    
    def can_continue(self, estimated_tokens):
        return self.total_tokens + estimated_tokens < self.budget
    
    def status(self):
        remaining = self.budget - self.total_tokens
        percent = (self.total_tokens / self.budget) * 100
        return f"Used: {self.total_tokens:,} ({percent:.2f}%) | Remaining: {remaining:,}"

tracker = TokenTracker()

def create_collections():
    """Create ChromaDB collections for different data types."""
    collections = {}
    
    collection_names = [
        ("knowledgebase", "Personal knowledge vault"),
        ("code", "Source code and projects"),
        ("conversations", "AI chat logs"),
        ("git_history", "Git commits and diffs"),
        ("documents", "General documents")
    ]
    
    for name, description in collection_names:
        try:
            collection = client.get_collection(name)
            print(f"✓ Using existing collection: {name}")
        except:
            collection = client.create_collection(
                name=name,
                metadata={"description": description}
            )
            print(f"✓ Created collection: {name}")
        collections[name] = collection
    
    return collections

def embed_file(file_path: Path, collection_name: str, collections: dict):
    """Embed a single file."""
    try:
        # Read content
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        if not content.strip():
            return 0
        
        # Skip if too small
        if len(content) < 100:
            return 0
        
        # Extract metadata
        timestamp = extract_timestamp(content, str(file_path))
        file_type = get_file_type(str(file_path))
        
        # Use hierarchical chunking for better structure preservation
        chunk_groups = hierarchical_chunk_document(content, str(file_path))
        
        if not chunk_groups:
            return 0
        
        # Count actual tokens
        estimated_tokens = sum(sum(count_tokens(chunk) for chunk in group) for group in chunk_groups)
        
        if not tracker.can_continue(estimated_tokens):
            print(f"⚠️ Token budget would be exceeded. Skipping {file_path}")
            return 0
        
        # Embed each group
        collection = collections[collection_name]
        total_tokens = 0
        
        for group_idx, chunk_group in enumerate(chunk_groups):
            # Use Voyage contextualized embeddings
            # For code, we use voyage-code-3 if available, otherwise voyage-context-3
            model = "voyage-code-3" if "code" in file_type and "code" in collection_name else "voyage-context-3"
            
            try:
                # Embed the group with rate limit handling
                for retry in range(3):
                    try:
                        embeds_obj = vo.contextualized_embed(
                            inputs=[chunk_group],
                            model="voyage-context-3",  # voyage-code-3 might not support contextualized
                            input_type="document"
                        )
                        break
                    except Exception as e:
                        if "rate limit" in str(e).lower() and retry < 2:
                            time.sleep(20)  # Wait 20 seconds for rate limit (3 RPM = 20s between requests)
                            continue
                        else:
                            raise
                
                # Store each chunk with its embedding
                for chunk_idx, (chunk, embedding) in enumerate(zip(chunk_group, embeds_obj.results[0].embeddings)):
                    chunk_id = f"{file_path}#g{group_idx}#c{chunk_idx}"
                    
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "source": str(file_path),
                            "timestamp": timestamp,
                            "file_type": file_type,
                            "group_index": group_idx,
                            "chunk_index": chunk_idx,
                            "total_groups": len(chunk_groups),
                            "total_chunks_in_group": len(chunk_group)
                        }],
                        ids=[chunk_id]
                    )
                
                total_tokens += embeds_obj.total_tokens
                
            except Exception as e:
                print(f"  Error embedding group {group_idx}: {e}")
                continue
        
        tracker.add(total_tokens)
        return total_tokens
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return 0

def embed_git_history(repo_path: Path, collection_name: str, collections: dict):
    """Embed git commits with diffs."""
    try:
        repo = git.Repo(repo_path)
        collection = collections[collection_name]
        
        print(f"\n📦 Processing git history for {repo_path}...")
        
        commits_to_process = []
        
        # Get all commits
        for commit in tqdm(repo.iter_commits(), desc="Reading commits"):
            commit_data = {
                "hash": commit.hexsha,
                "message": commit.message,
                "author": str(commit.author),
                "date": datetime.fromtimestamp(commit.committed_date).isoformat(),
                "diff": ""
            }
            
            # Get diff (limit size to avoid huge diffs)
            try:
                if commit.parents:
                    diff = repo.git.diff(commit.parents[0].hexsha, commit.hexsha, '--stat')
                    diff += "\n\n"
                    diff += repo.git.diff(commit.parents[0].hexsha, commit.hexsha, '--', '*.rs', '*.py', '*.md', '*.sql', '*.toml')
                    
                    # Limit diff size
                    if len(diff) > 50000:
                        diff = diff[:50000] + "\n... [diff truncated]"
                    
                    commit_data["diff"] = diff
            except:
                pass
            
            commits_to_process.append(commit_data)
        
        # Group commits for embedding (batch by time periods or size)
        commit_batches = []
        current_batch = []
        current_size = 0
        
        for commit in commits_to_process:
            commit_text = f"Commit: {commit['hash'][:8]}\nAuthor: {commit['author']}\nDate: {commit['date']}\nMessage: {commit['message']}\n\nDiff:\n{commit['diff']}"
            size = count_tokens(commit_text)
            
            if current_size + size > 25000:  # Keep batches under 25K tokens
                if current_batch:
                    commit_batches.append(current_batch)
                current_batch = [commit_text]
                current_size = size
            else:
                current_batch.append(commit_text)
                current_size += size
        
        if current_batch:
            commit_batches.append(current_batch)
        
        # Embed batches
        total_tokens = 0
        for batch_idx, batch in enumerate(tqdm(commit_batches, desc="Embedding commits")):
            try:
                for retry in range(3):
                    try:
                        embeds_obj = vo.contextualized_embed(
                            inputs=[batch],
                            model="voyage-context-3",
                            input_type="document"
                        )
                        break
                    except Exception as e:
                        if "rate limit" in str(e).lower() and retry < 2:
                            time.sleep(20)  # Wait 20 seconds for rate limit
                            continue
                        else:
                            raise
                
                for commit_idx, (commit_text, embedding) in enumerate(zip(batch, embeds_obj.results[0].embeddings)):
                    commit_id = f"{repo_path}#batch{batch_idx}#commit{commit_idx}"
                    
                    # Extract hash from commit text
                    hash_match = re.search(r"Commit: ([a-f0-9]{8})", commit_text)
                    commit_hash = hash_match.group(1) if hash_match else "unknown"
                    
                    collection.add(
                        embeddings=[embedding],
                        documents=[commit_text],
                        metadatas=[{
                            "source": str(repo_path),
                            "type": "git_commit",
                            "commit_hash": commit_hash,
                            "batch_index": batch_idx,
                            "commit_index": commit_idx
                        }],
                        ids=[commit_id]
                    )
                
                total_tokens += embeds_obj.total_tokens
                tracker.add(embeds_obj.total_tokens)
                
            except Exception as e:
                print(f"  Error embedding batch {batch_idx}: {e}")
                continue
        
        print(f"  ✓ Embedded {len(commits_to_process)} commits using {total_tokens:,} tokens")
        return total_tokens
        
    except Exception as e:
        print(f"  Error processing git history: {e}")
        return 0

def embed_directory(directory: Path, collection_name: str, collections: dict):
    """Recursively embed all files in a directory."""
    total_tokens = 0
    files_processed = 0
    
    # Get all files
    all_files = []
    for file_path in directory.rglob("*"):
        if file_path.is_file() and not should_skip_file(file_path):
            all_files.append(file_path)
    
    print(f"\n📁 Processing {len(all_files)} files from {directory}...")
    
    for file_path in tqdm(all_files, desc=f"Embedding {directory.name}"):
        tokens = embed_file(file_path, collection_name, collections)
        if tokens > 0:
            total_tokens += tokens
            files_processed += 1
    
    print(f"  ✓ Processed {files_processed} files using {total_tokens:,} tokens")
    return total_tokens

def main():
    """Main embedding pipeline."""
    print("🚀 Voyage Embedding Pipeline")
    print(f"📊 Token Budget: {tracker.status()}")
    
    # Create collections
    collections = create_collections()
    
    # Define what to embed
    tasks = [
        # Knowledgebase
        (Path("/realm/knowledgebase"), "knowledgebase", "directory"),
        
        # Sinex project
        (Path("/realm/project/sinex"), "code", "directory"),
        (Path("/realm/project/sinex"), "git_history", "git"),
        
        # Sinex analysis docs
        (Path("/realm/project/sinex-analysis/gemini-sinex-discussions"), "conversations", "directory"),
        
        # Chat logs (excluding claude jsonl files for now)
        (Path("/realm/data/chatlog/chatgpt-data-2025-01-31-02-51-50.zip"), "conversations", "archive"),
        (Path("/realm/data/chatlog/claude-ai-data-2025-01-31-01-35-27.zip"), "conversations", "archive"),
        
        # Knowledgebase git history
        (Path("/realm/knowledgebase"), "git_history", "git"),
    ]
    
    # Process each task
    for path, collection_name, task_type in tasks:
        if not tracker.can_continue(1000000):  # Check if we have at least 1M tokens left
            print(f"\n⚠️ Approaching token limit. Stopping.")
            break
        
        if task_type == "directory":
            if path.exists():
                embed_directory(path, collection_name, collections)
        
        elif task_type == "git":
            if path.exists() and (path / ".git").exists():
                embed_git_history(path, collection_name, collections)
        
        elif task_type == "archive":
            # Extract and process archives
            if path.exists():
                print(f"\n📦 Processing archive {path.name}...")
                import zipfile
                import tempfile
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmppath = Path(tmpdir)
                    
                    try:
                        with zipfile.ZipFile(path, 'r') as zf:
                            zf.extractall(tmppath)
                        
                        # Process extracted files
                        embed_directory(tmppath, collection_name, collections)
                    except Exception as e:
                        print(f"  Error processing archive: {e}")
        
        print(f"\n📊 Status: {tracker.status()}")
    
    print("\n✅ Embedding complete!")
    print(f"📊 Final: {tracker.status()}")

if __name__ == "__main__":
    import re
    main()