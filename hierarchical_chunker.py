"""Hierarchical chunking with content-aware strategies."""

import tiktoken
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

@dataclass
class ChunkNode:
    """Represents a chunk in the hierarchy."""
    content: str
    level: int  # 0 = document, 1 = section, 2 = subsection, etc.
    parent: Optional['ChunkNode'] = None
    children: List['ChunkNode'] = None
    metadata: Dict = None
    tokens: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}
        self.tokens = len(tokenizer.encode(self.content))

def detect_content_type(path: str, content: str) -> str:
    """Detect specific content type from path and content."""
    path_lower = path.lower()
    
    # Check path patterns first
    if 'journal' in path_lower or 'log-journal' in path_lower:
        return 'journal'
    elif 'raw-log' in path_lower:
        return 'raw_log'
    elif 'moc-' in path_lower or '/moc/' in path_lower:
        return 'moc'
    elif 'claude' in path_lower or 'chatgpt' in path_lower or 'gemini' in path_lower:
        return 'ai_conversation'
    elif 'reflection-' in path_lower or 'incident-' in path_lower:
        return 'reflection'
    elif 'idea-' in path_lower or 'concept-' in path_lower:
        return 'concept_note'
    elif path_lower.endswith('.rs'):
        return 'rust_code'
    elif path_lower.endswith('.py'):
        return 'python_code'
    elif path_lower.endswith('.sql'):
        return 'sql'
    
    # Check content patterns
    if re.search(r'^\*\*\d{4}-\d{2}-\d{2}', content, re.MULTILINE):
        return 'journal'
    elif re.search(r'^\s*- \*\*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\*\*', content, re.MULTILINE):
        return 'raw_log'
    elif 'Human:' in content and 'Assistant:' in content:
        return 'ai_conversation'
    elif re.search(r'^## Core.*\n|^## Main Content|^## Related', content, re.MULTILINE):
        return 'moc'
    
    # Default markdown types based on structure
    if content.count('\n## ') > 3:
        return 'structured_doc'
    else:
        return 'general_note'

def parse_journal_hierarchy(content: str) -> ChunkNode:
    """Parse journal/log entries into hierarchy."""
    root = ChunkNode(content=content, level=0, metadata={'type': 'journal'})
    
    # Pattern for date entries (various formats)
    date_patterns = [
        (r'^## (\d{4}-\d{2}-\d{2})', 1),  # ## 2024-01-15
        (r'^\*\*(\d{4}-\d{2}-\d{2})', 1),  # **2024-01-15
        (r'^### (\d{4}-\d{2}-\d{2})', 1),  # ### 2024-01-15
        (r'^- \*\*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\*\*', 2),  # raw-log style
    ]
    
    for pattern, level in date_patterns:
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        if matches:
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i+1].start() if i+1 < len(matches) else len(content)
                
                entry_content = content[start:end].strip()
                entry_node = ChunkNode(
                    content=entry_content,
                    level=level,
                    parent=root,
                    metadata={'date': match.group(1), 'type': 'journal_entry'}
                )
                root.children.append(entry_node)
                
                # Parse sub-entries within each date
                parse_journal_subentries(entry_node)
            break
    
    # If no date pattern matched, treat as single entry
    if not root.children:
        root.children.append(ChunkNode(
            content=content,
            level=1,
            parent=root,
            metadata={'type': 'journal_entry_unstructured'}
        ))
    
    return root

def parse_journal_subentries(entry_node: ChunkNode):
    """Parse sub-entries within a journal entry (thoughts, events, etc)."""
    content = entry_node.content
    
    # Look for time-stamped sub-entries or bullet points
    time_pattern = r'^(?:- )?(?:\*\*)?(\d{1,2}:\d{2}(?::\d{2})?)'
    bullet_pattern = r'^- (?!\*\*\d)'  # Bullets not followed by timestamp
    
    # Try time-based splitting first
    time_matches = list(re.finditer(time_pattern, content, re.MULTILINE))
    if time_matches:
        for i, match in enumerate(time_matches):
            start = match.start()
            end = time_matches[i+1].start() if i+1 < len(time_matches) else len(content)
            
            sub_content = content[start:end].strip()
            sub_node = ChunkNode(
                content=sub_content,
                level=entry_node.level + 1,
                parent=entry_node,
                metadata={'time': match.group(1), 'type': 'timed_entry'}
            )
            entry_node.children.append(sub_node)
    
    # Otherwise try bullet-based splitting
    elif re.search(bullet_pattern, content, re.MULTILINE):
        bullets = re.split(r'^- ', content, flags=re.MULTILINE)[1:]  # Skip first empty
        for bullet in bullets:
            if bullet.strip():
                sub_node = ChunkNode(
                    content='- ' + bullet.strip(),
                    level=entry_node.level + 1,
                    parent=entry_node,
                    metadata={'type': 'bullet_point'}
                )
                entry_node.children.append(sub_node)

def parse_moc_hierarchy(content: str) -> ChunkNode:
    """Parse MOC (Map of Content) into hierarchy."""
    root = ChunkNode(content=content, level=0, metadata={'type': 'moc'})
    
    # MOCs typically have structured sections
    sections = re.split(r'^## ', content, flags=re.MULTILINE)[1:]  # Skip before first ##
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.split('\n', 1)
        section_title = lines[0].strip()
        section_content = lines[1] if len(lines) > 1 else ''
        
        section_node = ChunkNode(
            content=f"## {section_title}\n{section_content}",
            level=1,
            parent=root,
            metadata={'section': section_title, 'type': 'moc_section'}
        )
        root.children.append(section_node)
        
        # Parse subsections (### headers)
        subsections = re.split(r'^### ', section_content, flags=re.MULTILINE)[1:]
        for subsection in subsections:
            if subsection.strip():
                sub_lines = subsection.split('\n', 1)
                sub_title = sub_lines[0].strip()
                sub_content = sub_lines[1] if len(sub_lines) > 1 else ''
                
                subsection_node = ChunkNode(
                    content=f"### {sub_title}\n{sub_content}",
                    level=2,
                    parent=section_node,
                    metadata={'subsection': sub_title, 'type': 'moc_subsection'}
                )
                section_node.children.append(subsection_node)
    
    return root

def parse_code_hierarchy(content: str, language: str) -> ChunkNode:
    """Parse code into hierarchical structure."""
    root = ChunkNode(content=content, level=0, metadata={'type': f'{language}_code'})
    
    if language == 'python':
        # Parse classes and functions
        class_pattern = r'^class\s+(\w+)'
        func_pattern = r'^(?:async\s+)?def\s+(\w+)'
        
        # Find all classes
        for class_match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = class_match.group(1)
            class_start = class_match.start()
            
            # Find end of class (next class or end of file)
            next_class = re.search(r'^class\s+\w+', content[class_start+1:], re.MULTILINE)
            class_end = class_start + 1 + next_class.start() if next_class else len(content)
            
            class_content = content[class_start:class_end]
            class_node = ChunkNode(
                content=class_content,
                level=1,
                parent=root,
                metadata={'class': class_name, 'type': 'class'}
            )
            root.children.append(class_node)
            
            # Parse methods within class
            for method_match in re.finditer(func_pattern, class_content, re.MULTILINE):
                method_name = method_match.group(1)
                method_node = ChunkNode(
                    content=f"Method: {method_name}",
                    level=2,
                    parent=class_node,
                    metadata={'method': method_name, 'type': 'method'}
                )
                class_node.children.append(method_node)
    
    elif language == 'rust':
        # Parse modules, structs, impls, functions
        patterns = [
            (r'^mod\s+(\w+)', 'module'),
            (r'^struct\s+(\w+)', 'struct'),
            (r'^impl\s+(?:\w+\s+for\s+)?(\w+)', 'impl'),
            (r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', 'function'),
        ]
        
        for pattern, node_type in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                name = match.group(1)
                node = ChunkNode(
                    content=f"{node_type}: {name}",
                    level=1,
                    parent=root,
                    metadata={'name': name, 'type': node_type}
                )
                root.children.append(node)
    
    return root

def parse_ai_conversation_hierarchy(content: str) -> ChunkNode:
    """Parse AI conversation into turns."""
    root = ChunkNode(content=content, level=0, metadata={'type': 'ai_conversation'})
    
    # Split by conversation turns
    turn_pattern = r'(Human:|Assistant:|User:|Model:|<\|.*?\|>:?)'
    parts = re.split(turn_pattern, content)
    
    current_speaker = None
    current_content = []
    
    for part in parts:
        if re.match(turn_pattern, part):
            # Save previous turn if any
            if current_speaker and current_content:
                turn_node = ChunkNode(
                    content=f"{current_speaker} {''.join(current_content)}",
                    level=1,
                    parent=root,
                    metadata={'speaker': current_speaker.rstrip(':'), 'type': 'turn'}
                )
                root.children.append(turn_node)
            
            current_speaker = part
            current_content = []
        else:
            current_content.append(part)
    
    # Don't forget last turn
    if current_speaker and current_content:
        turn_node = ChunkNode(
            content=f"{current_speaker} {''.join(current_content)}",
            level=1,
            parent=root,
            metadata={'speaker': current_speaker.rstrip(':'), 'type': 'turn'}
        )
        root.children.append(turn_node)
    
    return root

def build_hierarchy(content: str, path: str) -> ChunkNode:
    """Build hierarchical structure based on content type."""
    content_type = detect_content_type(path, content)
    
    if content_type in ['journal', 'raw_log']:
        return parse_journal_hierarchy(content)
    elif content_type == 'moc':
        return parse_moc_hierarchy(content)
    elif content_type == 'ai_conversation':
        return parse_ai_conversation_hierarchy(content)
    elif content_type == 'python_code':
        return parse_code_hierarchy(content, 'python')
    elif content_type == 'rust_code':
        return parse_code_hierarchy(content, 'rust')
    else:
        # Generic markdown parsing by headers
        root = ChunkNode(content=content, level=0, metadata={'type': content_type})
        
        # Split by main headers
        sections = re.split(r'^# ', content, flags=re.MULTILINE)[1:]
        for section in sections:
            if section.strip():
                lines = section.split('\n', 1)
                title = lines[0].strip()
                section_content = lines[1] if len(lines) > 1 else ''
                
                section_node = ChunkNode(
                    content=f"# {title}\n{section_content}",
                    level=1,
                    parent=root,
                    metadata={'title': title, 'type': 'section'}
                )
                root.children.append(section_node)
        
        # If no main headers, try ## headers
        if not root.children:
            sections = re.split(r'^## ', content, flags=re.MULTILINE)[1:]
            for section in sections:
                if section.strip():
                    lines = section.split('\n', 1)
                    title = lines[0].strip()
                    section_content = lines[1] if len(lines) > 1 else ''
                    
                    section_node = ChunkNode(
                        content=f"## {title}\n{section_content}",
                        level=1,
                        parent=root,
                        metadata={'title': title, 'type': 'section'}
                    )
                    root.children.append(section_node)
        
        # If still no structure, treat as single chunk
        if not root.children:
            root.children.append(ChunkNode(
                content=content,
                level=1,
                parent=root,
                metadata={'type': 'unstructured'}
            ))
    
    return root

def flatten_hierarchy_smart(root: ChunkNode, max_tokens: int = 30000) -> List[List[str]]:
    """
    Flatten hierarchy into chunks, preserving structure where possible.
    Returns list of chunk groups for contextualized embedding.
    """
    groups = []
    
    def process_node(node: ChunkNode, include_ancestors: bool = True):
        """Process a node and its children."""
        chunks = []
        
        # For small nodes, include with parent context
        if node.tokens < 1000 and include_ancestors and node.parent:
            # Build context from ancestors
            context_parts = []
            current = node.parent
            while current and current.level > 0:
                # Add minimal context from parent (just title/header)
                header = current.content.split('\n')[0][:200]
                context_parts.insert(0, header)
                current = current.parent
            
            context = '\n'.join(context_parts) + '\n\n'
            chunks.append(context + node.content)
        else:
            chunks.append(node.content)
        
        # Process children
        if node.children:
            # Group small children together
            small_children = []
            small_tokens = 0
            
            for child in node.children:
                if child.tokens < 500:
                    small_children.append(child)
                    small_tokens += child.tokens
                    
                    # Flush small children if getting large
                    if small_tokens > 5000:
                        combined = '\n\n'.join([c.content for c in small_children])
                        chunks.append(combined)
                        small_children = []
                        small_tokens = 0
                else:
                    # Process large child independently
                    if small_children:
                        combined = '\n\n'.join([c.content for c in small_children])
                        chunks.append(combined)
                        small_children = []
                        small_tokens = 0
                    
                    chunks.extend(process_node(child, include_ancestors=False))
            
            # Don't forget remaining small children
            if small_children:
                combined = '\n\n'.join([c.content for c in small_children])
                chunks.append(combined)
        
        return chunks
    
    # Get flattened chunks
    all_chunks = process_node(root)
    
    # Group chunks for contextualized embedding
    current_group = []
    current_tokens = 0
    
    for chunk in all_chunks:
        chunk_tokens = len(tokenizer.encode(chunk))
        
        if chunk_tokens > max_tokens:
            # Split oversized chunk
            if current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
            
            # Split the large chunk (reuse shared splitter)
            from embed_utils import split_long_text
            sub_chunks = split_long_text(chunk, max_tokens - 1000)
            for sub in sub_chunks:
                groups.append([sub])
        elif current_tokens + chunk_tokens > max_tokens:
            # Start new group
            groups.append(current_group)
            current_group = [chunk]
            current_tokens = chunk_tokens
        else:
            current_group.append(chunk)
            current_tokens += chunk_tokens
    
    if current_group:
        groups.append(current_group)
    
    return groups

def hierarchical_chunk_document(content: str, path: str) -> List[List[str]]:
    """
    Main entry point for hierarchical chunking.
    Returns list of chunk groups ready for embedding.
    """
    # Build hierarchy
    root = build_hierarchy(content, path)
    
    # Convert to chunk groups
    groups = flatten_hierarchy_smart(root)
    
    # Log statistics
    total_chunks = sum(len(g) for g in groups)
    print(f"  Hierarchical chunking: {len(groups)} groups, {total_chunks} total chunks")
    print(f"  Content type: {root.metadata.get('type', 'unknown')}")
    
    return groups
