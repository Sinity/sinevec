#!/usr/bin/env python3
"""
Embed Raindrop.io bookmarks from CSV into the unified ChromaDB collection.

Default strategy (simple and robust):
- One compact summary chunk per bookmark (title, url, folder, tags, excerpt, note).
- One chunk per highlight (split into parts if too long).
- Uses voyage-3 (non-contextualized) by default.

Optionally, you can enable contextualized embeddings per bookmark with
--contextualized to embed [summary + all highlights] together using
voyage-context-3, which injects header context into each highlight vector.

IDs:
  - Summary:  raindrop#<bookmark_id>#summary
  - Highlight: raindrop#<bookmark_id>#hl<index> (with optional _part<k>)

Metadata includes category='bookmarks', domain, folder, tags, created, favorite,
and item_type ('bookmark_summary'|'bookmark_highlight').
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set

import chromadb
import tiktoken
import voyageai
from dotenv import load_dotenv


load_dotenv()

from voyage_embeddings.embed_utils import (
    get_clients, ensure_collection as _ensure_collection,
    count_tokens, split_long_text, domain_of, contextual_windows,
    CONTEXT_MODEL, EMBED_DIM
)

CSV_PATH = Path(os.environ.get('RAINDROP_CSV', 'data/raindrop/raindrop_bookmarks_19_08_2025.csv'))
tokenizer = tiktoken.get_encoding('cl100k_base')


def parse_highlights(raw: str) -> List[str]:
    s = (raw or '').strip()
    if not s:
        return []
    # Highlights are prefixed by 'Highlight:'
    chunks = re.split(r"(?:^|\n)\s*Highlight:\s*", s)
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


def build_summary_text(row: Dict[str, str]) -> str:
    title = row.get('title') or ''
    url = row.get('url') or ''
    folder = row.get('folder') or ''
    tags = (row.get('tags') or '').strip()
    created = row.get('created') or ''
    excerpt = (row.get('excerpt') or '').strip()
    note = (row.get('note') or '').strip()
    lines = [f"Title: {title}", f"URL: {url}", f"Folder: {folder}", f"Tags: {tags}", f"Created: {created}"]
    if excerpt:
        lines += ["", "Excerpt:", excerpt]
    if note:
        lines += ["", "Note:", note]
    return "\n".join(lines)


def domain_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or '')
    return m.group(1).lower() if m else ''


def embed_bookmarks(
    csv_path: Path,
    contextualized: bool = True,
    limit: int = 0,
) -> Tuple[int, int, int]:
    """Embed bookmarks into unified.
    Returns (bookmarks_processed, chunks_embedded, tokens_used if available).
    """
    vo, client = get_clients()
    col = _ensure_collection(client)

    # Simple resumable state of processed bookmark IDs
    state_path = Path('var/state/raindrop_embed_state.json')
    processed_ids: Set[str] = set()
    if state_path.exists():
        try:
            import json
            with state_path.open('r', encoding='utf-8') as sf:
                data = json.load(sf)
                processed_ids = set(data.get('processed_ids', []))
        except Exception:
            processed_ids = set()

    processed = 0
    embedded = 0
    total_tokens = 0

    with csv_path.open('r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and processed >= limit:
                break
            bid = str(row.get('id') or '').strip()
            if not bid:
                continue
            if bid in processed_ids:
                continue

            title = row.get('title') or ''
            url = row.get('url') or ''
            tags = [t.strip() for t in (row.get('tags') or '').split(',') if t.strip()]
            folder = row.get('folder') or ''
            created = row.get('created') or ''
            favorite = str(row.get('favorite') or '').lower() == 'true'
            cover = row.get('cover') or ''
            dom = domain_of(url)

            summary_text = build_summary_text(row)
            highlights = parse_highlights(row.get('highlights') or '')

            # Build chunks
            chunk_texts: List[str] = []
            chunk_ids: List[str] = []
            chunk_metas: List[Dict] = []

            # Summary
            summary_id = f"raindrop#{bid}#summary"
            chunk_texts.append(summary_text)
            chunk_ids.append(summary_id)
            chunk_metas.append({
                'category': 'bookmarks',
                'subcategory': folder or dom or 'general',
                'source': f'raindrop://{bid}',
                'file_type': 'bookmark_summary',
                'title': title[:500],
                'url': url,
                'domain': dom,
                'tags': ', '.join(tags) if tags else '',
                'created': created,
                'favorite': favorite,
                'cover': cover,
            })

            # Highlights
            for hi, h in enumerate(highlights):
                parts = split_long_text(h, max_tokens=8000)
                for pi, part in enumerate(parts):
                    hid = f"raindrop#{bid}#hl{hi}"
                    if len(parts) > 1:
                        hid += f"_part{pi}"
                    chunk_texts.append(part)
                    chunk_ids.append(hid)
                    chunk_metas.append({
                        'category': 'bookmarks',
                        'subcategory': folder or dom or 'general',
                        'source': f'raindrop://{bid}',
                        'file_type': 'bookmark_highlight',
                        'title': title[:500],
                        'url': url,
                        'domain': dom,
                        'tags': ', '.join(tags) if tags else '',
                        'created': created,
                        'favorite': favorite,
                        'cover': cover,
                        'highlight_index': hi,
                        'part_index': pi if len(parts) > 1 else 0,
                    })

            # Upsert: delete existing ids before adding
            if chunk_ids:
                try:
                    col.delete(ids=chunk_ids)
                except Exception:
                    pass

            # Embed chunk_texts
            try:
                if contextualized:
                    # Compute windows that always include the summary at index 0
                    windows = contextual_windows(chunk_texts, always_include_first=True)

                    seen_ids = set()
                    for w in windows:
                        s, e = w
                        inputs = [chunk_texts[s:e]]
                        try:
                            embeds = vo.contextualized_embed(
                                inputs=inputs,
                                model=CONTEXT_MODEL,
                                input_type='document',
                                output_dimension=EMBED_DIM,
                            )
                        except Exception as ee:
                            # Fallback: embed these parts with voyage-3
                            emb = vo.embed(inputs[0], model='voyage-3', input_type='document')
                            vectors = emb.embeddings
                            total_tokens += int(getattr(emb, 'total_tokens', 0) or 0)
                        else:
                            vectors = embeds.results[0].embeddings
                            total_tokens += int(getattr(embeds, 'total_tokens', 0) or 0)

                        # Map back to global ids/metas
                        add_ids = []
                        add_vecs = []
                        add_docs = []
                        add_metas = []
                        for i, vec in enumerate(vectors):
                            global_idx = s + i
                            cid = chunk_ids[global_idx]
                            if cid in seen_ids:
                                continue
                            seen_ids.add(cid)
                            add_ids.append(cid)
                            add_vecs.append(vec)
                            add_docs.append(inputs[0][i][:65536])
                            m = dict(chunk_metas[global_idx])
                            m['embedding_model'] = CONTEXT_MODEL
                            add_metas.append(m)
                        if add_ids:
                            col.add(ids=add_ids, embeddings=add_vecs, documents=add_docs, metadatas=add_metas)
                            embedded += len(add_ids)
                else:
                    emb = vo.embed(chunk_texts, model='voyage-3', input_type='document')
                    vectors = emb.embeddings
                    total_tokens += int(getattr(emb, 'total_tokens', 0) or 0)
                    metas = []
                    for m in chunk_metas:
                        mm = dict(m)
                        mm['embedding_model'] = 'voyage-3'
                        metas.append(mm)
                    col.add(ids=chunk_ids, embeddings=vectors, documents=[t[:65536] for t in chunk_texts], metadatas=metas)
                    embedded += len(chunk_ids)
            except Exception as e:
                print(f"  ⚠️ Embed failed for bookmark {bid}: {str(e)[:200]}")
                continue

            processed += 1
            processed_ids.add(bid)
            # Save state every 500 items
            state_path.parent.mkdir(parents=True, exist_ok=True)
            if processed % 500 == 0:
                try:
                    import json
                    with state_path.open('w', encoding='utf-8') as sf:
                        json.dump({'processed_ids': sorted(processed_ids)}, sf)
                except Exception:
                    pass

    # Final state save
    try:
        import json
        with state_path.open('w', encoding='utf-8') as sf:
            json.dump({'processed_ids': sorted(processed_ids)}, sf)
    except Exception:
        pass

    return processed, embedded, total_tokens


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Embed Raindrop.io bookmarks CSV into ChromaDB')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of bookmarks to process')
    parser.add_argument('--contextualized', action='store_true', help='Use voyage-context-3 per-bookmark (summary + highlights)')
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f'❌ CSV not found: {CSV_PATH}')
        return

    print('📚 Raindrop CSV:', CSV_PATH)
    print('🗄️  Target DB:', DB_PATH, 'collection:', COLLECTION)
    # Contextualized by default unless user disables
    ctx = True if args.contextualized or args.contextualized is None else args.contextualized
    print('🔧 Mode:', 'contextualized' if ctx else 'standard (voyage-3)')
    processed, embedded, tokens = embed_bookmarks(CSV_PATH, contextualized=ctx, limit=args.limit)
    print('\n✅ Raindrop embedding complete')
    print(f'📦 Bookmarks processed: {processed}')
    print(f'🧩 Chunks embedded: {embedded}')
    if tokens:
        print(f'🔢 Tokens used (reported): {tokens:,}')


if __name__ == '__main__':
    main()
