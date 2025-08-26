from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Tuple

from voyage_embeddings.embed_utils import (
    get_clients, ensure_collection, count_tokens, split_long_text, contextual_windows,
    CONTEXT_MODEL, EMBED_DIM, domain_of,
)


def parse_highlights(raw: str) -> List[str]:
    s = (raw or '').strip()
    if not s:
        return []
    import re
    chunks = re.split(r"(?:^|\n)\s*Highlight:\s*", s)
    return [c.strip() for c in chunks if c.strip()]


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


def embed_bookmarks_csv(csv_path: Path, limit: int = 0) -> Tuple[int, int, int]:
    vo, client = get_clients()
    col = ensure_collection(client)

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

            chunk_texts: List[str] = [summary_text]
            chunk_ids: List[str] = [f"raindrop#{bid}#summary"]
            chunk_metas: List[Dict] = [{
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
            }]

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

            try:
                col.delete(ids=chunk_ids)
            except Exception:
                pass

            windows = contextual_windows(chunk_texts, always_include_first=True)
            seen = set()
            for s, e in windows:
                inputs = [chunk_texts[s:e]]
                try:
                    embeds = vo.contextualized_embed(inputs, model=CONTEXT_MODEL, input_type='document', output_dimension=EMBED_DIM)
                    vectors = embeds.results[0].embeddings
                    total_tokens += int(getattr(embeds, 'total_tokens', 0) or 0)
                except Exception:
                    continue
                add_ids: List[str] = []
                add_vecs: List[List[float]] = []
                add_docs: List[str] = []
                add_meta: List[Dict] = []
                for i, vec in enumerate(vectors):
                    gid = chunk_ids[s+i]
                    if gid in seen:
                        continue
                    seen.add(gid)
                    add_ids.append(gid)
                    add_vecs.append(vec)
                    add_docs.append(inputs[0][i][:65536])
                    meta = dict(chunk_metas[s+i])
                    meta['embedding_model'] = CONTEXT_MODEL
                    add_meta.append(meta)
                if add_ids:
                    col.add(ids=add_ids, embeddings=add_vecs, documents=add_docs, metadatas=add_meta)
                    embedded += len(add_ids)

            processed += 1

    return processed, embedded, total_tokens
