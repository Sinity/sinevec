from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from voyage_embeddings.embed_utils import (
    get_clients, ensure_collection, count_tokens, detect_code,
    CONTEXT_MODEL, EMBED_DIM, MAX_DOC_TOKENS,
)


def format_msg(role: str, content: str) -> str:
    role = (role or "unknown").upper()
    return f"{role}:\n{content}"


def embed_conversation_messages(conv: Dict, collection) -> Tuple[int, int]:
    vo, _ = get_clients()
    total_tokens = 0
    embedded = 0
    platform = conv.get("source", "unknown")
    conv_id = str(conv.get("id", ""))
    messages = conv.get("messages", [])
    if not messages:
        return 0, 0

    msg_texts: List[str] = [format_msg(m.get("role", "unknown"), str(m.get("content", ""))) for m in messages]
    msg_tokens: List[int] = [count_tokens(t) for t in msg_texts]

    windows: List[Tuple[int, int]] = []
    start = 0
    while start < len(msg_texts):
        total = 0
        end = start
        while end < len(msg_texts) and total + msg_tokens[end] <= MAX_DOC_TOKENS:
            total += msg_tokens[end]
            end += 1
        if end == start:
            end = start + 1
        windows.append((start, end))
        start = end

    title = str(conv.get("title", "Untitled"))
    n = len(messages)

    for (w_start, w_end) in windows:
        window_texts = msg_texts[w_start:w_end]
        try:
            ctx = vo.contextualized_embed(
                inputs=[window_texts], model=CONTEXT_MODEL, input_type="document", output_dimension=EMBED_DIM
            )
        except Exception:
            continue
        vectors = ctx.results[0].embeddings if ctx.results else []
        total_tokens += int(getattr(ctx, "total_tokens", 0) or 0)

        to_add_ids: List[str] = []
        to_add_embs: List[List[float]] = []
        to_add_docs: List[str] = []
        to_add_meta: List[Dict] = []

        for offset, (msg, vec) in enumerate(zip(messages[w_start:w_end], vectors)):
            i = w_start + offset
            msg_id = f"message#{platform}#{conv_id}#msg{i}"
            try:
                existing = collection.get(ids=[msg_id])
                if existing.get("ids"):
                    continue
            except Exception:
                pass
            doc = msg_texts[i][:65536]
            metadata = {
                "granularity": "message",
                "contextualized": True,
                "embedding_model": CONTEXT_MODEL,
                "category": "conversations",
                "subcategory": platform,
                "source": f"chatlog/{platform}",
                "file_type": "ai_conversation",
                "conversation_id": conv_id,
                "conversation_title": title[:500],
                "message_index": i,
                "num_messages": n,
                "role": str(msg.get("role", "unknown")),
                "has_code": detect_code(doc),
                "created": str(conv.get("created", "")),
                "updated": str(conv.get("updated", "")),
            }
            to_add_ids.append(msg_id)
            to_add_embs.append(vec)
            to_add_docs.append(doc)
            to_add_meta.append(metadata)

        if to_add_ids:
            collection.add(ids=to_add_ids, embeddings=to_add_embs, documents=to_add_docs, metadatas=to_add_meta)
            embedded += len(to_add_ids)

    return embedded, total_tokens


