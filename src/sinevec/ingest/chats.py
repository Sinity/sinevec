from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json, zipfile

from sinevec.embed_utils import (
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


def load_chatgpt_conversations(file_path: Path) -> List[Dict]:
    conversations: List[Dict] = []
    if not file_path.exists():
        return conversations
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for conv in data:
            messages: List[Dict] = []
            mapping = conv.get("mapping", {})
            for _msg_id, msg_data in mapping.items():
                msg = msg_data.get("message")
                if not msg or not msg.get("content"):
                    continue
                role = msg.get("author", {}).get("role", "unknown")
                content = msg["content"]
                if content.get("content_type") == "text":
                    parts = content.get("parts", [])
                    text = None
                    if parts and isinstance(parts[0], str):
                        text = parts[0]
                    if text:
                        messages.append({
                            "role": role,
                            "content": text,
                            "timestamp": msg.get("create_time", ""),
                        })
            try:
                messages.sort(key=lambda m: (m.get("timestamp") is None, str(m.get("timestamp", ""))))
            except Exception:
                pass
            if messages:
                conversations.append({
                    "id": conv.get("id", ""),
                    "title": conv.get("title", "Untitled"),
                    "messages": messages,
                    "created": conv.get("create_time", ""),
                    "updated": conv.get("update_time", ""),
                    "source": "chatgpt",
                })
    except Exception:
        pass
    return conversations


def load_claude_conversations_from_extracted(base_path: Path) -> List[Dict]:
    conversations: List[Dict] = []
    if not base_path.exists():
        return conversations
    for folder in base_path.iterdir():
        if not folder.is_dir() or len(folder.name) != 36:
            continue
        chat_file = folder / "chat.json"
        if not chat_file.exists():
            continue
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages: List[Dict] = []
            for msg in data.get("messages", []):
                role = msg.get("sender", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list) and content:
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    content = "\n".join(text_parts)
                messages.append({"role": role, "content": content})
            if messages:
                conversations.append({
                    "id": folder.name,
                    "title": data.get("name", "Untitled"),
                    "messages": messages,
                    "created": data.get("created_at", ""),
                    "updated": data.get("updated_at", ""),
                    "source": "claude",
                })
        except Exception:
            pass
    return conversations


def load_claude_conversations_from_zip(archive_path: Path) -> List[Dict]:
    conversations: List[Dict] = []
    if not archive_path.exists():
        return conversations
    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            with z.open("conversations.json") as f:
                data = json.load(f)
        if isinstance(data, list):
            for conv in data:
                messages: List[Dict] = []
                for msg in conv.get("chat_messages", []):
                    messages.append({
                        "role": msg.get("sender", "unknown"),
                        "content": msg.get("text", ""),
                        "timestamp": msg.get("created_at", ""),
                    })
                try:
                    messages.sort(key=lambda m: str(m.get("timestamp", "")))
                except Exception:
                    pass
                if messages:
                    conversations.append({
                        "id": conv.get("uuid", ""),
                        "title": conv.get("name", "Untitled"),
                        "messages": messages,
                        "created": conv.get("created_at", ""),
                        "updated": conv.get("updated_at", ""),
                        "source": "claude",
                    })
    except Exception:
        pass
    return conversations


def load_cody_conversations(file_path: Path) -> List[Dict]:
    conversations: List[Dict] = []
    if not file_path.exists():
        return conversations
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for conv in data:
            messages: List[Dict] = []
            for inter in conv.get("interactions", []):
                if inter.get("humanMessage"):
                    messages.append({
                        "role": "user",
                        "content": inter["humanMessage"].get("text", ""),
                    })
                if inter.get("assistantMessage"):
                    messages.append({
                        "role": "assistant",
                        "content": inter["assistantMessage"].get("text", ""),
                    })
            if messages:
                conversations.append({
                    "id": str(conv.get("id", "")),
                    "title": messages[0]["content"][:100] if messages else "Untitled",
                    "messages": messages,
                    "created": conv.get("timestamp", ""),
                    "updated": conv.get("timestamp", ""),
                    "source": "cody",
                })
    except Exception:
        pass
    return conversations


def embed_chats_pipeline(platform: str = "all", limit: int = 0) -> Tuple[int, int, int]:
    """Embed conversations from chatlog sources via contextualized per-message embeddings.
    Returns (conversations_processed, messages_embedded, tokens_used).
    """
    vo, client = get_clients()
    collection = ensure_collection(client, "unified")

    base = Path("data/chatlog")
    conversations: List[Dict] = []

    chatgpt_file = base / "conversations.json"
    chatgpt_convs = load_chatgpt_conversations(chatgpt_file)
    claude_extracted = load_claude_conversations_from_extracted(base)
    claude_zip_convs: List[Dict] = []
    if not claude_extracted:
        for z in sorted(base.glob("claude-ai-data-*.zip")):
            more = load_claude_conversations_from_zip(z)
            if more:
                claude_zip_convs = more
                break
    cody_candidates = sorted(base.glob("cody-chat-history-*.json"))
    cody_convs: List[Dict] = load_cody_conversations(cody_candidates[-1]) if cody_candidates else []

    if platform in ("all", "chatgpt"):
        conversations.extend(chatgpt_convs)
    if platform in ("all", "claude"):
        conversations.extend(claude_extracted if claude_extracted else claude_zip_convs)
    if platform in ("all", "cody"):
        conversations.extend(cody_convs)

    if not conversations:
        return 0, 0, 0
    conversations.sort(key=lambda c: str(c.get("updated", "")), reverse=True)
    if limit and limit > 0:
        conversations = conversations[: limit]

    total_msgs = 0
    total_embedded = 0
    total_tokens = 0
    for conv in conversations:
        msgs = len(conv.get("messages", []))
        total_msgs += msgs
        emb_count, tok = embed_conversation_messages(conv, collection)
        total_embedded += emb_count
        total_tokens += tok
    return len(conversations), total_embedded, total_tokens
