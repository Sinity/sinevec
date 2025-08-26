# Voyage Embeddings

Minimal, clean embedding pipelines using Voyage AI with a unified ChromaDB store.

Core scripts
- `embed_ai_messages.py`: Per-message embeddings for AI chats (ChatGPT/Claude/Cody) using `voyage-context-3`. Stores `granularity=message` in `unified`.
- `embed_raindrop_bookmarks.py`: Per-bookmark contextualized embeddings (summary + highlights in one context). Stores `category=bookmarks` in `unified`.
- `embed_knowledge_code.py`: Contextualized embedding for knowledgebase/code trees; resumable state and simple chunking. Uses `KB_DIR`/`CODE_DIR` or `data/knowledgebase` and `data/code`.
- `search.py`: Search with metadata filters over `unified`.

Shared utilities
- `embed_utils.py` centralizes clients, tokenization, splitting, context windows, file helpers, and config (DB path, model, dimensions).

Data layout
- Vector DB: `chroma_db/` (single current DB)
- Data inputs: `data/` (e.g., `data/chatlog/`, `data/raindrop/`, optionally `data/knowledgebase/`, `data/code/`)
- State/Logs: `var/state/` and `var/log/`

Quick start
- Ensure `.env` contains `VOYAGE_API_KEY`.
- Enter the dev shell: `nix develop` (creates/activates `.venv`, installs deps, and exposes a reliable `ve` command).
- CLI (three reliable ways):
  - `ve ...` (shim added to PATH in dev shell)
  - `nix run .#ve -- ...` (no shell needed; use `--` before args)
  - `python -m voyage_embeddings.cli ...` (direct module run)
  - Examples:
    - Search: `ve search "your query"`
    - Bookmarks: `ve embed-bookmarks --csv data/raindrop/raindrop_bookmarks_19_08_2025.csv --limit 1000`
    - Inspect DB: `ve inspect-db`
- Scripts (still supported):
  - AI chats: `python embed_ai_messages.py --platform all`
  - Raindrop: `python embed_raindrop_bookmarks.py`
  - Knowledgebase/code: `python embed_knowledge_code.py --resume`
  - Search: `python search.py "your query"`

Notes
- This repo removed obsolete one-offs and legacy scripts. Only the above entry points are supported.
- All embeddings target `chroma_db` → `unified` with 1024‑dim vectors.
