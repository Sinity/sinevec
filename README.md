# Voyage Embeddings

Minimal, clean embedding pipelines using Voyage AI with a unified ChromaDB store.

Core
- Prefer the CLI (`ve ...`) for all workflows. See Quick start below.

Shared utilities
- `src/voyage_embeddings/embed_utils.py` centralizes clients, tokenization, splitting, context windows, file helpers, and config (DB path, model, dimensions).

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
    - Search: `ve search "your query"` (filters: `--category`, `--channel`, `--date-from`, etc.)
    - Bookmarks: `ve embed-bookmarks --csv data/raindrop/raindrop_bookmarks_19_08_2025.csv --limit 1000`
    - Chats: `ve embed-chats --platform all --limit 200`
    - Knowledge/Code: `ve embed-knowledge --kb-dir data/knowledgebase --code-dir data/code`
    - Inspect/Audit: `ve inspect-db`, `ve audit-models`

Notes
- This repo removed obsolete one-offs and legacy scripts. Only the above entry points are supported.
- All embeddings target `chroma_db` → `unified` with 1024‑dim vectors.
