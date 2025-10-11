# Sinevec

Sinevec centralizes Voyage AI embeddings in a unified Qdrant vector store.

Core
- Prefer the CLI (`sinevec ...`) for all workflows. See Quick start below.

Shared utilities
- `src/sinevec/embed_utils.py` centralizes clients, tokenization, splitting, context windows, file helpers, and config (DB path, model, dimensions).

Data layout
- Vector DB: Qdrant service on `127.0.0.1:6333` (data under `/realm/data/qdrant`).
- Data inputs: `data/` (e.g., `data/chatlog/`, `data/raindrop/`, optionally `data/knowledgebase/`, `data/code/`)
- State/Logs: `var/state/` and `var/log/`

Quick start
- Ensure `.env` contains `VOYAGE_API_KEY`.
- Enter the dev shell: `nix develop` (creates/activates `.venv`, installs deps, and exposes a reliable `sinevec` command).
- CLI (three reliable ways):
  - `sinevec ...` (shim added to PATH in dev shell)
  - `nix run .#sinevec -- ...` (no shell needed; use `--` before args)
  - `python -m sinevec.cli ...` (direct module run)
  - Examples:
    - Search: `sinevec search "your query"` (filters: `--category`, `--channel`, `--date-from`, etc.; add `--json` for machine-readable output)
    - Bookmarks: `sinevec embed-bookmarks --csv data/raindrop/raindrop_bookmarks_19_08_2025.csv --limit 1000`
    - Chats: `sinevec embed-chats --platform all --limit 200`
    - Knowledge/Code: `sinevec embed-knowledge --kb-dir data/knowledgebase --code-dir data/code`
    - Inspect: `sinevec inspect-db`
    - Serve UI: `sinevec serve`
    - List indexed filters: `sinevec options [--category knowledgebase]`

Vector store configuration
- Qdrant connection is controlled via `QDRANT_HOST`, `QDRANT_HTTP_PORT`, `QDRANT_GRPC_PORT`, `QDRANT_API_KEY`, `QDRANT_VECTOR_SIZE` (defaults to 1024), and `QDRANT_CLIENT_TIMEOUT`.
- `sinevec inspect-db` reports available collections and counts so you can confirm ingest progress.

Notes
- This repo removed obsolete one-offs and legacy scripts. Only the above entry points are supported.
- Qdrant stores point IDs as deterministic UUIDs but keeps the original ID in payloads so existing tooling (search, deletes) continues to work.
