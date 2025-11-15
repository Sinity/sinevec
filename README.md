# Sinevec

Sinevec centralizes Voyage AI embeddings in a unified Qdrant vector store.

Core
- Prefer the CLI (`sinevec ...`) for all workflows. See Quick start below.

Shared utilities
- `src/sinevec/embed_utils.py` centralizes clients, tokenization, splitting, context windows, file helpers, and configuration (vector DB host, data/state roots, model defaults).

Data layout
- Vector DB: Qdrant service on `127.0.0.1:6333` (data under `/realm/data/qdrant`).
- Content roots: `SINEVEC_DATA_ROOT` (default `~/.local/share/sinevec`) and `SINEVEC_STATE_DIR` (default `~/.local/state/sinevec`).
- Logs: `SINEVEC_LOG_DIR` (default `~/.local/state/sinevec/log`).

Quick start
- Ensure `.env` contains `VOYAGE_API_KEY`.
- Enter the dev shell: `nix develop` (creates/activates `.venv`, installs deps, and exposes a reliable `sinevec` command).
- CLI (three reliable ways):
  - `sinevec ...` (shim added to PATH in dev shell)
  - `nix run .#sinevec -- ...` (no shell needed; use `--` before args)
  - `python -m sinevec.cli ...` (direct module run)
  - Examples:
    - Search: `sinevec search "your query"` (filters: `--category`, `--channel`, `--date-from`, etc.; add `--json` for machine-readable output)
    - Bookmarks: `sinevec embed-bookmarks --limit 1000` (reads from `$SINEVEC_DATA_ROOT/raindrop/…`; add `--force` to re-embed)
    - Chats: `sinevec embed-chats --platform all --limit 200` (state-aware; use `--force` to reprocess)
    - Knowledge/Code: `sinevec embed-knowledge` (defaults follow `$SINEVEC_DATA_ROOT`; `--force` re-embeds everything)
    - Inspect: `sinevec inspect-db`
    - Backfill metadata: `sinevec backfill-embedding-model --dry-run`
- Serve UI: `sinevec serve`
- Build package: `nix build .#sinevec` (produces a runnable CLI binary in `result/bin/sinevec`)
- List indexed filters: `sinevec options [--category knowledgebase]`

GraphRAG experiments
- A ready-to-use GraphRAG dev shell and helper script live in this repo. See [`docs/graphrag.md`](docs/graphrag.md) for instructions on `nix develop .#graphrag`, initializing a workspace under `var/graphrag`, and running `graphrag index/query` against local data.

Vector store configuration
- Qdrant connection is controlled via `QDRANT_HOST`, `QDRANT_HTTP_PORT`, `QDRANT_GRPC_PORT`, `QDRANT_API_KEY`, `QDRANT_VECTOR_SIZE` (defaults to 1024), and `QDRANT_CLIENT_TIMEOUT`.
- `sinevec inspect-db` reports available collections and counts so you can confirm ingest progress.

Notes
- This repo removed obsolete one-offs and legacy scripts. Only the above entry points are supported.
- Qdrant stores point IDs as deterministic UUIDs but keeps the original ID in payloads so existing tooling (search, deletes) continues to work.
- Ingestion pipelines persist their progress under `$SINEVEC_STATE_DIR`; if you move state between machines, copy that directory alongside the data.
