# Repository Guidelines

## Project Structure & Module Organization
- Source: Python scripts live at the repo root (e.g., `embed_*.py`, `search*.py`, `hierarchical_chunker.py`, `utils.py`, `smart_utils.py`).
- Data stores: `chroma_db*/` directories hold local ChromaDB collections (generated, git-ignored).
- Logs & state: `logs/`, `*.log`, and `*_state.json` track progress and recoveries (git-ignored).
- Environment: `flake.nix` provides a Nix dev shell and bootstraps `.venv`; secrets live in `.env`.

## Build, Test, and Development Commands
- Dev shell: `nix develop` — creates/activates `.venv`, installs deps (`voyageai`, `chromadb`, `langchain-text-splitters`, `numpy`, `tqdm`, `python-dotenv`, `gitpython`, `tiktoken`) and loads `VOYAGE_API_KEY` from `.env`.
- Run embeddings: `python embed_everything.py`, `python embed_reddit.py`, `python embed_ai_conversations.py`, `python embed_v3.py`.
- Search: `python search.py "your query"` or `python search_v2.py "your query"`.
- Maintenance: `python inspect_chromadb.py`, `python migrate_chromadb.py`, `python recover_embeddings.py`, `python unify_collections.py`.
- Quick check: `python simple_embed_test.py` — minimal end-to-end sanity test.

## Coding Style & Naming Conventions
- Python 3.11; follow PEP 8 with 4-space indentation and type hints where helpful.
- Scripts: `verb_subject.py` (e.g., `embed_source.py`); utilities in `*_utils.py` or `utils.py`.
- Functions/vars use `snake_case`; classes use `CamelCase`; add short docstrings to new public functions.
- Do not commit data, logs, or secrets (`.gitignore` covers `.env`, `.venv/`, `chroma_db/`, `logs/`, `*.log`).

## Testing Guidelines
- Prefer small, runnable test scripts next to code (e.g., `something_test.py`), callable via `python file_test.py`.
- Keep tests deterministic; avoid external calls unless `VOYAGE_API_KEY` is present. Log to `logs/` if needed.
- If adopting a framework later, use `tests/` with `test_*.py` naming and keep fixtures minimal.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (<72 chars), focused scope. Example: `embed: add Reddit ingestion with rate limit`.
- PRs: include summary, linked issues, example commands and outputs, notes on schema/collection changes, and any migration steps.

## Security & Configuration Tips
- Create `.env` with `VOYAGE_API_KEY=<your key>`; never commit `.env` or tokens.
- Be careful with logs; avoid writing raw secrets. Large artifacts belong in `chroma_db*/` or `logs/` (already ignored).
