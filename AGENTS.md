# Repository Guidelines

## Project Structure & Module Organization
- Source: library and CLI under `src/voyage_embeddings` (entrypoint: `ve`).
- Data stores: `chroma_db/` (current), older `chroma_db_*` are ignored.
- Logs & state: `var/log/`, `var/state/` (git-ignored).
- Environment: `flake.nix` provides a dev shell and bootstraps `.venv`; secrets live in `.env` (ignored).

## Build, Test, and Development Commands
- Dev shell: `nix develop` — creates/activates `.venv`, installs deps and the package, exposes `ve`.
- Run embeddings (CLI): `ve embed-bookmarks --csv <path> [--limit N]`.
- Search: `ve search "your query"` (use `--model` to override).
- Maintenance: `ve inspect-db`, `ve audit-models`, `ve infer-category-models`.
 

## Coding Style & Naming Conventions
- Python 3.11; follow PEP 8 with 4-space indentation and type hints where helpful.
- Scripts: `verb_subject.py` (e.g., `embed_source.py`); shared utilities live in `src/voyage_embeddings/embed_utils.py`.
- Functions/vars use `snake_case`; classes use `CamelCase`; add short docstrings to new public functions.
- Do not commit data, logs, or secrets (ignored: `.env`, `.venv/`, `chroma_db/`, `var/`, `*.log`).

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
