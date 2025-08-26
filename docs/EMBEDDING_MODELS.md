# Embedding Models Inventory

This document tracks which Voyage models are used across the local ChromaDB store and in the pipelines.

Summary (generated via `ve audit-models` on this machine):

- By model:
  - voyage-context-3: 24,395 (conversations)
  - <missing>: 46,704 (bookmarks, reddit, irc, knowledgebase, code)

- By model x category (top):
  - voyage-context-3 × conversations: 24,395
  - <missing> × bookmarks: 23,263
  - <missing> × reddit: 14,902
  - <missing> × irc: 4,301
  - <missing> × knowledgebase: 2,349
  - <missing> × code: 1,889

Notes:
- Conversations (ChatGPT/Claude/Cody) are embedded with `voyage-context-3` (contextualized, 32K context).
- Prior bookmark/code/knowledgebase runs didn’t record `embedding_model` in metadata; the audit lists them as `<missing>`.
- Knowledgebase/code pipelines have been using `voyage-context-3`; starting now they also record `embedding_model`.
- Bookmarks pipeline now records `embedding_model` as either `voyage-context-3` (contextualized path) or `voyage-3` (fallback). Older entries remain without this field.

Operational defaults (current):
- Query default: `VOYAGE_QUERY_MODEL` (CLI `ve search`) can be set; if unset, the CLI auto‑routes and falls back to `voyage-2` for broad compatibility. You can force contextual with `--model voyage-context-3`.
- Ingestion defaults: `VOYAGE_CONTEXT_MODEL` and `VOYAGE_EMBED_MODEL` (see `src/voyage_embeddings/embed_utils.py`). Override via environment to match your stored vectors.

How to refresh this report:
1) Run `ve audit-models` to print a fresh summary and write `var/reports/embedding_model_audit.md`.
2) Update this document if notable changes occur (e.g., switching default models or re‑embedding).

