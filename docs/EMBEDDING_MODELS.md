# Embedding Models Inventory

Sinevec now records the embedding model name for every new vector that lands in Qdrant.

## Current defaults

- **Contextual pipelines** (chats, knowledgebase, code, bookmarks): `voyage-context-3`
- **Standard embeddings / fallbacks**: `voyage-3`
- **Search queries** default to `VOYAGE_QUERY_MODEL` if set, otherwise `voyage-context-3`.

Adjust these via environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `VOYAGE_CONTEXT_MODEL` | contextual ingestion | `voyage-context-3` |
| `VOYAGE_EMBED_MODEL` | non-contextual ingestion fallback | `voyage-3` |
| `VOYAGE_QUERY_MODEL` | search queries | (inherits contextual default) |

## Backfilling legacy vectors

Older points (pre-Qdrant migration) may still have a missing `embedding_model` payload. Use the CLI helper to fill these gaps in-place:

```bash
# Dry run – see how many vectors would be updated
sinevec backfill-embedding-model --dry-run

# Apply updates, forcing bookmarks to record voyage-3 explicitly
sinevec backfill-embedding-model --category bookmarks --model voyage-3
```

The command iterates through the `unified` collection and updates only records missing the field. Restrict by `--collection`, `--category`, or `--model` as needed.

## Auditing live totals

Run `python tools/quick_stats.py` to print the latest embedding counts, token usage, and per-source summaries. The script reads from the state files managed by the ingestion pipelines and from Qdrant directly. Update the script if you introduce new pipelines or metadata fields.
