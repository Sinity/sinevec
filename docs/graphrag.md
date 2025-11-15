# GraphRAG in the Sinevec workspace

Microsoft’s [GraphRAG](https://github.com/microsoft/graphrag) CLI ships with this repo so you can experiment without leaving the Sinevec toolchain.

## Prerequisites

- `OPENAI_API_KEY` (or Azure equivalents) in `.env`. The dev shell mirrors it to `GRAPHRAG_API_KEY` automatically.
- Documents to ingest. Either drop files under `var/graphrag/input` or point the config at existing sources (e.g., `/realm/data/irc_logs`, rendered knowledgebase exports, bookmarks CSVs).

## Environment setup

```bash
# enter the standard shell (Python 3.12 + GraphRAG virtualenv + CLI shims)
nix develop

# bootstrap a workspace under var/graphrag (override via GRAPHRAG_ROOT if needed)
nix develop -c graphrag-init
# -> edits go into var/graphrag/settings.yaml (re-run with --force to reset)
```

The helper runs `graphrag init --root <path> --force` using the managed virtualenv at `var/graphrag/.venv`. `GRAPHRAG_ROOT` defaults to `var/graphrag` and is created on demand; add `--force` to regenerate `settings.yaml` if you want a clean slate. It also seeds `var/graphrag/input/sanity.txt` so you can test the pipeline before pointing it at your own corpus.

### Minimum config changes

Structured outputs require a chat model that supports JSON schema. Adjust `var/graphrag/settings.yaml` accordingly:

```yaml
models:
  default_chat_model:
    type: chat
    model_provider: openai
    model: gpt-5-mini          # structured outputs + GPT5 cost tier
    api_key: ${GRAPHRAG_API_KEY}
    concurrent_requests: 10
    tokens_per_minute: 30000  # optional spend clamp
    retry_strategy: exponential_backoff
    max_retries: 6
  default_embedding_model:
    type: embedding
    model_provider: openai
    model: text-embedding-3-large
    api_key: ${GRAPHRAG_API_KEY}

global_search:
  chat_model_id: default_chat_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"
  response_model_id: gpt-5-nano     # optional cheaper query-time override

input:
  storage:
    type: file
    base_dir: "/realm/data/irc_logs"
  file_type: text
```

## Running the pipeline

```bash
graphrag index --root $GRAPHRAG_ROOT
graphrag query --root $GRAPHRAG_ROOT --method local --query "Summarize Sinevec projects"

# optional prompt auto-tuning (writes tuned prompts to $GRAPHRAG_ROOT/prompts)
graphrag prompt-tune --root $GRAPHRAG_ROOT --domain "sinevec knowledgebase"
```

Tips:

- Everything under `$GRAPHRAG_ROOT` is git-ignored; keep multiple sandboxes by overriding `GRAPHRAG_ROOT=/realm/project/sinevec/var/graphrag-<name>`.
- Set `cache.base_dir` to a fast disk (`var/graphrag/cache` by default) to avoid paying for repeat indexing when prompts change.
- Use the `fast` indexing method (see upstream docs) when you only need summaries; it removes most LLM calls and cuts indexing cost to ~25% of the full pipeline.
- The CLI works from the main `nix develop` shell; no secondary shell is required.
- Sample input lives in `var/graphrag/input/sanity.txt`. Run the index/query commands above to reproduce the demo output.

## Cost sketch & controls

| Workflow | Model (current config) | Price (on-demand) | Notes |
|----------|-----------------------|-------------------|-------|
| Indexing **and** query LLMs | `gpt-5-mini` | $0.25 / **1M** input, $2.00 / **1M** output | Used for extraction, summaries, and downstream search responses. |
| Embeddings | `text-embedding-3-large` | $0.13 / **1M** input | Powers `embed_text` / `embed_graph`. |

> **Batch API:** OpenAI’s batch endpoint halves those rates (~$0.125 / 1M input + $1 / 1M output for GPT‑5 mini, ~$0.065 / 1M for embeddings). GraphRAG doesn’t natively submit batch jobs yet; enabling that would require a custom submit/poll wrapper around each workflow.

Budgeting example for a 1 M-token corpus:

- Extraction/chat cost ≈ (3 M input tokens × $0.25/1M) + (0.6 M output tokens × $2/1M) ≈ **$1.95**. (Batch API, once supported externally: ≈ $0.98.)
- Embeddings ≈ 1 M × $0.13/1M ≈ **$0.13**. (Batch API: ≈ $0.065.)

Savings knobs:

1. **Smaller models** – Stay on `gpt-5-mini` for indexing and `gpt-5-nano` for conversational queries; jump to `gpt-5-pro` only when answers demand it.
2. **Fast mode** – Set `index.method: fast` to lean on NLP heuristics (no extraction chat model) when you only need community reports.
3. **Rate caps** – Populate `tokens_per_minute` and `requests_per_minute`; fnllm/LiteLLM respect those ceilings and act as a circuit breaker with exponential backoff + `max_retries`.
4. **Caching** – Leave the file cache enabled (`var/graphrag/cache`) so reruns reuse LLM outputs unless the inputs change.
5. **Prompt tuning** – Run `graphrag prompt-tune` to generate domain-specific prompts, which reduces retries and noisy completions.

## Failure modes & recovery

- **`response_format` not supported** – Switch to a model that advertises structured outputs (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`).
- **Hangs** – Use `--verbose` to watch workflow progress; detailed traces land in `var/graphrag/logs/indexing-engine.log`.
- **Rate limiting** – Set the per-model RPM/TPM fields; fnllm queues and backs off automatically.
- **Cost runaway** – Reduce `chunks.size`, `max_gleanings`, or use `fast` indexing until prompts are tuned.

Refer to [https://microsoft.github.io/graphrag](https://microsoft.github.io/graphrag) for full configuration, prompt tuning, and troubleshooting guidance. Delete `var/graphrag` when you want to reset the workspace.
