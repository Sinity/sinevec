from __future__ import annotations

import os
from pathlib import Path
import typer

from voyage_embeddings.embed_utils import get_clients, ensure_collection
from pathlib import Path
from voyage_embeddings.ingest.bookmarks import embed_bookmarks_csv
from voyage_embeddings.ingest.chats import embed_conversation_messages, embed_chats_pipeline
from voyage_embeddings.ingest.knowledge_code import embed_knowledge_code_pipeline

app = typer.Typer(help="Voyage Embeddings CLI")


@app.command("search")
def search_cmd(
    query: str,
    n: int = typer.Option(10, "--n"),
    model: str | None = typer.Option(None, "--model", help="Override query model; defaults to VOYAGE_QUERY_MODEL or auto"),
    category: str | None = typer.Option(None, "--category"),
    subcategory: str | None = typer.Option(None, "--subcategory"),
    channel: str | None = typer.Option(None, "--channel"),
    date_from: str | None = typer.Option(None, "--date-from"),
    date_to: str | None = typer.Option(None, "--date-to"),
    has_code: bool = typer.Option(False, "--has-code"),
    has_urls: bool = typer.Option(False, "--has-urls"),
):
    import voyageai, chromadb
    vo, client = get_clients()
    col = ensure_collection(client)
    model = model or os.environ.get("VOYAGE_QUERY_MODEL") or os.environ.get("VOYAGE_CONTEXT_MODEL") or "voyage-2"

    # Use contextualized endpoint when requesting a contextualized model
    try:
        if "context" in model:
            ctx = vo.contextualized_embed(inputs=[[query]], model=model, input_type="query")
            qv = ctx.results[0].embeddings[0]
        else:
            qv = vo.embed([query], model=model, input_type="query").embeddings[0]
    except Exception as e:
        typer.echo(f"Error embedding query with model '{model}': {e}")
        typer.echo("Hint: try --model voyage-2 or set VOYAGE_QUERY_MODEL.")
        raise typer.Exit(1)

    # Build filter
    where_filter = {}
    conditions = []
    if category:
        conditions.append({"category": category})
    if subcategory:
        conditions.append({"subcategory": subcategory})
    if channel:
        conditions.append({"channel": channel})
    if has_code:
        conditions.append({"has_code": True})
    if has_urls:
        conditions.append({"has_urls": True})
    if date_from or date_to:
        df = {}
        if date_from:
            df["$gte"] = date_from
        if date_to:
            df["$lte"] = date_to
        conditions.append({"date": df})
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    try:
        res = col.query(query_embeddings=[qv], n_results=n, where=where_filter or None)
    except Exception as e:
        typer.echo(f"Query error: {e}")
        raise typer.Exit(2)

    ids = res.get('ids') or []
    if not ids or not ids[0]:
        typer.echo("No results found.")
        raise typer.Exit(0)

    for i in range(len(ids[0])):
        print(f"\n[{i+1}] {ids[0][i]} :: {res['distances'][0][i]:.4f}")
        meta = res['metadatas'][0][i]
        print(meta)
        print(res['documents'][0][i][:300])


@app.command("embed-chats")
def embed_chats(platform: str = typer.Option("all", "--platform"), limit: int = typer.Option(0, "--limit")):
    processed, embedded, tokens = embed_chats_pipeline(platform=platform, limit=limit)
    print("\nDone:")
    print(f" conversations={processed} messages_embedded={embedded} tokens={tokens}")


@app.command("embed-knowledge")
def embed_knowledge(kb_dir: Path = typer.Option(Path("data/knowledgebase"), "--kb-dir"), code_dir: Path = typer.Option(Path("data/code"), "--code-dir"), force: bool = typer.Option(False, "--force")):
    processed, tokens = embed_knowledge_code_pipeline(kb_dir=kb_dir, code_dir=code_dir, force=force)
    print("\nDone:")
    print(f" files_processed={processed} tokens={tokens}")


@app.command("audit-models")
def audit_models(write_report: bool = typer.Option(True, "--write-report/--no-write-report")):
    """Summarize which embedding models were used across the DB."""
    import sqlite3
    from pathlib import Path

    db_path = Path("chroma_db/chroma.sqlite3")
    if not db_path.exists():
        print("DB not found at chroma_db/chroma.sqlite3")
        raise SystemExit(1)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    def q(sql: str):
        cur.execute(sql)
        return cur.fetchall()

    rows = q(
        """
        SELECT coalesce(em_model.string_value,'<missing>') AS model,
               COUNT(DISTINCT e.id) AS count
        FROM embeddings e
        JOIN segments s ON e.segment_id = s.id
        JOIN collections c ON s.collection = c.id
        LEFT JOIN embedding_metadata em_model ON em_model.id = e.id AND em_model.key='embedding_model'
        GROUP BY model
        ORDER BY count DESC
        """
    )
    print("\nBy model:")
    for model, count in rows:
        print(f"- {model}: {count}")

    rows2 = q(
        """
        SELECT coalesce(em_model.string_value,'<missing>') AS model,
               coalesce(em_cat.string_value,'<uncategorized>') AS category,
               COUNT(DISTINCT e.id) AS count
        FROM embeddings e
        JOIN segments s ON e.segment_id = s.id
        JOIN collections c ON s.collection = c.id
        LEFT JOIN embedding_metadata em_model ON em_model.id = e.id AND em_model.key='embedding_model'
        LEFT JOIN embedding_metadata em_cat ON em_cat.id = e.id AND em_cat.key='category'
        GROUP BY model, category
        ORDER BY count DESC
        """
    )

    print("\nBy model x category (top 20):")
    for model, category, count in rows2[:20]:
        print(f"- {model} x {category}: {count}")

    rows_missing = [r for r in rows2 if r[0] == '<missing>']
    if rows_missing:
        print("\nMissing model metadata (top categories):")
        for _, category, count in rows_missing[:10]:
            print(f"- {category}: {count}")

    if write_report:
        out = Path("var/reports")
        out.mkdir(parents=True, exist_ok=True)
        md = out / "embedding_model_audit.md"
        with md.open("w", encoding="utf-8") as f:
            f.write("# Embedding Model Audit\n\n")
            f.write("## By Model\n")
            for model, count in rows:
                f.write(f"- {model}: {count}\n")
            f.write("\n## By Model x Category (top 50)\n")
            for model, category, count in rows2[:50]:
                f.write(f"- {model} x {category}: {count}\n")
            if rows_missing:
                f.write("\n## Missing Model Metadata (top categories)\n")
                for _, category, count in rows_missing[:20]:
                    f.write(f"- {category}: {count}\n")
        print(f"\nReport written: {md}")


@app.command("backfill-models")
def backfill_models(
    categories: str = typer.Option("knowledgebase,code", "--categories", help="Comma-separated categories to backfill"),
    model: str = typer.Option("voyage-context-3", "--model", help="Model name to write"),
    apply: bool = typer.Option(False, "--apply", help="Actually write changes; otherwise dry-run"),
):
    """Backfill missing embedding_model metadata for selected categories.

    NOTE: Defaults target knowledgebase and code which were produced with contextualized embeddings.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path("chroma_db/chroma.sqlite3")
    if not db_path.exists():
        print("DB not found at chroma_db/chroma.sqlite3")
        raise SystemExit(1)

    cats = [c.strip() for c in categories.split(",") if c.strip()]
    if not cats:
        print("No categories specified")
        raise SystemExit(2)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Count targets
    q = (
        "SELECT COUNT(DISTINCT e.id)\n"
        "FROM embeddings e\n"
        "LEFT JOIN embedding_metadata em_model ON em_model.id = e.id AND em_model.key='embedding_model'\n"
        "JOIN embedding_metadata em_cat ON em_cat.id = e.id AND em_cat.key='category'\n"
        f"WHERE em_model.id IS NULL AND em_cat.string_value IN ({','.join('?' for _ in cats)})"
    )
    cur.execute(q, cats)
    (missing_count,) = cur.fetchone()
    print(f"Missing embedding_model in categories {cats}: {missing_count}")

    if missing_count == 0:
        conn.close()
        return

    if not apply:
        print("Dry-run. Use --apply to write metadata.")
        conn.close()
        return

    print(f"Writing embedding_model='{model}' ...")
    cur.execute("BEGIN")
    ins = (
        "INSERT INTO embedding_metadata (id, key, string_value)\n"
        "SELECT DISTINCT e.id, 'embedding_model', ?\n"
        "FROM embeddings e\n"
        "LEFT JOIN embedding_metadata em_model ON em_model.id = e.id AND em_model.key='embedding_model'\n"
        "JOIN embedding_metadata em_cat ON em_cat.id = e.id AND em_cat.key='category'\n"
        f"WHERE em_model.id IS NULL AND em_cat.string_value IN ({','.join('?' for _ in cats)})"
    )
    cur.execute(ins, (model, *cats))
    conn.commit()
    print("Done.")
    conn.close()


def _cosine(a, b):
    import math
    if a is None or b is None:
        return 0.0
    try:
        la = len(a)
        lb = len(b)
    except Exception:
        return 0.0
    if la != lb:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        x = float(x)
        y = float(y)
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@app.command("infer-bookmark-models")
def infer_bookmark_models(
    limit: int = typer.Option(100, "--limit", help="Number of bookmarks to sample (0=all)"),
    threshold: float = typer.Option(0.02, "--threshold", help="Similarity margin for decision"),
    apply: bool = typer.Option(False, "--apply", help="Write inferred embedding_model back to DB for confident items"),
):
    """Infer whether bookmark embeddings are contextualized or standard.

    Approach: for each bookmark, reconstruct the chunk list (summary + highlights),
    then re-embed:
      - Standard: per-chunk via `embed(model=voyage-3)`
      - Contextualized: windows including the summary via `contextualized_embed(model=VOYAGE_CONTEXT_MODEL)`
    Compare cosine similarity of stored vectors to both re-embeddings and decide.
    """
    import os
    import sqlite3
    from voyage_embeddings.embed_utils import contextual_windows, CONTEXT_MODEL, EMBED_DIM

    vo, client = get_clients()
    col = ensure_collection(client)

    # Fetch a sample of bookmark IDs via SQL for efficiency
    db_path = Path("chroma_db/chroma.sqlite3")
    if not db_path.exists():
        print("DB not found at chroma_db/chroma.sqlite3")
        raise SystemExit(1)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT substr(em_src.string_value, length('raindrop://')+1) AS bid
        FROM embedding_metadata em_cat
        JOIN embedding_metadata em_src ON em_src.id = em_cat.id AND em_src.key='source' AND em_src.string_value LIKE 'raindrop://%'
        WHERE em_cat.key='category' AND em_cat.string_value='bookmarks'
        LIMIT ?
        """,
        (limit if limit and limit > 0 else 1000000,),
    )
    bids = [r[0] for r in cur.fetchall()]
    conn.close()
    if not bids:
        print("No bookmarks found.")
        return

    print(f"Sampling {len(bids)} bookmarks...")

    decided = {"voyage-context-3": 0, "voyage-3": 0, "uncertain": 0}
    per_bookmark_decision: dict[str, str] = {}
    write_ids: list[tuple[str, str]] = []  # (embedding_id, model)

    for bi, bid in enumerate(bids, 1):
        where = {"source": f"raindrop://{bid}"}
        got = col.get(where=where, include=["ids", "documents", "metadatas", "embeddings"])
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        vecs = got.get("embeddings") or []
        if not ids:
            continue
        # Order: summary first, then highlights by (highlight_index, part_index)
        items = []
        for i in range(len(ids)):
            m = metas[i] or {}
            ft = m.get("file_type", "")
            if ft == "bookmark_summary":
                key = (0, 0, 0)
            else:
                hi = int(m.get("highlight_index", 0) or 0)
                pi = int(m.get("part_index", 0) or 0)
                key = (1, hi, pi)
            items.append((key, ids[i], docs[i] or "", metas[i] or {}, vecs[i]))
        items.sort(key=lambda x: x[0])
        chunk_ids = [it[1] for it in items]
        chunk_texts = [it[2] for it in items]
        stored_vecs = [it[4] for it in items]
        if not chunk_texts or not stored_vecs:
            continue

        # Standard re-embed per chunk
        try:
            emb_std = vo.embed(chunk_texts, model=os.environ.get("VOYAGE_EMBED_MODEL", "voyage-3"), input_type="document")
            std_vecs = emb_std.embeddings
        except Exception as e:
            print(f"  ⚠️ embed(voyage-3) failed for {bid}: {e}")
            std_vecs = None

        # Contextualized re-embed via windows including summary
        index_to_ctx: dict[int, list[float]] = {}
        try:
            windows = contextual_windows(chunk_texts, always_include_first=True)
            for (s, e) in windows:
                ctx = vo.contextualized_embed(inputs=[chunk_texts[s:e]], model=os.environ.get("VOYAGE_CONTEXT_MODEL", CONTEXT_MODEL), input_type="document", output_dimension=EMBED_DIM)
                vecs_ctx = ctx.results[0].embeddings if ctx.results else []
                for i, v in enumerate(vecs_ctx):
                    idx = s + i
                    # only set if not already assigned (first window wins)
                    if idx not in index_to_ctx:
                        index_to_ctx[idx] = v
        except Exception as e:
            print(f"  ⚠️ contextualized_embed failed for {bid}: {e}")
            index_to_ctx = {}

        # Compare cosines per chunk
        deltas = []
        per_id_assignment: list[tuple[str, str]] = []
        for i in range(len(stored_vecs)):
            sv = stored_vecs[i]
            s_std = _cosine(sv, std_vecs[i]) if std_vecs else 0.0
            s_ctx = _cosine(sv, index_to_ctx.get(i)) if index_to_ctx else 0.0
            deltas.append(s_ctx - s_std)
            if abs(s_ctx - s_std) >= threshold:
                per_id_assignment.append((chunk_ids[i], "voyage-context-3" if s_ctx > s_std else "voyage-3"))

        # Decide per-bookmark by median delta
        if deltas:
            d_sorted = sorted(deltas)
            med = d_sorted[len(d_sorted)//2]
            if med >= threshold:
                decided["voyage-context-3"] += 1
                per_bookmark_decision[bid] = "voyage-context-3"
                if apply:
                    write_ids.extend((cid, "voyage-context-3") for cid, m in per_id_assignment)
            elif med <= -threshold:
                decided["voyage-3"] += 1
                per_bookmark_decision[bid] = "voyage-3"
                if apply:
                    write_ids.extend((cid, "voyage-3") for cid, m in per_id_assignment)
            else:
                decided["uncertain"] += 1
                per_bookmark_decision[bid] = "uncertain"
        else:
            decided["uncertain"] += 1
            per_bookmark_decision[bid] = "uncertain"

        if bi % 25 == 0:
            print(f"  Processed {bi}/{len(bids)}...")

    print("\nDecisions:")
    for k, v in decided.items():
        print(f"- {k}: {v}")

    if apply and write_ids:
        # write embedding_model for confident per-chunk assignments using SQL (avoids overwriting other metadata)
        conn = sqlite3.connect("chroma_db/chroma.sqlite3")
        cur = conn.cursor()
        cur.execute("BEGIN")
        # batch by 500
        B = 500
        for i in range(0, len(write_ids), B):
            batch = write_ids[i:i+B]
            ids_only = [cid for cid, _ in batch]
            # build temp table for mapping embedding_id -> model
            cur.execute("CREATE TEMP TABLE IF NOT EXISTS t_assign (embedding_id TEXT PRIMARY KEY, model TEXT)")
            cur.execute("DELETE FROM t_assign")
            cur.executemany("INSERT INTO t_assign (embedding_id, model) VALUES (?, ?)", batch)
            cur.execute(
                """
                INSERT INTO embedding_metadata (id, key, string_value)
                SELECT e.id, 'embedding_model', t.model
                FROM embeddings e
                JOIN t_assign t ON t.embedding_id = e.embedding_id
                LEFT JOIN embedding_metadata em ON em.id=e.id AND em.key='embedding_model'
                WHERE em.id IS NULL
                """
            )
            conn.commit()
        conn.close()
        print(f"Applied embedding_model to {len(write_ids)} vectors (confident per-chunk assignments).")


@app.command("infer-category-models")
def infer_category_models(
    categories: str = typer.Option("bookmarks,knowledgebase,code,reddit,irc", "--categories", help="Comma-separated categories to process"),
    sample_size: int = typer.Option(200, "--sample-size", help="Units to sample per category (bookmarks: bookmarks; kb/code: groups; others: chunks)"),
    threshold: float = typer.Option(0.02, "--threshold", help="Similarity margin for decision per unit"),
    apply_if_consistent: bool = typer.Option(False, "--apply-if-consistent", help="If all sampled units agree, backfill category with inferred model"),
):
    """Spot-check categories and auto-assign model if samples are unanimous.

    - bookmarks: samples per-bookmark (summary + highlights)
    - knowledgebase/code: samples per (source, group_index) group
    - reddit/irc: samples per embedding (single-chunk heuristic)
    """
    import sqlite3
    import os
    from collections import defaultdict

    vo, client = get_clients()
    col = ensure_collection(client)

    cats = [c.strip() for c in categories.split(',') if c.strip()]
    results = {}

    def decide_unit(std_vecs, ctx_vecs, stored_vecs):
        deltas = []
        for i in range(len(stored_vecs)):
            s_std = _cosine(stored_vecs[i], std_vecs[i]) if std_vecs else 0.0
            s_ctx = _cosine(stored_vecs[i], ctx_vecs[i]) if ctx_vecs and i < len(ctx_vecs) else 0.0
            deltas.append(s_ctx - s_std)
        if not deltas:
            return 'uncertain', 0.0
        d_sorted = sorted(deltas)
        med = d_sorted[len(d_sorted)//2]
        if med >= threshold:
            return 'voyage-context-3', med
        if med <= -threshold:
            return 'voyage-3', med
        return 'uncertain', med

    db_path = Path("chroma_db/chroma.sqlite3")
    if not db_path.exists():
        print("DB not found at chroma_db/chroma.sqlite3")
        raise SystemExit(1)

    for cat in cats:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        units = []
        if cat == 'bookmarks':
            cur.execute(
                """
                SELECT DISTINCT substr(em_src.string_value, length('raindrop://')+1) AS bid
                FROM embedding_metadata em_cat
                JOIN embedding_metadata em_src ON em_src.id = em_cat.id AND em_src.key='source' AND em_src.string_value LIKE 'raindrop://%'
                WHERE em_cat.key='category' AND em_cat.string_value='bookmarks'
                LIMIT ?
                """,
                (sample_size,),
            )
            units = [('bookmark', r[0]) for r in cur.fetchall()]
        elif cat in ('knowledgebase', 'code'):
            cur.execute(
                """
                SELECT em_src.string_value AS source, em_gi.int_value AS gidx
                FROM embedding_metadata em_cat
                JOIN embedding_metadata em_src ON em_src.id=em_cat.id AND em_src.key='source'
                JOIN embedding_metadata em_gi ON em_gi.id=em_cat.id AND em_gi.key='group_index'
                WHERE em_cat.key='category' AND em_cat.string_value=?
                GROUP BY source, gidx
                LIMIT ?
                """,
                (cat, sample_size),
            )
            units = [('kbcode_group', (r[0], int(r[1]))) for r in cur.fetchall()]
        else:
            cur.execute(
                """
                SELECT e.embedding_id
                FROM embeddings e
                JOIN embedding_metadata em_cat ON em_cat.id=e.id AND em_cat.key='category' AND em_cat.string_value=?
                LIMIT ?
                """,
                (cat, sample_size),
            )
            units = [('single', r[0]) for r in cur.fetchall()]
        conn.close()

        if not units:
            results[cat] = {'voyage-context-3':0,'voyage-3':0,'uncertain':0,'decision': 'no_data'}
            continue

        counts = defaultdict(int)
        assignments = []

        for kind, u in units:
            if kind == 'bookmark':
                where = {"source": f"raindrop://{u}"}
                got = col.get(where=where, include=["documents","embeddings","metadatas"]) 
                docs = got.get('documents') or []
                metas = got.get('metadatas') or []
                if not docs:
                    counts['uncertain'] += 1
                    continue
                # order: summary first then highlights
                items = []
                for i in range(len(docs)):
                    m = metas[i] or {}
                    ft = m.get('file_type','')
                    if ft=='bookmark_summary':
                        key=(0,0,0)
                    else:
                        hi=int(m.get('highlight_index',0) or 0)
                        pi=int(m.get('part_index',0) or 0)
                        key=(1,hi,pi)
                    items.append((key, docs[i], got['embeddings'][i]))
                items.sort(key=lambda x: x[0])
                cd = [it[1] for it in items]
                sv = [it[2] for it in items]
                # standard
                try:
                    std_vecs = vo.embed(cd, model=os.environ.get('VOYAGE_EMBED_MODEL','voyage-3'), input_type='document').embeddings
                except Exception:
                    std_vecs = None
                # contextual windows
                from voyage_embeddings.embed_utils import contextual_windows, CONTEXT_MODEL, EMBED_DIM
                idx_to_ctx={}
                try:
                    for (s,e) in contextual_windows(cd, always_include_first=True):
                        ctx = vo.contextualized_embed(inputs=[cd[s:e]], model=os.environ.get('VOYAGE_CONTEXT_MODEL', CONTEXT_MODEL), input_type='document', output_dimension=EMBED_DIM)
                        vecs = ctx.results[0].embeddings if ctx.results else []
                        for i,v in enumerate(vecs):
                            idx=s+i
                            if idx not in idx_to_ctx:
                                idx_to_ctx[idx]=v
                except Exception:
                    pass
                ctx_vecs=[idx_to_ctx.get(i) for i in range(len(cd))]
                decision, _ = decide_unit(std_vecs, ctx_vecs, sv)
                counts[decision]+=1
                assignments.append(decision)
            elif kind == 'kbcode_group':
                source, gidx = u
                got = col.get(where={"source": source, "group_index": gidx}, include=["documents","embeddings"]) 
                docs = got.get('documents') or []
                sv = got.get('embeddings') or []
                if not docs:
                    counts['uncertain']+=1
                    continue
                try:
                    std_vecs = vo.embed(docs, model=os.environ.get('VOYAGE_EMBED_MODEL','voyage-3'), input_type='document').embeddings
                except Exception:
                    std_vecs = None
                try:
                    from voyage_embeddings.embed_utils import CONTEXT_MODEL, EMBED_DIM
                    ctx = vo.contextualized_embed(inputs=[docs], model=os.environ.get('VOYAGE_CONTEXT_MODEL', CONTEXT_MODEL), input_type='document', output_dimension=EMBED_DIM)
                    ctx_vecs = ctx.results[0].embeddings if ctx.results else []
                except Exception:
                    ctx_vecs=None
                decision,_=decide_unit(std_vecs, ctx_vecs, sv)
                counts[decision]+=1
                assignments.append(decision)
            else:
                # single-chunk heuristic for reddit/irc
                eid = u
                got = col.get(ids=[eid], include=["documents","embeddings"]) 
                if not got.get('ids'):
                    counts['uncertain']+=1
                    continue
                doc = got['documents'][0]
                sv = [got['embeddings'][0]]
                try:
                    std_vecs = vo.embed([doc], model=os.environ.get('VOYAGE_EMBED_MODEL','voyage-3'), input_type='document').embeddings
                except Exception:
                    std_vecs=None
                try:
                    from voyage_embeddings.embed_utils import CONTEXT_MODEL, EMBED_DIM
                    ctx = vo.contextualized_embed(inputs=[[doc]], model=os.environ.get('VOYAGE_CONTEXT_MODEL', CONTEXT_MODEL), input_type='document', output_dimension=EMBED_DIM)
                    ctx_vecs = [ctx.results[0].embeddings[0]] if ctx.results else None
                except Exception:
                    ctx_vecs=None
                decision,_=decide_unit(std_vecs, ctx_vecs, sv)
                counts[decision]+=1
                assignments.append(decision)

        unanimous = [k for k in ('voyage-context-3','voyage-3') if counts[k]>0 and counts['uncertain']==0 and counts[k]==sum(counts.values())]
        decision = unanimous[0] if unanimous else 'inconclusive'
        results[cat] = {**counts, 'decision': decision}

        print(f"\nCategory {cat} sample results:")
        for k in ('voyage-context-3','voyage-3','uncertain'):
            print(f"- {k}: {counts[k]}")
        print(f"Decision: {decision}")

        if apply_if_consistent and decision in ('voyage-context-3','voyage-3'):
            print(f"Applying backfill for category '{cat}' with model={decision} ...")
            # Write for all items in category missing embedding_model
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("BEGIN")
            cur.execute(
                """
                INSERT INTO embedding_metadata (id, key, string_value)
                SELECT e.id, 'embedding_model', ?
                FROM embeddings e
                JOIN embedding_metadata em_cat ON em_cat.id=e.id AND em_cat.key='category' AND em_cat.string_value=?
                LEFT JOIN embedding_metadata em ON em.id=e.id AND em.key='embedding_model'
                WHERE em.id IS NULL
                """,
                (decision, cat),
            )
            conn.commit()
            conn.close()
            print("Backfill applied.")


@app.command("embed-bookmarks")
def embed_bookmarks(csv: Path = typer.Option(Path("data/raindrop/raindrop_bookmarks_19_08_2025.csv"), "--csv"), limit: int = 0):
    processed, embedded, tokens = embed_bookmarks_csv(csv_path=csv, limit=limit)
    print("\nDone:")
    print(f" processed={processed} embedded={embedded} tokens={tokens}")


@app.command("inspect-db")
def inspect_db():
    import sqlite3
    db_path = Path("chroma_db/chroma.sqlite3")
    if not db_path.exists():
        print("DB not found")
        raise SystemExit(1)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT name FROM collections")
    print("Collections:", cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM embeddings")
    print("Embeddings:", cur.fetchone()[0])
    conn.close()
