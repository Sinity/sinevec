from __future__ import annotations

import os
from pathlib import Path
import typer

from voyage_embeddings.embed_utils import get_clients, ensure_collection
from voyage_embeddings.ingest.bookmarks import embed_bookmarks_csv
from voyage_embeddings.ingest.chats import embed_conversation_messages

app = typer.Typer(help="Voyage Embeddings CLI")


@app.command("search")
def search_cmd(
    query: str,
    n: int = typer.Option(10, "--n"),
    model: str | None = typer.Option(None, "--model", help="Override query model; defaults to VOYAGE_QUERY_MODEL or auto")
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

    try:
        res = col.query(query_embeddings=[qv], n_results=n)
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
