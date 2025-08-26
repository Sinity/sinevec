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
