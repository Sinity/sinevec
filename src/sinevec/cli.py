from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import typer

from sinevec.embed_utils import (
    UNIFIED,
    ensure_collection,
    get_clients,
    get_qdrant_http_client,
)
from sinevec.ingest.bookmarks import embed_bookmarks_csv
from sinevec.ingest.chats import embed_chats_pipeline
from sinevec.ingest.knowledge_code import embed_knowledge_code_pipeline
from sinevec.search_core import SearchError, run_search

app = typer.Typer(help="Sinevec CLI")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Show the top-level help when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


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
    details: bool = typer.Option(True, "--details/--no-details", help="Show per-result metadata cards"),
    reverse: bool = typer.Option(False, "--reverse", help="List farthest matches first"),
):
    try:
        raw_results = run_search(
            query,
            n=n,
            model=model,
            category=category,
            subcategory=subcategory,
            channel=channel,
            date_from=date_from,
            date_to=date_to,
            has_code=has_code,
            has_urls=has_urls,
            reverse=reverse,
        )
    except SearchError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    if not raw_results:
        typer.echo("No results found. You may need to embed more content or adjust filters.")
        raise typer.Exit(0)

    def _shorten_for_table(value: str, width: int = 48) -> str:
        if not value:
            return ""
        clean = value.replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")
        return textwrap.shorten(clean, width=width, placeholder="…")

    results: list[dict[str, Any]] = []
    for row in raw_results:
        score_val = row.get("score")
        score_str = f"{float(score_val):.4f}" if isinstance(score_val, (int, float)) else "n/a"

        title = row.get("title") or row.get("id")
        url = row.get("url") or ""
        source = row.get("source") or ""
        category_value = row.get("category") or ""

        title_short = _shorten_for_table(title)
        title_cell = f"[{title_short}]({url})" if url else title_short

        results.append(
            {
                "index": row.get("index", len(results) + 1),
                "id": row.get("id"),
                "score": score_str,
                "title": title,
                "title_cell": title_cell,
                "snippet": row.get("snippet") or "",
                "category": category_value,
                "subcategory": row.get("subcategory") or "",
                "source": source,
                "url": url,
                "meta": row.get("meta") or {},
            }
        )

    use_gum = details and shutil.which("gum")
    if use_gum:
        table_rows = [
            (
                f"{r['index']:3d}",
                r["score"],
                r["title_cell"],
                r["category"],
                r["source"],
                r["url"],
            )
            for r in results
        ]
        header = ["#", "Score", "Title", "Category", "Source", "URL"]
        table_text = "\n".join(["  |  ".join(header)] + ["  |  ".join(row) for row in table_rows])
        proc = subprocess.run(["gum", "table"], input=table_text.encode("utf-8"))
        if proc.returncode != 0:
            use_gum = False

    if not use_gum:
        for r in results:
            header_line = f"[{r['index']}] {r['id']} :: {r['score']}"
            if r["title"]:
                header_line += f" :: {r['title']}"
            typer.echo(f"\n{header_line}")
            if r["snippet"]:
                typer.echo(r["snippet"])

    if details and use_gum:
        meta_fields = [
            "category",
            "subcategory",
            "source",
            "url",
            "file_type",
            "channel",
            "created",
            "domain",
            "tags",
            "embedding_model",
        ]
        for row in results:
            meta = row["meta"]
            info_lines = []
            for key in meta_fields:
                value = meta.get(key)
                if value not in (None, ""):
                    info_lines.append(f"- **{key}**: {value}")
            title_line = f"### [{row['index']}] {row['title']}"
            if row["url"]:
                title_line = f"### [{row['index']}] [{row['title']}]({row['url']})"

            detail_lines = [title_line, f"- **Score:** `{row['score']}`"]
            if row["url"]:
                detail_lines.append(f"- **URL:** {row['url']}")
            if info_lines:
                detail_lines.extend(info_lines)
            if row["snippet"]:
                detail_lines.append("")
                detail_lines.append(f"> {row['snippet']}")
            subprocess.run(["gum", "format"], input="\n".join(detail_lines).encode("utf-8"))


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", help="Listen address"),
    port: int = typer.Option(8000, help="Listen port"),
    reload: bool = typer.Option(False, "--reload/--no-reload", help="Enable autoreload in development"),
):
    """Launch the FastAPI web interface for exploring embeddings."""
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        typer.echo("uvicorn is required for 'sinevec serve'. Install fastapi extras first.")
        raise typer.Exit(1) from exc

    from sinevec.server import app as fastapi_app

    uvicorn.run(fastapi_app, host=host, port=port, reload=reload, log_level="info")


@app.command("embed-chats")
def embed_chats(platform: str = typer.Option("all", "--platform"), limit: int = typer.Option(0, "--limit")):
    processed, embedded, tokens = embed_chats_pipeline(platform=platform, limit=limit)
    typer.echo("\nDone:")
    typer.echo(f" conversations={processed} messages_embedded={embedded} tokens={tokens}")


@app.command("embed-knowledge")
def embed_knowledge(
    kb_dir: Path = typer.Option(Path("data/knowledgebase"), "--kb-dir"),
    code_dir: Path = typer.Option(Path("data/code"), "--code-dir"),
    force: bool = typer.Option(False, "--force"),
):
    processed, tokens = embed_knowledge_code_pipeline(kb_dir=kb_dir, code_dir=code_dir, force=force)
    typer.echo("\nDone:")
    typer.echo(f" files_processed={processed} tokens={tokens}")


@app.command("embed-bookmarks")
def embed_bookmarks(csv: Path = typer.Option(Path("data/raindrop/raindrop_bookmarks_19_08_2025.csv"), "--csv"), limit: int = 0):
    processed, embedded, tokens = embed_bookmarks_csv(csv_path=csv, limit=limit)
    typer.echo("\nDone:")
    typer.echo(f" processed={processed} embedded={embedded} tokens={tokens}")


@app.command("inspect-db")
def inspect_db(collection: str | None = typer.Option(None, "--collection", help="Collection to inspect (default: all)")):
    """Show summary information about the Qdrant vector store."""
    client = get_qdrant_http_client()
    if client is None:
        typer.echo("qdrant-client is not installed; unable to inspect the vector store.")
        raise typer.Exit(1)

    try:
        collections = client.get_collections().collections or []
    except Exception as exc:  # pragma: no cover - runtime failure
        typer.echo(f"Failed to query Qdrant: {exc}")
        raise typer.Exit(1) from exc

    if collection:
        collections = [c for c in collections if c.name == collection]
        if not collections:
            typer.echo(f"Collection '{collection}' not found.")
            raise typer.Exit(1)

    if not collections:
        typer.echo("No collections present.")
        raise typer.Exit(0)

    for info in collections:
        name = info.name
        typer.echo(f"\nCollection: {name}")
        try:
            count = client.count(collection_name=name, exact=True).count
        except Exception as exc:
            typer.echo(f"  count: error ({exc})")
        else:
            typer.echo(f"  count: {count}")

        try:
            full = client.get_collection(name)
        except Exception:
            full = None
        vectors = getattr(getattr(getattr(full, "config", None), "params", None), "vectors", None)
        if vectors and getattr(vectors, "size", None):
            typer.echo(f"  vector_size: {vectors.size}")

        try:
            status = client.get_collection(name)
            optimizer = getattr(getattr(status, "status", None), "optimizer_status", None)
            if optimizer:
                typer.echo(f"  optimizer: {optimizer}")
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover
    app()
