from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from molecule.config import load_config
from molecule.core.eval import run_eval
from molecule.core.extractor import extract_memories
from molecule.core.feedback import feedback_stats, record_feedback
from molecule.core.ingest import do_ingest
from molecule.core.labeler import default_llm_fn
from molecule.core.memory import expire_stale, merge_duplicates
from molecule.core.retriever import rank_candidates
from molecule.core.telemetry import get_stats
from molecule.db import get_connection, init_schema


def _get_conn():
    cfg = load_config()
    conn = get_connection(cfg.db_path)
    init_schema(conn)
    expire_stale(conn)
    return conn, cfg


def _force_utf8_stdio() -> None:
    """Windows consoles default to cp1252, which can't print ✓/✗/▲.
    Reconfigure stdout/stderr to UTF-8 so CLI output never crashes."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@click.group()
def cli() -> None:
    """Molecule — local-first context router and memory for AI agents."""


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Re-embed even unchanged sections.")
def ingest(file_path: str, force: bool) -> None:
    """Ingest a markdown file into the chunk index."""
    conn, cfg = _get_conn()
    result = do_ingest(Path(file_path).resolve(), conn, cfg, force)
    if "error" in result:
        click.echo(f"✗ {result['error']}", err=True)
        raise SystemExit(1)
    click.echo(
        f"✓ {Path(result['file']).name}: "
        f"{result['new']} new, {result['updated']} updated, {result['skipped']} skipped "
        f"({result['total']} total)"
    )
    timing = result.get("timing", {})
    if timing.get("embed_s") or timing.get("model_load_s"):
        click.echo(
            f"  timing: model load {timing['model_load_s']}s, embed {timing['embed_s']}s"
        )


@cli.command()
@click.argument("query")
@click.option("-k", default=5, show_default=True, help="Number of results.")
def search(query: str, k: int) -> None:
    """Search the chunk index and memories."""
    conn, cfg = _get_conn()
    results = rank_candidates(query, conn, k, cfg)
    if not results:
        click.echo("No results.")
        return
    for r in results:
        click.echo(f"\n[{r['type']}] {r['id']}")
        click.echo(f"  {r['summary']}")
        click.echo(f"  reason: {r['reason']}  score: {r['score']:.3f}")


@cli.command()
def sources() -> None:
    """List all ingested source files."""
    conn, _ = _get_conn()
    rows = conn.execute(
        "SELECT source_file, COUNT(*) as chunks, SUM(token_count) as total_tokens "
        "FROM chunks GROUP BY source_file"
    ).fetchall()
    if not rows:
        click.echo("No sources ingested yet. Run: molecule ingest <file>")
        return
    for r in rows:
        click.echo(f"{r['source_file']}  ({r['chunks']} chunks, {r['total_tokens']} tokens)")


@cli.command()
@click.argument("source_file")
def remove(source_file: str) -> None:
    """Remove all chunks for a source file from the index."""
    conn, _ = _get_conn()
    rows = conn.execute(
        "SELECT COUNT(*) as n FROM chunks WHERE source_file=?", (source_file,)
    ).fetchone()
    if not rows or rows["n"] == 0:
        click.echo(f"✗ No chunks found for: {source_file}", err=True)
        raise SystemExit(1)
    conn.execute("DELETE FROM chunks WHERE source_file=?", (source_file,))
    conn.commit()
    click.echo(f"✓ Removed {rows['n']} chunks for {source_file}")


@cli.command()
def stats() -> None:
    """Show token savings for the current session."""
    conn, _ = _get_conn()
    s = get_stats(conn)
    click.echo(f"Tokens used:     {s['tokens_used_this_session']}")
    click.echo(f"Tokens baseline: {s['tokens_if_full_file_loaded']}")
    click.echo(f"Savings:         {s['savings_pct']}%")


@cli.command()
@click.argument("eval_file", type=click.Path(exists=True))
@click.option("-k", default=3, show_default=True)
def eval(eval_file: str, k: int) -> None:
    """Run retrieval eval from a JSON file of {query, expected_chunk_id} pairs."""
    conn, cfg = _get_conn()
    data = json.loads(Path(eval_file).read_text(encoding="utf-8"))
    pairs = [(q["query"], q["expected_chunk_id"]) for q in data]
    result = run_eval(pairs, conn, cfg, k)
    click.echo(f"Top-{k} precision: {result[f'top{k}_precision']} ({result['hits']}/{result['total']})")
    click.echo(f"Target 80%: {'✓ PASS' if result['passes'] else '✗ FAIL'}")
    if result["misses"]:
        click.echo("\nMisses:")
        for m in result["misses"]:
            click.echo(f"  Q: {m['query']}")
            click.echo(f"  Expected: {m['expected']}")
            click.echo(f"  Got:      {m['got']}")


@cli.command()
def health() -> None:
    """Check DB status and indexed content."""
    conn, cfg = _get_conn()
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    memories = conn.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
    click.echo("Status:   ok")
    click.echo(f"DB:       {cfg.db_path}")
    click.echo(f"Chunks:   {chunks}")
    click.echo(f"Memories: {memories}")
    click.echo(f"Model:    {cfg.embedding_model}")


@cli.command()
def merge() -> None:
    """Merge near-duplicate memories into single clean facts."""
    conn, cfg = _get_conn()
    n = merge_duplicates(conn, cfg, default_llm_fn)
    click.echo(f"✓ Merged {n} duplicate group(s)")


@cli.command()
def setup() -> None:
    """One-time setup: download and cache the embedding model (~80MB)."""
    import time
    cfg = load_config()
    click.echo(f"Downloading/loading embedding model '{cfg.embedding_model}'…")
    t0 = time.perf_counter()
    from molecule.core.embedder import embed
    embed(["warm up"], cfg.embedding_model)
    click.echo(f"✓ Model ready and cached in {time.perf_counter() - t0:.1f}s")
    click.echo("  Future ingests will skip the download entirely.")


@cli.command()
@click.argument("chunk_id")
@click.option("--useful/--not-useful", default=True, help="Was the chunk useful?")
@click.option("--query", default="", help="The query this feedback is about.")
def feedback(chunk_id: str, useful: bool, query: str) -> None:
    """Teach the router: mark a chunk as useful or not for a query."""
    conn, cfg = _get_conn()
    result = record_feedback(chunk_id, useful, query, conn, cfg)
    if "error" in result:
        click.echo(f"✗ {result['error']}", err=True)
        raise SystemExit(1)
    sign = "▲" if result["delta"] >= 0 else "▼"
    click.echo(
        f"✓ {chunk_id}: learned weight {result['learned_weight']} "
        f"({sign} {result['delta']:+.4f})"
    )


@cli.command("feedback-stats")
def feedback_stats_cmd() -> None:
    """Show what the router has learned from feedback."""
    conn, _ = _get_conn()
    s = feedback_stats(conn)
    click.echo(f"Votes: {s['votes']} ({s['useful']} useful, {s['not_useful']} not)")
    if s["learned"]:
        click.echo("Learned weights:")
        for entry in s["learned"]:
            click.echo(f"  {entry['learned_weight']:.4f}  {entry['chunk_id']}")
    else:
        click.echo("No learned weights yet — use 'molecule feedback <chunk_id>'.")


@cli.command()
@click.argument("transcript_file", type=click.Path(exists=True))
@click.option("--task", default="session", help="Source task label for stored facts.")
def extract(transcript_file: str, task: str) -> None:
    """Extract durable facts from a transcript file and store them as memories."""
    conn, cfg = _get_conn()
    text = Path(transcript_file).read_text(encoding="utf-8")
    result = extract_memories(text, task, conn, cfg)
    click.echo(
        f"✓ Extracted {result['extracted']} fact(s): "
        f"{result['stored']} stored, {result['skipped']} skipped"
    )
    for f in result["facts"]:
        click.echo(f"  [{f['outcome']}] ({f['confidence']}, {f['type']}) {f['text']}")


@cli.command("serve")
def serve() -> None:
    """Start the MCP server."""
    from molecule.server import mcp
    mcp.run()


def main() -> None:
    _force_utf8_stdio()
    cli()
