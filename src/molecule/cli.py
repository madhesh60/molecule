"""cli.py — the `molecule` command. A thin shell over the engine.

Every command builds one `Molecule` for the current directory (or
`$MOLECULE_ROOT`) and prints what the matching engine method returns —
human-readable by default, raw JSON with `--json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import click

from molecule import Molecule, __version__


def _mol() -> Molecule:
    return Molecule()


def _force_utf8_stdio() -> None:
    """Windows consoles default to cp1252, which can't print ✓/✗/▲.
    Reconfigure stdout/stderr to UTF-8 so CLI output never crashes."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _print_json(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))


def _fail(message: str) -> None:
    click.echo(f"✗ {message}", err=True)
    raise SystemExit(1)


json_option = click.option("--json", "as_json", is_flag=True, help="Print raw JSON.")


@click.group()
@click.version_option(__version__, prog_name="molecule")
def cli() -> None:
    """Molecule — local-first context router and memory for AI agents."""


# --- Indexing ----------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Re-embed even unchanged sections.")
@json_option
def ingest(path: str, force: bool, as_json: bool) -> None:
    """Index a markdown file, or every markdown file under a directory."""
    result = _mol().ingest(Path(path).resolve(), force)
    if as_json:
        return _print_json(result)
    if "error" in result:
        _fail(result["error"])
    label = f"{result['files']} files" if "files" in result else Path(result["file"]).name
    click.echo(
        f"✓ {label}: {result['new']} new, {result['updated']} updated, "
        f"{result['skipped']} skipped, {result['removed']} removed ({result['total']} total)"
    )
    timing = result.get("timing", {})
    if any(timing.values()):
        click.echo(
            f"  timing: model load {timing['model_load_s']}s, "
            f"chunk vectors {timing['embed_s']}s, passage vectors {timing['passages_s']}s"
        )
    for error in result.get("errors", []):
        click.echo(f"  ! {error}", err=True)


@cli.command()
@click.option("--prune", is_flag=True, help="Also drop chunks of deleted files.")
@click.option("--watch", type=float, default=0.0, metavar="SECONDS",
              help="Keep running and re-check every SECONDS.")
def sync(prune: bool, watch: float) -> None:
    """Re-index source files that changed on disk."""
    mol = _mol()
    while True:
        report = mol.sync(prune)
        for item in report["reingested"]:
            click.echo(f"✓ {item['source']}: {item['new']} new, {item['updated']} updated, "
                       f"{item['removed']} removed")
        for source in report["missing"]:
            click.echo(f"{'✓ pruned' if prune else '! missing'}: {source}")
        for error in report["errors"]:
            click.echo(f"✗ {error}", err=True)
        if not watch:
            click.echo(f"{report['fresh']} source(s) already fresh.")
            return
        time.sleep(watch)


@cli.command()
def sources() -> None:
    """List all ingested source files."""
    rows = _mol().sources()
    if not rows:
        click.echo("No sources ingested yet. Run: molecule ingest <file>")
        return
    for r in rows:
        click.echo(f"{r['source_file']}  ({r['chunks']} chunks, {r['total_tokens']} tokens)")


@cli.command()
@click.argument("source_file")
def remove(source_file: str) -> None:
    """Remove all chunks for a source file from the index."""
    result = _mol().remove_source(source_file)
    if "error" in result:
        _fail(result["error"])
    click.echo(f"✓ Removed {result['chunks_deleted']} chunks for {result['removed']}")


# --- Retrieval -----------------------------------------------------------------------

@cli.command()
@click.option("--max-tokens", default=0, help="Fit the index into this many tokens.")
@click.option("--prefix", default="", help="Only chunk ids starting with this.")
def index(max_tokens: int, prefix: str) -> None:
    """Print the router index (table of contents)."""
    _print_json(_mol().index(max_tokens or None, prefix or None))


@cli.command()
@click.argument("query")
@click.option("-k", default=5, show_default=True, help="Number of results.")
@json_option
def search(query: str, k: int, as_json: bool) -> None:
    """Search the chunk index and memories."""
    results = _mol().search(query, k)
    if as_json:
        return _print_json(results)
    if not results:
        click.echo("No results.")
        return
    for r in results:
        click.echo(f"\n[{r['type']}] {r['id']}")
        click.echo(f"  {r['summary']}")
        click.echo(f"  reason: {r['reason']}  score: {r['score']:.3f}  "
                   f"relevance: {r['relevance']}")
        for related in r.get("related", []):
            click.echo(f"  ↳ related ({related['bond']}, {related['strength']}): {related['id']}")


@cli.command()
@click.argument("query")
@click.option("--budget", default=0, help="Token budget (default from config).")
@json_option
def context(query: str, budget: int, as_json: bool) -> None:
    """Compile ready-to-paste context for a query within a token budget."""
    result = _mol().context(query, budget or None)
    if as_json:
        return _print_json(result)
    if "error" in result:
        _fail(result["error"])
    click.echo(result["context"] or "(nothing relevant)")
    click.echo(
        f"\n— {result['tokens_used']}/{result['budget']} tokens · "
        f"confidence: {result['confidence']} · {len(result['items'])} item(s)"
        + (f" · {len(result['omitted'])} omitted" if result["omitted"] else ""),
        err=True,
    )


@cli.command()
@click.argument("chunk_id")
def bonds(chunk_id: str) -> None:
    """Show the chunks bonded to a chunk."""
    related = _mol().bonds(chunk_id)
    if not related:
        click.echo("No bonds.")
        return
    for r in related:
        click.echo(f"{r['strength']:.2f}  {r['bond']:<13} {r['id']}")


# --- Learning ----------------------------------------------------------------------

@cli.command()
@click.argument("chunk_id")
@click.option("--useful/--not-useful", default=True, help="Was the chunk useful?")
@click.option("--query", default="", help="The query this feedback is about.")
def feedback(chunk_id: str, useful: bool, query: str) -> None:
    """Teach the router: mark a chunk as useful or not for a query."""
    result = _mol().feedback(chunk_id, useful, query)
    if "error" in result:
        _fail(result["error"])
    sign = "▲" if result["delta"] >= 0 else "▼"
    click.echo(
        f"✓ {chunk_id}: learned weight {result['learned_weight']} "
        f"({sign} {result['delta']:+.4f}, applies to {result['scope']})"
    )


@cli.command("feedback-stats")
def feedback_stats_cmd() -> None:
    """Show what the router has learned from feedback."""
    s = _mol().feedback_stats()
    click.echo(f"Votes: {s['votes']} ({s['useful']} useful, {s['not_useful']} not; "
               f"{s['query_aware_votes']} query-aware)")
    if s["learned"]:
        click.echo("Learned weights:")
        for entry in s["learned"]:
            click.echo(f"  {entry['learned_weight']:.4f}  {entry['chunk_id']}")
    else:
        click.echo("No learned weights yet — use 'molecule feedback <chunk_id>'.")
    click.echo(f"Bonds: {s['bonds'] or 'none'}")


# --- Memory ------------------------------------------------------------------------

@cli.command()
@click.argument("transcript_file", type=click.Path(exists=True))
@click.option("--task", default="session", help="Source task label for stored facts.")
def extract(transcript_file: str, task: str) -> None:
    """Extract durable facts from a transcript file and store them as memories."""
    text = Path(transcript_file).read_text(encoding="utf-8")
    result = _mol().extract(text, task)
    click.echo(
        f"✓ Extracted {result['extracted']} fact(s): "
        f"{result['stored']} stored, {result['skipped']} skipped"
    )
    for f in result["facts"]:
        click.echo(f"  [{f['outcome']}] ({f['confidence']}, {f['type']}) {f['text']}")


@cli.command()
def merge() -> None:
    """Merge near-duplicate memories into single clean facts."""
    click.echo(f"✓ Merged {_mol().merge_duplicates()['merged_groups']} duplicate group(s)")


@cli.command("export")
@click.argument("path", type=click.Path())
def export_cmd(path: str) -> None:
    """Export memories, feedback and learned bonds to a JSON snapshot."""
    r = _mol().export_snapshot(path)
    click.echo(f"✓ {r['path']}: {r['memories']} memories, {r['feedback']} votes, "
               f"{r['bonds']} bonds")


@cli.command("import")
@click.argument("path", type=click.Path(exists=True))
def import_cmd(path: str) -> None:
    """Merge a JSON snapshot into this store (safe to repeat)."""
    r = _mol().import_snapshot(path)
    if "error" in r:
        _fail(r["error"])
    click.echo(f"✓ Imported {r['memories']} memories, {r['feedback']} votes, "
               f"{r['bonds']} bonds ({r['skipped']} skipped)")


@cli.command()
@click.argument("path", type=click.Path())
def backup(path: str) -> None:
    """Copy the whole store to PATH, safely, while it is in use."""
    r = _mol().backup(path)
    click.echo(f"✓ {r['path']} ({r['bytes']} bytes)")


# --- Operations --------------------------------------------------------------------

@cli.command()
@json_option
def stats(as_json: bool) -> None:
    """Show token savings (this session and lifetime)."""
    s = _mol().stats()
    if as_json:
        return _print_json(s)
    life = s["lifetime"]
    click.echo(f"Tokens used:     {s['tokens_used_this_session']}")
    click.echo(f"Tokens baseline: {s['tokens_if_full_file_loaded']}")
    click.echo(f"Savings:         {s['savings_pct']}%")
    click.echo(
        f"Lifetime:        {life['tokens_used']} used vs {life['tokens_if_full_files_loaded']} "
        f"baseline over {life['sessions']} session(s) → {life['savings_pct']}% "
        f"(~${life['est_usd_saved']} saved)"
    )


@cli.command()
@click.argument("eval_file", type=click.Path(exists=True))
@click.option("-k", default=3, show_default=True)
@click.option("--ablate", is_flag=True, help="Also score with each signal switched off.")
@json_option
def eval(eval_file: str, k: int, ablate: bool, as_json: bool) -> None:
    """Run a retrieval eval from a JSON file of labeled queries."""
    queries = json.loads(Path(eval_file).read_text(encoding="utf-8"))
    result = _mol().evaluate(queries, k, ablate)
    if as_json:
        return _print_json(result)
    if "error" in result:
        _fail(result["error"])
    click.echo(
        f"Top-{k} precision: {result[f'top{k}_precision']} ({result['hits']}/{result['total']})"
        f"   Hit@1: {result['hit_at_1']}   MRR: {result['mrr']}"
    )
    click.echo(f"Search latency: p50 {result['latency_ms']['p50']} ms, "
               f"p95 {result['latency_ms']['p95']} ms")
    click.echo(f"Target 80%: {'✓ PASS' if result['passes'] else '✗ FAIL'}")
    if result["misses"]:
        click.echo("\nMisses:")
        for m in result["misses"]:
            click.echo(f"  Q: {m['query']}")
            click.echo(f"  Expected: {m['expected']}")
            click.echo(f"  Got:      {m['got']}")
    for row in result.get("ablation", []):
        click.echo(f"  {row['variant']:<22} top-{k} {row[f'top{k}_precision']:.3f}   "
                   f"Hit@1 {row['hit_at_1']:.3f}   MRR {row['mrr']:.3f}")
    if not result["passes"]:
        raise SystemExit(1)


@cli.command()
@json_option
def health(as_json: bool) -> None:
    """Check store status and indexed content."""
    h = _mol().health()
    if as_json:
        return _print_json(h)
    click.echo(f"Status:   {h['status']}  (molecule {h['version']}, schema v{h['schema_version']})")
    click.echo(f"DB:       {h['db_path']}")
    click.echo(f"Chunks:   {h['chunks_indexed']} from {h['sources']} source(s)")
    click.echo(f"Memories: {h['active_memories']}")
    click.echo(f"Model:    {h['embedding_id']}   Tokenizer: {h['tokenizer']}")
    if h["stale_sources"]:
        click.echo(f"Stale:    {', '.join(h['stale_sources'])}  (run: molecule sync)")


@cli.command()
def doctor() -> None:
    """Run every health check and say what needs fixing."""
    report = _mol().doctor()
    for check in report["checks"]:
        click.echo(f"{'✓' if check['ok'] else '✗'} {check['check']}: {check['detail']}")
    if not report["healthy"]:
        raise SystemExit(1)


@cli.command()
def setup() -> None:
    """One-time setup: download and cache the embedding model (~90MB)."""
    from molecule.config import load_config
    from molecule.core.embedder import embedder_for
    from molecule.core.tokens import tokenizer_name

    cfg = load_config()
    click.echo(f"Loading embedding model '{cfg.embedding_model}'…")
    t0 = time.perf_counter()
    embedder = embedder_for(cfg)
    embedder.encode(["warm up"], 16)
    click.echo(f"✓ {embedder.name} ready ({embedder.dim} dims) in "
               f"{time.perf_counter() - t0:.1f}s — cached for offline use")
    click.echo(f"✓ Tokenizer: {tokenizer_name()}")


@cli.command("serve")
def serve() -> None:
    """Start the MCP server (stdio)."""
    from molecule.server import mcp

    mcp.run()


def main() -> None:
    _force_utf8_stdio()
    cli()
