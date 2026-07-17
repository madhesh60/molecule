from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp.server.fastmcp import FastMCP

from molecule.config import load_config
from molecule.core.eval import run_eval
from molecule.core.extractor import extract_memories
from molecule.core.feedback import feedback_stats, record_feedback
from molecule.core.ingest import do_ingest
from molecule.core.labeler import default_llm_fn
from molecule.core.memory import expire_stale, merge_duplicates, search_memories, write_memory
from molecule.core.retriever import fetch_chunks, load_router_index, rank_candidates
from molecule.core.telemetry import get_stats, record_usage
from molecule.db import get_connection, init_schema
from molecule.logging_setup import get_logger

mcp = FastMCP("molecule")
_config = load_config()
_conn = get_connection(_config.db_path)
init_schema(_conn)
expire_stale(_conn)
_log = get_logger(_config)

_session_baselines: set[str] = set()


def _safe(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Catch tool failures: log the traceback, return a structured error
    instead of crashing the MCP server."""
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — server must survive any tool error
            _log.exception("tool %s failed", fn.__name__)
            return {"error": f"{type(exc).__name__}: {exc}"}
    return wrapper


@mcp.tool()
@_safe
async def mol_ingest(file_path: str, force: bool = False) -> dict:
    """(Re)chunk a source file and store in the index. Skips unchanged sections unless force=True."""
    path = Path(file_path).expanduser().resolve()
    return do_ingest(path, _conn, _config, force)


@mcp.tool()
@_safe
async def mol_load_index() -> list[dict]:
    """Return the cheap router index: chunk_id, heading_path, summary, token_count.
    Always call this first — it's the table of contents, not the full content."""
    return load_router_index(_conn)


@mcp.tool()
@_safe
async def mol_search(query: str, k: int = 5) -> list[dict]:
    """Return ranked chunk and memory candidates (ids + summaries only, not full bodies).
    Uses hybrid BM25 keyword + embedding + heading-match ranking, adjusted by
    learned feedback weights. Each result includes a 'reason' field."""
    return rank_candidates(query, _conn, k, _config)


@mcp.tool()
@_safe
async def mol_fetch(chunk_ids: list[str]) -> list[dict]:
    """Return the full body text for the given chunk ids.
    Only call this after mol_search — fetching full bodies is the expensive step."""
    chunks = fetch_chunks(chunk_ids, _conn)
    tokens_used = sum(c["token_count"] for c in chunks)
    source_stems = {cid.split("::")[0] for cid in chunk_ids}
    baseline = 0
    for stem in source_stems:
        if stem not in _session_baselines:
            row = _conn.execute(
                "SELECT SUM(token_count) FROM chunks WHERE chunk_id LIKE ?", (f"{stem}::%",)
            ).fetchone()
            baseline += row[0] or 0
            _session_baselines.add(stem)
    record_usage(_conn, tokens_used, baseline)
    return chunks


@mcp.tool()
@_safe
async def mol_feedback(chunk_id: str, useful: bool, query: str = "") -> dict:
    """Tell the router whether a fetched chunk was actually useful for your query.
    The router learns: useful chunks rank higher next time, useless ones lower.
    Call after mol_fetch once you know whether the chunk answered the question."""
    return record_feedback(chunk_id, useful, query, _conn, _config)


@mcp.tool()
@_safe
async def mol_feedback_stats() -> dict:
    """Show what the router has learned: vote counts and per-chunk learned weights."""
    return feedback_stats(_conn)


@mcp.tool()
@_safe
async def mem_write(
    text: str,
    source_task: str,
    confidence: float = 0.8,
    mem_type: str = "preference",
) -> dict:
    """Store a durable fact. mem_type: 'preference' (permanent) or 'task_state' (expires in 7 days).
    Automatically deduplicates, detects conflicts, and redacts secrets."""
    return write_memory(text, source_task, confidence, _conn, _config, mem_type)


@mcp.tool()
@_safe
async def mem_search(query: str, k: int = 5) -> list[dict]:
    """Retrieve relevant stored memories. Each result includes a 'reason' field."""
    return search_memories(query, k, _conn, _config)


@mcp.tool()
@_safe
async def mem_extract(transcript: str, source_task: str = "session") -> dict:
    """Extract up to 5 durable facts from a session transcript and store them.
    Heuristic (no LLM needed): decisions, preferences, corrections. Direct
    statements get confidence 0.9, hedged ones 0.6; ephemeral facts expire in 7 days.
    Call at the end of a task with the conversation text."""
    return extract_memories(transcript, source_task, _conn, _config)


@mcp.tool()
@_safe
async def mol_forget(memory_id: str) -> dict:
    """Permanently delete a stored memory by its id. Use mem_search first to find the id."""
    row = _conn.execute("SELECT memory_id FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
    if not row:
        return {"error": f"memory_id not found: {memory_id}"}
    _conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
    _conn.commit()
    return {"deleted": memory_id}


@mcp.tool()
@_safe
async def mol_list_sources() -> list[dict]:
    """List all ingested source files with chunk counts and total tokens."""
    rows = _conn.execute(
        "SELECT source_file, COUNT(*) as chunks, SUM(token_count) as total_tokens "
        "FROM chunks GROUP BY source_file"
    ).fetchall()
    return [{"source_file": r["source_file"], "chunks": r["chunks"], "total_tokens": r["total_tokens"]} for r in rows]


@mcp.tool()
@_safe
async def mol_remove_source(source_file: str) -> dict:
    """Remove all chunks for a given source file from the index."""
    rows = _conn.execute(
        "SELECT COUNT(*) as n FROM chunks WHERE source_file=?", (source_file,)
    ).fetchone()
    if not rows or rows["n"] == 0:
        return {"error": f"No chunks found for: {source_file}"}
    _conn.execute("DELETE FROM chunks WHERE source_file=?", (source_file,))
    _conn.commit()
    return {"removed": source_file, "chunks_deleted": rows["n"]}


@mcp.tool()
@_safe
async def mem_merge_duplicates() -> dict:
    """Merge near-duplicate memories into single clean facts. Run periodically to keep memory lean."""
    merged = merge_duplicates(_conn, _config, default_llm_fn)
    return {"merged_groups": merged}


@mcp.tool()
@_safe
async def mol_eval(queries: list[dict], k: int = 3) -> dict:
    """Run retrieval precision eval. queries: list of {query, expected_chunk_id}.
    Returns top-k precision, per-query hits/misses, and whether the 80% target is met."""
    pairs = [(q["query"], q["expected_chunk_id"]) for q in queries]
    return run_eval(pairs, _conn, _config, k)


@mcp.tool()
@_safe
async def mol_health() -> dict:
    """Health check — returns DB path, chunk count, memory count, and server status."""
    chunks = _conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    memories = _conn.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
    return {
        "status": "ok",
        "db_path": str(_config.db_path),
        "chunks_indexed": chunks,
        "active_memories": memories,
        "embedding_model": _config.embedding_model,
    }


@mcp.tool()
@_safe
async def mol_stats() -> dict:
    """Return token telemetry: tokens_used_this_session, tokens_if_full_file_loaded, savings_pct."""
    return get_stats(_conn)
