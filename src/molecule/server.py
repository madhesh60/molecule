"""server.py — the MCP server: 18 thin tools over one `Molecule` engine.

Works with both major versions of the MCP Python SDK (1.x `FastMCP`, 2.x
`MCPServer`).

Every tool is a one-line delegation; all behaviour lives in the engine and
`core/`, where it is unit-tested without spinning up a server. The engine is
created on first use (not at import), so importing this module has no side
effects and tests can inject their own engine with `set_engine()`.

Tool failures never crash the server: `_safe` logs the traceback to
`.molecule/molecule.log` (stdout is the MCP JSON-RPC channel and must stay
clean) and returns a structured `{"error": ...}`.
"""
from __future__ import annotations

import functools
import threading
from collections.abc import Awaitable, Callable
from typing import Any

try:  # mcp >= 2 renamed FastMCP to MCPServer; the decorator API is the same
    from mcp.server.mcpserver import MCPServer as _McpServer
except ModuleNotFoundError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _McpServer

from molecule.engine import Molecule
from molecule.logging_setup import get_logger

_INSTRUCTIONS = """\
Molecule routes you to the part of the project docs you need and remembers facts \
across sessions.

Fast path — one call:
  mol_context(query, budget)   → ready-to-paste context that fits the token budget.

Step by step, when you want control:
  mol_load_index()             → table of contents (cheap)
  mol_search(query)            → ranked ids + summaries + why (no bodies)
  mol_fetch(chunk_ids)         → full text of the chunks you chose
  mol_feedback(chunk_id, useful, query)  → teach the router

Memory: mem_write / mem_search / mem_extract (end of task). mol_stats shows token savings.
If a result's relevance is "low", the docs probably do not cover the question."""

mcp = _McpServer("molecule", instructions=_INSTRUCTIONS)

_engine: Molecule | None = None
_engine_lock = threading.Lock()


def get_engine() -> Molecule:
    """The process-wide engine, created on first use from cwd / $MOLECULE_ROOT."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = Molecule()
    return _engine


def set_engine(engine: Molecule | None) -> None:
    """Replace the engine (tests, or embedding the server in another app)."""
    global _engine
    _engine = engine


def _safe(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Catch tool failures: log the traceback, return a structured error
    instead of crashing the MCP server."""
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — server must survive any tool error
            try:
                get_logger(get_engine().config).exception("tool %s failed", fn.__name__)
            except Exception:  # noqa: BLE001 — logging must not mask the tool error
                pass
            return {"error": f"{type(exc).__name__}: {exc}"}
    return wrapper


# --- Context ---------------------------------------------------------------------

@mcp.tool()
@_safe
async def mol_context(query: str, budget: int = 0) -> dict:
    """One-call context: the most relevant doc sections (and remembered facts)
    for `query`, compiled to fit `budget` tokens (0 = default budget).
    Returns ready-to-paste `context`, per-item provenance, a `confidence`
    label and anything `omitted` for lack of room. Prefer this over
    search+fetch when you just need the answer material."""
    return get_engine().context(query, budget or None)


@mcp.tool()
@_safe
async def mol_load_index(max_tokens: int = 0, prefix: str = "") -> list[dict]:
    """The cheap table of contents: chunk ids, one-line summaries, token counts.
    Always fits `max_tokens` (0 = default) by reducing detail; large stores
    come back as an outline of groups — pass a group's `prefix` to drill in."""
    engine = get_engine()
    return engine.index(max_tokens or engine.config.index_token_budget, prefix or None)


@mcp.tool()
@_safe
async def mol_search(query: str, k: int = 5) -> list[dict]:
    """Ranked chunk and memory candidates (ids + summaries only, not bodies).
    Hybrid ranking: sentence-level embeddings + keyword coverage + heading
    match, adjusted by learned feedback. Each result has a `reason`, a
    `relevance` label (high/medium/low) and `related` chunks worth fetching
    with it."""
    return get_engine().search(query, k)


@mcp.tool()
@_safe
async def mol_fetch(chunk_ids: list[str]) -> list[dict]:
    """Full body text for the given chunk ids.
    Call after mol_search — fetching bodies is the expensive step."""
    return get_engine().fetch(chunk_ids)


@mcp.tool()
@_safe
async def mol_ingest(file_path: str, force: bool = False) -> dict:
    """(Re)index a markdown file, or every markdown file under a directory.
    Unchanged content is skipped unless force=True. Edits to an ingested file
    are picked up automatically afterwards."""
    return get_engine().ingest(file_path, force)


@mcp.tool()
@_safe
async def mol_sync(prune: bool = False) -> dict:
    """Re-index every source file that changed on disk. prune=True also
    removes the chunks of files that were deleted."""
    return get_engine().sync(prune)


@mcp.tool()
@_safe
async def mol_list_sources() -> list[dict]:
    """List all ingested source files with chunk counts and total tokens."""
    return get_engine().sources()


@mcp.tool()
@_safe
async def mol_remove_source(source_file: str) -> dict:
    """Remove all chunks for a given source file from the index."""
    return get_engine().remove_source(source_file)


# --- Self-learning router ------------------------------------------------------------

@mcp.tool()
@_safe
async def mol_feedback(chunk_id: str, useful: bool, query: str = "") -> dict:
    """Tell the router whether a fetched chunk was useful for your query.
    Pass the query: the lesson then applies to similar future questions
    instead of to every question. Bounded — one vote can never dominate."""
    return get_engine().feedback(chunk_id, useful, query)


@mcp.tool()
@_safe
async def mol_feedback_stats() -> dict:
    """What the router has learned: vote counts, per-chunk weights, and the
    strongest learned bonds between chunks."""
    return get_engine().feedback_stats()


# --- Memory ----------------------------------------------------------------------

@mcp.tool()
@_safe
async def mem_write(
    text: str,
    source_task: str,
    confidence: float = 0.8,
    mem_type: str = "preference",
) -> dict:
    """Store a durable fact. mem_type: 'preference' (permanent) or
    'task_state' (expires in 7 days). Automatically redacts secrets, skips
    duplicates and retires the fact it contradicts."""
    return get_engine().remember(text, source_task, confidence, mem_type)


@mcp.tool()
@_safe
async def mem_search(query: str, k: int = 5) -> list[dict]:
    """Retrieve relevant stored memories. Each result includes a 'reason'."""
    return get_engine().recall(query, k)


@mcp.tool()
@_safe
async def mem_extract(transcript: str, source_task: str = "session") -> dict:
    """Extract up to 5 durable facts from a session transcript and store them.
    Heuristic (no LLM needed): decisions, preferences, corrections. Direct
    statements get confidence 0.9, hedged ones 0.6; ephemeral facts expire in
    7 days. Call at the end of a task with the conversation text."""
    return get_engine().extract(transcript, source_task)


@mcp.tool()
@_safe
async def mol_forget(memory_id: str) -> dict:
    """Permanently delete a stored memory by its id (find it with mem_search)."""
    return get_engine().forget(memory_id)


@mcp.tool()
@_safe
async def mem_merge_duplicates() -> dict:
    """Merge near-duplicate memories into single clean facts. Run periodically."""
    return get_engine().merge_duplicates()


# --- Operations ------------------------------------------------------------------

@mcp.tool()
@_safe
async def mol_eval(queries: list[dict], k: int = 3) -> dict:
    """Run a retrieval eval. queries: list of {query, expected_chunk_id} (or
    {query, answer}). Returns top-k precision, Hit@1, MRR, per-query hits and
    whether the 80% target is met."""
    return get_engine().evaluate(queries, k)


@mcp.tool()
@_safe
async def mol_health() -> dict:
    """Health check: store status, counts, stale sources and per-operation
    latency percentiles."""
    return get_engine().health()


@mcp.tool()
@_safe
async def mol_stats() -> dict:
    """Token telemetry: tokens_used_this_session, tokens_if_full_file_loaded,
    savings_pct — plus a per-tool breakdown and lifetime totals. Counts every
    token Molecule returned, not just fetched chunks."""
    return get_engine().stats()
