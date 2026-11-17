"""engine.py — the Molecule engine: one object, every capability.

`Molecule` is the single entry point. The MCP server and the CLI are thin
wrappers over it, and so is your own code:

    from molecule import Molecule

    mol = Molecule()                      # opens ./.molecule/store.db
    mol.ingest("AGENTS.md")               # once; edits are picked up automatically

    ctx = mol.context("how do deploys work?", budget=800)
    print(ctx["context"])                 # ready-to-paste, never over budget

    mol.remember("User prefers small PRs")
    print(mol.stats()["savings_pct"])

What the engine adds on top of the `core/` functions:

  * **Thread safety** — one re-entrant lock around every operation, so a
    single instance can be shared by threads (and by an async server).
  * **Freshness** — edited source files are re-ingested automatically
    before a read, throttled to one cheap `stat()` pass every few seconds.
  * **Honest accounting** — every token returned to the caller is recorded.
  * **Learning hooks** — chunks fetched together bond together.
  * **Latency metrics** for every operation.
"""
from __future__ import annotations

import functools
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from molecule.config import MoleculeConfig, load_config
from molecule.core import bonds as bonds_mod
from molecule.core import telemetry
from molecule.core.context import build_context
from molecule.core.doctor import run_doctor
from molecule.core.embedder import embedder_name
from molecule.core.eval import run_ablation, run_eval
from molecule.core.extractor import extract_memories
from molecule.core.feedback import feedback_stats, record_feedback
from molecule.core.ingest import do_ingest, list_sources, remove_source
from molecule.core.memory import (
    default_llm_fn,
    enforce_capacity,
    expire_stale,
    forget_memory,
    memory_history,
    merge_duplicates,
    search_memories,
    write_memory,
)
from molecule.core.metrics import Metrics
from molecule.core.retriever import clamp_k, fetch_chunks, load_router_index, rank_candidates
from molecule.core.snapshot import backup_database, export_snapshot, import_snapshot
from molecule.core.sync import source_states, sync_sources
from molecule.core.tokens import count_tokens, tokenizer_name
from molecule.db import get_connection, init_schema, schema_version

_log = logging.getLogger("molecule")

_MAX_QUERY_CHARS = 2000
F = TypeVar("F", bound=Callable[..., Any])


def _operation(name: str, fresh: bool = False) -> Callable[[F], F]:
    """Run a method under the engine lock, time it, and (for reads that must
    see the latest files) sync edited sources first."""
    def decorate(method: F) -> F:
        @functools.wraps(method)
        def wrapper(self: Molecule, *args: Any, **kwargs: Any) -> Any:
            with self._lock, self._metrics.timed(name):
                if fresh:
                    self._auto_sync()
                return method(self, *args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorate


class Molecule:
    """Drop-in context router and memory for any AI agent.

    Args:
        root: the repo whose `.molecule/store.db` to use (default: cwd, or
            `$MOLECULE_ROOT`).
        config: an explicit `MoleculeConfig`; overrides `root`-based loading.
    """

    def __init__(self, root: str | Path | None = None, config: MoleculeConfig | None = None):
        self._config = config.validate() if config else load_config(Path(root) if root else None)
        self._conn = get_connection(self._config.db_path)
        init_schema(self._conn)
        expire_stale(self._conn)
        enforce_capacity(self._conn, self._config)
        self._lock = threading.RLock()
        self._metrics = Metrics()
        self._recent_fetches: deque[tuple[float, str]] = deque(maxlen=64)
        self._recent_useful: deque[tuple[float, str]] = deque(maxlen=64)
        self._last_sync_check = 0.0
        self._closed = False

    # --- lifecycle ---------------------------------------------------------------

    @property
    def config(self) -> MoleculeConfig:
        return self._config

    def close(self) -> None:
        """Release the database connection. The instance is unusable afterwards."""
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> Molecule:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- internals ---------------------------------------------------------------

    def _auto_sync(self) -> None:
        """Re-ingest source files edited since the last look (cheap, throttled)."""
        cfg = self._config
        if not cfg.auto_sync:
            return
        now = time.monotonic()
        if now - self._last_sync_check < cfg.auto_sync_interval_s:
            return
        self._last_sync_check = now
        try:
            if any(s["state"] == "changed" for s in source_states(self._conn)):
                report = sync_sources(self._conn, cfg)
                for item in report["reingested"]:
                    _log.info("auto-sync re-ingested %s", item["source"])
                for error in report["errors"]:
                    _log.warning("auto-sync: %s", error)
        except Exception:  # noqa: BLE001 — freshness must never break a read
            _log.exception("auto-sync failed")

    def _baseline_for(self, sources: set[str]) -> int:
        """Full-file tokens for sources delivered for the first time this session."""
        return telemetry.claim_baseline(self._conn, str(self._config.db_path), sources)

    def _account(self, kind: str, payload: Any) -> None:
        """Record the tokens of a response that carries no chunk bodies."""
        if payload:
            tokens = count_tokens(json.dumps(payload, ensure_ascii=False))
            telemetry.record_usage(self._conn, tokens, 0, kind)

    def _recent_partners(self, recent: deque[tuple[float, str]], ids: list[str]) -> list[str]:
        """Chunks used within the co-fetch window before `ids` (most recent
        first); then remember `ids` as used now."""
        now = time.monotonic()
        window = self._config.cofetch_window_s
        partners = [cid for ts, cid in reversed(recent) if now - ts <= window]
        recent.extend((now, cid) for cid in ids)
        return list(dict.fromkeys(partners))

    @staticmethod
    def _clean_query(query: str) -> str:
        return (query or "").strip()[:_MAX_QUERY_CHARS]

    # --- Context / chunks --------------------------------------------------------

    @_operation("ingest")
    def ingest(self, file_path: str | Path, force: bool = False) -> dict:
        """Chunk and index a markdown file, or every markdown file under a
        directory. Safe to call repeatedly — unchanged content is skipped."""
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            candidate = self._config.root / path
            path = candidate if candidate.exists() else path
        return do_ingest(path.resolve(), self._conn, self._config, force)

    @_operation("sync")
    def sync(self, prune: bool = False) -> dict:
        """Re-ingest every source file that changed on disk. With `prune`,
        also drop the chunks of files that were deleted."""
        self._last_sync_check = time.monotonic()
        return sync_sources(self._conn, self._config, prune)

    @_operation("index", fresh=True)
    def index(self, max_tokens: int | None = None, prefix: str | None = None) -> list[dict]:
        """The cheap router index (summaries only, never bodies).

        With `max_tokens`, detail degrades gracefully so the index always
        fits; `prefix` drills into one file or heading group."""
        entries = load_router_index(self._conn, max_tokens, prefix)
        self._account("index", entries)
        return entries

    @_operation("search", fresh=True)
    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Find relevant chunks and memories. Returns ids + summaries + the
        reason each was picked — no bodies."""
        query = self._clean_query(query)
        if not query:
            return []
        results = rank_candidates(query, self._conn, clamp_k(k, self._config), self._config)
        self._account("search", results)
        return results

    @_operation("fetch")
    def fetch(self, chunk_ids: list[str]) -> list[dict]:
        """Full content for the given chunk ids. Call after search()."""
        chunks = fetch_chunks(list(chunk_ids), self._conn)
        if not chunks:
            return []
        fetched = [c["chunk_id"] for c in chunks]
        for chunk in chunks:
            related = bonds_mod.related_for(
                self._conn, self._config, chunk["chunk_id"], exclude=set(fetched)
            )
            if related:
                chunk["related"] = related
        used = sum(c["token_count"] for c in chunks)
        baseline = self._baseline_for({cid.split("::")[0] for cid in fetched})
        telemetry.record_usage(self._conn, used, baseline, "fetch")
        # Chunks an agent chooses to read together are probably needed together.
        partners = self._recent_partners(self._recent_fetches, fetched)
        if bonds_mod.strengthen_cofetch(self._conn, fetched, self._config, partners):
            self._conn.commit()
        return chunks

    @_operation("context", fresh=True)
    def context(
        self,
        query: str,
        budget: int | None = None,
        compress: bool = True,
        include_memories: bool = True,
    ) -> dict:
        """One call instead of index → search → fetch: the best context for
        `query` that fits in `budget` tokens, with provenance per item."""
        query = self._clean_query(query)
        if not query:
            return {"error": "empty query"}
        result = build_context(query, self._conn, self._config, budget, compress,
                               include_memories)
        baseline = self._baseline_for(set(result["sources"]))
        telemetry.record_usage(self._conn, result["tokens_used"], baseline, "context")
        return result

    @_operation("sources")
    def sources(self) -> list[dict]:
        """List all ingested source files."""
        return list_sources(self._conn)

    @_operation("remove_source")
    def remove_source(self, source_file: str) -> dict:
        """Remove all chunks for a source file."""
        return remove_source(self._conn, source_file)

    # --- Memory ------------------------------------------------------------------

    @_operation("remember")
    def remember(
        self,
        text: str,
        source_task: str = "agent",
        confidence: float = 0.8,
        mem_type: str = "preference",
    ) -> dict:
        """Store a durable fact. mem_type: 'preference' (permanent) or
        'task_state' (expires after `task_state_expiry_days`)."""
        result = write_memory(text, source_task, confidence, self._conn, self._config, mem_type)
        if result.get("stored"):
            enforce_capacity(self._conn, self._config)
        return result

    @_operation("recall")
    def recall(self, query: str, k: int | None = None) -> list[dict]:
        """Retrieve relevant memories by query."""
        query = self._clean_query(query)
        if not query:
            return []
        hits = search_memories(query, clamp_k(k, self._config), self._conn, self._config)
        self._account("memory", hits)
        return hits

    @_operation("forget")
    def forget(self, memory_id: str) -> dict:
        """Delete a stored memory by id."""
        return forget_memory(self._conn, memory_id)

    @_operation("extract")
    def extract(self, transcript: str, source_task: str = "session") -> dict:
        """Extract up to `extract_max_facts` durable facts from a transcript
        and store them. Heuristic — no LLM call needed."""
        return extract_memories(transcript, source_task, self._conn, self._config)

    @_operation("merge_duplicates")
    def merge_duplicates(self) -> dict:
        """Merge clusters of near-duplicate memories into single facts."""
        return {"merged_groups": merge_duplicates(self._conn, self._config, default_llm_fn)}

    @_operation("memory_history")
    def memory_history(self, memory_id: str) -> list[dict]:
        """The chain of facts that replaced each other on one topic."""
        return memory_history(self._conn, memory_id)

    # --- Self-learning router ----------------------------------------------------

    @_operation("feedback")
    def feedback(self, chunk_id: str, useful: bool, query: str = "") -> dict:
        """Tell the router whether a chunk was useful for a query. The effect
        is strongest on similar future queries and bounded everywhere."""
        result = record_feedback(chunk_id, useful, self._clean_query(query), self._conn,
                                 self._config)
        if useful and "error" not in result:
            # Two chunks that both helped in the same short window belong together.
            partners = self._recent_partners(self._recent_useful, [chunk_id])
            if bonds_mod.strengthen_cofetch(self._conn, [chunk_id], self._config, partners):
                self._conn.commit()
        return result

    @_operation("feedback_stats")
    def feedback_stats(self) -> dict:
        """What the router has learned: votes, weights and learned bonds."""
        return {**feedback_stats(self._conn), **bonds_mod.bond_stats(self._conn)}

    @_operation("bonds")
    def bonds(self, chunk_id: str) -> list[dict]:
        """Chunks bonded to `chunk_id` (cross-references, continuations, co-use)."""
        return bonds_mod.related_for(self._conn, self._config, chunk_id, limit=20)

    # --- Telemetry and operations ------------------------------------------------

    @_operation("stats")
    def stats(self) -> dict:
        """Live token savings for this session and for the store's lifetime."""
        return telemetry.get_stats(self._conn, self._config.cost_per_mtok_usd)

    @_operation("health")
    def health(self) -> dict:
        """Store status, counts and per-operation latency."""
        from molecule import __version__

        chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        memories = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]
        states = source_states(self._conn)
        return {
            "status": "ok",
            "version": __version__,
            "db_path": str(self._config.db_path),
            "schema_version": schema_version(self._conn),
            "chunks_indexed": chunks,
            "active_memories": memories,
            "sources": len(states),
            "stale_sources": [s["source"] for s in states if s["state"] != "fresh"],
            "embedding_model": self._config.embedding_model,
            "embedding_id": embedder_name(self._config),
            "tokenizer": tokenizer_name(),
            "metrics": self._metrics.snapshot(),
        }

    def metrics(self) -> dict:
        """Per-operation call counts and latency percentiles since start-up."""
        return self._metrics.snapshot()

    @_operation("doctor")
    def doctor(self) -> dict:
        """Run every health check and say what (if anything) needs fixing."""
        return run_doctor(self._conn, self._config)

    @_operation("evaluate")
    def evaluate(self, queries: list, k: int = 3, ablate: bool = False) -> dict:
        """Score a labeled query set (see core/eval.py for the formats)."""
        result = run_eval(queries, self._conn, self._config, k)
        if ablate and "error" not in result:
            result["ablation"] = run_ablation(queries, self._conn, self._config, k)
        return result

    @_operation("export")
    def export_snapshot(self, path: str | Path) -> dict:
        """Write memories, feedback and learned bonds to a JSON file."""
        return export_snapshot(self._conn, Path(path))

    @_operation("import")
    def import_snapshot(self, path: str | Path) -> dict:
        """Merge a snapshot written by `export_snapshot` into this store."""
        return import_snapshot(self._conn, Path(path))

    @_operation("backup")
    def backup(self, path: str | Path) -> dict:
        """Copy the whole store to `path`, safely, while it is in use."""
        return backup_database(self._conn, Path(path))
