from __future__ import annotations

from pathlib import Path

from molecule.config import MoleculeConfig, load_config
from molecule.core.extractor import extract_memories
from molecule.core.feedback import feedback_stats, record_feedback
from molecule.core.ingest import do_ingest
from molecule.core.memory import expire_stale, search_memories, write_memory
from molecule.core.retriever import fetch_chunks, load_router_index, rank_candidates
from molecule.core.telemetry import get_stats, record_usage
from molecule.db import get_connection, init_schema


class Molecule:
    """
    Drop-in context router and memory for any AI agent.

    Usage:
        from molecule import Molecule

        mol = Molecule()
        mol.ingest("AGENTS.md")

        chunks = mol.search("how does auth work")
        content = mol.fetch([chunks[0]["id"]])

        mol.remember("User prefers dark mode")
        facts = mol.recall("UI preferences")

        print(mol.stats())
    """

    def __init__(self, root: str | Path | None = None, config: MoleculeConfig | None = None):
        self._config = config or load_config(Path(root) if root else None)
        self._conn = get_connection(self._config.db_path)
        init_schema(self._conn)
        expire_stale(self._conn)
        self._session_baselines: set[str] = set()

    # --- Context / chunks ---

    def ingest(self, file_path: str | Path, force: bool = False) -> dict:
        """Chunk and index a markdown file. Safe to call repeatedly — skips unchanged sections."""
        return do_ingest(Path(file_path).resolve(), self._conn, self._config, force)

    def index(self) -> list[dict]:
        """Return the cheap router index (summaries only, no full content)."""
        return load_router_index(self._conn)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Find relevant chunks and memories by query. Returns ids + summaries + reason."""
        return rank_candidates(query, self._conn, k or self._config.default_k, self._config)

    def fetch(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch full content for the given chunk ids. Call after search()."""
        chunks = fetch_chunks(chunk_ids, self._conn)
        tokens_used = sum(c["token_count"] for c in chunks)
        baseline = 0
        for stem in {cid.split("::")[0] for cid in chunk_ids}:
            if stem not in self._session_baselines:
                row = self._conn.execute(
                    "SELECT SUM(token_count) FROM chunks WHERE chunk_id LIKE ?", (f"{stem}::%",)
                ).fetchone()
                baseline += row[0] or 0
                self._session_baselines.add(stem)
        record_usage(self._conn, tokens_used, baseline)
        return chunks

    def remove_source(self, source_file: str) -> dict:
        """Remove all chunks for a source file."""
        row = self._conn.execute(
            "SELECT COUNT(*) as n FROM chunks WHERE source_file=?", (source_file,)
        ).fetchone()
        if not row or row["n"] == 0:
            return {"error": f"No chunks found for: {source_file}"}
        self._conn.execute("DELETE FROM chunks WHERE source_file=?", (source_file,))
        self._conn.commit()
        return {"removed": source_file, "chunks_deleted": row["n"]}

    def sources(self) -> list[dict]:
        """List all ingested source files."""
        rows = self._conn.execute(
            "SELECT source_file, COUNT(*) as chunks, SUM(token_count) as total_tokens "
            "FROM chunks GROUP BY source_file"
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Memory ---

    def remember(
        self,
        text: str,
        source_task: str = "agent",
        confidence: float = 0.8,
        mem_type: str = "preference",
    ) -> dict:
        """Store a durable fact. mem_type: 'preference' (permanent) or 'task_state' (7-day expiry)."""
        return write_memory(text, source_task, confidence, self._conn, self._config, mem_type)

    def recall(self, query: str, k: int | None = None) -> list[dict]:
        """Retrieve relevant memories by query."""
        return search_memories(query, k or self._config.default_k, self._conn, self._config)

    def forget(self, memory_id: str) -> dict:
        """Delete a stored memory by id."""
        row = self._conn.execute(
            "SELECT memory_id FROM memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if not row:
            return {"error": f"memory_id not found: {memory_id}"}
        self._conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
        self._conn.commit()
        return {"deleted": memory_id}

    def extract(self, transcript: str, source_task: str = "session") -> dict:
        """Extract up to 5 durable facts from a transcript and store them.
        Heuristic — no LLM call needed. Direct statements → 0.9 confidence,
        hedged → 0.6, ephemeral facts become task_state (7-day expiry)."""
        return extract_memories(transcript, source_task, self._conn, self._config)

    # --- Self-learning router ---

    def feedback(self, chunk_id: str, useful: bool, query: str = "") -> dict:
        """Tell the router whether a chunk was useful. Useful chunks rank
        higher on future searches; useless ones sink. Bounded EMA — one vote
        can never dominate."""
        return record_feedback(chunk_id, useful, query, self._conn, self._config)

    def feedback_stats(self) -> dict:
        """What the router has learned: vote counts + per-chunk learned weights."""
        return feedback_stats(self._conn)

    # --- Telemetry ---

    def stats(self) -> dict:
        """Return live token savings: tokens_used, tokens_baseline, savings_pct."""
        return get_stats(self._conn)

    def health(self) -> dict:
        """Return DB status, chunk count, memory count."""
        chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        memories = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='active'"
        ).fetchone()[0]
        return {
            "status": "ok",
            "db_path": str(self._config.db_path),
            "chunks_indexed": chunks,
            "active_memories": memories,
            "embedding_model": self._config.embedding_model,
        }


__all__ = ["Molecule", "MoleculeConfig", "load_config"]
