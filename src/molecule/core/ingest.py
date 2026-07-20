from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from molecule.config import MoleculeConfig
from molecule.core.chunker import chunk_file
from molecule.core.embed_text import EMBEDDING_TEXT_VERSION, build_embedding_text
from molecule.core.embedder import embed, get_model
from molecule.core.labeler import default_llm_fn, generate_labels
from molecule.db import utc_now_iso


def do_ingest(
    path: Path,
    conn: sqlite3.Connection,
    cfg: MoleculeConfig,
    force: bool = False,
) -> dict:
    """Shared ingest logic used by both CLI and MCP server."""
    if not path.exists():
        return {"error": f"File not found: {path}"}

    try:
        sections = chunk_file(path, cfg)
    except UnicodeDecodeError:
        return {"error": f"Not a UTF-8 text file: {path}"}
    # Drop empty-bodied sections (empty/whitespace-only files) — an empty
    # chunk can only pollute search results.
    sections = [s for s in sections if s.content.strip()]
    if not sections:
        return {"error": f"No indexable content in: {path}"}
    labels = generate_labels(sections, conn, default_llm_fn)
    now = utc_now_iso()
    new_count = updated = skipped = 0

    # Remove stale chunks whose heading (and therefore chunk_id) no longer
    # exists in the file — otherwise renamed sections leave orphan rows that
    # pollute search results.
    current_ids = {
        f"{path.name}::{':'.join(sec.heading_path)}" for sec in sections
    }
    stale = [
        r[0] for r in conn.execute(
            "SELECT chunk_id FROM chunks WHERE source_file=?", (str(path),)
        ).fetchall()
        if r[0] not in current_ids
    ]
    removed = len(stale)
    for cid in stale:
        conn.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))

    # Separate changed vs unchanged — only embed changed chunks (batched)
    to_embed: list[str] = []  # embedding text per changed chunk
    rows_to_upsert = []

    for sec, label in zip(sections, labels):
        chunk_id = f"{path.name}::{':'.join(sec.heading_path)}"
        existing = conn.execute(
            "SELECT content_hash, embedding, embedding_text_version FROM chunks WHERE chunk_id=?",
            (chunk_id,),
        ).fetchone()

        h = sec.hash
        unchanged = (
            existing
            and existing["content_hash"] == h
            and existing["embedding"] is not None
            and existing["embedding_text_version"] >= EMBEDDING_TEXT_VERSION
            and not force
        )
        if unchanged:
            skipped += 1
            continue

        weight = cfg.always_include_weight if sec.always_include else 1.0
        rows_to_upsert.append((
            chunk_id, str(path), json.dumps(sec.heading_path), label, sec.content,
            sec.token_count, h, int(sec.always_include), weight,
            EMBEDDING_TEXT_VERSION, now,
            bool(existing),
        ))
        # Embed heading path + summary + content together — much stronger
        # routing signal than raw content alone.
        to_embed.append(build_embedding_text(sec.heading_path, label, sec.content))

    # Batch embed all changed chunks in ONE model.encode() call.
    # Time model load and embedding separately: first-run model download
    # dominates wall time and shouldn't count against the <5s embed target.
    model_load_s = embed_s = 0.0
    if to_embed:
        t0 = time.perf_counter()
        get_model(cfg.embedding_model)  # warm/load (no-op if cached)
        model_load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        vecs = embed(to_embed, cfg.embedding_model)  # shape (n, dim)
        embed_s = time.perf_counter() - t0

        for row, vec in zip(rows_to_upsert, vecs):
            is_existing = row[-1]
            chunk_id = row[0]
            conn.execute(
                """INSERT INTO chunks (chunk_id, source_file, heading_path, summary, content,
                   token_count, content_hash, always_include, weight,
                   embedding_text_version, last_updated)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                   summary=excluded.summary, content=excluded.content,
                   token_count=excluded.token_count, content_hash=excluded.content_hash,
                   always_include=excluded.always_include, weight=excluded.weight,
                   embedding_text_version=excluded.embedding_text_version,
                   last_updated=excluded.last_updated""",
                row[:-1],  # exclude is_existing flag
            )
            # NOTE: learned_weight is intentionally NOT touched — router feedback
            # survives re-ingest.
            conn.execute(
                "UPDATE chunks SET embedding=? WHERE chunk_id=?",
                (vec.tobytes(), chunk_id),
            )
            updated += 1 if is_existing else 0
            new_count += 0 if is_existing else 1

    conn.commit()
    return {
        "file": str(path),
        "new": new_count,
        "updated": updated,
        "skipped": skipped,
        "removed": removed,
        "total": len(sections),
        "timing": {
            "model_load_s": round(model_load_s, 3),
            "embed_s": round(embed_s, 3),
        },
    }
