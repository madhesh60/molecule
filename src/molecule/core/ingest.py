"""ingest.py — turn source files into indexed chunks.

    read → chunk → label → embed (chunk + passages) → store, atomically

Design points:

  * **Incremental by default.** A file whose bytes have not changed is
    skipped after one hash comparison. Inside a changed file, only chunks
    whose content changed are re-embedded.
  * **Atomic.** All writes for one file happen in a single transaction: a
    crash or error leaves the previous index intact, never half a file.
  * **Self-describing.** Each source records the chunker/embedder settings
    it was built with (`signature`); changing the model or chunk size
    rebuilds that source on the next ingest instead of mixing indexes.
  * **Learning survives edits.** Feedback and learned bonds are keyed by
    chunk id; when a heading is renamed but the body is unchanged, they are
    carried over to the new id.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from molecule.config import MoleculeConfig
from molecule.core.bonds import rebuild_structural_bonds
from molecule.core.chunker import CHUNKER_VERSION, RawSection, chunk_text
from molecule.core.embed_text import EMBEDDING_TEXT_VERSION, build_embedding_text
from molecule.core.embedder import (
    EmbeddingUnavailableError,
    embedder_for,
    embedder_name,
    vector_is_current,
)
from molecule.core.labeler import default_llm_fn, generate_labels
from molecule.core.passages import passage_embedding_texts, split_passages
from molecule.core.tokens import count_tokens
from molecule.db import transaction, utc_now_iso

MARKDOWN_SUFFIXES = (".md", ".markdown", ".mdx")
_SKIP_DIRS = {"node_modules", "venv", "env", "__pycache__", "dist", "build", "target"}

_UPSERT_SQL = """
INSERT INTO chunks (
    chunk_id, source_file, heading_path, summary, content, token_count,
    content_hash, always_include, weight, embedding, embedding_model,
    passage_embeddings, passage_count, position, embedding_text_version, last_updated
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(chunk_id) DO UPDATE SET
    source_file=excluded.source_file, heading_path=excluded.heading_path,
    summary=excluded.summary, content=excluded.content,
    token_count=excluded.token_count, content_hash=excluded.content_hash,
    always_include=excluded.always_include, weight=excluded.weight,
    embedding=excluded.embedding, embedding_model=excluded.embedding_model,
    passage_embeddings=excluded.passage_embeddings,
    passage_count=excluded.passage_count, position=excluded.position,
    embedding_text_version=excluded.embedding_text_version,
    last_updated=excluded.last_updated
"""
# NOTE: learned_weight is intentionally absent — router feedback survives re-ingest.


def source_key(path: Path, root: Path) -> str:
    """Stable, portable name for a source file: its repo-relative POSIX path
    (or just the file name when it lives outside the repo). Used as the
    chunk id prefix, so ids do not change when the repo is moved."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def index_signature(config: MoleculeConfig, embedder_id: str) -> str:
    """Everything that, if changed, makes an existing index for a file stale."""
    return ".".join(str(part) for part in (
        f"c{CHUNKER_VERSION}", f"e{EMBEDDING_TEXT_VERSION}", embedder_id,
        config.max_chunk_tokens, config.min_chunk_tokens, config.heading_split_depth,
        config.chunk_embed_tokens, int(config.passage_vectors),
        config.passage_embed_tokens, config.max_passages_per_chunk,
    ))


def chunk_id_for(source: str, heading_path: list[str]) -> str:
    return f"{source}::{':'.join(heading_path)}"


def iter_markdown_files(directory: Path, limit: int) -> list[Path]:
    """Markdown files under a directory, skipping hidden and vendored folders."""
    found: list[Path] = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).parts
        if any(part.startswith(".") or part in _SKIP_DIRS for part in relative[:-1]):
            continue
        if path.is_file() and path.suffix.lower() in MARKDOWN_SUFFIXES:
            found.append(path)
            if len(found) >= limit:
                break
    return found


def _timing(model_load: float = 0.0, embed: float = 0.0, passages: float = 0.0) -> dict:
    return {
        "model_load_s": round(model_load, 3),
        "embed_s": round(embed, 3),
        "passages_s": round(passages, 3),
    }


def _embed_passages(
    sections: list[RawSection], config: MoleculeConfig, embedder
) -> list[tuple[bytes | None, int]]:
    """(blob, count) of passage vectors per section, embedded in one batch."""
    if not config.passage_vectors:
        return [(None, 0)] * len(sections)
    texts: list[str] = []
    counts: list[int] = []
    for section in sections:
        passages = split_passages(section.content, config.max_passages_per_chunk)
        counts.append(len(passages))
        texts.extend(passage_embedding_texts(section.heading_path[-1], passages))
    vectors = embedder.encode(texts, config.passage_embed_tokens) if texts else None
    out: list[tuple[bytes | None, int]] = []
    offset = 0
    for count in counts:
        if count and vectors is not None:
            out.append((np.ascontiguousarray(vectors[offset : offset + count]).tobytes(), count))
        else:
            out.append((None, 0))
        offset += count
    return out


def _carry_learning(conn: sqlite3.Connection, old_id: str, new_id: str) -> None:
    """A heading was renamed but the body is the same chunk: move what the
    router learned about it (votes, weight, co-fetch bonds) to the new id."""
    conn.execute(
        "UPDATE chunks SET learned_weight = "
        "(SELECT learned_weight FROM chunks WHERE chunk_id=?) WHERE chunk_id=?",
        (old_id, new_id),
    )
    conn.execute("UPDATE chunk_feedback SET chunk_id=? WHERE chunk_id=?", (new_id, old_id))
    for column in ("src", "dst"):
        conn.execute(
            f"UPDATE OR IGNORE chunk_bonds SET {column}=? WHERE {column}=? AND kind='cofetch'",
            (new_id, old_id),
        )


def ingest_file(
    path: Path,
    conn: sqlite3.Connection,
    cfg: MoleculeConfig,
    force: bool = False,
) -> dict:
    """(Re)index one markdown file. Returns counts, or {"error": ...}."""
    if not path.exists():
        return {"error": f"File not found: {path}"}
    if not path.is_file():
        return {"error": f"Not a file: {path}"}
    stat = path.stat()
    if stat.st_size > cfg.max_file_bytes:
        return {"error": f"File too large ({stat.st_size} bytes > {cfg.max_file_bytes}): {path}"}

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {"error": f"Not a UTF-8 text file: {path}"}

    source = source_key(path, cfg.root)
    file_hash = hashlib.sha256(raw).hexdigest()
    known = conn.execute("SELECT * FROM sources WHERE source_file=?", (source,)).fetchone()

    # Fast path: same bytes, same settings → nothing to do (no model load).
    expected_signature = index_signature(cfg, embedder_name(cfg))
    if (
        known and not force
        and known["file_hash"] == file_hash
        and known["signature"] == expected_signature
    ):
        if known["mtime_ns"] != stat.st_mtime_ns:
            conn.execute(
                "UPDATE sources SET mtime_ns=? WHERE source_file=?", (stat.st_mtime_ns, source)
            )
            conn.commit()
        return {
            "file": str(path), "source": source, "unchanged": True,
            "new": 0, "updated": 0, "skipped": known["chunk_count"], "removed": 0,
            "total": known["chunk_count"], "timing": _timing(),
        }

    # Drop empty-bodied sections — an empty chunk can only pollute results.
    sections = [s for s in chunk_text(text, cfg) if s.content.strip()]
    if not sections:
        return {"error": f"No indexable content in: {path}"}
    labels = generate_labels(sections, conn, default_llm_fn)

    t0 = time.perf_counter()
    try:
        embedder = embedder_for(cfg)
    except EmbeddingUnavailableError as exc:
        return {"error": str(exc)}
    model_load_s = time.perf_counter() - t0
    signature = index_signature(cfg, embedder.name)
    rebuild_all = force or not known or known["signature"] != signature

    # Rows currently stored for this file (also under the pre-0.2 absolute path).
    existing = {
        r["chunk_id"]: r
        for r in conn.execute(
            "SELECT chunk_id, content_hash, embedding, embedding_model, "
            "embedding_text_version, position FROM chunks WHERE source_file IN (?, ?)",
            (source, str(path)),
        )
    }

    ids = [chunk_id_for(source, s.heading_path) for s in sections]
    changed: list[int] = []
    for i, (chunk_id, section) in enumerate(zip(ids, sections)):
        row = existing.get(chunk_id)
        unchanged = (
            row is not None
            and not rebuild_all
            and row["content_hash"] == section.hash
            and row["embedding_text_version"] >= EMBEDDING_TEXT_VERSION
            and vector_is_current(row["embedding"], row["embedding_model"], embedder)
        )
        if not unchanged:
            changed.append(i)

    # Embed everything that changed in two batched calls (chunks, passages).
    embed_s = passages_s = 0.0
    vectors = None
    passage_blobs: list[tuple[bytes | None, int]] = []
    if changed:
        t0 = time.perf_counter()
        vectors = embedder.encode(
            [build_embedding_text(sections[i].heading_path, labels[i], sections[i].content)
             for i in changed],
            cfg.chunk_embed_tokens,
        )
        embed_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        passage_blobs = _embed_passages([sections[i] for i in changed], cfg, embedder)
        passages_s = time.perf_counter() - t0

    now = utc_now_iso()
    current_ids = set(ids)
    stale = [cid for cid in existing if cid not in current_ids]
    stale_by_hash = {existing[cid]["content_hash"]: cid for cid in stale}
    new_count = updated = 0

    with transaction(conn):
        for slot, i in enumerate(changed):
            section, chunk_id = sections[i], ids[i]
            blob, passage_count = passage_blobs[slot]
            conn.execute(_UPSERT_SQL, (
                chunk_id, source, json.dumps(section.heading_path), labels[i],
                section.content, section.token_count, section.hash,
                int(section.always_include),
                cfg.always_include_weight if section.always_include else 1.0,
                vectors[slot].tobytes(), embedder.name, blob, passage_count,
                section.position, EMBEDDING_TEXT_VERSION, now,
            ))
            if chunk_id in existing:
                updated += 1
            else:
                new_count += 1
                renamed_from = stale_by_hash.get(section.hash)
                if renamed_from:
                    _carry_learning(conn, renamed_from, chunk_id)

        changed_set = set(changed)
        for i, (chunk_id, section) in enumerate(zip(ids, sections)):
            if i not in changed_set and existing[chunk_id]["position"] != section.position:
                conn.execute(
                    "UPDATE chunks SET position=?, source_file=? WHERE chunk_id=?",
                    (section.position, source, chunk_id),
                )

        # Remove chunks whose heading (and therefore id) no longer exists in the
        # file — otherwise renamed sections leave orphans that pollute search.
        conn.executemany("DELETE FROM chunks WHERE chunk_id=?", [(cid,) for cid in stale])

        bonds = rebuild_structural_bonds(conn, source, cfg)
        conn.execute(
            "INSERT OR REPLACE INTO sources (source_file, abs_path, file_hash, size_bytes, "
            "mtime_ns, total_tokens, chunk_count, signature, ingested_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (source, str(path), file_hash, stat.st_size, stat.st_mtime_ns,
             count_tokens(text), len(sections), signature, now),
        )

    return {
        "file": str(path),
        "source": source,
        "unchanged": False,
        "new": new_count,
        "updated": updated,
        "skipped": len(sections) - len(changed),
        "removed": len(stale),
        "total": len(sections),
        "passages": sum(count for _, count in passage_blobs),
        "bonds": bonds,
        "timing": _timing(model_load_s, embed_s, passages_s),
    }


def ingest_directory(
    directory: Path, conn: sqlite3.Connection, cfg: MoleculeConfig, force: bool = False
) -> dict:
    """Index every markdown file under a directory (recursively)."""
    files = iter_markdown_files(directory, cfg.max_ingest_files)
    if not files:
        return {"error": f"No markdown files found under: {directory}"}
    totals = {"new": 0, "updated": 0, "skipped": 0, "removed": 0, "total": 0}
    timing = {"model_load_s": 0.0, "embed_s": 0.0, "passages_s": 0.0}
    results, errors = [], []
    for file in files:
        result = ingest_file(file, conn, cfg, force)
        if "error" in result:
            errors.append(result["error"])
            continue
        results.append({k: result[k] for k in ("source", "new", "updated", "skipped", "total")})
        for key in totals:
            totals[key] += result[key]
        for key in timing:
            timing[key] = round(timing[key] + result["timing"][key], 3)
    return {
        "file": str(directory), "files": len(results), **totals,
        "timing": timing, "results": results, "errors": errors,
    }


def do_ingest(
    path: Path,
    conn: sqlite3.Connection,
    cfg: MoleculeConfig,
    force: bool = False,
) -> dict:
    """Index a markdown file, or every markdown file under a directory."""
    if path.is_dir():
        return ingest_directory(path, conn, cfg, force)
    return ingest_file(path, conn, cfg, force)


def remove_source(conn: sqlite3.Connection, name: str) -> dict:
    """Remove a source file's chunks. `name` may be the source key, the
    absolute path, or a bare file name when that is unambiguous."""
    candidates = [
        r["source_file"] for r in conn.execute(
            "SELECT DISTINCT source_file FROM chunks"
        )
    ]
    abs_paths = {
        r["abs_path"]: r["source_file"]
        for r in conn.execute("SELECT source_file, abs_path FROM sources")
    }
    if name in candidates:
        target = name
    elif name in abs_paths:
        target = abs_paths[name]
    else:
        by_name = [c for c in candidates if Path(c).name == Path(name).name]
        if len(by_name) != 1:
            return {"error": f"No chunks found for: {name}"}
        target = by_name[0]
    with transaction(conn):
        deleted = conn.execute("DELETE FROM chunks WHERE source_file=?", (target,)).rowcount
        conn.execute("DELETE FROM sources WHERE source_file=?", (target,))
    return {"removed": target, "chunks_deleted": deleted}


def list_sources(conn: sqlite3.Connection) -> list[dict]:
    """Ingested files with chunk counts and their true token size."""
    rows = conn.execute(
        "SELECT c.source_file AS source_file, COUNT(*) AS chunks, "
        "COALESCE(MAX(s.total_tokens), SUM(c.token_count)) AS total_tokens, "
        "MAX(s.ingested_at) AS ingested_at "
        "FROM chunks c LEFT JOIN sources s ON s.source_file = c.source_file "
        "GROUP BY c.source_file ORDER BY c.source_file"
    ).fetchall()
    return [dict(r) for r in rows]
