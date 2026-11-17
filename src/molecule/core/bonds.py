"""bonds.py — the chunk graph: which sections belong together.

Chunks are the atoms; bonds are what makes it a molecule. A bond says "an
agent that reads this chunk will probably need that one too". Three kinds:

  * **ref** — the text says so itself: "see §7.3", "[Rollback](#rollback)",
    "[setup](docs/setup.md)". Parsed at ingest, free, deterministic.
  * **continuation** — consecutive parts of one section that was too large
    to stay whole.
  * **cofetch** — learned from use, Hebbian style ("chunks that are fetched
    together bond together"): every time two chunks are fetched within the
    same short window their bond moves a step toward 1,

        strength ← strength + rate · (1 − strength)

    and it fades with a half-life when it stops being reinforced, so a
    one-off coincidence never becomes permanent.

Bonds never override relevance. They are used for (a) the `related` hint on
search results — so the agent can fetch a companion section without another
search round-trip, and (b) the context compiler, which may add a bonded
neighbour of a selected chunk when the token budget allows.
"""
from __future__ import annotations

import json
import posixpath
import re
import sqlite3
from datetime import datetime, timezone
from itertools import combinations

from molecule.config import MoleculeConfig
from molecule.core.cache import versioned
from molecule.db import utc_now, utc_now_iso

STRUCTURAL_KINDS = ("ref", "continuation")
_MAX_REFS_PER_CHUNK = 8
_MAX_COFETCH_GROUP = 8

_SECTION_REF_RE = re.compile(r"(?:§\s?|\b[Ss]ection\s+|\b[Ss]ec\.\s*)(\d+(?:\.\d+)*)")
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_HEADING_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?(?:\s|$)")
_PART_SUFFIX_RE = re.compile(r"(?: §\d+)?(?: \(\d+\))?$")
_SLUG_DROP_RE = re.compile(r"[^\w\- ]")


def _base_heading(heading: str) -> str:
    """Heading without the ' §n' part marker / ' (n)' duplicate marker."""
    return _PART_SUFFIX_RE.sub("", heading)


def slugify(heading: str) -> str:
    """GitHub-style anchor for a heading ("7.3 Rollback" → "73-rollback")."""
    return _SLUG_DROP_RE.sub("", heading.strip().lower()).replace(" ", "-")


def _heading_number(heading: str) -> str | None:
    match = _HEADING_NUMBER_RE.match(heading)
    return match.group(1) if match else None


class _SourceMap:
    """Lookup tables for resolving references into one source's chunks."""

    def __init__(self, rows: list[sqlite3.Row]) -> None:
        self.by_number: dict[str, list[str]] = {}
        self.inside_number: dict[str, str] = {}
        self.by_slug: dict[str, list[str]] = {}
        self.first: str | None = rows[0]["chunk_id"] if rows else None
        for row in rows:
            path = json.loads(row["heading_path"])
            last = _base_heading(path[-1])
            number = _heading_number(last)
            if number:
                self.by_number.setdefault(number, []).append(row["chunk_id"])
            for heading in path[:-1]:
                parent = _heading_number(heading)
                if parent:
                    self.inside_number.setdefault(parent, row["chunk_id"])
            self.by_slug.setdefault(slugify(last), []).append(row["chunk_id"])

    def number(self, ref: str) -> list[str]:
        if ref in self.by_number:
            return self.by_number[ref]
        inside = self.inside_number.get(ref)   # "§6" → first chunk under heading 6
        return [inside] if inside else []


def _source_rows(conn: sqlite3.Connection, source: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT chunk_id, heading_path, content FROM chunks WHERE source_file=? "
        "ORDER BY position",
        (source,),
    ).fetchall()


def rebuild_structural_bonds(
    conn: sqlite3.Connection, source: str, config: MoleculeConfig
) -> int:
    """Recompute `ref` and `continuation` bonds that start in one source file.
    Called inside the ingest transaction; returns the number of bonds."""
    rows = _source_rows(conn, source)
    marks = ",".join("?" * len(STRUCTURAL_KINDS))
    conn.execute(
        f"DELETE FROM chunk_bonds WHERE kind IN ({marks}) AND src IN "
        "(SELECT chunk_id FROM chunks WHERE source_file=?)",
        (*STRUCTURAL_KINDS, source),
    )
    if not rows:
        return 0

    own = _SourceMap(rows)
    other_maps: dict[str, _SourceMap | None] = {}
    bonds: dict[tuple[str, str, str], float] = {}

    def other(target_file: str) -> _SourceMap | None:
        if target_file not in other_maps:
            target_rows = _source_rows(conn, target_file)
            other_maps[target_file] = _SourceMap(target_rows) if target_rows else None
        return other_maps[target_file]

    # Consecutive parts of one split section point at each other.
    by_base: dict[tuple[str, ...], list[str]] = {}
    for row in rows:
        path = json.loads(row["heading_path"])
        if re.search(r" §\d+", path[-1]):
            by_base.setdefault((*path[:-1], _base_heading(path[-1])), []).append(row["chunk_id"])
    for parts in by_base.values():
        for a, b in zip(parts, parts[1:]):
            bonds[(a, b, "continuation")] = config.bond_continuation_strength
            bonds[(b, a, "continuation")] = config.bond_continuation_strength

    # Explicit references in the text.
    for row in rows:
        src = row["chunk_id"]
        targets: list[str] = []
        for number in dict.fromkeys(_SECTION_REF_RE.findall(row["content"])):
            targets.extend(own.number(number))
        for link in dict.fromkeys(_LINK_RE.findall(row["content"])):
            if "://" in link or link.startswith("mailto:"):
                continue
            file_part, _, anchor = link.partition("#")
            if not file_part:
                targets.extend(own.by_slug.get(anchor.lower(), []))
                continue
            target_file = posixpath.normpath(
                posixpath.join(posixpath.dirname(source), file_part)
            )
            target_map = other(target_file)
            if target_map is None:
                continue
            hits = target_map.by_slug.get(anchor.lower(), []) if anchor else []
            targets.extend(hits or ([target_map.first] if target_map.first else []))
        for dst in list(dict.fromkeys(t for t in targets if t != src))[:_MAX_REFS_PER_CHUNK]:
            bonds[(src, dst, "ref")] = config.bond_ref_strength

    now = utc_now_iso()
    conn.executemany(
        "INSERT OR REPLACE INTO chunk_bonds (src, dst, kind, strength, hits, updated_at) "
        "VALUES (?,?,?,?,1,?)",
        [(src, dst, kind, strength, now) for (src, dst, kind), strength in bonds.items()],
    )
    return len(bonds)


def strengthen_cofetch(
    conn: sqlite3.Connection,
    chunk_ids: list[str],
    config: MoleculeConfig,
    partners: list[str] | tuple[str, ...] = (),
) -> int:
    """Hebbian update for one co-occurrence event.

    `chunk_ids` were just used together; `partners` were used shortly before
    (most recent first). Each pair *within* `chunk_ids`, and each of
    `chunk_ids` with each partner, moves one step toward 1. Pairs among the
    partners themselves are not touched again — that co-occurrence was
    already counted when it happened. Returns the number of bonds touched;
    the caller commits."""
    new = list(dict.fromkeys(chunk_ids))[:_MAX_COFETCH_GROUP]
    earlier = [p for p in dict.fromkeys(partners) if p not in new][:_MAX_COFETCH_GROUP]
    candidates = new + earlier
    if len(candidates) < 2 or not new:
        return 0
    marks = ",".join("?" * len(candidates))
    known = {
        r[0] for r in conn.execute(
            f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({marks})", candidates
        )
    }
    pairs = {tuple(sorted(pair)) for pair in combinations([i for i in new if i in known], 2)}
    pairs |= {
        tuple(sorted((a, b))) for a in new for b in earlier if a in known and b in known
    }
    rate = config.bond_learning_rate
    now = utc_now_iso()
    for a, b in sorted(pairs):
        conn.execute(
            "INSERT INTO chunk_bonds (src, dst, kind, strength, hits, updated_at) "
            "VALUES (?,?,'cofetch',?,1,?) "
            "ON CONFLICT(src, dst, kind) DO UPDATE SET "
            "strength = strength + ? * (1.0 - strength), hits = hits + 1, updated_at = ?",
            (a, b, rate, now, rate, now),
        )
    return len(pairs)


def _age_days(timestamp: str, now: datetime) -> float:
    try:
        then = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    return max(0.0, (now - then).total_seconds() / 86400.0)


def _build_neighbors(
    conn: sqlite3.Connection, config: MoleculeConfig
) -> dict[str, list[tuple[str, float, str]]]:
    now = utc_now()
    combined: dict[tuple[str, str], tuple[float, str]] = {}

    def add(a: str, b: str, strength: float, kind: str) -> None:
        old_strength, old_kind = combined.get((a, b), (0.0, kind))
        # Independent reasons to bond reinforce each other (noisy-or).
        merged = 1.0 - (1.0 - old_strength) * (1.0 - strength)
        combined[(a, b)] = (merged, kind if strength >= old_strength else old_kind)

    for row in conn.execute("SELECT src, dst, kind, strength, updated_at FROM chunk_bonds"):
        strength = float(row["strength"])
        if row["kind"] == "cofetch":
            strength *= 0.5 ** (_age_days(row["updated_at"], now) / config.bond_half_life_days)
            add(row["src"], row["dst"], strength, "cofetch")
            add(row["dst"], row["src"], strength, "cofetch")   # stored once, used both ways
        else:
            add(row["src"], row["dst"], strength, row["kind"])

    neighbors: dict[str, list[tuple[str, float, str]]] = {}
    for (a, b), (strength, kind) in combined.items():
        if strength >= config.bond_min_strength:
            neighbors.setdefault(a, []).append((b, round(strength, 4), kind))
    for links in neighbors.values():
        links.sort(key=lambda link: -link[1])
    return neighbors


def neighbors(
    conn: sqlite3.Connection, config: MoleculeConfig
) -> dict[str, list[tuple[str, float, str]]]:
    """chunk id -> [(bonded chunk id, strength, kind)], strongest first.
    Cached until a bond or chunk changes (and re-aged once a day)."""
    # The result also depends on two settings and on today's date (decay).
    salt = (utc_now().strftime("%Y-%m-%d"), config.bond_min_strength, config.bond_half_life_days)
    return versioned(
        conn, "bond_neighbors", lambda: _build_neighbors(conn, config),
        domains=("bonds", "chunks"), salt=salt,
    )


def related_for(
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    chunk_id: str,
    exclude: set[str] | frozenset[str] = frozenset(),
    limit: int | None = None,
) -> list[dict]:
    """Bonded neighbours of one chunk, as small dicts for tool results."""
    limit = config.related_k if limit is None else limit
    out = []
    for other, strength, kind in neighbors(conn, config).get(chunk_id, []):
        if other in exclude:
            continue
        out.append({"id": other, "bond": kind, "strength": strength})
        if len(out) >= limit:
            break
    return out


def bond_stats(conn: sqlite3.Connection) -> dict:
    """Counts per bond kind plus the strongest learned bonds."""
    counts = {
        r["kind"]: r["n"]
        for r in conn.execute("SELECT kind, COUNT(*) AS n FROM chunk_bonds GROUP BY kind")
    }
    learned = [
        {"a": r["src"], "b": r["dst"], "strength": round(r["strength"], 4), "hits": r["hits"]}
        for r in conn.execute(
            "SELECT src, dst, strength, hits FROM chunk_bonds WHERE kind='cofetch' "
            "ORDER BY strength DESC LIMIT 10"
        )
    ]
    return {"bonds": counts, "strongest_learned": learned}
