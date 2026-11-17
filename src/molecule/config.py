"""config.py — every tunable in one typed place.

Three layers, later wins:  defaults  <  .molecule/config.json  <  environment.

Any field can be overridden with an env var named `MOLECULE_<FIELD>` in upper
case (`MOLECULE_DEFAULT_K=3`, `MOLECULE_EMBEDDING_MODEL=hashing`, ...). Bad
values never crash start-up: they are reported on stderr and the default is
kept, because a context layer that refuses to boot is worse than one running
on defaults.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass
class MoleculeConfig:
    db_path: Path = Path(".molecule/store.db")

    # --- Chunking -----------------------------------------------------------
    max_chunk_tokens: int = 400
    min_chunk_tokens: int = 50
    heading_split_depth: int = 3          # H1..H3 start a new chunk; deeper stay inline
    max_file_bytes: int = 10_000_000      # refuse to ingest files larger than this
    max_ingest_files: int = 2000          # cap for directory ingest

    # --- Embeddings ---------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"   # or "hashing" for zero-download mode
    embedding_backend: str = "auto"       # auto | onnx | torch | hashing
    onnx_model_file: str = "onnx/model.onnx"
    chunk_embed_tokens: int = 256         # wordpieces of heading+summary+content per chunk vector
    passage_embed_tokens: int = 64        # wordpieces per passage (sentence) vector
    # Also store one vector per sentence at ingest and let the best sentence
    # lift its chunk's rank. Measured: ~+1% top-3 for ~3x the ingest time, so
    # it is off by default (mol_context embeds passages on demand instead).
    passage_vectors: bool = False
    max_passages_per_chunk: int = 32

    # --- Retrieval ----------------------------------------------------------
    default_k: int = 5
    emb_weight: float = 0.6               # dense (meaning) signal
    keyword_weight: float = 0.6           # coverage-normalised BM25 signal
    heading_boost: float = 0.15           # query terms that appear in the heading
    passage_weight: float = 0.4           # share of dense taken from the best sentence, if stored
    query_expansion: bool = True          # concept-cluster synonyms at reduced weight
    always_include_weight: float = 2.0
    relevance_high: float = 0.45          # evidence at/above this → "high"
    relevance_low: float = 0.25           # evidence below this → "low" (probably not covered)

    # --- Self-learning router (mol_feedback) ---------------------------------
    feedback_ema_alpha: float = 0.2
    learned_weight_min: float = 0.5
    learned_weight_max: float = 1.5
    feedback_kernel_floor: float = 0.3    # query similarity below this → vote does not apply
    feedback_kernel_full: float = 0.85    # similarity at/above this → vote applies in full
    feedback_global_leak: float = 0.1     # tiny share of every vote that applies to all queries
    feedback_window: int = 64             # most recent votes kept per chunk

    # --- Bonds (chunk graph) -------------------------------------------------
    bond_ref_strength: float = 0.5        # explicit cross-reference in the text
    bond_continuation_strength: float = 0.6   # consecutive parts of one split section
    bond_learning_rate: float = 0.2       # Hebbian step for co-fetched chunks
    bond_half_life_days: float = 30.0     # learned bonds fade when not reinforced
    bond_min_strength: float = 0.25       # weaker bonds are ignored
    cofetch_window_s: float = 180.0       # fetches this close together count as "together"
    related_k: int = 3                    # bonded neighbours listed per search result

    # --- Context compiler (mol_context) --------------------------------------
    context_budget: int = 1200            # default token budget
    context_min_score_ratio: float = 0.6  # chunks scoring below ratio × best are left out
    context_excerpt: str = "head"         # how an overflowing chunk is cut: head | focus
    context_min_excerpt_tokens: int = 60  # never squeeze a chunk below this
    context_max_memories: int = 3
    index_token_budget: int = 400         # mol_load_index always fits in this

    # --- Memory --------------------------------------------------------------
    dedup_threshold: float = 0.92
    conflict_threshold: float = 0.85
    task_state_expiry_days: int = 7
    low_confidence_threshold: float = 0.7
    extract_max_facts: int = 5
    max_active_memories: int = 5000       # least-valuable memories are evicted past this

    # --- Telemetry -----------------------------------------------------------
    cost_per_mtok_usd: float = 3.0        # $ per million input tokens, for the savings estimate

    # --- Freshness -----------------------------------------------------------
    auto_sync: bool = True                # re-ingest edited source files automatically
    auto_sync_interval_s: float = 5.0     # how often to stat() the source files

    # --- Logging -------------------------------------------------------------
    log_to_file: bool = True

    @property
    def root(self) -> Path:
        """The repo the store belongs to (`<root>/.molecule/store.db`)."""
        return self.db_path.parent.parent

    def problems(self) -> list[str]:
        """Human-readable list of invalid settings (empty when all is well)."""
        issues: list[str] = []

        def check(ok: bool, message: str) -> None:
            if not ok:
                issues.append(message)

        check(self.max_chunk_tokens >= 32, "max_chunk_tokens must be >= 32")
        check(0 <= self.min_chunk_tokens < self.max_chunk_tokens,
              "min_chunk_tokens must be >= 0 and smaller than max_chunk_tokens")
        check(1 <= self.heading_split_depth <= 6, "heading_split_depth must be 1..6")
        check(self.default_k >= 1, "default_k must be >= 1")
        check(self.chunk_embed_tokens >= 16, "chunk_embed_tokens must be >= 16")
        check(self.passage_embed_tokens >= 16, "passage_embed_tokens must be >= 16")
        for name in ("emb_weight", "keyword_weight", "heading_boost"):
            check(getattr(self, name) >= 0, f"{name} must be >= 0")
        check(self.emb_weight + self.keyword_weight > 0,
              "emb_weight and keyword_weight cannot both be 0")
        check(0 <= self.passage_weight <= 1, "passage_weight must be in [0, 1]")
        check(0 < self.feedback_ema_alpha <= 1, "feedback_ema_alpha must be in (0, 1]")
        check(0 < self.learned_weight_min <= 1 <= self.learned_weight_max,
              "need learned_weight_min <= 1 <= learned_weight_max, with min > 0")
        check(0 <= self.feedback_kernel_floor < self.feedback_kernel_full <= 1,
              "need 0 <= feedback_kernel_floor < feedback_kernel_full <= 1")
        check(0 < self.dedup_threshold <= 1, "dedup_threshold must be in (0, 1]")
        check(0 < self.conflict_threshold <= 1, "conflict_threshold must be in (0, 1]")
        check(self.context_budget >= 64, "context_budget must be >= 64")
        check(self.index_token_budget >= 32, "index_token_budget must be >= 32")
        check(self.context_excerpt in {"head", "focus"},
              "context_excerpt must be 'head' or 'focus'")
        check(self.embedding_backend in {"auto", "onnx", "torch", "hashing"},
              "embedding_backend must be one of auto, onnx, torch, hashing")
        return issues

    def validate(self) -> MoleculeConfig:
        """Raise ValueError when a setting is out of range; return self otherwise."""
        issues = self.problems()
        if issues:
            raise ValueError("invalid Molecule configuration: " + "; ".join(issues))
        return self


def _coerce(current: Any, raw: Any) -> Any:
    """Convert `raw` (from JSON or an env string) to the type of `current`."""
    if isinstance(current, bool):
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, Path):
        return Path(str(raw))
    return str(raw)


def _warn(message: str) -> None:
    print(f"molecule: {message}", file=sys.stderr)


def _apply(cfg: MoleculeConfig, key: str, raw: Any, origin: str) -> None:
    """Set one field from an override, keeping the old value on bad input."""
    try:
        setattr(cfg, key, _coerce(getattr(cfg, key), raw))
    except (TypeError, ValueError) as exc:
        _warn(f"ignoring {origin} {key}={raw!r}: {exc}")


def _apply_config_file(cfg: MoleculeConfig, root: Path) -> None:
    """Apply .molecule/config.json overrides. Malformed files warn, never crash."""
    path = root / ".molecule" / "config.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config.json must contain a JSON object")
    except (ValueError, OSError) as exc:
        _warn(f"ignoring invalid {path}: {exc}")
        return
    known = {f.name for f in fields(cfg)}
    for key, value in data.items():
        # db_path stays derived from the repo root — never file-overridable.
        if key == "db_path" or key not in known:
            continue
        _apply(cfg, key, value, "config.json")


def _apply_env(cfg: MoleculeConfig) -> None:
    """Apply MOLECULE_<FIELD> overrides — useful for Docker / CI."""
    for f in fields(cfg):
        if f.name == "db_path":
            continue
        raw = os.environ.get(f"MOLECULE_{f.name.upper()}")
        if raw is not None and raw != "":
            _apply(cfg, f.name, raw, "env")


def load_config(repo_root: Path | None = None) -> MoleculeConfig:
    """Build the effective config for a repo (defaults < config.json < env)."""
    root = repo_root or Path(os.environ.get("MOLECULE_ROOT", str(Path.cwd())))
    cfg = MoleculeConfig(db_path=Path(root) / ".molecule" / "store.db")
    _apply_config_file(cfg, Path(root))
    _apply_env(cfg)

    issues = cfg.problems()
    if issues:
        # Fall back to defaults for the whole config rather than run with a
        # combination that makes no sense; say exactly why.
        _warn("invalid settings, using defaults instead: " + "; ".join(issues))
        cfg = MoleculeConfig(db_path=cfg.db_path)
    return cfg
