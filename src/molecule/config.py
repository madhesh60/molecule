from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MoleculeConfig:
    db_path: Path = Path(".molecule/store.db")
    max_chunk_tokens: int = 400
    min_chunk_tokens: int = 50
    default_k: int = 5
    dedup_threshold: float = 0.92
    conflict_threshold: float = 0.85
    embedding_model: str = "all-MiniLM-L6-v2"
    always_include_weight: float = 2.0
    task_state_expiry_days: int = 7
    low_confidence_threshold: float = 0.7
    # Retrieval scoring
    emb_weight: float = 0.6
    keyword_weight: float = 0.3
    heading_boost: float = 0.15
    # Self-learning router (mol_feedback)
    feedback_ema_alpha: float = 0.2
    learned_weight_min: float = 0.5
    learned_weight_max: float = 1.5
    # Auto memory extraction (mem_extract)
    extract_max_facts: int = 5
    # Logging
    log_to_file: bool = True


def _apply_config_file(cfg: MoleculeConfig, root: Path) -> None:
    """Apply .molecule/config.json overrides. Malformed files warn, never crash."""
    path = root / ".molecule" / "config.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config.json must contain a JSON object")
        for key, value in data.items():
            # db_path stays derived from the repo root — never file-overridable.
            if key == "db_path" or not hasattr(cfg, key):
                continue
            current = getattr(cfg, key)
            setattr(cfg, key, type(current)(value))
    except (ValueError, TypeError, OSError) as exc:
        print(f"molecule: ignoring invalid {path}: {exc}", file=sys.stderr)


def load_config(repo_root: Path | None = None) -> MoleculeConfig:
    root = repo_root or Path(os.environ.get("MOLECULE_ROOT", str(Path.cwd())))
    cfg = MoleculeConfig(db_path=root / ".molecule" / "store.db")

    # Precedence: defaults < .molecule/config.json < env vars.
    _apply_config_file(cfg, root)

    # Env var overrides — useful for Docker / CI
    if v := os.environ.get("MOLECULE_MAX_CHUNK_TOKENS"):
        cfg.max_chunk_tokens = int(v)
    if v := os.environ.get("MOLECULE_MIN_CHUNK_TOKENS"):
        cfg.min_chunk_tokens = int(v)
    if v := os.environ.get("MOLECULE_DEFAULT_K"):
        cfg.default_k = int(v)
    if v := os.environ.get("MOLECULE_EMBEDDING_MODEL"):
        cfg.embedding_model = v
    if v := os.environ.get("MOLECULE_DEDUP_THRESHOLD"):
        cfg.dedup_threshold = float(v)

    return cfg
