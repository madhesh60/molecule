from __future__ import annotations

import json

from molecule.config import load_config


def test_defaults(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.max_chunk_tokens == 400
    assert cfg.emb_weight == 0.6
    assert cfg.learned_weight_min == 0.5
    assert cfg.db_path == tmp_path / ".molecule" / "store.db"


def test_config_file_applies(tmp_path):
    (tmp_path / ".molecule").mkdir()
    (tmp_path / ".molecule" / "config.json").write_text(
        json.dumps({"max_chunk_tokens": 123, "heading_boost": 0.2})
    )
    cfg = load_config(tmp_path)
    assert cfg.max_chunk_tokens == 123
    assert cfg.heading_boost == 0.2


def test_config_file_cannot_override_db_path(tmp_path):
    (tmp_path / ".molecule").mkdir()
    (tmp_path / ".molecule" / "config.json").write_text(
        json.dumps({"db_path": "/somewhere/else.db"})
    )
    cfg = load_config(tmp_path)
    assert cfg.db_path == tmp_path / ".molecule" / "store.db"


def test_env_wins_over_config_file(tmp_path, monkeypatch):
    (tmp_path / ".molecule").mkdir()
    (tmp_path / ".molecule" / "config.json").write_text(
        json.dumps({"max_chunk_tokens": 123})
    )
    monkeypatch.setenv("MOLECULE_MAX_CHUNK_TOKENS", "999")
    cfg = load_config(tmp_path)
    assert cfg.max_chunk_tokens == 999


def test_malformed_config_file_is_ignored(tmp_path):
    (tmp_path / ".molecule").mkdir()
    (tmp_path / ".molecule" / "config.json").write_text("{not json")
    cfg = load_config(tmp_path)  # must not raise
    assert cfg.max_chunk_tokens == 400


def test_unknown_keys_ignored(tmp_path):
    (tmp_path / ".molecule").mkdir()
    (tmp_path / ".molecule" / "config.json").write_text(
        json.dumps({"no_such_key": 1, "default_k": 3})
    )
    cfg = load_config(tmp_path)
    assert cfg.default_k == 3
