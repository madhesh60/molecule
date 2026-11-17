from __future__ import annotations

import json

import pytest

from molecule.config import MoleculeConfig, load_config


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


def test_any_field_can_be_set_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MOLECULE_KEYWORD_WEIGHT", "0.25")
    monkeypatch.setenv("MOLECULE_EMBEDDING_MODEL", "hashing")
    monkeypatch.setenv("MOLECULE_AUTO_SYNC", "false")
    cfg = load_config(tmp_path)
    assert cfg.keyword_weight == 0.25
    assert cfg.embedding_model == "hashing"
    assert cfg.auto_sync is False


@pytest.mark.parametrize("raw,expected", [("1", True), ("yes", True), ("off", False), ("0", False)])
def test_boolean_parsing(tmp_path, monkeypatch, raw, expected):
    monkeypatch.setenv("MOLECULE_QUERY_EXPANSION", raw)
    assert load_config(tmp_path).query_expansion is expected


def test_unparseable_env_value_keeps_the_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MOLECULE_DEFAULT_K", "many")
    cfg = load_config(tmp_path)
    assert cfg.default_k == 5
    assert "MOLECULE" not in capsys.readouterr().out  # warnings go to stderr only


def test_boolean_false_in_config_file_is_really_false(tmp_path):
    """bool("false") is True in Python — the loader must not fall for that."""
    (tmp_path / ".molecule").mkdir()
    (tmp_path / ".molecule" / "config.json").write_text(json.dumps({"auto_sync": "false"}))
    assert load_config(tmp_path).auto_sync is False


def test_invalid_combination_falls_back_to_defaults(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MOLECULE_MIN_CHUNK_TOKENS", "900")   # larger than max
    cfg = load_config(tmp_path)
    assert cfg.min_chunk_tokens == 50
    assert "min_chunk_tokens" in capsys.readouterr().err


def test_validate_raises_on_bad_values():
    with pytest.raises(ValueError, match="default_k"):
        MoleculeConfig(default_k=0).validate()
    with pytest.raises(ValueError, match="embedding_backend"):
        MoleculeConfig(embedding_backend="quantum").validate()


def test_root_is_derived_from_db_path(tmp_path):
    assert load_config(tmp_path).root == tmp_path
