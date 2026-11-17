"""Embedding backends: the hashing embedder, backend selection, model identity."""
from __future__ import annotations

import numpy as np
import pytest

from molecule.core.embedder import (
    DEFAULT_MODEL,
    EmbeddingUnavailableError,
    HashingEmbedder,
    cosine_similarity,
    embed_query,
    embedder_for,
    embedder_name,
    get_embedder,
    vector_is_current,
)
from tests.conftest import make_config


def test_hashing_vectors_are_unit_length_and_deterministic():
    embedder = HashingEmbedder()
    a = embedder.encode(["rollback a deploy", "structured json logs"])
    b = embedder.encode(["rollback a deploy", "structured json logs"])
    assert a.shape == (2, embedder.dim) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, b)


def test_hashing_similarity_follows_shared_wording():
    embedder = HashingEmbedder()
    query, same, other = embedder.encode([
        "how do I roll back a deploy",
        "roll back the deploy with deployctl",
        "bundle size budget for the dashboard",
    ])
    assert float(query @ same) > float(query @ other) + 0.2


def test_hashing_similarity_follows_concept_clusters():
    """'password' and 'secret' share no characters but share a cluster."""
    embedder = HashingEmbedder()
    a, b, c = embedder.encode(["rotate the password", "keep the secret safe", "paginate results"])
    assert float(a @ b) > float(a @ c)


def test_hashing_handles_empty_text():
    vectors = HashingEmbedder().encode(["", "   "])
    assert vectors.shape[0] == 2 and not np.isnan(vectors).any()


def test_hashing_backend_is_selected_by_name_or_backend():
    assert isinstance(get_embedder("hashing"), HashingEmbedder)
    assert isinstance(get_embedder(DEFAULT_MODEL, backend="hashing"), HashingEmbedder)


def test_embedder_name_needs_no_model_load(tmp_path):
    assert embedder_name(make_config(tmp_path)) == "hashing-384"
    assert embedder_name(make_config(tmp_path, embedding_model="all-MiniLM-L6-v2")) == (
        "all-MiniLM-L6-v2"
    )
    quantised = make_config(tmp_path, embedding_model="all-MiniLM-L6-v2",
                            onnx_model_file="onnx/model_quint8_avx2.onnx")
    assert embedder_name(quantised) == "all-MiniLM-L6-v2+model_quint8_avx2"


def test_unknown_model_fails_with_actionable_message(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(EmbeddingUnavailableError, match="molecule setup"):
        get_embedder("definitely-not-a-real-model-xyz", backend="onnx")


def test_query_embedding_is_cached(tmp_path):
    embedder = embedder_for(make_config(tmp_path))
    first = embed_query("the same question", embedder, 64)
    second = embed_query("the same question", embedder, 64)
    assert np.array_equal(first, second)
    assert not first.flags.writeable   # shared cache entry must be read-only


def test_vector_is_current_rules():
    embedder = HashingEmbedder()
    good = np.zeros(embedder.dim, dtype=np.float32).tobytes()
    assert vector_is_current(good, embedder.name, embedder)
    assert vector_is_current(good, None, embedder)             # untagged, right width
    assert not vector_is_current(good, "another-model", embedder)
    assert not vector_is_current(good[:16], None, embedder)    # wrong width
    assert not vector_is_current(None, None, embedder)


def test_cosine_similarity_of_unit_vectors():
    m = np.eye(3, dtype=np.float32)
    assert cosine_similarity(m[0], m).tolist() == [1.0, 0.0, 0.0]


# --- The real model (skipped when it is not cached) --------------------------------

@pytest.mark.model
def test_real_model_understands_paraphrase(real_model):
    query, same, other = real_model.encode([
        "how do I undo a bad release",
        "To roll back a service run deployctl rollback.",
        "The dashboard bundle must stay under 180 KB.",
    ])
    assert real_model.dim == 384
    assert float(query @ same) > float(query @ other) + 0.1


@pytest.mark.model
def test_real_model_truncation_changes_only_long_inputs(real_model):
    short = "A short sentence."
    assert np.allclose(real_model.encode([short], 16), real_model.encode([short], 128), atol=1e-5)
