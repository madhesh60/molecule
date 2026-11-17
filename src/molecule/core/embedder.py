"""embedder.py — local embedding backends behind one small interface.

Three backends, same contract (`encode(texts, max_tokens) -> unit vectors`):

  * **onnx** (default) — runs the model with ONNX Runtime + `tokenizers`.
    Same weights and the same vectors as sentence-transformers (cosine
    1.0000 on our corpus), but the install is ~80 MB instead of several GB
    of PyTorch, and a cold start takes ~0.15 s instead of ~4 s.
  * **torch** — sentence-transformers, for models that ship no ONNX export
    (`pip install molecule-mcp[torch]`).
  * **hashing** — a dependency-free feature-hashing embedder. No download,
    no model, fully deterministic. Weaker than a neural model on meaning,
    fine on wording; used for zero-setup mode and for hermetic tests.

Every stored vector is tagged with `Embedder.name`. If the configured model
changes, the stale vectors are re-embedded instead of being compared across
two unrelated vector spaces (see `heal_embeddings`).

Models are looked up offline first (`local_files_only`) so a machine that
ran `molecule setup` once never touches the network again.
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import zlib
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from molecule.db import content_hash as _hash

if TYPE_CHECKING:
    from molecule.config import MoleculeConfig

_log = logging.getLogger("molecule")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_ONNX_FILE = "onnx/model.onnx"
HASHING_DIM = 384
_BATCH_SIZE = 16   # small batches of similar length beat big padded ones on CPU
_DEFAULT_MAX_TOKENS = 128

# Pooling and prompt conventions for models known to differ from the
# sentence-transformers default (mean pooling, no prompts).
_PRESETS: dict[str, dict[str, str]] = {
    "BAAI/bge-small-en-v1.5": {
        "pooling": "cls",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "intfloat/e5-small-v2": {"query_prefix": "query: ", "passage_prefix": "passage: "},
}


class EmbeddingUnavailableError(RuntimeError):
    """The configured embedding model could not be loaded."""


class Embedder(Protocol):
    name: str   # identity stored next to every vector this backend produces
    dim: int

    def encode(
        self, texts: Sequence[str], max_tokens: int = _DEFAULT_MAX_TOKENS, query: bool = False
    ) -> np.ndarray:
        """Return L2-normalised float32 vectors, one row per text."""
        ...


def _empty(dim: int) -> np.ndarray:
    return np.zeros((0, dim), dtype=np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)


# --- ONNX Runtime backend ------------------------------------------------------

def _hf_file(repo: str, filename: str) -> str:
    """Path of a model file: the local cache first, the network only if needed."""
    local_dir = Path(repo)
    if local_dir.is_dir():
        path = local_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached yet → fetch once
        return hf_hub_download(repo, filename)


class OnnxEmbedder:
    """A sentence-embedding model run with ONNX Runtime on the CPU."""

    def __init__(self, model_name: str, onnx_file: str = DEFAULT_ONNX_FILE) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        repo = model_name if "/" in model_name or Path(model_name).is_dir() else (
            f"sentence-transformers/{model_name}"
        )
        preset = _PRESETS.get(repo, {})
        self._pooling = preset.get("pooling", "mean")
        self._query_prefix = preset.get("query_prefix", "")
        self._passage_prefix = preset.get("passage_prefix", "")

        self._tokenizer = Tokenizer.from_file(_hf_file(repo, "tokenizer.json"))
        pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.log_severity_level = 3
        threads = int(os.environ.get("MOLECULE_EMBED_THREADS", "0"))
        if threads > 0:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            _hf_file(repo, onnx_file), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._lock = threading.Lock()  # tokenizer truncation is shared state

        # The default export produces the reference vectors; any other file
        # (e.g. an int8-quantised one) is a slightly different vector space.
        variant = "" if onnx_file == DEFAULT_ONNX_FILE else f"+{Path(onnx_file).stem}"
        self.name = f"{model_name}{variant}"
        self.dim = int(self.encode(["dimension probe"], 8).shape[1])

    def encode(
        self, texts: Sequence[str], max_tokens: int = _DEFAULT_MAX_TOKENS, query: bool = False
    ) -> np.ndarray:
        if not texts:
            return _empty(getattr(self, "dim", 0))
        prefix = self._query_prefix if query else self._passage_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        # Longest first: each batch then holds similar lengths → minimal padding.
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out: np.ndarray | None = None
        with self._lock:
            self._tokenizer.enable_truncation(max_length=max(8, int(max_tokens)))
            for start in range(0, len(order), _BATCH_SIZE):
                batch = order[start : start + _BATCH_SIZE]
                encodings = self._tokenizer.encode_batch([texts[i] for i in batch])
                ids = np.array([e.ids for e in encodings], dtype=np.int64)
                mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
                feed = {"input_ids": ids, "attention_mask": mask}
                if "token_type_ids" in self._inputs:
                    feed["token_type_ids"] = np.zeros_like(ids)
                hidden = self._session.run(None, feed)[0]
                if self._pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    weights = mask[..., None].astype(np.float32)
                    pooled = (hidden * weights).sum(axis=1) / np.clip(
                        weights.sum(axis=1), 1e-9, None
                    )
                if out is None:
                    out = np.zeros((len(texts), pooled.shape[1]), dtype=np.float32)
                out[batch] = pooled
        assert out is not None
        return _normalize(out)


# --- sentence-transformers backend ----------------------------------------------

class TorchEmbedder:
    """sentence-transformers backend for models without an ONNX export."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._native_max = int(self._model.max_seq_length or 256)
        self._lock = threading.Lock()
        self.name = model_name
        self.dim = int(self.encode(["dimension probe"], 8).shape[1])

    def encode(
        self, texts: Sequence[str], max_tokens: int = _DEFAULT_MAX_TOKENS, query: bool = False
    ) -> np.ndarray:
        if not texts:
            return _empty(getattr(self, "dim", 0))
        with self._lock:
            self._model.max_seq_length = min(self._native_max, max(8, int(max_tokens)))
            vectors = self._model.encode(
                list(texts), batch_size=_BATCH_SIZE, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
        return vectors.astype(np.float32)


# --- Hashing backend (no model, no download) -------------------------------------

class HashingEmbedder:
    """Signed feature hashing over stems, stem bigrams and concept clusters.

    A text becomes a bag of features; each feature is hashed to one of `dim`
    buckets with a ±1 sign (so unrelated features cancel instead of piling
    up), weighted by a damped term frequency, then L2-normalised. Two texts
    that share words — or words from the same concept cluster — end up with
    a high cosine. Deterministic across machines and Python versions
    (crc32, not the salted built-in hash)."""

    def __init__(self, dim: int = HASHING_DIM) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> Counter:
        from molecule.core.expansion import cluster_id
        from molecule.core.scoring import tokenize

        stems = tokenize(text)
        features: Counter = Counter()
        for stem in stems:
            features[stem] += 1.0
            cluster = cluster_id(stem)
            if cluster is not None:
                features[f"\x00cluster:{cluster}"] += 0.6
        for a, b in zip(stems, stems[1:]):
            features[f"{a}\x01{b}"] += 0.5
        return features

    def encode(
        self, texts: Sequence[str], max_tokens: int = _DEFAULT_MAX_TOKENS, query: bool = False
    ) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text).items():
                h = zlib.crc32(feature.encode("utf-8"))
                sign = 1.0 if (h >> 16) & 1 else -1.0
                # Damped term frequency: the 5th repeat matters less than the 1st.
                damped = 1.0 + math.log(weight) if weight >= 1.0 else weight
                out[row, h % self.dim] += sign * damped
        return _normalize(out)


# --- Backend selection -----------------------------------------------------------

def _is_hashing(model_name: str, backend: str) -> bool:
    return backend == "hashing" or model_name.lower().startswith("hashing")


@lru_cache(maxsize=4)
def get_embedder(
    model_name: str = DEFAULT_MODEL,
    backend: str = "auto",
    onnx_file: str = DEFAULT_ONNX_FILE,
) -> Embedder:
    """Load (and cache) the embedding backend for a model."""
    if _is_hashing(model_name, backend):
        return HashingEmbedder()

    errors: list[str] = []
    if backend in ("auto", "onnx"):
        try:
            return OnnxEmbedder(model_name, onnx_file)
        except Exception as exc:  # noqa: BLE001 — try the next backend, report all
            errors.append(f"onnx: {type(exc).__name__}: {exc}")
    if backend in ("auto", "torch"):
        try:
            return TorchEmbedder(model_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"torch: {type(exc).__name__}: {exc}")

    raise EmbeddingUnavailableError(
        f"could not load embedding model {model_name!r} ({'; '.join(errors)}). "
        "Run `molecule setup` once while online to cache it, or set "
        "MOLECULE_EMBEDDING_MODEL=hashing to run with the built-in zero-download embedder."
    )


def embedder_for(config: MoleculeConfig) -> Embedder:
    """The embedding backend selected by a config."""
    return get_embedder(config.embedding_model, config.embedding_backend, config.onnx_model_file)


def embedder_name(config: MoleculeConfig) -> str:
    """Identity of the configured backend — computed without loading it, so
    cheap checks ("is this index still current?") never pay for a model load."""
    if _is_hashing(config.embedding_model, config.embedding_backend):
        return f"hashing-{HASHING_DIM}"
    quantised = (
        config.embedding_backend != "torch" and config.onnx_model_file != DEFAULT_ONNX_FILE
    )
    variant = f"+{Path(config.onnx_model_file).stem}" if quantised else ""
    return f"{config.embedding_model}{variant}"


def get_model(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Load (and cache) the embedding model — call once to warm the cache."""
    return get_embedder(model_name)


def embed(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> np.ndarray:
    """Embed texts with the named model (auto-selected backend)."""
    return get_embedder(model_name).encode(texts, max_tokens)


@lru_cache(maxsize=512)
def _embed_query_cached(text: str, embedder: Any, max_tokens: int) -> bytes:
    # bytes are hashable/immutable — safe to share from an lru_cache.
    return embedder.encode([text], max_tokens, query=True)[0].tobytes()


def embed_query(
    text: str, model: str | Embedder = DEFAULT_MODEL, max_tokens: int = _DEFAULT_MAX_TOKENS
) -> np.ndarray:
    """Embed a single query string with an LRU cache.

    Model inference dominates search latency; agents often retry or
    paraphrase-loop the same query, so repeats become free. Returns a
    read-only view over the cached bytes."""
    embedder = get_embedder(model) if isinstance(model, str) else model
    return np.frombuffer(_embed_query_cached(text, embedder, max_tokens), dtype=np.float32)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine of `query` against each row — both sides are already unit vectors."""
    return (matrix @ query).astype(np.float32)


# --- Storage helpers ---------------------------------------------------------------

def embed_and_store(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    row_id: str,
    text: str,
    model_name: str,
) -> None:
    """Embed one row's text and store it — skipped when the hash is unchanged."""
    h = _hash(text)
    existing = conn.execute(
        f"SELECT content_hash, embedding FROM {table} WHERE {id_col} = ?", (row_id,)
    ).fetchone()
    # Skip only when hash matches AND embedding already exists
    if existing and existing["content_hash"] == h and existing["embedding"] is not None:
        return
    vec = embed([text], model_name)[0]
    conn.execute(
        f"UPDATE {table} SET embedding = ?, content_hash = ? WHERE {id_col} = ?",
        (vec.tobytes(), h, row_id),
    )
    conn.commit()


def load_embeddings(
    conn: sqlite3.Connection, table: str, id_col: str, filter_clause: str = ""
) -> tuple[list[str], np.ndarray]:
    """All stored vectors of a table as (ids, matrix). Rows whose vector has a
    different width than the first one are skipped rather than crashing."""
    sql = f"SELECT {id_col}, embedding FROM {table} WHERE embedding IS NOT NULL"
    if filter_clause:
        sql += f" AND {filter_clause}"
    rows = conn.execute(sql).fetchall()
    if not rows:
        return [], np.empty((0,), dtype=np.float32)
    width = len(rows[0][1])
    rows = [r for r in rows if len(r[1]) == width]
    matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    return [r[0] for r in rows], matrix


def vector_is_current(blob: bytes | None, tag: str | None, embedder: Embedder) -> bool:
    """True when a stored vector can be compared with this embedder's output.

    Untagged vectors (written by Molecule 0.1 or by hand) are trusted when
    their width matches; tagged ones must name the same model."""
    if blob is None or len(blob) != embedder.dim * 4:
        return False
    return tag is None or tag == embedder.name
