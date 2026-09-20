"""Embedding service (pipeline.md 11.3 L903-937 / scaf.md section 11; ruling C10).

- main path: sentence-transformers (C10 initial value: all-MiniLM-L6-v2), lazily loaded
- degraded path: HashEmbedder (deterministic character n-gram vectors; tests/smoke only, flagged as degraded)
- cache: results/embeddings/, keyed by a hash of the text (avoids thousands of repeated encodings)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_DIM = 256


def _model_cached(model_name: str) -> bool:
    """Whether the HF model already has a local snapshot (snapshots/*/model.safetensors)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = model_name.replace("/", "--")
        snap = Path(HF_HUB_CACHE) / ("models--" + repo) / "snapshots"
        return snap.is_dir() and any(snap.glob("*/model.safetensors"))
    except Exception:
        return False


def _ensure_local_hf_offline(model_name: str) -> None:
    """If the sentence-embedding model is already in the local HF cache, switch to offline mode so no network requests are made.

    Even on a cache hit, sentence-transformers first performs a hub HEAD probe (adapter_config.json
    etc.); on offline data/evaluation machines this raises getaddrinfo / connection errors. Once a
    local snapshot is detected, HF_HUB_OFFLINE=1 is set.
    """
    try:
        import os

        if _model_cached(model_name):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    except Exception:
        pass  # exceptions here are not fatal; the degraded layer handles them


class HashEmbedder:
    """Deterministic degraded embedder: character n-gram hash vectors (non-semantic; pipeline tests only)."""

    def __init__(self, dim: int = _DIM):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            lowered = text.lower()
            for i in range(len(lowered) - 2):
                gram = lowered[i : i + 3]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
                vec[h % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        return np.stack(vectors)


class EmbeddingService:
    def __init__(self, model_name: str, cache_dir: str = "results/embeddings",
                 embedder=None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._st = None
        self._hash = None
        self._embedder = embedder  # injectable for tests
        self.degraded = False  # set to True when sentence-transformers is unavailable (visible in provenance/reproducibility checks)

    @staticmethod
    def _pick_device() -> str:
        """Use cuda when available, otherwise cpu (much faster when analyze encodes large amounts of text)."""
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _get_backend(self):
        if self._embedder is not None:
            return self._embedder
        if self._st is None and self._hash is None:
            local_only = _model_cached(self.model_name)
            _ensure_local_hf_offline(self.model_name)
            try:
                from sentence_transformers import SentenceTransformer

                device = self._pick_device()
                # with a local snapshot, local_files_only=True prevents any HF HEAD/GET in offline environments;
                # device is chosen automatically: encode on the GPU when the cloud machine has one (tens of times faster for 26,070 texts)
                self._st = SentenceTransformer(self.model_name,
                                               local_files_only=local_only,
                                               device=device)
            except ImportError:
                import logging
                self.degraded = True
                logging.getLogger("iple.embedding").warning(
                    "sentence-transformers unavailable; falling back to "
                    "HashEmbedder (deterministic n-gram, NOT semantic). "
                    "Similarity/confidence metrics are not research-valid here."
                )
                self._hash = HashEmbedder()
        return self._st if self._st is not None else self._hash

    def encode(self, texts: list[str]) -> np.ndarray:
        """Batch encoding with cache reuse (keyed by a hash of the response content)."""
        keys = [
            hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts
        ]
        cached: dict[str, np.ndarray] = {}
        missing: list[int] = []
        for i, key in enumerate(keys):
            path = self.cache_dir / f"{key}.npy"
            if path.exists():
                cached[i] = np.load(path)
            else:
                missing.append(i)

        if missing:
            backend = self._get_backend()
            encoded = backend.encode([texts[i] for i in missing])
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)
            for j, i in enumerate(missing):
                vec = np.asarray(encoded[j], dtype=np.float32)
                np.save(self.cache_dir / f"{keys[i]}.npy", vec)
                cached[i] = vec

        order = [cached[i] for i in range(len(texts))]
        return np.stack(order)

    def cosine_similarity(self, a: str, b: str) -> float:
        va, vb = self.encode([a, b])
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)


def text_hash(text: str) -> str:
    """Cache key for a text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_stats(cache_dir: str = "results/embeddings") -> dict:
    """Cache statistics (used by the performance pre-check)."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {"files": 0}
    return {"files": len(list(cache_dir.glob("*.npy")))}
