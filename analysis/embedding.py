"""Embedding 服务（pipeline.md 十一.3 L903-937 / scaf.md 11 节，裁决 C10）。

- 主路径：sentence-transformers（C10 初值：all-MiniLM-L6-v2），懒加载
- 降级路径：HashEmbedder（确定性字符 n-gram 向量，仅测试/冒烟用，标注降级）
- 缓存：results/embeddings/，以文本 hash 为键（避免数千次重复编码）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_DIM = 256


def _model_cached(model_name: str) -> bool:
    """该 HF 模型是否已有本地快照（snapshots/*/model.safetensors）。"""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = model_name.replace("/", "--")
        snap = Path(HF_HUB_CACHE) / ("models--" + repo) / "snapshots"
        return snap.is_dir() and any(snap.glob("*/model.safetensors"))
    except Exception:
        return False


def _ensure_local_hf_offline(model_name: str) -> None:
    """若该句向量模型已在本地 HF 缓存，则置离线环境，使其不再发网络请求。

    sentence-transformers 即使命中缓存也会先做一轮 hub HEAD（adapter_config.json 等）
    探测；离线数据/评测机上会抛 getaddrinfo/连接失败。已探到本地快照即 HF_HUB_OFFLINE=1。
    """
    try:
        import os

        if _model_cached(model_name):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    except Exception:
        pass  # 命中异常不阻断，交由降级层处理


class HashEmbedder:
    """确定性降级 embedder：字符 n-gram 哈希向量（非语义，仅链路测试）。"""

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
        self._embedder = embedder  # 测试注入
        self.degraded = False  # 若 sentence-transformers 不可用则置 True（保/复现可查）

    @staticmethod
    def _pick_device() -> str:
        """有可用 CUDA 用 cuda，否则 cpu（analyze 大量文本编码时显著提速）。"""
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
                # 本地有快照则 local_files_only=True，杜绝离线环境对 HF 发 HEAD/GET；
                # device 自动选择：云端有 GPU 用 GPU 编码（26070 条文本提速数十倍）
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
        """批量编码，命中缓存即复用（response 内容 hash 为键）。"""
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
    """文本缓存键。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_stats(cache_dir: str = "results/embeddings") -> dict:
    """缓存统计（性能预检用）。"""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {"files": 0}
    return {"files": len(list(cache_dir.glob("*.npy")))}
