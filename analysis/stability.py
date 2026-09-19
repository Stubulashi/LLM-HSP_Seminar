"""Stability 指标（scaf.md 11.3 L1166-1193，裁决 C12 定义）。

Stability = 相邻 step 解释的 embedding 余弦距离均值（即 trajectory 语义，
对应 pipeline.md 十一.3 的 Semantic Trajectory；产出 trajectory.png）。
"""

from __future__ import annotations

import numpy as np

from analysis.confidence import extract_interpretation


def calculate_stability(records: list[dict], service) -> float:
    """返回相邻 step 距离均值；不足 2 步返回 0.0。"""
    ordered = sorted(records, key=lambda r: r["step"])
    if len(ordered) < 2:
        return 0.0
    texts = [extract_interpretation(r["response"]) for r in ordered]
    vectors = service.encode(texts)
    distances = []
    for i in range(len(vectors) - 1):
        a, b = vectors[i], vectors[i + 1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            continue
        distances.append(1.0 - float(np.dot(a, b) / denom))  # 余弦距离
    return float(np.mean(distances)) if distances else 0.0


def step_distances(records: list[dict], service) -> list[tuple[int, float]]:
    """相邻 step 距离序列（trajectory.png 绘图数据）：[(step, distance)]。"""
    ordered = sorted(records, key=lambda r: r["step"])
    texts = [extract_interpretation(r["response"]) for r in ordered]
    vectors = service.encode(texts)
    out = []
    for i in range(len(vectors) - 1):
        a, b = vectors[i], vectors[i + 1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            continue
        out.append((ordered[i + 1]["step"], 1.0 - float(np.dot(a, b) / denom)))
    return out
