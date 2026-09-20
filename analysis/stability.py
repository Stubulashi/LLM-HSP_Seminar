"""Stability metric (scaf.md 11.3 L1166-1193; defined by ruling C12).

Stability = mean cosine distance between the interpretations of adjacent steps (the semantic
trajectory of pipeline.md 11.3; produces trajectory.png).
"""

from __future__ import annotations

import numpy as np

from analysis.confidence import extract_interpretation


def calculate_stability(records: list[dict], service) -> float:
    """Mean distance between adjacent steps; 0.0 when there are fewer than 2 steps."""
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
        distances.append(1.0 - float(np.dot(a, b) / denom))  # cosine distance
    return float(np.mean(distances)) if distances else 0.0


def step_distances(records: list[dict], service) -> list[tuple[int, float]]:
    """Sequence of adjacent-step distances (the plotting data for trajectory.png): [(step, distance)]."""
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
