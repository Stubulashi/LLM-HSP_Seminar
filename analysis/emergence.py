"""Emergence point metric (pipeline.md 11.2 L880-899 / scaf.md 11.2 L1134-1162).

Definition: the first step that reaches a correct interpretation (proj.md L706-734); None if none does.
Comparison: embedding similarity > threshold (ruling C10; same backend as accuracy).
Multiple gold references are supported (a gold_points list): reaching any reference counts the step as correct (P1-4).
"""

from __future__ import annotations

from analysis.accuracy import best_similarity
from analysis.confidence import extract_interpretation


def calculate_emergence(records: list[dict], gold, service, threshold: float, *,
                        judge_fn=None, question: str | None = None):
    """records sorted by step; returns the first passing step, or None.

    With judge_fn, the judge's equivalence decision picks the first passing step (the full P1-4 route); otherwise cosine > threshold.
    """
    ordered = sorted(records, key=lambda r: r["step"])
    for record in ordered:
        if judge_fn is not None:
            label = judge_fn(question or "", extract_interpretation(record["response"]), gold)
            if label is True:
                return record["step"]
        elif best_similarity(record["response"], gold, service) > threshold:
            return record["step"]
    return None
