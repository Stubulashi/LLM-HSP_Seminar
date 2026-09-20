"""Accuracy metric (scaf.md 11.1 L1109-1130; ruling C10: embedding similarity + threshold).

Design notes (P1-4):
- embedding_similarity is only one of the scoring methods (experiment.yaml `scoring.method`
  keeps a slot for an LLM judge replacement). Comparing a full long gold prose against a
  full model response is crude: the cosine of long texts is dominated by generic wording
  overlap, and the fixed 0.7 threshold has no calibration basis, producing near-noise 0/1.
- This implementation allows gold to be "multiple reference key points" (a gold_points list):
  when present, the maximum similarity to any reference is used, so a single long gold
  cannot void the threshold; when absent, it falls back to the single gold_answer (old behaviour).
- Callers should normalise the response first (extract_interpretation) and calibrate the
  threshold on a healthy subset; do not trust the fixed 0.7 as is.
"""

from __future__ import annotations

from typing import Iterable

from analysis.confidence import extract_interpretation


def _references(gold) -> Iterable[str]:
    """Normalise a gold_points list / gold_answer string into an iterable of reference texts."""
    if isinstance(gold, (list, tuple)):
        return [g for g in gold if g]
    return [] if not gold else [gold]


def best_similarity(response: str, gold, service) -> float:
    """Maximum cosine similarity between the response interpretation and any gold reference point (robust with multiple references)."""
    refs = _references(gold)
    if not refs:
        return 0.0
    sims = [
        service.cosine_similarity(extract_interpretation(response), ref)
        for ref in refs
    ]
    return max(sims)


def calculate_accuracy(response: str, gold, service, threshold: float, *,
                       judge_fn=None, question: str | None = None) -> float:
    """accuracy: cosine > threshold scores 1; when judge_fn is given, its equivalence decision takes precedence (the full P1-4 route).

    judge_fn(question, answer_norm, gold) -> True/False/None; both None and False score 0 (undecidable cases are not counted as correct).
    """
    if judge_fn is not None:
        ans = extract_interpretation(response)
        label = judge_fn(question or "", ans, gold)
        return 1.0 if label is True else 0.0
    return 1.0 if best_similarity(response, gold, service) > threshold else 0.0
