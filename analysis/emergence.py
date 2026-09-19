"""Emergence Point 指标（pipeline.md 十一.2 L880-899 / scaf.md 11.2 L1134-1162）。

定义：第一个达到正确解释的 step（proj.md L706-734）；全部不达标返回 None。
比较机制：embedding 相似度 > threshold（裁决 C10，与 accuracy 同一后端）。
支持 gold 多参考（gold_points 列表）：任一参考达标即视为该 step 正确（P1-4）。
"""

from __future__ import annotations

from analysis.accuracy import best_similarity
from analysis.confidence import extract_interpretation


def calculate_emergence(records: list[dict], gold, service, threshold: float, *,
                        judge_fn=None, question: str | None = None):
    """records 按 step 升序；返回首个达标 step，否则 None。

    judge_fn 给定则以 judge 等价判定首个达标 step（P1-4 完整路线）；否则余弦 > threshold。
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
