"""Accuracy 指标（scaf.md 11.1 L1109-1130，裁决 C10：embedding 相似度 + threshold）。

设计说明（P1-4）：
- embedding_similarity 仅是评分方法之一（experiment.yaml `scoring.method` 预留 LLM
  judge 替换位）。它对“整段长 gold 散文 vs 模型整段 response”的比对很片面：长文本的
  cosine 被通用措辞重叠主导，0.7 固定阈值无校准依据，会产出接近噪声的 0/1。
- 本实现允许 gold 为“多参考关键点”（gold_points 列表）：存在时取与任一参考点的
  最大相似度，避免单一长 gold 使阈值失效；缺失时回退单条 gold_answer（旧行为）。
- 调用方应先对 response 做归一化（extract_interpretation），并把 threshold 在健康
  子集上校准，勿直接信任固定 0.7。
"""

from __future__ import annotations

from typing import Iterable

from analysis.confidence import extract_interpretation


def _references(gold) -> Iterable[str]:
    """将 gold_points 列表 / gold_answer 字符串统一为可遍历的参考文本。"""
    if isinstance(gold, (list, tuple)):
        return [g for g in gold if g]
    return [] if not gold else [gold]


def best_similarity(response: str, gold, service) -> float:
    """response 解释与任一 gold 参考点的最大余弦相似度（多参考时的稳健度量）。"""
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
    """accuracy：余弦 > threshold 记 1；若给 judge_fn 则以其等价判定为准（P1-4 完整路线）。

    judge_fn(question, answer_norm, gold) -> True/False/None；None 与 False 都记 0（无法判定不虚报）。
    """
    if judge_fn is not None:
        ans = extract_interpretation(response)
        label = judge_fn(question or "", ans, gold)
        return 1.0 if label is True else 0.0
    return 1.0 if best_similarity(response, gold, service) > threshold else 0.0
