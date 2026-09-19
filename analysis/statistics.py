"""统计分析（proj.md 10 节 L831-868 / scaf.md 12 节 L1197-1252，裁决 C12 扩展）。

公式（proj.md L854/L862 + C12）：
    Emergence ~ Model + Task + ModelSize + (1|Story)
    Confidence ~ SentencePosition + Model + Task + (1|Story)
    Stability ~ Model + Task + SentencePosition + (1|Story)
依赖 statsmodels（依据缺失，补充约定）；未安装或数据不足时 WARN 跳过并返回 None。
"""

from __future__ import annotations

import numpy as np


def run_statistics(metrics_df, logger=None):
    """拟合混合效应模型；返回 {formula: summary}；不可用时返回 None。"""
    if metrics_df is None or len(metrics_df) < 10:
        if logger is not None:
            logger.warning("statistics: insufficient data (<10 rows), skip MixedLM")
        return None
    try:
        import statsmodels.formula.api as smf  # noqa: PLC0415
    except ImportError:
        if logger is not None:
            logger.warning("statistics: statsmodels not installed, skip MixedLM")
        return None

    results = {}
    # 修复：statsmodels MixedLM 不支持 lme4 的 (1|group) 随机效应语法（patsy 会把 | 当按位或
    # 导致 TypeError），随机效应已由 groups= 指定；公式中仅保留固定效应项。
    # 注：emergence 不放入 C(size)：当模型集合使 model↔size 共线（如仅 ds7b/ds14b）时
    # 设计矩阵奇异；规模效应由 scripts/mixed_effects.py 的 size_centered 专模覆盖。
    formulas = {
        "emergence": "emergence_point ~ C(model) + C(task)",
        "confidence": "confidence ~ C(sentence_position) + C(model) + C(task)",
        "stability": "stability ~ C(model) + C(task)",
    }
    for name, formula in formulas.items():
        # 修复：判断应针对实际列名（emergence 的列是 emergence_point），避免永久跳过该模型
        col = {"emergence": "emergence_point"}.get(name, name)
        if col not in metrics_df.columns:
            continue
        # 修复：emergence_point 可能为空（未涌现）→ NaN，MixedLM 不自动丢弃，需显式 dropna
        sub = metrics_df.dropna(subset=[col])
        if len(sub) < 10:
            if logger is not None:
                logger.warning(f"statistics: {name} insufficient non-missing rows, skip MixedLM")
            continue
        try:
            model = smf.mixedlm(formula, sub, groups=sub["story_id"])
            fit = model.fit(reml=True)
            results[name] = str(fit.summary())
        except Exception as exc:  # noqa: BLE001 - 拟合失败不阻塞管线
            if logger is not None:
                logger.warning(f"statistics: {name} model fit failed: {exc}")
    return results or None


def trajectory_similarity(records_a: list[dict], records_b: list[dict], service) -> float:
    """跨模型轨迹相似度（proj.md Metric 5 L807-829）：逐 step 解释 cosine 均值。"""
    by_step_a = {r["step"]: r for r in records_a}
    by_step_b = {r["step"]: r for r in records_b}
    common = sorted(set(by_step_a) & set(by_step_b))
    if not common:
        return 0.0
    sims = []
    for step in common:
        sims.append(service.cosine_similarity(
            by_step_a[step]["response"], by_step_b[step]["response"]
        ))
    return float(np.mean(sims)) if sims else 0.0
