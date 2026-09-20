"""Statistical analysis (proj.md section 10 L831-868 / scaf.md section 12 L1197-1252; extended by ruling C12).

Formulas (proj.md L854/L862 + C12):
    Emergence ~ Model + Task + ModelSize + (1|Story)
    Confidence ~ SentencePosition + Model + Task + (1|Story)
    Stability ~ Model + Task + SentencePosition + (1|Story)
Depends on statsmodels (basis missing; supplementary convention); when it is not installed or
there is not enough data, a WARN is logged, the fit is skipped and None is returned.
"""

from __future__ import annotations

import numpy as np


def run_statistics(metrics_df, logger=None):
    """Fit the mixed-effects models; returns {formula: summary}; None when unavailable."""
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
    # fix: statsmodels MixedLM does not accept lme4's (1|group) random-effect syntax (patsy treats |
    # as bitwise or and raises TypeError); the random effect is specified via groups=, so the
    # formulas keep fixed-effect terms only.
    # note: emergence does not include C(size): when the model set makes model and size collinear
    # (e.g. only ds7b/ds14b), the design matrix is singular; size effects are covered by the
    # dedicated size_centered model in scripts/mixed_effects.py.
    formulas = {
        "emergence": "emergence_point ~ C(model) + C(task)",
        "confidence": "confidence ~ C(sentence_position) + C(model) + C(task)",
        "stability": "stability ~ C(model) + C(task)",
    }
    for name, formula in formulas.items():
        # fix: check against the real column name (emergence uses emergence_point), otherwise the model would be skipped forever
        col = {"emergence": "emergence_point"}.get(name, name)
        if col not in metrics_df.columns:
            continue
        # fix: emergence_point can be empty (no emergence) → NaN; MixedLM does not drop these automatically, so dropna explicitly
        sub = metrics_df.dropna(subset=[col])
        if len(sub) < 10:
            if logger is not None:
                logger.warning(f"statistics: {name} insufficient non-missing rows, skip MixedLM")
            continue
        try:
            model = smf.mixedlm(formula, sub, groups=sub["story_id"])
            fit = model.fit(reml=True)
            results[name] = str(fit.summary())
        except Exception as exc:  # noqa: BLE001 - a failed fit must not block the pipeline
            if logger is not None:
                logger.warning(f"statistics: {name} model fit failed: {exc}")
    return results or None


def trajectory_similarity(records_a: list[dict], records_b: list[dict], service) -> float:
    """Cross-model trajectory similarity (proj.md metric 5 L807-829): mean cosine over per-step interpretations."""
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
