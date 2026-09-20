"""Confidence curves and per-step modelling (proj.md statistics plan: Confidence ~ Position × Model + Task).

Reads results/raw/<model>/<task>/<story>/run*.json (the confidence self-report inside each record.response) and joins results/processed_judge/accuracy.csv (last-step decision):
  1) per-model confidence means by sentence-position quartile (Q1-Q4);
  2) MixedLM: conf ~ pos_frac + C(model) + C(task) (groups=story_id; skipped when statsmodels is missing);
Artifacts:
  results/processed_judge/confidence_by_position.csv (model × quartile means)
Usage: python -X utf8 scripts/confidence_analysis.py --models qwen7b,deepseek7b,qwen14b,deepseek14b,deepseek32b,qwen32b
"""
import argparse, csv, glob, json, os, statistics, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.confidence import extract_confidence

BASE = "results"
PJ = "results/processed_judge"


def _n_sent(sid):
    try:
        return len(json.load(open(f"data/annotated/{sid}.json", encoding="utf-8")).get("sentences", []))
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen7b,deepseek7b,qwen14b,deepseek14b,deepseek32b,qwen32b")
    ap.add_argument("--out", default=os.path.join(PJ, "confidence_by_position.csv"))
    args = ap.parse_args()
    keep = {m.strip() for m in args.models.split(",") if m.strip()}

    rows = []
    for p in sorted(glob.glob(os.path.join(BASE, "raw", "*", "*", "*", "run*.json"))):
        parts = os.path.normpath(p).split(os.sep)
        model, story = parts[2], parts[4]
        if model not in keep:
            continue
        try:
            ann = json.load(open(f"data/annotated/{story}.json", encoding="utf-8"))
        except Exception:
            continue
        task = ann.get("task")
        ns = len(ann.get("sentences", []))
        if ns <= 0:
            continue
        recs = [json.loads(l) for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
        for r in recs:
            c = extract_confidence(r.get("response", ""))
            if c is None:
                continue
            rows.append({"model": model, "task": task, "story_id": story,
                         "step": r["step"], "pos_frac": r["step"] / ns, "conf": c})
    if not rows:
        print("[confidence] no usable records")
        return 1
    print(f"[confidence] records={len(rows)} models={sorted({r['model'] for r in rows})}")

    # 1) position-quartile means (per model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    buckets = {m: {q: [] for q in ("Q1", "Q2", "Q3", "Q4")} for m in sorted(keep)}
    for r in rows:
        q = "Q1" if r["pos_frac"] <= 0.25 else "Q2" if r["pos_frac"] <= 0.5 \
            else "Q3" if r["pos_frac"] <= 0.75 else "Q4"
        buckets[r["model"]][q].append(r["conf"])
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "Q1", "Q2", "Q3", "Q4", "n"])
        print(f"{'model':12} {'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6}   n")
        for m, b in buckets.items():
            vals = [round(statistics.mean(b[q]), 1) if b[q] else float("nan") for q in ("Q1", "Q2", "Q3", "Q4")]
            n = sum(len(b[q]) for q in b)
            w.writerow([m] + vals + [n])
            print(f"{m:12} {vals[0]:6.1f} {vals[1]:6.1f} {vals[2]:6.1f} {vals[3]:6.1f} {n:6d}")

    # 2) MixedLM (optional dependency)
    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        df = pd.DataFrame(rows)
        for formula in ("conf ~ pos_frac + C(model) + C(task)",
                        "conf ~ pos_frac * C(model) + C(task)"):
            print("\n" + "=" * 70)
            print(f"[confidence] MixedLM: {formula}")
            try:
                mf = smf.mixedlm(formula, df, groups=df["story_id"]).fit(reml=False, maxiter=500)
                print(mf.summary())
                print(f"converged={mf.converged} llf={mf.llf:.1f}")
            except Exception as exc:  # noqa: BLE001
                print(f"fit failed: {type(exc).__name__}: {str(exc)[:200]}")
    except ImportError:
        print("[confidence] statsmodels/pandas missing; skipping MixedLM (the table has been written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
