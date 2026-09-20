"""A5: calibrate the cosine threshold with the manual equivalence subset (instead of the fixed 0.7) and report the emergence-definition comparison.

For each pooled row, the t that maximises the agreement between human_label (1/0) and (cos_sim > t) is the suggested threshold; the suggested location to write it is config/experiment.yaml `scoring.threshold`.
Emergence note: the judge-based emergence is defined as "the smallest step first judged equivalent"; its agreement with the first cos>t step is assessed from the judge columns (after per-step backfill); this script lays out the method framework first.

Usage: python -X utf8 scripts/calibrate_cosine_threshold.py
"""
import argparse, csv, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assess_judges import _parse, metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("data", "manual_equivalence", "labels.csv"))
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.labels, encoding="utf-8-sig"))]
    paired = [(float(r["cos_sim"]), _parse(r.get("human_label")))
              for r in rows if r.get("human_label") not in (None, "", "U")]
    paired = [(c, h) for c, h in paired if h is not None]
    print(f"[calib] paired (human 1/0 and readable cos)={len(paired)}")
    if not paired:
        print("  [warn] no human labels yet; fill in human_label and rerun.")
        return 0
    # scan thresholds
    best = None
    for tenth in range(0, 1001, 10):  # 0.00..1.00
        t = tenth / 1000.0
        pred = [c > t for c, _ in paired]
        hy = [h for _, h in paired]
        m = metrics(hy, pred)
        if best is None or m["acc"] > best[1]["acc"]:
            best = (t, m)
    t, m = best
    print(f"[calib] best threshold t={t:.3f} acc={m['acc']:.3f} "
          f"prec={m['prec']:.3f} rec={m['rec']:.3f} kappa={m['kappa']:.3f}")
    print(f"  -> if adopted, set config/experiment.yaml scoring.threshold to {t:.3f} (0.7 is the initial value)")
    # fixed 0.7 baseline
    m07 = metrics(hy, [c > 0.7 for c, _ in paired])
    print(f"[calib] fixed 0.7 acc={m07['acc']:.3f} kappa={m07['kappa']:.3f} (for comparison)")
    # emergence note
    print("[calib] emergence: judge definition = the first step with judge==1; cos definition = the first step with cos>t.")
    print("         after backfilling the judge_* columns per step within (model,task,story_id,repetition) groups in labels.csv,")
    print("         compare whether the two first-passage definitions agree (the judge columns must hold per-step values).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
