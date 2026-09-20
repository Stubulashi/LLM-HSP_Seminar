"""RQ3 "erroneous update / regression" audit: replay the per-step judge labels from the judge cache and count regressions.

proj.md RQ3 (L139-152) asks whether models make "fewer erroneous updates". This script replays the
per-step judge labels for every run (reusing judge_cache.jsonl; zero API calls):
  1) results/processed/step_labels.csv        per-run per-step judge label (1/0)
  2) results/processed/transition_stats.csv   per (model,task) transition statistics:
       regressions    total count of "correct→wrong" backoffs
       regressed_runs share of runs with at least one backoff
       recoveries     count of "wrong→correct" recoveries
       final_correct  last-step accuracy (for comparison with accuracy.csv)
       never_correct  share of runs never correct
   and prints a per-model summary (including the condB directory).

Usage:
  python -X utf8 scripts/stepwise_labels.py
  python -X utf8 scripts/stepwise_labels.py --results-dir results_condB
"""
import argparse, csv, json, os, sys
from collections import defaultdict
from hashlib import sha1

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.confidence import extract_interpretation
from analysis.pipeline import _filter_task_drift, load_records


def _load_cache(path: str) -> dict:
    cache = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            cache[o["k"]] = o["v"]
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--cache", default=None, help="defaults to all judge_cache.jsonl files under <results-dir>/processed_judge*")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    rd = args.results_dir
    out_dir = args.out_dir or os.path.join(rd, "processed")
    os.makedirs(out_dir, exist_ok=True)

    from config.config_manager import ConfigManager
    from data_manager import DataManager

    cfg = ConfigManager()
    dm = DataManager()
    allow = set(cfg.get("models").keys())

    # merge caches: processed_judge of the main results and processed_judge_condB of condB
    cache = {}
    if args.cache:
        cache.update(_load_cache(args.cache))
    else:
        # try all the common cache locations (for condB, rd=results_condB, but the cache usually sits under results/processed_judge_condB)
        for c in (os.path.join(rd, "processed_judge", "judge_cache.jsonl"),
                  os.path.join(rd, "processed_judge_condB", "judge_cache.jsonl"),
                  os.path.join("results", "processed_judge", "judge_cache.jsonl"),
                  os.path.join("results", "processed_judge_condB", "judge_cache.jsonl")):
            cache.update(_load_cache(c))
    print(f"[stepwise] cache keys={len(cache)}")

    records = _filter_task_drift(load_records(rd), dm)
    records = [r for r in records if r.get("model") in allow]
    groups = {}
    for rec in records:
        groups.setdefault(tuple(rec[k] for k in ("model", "task", "story_id", "repetition")),
                          []).append(rec)

    label_rows = []
    stats = defaultdict(lambda: {"runs": 0, "regressions": 0, "regressed_runs": 0,
                                 "recoveries": 0, "final_correct": 0, "never_correct": 0})
    for (model, task, story_id, rep), group in groups.items():
        ann = dm.load_annotation(story_id)
        q = ann.get("question", "")
        gold = ann.get("gold_points") or ann.get("gold_answer")
        ordered = sorted(group, key=lambda r: r["step"])
        seq = []
        for r in ordered:
            norm = extract_interpretation(r["response"]).strip()
            if not norm:
                lab = 0  # an empty interpretation counts as not equivalent (consistent with judge_analyze)
            else:
                key = sha1("|{0}|{1}|{2}".format(q, norm, gold).encode("utf-8")).hexdigest()
                v = cache.get(key)
                lab = None if v is None else int(v == "1")
            seq.append(lab)
            label_rows.append({"model": model, "task": task, "story_id": story_id,
                               "repetition": rep, "step": r["step"],
                               "label": "" if lab is None else lab})
        known = [x for x in seq if x is not None]
        if not known:
            continue
        s = stats[(model, task)]
        s["runs"] += 1
        reg = rec = 0
        for a, b in zip(known, known[1:]):
            if a == 1 and b == 0:
                reg += 1
            elif a == 0 and b == 1:
                rec += 1
        s["regressions"] += reg
        s["recoveries"] += rec
        if reg:
            s["regressed_runs"] += 1
        if known[-1] == 1:
            s["final_correct"] += 1
        if all(x == 0 for x in known):
            s["never_correct"] += 1

    with open(os.path.join(out_dir, "step_labels.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "task", "story_id", "repetition", "step", "label"])
        w.writeheader()
        w.writerows(label_rows)
    with open(os.path.join(out_dir, "transition_stats.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "runs", "regressions", "regressed_runs", "regression_rate",
                    "recoveries", "final_correct", "final_acc", "never_correct"])
        for (m, t) in sorted(stats):
            s = stats[(m, t)]
            w.writerow([m, t, s["runs"], s["regressions"], s["regressed_runs"],
                        round(s["regressed_runs"] / s["runs"], 4) if s["runs"] else "",
                        s["recoveries"], s["final_correct"],
                        round(s["final_correct"] / s["runs"], 4) if s["runs"] else "",
                        s["never_correct"]])

    print(f"[stepwise] labels={len(label_rows)} rows -> {out_dir}")
    print(f"{'model':12} {'runs':>5} {'regr/run':>8} {'regr%':>6} {'recov/run':>9} "
          f"{'final_acc':>9} {'never%':>7}")
    agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for (m, t), s in stats.items():
        a = agg[m]
        a[0] += s["runs"]; a[1] += s["regressions"]; a[2] += s["regressed_runs"]
        a[3] += s["recoveries"]; a[4] += s["final_correct"]; a[5] += s["never_correct"]
    for m in sorted(agg):
        runs, reg, regr, rec, fc, nc = agg[m]
        if not runs:
            continue
        print(f"{m:12} {runs:5d} {reg / runs:8.2f} {regr / runs * 100:5.1f}% "
              f"{rec / runs:9.2f} {fc / runs:9.3f} {nc / runs * 100:6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
