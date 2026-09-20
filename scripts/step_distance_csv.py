"""Metric 3 (Interpretation Revision) step-distance numeric tables.

Same convention as analysis/stability.step_distances (extract_interpretation cleaning +
EmbeddingService cosine distance), but every adjacent step-pair distance is written to CSV
(for the paper's numeric tables):
  results/processed/step_distance_detail.csv       per-run detail of each adjacent distance
  results/processed/step_distance_by_position.csv  model × step means (revision-curve data)

Usage:
  python -X utf8 scripts/step_distance_csv.py                        # official results
  python -X utf8 scripts/step_distance_csv.py --results-dir results_condB
Note: it reuses the existing embeddings cache (results/embeddings); a cache hit finishes in minutes.
"""
import argparse, csv, os, sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.confidence import extract_interpretation
from analysis.embedding import EmbeddingService
from analysis.pipeline import _filter_task_drift, load_records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=None, help="default <results-dir>/processed")
    ap.add_argument("--models", default=None, help="comma-separated; defaults to all keys in models.yaml")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)

    from config.config_manager import ConfigManager
    from data_manager import DataManager

    cfg = ConfigManager()
    dm = DataManager()
    allow = ({m.strip() for m in args.models.split(",") if m.strip()}
             if args.models else set(cfg.get("models").keys()))
    scoring = cfg.get("scoring")
    service = EmbeddingService(scoring["embedding_model"],
                               cache_dir=os.path.join(args.results_dir, "embeddings"))

    records = _filter_task_drift(load_records(args.results_dir), dm)
    records = [r for r in records if r.get("model") in allow]
    if not records:
        print("[step_distance] no usable records")
        return 1

    groups = {}
    for rec in records:
        groups.setdefault(tuple(rec[k] for k in ("model", "task", "story_id", "repetition")),
                          []).append(rec)

    detail_rows = []
    by_pos = defaultdict(list)  # (model, step_to) -> [dist]
    for (model, task, story_id, rep), group in groups.items():
        ordered = sorted(group, key=lambda r: r["step"])
        texts = [extract_interpretation(r["response"]) for r in ordered]
        vectors = service.encode(texts)
        for i in range(len(vectors) - 1):
            a, b = vectors[i], vectors[i + 1]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0.0:
                continue
            dist = 1.0 - float(np.dot(a, b) / denom)
            detail_rows.append({"model": model, "task": task, "story_id": story_id,
                                "repetition": rep, "step_from": ordered[i]["step"],
                                "step_to": ordered[i + 1]["step"], "distance": round(dist, 6)})
            by_pos[(model, ordered[i + 1]["step"])].append(dist)

    detail_path = os.path.join(out_dir, "step_distance_detail.csv")
    by_pos_path = os.path.join(out_dir, "step_distance_by_position.csv")
    with open(detail_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "task", "story_id", "repetition",
                                           "step_from", "step_to", "distance"])
        w.writeheader()
        w.writerows(detail_rows)
    agg = {k: (float(np.mean(v)), len(v)) for k, v in by_pos.items()}
    with open(by_pos_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "step_to", "mean_distance", "n"])
        for (m, s) in sorted(agg):
            w.writerow([m, s, round(agg[(m, s)][0], 6), agg[(m, s)][1]])

    models = sorted({m for m, _ in agg})
    max_step = 10
    print(f"[step_distance] detail={len(detail_rows)} rows -> {detail_path}")
    print(f"[step_distance] by-position -> {by_pos_path}")
    print(f"{'model':12} " + " ".join(f"S{s:>4}" for s in range(2, max_step + 1)))
    for m in models:
        vals = [f"{agg[(m, s)][0]:5.3f}" if (m, s) in agg else "  -  "
                for s in range(2, max_step + 1)]
        print(f"{m:12} " + " ".join(vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
