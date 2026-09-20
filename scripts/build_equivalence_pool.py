"""A1: build the "manual equivalence judgement" candidate pool (data/manual_equivalence/labels.csv).

Layered sampling used to check/calibrate the judge (local or API):
- forced_conflict: samples where the judge and cosine are known to conflict (including the
  faux_pas/sfp001 cosine false positive and the implicature/swmimp001 cosine miss) and their qwen15b counterparts;
- boundary: cos_sim ∈ [0.5, 0.9] (around the 0.7 threshold);
- high_cos: cos_sim ≥ 0.85 with a negation word / suspected inverted meaning (possible false positives);
- low_cos_shared: cos_sim ≤ 0.30 but with word/character overlap with the gold (possible misses);
- random: a deterministic-seed random baseline (per task).

Fields are in the CSV header; human_label / judge_* are left blank for human and judge backfill.
Usage: python -X utf8 scripts/build_equivalence_pool.py
"""
import argparse, csv, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.accuracy import best_similarity
from analysis.confidence import extract_interpretation
from analysis.embedding import EmbeddingService
from data_manager import DataManager

OUT = os.path.join("data", "manual_equivalence", "labels.csv")
FIELDS = ["id", "task", "story_id", "step", "model", "repetition", "run_path",
          "question", "gold_answer", "answer_norm", "cos_sim", "reason",
          "human_label", "label_notes",
          "judge_local", "judge_api_deepseek", "judge_api_kimi", "annotator_id"]
NEG = (" no ", " not ", " nobody ", " nothing ", " never ", " doesn't ", " don't ",
       " did not ", " wasn't ", " isn't ", "no one", "none")


def _norm(text: str) -> str:
    return extract_interpretation(text)[:600]


def _has_negation(text: str) -> bool:
    low = (" " + text.lower() + " ")
    return any(n in low for n in NEG)


def _shared_tokens(a: str, b: str) -> bool:
    # Chinese: share any 2-gram; English: share content words (stop words removed)
    if re.search(r"[\u4e00-\u9fff]", a + b):
        bigrams = {a[i:i + 2] for i in range(len(a) - 1)}
        return any(bb in b for bb in bigrams if bb.strip())
    toks = set(re.findall(r"[a-z']+", a.lower()))
    toks = {t for t in toks if len(t) > 2 and t not in
            {"the", "and", "that", "this", "with", "from", "was", "were", "have", "has"}}
    return any(t in b.lower() for t in toks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-layer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(OUT) and not args.overwrite:
        print(f"already exists: {OUT} (use --overwrite)")
        return 0

    dm = DataManager()
    svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2",
                           cache_dir="results/embeddings")

    # per-record computation of (cos_sim, the information needed for reason candidates)
    rows = {}  # (model,task,story,rep,step) -> candidate dict
    import glob
    files = sorted(glob.glob(os.path.join("results", "raw", "qwen05b", "*", "*", "run*.json")))
    files += sorted(glob.glob(os.path.join("results", "raw", "qwen15b", "*", "*", "run*.json")))
    for p in files:
        parts = os.path.normpath(p).split(os.sep)  # results/raw/<model>/<task>/<story>/runN.json
        task_dir, story = parts[3], parts[4]
        recs = [json.loads(l) for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
        if not recs:
            continue
        model = recs[0]["model"]
        try:
            ann = dm.load_annotation(story)
        except FileNotFoundError:
            continue
        ann_task = ann.get("task")
        if ann_task != task_dir:  # task drift; same filter as analyze
            continue
        gold = ann.get("gold_points") or ann.get("gold_answer")
        q = ann.get("question", "")
        rep = recs[0].get("repetition", 1)
        for r in recs:
            cos = best_similarity(r["response"], gold, svc)
            key = (model, ann_task, story, rep, r["step"])
            rows[key] = {
                "task": ann_task, "story_id": story, "step": r["step"], "model": model,
                "repetition": rep, "run_path": p, "question": q, "gold_answer": gold,
                "answer_norm": _norm(r["response"]), "cos_sim": round(cos, 4),
            }

    forced_keys = [
        ("qwen05b", "faux_pas", "sfp001"), ("qwen05b", "implicature", "swmimp001"),
        ("qwen15b", "faux_pas", "sfp001"), ("qwen15b", "implicature", "swmimp001"),
    ]
    last_by_story = {}
    for (model, task, story, rep, step), info in rows.items():
        last_by_story.setdefault((model, task, story, rep), []).append(info)
    last = {k: max(v, key=lambda x: x["step"]) for k, v in last_by_story.items()}

    selected: dict[tuple, str] = {}  # key -> reason

    def add(key: tuple, reason: str):
        if key in rows and key not in selected:
            selected[key] = reason

    # L0 forced (take the last-step key of that story)
    for m, t, s in forced_keys:
        cand = [k for k in rows if k[0] == m and k[1] == t and k[2] == s]
        if cand:
            add(max(cand, key=lambda k: k[4]), "forced_conflict")

    # remaining layers: last steps plus some boundary mid-steps
    per_layer = {}
    for (m, t, s, rep, step), info in rows.items():
        cos = info["cos_sim"]
        txt = info["answer_norm"].lower()
        if 0.5 <= cos <= 0.9:
            per_layer.setdefault("boundary", []).append((info, (m, t, s, rep, step)))
        elif cos >= 0.85 and _has_negation(info["answer_norm"]):
            per_layer.setdefault("high_cos", []).append((info, (m, t, s, rep, step)))
        elif cos <= 0.30 and _shared_tokens(info["answer_norm"], info["gold_answer"]):
            per_layer.setdefault("low_cos_shared", []).append((info, (m, t, s, rep, step)))

    import random
    rng = random.Random(args.seed)
    for layer in ("boundary", "high_cos", "low_cos_shared"):
        bucket = per_layer.get(layer, [])
        # at most max-per-layer rows per task
        by_task = {}
        for info, key in bucket:
            by_task.setdefault(info["task"], []).append(key)
        for t, keys in by_task.items():
            rng.shuffle(keys)
            for k in keys[:args.max_per_layer]:
                add(k, layer)
    # random baseline
    for t in ("false_belief", "faux_pas", "implicature"):
        cand = [k for k, info in rows.items() if info["task"] == t and k not in selected]
        rng.shuffle(cand)
        for k in cand[:args.max_per_layer]:
            add(k, "random")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for idx, (key, reason) in enumerate(sorted(selected.items()), 1):
            info = rows[key]
            w.writerow({**{f: "" for f in FIELDS}, **{
                "id": f"EQ{idx:04d}", "task": info["task"], "story_id": info["story_id"],
                "step": info["step"], "model": info["model"], "repetition": info["repetition"],
                "run_path": info["run_path"], "question": info["question"],
                "gold_answer": info["gold_answer"], "answer_norm": info["answer_norm"],
                "cos_sim": info["cos_sim"], "reason": reason}})
    from collections import Counter
    print(f"[pool] wrote {OUT} rows={len(selected)} by_reason={dict(Counter(selected.values()))}")
    print(f"[pool] by_task={dict(Counter(rows[k]['task'] for k in selected))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
