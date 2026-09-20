"""Content-level equivalence judging (judge) with the local qwen15b on the answers of six sample stories.

Compares the processed accuracy (cosine 0.7) with the judge (semantic equivalence) to expose the
cosine criterion's misses and false positives.
Usage: python -X utf8 scripts/judge_sample.py   (requires the repo root on PYTHONPATH)
"""
import csv, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.confidence import extract_interpretation
from analysis.judge import LLMEquivalenceJudge

SAMPLE = [("false_belief", "fb001"), ("false_belief", "fb002"),
          ("faux_pas", "sfp001"), ("faux_pas", "sfp002"),
          ("implicature", "swmimp001"), ("implicature", "swmimp002")]

acc = {r["story_id"]: float(r["accuracy"])
       for r in csv.DictReader(open("results/processed/accuracy.csv", encoding="utf-8"))
       if r["model"] == "qwen15b"}


def main() -> int:
    judge = LLMEquivalenceJudge(model_name="qwen15b")
    judge.load()
    try:
        print(f"{'story':14} {'cos_acc':>7} {'judge':>6}  rationale")
        correct = 0
        for task, sid in SAMPLE:
            ann = json.load(open(f"data/annotated/{sid}.json", encoding="utf-8"))
            q = ann.get("question", "")
            gold = ann.get("gold_answer", "")
            p = os.path.join("results", "raw", "qwen15b", task, sid, "run1.json")
            recs = [json.loads(l) for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
            last = max(recs, key=lambda r: r["step"])
            ans = extract_interpretation(last["response"])[:600]
            res = judge.judge(q, ans, gold)
            cos = acc.get(sid, None)
            print(f"{sid:14} {str(cos):>7} {str(res['label']):>6}  {res['rationale'][:120]}")
            correct += bool(res["label"])
        print(f"\njudge-accuracy(qwen15b 6-sample) = {correct}/{len(SAMPLE)}")
    finally:
        judge.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
