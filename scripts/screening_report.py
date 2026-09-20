"""Scientific-value screening of the data (read-only statistics; nothing is modified).

Reads:
  results/processed_judge/accuracy.csv, emergence.csv   (the official judge-based results)
  results/raw/<model>/<task>/<story>/runN.json          (NDJSON raw records)
  data/annotated/<story>.json                           (sentence counts/lengths)

Prints four sections:
  S1 emergence-point distribution (including the "last step only" rate) — how strong the incremental emergence is for RQ1
  S2 empty-response / pure-echo rate (per model × task, with the judge-acc contrast) — the data-quality gate
  S3 confidence parse rate + confidence × correctness association — feasibility of the confidence-curve study
  S4 summary (for pasting back)

Usage: python -X utf8 scripts/screening_report.py
"""
import csv, glob, json, os, statistics, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.confidence import extract_confidence, extract_interpretation

MODELS = ("qwen7b", "deepseek7b", "qwen14b", "deepseek14b", "deepseek32b")
TASKS = ("false_belief", "faux_pas", "implicature")
BASE = "results"


def _load_judge():
    acc = {r["story_id"] + "|" + r["model"] + "|" + r["repetition"]: r
           for r in csv.DictReader(open(f"{BASE}/processed_judge/accuracy.csv", encoding="utf-8"))}
    emg = {r["story_id"] + "|" + r["model"] + "|" + r["repetition"]: r
           for r in csv.DictReader(open(f"{BASE}/processed_judge/emergence.csv", encoding="utf-8"))}
    return acc, emg


def _n_sent(story_id: str) -> int:
    try:
        ann = json.load(open(f"data/annotated/{story_id}.json", encoding="utf-8"))
        return len(ann.get("sentences", []))
    except Exception:
        return -1


def _load_raw_runs():
    """Returns {model: [(task, story_id, rep, [records])]}"""
    runs = []
    for p in sorted(glob.glob(os.path.join(BASE, "raw", "*", "*", "*", "run*.json"))):
        parts = os.path.normpath(p).split(os.sep)  # results/raw/<model>/<task>/<story>/runN.json
        model, task_dir, story = parts[2], parts[3], parts[4]
        recs = [json.loads(l) for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
        if not recs:
            continue
        rep = recs[0].get("repetition", 1)
        # skip task drift (consistent with analyze)
        try:
            ann = json.load(open(f"data/annotated/{story}.json", encoding="utf-8"))
        except Exception:
            continue
        if ann.get("task") != task_dir:
            continue
        if model not in MODELS:
            continue
        runs.append((model, ann["task"], story, rep, recs))
    return runs


def sec1(acc, emg):
    print("=" * 60)
    print("S1 emergence-point distribution (RQ1)")
    print(f"{'model':11} {'task':13} {'n':>4} {'valid':>5} {'mean':>5} {'med':>4} "
          f"{'laststep%':>9} {'frac_of_len':>11}")
    for m in MODELS:
        for t in TASKS:
            pts, lens = [], []
            key_t = [k for k in emg if emg[k]["model"] == m and emg[k]["task"] == t]
            for k in key_t:
                v = emg[k]["emergence_point"]
                if v == "":
                    continue
                pts.append(int(v))
                ns = _n_sent(emg[k]["story_id"])
                lens.append(ns if ns > 0 else -1)
            n = len(key_t)
            if not pts:
                print(f"{m:11} {t:13} {n:4d} {0:5d}  -    -     -        -")
                continue
            valid = len(pts)
            lastfrac = sum(1 for p, ln in zip(pts, lens) if ln > 0 and p == ln) / valid
            fracs = [p / ln for p, ln in zip(pts, lens) if ln > 0]
            print(f"{m:11} {t:13} {n:4d} {valid:5d} {statistics.mean(pts):5.1f} "
                  f"{statistics.median(pts):4.0f} {lastfrac * 100:8.0f}% "
                  f"{statistics.mean(fracs):11.2f}")


def sec2(acc, emg, runs):
    print("=" * 60)
    print("S2 empty-response / pure-echo rate (data-quality gate)")
    print(f"{'model':11} {'task':13} {'runs':>5} {'empty%':>7} {'judge_acc':>9}")
    from collections import defaultdict

    grp = defaultdict(list)
    for model, task, story, rep, recs in runs:
        last = max(recs, key=lambda r: r["step"])
        norm = extract_interpretation(last["response"]).strip()
        grp[(model, task)].append((len(norm) < 10, story, rep))
    for m in MODELS:
        for t in TASKS:
            rows = grp.get((m, t), [])
            if not rows:
                continue
            empty = sum(1 for e, _, _ in rows if e) / len(rows)
            ok = sum(1 for _, s, r in rows if (acc.get(f"{s}|{m}|{r}") or {}).get("accuracy") == "1.0")
            print(f"{m:11} {t:13} {len(rows):5d} {empty * 100:6.1f}% "
                  f"{ok / len(rows):9.3f}")


def sec3(acc, runs):
    print("=" * 60)
    print("S3 confidence parse rate and confidence × correctness association")
    from collections import defaultdict

    conf_parse = defaultdict(lambda: [0, 0])  # (model,task): [parsed, total]
    conf_by_acc = defaultdict(list)  # (model,task,acc01): [conf...]
    for model, task, story, rep, recs in runs:
        key_judge = f"{story}|{model}|{rep}"
        arow = acc.get(key_judge)
        if arow is None:
            continue
        is1 = arow["accuracy"] == "1.0"
        last = max(recs, key=lambda r: r["step"])
        v = extract_confidence(last["response"])
        conf_parse[(model, task)][1] += 1
        if v is not None:
            conf_parse[(model, task)][0] += 1
            conf_by_acc[(model, task, int(is1))].append(v)
    print(f"{'model':11} {'task':13} {'parse%':>7} {'acc=1 conf':>11} {'acc=0 conf':>11} {'delta':>6}")
    for m in MODELS:
        for t in TASKS:
            par, tot = conf_parse.get((m, t), (0, 0))
            c1 = conf_by_acc.get((m, t, 1), [])
            c0 = conf_by_acc.get((m, t, 0), [])
            if tot == 0:
                continue
            m1 = statistics.mean(c1) if c1 else float("nan")
            m0 = statistics.mean(c0) if c0 else float("nan")
            d = (m1 - m0) if c1 and c0 else float("nan")
            print(f"{m:11} {t:13} {par / tot * 100:6.1f}% "
                  f"{m1:11.1f} {m0:11.1f} {d:6.1f}")


def main() -> int:
    acc, emg = _load_judge()
    sec1(acc, emg)
    runs = _load_raw_runs()
    sec2(acc, emg, runs)
    sec3(acc, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
