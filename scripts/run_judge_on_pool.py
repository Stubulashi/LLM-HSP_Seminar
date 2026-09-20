"""B3 companion: backfill the equivalence-subset labels with a chosen judge (local/deepseek/kimi).

By default the judge_* columns of data/manual_equivalence/labels.csv are updated; if that file is
locked by Excel (a Windows PermissionError), point --out at a separate sidecar:
    python -X utf8 scripts/run_judge_on_pool.py --provider local --out data/manual_equivalence/judge_sidecar.csv
The sidecar columns are [id, judge_local, judge_api_deepseek, judge_api_kimi], and
assess_judges.py --judge-file can merge it into the assessment.

- --provider local    → local qwen15b (must be cached); writes judge_local
- --provider deepseek → reads .env DEEPSEEK_*; writes judge_api_deepseek
- --provider kimi     → reads .env KIMI_*; writes judge_api_kimi
Written values: 1 / 0 / NA (NA = the judge could not decide; label=None). Already filled cells are
skipped (resume after interruption); --force recomputes them.
"""
import argparse, csv, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ensure the repo root modules import from any cwd (the same on the cloud and locally)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.judge import LLMEquivalenceJudge

DEFAULT_CSV = os.path.join("data", "manual_equivalence", "labels.csv")
PROVIDER_COL = {"local": "judge_local", "deepseek": "judge_api_deepseek",
                "kimi": "judge_api_kimi"}
SIDECAR_COLS = ["id", "judge_local", "judge_api_deepseek", "judge_api_kimi"]


def _load_sidecar(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def _save_sidecar(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for sid, d in data.items():
        row = {c: d.get(c, "") for c in SIDECAR_COLS}
        row["id"] = sid
        rows.append(row)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SIDECAR_COLS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_CSV)
    ap.add_argument("--out", default=None, help="output file; defaults to labels.csv; give a sidecar path when Excel holds the file")
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDER_COL))
    ap.add_argument("--model", default=None,
                    help="model name for the local provider (default qwen15b; on the cloud qwen7b via vLLM also works)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    col = PROVIDER_COL[args.provider]
    out = args.out or args.labels
    sidecar_mode = os.path.basename(out) != os.path.basename(DEFAULT_CSV)

    main_rows = list(csv.DictReader(open(args.labels, encoding="utf-8-sig")))
    if sidecar_mode:
        store = _load_sidecar(out)
        for r in main_rows:
            store.setdefault(r["id"], {})
        todo = [r for r in main_rows
                if (args.force or not str(store[r["id"]].get(col) or "").strip())]
    else:
        todo = [r for r in main_rows
                if (args.force or not str(r.get(col) or "").strip())]
    if args.reason:
        todo = [r for r in todo if r.get("reason") == args.reason]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[pool-judge:{args.provider}] to_fill={len(todo)} / {len(main_rows)} sidecar={sidecar_mode}")

    judge = LLMEquivalenceJudge(provider=args.provider, model_name=args.model or "qwen15b")
    judge.load()  # local loads the model / api validates the .env config
    try:
        for i, r in enumerate(todo, 1):
            res = judge.judge(r.get("question") or "", r.get("answer_norm") or "",
                              r.get("gold_answer") or "")
            val = "1" if res["label"] is True else ("0" if res["label"] is False else "NA")
            if sidecar_mode:
                store[r["id"]][col] = val
            else:
                r[col] = val
            if i % 5 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {r['id']} -> {val} | {res['rationale'][:60]}")
            # write through promptly so the run can resume
            if sidecar_mode:
                _save_sidecar(out, store)
            else:
                with open(out, "w", encoding="utf-8-sig", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(main_rows[0].keys()))
                    w.writeheader()
                    w.writerows(main_rows)
    finally:
        judge.unload()
    print(f"[pool-judge:{args.provider}] done -> {out}. Use --judge-file in the assess step to merge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
