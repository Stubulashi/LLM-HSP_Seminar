"""B3 配套：用指定 judge（local/deepseek/kimi）回填等价子集 labels。

默认写回 data/manual_equivalence/labels.csv 的 judge_* 列；
若该文件正被 Excel 占用（Windows PermissionError），用 --out 指向独立 sidecar：
    python -X utf8 scripts/run_judge_on_pool.py --provider local --out data/manual_equivalence/judge_sidecar.csv
sidecar 列为 [id, judge_local, judge_api_deepseek, judge_api_kimi]，assess_judges.py --judge-file 可合并评估。

- --provider local   → 本地 qwen15b（需已缓存）；写 judge_local
- --provider deepseek→ 读 .env DEEPSEEK_*；写 judge_api_deepseek
- --provider kimi    → 读 .env KIMI_*；写 judge_api_kimi
写回值：1 / 0 / NA（NA=judge 无法判定，label=None）。已填单元格跳过（断点续跑）；--force 重算。
"""
import argparse, csv, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 保证从任意 cwd 直接运行都能导入仓库根模块（云端/本地一致）
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
    ap.add_argument("--out", default=None, help="写回文件；默认 labels.csv；Excel 占用时给 sidecar 路径")
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDER_COL))
    ap.add_argument("--model", default=None,
                    help="local provider 用的模型名（默认 qwen15b；云端可用 qwen7b 走 vllm）")
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
    judge.load()  # local 加载模型 / api 校验 .env 配置
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
            # 及时落盘，支持断点
            if sidecar_mode:
                _save_sidecar(out, store)
            else:
                with open(out, "w", encoding="utf-8-sig", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(main_rows[0].keys()))
                    w.writeheader()
                    w.writerows(main_rows)
    finally:
        judge.unload()
    print(f"[pool-judge:{args.provider}] done -> {out}。assess 用 --judge-file 合并。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
