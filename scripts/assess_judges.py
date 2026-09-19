"""A3: 评估 judge 与人工标注的一致性（acc/precision/recall/F1/Cohen's kappa）。

输入：data/manual_equivalence/labels.csv
- human_label 列（人工）：1=等价 0=不等价 U=无法判断
- judge_local / judge_api_deepseek / judge_api_kimi 列：1 / 0 / NA（NA=无法判定或未跑）
输出：每 judge vs 人工的指标表；U 的两种处理：
  - exclude_U：剔除 human=U 或 judge=NA 后算
  - U_as_wrong：把 human=U 计为错误（敏感性）
用法： python -X utf8 scripts/assess_judges.py
"""
import argparse, csv, math, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JUDGE_COLS = ["judge_local", "judge_api_deepseek", "judge_api_kimi"]


def _parse(v):
    """列值 -> True/False/None(NA/U)。"""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None  # NA / 空 / U


def cohen_kappa(hy, jy):
    """2x2 二值 Cohen's kappa。hy/jy: bool 列表等长。"""
    n = len(hy)
    if n == 0:
        return float("nan")
    a = sum(1 for h, j in zip(hy, jy) if h and j)        # both 1
    d = sum(1 for h, j in zip(hy, jy) if not h and not j)  # both 0
    b = sum(1 for h, j in zip(hy, jy) if h and not j)     # human1 judge0
    c = sum(1 for h, j in zip(hy, jy) if not h and j)     # human0 judge1
    po = (a + d) / n
    p1 = (a + b) / n * (a + c) / n
    p0 = (c + d) / n * (b + d) / n
    pe = p1 + p0
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def metrics(hy, jy):
    n = len(hy)
    tp = sum(1 for h, j in zip(hy, jy) if h and j)
    fp = sum(1 for h, j in zip(hy, jy) if not h and j)
    fn = sum(1 for h, j in zip(hy, jy) if h and not j)
    tn = sum(1 for h, j in zip(hy, jy) if not h and not j)
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {"n": n, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "kappa": cohen_kappa(hy, jy)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("data", "manual_equivalence", "labels.csv"))
    ap.add_argument("--judge-file", default=None,
                    help="judge sidecar csv（id+judge_* 列）；未给时用 labels.csv 内的 judge_* 列")
    ap.add_argument("--exclude-file", default=None,
                    help="排除表 csv（id 列），如空响应行；这些行不参与一致性评估（等效 U）")
    args = ap.parse_args()

    excluded = set()
    if args.exclude_file:
        with open(args.exclude_file, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                excluded.add(r["id"])
        print(f"[assess] exclude-file={args.exclude_file} ids={len(excluded)}")

    rows = list(csv.DictReader(open(args.labels, encoding="utf-8-sig")))
    side = {}
    if args.judge_file:
        with open(args.judge_file, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                side[r["id"]] = r
        print(f"[assess] sidecar={args.judge_file} rows={len(side)}")
    print(f"[assess] rows={len(rows)}")
    humans = [_parse(r.get("human_label")) for r in rows]
    n_human = sum(1 for h in humans if h is not None)
    n_u = sum(1 for h in humans if h is None) + len(excluded)
    print(f"  human labeled={n_human} (U/空={n_u - len(excluded)} + excluded={len(excluded)})")
    if n_human == 0:
        print("  [warn] 尚无人工标注；请先填写 human_label（1/0/U）再运行。")
    for col in JUDGE_COLS:
        judges = []
        for r in rows:
            val = r.get(col)
            if side:
                sval = (side.get(r["id"]) or {}).get(col)
                if sval is not None and str(sval).strip():
                    val = sval
            judges.append(_parse(val))
        n_na = sum(1 for j in judges if j is None)
        # exclude U（含排除表行）
        idx = [i for i in range(len(rows))
               if humans[i] is not None and judges[i] is not None
               and rows[i]["id"] not in excluded]
        if idx:
            m = metrics([humans[i] for i in idx], [judges[i] for i in idx])
            print(f"[{col}] exclude_U n={m['n']} acc={m['acc']:.3f} prec={m['prec']:.3f} "
                  f"rec={m['rec']:.3f} f1={m['f1']:.3f} kappa={m['kappa']:.3f} (judge_NA={n_na})")
        else:
            print(f"[{col}] 无有效配对（judge 列未回填或 human 空）")
        # U as wrong sensitivity（排除表行不计）
        idx2 = [i for i in range(len(rows)) if judges[i] is not None
                and rows[i]["id"] not in excluded]
        if idx2:
            hy = [bool(humans[i]) for i in idx2]  # None(U) -> False
            m2 = metrics(hy, [judges[i] for i in idx2])
            print(f"[{col}] U_as_wrong n={m2['n']} acc={m2['acc']:.3f} kappa={m2['kappa']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
