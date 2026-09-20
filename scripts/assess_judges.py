"""A3: assess judge vs human agreement (acc/precision/recall/F1/Cohen's kappa).

Input: data/manual_equivalence/labels.csv
- human_label column (manual): 1=equivalent 0=not equivalent U=undecidable
- judge_local / judge_api_deepseek / judge_api_kimi columns: 1 / 0 / NA (NA = undecidable or not run)
Output: a metrics table per judge vs the human labels; U is handled in two ways:
  - exclude_U: drop human=U or judge=NA rows before computing
  - U_as_wrong: count human=U as wrong (sensitivity)
Usage: python -X utf8 scripts/assess_judges.py
"""
import argparse, csv, math, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

JUDGE_COLS = ["judge_local", "judge_api_deepseek", "judge_api_kimi"]


def _parse(v):
    """Column value -> True/False/None (NA/U)."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None  # NA / empty / U


def cohen_kappa(hy, jy):
    """2x2 binary Cohen's kappa. hy/jy: equal-length bool lists."""
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
                    help="judge sidecar csv (id + judge_* columns); when omitted, the judge_* columns of labels.csv are used")
    ap.add_argument("--exclude-file", default=None,
                    help="exclusion csv (id column), e.g. empty-response rows; these rows take no part in the agreement assessment (equivalent to U)")
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
    print(f"  human labeled={n_human} (U/empty={n_u - len(excluded)} + excluded={len(excluded)})")
    if n_human == 0:
        print("  [warn] no human labels yet; fill in human_label (1/0/U) before running.")
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
        # exclude U (including the exclusion-list rows)
        idx = [i for i in range(len(rows))
               if humans[i] is not None and judges[i] is not None
               and rows[i]["id"] not in excluded]
        if idx:
            m = metrics([humans[i] for i in idx], [judges[i] for i in idx])
            print(f"[{col}] exclude_U n={m['n']} acc={m['acc']:.3f} prec={m['prec']:.3f} "
                  f"rec={m['rec']:.3f} f1={m['f1']:.3f} kappa={m['kappa']:.3f} (judge_NA={n_na})")
        else:
            print(f"[{col}] no valid pairs (judge columns not filled in, or human labels empty)")
        # U-as-wrong sensitivity (exclusion-list rows are not counted)
        idx2 = [i for i in range(len(rows)) if judges[i] is not None
                and rows[i]["id"] not in excluded]
        if idx2:
            hy = [bool(humans[i]) for i in idx2]  # None (U) -> False
            m2 = metrics(hy, [judges[i] for i in idx2])
            print(f"[{col}] U_as_wrong n={m2['n']} acc={m2['acc']:.3f} kappa={m2['kappa']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
