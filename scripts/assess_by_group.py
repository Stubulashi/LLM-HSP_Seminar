"""步骤2补：分 reason/task 的 judge-acc 与错误行清单（读 sidecar）。"""
import csv, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assess_judges import _parse

BASE = os.path.join("data", "manual_equivalence")
rows = list(csv.DictReader(open(os.path.join(BASE, "labels.csv"), encoding="utf-8-sig")))
side = {}
with open(os.path.join(BASE, "judge_sidecar.csv"), encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        side[r["id"]] = r


def judge_val(r, col):
    s = (side.get(r["id"]) or {}).get(col)
    if s is None or not str(s).strip():
        s = r.get(col)
    return _parse(s)


def human_val(r):
    return _parse(r.get("human_label"))


def agree_acc(rows_sub, col):
    pairs = [(human_val(r), judge_val(r, col)) for r in rows_sub]
    pairs = [(h, j) for h, j in pairs if h is not None and j is not None]
    if not pairs:
        return None
    ok = sum(1 for h, j in pairs if h == j)
    return len(pairs), ok / len(pairs)


for col in ("judge_local", "judge_api_deepseek"):
    print("==", col)
    for grp in ("forced_conflict", "boundary", "low_cos_shared", "random"):
        sub = [r for r in rows if r["reason"] == grp]
        m = agree_acc(sub, col)
        print(f"  reason {grp:16s} n={m[0] if m else 0:3d} acc={m[1]:.3f}" if m else f"  reason {grp}: -")
    for t in ("false_belief", "faux_pas", "implicature"):
        sub = [r for r in rows if r["task"] == t]
        m = agree_acc(sub, col)
        print(f"  task   {t:14s} n={m[0] if m else 0:3d} acc={m[1]:.3f}" if m else f"  task {t}: -")
    # 错误行清单
    errs = [(r, human_val(r), judge_val(r, col)) for r in rows
            if human_val(r) is not None and judge_val(r, col) is not None
            and human_val(r) != judge_val(r, col)]
    print(f"  errors n={len(errs)}:")
    for r, h, j in errs[:12]:
        print(f"    {r['id']} {r['task']}/{r['story_id']} step{r['step']} {r['model']} "
              f"cos={r['cos_sim']} human={int(h)} judge={int(j)} reason={r['reason']}")
