"""A5: 用人工等价子集校准余弦阈值（替代固定 0.7），并报告 emergence 定义对比。

对池内各行，human_label(1/0) 与 (cos_sim > t) 的一致性最优 t 即为建议阈值；
同时给出建议写入位置 config/experiment.yaml `scoring.threshold`。
Emergence 说明：judge 版 emergence 定义为"首个被 judge 判等价的最小 step"，
与"首个 cos>t" 的首达一致率由 judge 列（per-step 回填后）评估；此处先给方法框架。

用法： python -X utf8 scripts/calibrate_cosine_threshold.py
"""
import argparse, csv, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assess_judges import _parse, metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("data", "manual_equivalence", "labels.csv"))
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.labels, encoding="utf-8-sig"))]
    paired = [(float(r["cos_sim"]), _parse(r.get("human_label")))
              for r in rows if r.get("human_label") not in (None, "", "U")]
    paired = [(c, h) for c, h in paired if h is not None]
    print(f"[calib] paired(human 1/0 且 cos 可读)={len(paired)}")
    if not paired:
        print("  [warn] 尚无人工标注；填完 human_label 后重跑。")
        return 0
    # 扫描阈值
    best = None
    for tenth in range(0, 1001, 10):  # 0.00..1.00
        t = tenth / 1000.0
        pred = [c > t for c, _ in paired]
        hy = [h for _, h in paired]
        m = metrics(hy, pred)
        if best is None or m["acc"] > best[1]["acc"]:
            best = (t, m)
    t, m = best
    print(f"[calib] best threshold t={t:.3f} acc={m['acc']:.3f} "
          f"prec={m['prec']:.3f} rec={m['rec']:.3f} kappa={m['kappa']:.3f}")
    print(f"  -> 若采用，请把 config/experiment.yaml scoring.threshold 改为 {t:.3f}（当前 0.7 为初值）")
    # 固定 0.7 基线
    m07 = metrics(hy, [c > 0.7 for c, _ in paired])
    print(f"[calib] fixed 0.7 acc={m07['acc']:.3f} kappa={m07['kappa']:.3f} (对比用)")
    # emergence 说明
    print("[calib] emergence: judge 定义=首个 judge==1 的 step；cos 定义=首个 cos>t 的 step。")
    print("         在 labels.csv 中按 (model,task,story_id,repetition) 组内用 judge_* 列回填各 step 后，")
    print("         再比较两种首达是否一致（需 judge 列含 per-step 值）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
