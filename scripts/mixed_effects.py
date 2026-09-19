"""混合效应模型（MixedLM）显著性检验（只读）。

读取 results/processed_judge/{accuracy,emergence}.csv + config/models.yaml 的
size/training，对 judge 口径指标跑两类模型（groups=story_id）：
  acc01 ~ C(model) + C(task) + C(model):C(task)      （模型×任务差异）
  acc01 ~ size_centered + C(training) + C(task)      （规模/范式主效应）
emerged01（emergence 是否有效）同构。

用法： python -X utf8 scripts/mixed_effects.py
输出：各模型摘要表（含 p 值）到 stdout。
"""
import csv, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("[mixed_effects] statsmodels 未安装：pip install statsmodels 后重跑")
    sys.exit(1)

from config.config_manager import ConfigManager

PJ = "results/processed_judge"


def _size_num(s: str) -> float:
    m = re.search(r"([\d.]+)", s or "")
    return float(m.group(1)) if m else -1.0


def _load():
    cfg = ConfigManager()
    meta = cfg.get("models")
    acc = list(csv.DictReader(open(f"{PJ}/accuracy.csv", encoding="utf-8")))
    emg = list(csv.DictReader(open(f"{PJ}/emergence.csv", encoding="utf-8")))
    emgmap = {(r["model"], r["story_id"], r["repetition"]): r
              for r in emg}
    rows = []
    for r in acc:
        m, s, rep = r["model"], r["story_id"], r["repetition"]
        er = emgmap.get((m, s, rep), {})
        info = meta.get(m, {})
        rows.append({
            "acc01": 1.0 if r["accuracy"] == "1.0" else 0.0,
            "emerged01": 1.0 if (er.get("emergence_point") or "") != "" else 0.0,
            "model": m, "task": r["task"], "story_id": s,
            "size": _size_num(info.get("size", "")),
            "training": info.get("training", "?"),
        })
    return rows


def _fit(df, formula, y):
    print("\n" + "=" * 70)
    print(f"[mixed_effects] y={y}  formula: {formula}")
    df = df.copy()
    df["size_centered"] = df["size"] - df["size"].mean()
    try:
        md = smf.mixedlm(formula, df, groups=df["story_id"])
        mf = md.fit(reml=False, maxiter=1000)
        print(mf.summary())  # 兼容 statsmodels 0.14+/1.x 的 summary 结构
        print(f"[mixed_effects] converged={mf.converged} llf={mf.llf:.1f}")
    except Exception as exc:  # noqa: BLE001 - 收敛/奇异等输出诊断
        print(f"[mixed_effects] fit failed: {type(exc).__name__}: {str(exc)[:300]}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen7b,deepseek7b,qwen14b,deepseek14b,deepseek32b",
                    help="逗号分隔模型名（默认正式 5 模型；对照/扩展模型不入统计）")
    args = ap.parse_args()
    keep = {m.strip() for m in args.models.split(",") if m.strip()}
    rows = _load()
    rows = [r for r in rows if r["model"] in keep]
    df = __import__("pandas").DataFrame(rows)
    print(f"[mixed_effects] runs={len(df)} models={sorted(df['model'].unique())}")
    for y in ("acc01", "emerged01"):
        _fit(df, f"{y} ~ C(model) + C(task) + C(model):C(task)", y)
        _fit(df, f"{y} ~ size_centered + C(training) + C(task)", y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
