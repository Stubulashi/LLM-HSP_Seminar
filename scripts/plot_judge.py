"""judge 口径可视化（英文标签避免中文字体缺失）。

产物（results/processed_judge/figs/）：
  fig_acc_by_model_task.png     模型×任务 judge-acc 分组柱状
  fig_emergence_hist.png        emergence 首达 step 分布（按任务）
  fig_condb_compare.png         Condition B vs A（judge-acc，DeepSeek 7B/14B）
用法： python -X utf8 scripts/plot_judge.py
"""
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PJ = "results/processed_judge"
OUT = os.path.join(PJ, "figs")
MODELS = ("qwen05b_vllm", "qwen15b_vllm", "qwen7b", "qwen14b", "qwen32b",
          "deepseek7b", "deepseek14b", "deepseek32b")
TASKS = ("false_belief", "faux_pas", "implicature")


def _load(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def acc(rows, m, t):
    sub = [r for r in rows if r["model"] == m and r["task"] == t]
    if not sub:
        return None
    return sum(1 for r in sub if r["accuracy"] == "1.0") / len(sub)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    acc_rows = _load(os.path.join(PJ, "accuracy.csv"))
    emg_rows = _load(os.path.join(PJ, "emergence.csv"))

    # 1) acc by model×task
    x = range(len(MODELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, t in enumerate(TASKS):
        vals = [acc(acc_rows, m, t) or 0 for m in MODELS]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=t)
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("judge accuracy")
    ax.set_title("Accuracy by model x task (judge)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_acc_by_model_task.png"), dpi=150)
    plt.close(fig)

    # 2) emergence step histogram by task
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for t in TASKS:
        pts = [int(r["emergence_point"]) for r in emg_rows
               if r["task"] == t and r["emergence_point"] != ""]
        if pts:
            ax.hist(pts, bins=range(1, max(pts) + 2), alpha=0.5, label=f"{t} (n={len(pts)})")
    ax.set_xlabel("emergence step (first judge-equivalent step)")
    ax.set_ylabel("count")
    ax.set_title("Emergence point distribution (judge, all models)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_emergence_hist.png"), dpi=150)
    plt.close(fig)

    # 3) Condition B compare
    cpath = "results/processed_judge_condB/accuracy.csv"
    cb = _load(cpath)
    if cb:
        labels, a_vals, b_vals = [], [], []
        for m in ("deepseek7b", "deepseek14b"):
            for t in TASKS:
                labels.append(f"{m}\n{t}")
                a_vals.append(acc(acc_rows, m, t) or 0)
                b_vals.append(acc(cb, m, t) or 0)
        xi = range(len(labels))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar([i - 0.2 for i in xi], a_vals, 0.4, label="condition_a")
        ax.bar([i + 0.2 for i in xi], b_vals, 0.4, label="condition_b")
        ax.set_xticks(list(xi))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("judge accuracy")
        ax.set_title("Condition B (reasoning prompt) vs A - DeepSeek")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_condb_compare.png"), dpi=150)
        plt.close(fig)
    print(f"[plot_judge] figures -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
