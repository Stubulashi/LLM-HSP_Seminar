"""Report-specific figure set (English labels, single-column A4 size, same convention as plot_judge.py).

Data sources: cloud_backup/results/processed_judge/accuracy.csv
              cloud_backup/results/processed_judge_condB/accuracy.csv
Artifacts (docs/submission/figs/):
  report_fig_acc_by_model_task.png   model × task judge-acc (single column 6.5in wide, ~9pt fonts)
  report_fig_condb.png               Condition B vs A (DeepSeek 7B/14B/32B × task)
Usage: python -X utf8 scripts/plot_report_figs.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ACC_MAIN = os.path.join("cloud_backup", "results", "processed_judge", "accuracy.csv")
ACC_COND_B = os.path.join("cloud_backup", "results", "processed_judge_condB", "accuracy.csv")
OUT = os.path.join("docs", "submission", "figs")

MODELS = ("qwen05b_vllm", "qwen15b_vllm", "qwen7b", "qwen14b", "qwen32b",
          "deepseek7b", "deepseek14b", "deepseek32b")
MODEL_LABELS = ("Qwen-0.5B", "Qwen-1.5B", "Qwen-7B", "Qwen-14B", "Qwen-32B",
                "DS-7B", "DS-14B", "DS-32B")
TASKS = ("false_belief", "faux_pas", "implicature")
TASK_LABELS = {"false_belief": "False Belief", "faux_pas": "Faux Pas",
               "implicature": "Implicature"}
DS_MODELS = ("deepseek7b", "deepseek14b", "deepseek32b")
DS_LABELS = {"deepseek7b": "DS-7B", "deepseek14b": "DS-14B", "deepseek32b": "DS-32B"}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _acc(rows, model, task):
    sub = [r for r in rows if r["model"] == model and r["task"] == task]
    if not sub:
        return 0.0
    return sum(1 for r in sub if r["accuracy"] == "1.0") / len(sub)


def _style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def fig_acc_by_model_task(rows_acc):
    """Grouped bars of model × task judge accuracy (Condition A, all eight tiers)."""
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    xs = range(len(MODELS))
    for i, task in enumerate(TASKS):
        vals = [_acc(rows_acc, m, task) for m in MODELS]
        ax.bar([x + (i - 1) * width for x in xs], vals, width, label=TASK_LABELS[task])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(MODEL_LABELS, rotation=30, ha="right")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT, "report_fig_acc_by_model_task.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_condb(rows_a, rows_b):
    """Condition B (explicit reasoning hint) vs A, DeepSeek 7B/14B/32B × task."""
    labels, a_vals, b_vals = [], [], []
    for m in DS_MODELS:
        for t in TASKS:
            labels.append(f"{DS_LABELS[m]}\n{TASK_LABELS[t]}")
            a_vals.append(_acc(rows_a, m, t))
            b_vals.append(_acc(rows_b, m, t))
    xs = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar([x - 0.19 for x in xs], a_vals, 0.38, label="Condition A")
    ax.bar([x + 0.19 for x in xs], b_vals, 0.38, label="Condition B")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT, "report_fig_condb.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    _style()
    os.makedirs(OUT, exist_ok=True)
    rows_acc = _load(ACC_MAIN)
    rows_b = _load(ACC_COND_B)
    p1 = fig_acc_by_model_task(rows_acc)
    p2 = fig_condb(rows_acc, rows_b)
    print(f"[plot_report_figs] {p1}")
    print(f"[plot_report_figs] {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
