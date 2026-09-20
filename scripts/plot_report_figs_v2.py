"""Report draft package v2: figure generation (3 new figures + 2 reused copies).

Inputs:
  cloud_backup/results/processed_judge/accuracy.csv                     (aggregated by build_report_tables_v2.py)
  docs/submission/report_draft_v2/tables/table1_acc_ci.csv              (figure 1 data: model × task acc + 95% CI)
  cloud_backup/results/processed/transition_stats.csv                   (figure 2 data: backoff/recovery)
  cloud_backup/results/processed_judge/confidence_by_position.csv       (figure 3 data: CondA confidence)
  cloud_backup/results_condB/processed/confidence_by_position.csv       (figure 3 data: CondB confidence)
  docs/submission/figs/report_fig_acc_by_model_task.png                 (reused figure, copied)
  docs/submission/figs/report_fig_condb.png                             (reused figure, copied)

Artifacts (docs/submission/report_draft_v2/figs/):
  fig_acc_by_model_task_ci.png     figure 1: model × task judge-acc + story-level cluster bootstrap 95% CI
  fig_process_profile.png          figure 2: process profile (share of runs with at least one backoff / recoveries per run)
  fig_confidence_by_position.png   figure 3: confidence × sentence-position quartile (CondA solid + CondB dashed)
  report_fig_acc_by_model_task.png reused (source docs/submission/figs/)
  report_fig_condb.png             reused (source docs/submission/figs/)

Usage: python -X utf8 scripts/plot_report_figs_v2.py
"""
import csv
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = "cloud_backup"
OUT = os.path.join("docs", "submission", "report_draft_v2", "figs")
TABLE1 = os.path.join("docs", "submission", "report_draft_v2", "tables", "table1_acc_ci.csv")
REUSE_SRC = os.path.join("docs", "submission", "figs")

MODELS = ("qwen05b_vllm", "qwen15b_vllm", "qwen7b", "qwen14b", "qwen32b",
          "deepseek7b", "deepseek14b", "deepseek32b")
MODEL_LABELS = ("Qwen-0.5B", "Qwen-1.5B", "Qwen-7B", "Qwen-14B", "Qwen-32B",
                "DS-7B", "DS-14B", "DS-32B")
TASKS = ("false_belief", "faux_pas", "implicature")
TASK_LABELS = {"false_belief": "False Belief", "faux_pas": "Faux Pas", "implicature": "Implicature"}
# colour-blind friendly (a subset of Okabe-Ito)
COLORS = {"false_belief": "#0072B2", "faux_pas": "#D55E00", "implicature": "#009E73"}
CONF_MODELS = ("qwen7b", "qwen14b", "qwen32b", "deepseek7b", "deepseek14b", "deepseek32b")
CONF_COLORS = {"qwen7b": "#0072B2", "qwen14b": "#56B4E9", "qwen32b": "#009E73",
               "deepseek7b": "#D55E00", "deepseek14b": "#CC79A7", "deepseek32b": "#000000"}


def load_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "figure.dpi": 300, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig_acc_ci():
    rows = {(r["model"], r["task"]): r for r in load_csv(TABLE1)}
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    xs = list(range(len(MODELS)))
    for i, task in enumerate(TASKS):
        accs = [float(rows[(m, task)]["acc"]) for m in MODELS]
        los = [float(rows[(m, task)]["ci_lo"]) for m in MODELS]
        his = [float(rows[(m, task)]["ci_hi"]) for m in MODELS]
        pos = [x + (i - 1) * width for x in xs]
        ax.bar(pos, accs, width, label=TASK_LABELS[task], color=COLORS[task], alpha=0.85)
        ax.errorbar(pos, accs, yerr=[[a - l for a, l in zip(accs, los)],
                                     [h - a for a, h in zip(accs, his)]],
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(MODEL_LABELS, rotation=30, ha="right")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", ncol=3, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_acc_by_model_task_ci.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_process_profile():
    rows = load_csv(os.path.join(RAW, "results", "processed", "transition_stats.csv"))
    data = {(r["model"], r["task"]): r for r in rows}
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    titles = ["(a) runs with at least one backoff", "(b) recoveries per run"]
    for ax, key, title in ((axes[0], "regressed_run_share", titles[0]),
                           (axes[1], "recoveries_per_run", titles[1])):
        xs = list(range(len(MODELS)))
        for i, task in enumerate(TASKS):
            vals = []
            for m in MODELS:
                r = data.get((m, task))
                if not r:
                    vals.append(0.0)
                    continue
                if key == "regressed_run_share":
                    vals.append(int(r["regressed_runs"]) / int(r["runs"]))
                else:
                    vals.append(int(r["recoveries"]) / int(r["runs"]))
            ax.bar([x + (i - 1) * width for x in xs], vals, width,
                   label=TASK_LABELS[task], color=COLORS[task], alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(MODEL_LABELS, rotation=30, ha="right")
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("share of runs")
    axes[1].set_ylabel("count per run")
    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_process_profile.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_confidence():
    main = {r["model"]: r for r in load_csv(
        os.path.join(RAW, "results", "processed_judge", "confidence_by_position.csv"))}
    condb = {r["model"]: r for r in load_csv(
        os.path.join(RAW, "results_condB", "processed", "confidence_by_position.csv"))}
    qs = ["Q1", "Q2", "Q3", "Q4"]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for m in CONF_MODELS:
        vals = [float(main[m][q]) for q in qs]
        ax.plot(qs, vals, marker="o", ms=3, lw=1.2, color=CONF_COLORS[m], label=m)
    for m in ("deepseek7b", "deepseek14b", "deepseek32b"):
        if m not in condb:
            continue
        vals = [float(condb[m][q]) for q in qs]
        ax.plot(qs, vals, marker="s", ms=3, lw=1.0, ls="--", alpha=0.55,
                color=CONF_COLORS[m], label=f"{m} (CondB)")
    ax.set_xlabel("sentence-position quartile")
    ax.set_ylabel("mean self-reported confidence (0-100)")
    ax.set_ylim(60, 100)
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_confidence_by_position.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def copy_reused():
    out = []
    for name in ("report_fig_acc_by_model_task.png", "report_fig_condb.png"):
        src = os.path.join(REUSE_SRC, name)
        dst = os.path.join(OUT, name)
        shutil.copyfile(src, dst)
        out.append((src, dst))
    return out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    style()
    print(f"[plot_figs_v2] {fig_acc_ci()}")
    print(f"[plot_figs_v2] {fig_process_profile()}")
    print(f"[plot_figs_v2] {fig_confidence()}")
    for src, dst in copy_reused():
        print(f"[plot_figs_v2] copied {src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
