"""报告草稿包 v2：表格与中间数据的重算与固化（只读 cloud_backup 权威产物）。

输入（权威源 cloud_backup/，本地补跑件在仓库 data/、cloud_backup/*.txt）：
  cloud_backup/results/processed_judge/accuracy.csv           6770 行（8 档 × 280 story × 3 rep + bf16 50）
  cloud_backup/results/processed_judge/emergence.csv          同规模，含 emergence_point
  cloud_backup/results/processed_judge_official/accuracy.csv  4200 行（正式 5 档，§3.1 口径）
  cloud_backup/results/processed_judge_official/emergence.csv
  cloud_backup/results/processed/transition_stats.csv         回退/恢复汇总（28 行）
  cloud_backup/results/processed/trajectory_similarity.csv    每 run stability 列（6770 行）
  data/annotated/<story>.json                                 句数（frac_of_len 与末步判定）
  data/manual_equivalence/{labels,judge_sidecar,judge_exclude}.csv  判定准入复算

输出（docs/submission/report_draft_v2/）：
  tables/table1_acc_ci.csv                  模型×任务 judge-acc + story 级 cluster bootstrap 95% CI
  tables/table2_same_scale.csv              同规模 judge-acc 对比（Qwen vs DeepSeek，Δ）
  tables/table3_process_profile.csv         过程画像（正式 5 档 × 指标）
  tables/table4_emergence.csv               §3.1 口径涌现指标（official 4200）
  tables/table5_repetition_consistency.csv  story 级 3 次重复一致性
  tables/table6_source_subsets.csv          false_belief 内 OpenToM vs FauxPas 子集
  tables/table7_admission_quantization.csv  判定准入 + 量化对照（long 格式：block,scope,metric,value,note）
  data/acc_by_story.csv                     bootstrap 重采样单元（model×task×story 平均准确率）
  data/bootstrap_summary.csv                bootstrap 分布摘要（seed/迭代/均值/分位）
  data/repetition_detail.csv                重复明细（rep1..rep3 + all_equal）
  data/source_subset_runs.csv               false_belief 来源子集 run 明细
  data/quantization_detail.csv              bf16 与同 story AWQ 对照明细
  data/sources/<name>.csv                   草稿引用的源文件逐字副本（便于整包携带与核对）
  data/sources_manifest.csv                 源副本清单（原始路径/行数/sha256 前 16 位）

用法： python -X utf8 scripts/build_report_tables_v2.py
"""
import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = "cloud_backup"
OUT = os.path.join("docs", "submission", "report_draft_v2")
TABLES = os.path.join(OUT, "tables")
DATA = os.path.join(OUT, "data")
ANN = os.path.join("data", "annotated")
MANUAL = os.path.join("data", "manual_equivalence")

MAIN_MODELS = ("qwen05b_vllm", "qwen15b_vllm", "qwen7b", "qwen14b", "qwen32b",
               "deepseek7b", "deepseek14b", "deepseek32b")
FORMAL5 = ("qwen7b", "qwen14b", "deepseek7b", "deepseek14b", "deepseek32b")
TASKS = ("false_belief", "faux_pas", "implicature")
SCALE_PAIRS = (("7B", "qwen7b", "deepseek7b"),
               ("14B", "qwen14b", "deepseek14b"),
               ("32B", "qwen32b", "deepseek32b"))
# 草稿引用的源文件副本清单：(输出名, 原始路径)
SOURCE_FILES = (
    ("accuracy_processed_judge_6770.csv",
     "cloud_backup/results/processed_judge/accuracy.csv"),
    ("accuracy_processed_judge_official_4200.csv",
     "cloud_backup/results/processed_judge_official/accuracy.csv"),
    ("emergence_processed_judge_6770.csv",
     "cloud_backup/results/processed_judge/emergence.csv"),
    ("emergence_processed_judge_official_4200.csv",
     "cloud_backup/results/processed_judge_official/emergence.csv"),
    ("accuracy_processed_judge_condB_2520.csv",
     "cloud_backup/results/processed_judge_condB/accuracy.csv"),
    ("emergence_processed_judge_condB_2520.csv",
     "cloud_backup/results/processed_judge_condB/emergence.csv"),
    ("transition_stats_main.csv",
     "cloud_backup/results/processed/transition_stats.csv"),
    ("transition_stats_condB.csv",
     "cloud_backup/results_condB/processed/transition_stats.csv"),
    ("confidence_by_position_main.csv",
     "cloud_backup/results/processed_judge/confidence_by_position.csv"),
    ("confidence_by_position_condB.csv",
     "cloud_backup/results_condB/processed/confidence_by_position.csv"),
    ("trajectory_similarity_stability_6770.csv",
     "cloud_backup/results/processed/trajectory_similarity.csv"),
    ("manual_equivalence_labels.csv",
     "data/manual_equivalence/labels.csv"),
    ("manual_equivalence_judge_sidecar.csv",
     "data/manual_equivalence/judge_sidecar.csv"),
    ("manual_equivalence_judge_exclude.csv",
     "data/manual_equivalence/judge_exclude.csv"),
)

BOOT_ITERS = 2000
BOOT_SEED = 42

SENT_LEN = {}


def load_csv(path):
    with open(path, encoding="utf-8-sig") as fh:  # utf-8-sig 兼容 BOM（judge_sidecar.csv）
        return list(csv.DictReader(fh))


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[build_tables] wrote {path} ({len(rows)} rows)")


def sent_len(story_id):
    if story_id not in SENT_LEN:
        try:
            with open(os.path.join(ANN, f"{story_id}.json"), encoding="utf-8") as fh:
                SENT_LEN[story_id] = len(json.load(fh).get("sentences", []))
        except Exception:
            SENT_LEN[story_id] = -1
    return SENT_LEN[story_id]


def is_yes(v):
    return str(v).strip() == "1.0"


def acc_of(rows):
    return sum(1 for r in rows if is_yes(r["accuracy"])) / len(rows) if rows else float("nan")


# ---------------------------------------------------------------- T1: acc + CI
def build_acc_ci(rows_main):
    """模型×任务准确率与 story 级 cluster bootstrap 95% CI。"""
    by_story = defaultdict(list)          # (model,task,story) -> [1/0...]
    for r in rows_main:
        if r["model"] not in MAIN_MODELS:
            continue
        by_story[(r["model"], r["task"], r["story_id"])].append(1.0 if is_yes(r["accuracy"]) else 0.0)

    story_rows = [{"model": m, "task": t, "story_id": s, "n_runs": len(v),
                   "acc": round(sum(v) / len(v), 6)}
                  for (m, t, s), v in sorted(by_story.items())]
    write_csv(os.path.join(DATA, "acc_by_story.csv"),
              ["model", "task", "story_id", "n_runs", "acc"], story_rows)

    per_cell = defaultdict(list)          # (model,task) -> [(story_acc, n_runs)]
    for r in story_rows:
        per_cell[(r["model"], r["task"])].append((r["acc"], r["n_runs"]))

    rng = np.random.default_rng(BOOT_SEED)
    out, boot_summary = [], []
    for model in MAIN_MODELS:
        for task in TASKS:
            cells = per_cell.get((model, task))
            if not cells:
                continue
            accs = np.array([a for a, _ in cells])
            ns = np.array([n for _, n in cells], dtype=float)
            boot = np.empty(BOOT_ITERS)
            for i in range(BOOT_ITERS):
                idx = rng.integers(0, len(accs), len(accs))
                boot[i] = float(np.average(accs[idx], weights=ns[idx]))
            acc = float(np.average(accs, weights=ns))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append({"model": model, "task": task, "n_runs": int(ns.sum()),
                        "n_stories": len(cells), "acc": round(acc, 4),
                        "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
            boot_summary.append({"model": model, "task": task, "iterations": BOOT_ITERS,
                                 "seed": BOOT_SEED, "acc": round(acc, 6),
                                 "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
                                 "boot_mean": round(float(boot.mean()), 6),
                                 "boot_sd": round(float(boot.std(ddof=1)), 6)})
    write_csv(os.path.join(TABLES, "table1_acc_ci.csv"),
              ["model", "task", "n_runs", "n_stories", "acc", "ci_lo", "ci_hi"], out)
    write_csv(os.path.join(DATA, "bootstrap_summary.csv"),
              ["model", "task", "iterations", "seed", "acc", "ci_lo", "ci_hi",
               "boot_mean", "boot_sd"], boot_summary)
    return {(r["model"], r["task"]): r for r in out}


# --------------------------------------------------------- T2: same-scale gaps
def build_same_scale(rows_main, table1):
    def overall(model):
        sub = [r for r in rows_main if r["model"] == model]
        return acc_of(sub), len(sub)

    out = []
    for scale, qm, dm in SCALE_PAIRS:
        qa, qn = overall(qm)
        da, dn = overall(dm)
        out.append({"scale": scale, "qwen_model": qm, "qwen_acc": round(qa, 4), "qwen_n": qn,
                    "ds_model": dm, "ds_acc": round(da, 4), "ds_n": dn,
                    "delta_ds_minus_qwen": round(da - qa, 4)})
    write_csv(os.path.join(TABLES, "table2_same_scale.csv"),
              ["scale", "qwen_model", "qwen_acc", "qwen_n", "ds_model", "ds_acc", "ds_n",
               "delta_ds_minus_qwen"], out)
    return out


# ------------------------------------------------------- T3: process profile
def build_process_profile(rows_main, rows_official, rows_trans, rows_stab):
    def rate_official(model):
        sub = [r for r in rows_official if r["model"] == model]
        valid = [r for r in sub if r["emergence_point"] != ""]
        if not sub:
            return None
        fracs, last = [], 0
        for r in valid:
            n = sent_len(r["story_id"])
            if n > 0:
                fracs.append(int(r["emergence_point"]) / n)
                last += 1 if int(r["emergence_point"]) == n else 0
        return {"n": len(sub), "valid": len(valid), "valid_rate": len(valid) / len(sub),
                "last_share": last / len(valid) if valid else float("nan"),
                "frac_of_len": statistics.mean(fracs) if fracs else float("nan")}

    trans = defaultdict(lambda: {"runs": 0, "regressed": 0, "recoveries": 0})
    for r in rows_trans:
        if r["model"] not in FORMAL5:
            continue
        d = trans[r["model"]]
        d["runs"] += int(r["runs"])
        d["regressed"] += int(r["regressed_runs"])
        d["recoveries"] += int(r["recoveries"])
    stab = defaultdict(list)
    for r in rows_stab:
        if r["model"] in FORMAL5:
            stab[r["model"]].append(float(r["stability"]))

    out, detail = [], []
    for model in FORMAL5:
        e = rate_official(model)
        tr = trans[model]
        row = {"model": model,
               "acc_overall": round(acc_of([r for r in rows_main if r["model"] == model]), 4),
               "valid_emergence_rate": round(e["valid_rate"], 4) if e else "",
               "last_step_share": round(e["last_share"], 4) if e else "",
               "frac_of_len": round(e["frac_of_len"], 4) if e else "",
               "regressed_run_share": round(tr["regressed"] / tr["runs"], 4),
               "recoveries_per_run": round(tr["recoveries"] / tr["runs"], 4),
               "stability_mean": round(statistics.mean(stab[model]), 4)}
        out.append(row)
        for task in TASKS:
            ts = [r for r in rows_trans if r["model"] == model and r["task"] == task]
            if ts:
                detail.append({"model": model, "task": task, "runs": ts[0]["runs"],
                               "regression_rate": ts[0]["regression_rate"],
                               "recoveries_per_run": round(int(ts[0]["recoveries"]) / int(ts[0]["runs"]), 4),
                               "final_acc": ts[0]["final_acc"]})
    write_csv(os.path.join(TABLES, "table3_process_profile.csv"),
              ["model", "acc_overall", "valid_emergence_rate", "last_step_share", "frac_of_len",
               "regressed_run_share", "recoveries_per_run", "stability_mean"], out)
    write_csv(os.path.join(DATA, "process_profile_detail.csv"),
              ["model", "task", "runs", "regression_rate", "recoveries_per_run", "final_acc"], detail)
    return out


# ---------------------------------------------------------- T4: emergence §3.1
def build_emergence(rows_official):
    out = []
    for model in FORMAL5:
        for task in TASKS:
            sub = [r for r in rows_official if r["model"] == model and r["task"] == task]
            if not sub:
                continue
            valid = [r for r in sub if r["emergence_point"] != ""]
            poss = [int(r["emergence_point"]) for r in valid]
            ns = [sent_len(r["story_id"]) for r in valid]
            fracs = [p / n for p, n in zip(poss, ns) if n > 0]
            last = sum(1 for p, n in zip(poss, ns) if n > 0 and p == n)
            out.append({"model": model, "task": task, "n": len(sub), "valid": len(valid),
                        "valid_rate": round(len(valid) / len(sub), 4),
                        "mean_pos": round(statistics.mean(poss), 2) if poss else "",
                        "median_pos": round(statistics.median(poss), 1) if poss else "",
                        "last_step_rate": round(last / len(valid), 4) if valid else "",
                        "frac_of_len": round(statistics.mean(fracs), 4) if fracs else ""})
    write_csv(os.path.join(TABLES, "table4_emergence.csv"),
              ["model", "task", "n", "valid", "valid_rate", "mean_pos", "median_pos",
               "last_step_rate", "frac_of_len"], out)
    return out


# ---------------------------------------------- T5: repetition consistency
def build_repetition(rows_main):
    groups = defaultdict(dict)            # (model,task,story) -> {rep: acc}
    for r in rows_main:
        if r["model"] not in MAIN_MODELS:
            continue
        groups[(r["model"], r["task"], r["story_id"])][r["repetition"]] = 1.0 if is_yes(r["accuracy"]) else 0.0

    detail = []
    for (m, t, s), reps in sorted(groups.items()):
        vals = [reps.get(str(k), "") for k in (1, 2, 3)]
        known = [v for v in vals if v != ""]
        detail.append({"model": m, "task": t, "story_id": s,
                       "rep1": vals[0], "rep2": vals[1], "rep3": vals[2],
                       "all_equal": int(len(set(known)) == 1 and len(known) == 3)})
    write_csv(os.path.join(DATA, "repetition_detail.csv"),
              ["model", "task", "story_id", "rep1", "rep2", "rep3", "all_equal"], detail)

    agg = defaultdict(lambda: [0, 0])
    agg_task = defaultdict(lambda: [0, 0])
    for r in detail:
        for key in ((r["model"], "ALL"), (r["model"], r["task"])):
            agg[key][0] += 1
            agg[key][1] += 1 - r["all_equal"]
    for key in sorted(agg):
        agg_task[key] = agg[key]
    out = [{"model": m, "task": t, "n_stories": v[0], "inconsistent": v[1],
            "inconsistency_rate": round(v[1] / v[0], 4)}
           for (m, t), v in sorted(agg_task.items())]
    write_csv(os.path.join(TABLES, "table5_repetition_consistency.csv"),
              ["model", "task", "n_stories", "inconsistent", "inconsistency_rate"], out)
    return out


# ------------------------------------------------------ T6: source subsets
def build_source_subsets(rows_main):
    out, detail = [], []
    for model in MAIN_MODELS:
        for subset, pred in (("opentom_fb", lambda s: s.startswith("fb")),
                             ("fauxpas_control_sfp", lambda s: s.startswith("sfp"))):
            sub = [r for r in rows_main
                   if r["model"] == model and r["task"] == "false_belief" and pred(r["story_id"])]
            for r in sub:
                detail.append({"model": model, "story_id": r["story_id"], "subset": subset,
                               "repetition": r["repetition"], "accuracy": r["accuracy"]})
            out.append({"model": model, "subset": subset, "n_runs": len(sub),
                        "n_stories": len({r["story_id"] for r in sub}),
                        "acc": round(acc_of(sub), 4)})
    write_csv(os.path.join(TABLES, "table6_source_subsets.csv"),
              ["model", "subset", "n_runs", "n_stories", "acc"], out)
    write_csv(os.path.join(DATA, "source_subset_runs.csv"),
              ["model", "story_id", "subset", "repetition", "accuracy"], detail)
    return out


# ------------------------------------------- T7: judge admission + quantization
def build_admission_quantization(rows_main):
    rows = load_csv(os.path.join(MANUAL, "labels.csv"))
    side = {r["id"]: r for r in load_csv(os.path.join(MANUAL, "judge_sidecar.csv"))}
    excluded = {r["id"] for r in load_csv(os.path.join(MANUAL, "judge_exclude.csv"))}

    def parse(v):
        s = str(v).strip().lower()
        return True if s in ("1", "true", "yes") else False if s in ("0", "false", "no") else None

    def metrics(h, j):
        n = len(h)
        tp = sum(1 for a, b in zip(h, j) if a and b)
        fp = sum(1 for a, b in zip(h, j) if not a and b)
        fn = sum(1 for a, b in zip(h, j) if a and not b)
        tn = sum(1 for a, b in zip(h, j) if not a and not b)
        po = (tp + tn) / n
        pe = (tp + fp) / n * (tp + fn) / n + (tn + fn) / n * (tn + fp) / n
        kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
        return {"n": n, "acc": (tp + tn) / n, "prec": tp / (tp + fp) if tp + fp else float("nan"),
                "rec": tp / (tp + fn) if tp + fn else float("nan"), "kappa": kappa}

    out = []
    for col in ("judge_api_deepseek", "judge_local"):
        humans, judges = [], []
        for r in rows:
            if r["id"] in excluded:
                continue
            h = parse(r.get("human_label"))
            sval = (side.get(r["id"]) or {}).get(col)
            j = parse(sval if sval is not None and str(sval).strip() else r.get(col))
            if h is None or j is None:
                continue
            humans.append(h)
            judges.append(j)
        m = metrics(humans, judges)
        note = ("primary judge, admission basis (58 valid of 64; 4 empty-answer rows excluded)"
                if col == "judge_api_deepseek" else "local proxy judge, contrast only")
        out.append({"block": "judge_admission", "scope": col, "metric": "n", "value": m["n"], "note": note})
        for k in ("acc", "prec", "rec", "kappa"):
            out.append({"block": "judge_admission", "scope": col, "metric": k,
                        "value": round(m[k], 3), "note": note})

    quant_rows = []
    for model in ("qwen14b", "deepseek14b"):
        bf = [r for r in rows_main if r["model"] == f"{model}_bf16"]
        keys = {(r["task"], r["story_id"]) for r in bf}
        awq_all = [r for r in rows_main if r["model"] == model and (r["task"], r["story_id"]) in keys]
        awq_rep1 = [r for r in awq_all if r["repetition"] == "1"]
        for r in bf:
            quant_rows.append({"pair": f"{model}_bf16_vs_awq", "scope": "bf16_run", "task": r["task"],
                               "story_id": r["story_id"], "repetition": r["repetition"],
                               "accuracy": r["accuracy"]})
        for r in awq_all:
            quant_rows.append({"pair": f"{model}_bf16_vs_awq", "scope": "awq_run", "task": r["task"],
                               "story_id": r["story_id"], "repetition": r["repetition"],
                               "accuracy": r["accuracy"]})
        bfa, a75, a25 = acc_of(bf), acc_of(awq_all), acc_of(awq_rep1)
        note = (f"{len(bf)} bf16 runs on {len(keys)} stories vs {len(awq_all)} AWQ runs "
                f"(same stories, 3 reps) and {len(awq_rep1)} AWQ rep1 runs")
        out.append({"block": "quantization", "scope": model, "metric": "bf16_acc",
                    "value": round(bfa, 4), "note": note})
        out.append({"block": "quantization", "scope": model, "metric": "awq_acc_same_stories_75",
                    "value": round(a75, 4), "note": note})
        out.append({"block": "quantization", "scope": model, "metric": "awq_acc_rep1_25",
                    "value": round(a25, 4), "note": note})
        out.append({"block": "quantization", "scope": model, "metric": "diff_bf16_minus_awq75",
                    "value": round(bfa - a75, 4), "note": note})
        out.append({"block": "quantization", "scope": model, "metric": "diff_bf16_minus_awq_rep1",
                    "value": round(bfa - a25, 4), "note": note})
    write_csv(os.path.join(TABLES, "table7_admission_quantization.csv"),
              ["block", "scope", "metric", "value", "note"], out)
    write_csv(os.path.join(DATA, "quantization_detail.csv"),
              ["pair", "scope", "task", "story_id", "repetition", "accuracy"], quant_rows)
    return out


# ------------------------------------------- 源文件副本（自包含）
def sync_sources():
    """把草稿引用的源 CSV 复制进 data/sources/，并写清单（原始路径/行数/sha256 前 16 位）。"""
    dst_dir = os.path.join(DATA, "sources")
    os.makedirs(dst_dir, exist_ok=True)
    manifest = []
    for name, src in SOURCE_FILES:
        dst = os.path.join(dst_dir, name)
        shutil.copyfile(src, dst)
        with open(dst, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()[:16]
        with open(dst, encoding="utf-8-sig", newline="") as fh:
            rows = sum(1 for _ in csv.reader(fh)) - 1
        manifest.append({"copy": f"data/sources/{name}", "source": src,
                         "rows": rows, "sha256_16": digest})
    write_csv(os.path.join(DATA, "sources_manifest.csv"),
              ["copy", "source", "rows", "sha256_16"], manifest)
    return manifest


# ------------------------------------------------------------------- checks
def checks(rows_main, table1, scale, emerg, reps, subset, adm):
    def get(tbl, model, task=None):
        for r in tbl:
            if r["model"] == model and (task is None or r["task"] == task):
                return r
    print("[build_tables] ---- 与 experiment_report 对表（抽检）----")
    exp_overall = {"qwen7b": 0.496, "qwen14b": 0.567, "qwen32b": 0.548,
                   "deepseek7b": 0.139, "deepseek14b": 0.538, "deepseek32b": 0.604}
    for m, e in exp_overall.items():
        got = acc_of([r for r in rows_main if r["model"] == m])
        print(f"  acc[{m}]={got:.3f} expected {e:.3f} {'OK' if abs(got-e)<=0.001 else 'MISMATCH'}")
    for r in scale:
        print(f"  Δ[{r['scale']}]={r['delta_ds_minus_qwen']:+.3f} raw {r['ds_acc']:.3f}-{r['qwen_acc']:.3f}")
    exp_emerg = {("qwen7b", "false_belief"): 0.97, ("qwen7b", "implicature"): 0.60,
                 ("deepseek7b", "faux_pas"): 0.57, ("deepseek32b", "implicature"): 0.65}
    for (m, t), e in exp_emerg.items():
        g = get(emerg, m, t)
        ok = abs(g["valid_rate"] - e) <= 0.011 if g else False
        print(f"  valid_rate[{m},{t}]={g['valid_rate'] if g else 'NA'} expected {e} {'OK' if ok else 'CHECK'}")
    for m in ("qwen7b", "deepseek14b", "qwen14b", "deepseek32b"):
        g = next((r for r in reps if r["model"] == m and r["task"] == "ALL"), None)
        print(f"  repetition inconsistency[{m}]={g['inconsistency_rate'] if g else 'NA'}")
    for m in ("deepseek32b", "qwen7b"):
        for r in subset:
            if r["model"] == m:
                print(f"  subset[{m},{r['subset']}] n={r['n_runs']} acc={r['acc']}")
    for r in adm:
        if r["block"] == "judge_admission" and r["scope"] == "judge_api_deepseek":
            print(f"  admission[{r['metric']}]={r['value']}")
        if r["block"] == "quantization" and r["metric"].startswith("diff"):
            print(f"  quant[{r['scope']}][{r['metric']}]={r['value']}")


def main() -> int:
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    rows_main = load_csv(os.path.join(RAW, "results", "processed_judge", "accuracy.csv"))
    rows_official = load_csv(os.path.join(RAW, "results", "processed_judge_official", "accuracy.csv"))
    rows_emg_official = load_csv(os.path.join(RAW, "results", "processed_judge_official", "emergence.csv"))
    rows_trans = load_csv(os.path.join(RAW, "results", "processed", "transition_stats.csv"))
    rows_stab = load_csv(os.path.join(RAW, "results", "processed", "trajectory_similarity.csv"))
    print(f"[build_tables] main={len(rows_main)} official={len(rows_official)} "
          f"trans={len(rows_trans)} stability={len(rows_stab)}")

    t1 = build_acc_ci(rows_main)
    t2 = build_same_scale(rows_main, t1)
    build_process_profile(rows_main, rows_emg_official, rows_trans, rows_stab)
    t4 = build_emergence(rows_emg_official)
    t5 = build_repetition(rows_main)
    t6 = build_source_subsets(rows_main)
    t7 = build_admission_quantization(rows_main)
    sync_sources()
    checks(rows_main, t1, t2, t4, t5, t6, t7)
    return 0


if __name__ == "__main__":
    sys.exit(main())
