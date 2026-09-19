"""分析处理管线（pipeline.md 十一节 / scaf.md 10-12 节，裁决 C12 产物映射）。

run_analysis：
1. 读取 results/raw/{model}/{task}/{story_id}/run{N}.json（JSONL）
2. 回填 confidence（raw → processed，scaf.md Principle 3）
3. 计算指标：accuracy / emergence / stability / trajectory_similarity
4. 产出 results/processed/：accuracy.csv、emergence.csv、trajectory_similarity.csv、
   confidence_curve.png、emergence_distribution.png、trajectory.png、statistics.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.accuracy import calculate_accuracy
from analysis.confidence import extract_confidence
from analysis.embedding import EmbeddingService
from analysis.emergence import calculate_emergence
from analysis.stability import calculate_stability, step_distances
from analysis.statistics import run_statistics, trajectory_similarity
from config.config_manager import ConfigError
from visualization import plots

REQUIRED_COLS = ("model", "task", "story_id", "step", "response", "repetition")


# 任务的 authoritative 来源是标注文件的 task 字段（裁决 C7/C11）。历史 run 可能因
# 中途改标注/错分组，把某个 story 的 run 落到了与当前标注 task 不一致的目录下，
# 导致 accuracy/emergence 聚合污染（实测：sfp011-sfp020 曾落入 false_belief）。
# analyze 应信任标注真值过滤掉此类行，避免跨任务污染；本函数仅过滤，不改动 raw 文件。
def _resolve_task(data_manager, story_id: str) -> str | None:
    """返回 story 当前标注的 task；标注缺失时返回 None（交由调用方决定）。"""
    try:
        return data_manager.load_annotation(story_id)["task"]
    except Exception:
        return None


def _filter_task_drift(records: list[dict], data_manager, logger=None) -> list[dict]:
    """剔除 run 记录 task 与当前标注 task 不一致的行（脏路由），并记告警。

    标注缺失的记录保留（旧 analyze 行为：交由 load_annotation 抛错或正常处理），
    仅丢弃能确证冲突的行。
    """
    ok: list[dict] = []
    dropped = 0
    for rec in records:
        canonical = _resolve_task(data_manager, rec["story_id"])
        if canonical is not None and canonical != rec.get("task"):
            dropped += 1
            continue
        ok.append(rec)
    if dropped and logger is not None:
        logger.warning(f"analysis dropped {dropped} records whose run task does not match "
                       f"their annotation task (task-drift filter)")
    return ok


def load_records(results_dir: str | Path) -> list[dict]:
    """遍历 results/raw 下全部 JSONL run 文件。"""
    raw_dir = Path(results_dir) / "raw"
    if not raw_dir.exists():
        return []
    records: list[dict] = []
    for path in sorted(raw_dir.rglob("run*.json")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _group(records: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        groups.setdefault(tuple(rec[k] for k in keys), []).append(rec)
    return groups


def run_analysis(results_dir: str | Path, data_manager, config, logger=None,
                 allow_models: set[str] | None = None, embedder=None,
                 judge_fn=None) -> dict:
    """执行完整分析；返回产物路径映射。

    allow_models：若给定，仅保留属于该模型名集合的 run 记录（用于 CLI 排除 mock/
    杂散模型，避免把测试夹具混入科研口径；None 表示不作模型过滤，供单测直接用 mock）。
    embedder：若给定（如确定性 HashEmbedder），注入 EmbeddingService，使单元/回归
    测试无需联网下载 HF 编码器；None 表示用 config 指定的 embeddings 编码器（科研口径）。
    judge_fn：scoring.method 为 local_judge/api_judge 时必需；签名 judge_fn(question, answer_norm, gold)
    -> True/False/None。默认 None 即走 embedding_similarity（余弦）。
    """
    from config.config_manager import ConfigManager

    cfg: ConfigManager = config
    scoring = cfg.get("scoring")
    method = scoring.get("method", "embedding_similarity")
    use_judge = method in ("local_judge", "api_judge")
    if use_judge and judge_fn is None:
        raise RuntimeError(
            f"scoring.method={method} 需要 judge_fn；请装配 LLMEquivalenceJudge 后传入，"
            f"或把 experiment.yaml scoring.method 改回 embedding_similarity。"
        )
    service = EmbeddingService(scoring["embedding_model"],
                                cache_dir=str(Path(results_dir) / "embeddings"),
                                embedder=embedder)
    threshold = scoring["threshold"]

    records = load_records(results_dir)
    if not records:
        raise RuntimeError(f"no raw results found under {results_dir}/raw")

    # 任务漂移过滤（裁决 C11 权威 task 在标注文件；run 的 task 目录不一致时剔除）
    records = _filter_task_drift(records, data_manager, logger=logger)
    if allow_models is not None:
        kept = [r for r in records if r.get("model") in allow_models]
        dropped = len(records) - len(kept)
        if dropped and logger is not None:
            logger.warning(f"analysis dropped {dropped} records from models not in "
                           f"allow_models (mock/杂散): {sorted({r.get('model') for r in records} - allow_models)}")
        records = kept
    if not records:
        raise RuntimeError(
            f"all raw results under {results_dir}/raw were dropped by filters "
            f"(task-drift or allow_models); nothing to analyze"
        )

    # 1. 回填 confidence（processed 层）
    for rec in records:
        rec["confidence"] = extract_confidence(rec["response"])

    processed_dir = Path(results_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. 指标计算（model × task × story × repetition）
    acc_rows, emg_rows, stb_rows = [], [], []
    stability_by_model: dict[str, list[tuple[int, float]]] = {}
    for key, group in _group(records, ("model", "task", "story_id", "repetition")).items():
        model, task, story_id, repetition = key
        ann = data_manager.load_annotation(story_id)
        # gold 优先取多参考关键点 gold_points（P1-4），缺失回退单条 gold_answer
        gold = ann.get("gold_points") or ann.get("gold_answer")
        if not gold:
            raise KeyError(f"annotation {story_id} has no gold_answer/gold_points")
        question = ann.get("question")
        last = max(group, key=lambda r: r["step"])
        acc = calculate_accuracy(last["response"], gold, service, threshold,
                                 judge_fn=judge_fn, question=question)
        emg = calculate_emergence(group, gold, service, threshold,
                                  judge_fn=judge_fn, question=question)
        stb = calculate_stability(group, service)
        acc_rows.append({"model": model, "task": task, "story_id": story_id,
                         "repetition": repetition, "accuracy": acc})
        emg_rows.append({"model": model, "task": task, "story_id": story_id,
                         "repetition": repetition, "emergence_point": emg})
        stb_rows.append({"model": model, "task": task, "story_id": story_id,
                         "repetition": repetition, "stability": stb})
        for step, dist in step_distances(group, service):
            stability_by_model.setdefault(model, []).append((step, dist))

    acc_df = pd.DataFrame(acc_rows)
    emg_df = pd.DataFrame(emg_rows)
    stb_df = pd.DataFrame(stb_rows)

    # 3. 产物（裁决 C12）
    acc_df.to_csv(processed_dir / "accuracy.csv", index=False)
    emg_df.to_csv(processed_dir / "emergence.csv", index=False)
    stb_df.to_csv(processed_dir / "trajectory_similarity.csv", index=False)

    # confidence 曲线数据
    conf_rows = [
        {"model": r["model"], "story_id": r["story_id"], "repetition": r["repetition"],
         "sentence_position": r["step"], "confidence": r["confidence"]}
        for r in records
    ]
    conf_df = pd.DataFrame(conf_rows)

    outputs = {
        "accuracy.csv": processed_dir / "accuracy.csv",
        "emergence.csv": processed_dir / "emergence.csv",
        "trajectory_similarity.csv": processed_dir / "trajectory_similarity.csv",
        "confidence_curve.png": plots.plot_confidence(
            conf_df, processed_dir / "confidence_curve.png"),
        "emergence_distribution.png": plots.plot_emergence(
            emg_df, processed_dir / "emergence_distribution.png"),
        "trajectory.png": plots.plot_trajectory(
            stability_by_model, processed_dir / "trajectory.png"),
    }

    # 4. 跨模型轨迹相似度（proj.md Metric 5）
    ts_rows = []
    models = sorted({r["model"] for r in records})
    if len(models) >= 2:
        by_model = {m: [r for r in records if r["model"] == m] for m in models}
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                sim = trajectory_similarity(by_model[ma], by_model[mb], service)
                ts_rows.append({"model_a": ma, "model_b": mb, "similarity": sim})
        # 修复：跨模型表单独落盘，避免覆盖上面的 per-run stability 表（trajectory_similarity.csv）
        ts_path = processed_dir / "trajectory_similarity_models.csv"
        pd.DataFrame(ts_rows).to_csv(ts_path, index=False)
        outputs["trajectory_similarity_models.csv"] = ts_path

    # 5. 统计（数据充足且 statsmodels 可用时）
    metrics = acc_df.merge(emg_df, on=["model", "task", "story_id", "repetition"], how="left")
    metrics = metrics.merge(stb_df, on=["model", "task", "story_id", "repetition"], how="left")
    # size 来自 models.yaml（裁决 C14）；测试用 mock 模型等未知键容忍为空
    model_sizes = {}
    for m in metrics["model"].unique():
        try:
            model_sizes[m] = cfg.get("models", m).get("size", "")
        except ConfigError:
            model_sizes[m] = ""
    metrics["size"] = metrics["model"].map(model_sizes)
    stats = run_statistics(metrics, logger)
    if stats:
        stat_path = processed_dir / "statistics.txt"
        stat_path.write_text("\n\n".join(stats.values()), encoding="utf-8")
        outputs["statistics.txt"] = stat_path

    if logger is not None:
        logger.info(f"analysis done: {len(records)} records, "
                    f"{len(acc_df)} runs, outputs in {processed_dir}")
    return outputs
