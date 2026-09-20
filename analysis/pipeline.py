"""Analysis pipeline (pipeline.md section 11 / scaf.md sections 10-12; artifact mapping from ruling C12).

run_analysis:
1. read results/raw/{model}/{task}/{story_id}/run{N}.json (JSONL)
2. backfill confidence (raw → processed; scaf.md Principle 3)
3. compute metrics: accuracy / emergence / stability / trajectory_similarity
4. write results/processed/: accuracy.csv, emergence.csv, trajectory_similarity.csv,
   confidence_curve.png, emergence_distribution.png, trajectory.png, statistics.txt
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


# The authoritative source of a task is the task field of the annotation file (rulings C7/C11).
# Historical runs may have landed a story's run in a directory inconsistent with its current
# annotation task (mid-way annotation edits or mis-grouping), polluting the accuracy/emergence
# aggregates (measured: sfp011-sfp020 once fell into false_belief). analyze should trust the
# annotation truth and filter such rows out; this function only filters and never edits raw files.
def _resolve_task(data_manager, story_id: str) -> str | None:
    """Return the task of the story's current annotation; None when the annotation is missing (the caller decides what to do)."""
    try:
        return data_manager.load_annotation(story_id)["task"]
    except Exception:
        return None


def _filter_task_drift(records: list[dict], data_manager, logger=None) -> list[dict]:
    """Drop run rows whose task does not match their current annotation task (dirty routing) and log a warning.

    Rows with a missing annotation are kept (old analyze behaviour: either load_annotation raises or they pass through);
    only rows with a proven conflict are discarded.
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
    """Walk every JSONL run file under results/raw."""
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
    """Run the full analysis; returns a mapping of output paths.

    allow_models: when given, keep only run records whose model is in the set (the CLI uses this
    to exclude mock/stray models so test fixtures never mix into the research protocol; None skips
    model filtering, which lets unit tests use mock models directly).
    embedder: when given (e.g. the deterministic HashEmbedder), it is injected into EmbeddingService
    so unit/regression tests need no network to download the HF encoder; None uses the encoder
    named in the config (research protocol).
    judge_fn: required when scoring.method is local_judge/api_judge; signature judge_fn(question, answer_norm, gold)
    -> True/False/None. The default None goes through embedding_similarity (cosine).
    """
    from config.config_manager import ConfigManager

    cfg: ConfigManager = config
    scoring = cfg.get("scoring")
    method = scoring.get("method", "embedding_similarity")
    use_judge = method in ("local_judge", "api_judge")
    if use_judge and judge_fn is None:
        raise RuntimeError(
            f"scoring.method={method} requires judge_fn; wire in LLMEquivalenceJudge and pass it, "
            f"or set scoring.method back to embedding_similarity in experiment.yaml."
        )
    service = EmbeddingService(scoring["embedding_model"],
                                cache_dir=str(Path(results_dir) / "embeddings"),
                                embedder=embedder)
    threshold = scoring["threshold"]

    records = load_records(results_dir)
    if not records:
        raise RuntimeError(f"no raw results found under {results_dir}/raw")

    # task-drift filter (ruling C11: the authoritative task lives in the annotation file; rows whose run directory disagrees are dropped)
    records = _filter_task_drift(records, data_manager, logger=logger)
    if allow_models is not None:
        kept = [r for r in records if r.get("model") in allow_models]
        dropped = len(records) - len(kept)
        if dropped and logger is not None:
            logger.warning(f"analysis dropped {dropped} records from models not in "
                           f"allow_models (mock/stray): {sorted({r.get('model') for r in records} - allow_models)}")
        records = kept
    if not records:
        raise RuntimeError(
            f"all raw results under {results_dir}/raw were dropped by filters "
            f"(task-drift or allow_models); nothing to analyze"
        )

    # 1. backfill confidence (processed layer)
    for rec in records:
        rec["confidence"] = extract_confidence(rec["response"])

    processed_dir = Path(results_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. metric computation (model × task × story × repetition)
    acc_rows, emg_rows, stb_rows = [], [], []
    stability_by_model: dict[str, list[tuple[int, float]]] = {}
    for key, group in _group(records, ("model", "task", "story_id", "repetition")).items():
        model, task, story_id, repetition = key
        ann = data_manager.load_annotation(story_id)
        # prefer the multi-reference gold_points (P1-4); fall back to the single gold_answer when absent
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

    # 3. artifacts (ruling C12)
    acc_df.to_csv(processed_dir / "accuracy.csv", index=False)
    emg_df.to_csv(processed_dir / "emergence.csv", index=False)
    stb_df.to_csv(processed_dir / "trajectory_similarity.csv", index=False)

    # confidence curve data
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

    # 4. cross-model trajectory similarity (proj.md metric 5)
    ts_rows = []
    models = sorted({r["model"] for r in records})
    if len(models) >= 2:
        by_model = {m: [r for r in records if r["model"] == m] for m in models}
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                sim = trajectory_similarity(by_model[ma], by_model[mb], service)
                ts_rows.append({"model_a": ma, "model_b": mb, "similarity": sim})
        # fix: keep the cross-model table in its own file so it does not overwrite the per-run stability table above (trajectory_similarity.csv)
        ts_path = processed_dir / "trajectory_similarity_models.csv"
        pd.DataFrame(ts_rows).to_csv(ts_path, index=False)
        outputs["trajectory_similarity_models.csv"] = ts_path

    # 5. statistics (when there is enough data and statsmodels is available)
    metrics = acc_df.merge(emg_df, on=["model", "task", "story_id", "repetition"], how="left")
    metrics = metrics.merge(stb_df, on=["model", "task", "story_id", "repetition"], how="left")
    # size comes from models.yaml (ruling C14); unknown keys such as test mock models are tolerated as empty
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
