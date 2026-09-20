"""Cloud / large-GPU batch runner: runs all (model × task × story × rep) incremental experiments concurrently in "lock-step rounds".

Why it is faster than sequential main.py run: each round batches the same relative step of every
story still in progress into one vLLM generate_batch call (continuous batching), saturating an
80/90-series card; together with enable_prefix_caching, adjacent rounds of the same story share
their prefix, so throughput improves by roughly an order of magnitude.

Features:
- resumable: completed runs are skipped; partial runs resume from the saved line count (step rows are appended; nothing is duplicated or lost).
- deterministic: every (model, story, rep) has a fixed job seed, and each round uses seed = job_seed + round*7919.
- OOM: vLLM memory errors are converted to explicit guidance and exit code 2; nothing is silent.

Usage (on the cloud, with Python and vLLM ready):
    python scripts/cloud_run.py                    # default full matrix: qwen7b,deepseek7b,qwen14b,deepseek14b,deepseek32b
    python scripts/cloud_run.py --models qwen7b    # restrict to a subset (for preflight)
    python scripts/cloud_run.py --stories fb001,fb002
    # rerun / resume with the same command; complete runs are skipped idempotently.
"""
import argparse, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ensure the repo root modules import from any cwd (on AutoDL it is common to run python scripts/... directly)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_manager import ConfigManager
from data_manager import DataManager
from experiment import PromptBuilder, ResultRecorder
from models import ModelFactory
from utils.seeding import job_seed

STEP_SEED_STRIDE = 7919

# the official 24G single-card full matrix (RQ1-RQ3): all models by default; --models overrides with a subset
DEFAULT_MODELS = ["qwen7b", "deepseek7b", "qwen14b", "deepseek14b", "deepseek32b"]


def _count_saved(path) -> int:
    if not path.exists():
        return 0
    return len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])


def _collect_stories(dm: DataManager, tasks, stories, include_unreviewed, logger) -> dict[str, list[dict]]:
    import glob

    by_task: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(os.path.join("data", "annotated", "*.json"))):
        sid = os.path.basename(p)[:-5]
        ann = dm.load_annotation(sid)
        if ann.get("task") not in tasks:
            continue
        if not ann.get("question"):
            print(f"  [skip] {sid}: no question; not included in the cloud run", flush=True)
            continue
        if ann.get("reviewed") is not True and not include_unreviewed:
            print(f"  [skip] {sid}: not reviewed (pass --include-unreviewed to include it)", flush=True)
            continue
        if stories and sid not in stories:
            continue
        by_task.setdefault(ann["task"], []).append(ann)
    return by_task


def run_model_batch(model_name: str, jobs, cfg, dm, builder, model, seed_master,
                    condition: str = "condition_a") -> int:
    """Lock-step batch execution for one model. Returns the number of finished runs. condition is passed to PromptBuilder (condition_a/b)."""
    experiment = cfg.get("experiment")
    temperature = experiment["temperature"]
    recorder = ResultRecorder(dm)
    rec_total = 0
    done_total = 0
    # jobs: list of (task, story_dict, rep)
    active = []
    for task, story, rep in jobs:
        n = len(story["sentences"])
        if recorder.is_complete(model_name, task, story["id"], rep, n):
            done_total += 1
            continue
        saved = 0
        path = recorder.run_path(model_name, task, story["id"], rep)
        if path.exists():
            saved = _count_saved(path)
        active.append({"task": task, "story": story, "rep": rep,
                       "n": n, "start": saved, "round": saved})
    if not active:
        return done_total
    print(f"[batch:{model_name}] active={len(active)} done={done_total}", flush=True)
    max_round = max(a["n"] for a in active)
    for rnd in range(max_round):
        batch = [a for a in active if a["start"] <= rnd < a["n"]]
        if not batch:
            continue
        prompts, metas = [], []
        for a in batch:
            story = a["story"]
            ctx = " ".join(s["text"] for s in story["sentences"][: rnd + 1])
            q = story.get("question")
            prompts.append(builder.build(context=ctx, task=story["task"], question=q,
                                         condition=condition))
            base = job_seed(model_name, story["id"], a["rep"], seed_master)
            metas.append({"a": a, "seed": (base + rnd * STEP_SEED_STRIDE) & 0xFFFFFFFF})
        try:
            texts = model.generate_batch(prompts, seeds=[m["seed"] for m in metas])
        except RuntimeError:
            raise  # explicit errors such as OOM (with the vLLM guidance) are re-raised as-is
        for m, text in zip(metas, texts):
            a = m["a"]
            story = a["story"]
            sentence = story["sentences"][rnd]
            record = {
                "model": model_name,
                "story_id": story["id"],
                "task": story["task"],
                "step": sentence["id"],
                "context": " ".join(s["text"] for s in story["sentences"][: rnd + 1]),
                "response": text,
                "confidence": None,  # backfilled by analysis
                "repetition": a["rep"],
                "metadata": {
                    "model_version": getattr(model, "path", model_name),
                    "prompt_version": builder.config.get("prompt_version", "?"),
                    "dataset_version": recorder.dataset_version(story["id"]),
                    "temperature": temperature,
                    "seed": m["seed"],
                    "condition": condition,
                },
            }
            recorder.save_step(record, model_name, story["id"], a["rep"])
            rec_total += 1
        print(f"[batch:{model_name}] round {rnd + 1}/{max_round} seqs={len(batch)} "
              f"records={rec_total}", flush=True)
    return done_total + len(active)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help=f"comma-separated; default full matrix: {','.join(DEFAULT_MODELS)}")
    ap.add_argument("--tasks", default="false_belief,faux_pas,implicature")
    ap.add_argument("--stories", default=None, help="comma-separated subset; default all")
    ap.add_argument("--repetitions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="include annotations with reviewed!=True (all faux_pas ones are unreviewed)")
    ap.add_argument("--condition", default="condition_a",
                    choices=["condition_a", "condition_b"],
                    help="prompt condition (default condition_a; Condition B = the reasoning hint, an optional exploration in the proposal)")
    ap.add_argument("--results-dir", default="results",
                    help="output root directory; for Condition B use a separate directory (e.g. results_condB) so the official results are not overwritten")
    args = ap.parse_args(argv)

    cfg = ConfigManager()
    dm = DataManager(results_dir=args.results_dir)  # reads data/annotated, writes results_dir/raw
    experiment = dict(cfg.get("experiment"))
    if args.repetitions:
        experiment["repetitions"] = args.repetitions
    seed_master = args.seed if args.seed is not None else experiment["seed_master"]

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    stories = {s.strip() for s in args.stories.split(",")} if args.stories else None
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    by_task = _collect_stories(dm, tasks, stories, args.include_unreviewed, None)
    if not by_task:
        print("[cloud_run] no usable stories (check --include-unreviewed / --stories / question coverage)")
        return 1
    total_planned = sum(len(v) for v in by_task.values()) * experiment["repetitions"]
    print(f"[cloud_run] tasks={len(by_task)} stories={total_planned // max(experiment['repetitions'],1)} "
          f"reps={experiment['repetitions']} planned_jobs={total_planned}")

    factory = ModelFactory(cfg)
    builder = PromptBuilder(cfg.get("prompts"))
    overall = 0
    for model_name in models:
        model = factory.create(model_name)
        print(f"[cloud_run] loading {model_name} ...", flush=True)
        try:
            model.load()
        except Exception as exc:
            print(f"[cloud_run] failed to load model {model_name}: {exc}", flush=True)
            return 2
        try:
            jobs = []
            for task, stories_ in by_task.items():
                for story in stories_:
                    for rep in range(1, experiment["repetitions"] + 1):
                        jobs.append((task, story, rep))
            overall += run_model_batch(model_name, jobs, cfg, dm, builder, model,
                                        seed_master, condition=args.condition)
        finally:
            model.unload()
    print(f"[cloud_run] done. condition={args.condition} runs_completed/planned ≈ {overall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
