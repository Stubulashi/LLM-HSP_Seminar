"""IPLE main entry point (CLI dispatcher).

Commands (rulings C5/C6, see docs/decisions.md):
    setup / annotate / review / run / analyze
--mock is a test facility (supplementary convention, for smoke tests and reproducibility checks);
it does not load real models.
"""

import argparse
import sys

from config.config_manager import ConfigManager
from data_manager import DataManager
from utils.logger import get_logger

VALID_TASKS = ("faux_pas", "false_belief", "implicature")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="IPLE - Incremental Pragmatic LLM Evaluation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="environment self-check (Python/deps/GPU) + create data directories")

    p_annotate = sub.add_parser("annotate", help="run the annotation pipeline on raw stories")
    p_annotate.add_argument("--input", default="data/raw_stories/",
                            help="raw story directory (default data/raw_stories/)")
    p_annotate.add_argument("--task", default=None,
                            help=f"task name {VALID_TASKS} (default: inferred from the directory name; ruling C11)")
    p_annotate.add_argument("--mock", action="store_true",
                            help="use the deterministic MockModel (tests/smoke runs; supplementary convention)")

    p_review = sub.add_parser("review", help="human review of an annotation")
    p_review.add_argument("story_id", help="story id (without extension, e.g. fp001)")

    p_run = sub.add_parser("run", help="run the incremental experiment")
    p_run.add_argument("--model", required=True, help="model name (a key in config/models.yaml)")
    p_run.add_argument("--task", default=None,
                       help=f"task name {VALID_TASKS} (default: all annotated tasks)")
    p_run.add_argument("--stories", default=None, help="comma-separated subset of story ids (default: all)")
    p_run.add_argument("--repetitions", type=int, default=None,
                       help="number of repetitions (default from experiment.yaml)")
    p_run.add_argument("--seed", type=int, default=None,
                       help="master seed override (default from experiment.yaml)")
    p_run.add_argument("--mock", action="store_true",
                       help="use the deterministic MockModel (tests/smoke runs; supplementary convention)")

    p_analyze = sub.add_parser("analyze", help="compute metrics and produce CSVs and figures (results/processed/)")
    p_analyze.add_argument("--scoring-method", default=None,
                           choices=["embedding_similarity", "local_judge", "api_judge"],
                           help="scoring method; default: experiment.yaml scoring.method")
    p_analyze.add_argument("--judge-provider", default=None,
                           choices=["local", "deepseek", "kimi"],
                           help="judge provider; default: scoring.judge_provider (api_judge only)")
    p_analyze.add_argument("--results-dir", default="results",
                           help="analysis root directory (default results; use results_condB for Condition B)")

    return parser


def _get_model(config: ConfigManager, model_name: str, mock: bool, logger):
    """Create and load a model; mock mode uses the deterministic MockModel (supplementary convention)."""
    if mock:
        from models import MockModel

        logger.info(f"mock mode: using MockModel for {model_name}")
        model = MockModel(config={"path": model_name})
    else:
        from models import ModelFactory

        model = ModelFactory(config).create(model_name)
    model.load()
    return model


def cmd_setup() -> int:
    from utils.environment import initialize_project

    return 0 if initialize_project() else 1


def cmd_annotate(args, config: ConfigManager, dm: DataManager, logger) -> int:
    from annotation.pipeline import infer_task, run_annotation_pipeline

    task = args.task or infer_task(args.input)
    if task not in VALID_TASKS:
        print(f"[FAIL] cannot infer task from {args.input!r}; pass --task in {VALID_TASKS}")
        return 1
    logger.info(f"annotate: input={args.input}, task={task}, mock={args.mock}")

    model = _get_model(config, config.get("experiment")["annotation_model"],
                       args.mock, logger)
    template = config.get("prompts")["annotation"]["template"]
    results = run_annotation_pipeline(
        args.input, model, template, dm, task=task, logger=logger
    )
    print(f"[OK] annotated {len(results)} stories -> data/annotated/")
    return 0


def cmd_review(args, dm: DataManager, logger) -> int:
    from annotation.human_review import review_annotation

    review_annotation(args.story_id, dm, logger=logger)
    print(f"[OK] reviewed {args.story_id}")
    return 0


def cmd_run(args, config: ConfigManager, dm: DataManager, logger) -> int:
    from experiment import (ExperimentScheduler, IncrementalRunner,
                            PromptBuilder, ResultRecorder)
    from models import ModelFactory, MockModel

    experiment = dict(config.get("experiment"))
    if args.repetitions is not None:
        experiment["repetitions"] = args.repetitions
    if args.seed is not None:
        experiment["seed_master"] = args.seed

    # collect annotated stories (skip those not yet reviewed to protect data integrity)
    stories_by_task: dict[str, list[dict]] = {}
    for path in sorted(dm.annotated_dir.glob("*.json")):
        story = dm.load_annotation(path.stem)
        if story.get("reviewed") is not True:
            logger.warning(f"skip {path.stem}: not reviewed yet")
            continue
        if args.task and story["task"] != args.task:
            continue
        if args.stories and path.stem not in {s.strip() for s in args.stories.split(",")}:
            continue
        stories_by_task.setdefault(story["task"], []).append(story)
    if not stories_by_task:
        print("[FAIL] no reviewed annotated stories to run; run annotate + review first")
        return 1

    builder = PromptBuilder(config.get("prompts"))
    recorder = ResultRecorder(dm, logger)
    runner = IncrementalRunner(builder, recorder, logger)

    if args.mock:
        model = MockModel(config={"path": args.model})
        model.load()

        class _MockFactory:
            def create(self, name):
                return model

        factory = _MockFactory()
    else:
        factory = ModelFactory(config)

    scheduler = ExperimentScheduler(config, factory, runner, recorder, logger)
    jobs = scheduler.create_jobs(
        models=[args.model],
        tasks=list(stories_by_task),
        stories_by_task=stories_by_task,
        repetitions=experiment["repetitions"],
    )
    logger.info(f"run: {len(jobs)} jobs for model {args.model}")
    scheduler.execute(jobs)
    print(f"[OK] run finished: {len(jobs)} jobs")
    return 0


def cmd_analyze(args, config: ConfigManager, dm: DataManager, logger) -> int:
    from analysis.pipeline import run_analysis

    scoring = config.get("scoring")
    method = args.scoring_method or scoring.get("method", "embedding_similarity")
    # allow_models: analyze only real models declared in models.yaml (excludes mock/stray names; research protocol)
    allow = set(config.get("models").keys())
    judge_fn = None
    judge = None
    if method in ("local_judge", "api_judge"):
        from analysis.judge import LLMEquivalenceJudge

        provider = args.judge_provider or scoring.get("judge_provider")
        if not provider:
            raise RuntimeError("judge mode requires scoring.judge_provider or --judge-provider")
        logger.info(f"analyze method={method} provider={provider}")
        judge = LLMEquivalenceJudge(provider=provider)
        judge.load()
        # judge() returns {'label': ..., 'rationale': ...}; the pipeline needs bool/None
        judge_fn = lambda q, a, g: judge.judge(q or "", a, g)["label"]
    try:
        outputs = run_analysis(args.results_dir, dm, config, logger,
                               allow_models=allow, judge_fn=judge_fn)
    finally:
        if judge is not None:
            judge.unload()
    for name, path in outputs.items():
        print(f"[OK] {name} -> {path}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = ConfigManager()
    dm = DataManager()
    logger = get_logger()

    if args.command == "setup":
        return cmd_setup()
    if args.command == "annotate":
        return cmd_annotate(args, config, dm, logger)
    if args.command == "review":
        return cmd_review(args, dm, logger)
    if args.command == "run":
        return cmd_run(args, config, dm, logger)
    if args.command == "analyze":
        return cmd_analyze(args, config, dm, logger)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
