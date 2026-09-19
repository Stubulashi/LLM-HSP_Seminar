"""IPLE 主入口（CLI 分发）。

命令集（裁决 C5/C6，见 docs/decisions.md）：
    setup / annotate / review / run / analyze
--mock 为补充约定的测试设施（冒烟/复现性验证用），不加载真实模型。
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

    sub.add_parser("setup", help="环境自检（Python/依赖/GPU）+ 创建数据目录")

    p_annotate = sub.add_parser("annotate", help="对原始故事运行标注管线")
    p_annotate.add_argument("--input", default="data/raw_stories/",
                            help="原始故事目录（默认 data/raw_stories/）")
    p_annotate.add_argument("--task", default=None,
                            help=f"任务名 {VALID_TASKS}（默认从目录名推断，裁决 C11）")
    p_annotate.add_argument("--mock", action="store_true",
                            help="使用确定性 MockModel（测试/冒烟用，补充约定）")

    p_review = sub.add_parser("review", help="人工审查标注")
    p_review.add_argument("story_id", help="故事 id（无扩展名，如 fp001）")

    p_run = sub.add_parser("run", help="运行增量实验")
    p_run.add_argument("--model", required=True, help="模型名（config/models.yaml 中的键）")
    p_run.add_argument("--task", default=None,
                       help=f"任务名 {VALID_TASKS}（默认全部已标注任务）")
    p_run.add_argument("--stories", default=None, help="逗号分隔的故事 id 子集（默认全部）")
    p_run.add_argument("--repetitions", type=int, default=None,
                       help="重复次数（默认取 experiment.yaml）")
    p_run.add_argument("--seed", type=int, default=None,
                       help="seed 母种子覆盖（默认取 experiment.yaml）")
    p_run.add_argument("--mock", action="store_true",
                       help="使用确定性 MockModel（测试/冒烟用，补充约定）")

    p_analyze = sub.add_parser("analyze", help="计算指标并产出 CSV 与图（results/processed/）")
    p_analyze.add_argument("--scoring-method", default=None,
                           choices=["embedding_similarity", "local_judge", "api_judge"],
                           help="评分口径；默认取 experiment.yaml scoring.method")
    p_analyze.add_argument("--judge-provider", default=None,
                           choices=["local", "deepseek", "kimi"],
                           help="judge 模式供应商；默认取 scoring.judge_provider（api_judge 时）")
    p_analyze.add_argument("--results-dir", default="results",
                           help="分析根目录（默认 results；Condition B 用 results_condB）")

    return parser


def _get_model(config: ConfigManager, model_name: str, mock: bool, logger):
    """创建并加载模型；mock 模式用确定性 MockModel（补充约定）。"""
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

    # 收集已标注故事（跳过未 review 的，保护数据完整性）
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
    # allow_models：只分析 models.yaml 中声明的真实模型，排除 mock/杂散（科研口径）
    allow = set(config.get("models").keys())
    judge_fn = None
    judge = None
    if method in ("local_judge", "api_judge"):
        from analysis.judge import LLMEquivalenceJudge

        provider = args.judge_provider or scoring.get("judge_provider")
        if not provider:
            raise RuntimeError("judge 模式需配置 scoring.judge_provider 或 --judge-provider")
        logger.info(f"analyze method={method} provider={provider}")
        judge = LLMEquivalenceJudge(provider=provider)
        judge.load()
        # judge() 返回 {'label':…,"rationale":…}，pipeline 需要 bool/None
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
