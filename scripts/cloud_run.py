"""云端/大卡批量运行器：以"锁步轮次"方式并发跑全部 (model×task×story×rep) 增量实验。

为什么比顺序 main.py run 快：每轮把所有仍在进行中的 story 的同一相对步合成一个批次，
一次 vLLM generate_batch（continuous batching）压满一张 80/90 系显卡；配合
enable_prefix_caching，同 story 相邻轮共享前缀，吞吐可提升一个量级。

特性：
- 断点续跑：已完成 run 跳过；部分 run 按已存行数续跑（逐行 step 追加，不重复不丢）。
- 确定性：每 (model,story,rep) 固定 job seed，每轮 seed = job_seed + round*7919。
- OOM：vLLM 显存错误显式转指引并退出码 2，不静默。

用法（云端，Python 与 vLLM 就绪后）：
    python scripts/cloud_run.py                    # 默认全矩阵：qwen7b,deepseek7b,qwen14b,deepseek14b,deepseek32b
    python scripts/cloud_run.py --models qwen7b    # 覆盖为子集（preflight 用）
    python scripts/cloud_run.py --stories fb001,fb002
    # 补跑/续跑同一命令即可；完整 run 幂等跳过。
"""
import argparse, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 保证从任意 cwd 直接运行都能导入仓库根模块（AutoDL 上常直接 python scripts/...）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_manager import ConfigManager
from data_manager import DataManager
from experiment import PromptBuilder, ResultRecorder
from models import ModelFactory
from utils.seeding import job_seed

STEP_SEED_STRIDE = 7919

# 24G 单卡正式全矩阵（RQ1-RQ3）：默认全跑；--models 可覆盖为子集
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
            print(f"  [skip] {sid}: 缺 question，未纳入云跑", flush=True)
            continue
        if ann.get("reviewed") is not True and not include_unreviewed:
            print(f"  [skip] {sid}: 未 review（加 --include-unreviewed 可纳入）", flush=True)
            continue
        if stories and sid not in stories:
            continue
        by_task.setdefault(ann["task"], []).append(ann)
    return by_task


def run_model_batch(model_name: str, jobs, cfg, dm, builder, model, seed_master,
                    condition: str = "condition_a") -> int:
    """单模型锁步批量执行。返回完成 runs 数。condition 传给 PromptBuilder（condition_a/b）。"""
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
            raise  # OOM 等显式错误（含 vllm 指引）直接上抛
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
                "confidence": None,  # analysis 回填
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
                    help=f"逗号分隔；默认全矩阵：{','.join(DEFAULT_MODELS)}")
    ap.add_argument("--tasks", default="false_belief,faux_pas,implicature")
    ap.add_argument("--stories", default=None, help="逗号分隔子集；默认全部")
    ap.add_argument("--repetitions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="纳入 reviewed!=True 的标注（faux_pas 全部未审）")
    ap.add_argument("--condition", default="condition_a",
                    choices=["condition_a", "condition_b"],
                    help="prompt 条件（默认 condition_a；Condition B=reasoning 提示，proj 可选探究）")
    ap.add_argument("--results-dir", default="results",
                    help="输出根目录；Condition B 请用独立目录（如 results_condB）避免覆盖正式结果")
    args = ap.parse_args(argv)

    cfg = ConfigManager()
    dm = DataManager(results_dir=args.results_dir)  # 读 data/annotated，写 results_dir/raw
    experiment = dict(cfg.get("experiment"))
    if args.repetitions:
        experiment["repetitions"] = args.repetitions
    seed_master = args.seed if args.seed is not None else experiment["seed_master"]

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    stories = {s.strip() for s in args.stories.split(",")} if args.stories else None
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    by_task = _collect_stories(dm, tasks, stories, args.include_unreviewed, None)
    if not by_task:
        print("[cloud_run] 无可用故事（检查 --include-unreviewed / --stories / question 覆盖）")
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
            print(f"[cloud_run] 模型 {model_name} 加载失败：{exc}", flush=True)
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
