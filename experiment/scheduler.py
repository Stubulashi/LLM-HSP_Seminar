"""实验调度器（scaf.md 8 节 L902-963，裁决 C2/C11 扩展）。

- create_jobs：model × task × story × repetition 生成 Job（含 task 字段，裁决 C11）
- execute：按模型分组——加载一次跑完该组全部 job 再卸载（性能关键决策）
- job 幂等：run 文件完整即跳过（断点续跑）
- seed：job_seed 确定性派生 + apply_seed（补充约定，scaf.md Principle 2）
"""

from __future__ import annotations

from utils.seeding import apply_seed, job_seed


class ExperimentScheduler:
    def __init__(self, config, factory, runner, recorder, logger=None):
        self.config = config
        self.factory = factory
        self.runner = runner
        self.recorder = recorder
        self.logger = logger

    def create_jobs(self, models: list[str], tasks: list[str],
                    stories_by_task: dict[str, list[dict]],
                    repetitions: int) -> list[dict]:
        """生成 Job 列表：{model, task, story_id, repeat, story}。"""
        jobs: list[dict] = []
        for model in models:
            for task in tasks:
                for story in stories_by_task.get(task, []):
                    for repeat in range(1, repetitions + 1):
                        jobs.append({
                            "model": model,
                            "task": task,
                            "story_id": story["id"],
                            "repeat": repeat,
                            "story": story,
                        })
        return jobs

    def execute(self, jobs: list[dict]) -> None:
        """按模型分组执行：load → 跑完该组 → unload。"""
        experiment = self.config.get("experiment")
        temperature = experiment["temperature"]
        seed_master = experiment["seed_master"]

        by_model: dict[str, list[dict]] = {}
        for job in jobs:
            by_model.setdefault(job["model"], []).append(job)

        for model_name, model_jobs in by_model.items():
            if self.logger is not None:
                self.logger.info(f"loading model {model_name} (group of {len(model_jobs)} jobs)")
            model = self.factory.create(model_name)
            model.load()
            try:
                for job in model_jobs:
                    n_steps = len(job["story"]["sentences"])
                    if self.recorder.is_complete(
                        job["model"], job["task"], job["story_id"], job["repeat"], n_steps
                    ):
                        if self.logger is not None:
                            self.logger.info(
                                f"skip completed job {job['model']}/{job['story_id']}/run{job['repeat']}"
                            )
                        continue
                    seed = job_seed(job["model"], job["story_id"], job["repeat"], seed_master)
                    apply_seed(seed)
                    self.runner.run_story(
                        story=job["story"],
                        model=model,
                        model_name=job["model"],
                        repetition=job["repeat"],
                        temperature=temperature,
                        seed=seed,
                    )
            finally:
                model.unload()
                if self.logger is not None:
                    self.logger.info(f"unloaded model {model_name}")
