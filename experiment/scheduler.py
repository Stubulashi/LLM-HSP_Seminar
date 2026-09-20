"""Experiment scheduler (scaf.md section 8 L902-963; extended by rulings C2/C11).

- create_jobs: builds jobs over model × task × story × repetition (each job carries the task field; ruling C11)
- execute: groups by model — load once, finish all jobs of the group, then unload (a performance-critical decision)
- job idempotency: a complete run file is skipped (resume after interruption)
- seed: deterministic derivation via job_seed plus apply_seed (supplementary convention; scaf.md Principle 2)
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
        """Build the job list: {model, task, story_id, repeat, story}."""
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
        """Execute grouped by model: load → finish the group → unload."""
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
