"""M4 experiment engine unit tests: prompt_builder / incremental_runner / recorder / scheduler."""

import json

from config.config_manager import ConfigManager
from data_manager import DataManager
from experiment import ExperimentScheduler, IncrementalRunner, PromptBuilder, ResultRecorder
from models import MockModel

STORY = {
    "id": "fp001",
    "task": "faux_pas",
    "sentences": [
        {"id": 1, "text": "Anna bought a handmade sweater for Lisa.", "function": "background"},
        {"id": 2, "text": "Lisa visited Anna.", "function": "context_update"},
        {"id": 3, "text": "Lisa said the sweater looked like something her grandmother would wear.", "function": "critical_event"},
    ],
    "critical_sentence": 3,
    "gold_answer": "Lisa unintentionally offended Anna",
    "reviewed": True,
}


def _setup(tmp_path):
    cfg = ConfigManager()
    dm = DataManager(data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    dm.save_annotation("fp001", STORY)
    builder = PromptBuilder(cfg.get("prompts"))
    recorder = ResultRecorder(dm)
    runner = IncrementalRunner(builder, recorder)
    return cfg, dm, builder, recorder, runner


def test_prompt_builder_condition_a(tmp_path):
    cfg, _, builder, _, _ = _setup(tmp_path)
    prompt = builder.build(context="Anna bought a sweater.")
    assert "language comprehension experiment" in prompt
    assert "Your confidence score from 0 to 100" in prompt
    assert "Do not assume future sentences" in prompt
    # P0-1: the confidence must be the last line of the whole reply, with nothing appended after it (suppresses prose tails)
    assert "Stop right after the confidence line" in prompt


def test_prompt_builder_false_belief_question(tmp_path):
    cfg, _, builder, _, _ = _setup(tmp_path)
    prompt = builder.build(
        context="John puts a chocolate in drawer.",
        task="false_belief",
        question="Where does Mary think the chocolate is?",
    )
    assert "Where does Mary think the chocolate is?" in prompt
    assert "Give a direct answer to the question, then:" in prompt


def test_prompt_builder_implicature_direct_template(tmp_path):
    # P1-5 calibration (1.2): implicature with a question → single-answer template, no numbered three-part echo
    cfg, _, builder, _, _ = _setup(tmp_path)
    prompt = builder.build(
        context="Some implicature context.",
        task="implicature",
        question="这句话的言外之意是什么？",
    )
    assert "这句话的言外之意是什么？" in prompt
    assert "Give a short direct answer in Chinese" in prompt
    assert "1. Current situation interpretation" not in prompt  # numbered echo forbidden
    assert "Stop right after the confidence line" in prompt or "Write nothing after it" in prompt


def test_prompt_builder_faux_pas_direct_template(tmp_path):
    # P1-5 calibration (1.2): faux_pas with a question → single-answer template
    cfg, _, builder, _, _ = _setup(tmp_path)
    prompt = builder.build(
        context="A story context.",
        task="faux_pas",
        question="Did anyone say something they should not have said?",
    )
    assert "Did anyone say something they should not have said?" in prompt
    assert "Give a short direct answer" in prompt
    assert "1. Current situation interpretation" not in prompt


def test_runner_records_and_saves(tmp_path):
    cfg, dm, builder, recorder, runner = _setup(tmp_path)
    model = MockModel(config={"path": "mock"}, seed=1)
    model.load()
    records = runner.run_story(STORY, model, "mock7b", repetition=1,
                               temperature=0.7, seed=42)

    assert len(records) == 3
    r = records[0]
    assert r["model"] == "mock7b"
    assert r["task"] == "faux_pas"
    assert r["confidence"] is None
    assert r["metadata"]["temperature"] == 0.7
    assert r["metadata"]["seed"] == 42
    assert r["metadata"]["prompt_version"] == "1.2"
    assert "Confidence:" in r["response"]

    path = dm.result_path("mock7b", "faux_pas", "fp001", 1)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["step"] == 1


def test_scheduler_jobs_and_idempotent(tmp_path):
    cfg, dm, builder, recorder, runner = _setup(tmp_path)

    class MockFactory:
        def __init__(self):
            self.instances = {}

        def create(self, name):
            if name not in self.instances:
                m = MockModel(config={"path": name}, seed=1)
                m.load()
                self.instances[name] = m
            return self.instances[name]

    scheduler = ExperimentScheduler(cfg, MockFactory(), runner, recorder)
    jobs = scheduler.create_jobs(
        models=["mock7b"], tasks=["faux_pas"],
        stories_by_task={"faux_pas": [STORY]}, repetitions=1,
    )
    assert len(jobs) == 1
    assert jobs[0]["task"] == "faux_pas"  # ruling C11: the job carries the task

    scheduler.execute(jobs)
    path = dm.result_path("mock7b", "faux_pas", "fp001", 1)
    first_run = path.read_text(encoding="utf-8")

    scheduler.execute(jobs)  # idempotent: the second call skips and does not rewrite the file
    assert path.read_text(encoding="utf-8") == first_run


def test_scheduler_marks_complete(tmp_path):
    cfg, dm, builder, recorder, runner = _setup(tmp_path)
    assert recorder.is_complete("mock7b", "faux_pas", "fp001", 1, n_steps=3) is False

    model = MockModel(config={"path": "mock"}, seed=1)
    model.load()
    runner.run_story(STORY, model, "mock7b", 1, 0.7, 42)
    assert recorder.is_complete("mock7b", "faux_pas", "fp001", 1, n_steps=3) is True
    assert recorder.is_complete("mock7b", "faux_pas", "fp001", 1, n_steps=5) is False
