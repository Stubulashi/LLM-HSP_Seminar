"""Supplementary self-test (stage 4): reproducibility spot check + extensibility drill (scaf.md Principle 2 / L1419-1463)."""

from config.config_manager import ConfigManager
from data_manager import DataManager
from experiment import ExperimentScheduler, IncrementalRunner, PromptBuilder, ResultRecorder
from models import MockModel

STORY = {
    "id": "fp001", "task": "faux_pas",
    "sentences": [
        {"id": 1, "text": "Anna bought a handmade sweater for Lisa.", "function": "background"},
        {"id": 2, "text": "Lisa visited Anna.", "function": "context_update"},
        {"id": 3, "text": "Lisa said the sweater looked like something her grandmother would wear.", "function": "critical_event"},
    ],
    "critical_sentence": 3, "gold_answer": "Lisa unintentionally offended Anna",
    "reviewed": True,
}


def test_reproducibility_same_seed_same_output(tmp_path):
    """Two consecutive runs with the same seed and config produce identical raw response sequences (scaf.md Principle 2)."""
    cfg = ConfigManager()

    def run_once():
        dm = DataManager(data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / "r1"))
        dm.save_annotation("fp001", STORY)
        builder = PromptBuilder(cfg.get("prompts"))
        recorder = ResultRecorder(dm)
        runner = IncrementalRunner(builder, recorder)
        model = MockModel(config={"path": "mock"}, seed=1)
        model.load()
        records = runner.run_story(STORY, model, "mock7b", 1, 0.7, seed=42)
        return [r["response"] for r in records]

    assert run_once() == run_once()


def test_extension_new_model_config_no_code_change(tmp_path):
    """Extensibility: after adding a model config to models.yaml the factory works directly (no experiment-code changes)."""
    cfg = ConfigManager()
    # simulate a new model config (ruling C14: adding a model = new config + a generate implementation)
    cfg.config["models"]["mock2"] = {
        "family": "Mock", "path": "mock/mock-2B", "backend": "huggingface",
        "size": "2B", "training": "instruction",
    }
    from models import ModelFactory

    factory = ModelFactory(cfg)
    model = factory.create("mock2")
    assert model.path == "mock/mock-2B"
    # do not actually load (that would download a real model); only verify the factory dispatch chain
