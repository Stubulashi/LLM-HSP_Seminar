"""自测补充（阶段 4）：可复现性抽查 + 扩展性演练（scaf.md Principle 2 / L1419-1463）。"""

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
    """同 seed 同配置连续两次 run，raw 响应序列一致（scaf.md Principle 2）。"""
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
    """扩展性：models.yaml 新增模型配置后，factory 直接可用（实验代码零改动）。"""
    cfg = ConfigManager()
    # 模拟新增模型配置（裁决 C14：加模型 = 新增配置 + 实现 generate）
    cfg.config["models"]["mock2"] = {
        "family": "Mock", "path": "mock/mock-2B", "backend": "huggingface",
        "size": "2B", "training": "instruction",
    }
    from models import ModelFactory

    factory = ModelFactory(cfg)
    model = factory.create("mock2")
    assert model.path == "mock/mock-2B"
    # 不实际加载（真实模型下载），仅验证工厂分发链路
