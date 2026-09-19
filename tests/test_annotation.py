"""M2 标注管线单测：parse_json / validator / story_parser / pipeline（含重试）。"""

import json

import pytest

from annotation.annotation_agent import AnnotationAgent, AnnotationError
from annotation.json_parser import JsonParseError, parse_json
from annotation.pipeline import run_annotation_pipeline
from annotation.story_parser import StoryPreprocessor
from annotation.validator import AnnotationValidator


# ---------- json_parser ----------

def test_parse_json_extracts_from_noisy_response():
    resp = "Here is JSON:\n{\"a\": 1}\nthanks"
    assert parse_json(resp) == {"a": 1}


def test_parse_json_rejects_missing_braces():
    with pytest.raises(JsonParseError):
        parse_json("no json here")


def test_parse_json_rejects_empty():
    with pytest.raises(JsonParseError):
        parse_json("")


# ---------- validator ----------

VALID = {
    "id": "fp001",
    "task": "faux_pas",
    "sentences": [
        {"id": 1, "text": "Anna bought a handmade sweater for Lisa.", "function": "background"},
        {"id": 2, "text": "Lisa visited Anna.", "function": "context_update"},
        {"id": 3, "text": "Lisa said the sweater looked like something her grandmother would wear.", "function": "critical_event"},
    ],
    "critical_sentence": 3,
    "gold_answer": "Lisa unintentionally offended Anna",
}


def test_validator_accepts_valid():
    ok, errors = AnnotationValidator().validate(VALID)
    assert ok, errors


@pytest.mark.parametrize("field", ["id", "task", "sentences", "critical_sentence", "gold_answer"])
def test_validator_rejects_missing_required(field):
    data = json.loads(json.dumps(VALID))
    del data[field]
    ok, errors = AnnotationValidator().validate(data)
    assert not ok
    assert any(field in e for e in errors)


def test_validator_rejects_skipped_sentence_id():
    data = json.loads(json.dumps(VALID))
    data["sentences"][1]["id"] = 3  # 跳号
    ok, errors = AnnotationValidator().validate(data)
    assert not ok
    assert any("continuous" in e for e in errors)


def test_validator_rejects_invalid_function():
    data = json.loads(json.dumps(VALID))
    data["sentences"][0]["function"] = "bad_function"
    ok, errors = AnnotationValidator().validate(data)
    assert not ok


def test_validator_rejects_invalid_task():
    data = json.loads(json.dumps(VALID))
    data["task"] = "unknown_task"
    ok, errors = AnnotationValidator().validate(data)
    assert not ok


def test_validator_rejects_critical_sentence_out_of_range():
    data = json.loads(json.dumps(VALID))
    data["critical_sentence"] = 9
    ok, errors = AnnotationValidator().validate(data)
    assert not ok
    assert any("out of range" in e for e in errors)


# ---------- story_parser ----------

RAW_STORY = (
    "Anna bought a handmade sweater for Lisa.\n\n"
    "Lisa visited Anna.\n\n"
    "Lisa said the sweater looked like something her grandmother would wear."
)


def test_split_sentence_three_sentences():
    sentences = StoryPreprocessor().split_sentence(RAW_STORY)
    assert len(sentences) == 3
    assert sentences[0] == "Anna bought a handmade sweater for Lisa."


# ---------- pipeline（mock 模型全链）----------

class MockAnnotator:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


GOOD_RESPONSE = json.dumps(VALID)


def test_pipeline_annotates_and_saves(tmp_path, monkeypatch):
    story_dir = tmp_path / "faux_pas"
    story_dir.mkdir()
    (story_dir / "fp001.txt").write_text(RAW_STORY, encoding="utf-8")

    monkeypatch.chdir(tmp_path)  # 隔离 data/ 目录

    from data_manager import DataManager

    dm = DataManager(data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    results = run_annotation_pipeline(
        story_dir, MockAnnotator(GOOD_RESPONSE), "{sentences}", dm, task="faux_pas"
    )
    assert [r["id"] for r in results] == ["fp001"]
    assert results[0]["task"] == "faux_pas"
    assert results[0]["reviewed"] is False
    # 落盘可读
    loaded = dm.load_annotation("fp001")
    assert loaded["gold_answer"] == "Lisa unintentionally offended Anna"


def test_agent_retries_then_succeeds():
    class FlakyModel:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "Here is JSON: {broken"
            return GOOD_RESPONSE

    model = FlakyModel()
    agent = AnnotationAgent(model, "{sentences}")
    data = agent.annotate(["s1", "s2"])
    assert data["id"] == "fp001"
    assert model.calls == 2


def test_agent_raises_after_max_attempts():
    model = MockAnnotator("no json at all")
    agent = AnnotationAgent(model, "{sentences}", max_attempts=3)
    with pytest.raises(AnnotationError):
        agent.annotate(["s1"])
