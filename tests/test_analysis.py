"""M5 分析评估单测：confidence / embedding / accuracy / emergence / stability / pipeline。"""

import json
import os

import pandas as pd
import pytest

from analysis.accuracy import calculate_accuracy
from analysis.confidence import extract_confidence, extract_interpretation
from analysis.embedding import EmbeddingService, HashEmbedder
from analysis.emergence import calculate_emergence
from analysis.pipeline import run_analysis
from analysis.stability import calculate_stability, step_distances
from analysis.statistics import run_statistics, trajectory_similarity
from config.config_manager import ConfigManager
from data_manager import DataManager


# ---------- confidence ----------

def test_extract_confidence_basic():
    assert extract_confidence("Confidence: 80") == 80


def test_extract_confidence_variants():
    assert extract_confidence("confidence: 100") == 100
    assert extract_confidence("Confidence: 07") == 7
    assert extract_confidence("no number here") is None


def test_extract_confidence_takes_last():
    assert extract_confidence("Confidence: 40\nConfidence: 65") == 65


def test_extract_confidence_score_alias():
    # P0-2：真实低容量模型常写 "Confidence score: N" / "... N/100"，旧正则返回 None
    assert extract_confidence("Confidence score: 85") == 85
    assert extract_confidence("My confidence score is 85/100.") == 85
    assert extract_confidence("Confidence score: 0") == 0


def test_extract_confidence_not_misread_prose_tail():
    # P0-2：只取含置信关键字行内 <=100 的数；结尾散文不干扰
    text = (
        "Potential intention: prepare\nConfidence score: 75/100\n"
        "I've provided Jill's situation at length; please ask if you need more."
    )
    assert extract_confidence(text) == 75
    # 无数字 / 不含置信自报告的纯叙述 → None；不以“任意相遇的 confidence 词”误报
    assert extract_confidence("I have no strong confidence in that judgment.") is None
    assert extract_confidence("The speaker spoke with confidence and authority.") is None


def test_extract_confidence_star_score_and_zero_of_100():
    # 实测真实格式（含 markdown 强调与 “score of N out of 100”）必须正确取到
    assert extract_confidence("Confidence score: **80**. The story gives enough context.") == 80
    assert extract_confidence(
        "3. Confidence score of 0 out of 100. We cannot infer intentions.") == 0
    assert extract_confidence(
        "My confidence level remains low.\nConfidence score: 85/100 \nFinal answer: ...") == 85


def test_extract_interpretation_removes_confidence_score_line():
    # P1-5：含置信自报告行应从解释文本剔除，避免污染 embedding
    text = "Anna may want a gift.\nConfidence score: 85/100\nCharacter: pleased."
    out = extract_interpretation(text)
    assert "Confidence score" not in out
    assert "Anna may want a gift." in out
    assert "Character: pleased." in out


def test_extract_interpretation_removes_structural_echo():
    # P1-5：无正文的 prompt 章节/头回显与“ready”退化填充应剔除，正文保留
    src = (
        "5. Explanation\n"
        "The story states that both Mary and Giovanni noticed the turnip.\n"
        "1. Current situation interpretation\n2. Character intention\n"
        "I'm ready for the next sentence!\nCharacter intention: May keep looking."
    )
    out = extract_interpretation(src)
    assert "Explanation" not in out
    assert "2. Character intention" not in out
    assert "ready for the next sentence" not in out
    assert "The story states that both Mary" in out  # 正文保留
    assert "May keep looking" in out  # 带正文的意图说明保留


def test_extract_interpretation_removes_confidence():
    text = "Current interpretation: Anna likes it.\nConfidence: 65\n\nCharacter intention: gift."
    out = extract_interpretation(text)
    assert "Confidence: 65" not in out
    assert "Anna likes it." in out


def test_extract_interpretation_keeps_singleline_answer_before_confidence():
    # 回归：v1.2 单行“答案……Confidence: 85”不得整行删除，应保留答案正文
    text = 'Sarah said, "I was going to wear this to your party!" It was inappropriate '
    "because she unintentionally revealed the party. Confidence: 85"
    out = extract_interpretation(text)
    assert "Sarah said" in out
    assert "Confidence" not in out
    # 纯置信行（行首 confidence）仍整行删除
    assert extract_interpretation("Confidence: 85") == ""
    assert extract_interpretation("Answer line.\nConfidence score: 90\n").strip() == "Answer line."


# ---------- embedding ----------

def test_embedding_service_cache(tmp_path):
    service = EmbeddingService("mock-model", cache_dir=str(tmp_path / "emb"),
                               embedder=HashEmbedder())
    v1 = service.encode(["hello world"])
    files_after_first = len(list((tmp_path / "emb").glob("*.npy")))
    v2 = service.encode(["hello world"])  # 命中缓存
    assert files_after_first >= 1
    assert (v1 == v2).all()
    assert 0.0 <= service.cosine_similarity("a", "b") <= 1.0


def test_ensure_local_hf_offline_flags_offline(monkeypatch, tmp_path):
    # 离线健壮性回归: 本地已有该模型快照时, EmbeddingService 应激活 HF 离线,
    # 避免 sentence-transformers 在缓存命中也发网络 HEAD 而 getaddrinfo 崩溃。
    import huggingface_hub.constants as hf_c
    from analysis.embedding import _ensure_local_hf_offline

    # 构造假缓存: models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<h>/model.safetensors
    cache = tmp_path / "hub"
    snap = cache / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(hf_c, "HF_HUB_CACHE", str(cache))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    _ensure_local_hf_offline("sentence-transformers/all-MiniLM-L6-v2")
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


# ---------- metrics ----------

GOLD = "Lisa unintentionally offended Anna"

RECORDS = [
    {"model": "m", "task": "faux_pas", "story_id": "fp001", "step": 1,
     "response": "Current interpretation: greeting.\nConfidence: 40", "repetition": 1},
    {"model": "m", "task": "faux_pas", "story_id": "fp001", "step": 2,
     "response": "Current interpretation: Lisa unintentionally offended Anna.\nConfidence: 70", "repetition": 1},
]


def _service(tmp_path):
    return EmbeddingService("mock-model", cache_dir=str(tmp_path / "emb"),
                            embedder=HashEmbedder())


def test_accuracy_and_emergence(tmp_path):
    service = _service(tmp_path)
    acc = calculate_accuracy(RECORDS[1]["response"], GOLD, service, threshold=0.5)
    assert acc in (0.0, 1.0)
    emg = calculate_emergence(RECORDS, GOLD, service, threshold=0.5)
    assert emg in (1, 2, None)


def test_best_similarity_over_gold_points(tmp_path):
    # P1-4：gold 为多参考关键点列表（gold_points）时取最大相似度
    from analysis.accuracy import best_similarity, calculate_accuracy

    service = _service(tmp_path)
    # gold_points 第一个是完全不同的文本，第二个应更高（best-of）
    points = ["utterly unrelated topic keyboard chair",
              "Lisa unintentionally offended Anna after the party"]
    resp = RECORDS[1]["response"]  # “…Lisa unintentionally offended Anna.”
    sim_all = best_similarity(resp, points, service)
    sim_second = best_similarity(resp, points[1], service)
    assert sim_all >= sim_second  # 多参考不会比单正确参考更低
    assert calculate_accuracy(resp, points, service, threshold=0.5) in (0.0, 1.0)
    # 空参考 → 0（防止 None gold 崩溃）
    assert best_similarity(resp, [], service) == 0.0


def test_accuracy_emergence_with_judge_fn(tmp_path):
    # C2 judge 分支：judge_fn 决定 acc/emergence；None/False 不虚报
    from analysis.accuracy import calculate_accuracy

    service = _service(tmp_path)
    judge_hit = lambda q, ans, gold: "Lisa unintentionally offended Anna" in ans
    assert calculate_accuracy(RECORDS[1]["response"], GOLD, service, 0.7,
                              judge_fn=judge_hit, question="q?") == 1.0
    assert calculate_accuracy(RECORDS[0]["response"], GOLD, service, 0.7,
                              judge_fn=judge_hit, question="q?") == 0.0
    # judge 返回 None（无法判定）→ 记 0
    assert calculate_accuracy(RECORDS[1]["response"], GOLD, service, 0.7,
                              judge_fn=lambda q, a, g: None, question="q?") == 0.0
    # emergence 走 judge：正确句在 step2
    emg = calculate_emergence(RECORDS, GOLD, service, 0.7,
                              judge_fn=judge_hit, question="q?")
    assert emg == 2
    emg_none = calculate_emergence(RECORDS, GOLD, service, 0.7,
                                   judge_fn=lambda q, a, g: None, question="q?")
    assert emg_none is None


def test_run_analysis_judge_method_requires_judge_fn(tmp_path):
    # C2：scoring.method=api_judge 但未装配 judge_fn → 显式报错
    cfg = ConfigManager()
    cfg.update("scoring", {"method": "api_judge", "threshold": 0.7,
                           "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"})
    dm = DataManager(data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    dm.save_annotation("fp001", STORY)
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="需要 judge_fn"):
        run_analysis(tmp_path / "results", dm, cfg, embedder=HashEmbedder())


def test_stability(tmp_path):
    service = _service(tmp_path)
    stb = calculate_stability(RECORDS, service)
    assert 0.0 <= stb <= 1.0
    assert len(step_distances(RECORDS, service)) == 1


def test_trajectory_similarity(tmp_path):
    service = _service(tmp_path)
    sim = trajectory_similarity(RECORDS, RECORDS, service)
    assert sim > 0.9  # 相同数据自身对比应接近 1


# ---------- run_analysis 全链 ----------

STORY = {
    "id": "fp001", "task": "faux_pas",
    "sentences": [{"id": 1, "text": "Anna bought a sweater.", "function": "background"},
                  {"id": 2, "text": "Lisa said it looks old.", "function": "critical_event"}],
    "critical_sentence": 2, "gold_answer": GOLD, "reviewed": True,
}


def test_run_analysis_end_to_end(tmp_path):
    cfg = ConfigManager()
    dm = DataManager(data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    dm.save_annotation("fp001", STORY)

    from experiment import IncrementalRunner, PromptBuilder, ResultRecorder
    from models import MockModel

    builder = PromptBuilder(cfg.get("prompts"))
    recorder = ResultRecorder(dm)
    runner = IncrementalRunner(builder, recorder)
    model = MockModel(config={"path": "mock"}, seed=1)
    model.load()
    runner.run_story(STORY, model, "mock7b", 1, 0.7, 42)

    outputs = run_analysis(tmp_path / "results", dm, cfg, embedder=HashEmbedder())
    assert outputs["accuracy.csv"].exists()
    assert outputs["emergence.csv"].exists()
    assert outputs["trajectory_similarity.csv"].exists()
    assert outputs["trajectory.png"].exists()
    assert outputs["confidence_curve.png"].exists()
    assert outputs["emergence_distribution.png"].exists()

    acc = pd.read_csv(outputs["accuracy.csv"])
    assert len(acc) == 1
    assert acc.iloc[0]["model"] == "mock7b"


def test_run_analysis_filters_mock_via_allow_models(tmp_path):
    # P0/科研口径：mock/杂散模型不混入真实分析——CLI 传 allow_models 时应被剔除
    from experiment import IncrementalRunner, PromptBuilder, ResultRecorder
    from models import MockModel

    cfg = ConfigManager()
    dm = DataManager(data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    dm.save_annotation("fp001", STORY)  # mock 唯一 run 仍是 fp001

    builder = PromptBuilder(cfg.get("prompts"))
    recorder = ResultRecorder(dm)
    runner = IncrementalRunner(builder, recorder)
    model = MockModel(config={"path": "mock"}, seed=1)
    model.load()
    runner.run_story(STORY, model, "mock7b", 1, 0.7, 42)

    # allow_models 不含 mock7b → 全部被剔除 → 抛错（不再产出含 mock 的假科研数据）
    try:
        run_analysis(tmp_path / "results", dm, cfg, allow_models={"qwen05b"},
                     embedder=HashEmbedder())
    except RuntimeError as exc:
        assert "dropped by filters" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when only mock present + allow_models excludes it")

    # allow_models 为 None（库调用、供单测）则保留 mock：向后兼容
    outputs = run_analysis(tmp_path / "results", dm, cfg, embedder=HashEmbedder())
    acc = pd.read_csv(outputs["accuracy.csv"])
    assert acc.iloc[0]["model"] == "mock7b"


def test_analysis_filters_task_drift(tmp_path):
    # P0-3：run 的 task 与标注真值不一致时，analyze 应剔除（防跨任务污染）
    import json

    from analysis.pipeline import _filter_task_drift, load_records

    dm = DataManager(data_dir=str(tmp_path / "data"),
                     results_dir=str(tmp_path / "results"))
    # 标注真值为 faux_pas，但历史 run 被错放到 false_belief 目录且记录 task=false_belief
    dm.save_annotation("sfp011", dict(STORY, id="sfp011", task="faux_pas"))
    base = tmp_path / "results" / "raw" / "qwen05b" / "false_belief" / "sfp011"
    base.mkdir(parents=True)
    (base / "run1.json").write_text(json.dumps(
        {"model": "qwen05b", "task": "false_belief", "story_id": "sfp011",
         "step": 1, "response": "x", "repetition": 1}), encoding="utf-8")
    recs = load_records(tmp_path / "results")
    assert _filter_task_drift(recs, dm) == []  # 全部被剔除

    # 标注缺失的记录保留（不误杀），维持旧 analyze 行为
    ghost = tmp_path / "results" / "raw" / "qwen05b" / "faux_pas" / "ghost"
    ghost.mkdir(parents=True)
    (ghost / "run1.json").write_text(json.dumps(
        {"model": "qwen05b", "task": "faux_pas", "story_id": "ghost",
         "step": 1, "response": "y", "repetition": 1}), encoding="utf-8")
    recs2 = load_records(tmp_path / "results")
    kept = _filter_task_drift(recs2, dm)
    assert any(r["story_id"] == "ghost" for r in kept)
    assert not any(r["story_id"] == "sfp011" for r in kept)


def test_run_statistics_skips_without_deps(tmp_path):
    df = pd.DataFrame([
        {"model": "a", "task": "t", "story_id": f"s{i}", "repetition": 1,
         "emergence_point": 1, "confidence": 50, "stability": 0.1, "size": "7B"}
        for i in range(5)
    ])
    assert run_statistics(df) is None  # 数据不足 <10 行 → 跳过
