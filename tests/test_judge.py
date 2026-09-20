"""LLMEquivalenceJudge regression tests (offline; no real models loaded)."""
import os

import pytest

from analysis.judge import (
    LLMEquivalenceJudge,
    _env_or_raise,
    _parse_label,
    load_dotenv_silent,
)


def test_parse_label():
    assert _parse_label("YES because both agree") is True
    assert _parse_label("NO the meanings differ") is False
    assert _parse_label("yes, both correct") is True
    assert _parse_label("No.") is False
    assert _parse_label("") is None
    assert _parse_label("maybe") is None  # undecidable → None, not False


def test_invalid_provider_raises():
    with pytest.raises(ValueError):
        LLMEquivalenceJudge(provider="gpt-unknown")


def test_api_provider_missing_key_raises(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "KIMI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    import analysis.judge as jmod
    monkeypatch.setattr(jmod, "load_dotenv_silent", lambda: None)  # disable .env backfill
    judge = LLMEquivalenceJudge(provider="deepseek")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        judge.load()


def test_env_or_raise_message():
    with pytest.raises(RuntimeError, match="not set"):
        _env_or_raise("IPLE_NONEXISTENT_KEY_XYZ")


def test_dotenv_silent_runs():
    # just needs to not raise (works with or without .env)
    load_dotenv_silent()
    assert True


def test_judge_analyze_core(tmp_path):
    """judge_analyze core: a fake judge_fn decides concurrently and writes accuracy/emergence CSVs (scope last/all)."""
    import csv

    from data_manager import DataManager
    from scripts.judge_analyze import judge_analyze_groups

    dm = DataManager(data_dir=str(tmp_path / "data"), results_dir=str(tmp_path / "results"))
    dm.save_annotation("fp001", {
        "id": "fp001", "task": "faux_pas", "question": "did anyone slip?",
        "sentences": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}],
        "critical_sentence": 2, "gold_answer": "Lisa unintentionally offended Anna",
        "reviewed": True})

    recs = [
        {"model": "m", "task": "faux_pas", "story_id": "fp001", "repetition": 1,
         "step": 1, "response": "Current interpretation: greeting."},
        {"model": "m", "task": "faux_pas", "story_id": "fp001", "repetition": 1,
         "step": 2, "response": "Lisa unintentionally offended Anna."},
    ]
    groups = {("m", "faux_pas", "fp001", 1): recs}
    judge_hit = lambda q, ans, gold: "offended Anna" in ans

    out_last = str(tmp_path / "out_last")
    judge_analyze_groups(groups, dm, judge_hit, "last", out_last, max_workers=2)
    acc = list(csv.DictReader(open(out_last + "/accuracy.csv", encoding="utf-8")))
    emg = list(csv.DictReader(open(out_last + "/emergence.csv", encoding="utf-8")))
    assert acc[0]["accuracy"] == "1.0"
    assert emg[0]["emergence_point"] == ""  # scope=last does not judge step by step

    out_all = str(tmp_path / "out_all")
    judge_analyze_groups(groups, dm, judge_hit, "all", out_all, max_workers=2)
    emg2 = list(csv.DictReader(open(out_all + "/emergence.csv", encoding="utf-8")))
    assert emg2[0]["emergence_point"] == "2"  # first step judged equivalent


def test_cached_judge_fn_handles_dict_and_empty_selfheal(tmp_path):
    """Regression: a dict-returning judge should yield its label; a polluted empty cache ('') should self-heal and re-judge."""
    import threading
    from hashlib import sha1

    from scripts.judge_analyze import _cached_judge_fn

    lock = threading.Lock()
    key = sha1("|q|a|g".encode("utf-8")).hexdigest()
    cache = {key: ""}  # the empty value written by the old version (real key)
    calls = {"n": 0}

    def judge_like_dict(q, a, g):
        calls["n"] += 1
        return {"label": True, "rationale": "ok"}  # simulate the dict return of judge.judge

    jf = _cached_judge_fn(judge_like_dict, cache, lock)
    # a polluted empty value counts as a miss: re-judge and overwrite with '1'
    assert jf("q", "a", "g") is True
    assert cache[key] == "1"
    # a fresh key is written normally
    assert jf("q2", "a2", "g2") is True
    assert all(v in ("1", "0") for v in cache.values())
    # after a cache hit the judge is no longer called
    n0 = calls["n"]
    assert jf("q", "a", "g") is True
    assert calls["n"] == n0
