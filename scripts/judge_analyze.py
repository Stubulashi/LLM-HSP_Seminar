"""Concurrent judge analyser: computes judge-based accuracy (last step) and emergence (per step) over all runs.

Background: main.py analyze --scoring-method api_judge calls the judge serially inside the pipeline;
4,200+ API decisions take hours and produce no progress log. This script calls the judge concurrently
with a thread pool (the DeepSeek API is thread-safe: a new client per call) and writes the artifacts
to results/processed_judge/ (without overwriting the cosine-based processed/).

Usage (on the cloud; .env must contain DEEPSEEK_API_KEY):
    python -X utf8 scripts/judge_analyze.py --provider deepseek --scope last
    python -X utf8 scripts/judge_analyze.py --provider deepseek --scope all   # includes per-step emergence
Artifacts:
    results/processed_judge/accuracy.csv     (scope-independent; always written)
    results/processed_judge/emergence.csv    (values only with scope=all; empty columns with last)
"""
import argparse, csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from threading import Lock

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.confidence import extract_interpretation
from analysis.pipeline import _filter_task_drift, load_records


def _group_records(records):
    groups = {}
    for rec in records:
        groups.setdefault((rec["model"], rec["task"], rec["story_id"], rec["repetition"]), []).append(rec)
    return groups


def _judge_run(recs, ann, judge_fn, scope: str):
    """Judge a single run: returns (acc_label, emg_step). acc_label is True/False/None.

    An empty interpretation (no body text after cleaning) skips the judge: semantically not
    equivalent → False (saves API calls and avoids judging empty input).
    """
    ordered = sorted(recs, key=lambda r: r["step"])
    gold = ann.get("gold_points") or ann.get("gold_answer")
    q = ann.get("question", "")

    def judge_answer(response: str):
        norm = extract_interpretation(response).strip()
        if not norm:
            return False  # no body text → not equivalent
        return judge_fn(q or "", norm, gold)

    last = ordered[-1]
    acc = judge_answer(last["response"])
    emg = None
    if scope == "all":
        for r in ordered:
            if judge_answer(r["response"]) is True:
                emg = r["step"]
                break
    return acc, emg


def _load_cache(path: str) -> dict:
    cache = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[obj["k"]] = obj["v"]
    return cache


def _save_cache(path: str, cache: dict) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        for k, v in cache.items():
            fh.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def _cached_judge_fn(judge_fn, cache: dict, lock: Lock):
    """Wrap judge_fn (returning bool/None): cache keyed by the hash of (question|answer|gold).

    Hits are reused directly; None (undecidable) is not cached, so polluted empty values cannot
    appear and a rerun retries them.
    """
    def _jf(q: str, a: str, g: str):
        key = sha1("|{0}|{1}|{2}".format(q, a, g).encode("utf-8")).hexdigest()
        with lock:
            hit = cache.get(key)
        if hit == "1":
            return True
        if hit == "0":
            return False
        # hit is '' (polluted by the old version) or a miss → call again and overwrite the cache (self-healing)
        r = judge_fn(q, a, g)
        if isinstance(r, dict):  # tolerate judge.judge being passed by mistake (it returns {'label': ...})
            r = r.get("label")
        # r is bool/None; exceptions propagate → the caller records err
        if r is not None:
            with lock:
                cache[key] = "1" if r is True else "0"
        return r
    return _jf


def judge_analyze_groups(groups, dm, judge_fn, scope: str,
                         out_dir, max_workers=8, progress_every=50,
                         judge_cache_path=None):
    """Core: judge all runs concurrently and write accuracy.csv / emergence.csv. Returns (runs, errors).

    When judge_cache_path is given, a per-call cache is enabled: hits are reused, failed rows are
    not cached, and a rerun retries them automatically.
    """
    os.makedirs(out_dir, exist_ok=True)
    cache = _load_cache(judge_cache_path)
    lock = Lock()
    fn = _cached_judge_fn(judge_fn, cache, lock) if judge_cache_path else judge_fn
    acc_rows, emg_rows = [], []
    done = err = 0
    futures = {}
    err_msgs: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for key, recs in groups.items():
            model, task, story_id, rep = key
            try:
                ann = dm.load_annotation(story_id)
            except Exception:
                err += 1
                continue
            futures[pool.submit(_judge_run, recs, ann, fn, scope)] = (model, task, story_id, rep)
        for fut in as_completed(futures):
            model, task, story_id, rep = futures[fut]
            try:
                acc, emg = fut.result()
            except Exception as exc:
                err += 1
                if len(err_msgs) < 3:
                    err_msgs.append(f"{model}/{story_id} run{rep}: {type(exc).__name__}: {str(exc)[:160]}")
                acc, emg = None, None
            acc_rows.append({"model": model, "task": task, "story_id": story_id,
                             "repetition": rep,
                             "accuracy": "" if acc is None else ("1.0" if acc else "0.0")})
            emg_rows.append({"model": model, "task": task, "story_id": story_id,
                             "repetition": rep, "emergence_point": "" if emg is None else emg})
            done += 1
            if done % progress_every == 0 or done == len(futures):
                print(f"[judge_analyze] {done}/{len(futures)} runs judged (err={err})", flush=True)
    if judge_cache_path:
        _save_cache(judge_cache_path, cache)
    for msg in err_msgs:
        print(f"[judge_analyze] ERR e.g. {msg}", flush=True)
    with open(os.path.join(out_dir, "accuracy.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "task", "story_id", "repetition", "accuracy"])
        w.writeheader()
        w.writerows(acc_rows)
    with open(os.path.join(out_dir, "emergence.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "task", "story_id", "repetition", "emergence_point"])
        w.writeheader()
        w.writerows(emg_rows)
    print(f"[judge_analyze] done runs={done} errors={err} -> {out_dir}"
          + (f" (cache rows={len(cache)})" if judge_cache_path else ""), flush=True)
    return done, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek", choices=["local", "deepseek", "kimi"])
    ap.add_argument("--scope", default="last", choices=["last", "all"],
                    help="last = last-step accuracy only; all = per-step emergence (number of calls ≈ total steps)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=6, help="API retries per call (exponential backoff on 429/5xx)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/processed_judge")
    ap.add_argument("--task-filter", default=None, help="comma-separated; run only some tasks (to save money in a pilot)")
    args = ap.parse_args()

    from config.config_manager import ConfigManager
    from data_manager import DataManager
    from analysis.judge import LLMEquivalenceJudge

    cfg = ConfigManager()
    dm = DataManager()
    records = load_records(args.results_dir)
    records = _filter_task_drift(records, dm)
    allow = set(cfg.get("models").keys())
    records = [r for r in records if r.get("model") in allow]
    if args.task_filter:
        keep = {t.strip() for t in args.task_filter.split(",") if t.strip()}
        records = [r for r in records if r.get("task") in keep]
    if not records:
        print("[judge_analyze] no usable records")
        return 1
    groups = _group_records(records)
    cache_path = os.path.join(args.out_dir, "judge_cache.jsonl")
    print(f"[judge_analyze] runs={len(groups)} records={len(records)} scope={args.scope} "
          f"workers={args.workers} retries={args.retries}", flush=True)

    judge = LLMEquivalenceJudge(provider=args.provider, retries=args.retries)
    judge.load()
    try:
        # judge.judge returns a dict (label/rationale); only label (bool/None) is used here
        judge_fn = lambda q, a, g: judge.judge(q or "", a, g)["label"]
        judge_analyze_groups(groups, dm, judge_fn, args.scope,
                             args.out_dir, max_workers=args.workers,
                             judge_cache_path=cache_path)
    finally:
        judge.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
