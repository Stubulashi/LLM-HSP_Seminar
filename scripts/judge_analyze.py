"""并发版 judge 分析器：对全部 runs 做 judge 口径 accuracy（末步）与 emergence（逐 step）。

背景：main.py analyze --scoring-method api_judge 在 pipeline 内串行调用 judge，4200+ 次
API 判定耗时数小时且无进度日志。本脚本用线程池并发调用 judge（DeepSeek API 线程安全：
每次调用新建 client），并把产物写到 results/processed_judge/（不覆盖余弦版 processed）。

用法（云端，需 .env 含 DEEPSEEK_API_KEY）：
    python -X utf8 scripts/judge_analyze.py --provider deepseek --scope last
    python -X utf8 scripts/judge_analyze.py --provider deepseek --scope all   # 含逐 step emergence
产物：
    results/processed_judge/accuracy.csv     （scope 无关，恒有）
    results/processed_judge/emergence.csv    （scope=all 才有值；last 时列为空）
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
    """单 run 判定：返回 (acc_label, emg_step)。acc_label True/False/None。

    空解释（清洗后无正文）不调 judge：语义上不等价 → False（省 API 且避免空输入误判）。
    """
    ordered = sorted(recs, key=lambda r: r["step"])
    gold = ann.get("gold_points") or ann.get("gold_answer")
    q = ann.get("question", "")

    def judge_answer(response: str):
        norm = extract_interpretation(response).strip()
        if not norm:
            return False  # 无正文 → 不等价
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
    """包装 judge_fn（返回 bool/None）：以 (question|answer|gold) 哈希为键缓存。

    命中直接复用；None（无法判定）不写缓存，避免空值污染（重跑会重试）。
    """
    def _jf(q: str, a: str, g: str):
        key = sha1("|{0}|{1}|{2}".format(q, a, g).encode("utf-8")).hexdigest()
        with lock:
            hit = cache.get(key)
        if hit == "1":
            return True
        if hit == "0":
            return False
        # hit 为 ''（旧版污染）或未命中 → 重新调用并覆盖缓存（自愈）
        r = judge_fn(q, a, g)
        if isinstance(r, dict):  # 兼容误传 judge.judge（返回 {'label':…}）
            r = r.get("label")
        # r: bool/None；异常上抛 → 调用方记 err
        if r is not None:
            with lock:
                cache[key] = "1" if r is True else "0"
        return r
    return _jf


def judge_analyze_groups(groups, dm, judge_fn, scope: str,
                         out_dir, max_workers=8, progress_every=50,
                         judge_cache_path=None):
    """核心：并发判所有 run，写 accuracy.csv / emergence.csv。返回 (runs, errors)。

    judge_cache_path 给定则启用逐调用缓存：命中直接复用，失败行不写缓存，重跑自动重试。
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
                    help="last=仅末步 accuracy；all=逐 step 判定 emergence（调用量≈全部步数）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=6, help="单次调用 API 重试次数（429/5xx 指数退避）")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/processed_judge")
    ap.add_argument("--task-filter", default=None, help="逗号分隔，只跑某任务（试点省钱用）")
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
        print("[judge_analyze] 无可用记录")
        return 1
    groups = _group_records(records)
    cache_path = os.path.join(args.out_dir, "judge_cache.jsonl")
    print(f"[judge_analyze] runs={len(groups)} records={len(records)} scope={args.scope} "
          f"workers={args.workers} retries={args.retries}", flush=True)

    judge = LLMEquivalenceJudge(provider=args.provider, retries=args.retries)
    judge.load()
    try:
        # judge.judge 返回 dict（含 label/rationale），此处只取 label（bool/None）
        judge_fn = lambda q, a, g: judge.judge(q or "", a, g)["label"]
        judge_analyze_groups(groups, dm, judge_fn, args.scope,
                             args.out_dir, max_workers=args.workers,
                             judge_cache_path=cache_path)
    finally:
        judge.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
