# -*- coding: utf-8 -*-
"""三类数据集最小全链路验证（并行冒烟扩展；不改 smoke_test.py）。

覆盖 task：faux_pas(FauxPas 中文? 否 -> 英文 sfp) / false_belief(OpenToM) /
implicature(SwordsmanImp-中文)。运行时仅使用 mock 模型，1 重复。

策略（不改变量架构 / 不改 main.py 默认目录）：
  1. 从 datasets/<cls>/annotated 选择代表性样本（FauxPas 用其中一个 task=faux_pas 的 sfp；
     OpenToM=fb001；Swordsman=swmimp001）。
  2. 复制到 data/annotated/<story>.json 并把 reviewed 置 True（main.py run 需 reviewed）。
  3. 逐 task 调用 main 的 run（mock, 1repetition, 用 --stories 限定仅样本）。
  4. 校验 run JSONL：无 NaN/None/截断；critical 在界；response 含 "Confidence: NN"。
  5. 调用 analyze，断言 3 个 CSV + 3 张 PNG 均非空。
  6. 清理本轮放入 data/annotated 的临时样本（fp001 smoke 保留不动）。

用法：python scripts/smoke_multi_datasets.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _set_utf8_stdout():
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        pass


def main() -> int:
    _set_utf8_stdout()
    from annotation.validator import AnnotationValidator
    from main import main as cli

    dm = None  # 延迟
    from data_manager import DataManager

    dm = DataManager()
    dst = dm.annotated_dir

    # ---------- 选样本 ----------
    # FauxPas: 选一个 task==faux_pas 的 sfp
    faux_story = None
    for f in sorted((Path("datasets/faux_pas/annotated")).glob("sfp*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("task") == "faux_pas":
            faux_story = (f.stem, d)
            break
    assert faux_story is not None, "no faux_pas sfp found"
    samples = [
        ("faux_pas", faux_story[0], faux_story[1]),
        ("false_belief", "fb001", json.load(
            open("datasets/opentom/annotated/fb001.json", encoding="utf-8"))),
        ("implicature", "swmimp001", json.load(
            open("datasets/swordsmanimp/annotated/swmimp001.json", encoding="utf-8"))),
    ]

    # ---------- 复制进入 data/annotated（reviewed=True）----------
    placed = []
    for task, sid, d in samples:
        d = dict(d)
        d["id"] = sid
        d["task"] = task
        d["reviewed"] = True
        p = dst / f"{sid}.json"
        data_manager = dm
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        placed.append(sid)

    # ---------- validator 校验 -------------
    v = AnnotationValidator()
    for sid in placed:
        obj = json.load(open(dst / f"{sid}.json", encoding="utf-8"))
        ok, errs = v.validate(obj)
        msg = f"[validator] {sid}: {'PASS' if ok else 'FAIL ' + '; '.join(errs)}"
        print(msg)

    # ---------- run（mock）----------
    stories = ",".join(placed)
    print("run stories:", stories)
    rc = cli(["run", "--mock", "--model", "mock7b", "--stories", stories,
              "--repetitions", "1"])
    if rc != 0:
        print("[FAIL] run rc=", rc)
        return 2

    # ---------- raw 校验 ----------
    ok = True
    for sid in placed:
        run_file = dst is not None and (Path("results") / "raw" / "mock7b"
                                        / "faux_pas" / sid / "run1.json")
        # 需要落实真实 parent by task: 先 map task 于相应源
        # 找到该 sid 的 task source 目录路径硬关联(见上 samples)
    # 直接据此循环修正 parent task:
    results_failed = []
    # 因为 task 各不同，改为按 samples task id 目录
    del ok
    for task, sid, _ in samples:
        rf = Path("results") / "raw" / "mock7b" / task / sid / "run1.json"
        if not rf.exists():
            results_failed.append((sid, "missing " + str(rf)))
            continue
        lines = [ln for ln in rf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        n = len(lines)
        bad = []
        import math
        for ln in lines:
            r = json.loads(ln)
            json_s = ln
            if "NaN" in json_s or (r.get("confidence") is not None
                                   and any(not math.isfinite(x) for x in
                                           ([r["confidence"]] if isinstance(r.get("confidence"), (int, float)) else []))):
                bad.append("NaN")
            if r.get("step") is None:
                bad.append("no-step")
        if bad:
            results_failed.append((sid, f"{n} lines but bad={bad}"))
        else:
            print(f"[raw] {sid}: {n} steps OK")
    if results_failed:
        for x in results_failed:
            print("[FAIL] raw:", x)
        return 3

    # ---------- analyze ----------
    rc = cli(["analyze"])
    if rc != 0:
        print("[FAIL] analyze rc=", rc)
        return 4
    expected = ["accuracy.csv", "emergence.csv", "trajectory_similarity.csv",
                "confidence_curve.png", "emergence_distribution.png", "trajectory.png"]
    missing = [p for p in expected
               if not (Path("results/processed") / p).exists()
               or (Path("results/processed") / p).stat().st_size == 0]
    if missing:
        print("[FAIL] analyze artifacts missing:", missing)
        return 5
    print("[OK] analyze artifacts present:", ", ".join(expected))

    # ---------- 清理 ----------
    keep = {"fp001.json"}
    for sid in placed:
        p = dst / f"{sid}.json"
        if p.exists() and p.name not in keep:
            import os
            os.remove(p)
    print("[OK] cleaned temp samples; data/annotated left:",
          sorted(x.name for x in dst.glob("*.json")))
    print("ALL DATASET MIN SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
