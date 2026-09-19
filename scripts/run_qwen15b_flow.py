"""用 qwen15b（本地 4GB，4bit）对三任务代表性小样跑完整增量流程。

写入权威路径 results/raw/qwen15b/{task}/{story}/run1.json，使 `analyze` 能纳入。
用法：
    python -X utf8 scripts/run_qwen15b_flow.py
环境：PYTHONPATH 需含仓库根；模型权重已在本地 HF 缓存。
"""
import json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.config_manager import ConfigManager
from data_manager import DataManager
from experiment import IncrementalRunner, PromptBuilder, ResultRecorder
from models import ModelFactory

# 三任务代表性小样（均有 question：false_belief 原有；faux_pas/implicature 本轮补齐）
SAMPLE = {
    "false_belief": ["fb001", "fb002"],
    "faux_pas": ["sfp001", "sfp002"],
    "implicature": ["swmimp001", "swmimp002"],
}
MODEL = "qwen15b"


def main() -> int:
    cfg = ConfigManager()
    factory = ModelFactory(cfg)
    builder = PromptBuilder(cfg.get("prompts"))
    dm = DataManager()  # results_dir=results -> results/raw/qwen15b/...
    model = factory.create(MODEL)
    print(f"[flow] loading {MODEL} ...", flush=True)
    model.load()
    t0 = time.time()
    n_records = 0
    try:
        for task, ids in SAMPLE.items():
            for sid in ids:
                ann = dm.load_annotation(sid)
                if ann.get("task") != task:
                    print(f"  WARN {sid}: annotation task {ann.get('task')} != {task}", flush=True)
                recorder = ResultRecorder(dm)
                runner = IncrementalRunner(builder, recorder)
                recs = runner.run_story(
                    story=ann,
                    model=model,
                    model_name=MODEL,
                    repetition=1,
                    temperature=cfg.get("experiment")["temperature"],
                    seed=42,  # 确定性：与 run 阶段默认 master 无关，固定探针种子
                )
                n_records += len(recs)
                has_q = bool(ann.get("question"))
                print(f"[flow] {task}/{sid}: steps={len(recs)} question={has_q} elapsed={time.time()-t0:.0f}s",
                      flush=True)
    finally:
        model.unload()
    print(f"[flow] done. total records={n_records} elapsed={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
