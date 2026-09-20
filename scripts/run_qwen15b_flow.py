"""Run the full incremental pipeline with qwen15b (local 4GB, 4bit) on representative samples of the three tasks.

Writes to the authoritative path results/raw/qwen15b/{task}/{story}/run1.json so that `analyze` picks it up.
Usage:
    python -X utf8 scripts/run_qwen15b_flow.py
Environment: PYTHONPATH must contain the repo root; the model weights are already in the local HF cache.
"""
import json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.config_manager import ConfigManager
from data_manager import DataManager
from experiment import IncrementalRunner, PromptBuilder, ResultRecorder
from models import ModelFactory

# representative samples of the three tasks (all with a question: false_belief had them; faux_pas/implicature were filled in this round)
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
                    seed=42,  # deterministic: independent of the run-stage master seed; a fixed probe seed
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
