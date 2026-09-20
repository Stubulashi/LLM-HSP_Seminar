"""Probe: re-run a small Qwen2.5-1.5B-Instruct incremental sample to tell whether the 0.5B U+FFFD/degeneration is a model-layer artefact.

Non-destructive: records are written to the separate results/probe_qwen15b/{task}/{story}/run1.json
without touching the existing qwen05b results under results/raw or the processed outputs.
Usage: python -X utf8 scripts/_probe_qwen15b.py
"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.config_manager import ConfigManager
from data_manager import DataManager
from experiment import IncrementalRunner, PromptBuilder, ResultRecorder
from models import ModelFactory

MODEL = "qwen15b"  # quantization:4bit is configured for it in models.yaml (the local 4GB limit)
STORY_IDS = ["swmimp001", "swmimp002"]  # implicature samples


def main() -> int:
    cfg = ConfigManager()
    factory = ModelFactory(cfg)
    builder = PromptBuilder(cfg.get("prompts"))
    dm = DataManager(results_dir="results/probe_qwen15b")

    print("[probe] loading", MODEL, flush=True)
    model = factory.create(MODEL)
    model.load()
    rec_count = 0
    try:
        for sid in STORY_IDS:
            ann = dm.load_annotation(sid)
            task = ann.get("task", "implicature")
            print(f"[probe] story {sid} task={task} n_sent={len(ann['sentences'])}", flush=True)

            recorder = ResultRecorder(dm)
            runner = IncrementalRunner(builder, recorder)
            records = runner.run_story(
                story=ann,
                model=model,
                model_name=MODEL,
                repetition=1,
                temperature=cfg.get("experiment")["temperature"],
                seed=1,
            )
            rec_count += len(records)
            bad = sum("\ufffd" in r["response"] for r in records)
            echo = sum(
                any(w in (r.get("response") or "").lower()
                    for w in ["i'm ready", "ready for the next", "(enter the next sentence)"])
                for r in records
            )
            # print one clean sample from the start of each run for a quick eyeball check
            for r in records[:1]:
                print("   sample:", json.dumps(r["response"][:160], ensure_ascii=False), flush=True)
            print(f"[probe] {sid}: records={len(records)} U_FFFD_replies={bad} "
                  f"echo_degenerate_steps={echo}", flush=True)
    finally:
        model.unload()
    print("[probe] done. total probe records:", rec_count, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
