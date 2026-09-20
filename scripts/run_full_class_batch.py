# -*- coding: utf-8 -*-
"""Full-class (per-class, non-sampled) real batch-run script (qwen0.5B on the local 4GB machine; resumable).

Example usage:
    python scripts/run_full_class_batch.py --class faux_pas --model qwen05b --repetitions 1
    python scripts/run_full_class_batch.py --class opentom  --model qwen05b --repetitions 1
    python scripts/run_full_class_batch.py --class swordsmanimp --model qwen05b --repetitions 1
    ... --analyze  run analyze at the end

Behaviour:
- Enumerate every item under datasets/<cls>/annotated → place them all into data/annotated (reviewed=True, including sfp).
- Call main.py run --model <m> --repetitions <n> (no story filter; runs every placed item of that batch);
  rerunning the same command resumes idempotently (completed run{N}.json files are skipped by the scheduler).
- Optional --analyze runs analyze at the end.
Exit code 0 = success.
"""
import argparse, json, shutil, io, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CLASS_SOURCE = {
    "faux_pas":  "datasets/faux_pas/annotated",
    "opentom":   "datasets/opentom/annotated",
    "swordsmanimp": "datasets/swordsmanimp/annotated",
}
DST = "data/annotated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="klass", required=True,
                    choices=list(CLASS_SOURCE))
    ap.add_argument("--model", default="qwen05b")
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()

    src = Path(CLASS_SOURCE[a.klass])
    files = sorted(src.glob("*.json"))
    assert files, f"no annotated under {src}"
    # placement
    n_placed = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        d["reviewed"] = True
        # use the file name as the id (the story_id = filename convention)
        sid = f.stem
        d["id"] = sid
        outp = Path(DST) / f"{sid}.json"
        json.dump(d, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        n_placed += 1
    print(f"[place] {a.klass}: {n_placed} stories -> data/annotated")

    from main import main as cli
    stories = ",".join(f.stem for f in files)
    rc = cli(["run", "--model", a.model, "--stories", stories,
              "--repetitions", str(a.repetitions)])
    if rc != 0:
        print("[fail] run rc=", rc)
        return rc

    # count completed run files
    import os
    completed = []
    for root, _, fs in os.walk(f"results/raw/{a.model}"):
        for fn in fs:
            if fn.startswith("run") and fn.endswith(".json"):
                completed.append(os.path.join(root, fn).replace("\\", "/"))
    print(f"[done] raw under results/raw/{a.model}: {len(completed)} run files")

    if a.analyze:
        r2 = cli(["analyze"])
        print("[analyze] rc=", r2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
