"""全链路冒烟测试（补充约定）：1 story × 1 model × 1 repeat。

流程：annotate（mock）→ review（自动接受）→ run（mock）→ analyze。
断言四类产物：annotated JSON、raw response、metrics CSV、PNG。
用法：python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 脚本模式运行时确保项目根在 sys.path（python scripts/smoke_test.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from annotation.human_review import review_annotation
    from data_manager import DataManager
    from main import main as cli
    from utils.logger import get_logger

    logger = get_logger()
    dm = DataManager()
    root = Path.cwd()

    # 1. annotate（mock 模型，示例故事 fp001 已在 data/raw_stories/faux_pas/）
    print("== Step 1: annotate ==")
    rc = cli(["annotate", "--mock", "--task", "faux_pas",
              "--input", "data/raw_stories/faux_pas"])
    assert rc == 0, "annotate failed"
    annotated = root / "data/annotated/fp001.json"
    assert annotated.exists(), "annotated JSON missing"

    # 2. review（自动接受，无修改）
    print("== Step 2: review ==")
    data = review_annotation("fp001", dm, input_fn=lambda _: "n", logger=logger)
    assert data.get("reviewed") is True, "reviewed flag missing"
    assert data["task"] == "faux_pas"

    # 3. run（mock 模型，1 重复）
    print("== Step 3: run ==")
    rc = cli(["run", "--mock", "--model", "mock7b", "--repetitions", "1"])
    assert rc == 0, "run failed"
    run_file = root / "results/raw/mock7b/faux_pas/fp001/run1.json"
    assert run_file.exists(), "raw response missing"
    n_steps = len([l for l in run_file.read_text(encoding="utf-8").splitlines() if l.strip()])
    assert n_steps >= 3, f"expected >=3 steps, got {n_steps}"

    # 4. analyze
    print("== Step 4: analyze ==")
    rc = cli(["analyze"])
    assert rc == 0, "analyze failed"
    processed = root / "results/processed"
    for name in ("accuracy.csv", "emergence.csv", "trajectory_similarity.csv",
                 "trajectory.png", "confidence_curve.png", "emergence_distribution.png"):
        assert (processed / name).exists(), f"missing artifact: {name}"

    print("[OK] smoke test passed: annotate -> review -> run -> analyze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
