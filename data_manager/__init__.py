"""数据层（scaf.md 4 节）：读写 story / annotation / 实验结果。

所有落盘采用原子写（临时文件 + os.replace），防半写损坏（硬性规则 6）。
路径约定见 docs/decisions.md 裁决 C2/C7/C11。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class DataManager:
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.annotated_dir = self.data_dir / "annotated"

    # ---- 读取 ----

    def load_raw_story(self, path: str | Path) -> str:
        """读取原始故事文本（scaf.md L341-347）。"""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_annotation(self, story_id: str) -> dict:
        """读取标注 JSON（scaf.md L363-371）。"""
        path = self.annotated_dir / f"{story_id}.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- 写入（原子写）----

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)  # 清理半写临时文件
            raise

    def save_annotation(self, story_id: str, data: dict) -> Path:
        """保存标注 JSON 至 data/annotated/{story_id}.json（scaf.md L351-359）。"""
        path = self.annotated_dir / f"{story_id}.json"
        self._atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))
        return path

    def save_result(self, record: dict, path: str | Path) -> Path:
        """原子写单条结果 JSON（scaf.md L375-385）。"""
        path = Path(path)
        self._atomic_write(path, json.dumps(record, ensure_ascii=False, indent=2))
        return path

    def append_result(self, record: dict, path: str | Path) -> Path:
        """JSONL 追加一条结果（逐 step 追加，抗中断；补充约定：结果文件粒度）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def result_path(self, model: str, task: str, story_id: str, run: int) -> Path:
        """结果路径：results/raw/{model}/{task}/{story_id}/run{N}.json（裁决 C2）。"""
        return self.results_dir / "raw" / model / task / story_id / f"run{run}.json"
