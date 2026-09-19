"""结果记录器（pipeline.md 十节 L795-855 / scaf.md 9 节，裁决 C2/C3）。

路径：results/raw/{model}/{task}/{story_id}/run{repetition}.json（裁决 C2）。
格式：JSONL 逐 step 追加（补充约定：结果文件粒度，抗中断）。
禁止覆盖已存在实验结果（禁止事项 3）：完整 run 由 scheduler 幂等跳过。
"""

from __future__ import annotations

import hashlib
import json

from data_manager import DataManager


class ResultRecorder:
    def __init__(self, data_manager: DataManager, logger=None):
        self.dm = data_manager
        self.logger = logger

    def dataset_version(self, story_id: str) -> str:
        """标注文件内容哈希，作为 dataset_version（scaf.md Principle 2）。"""
        data = self.dm.load_annotation(story_id)
        blob = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def save_step(self, record: dict, model: str, story_id: str, repetition: int) -> None:
        """逐 step JSONL 追加（裁决 C2 路径 + C3 字段）。"""
        path = self.dm.result_path(model, record["task"], story_id, repetition)
        self.dm.append_result(record, path)

    def run_path(self, model: str, task: str, story_id: str, repetition: int):
        return self.dm.result_path(model, task, story_id, repetition)

    def is_complete(self, model: str, task: str, story_id: str, repetition: int, n_steps: int) -> bool:
        """run 文件存在且 step 数完整 → 幂等跳过（断点续跑不重算）。"""
        path = self.run_path(model, task, story_id, repetition)
        if not path.exists():
            return False
        try:
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        except OSError:
            return False
        return len(lines) >= n_steps
