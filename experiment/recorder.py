"""Result recorder (pipeline.md section 10 L795-855 / scaf.md section 9; rulings C2/C3).

Path: results/raw/{model}/{task}/{story_id}/run{repetition}.json (ruling C2).
Format: JSONL, appended step by step (supplementary convention: result-file granularity, interruption-safe).
Overwriting existing experiment results is forbidden (forbidden item 3); a complete run is skipped idempotently by the scheduler.
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
        """Content hash of the annotation file, used as dataset_version (scaf.md Principle 2)."""
        data = self.dm.load_annotation(story_id)
        blob = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def save_step(self, record: dict, model: str, story_id: str, repetition: int) -> None:
        """Append one step as JSONL (the path from ruling C2 plus the fields from ruling C3)."""
        path = self.dm.result_path(model, record["task"], story_id, repetition)
        self.dm.append_result(record, path)

    def run_path(self, model: str, task: str, story_id: str, repetition: int):
        return self.dm.result_path(model, task, story_id, repetition)

    def is_complete(self, model: str, task: str, story_id: str, repetition: int, n_steps: int) -> bool:
        """The run file exists and holds a complete step count → skip idempotently (resume without recomputation)."""
        path = self.run_path(model, task, story_id, repetition)
        if not path.exists():
            return False
        try:
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        except OSError:
            return False
        return len(lines) >= n_steps
