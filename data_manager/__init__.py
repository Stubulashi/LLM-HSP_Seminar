"""Data layer (scaf.md section 4): read and write stories / annotations / experiment results.

All writes are atomic (temporary file + os.replace) to prevent half-written corruption (hard rule 6).
Path conventions follow rulings C2/C7/C11 in docs/decisions.md.
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

    # ---- reads ----

    def load_raw_story(self, path: str | Path) -> str:
        """Read raw story text (scaf.md L341-347)."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_annotation(self, story_id: str) -> dict:
        """Read an annotation JSON (scaf.md L363-371)."""
        path = self.annotated_dir / f"{story_id}.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- writes (atomic) ----

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)  # clean up the half-written temporary file
            raise

    def save_annotation(self, story_id: str, data: dict) -> Path:
        """Save an annotation JSON to data/annotated/{story_id}.json (scaf.md L351-359)."""
        path = self.annotated_dir / f"{story_id}.json"
        self._atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))
        return path

    def save_result(self, record: dict, path: str | Path) -> Path:
        """Atomically write a single result JSON (scaf.md L375-385)."""
        path = Path(path)
        self._atomic_write(path, json.dumps(record, ensure_ascii=False, indent=2))
        return path

    def append_result(self, record: dict, path: str | Path) -> Path:
        """Append one result as JSONL (step-by-step appends survive interruption; supplementary convention: result-file granularity)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def result_path(self, model: str, task: str, story_id: str, run: int) -> Path:
        """Result path: results/raw/{model}/{task}/{story_id}/run{N}.json (ruling C2)."""
        return self.results_dir / "raw" / model / task / story_id / f"run{run}.json"
