"""Logging system (scaf.md section 3): info/warning/error + timestamps + file output.

Every module must log through the logger (scaf.md L302-310).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str = "logs", name: str = "iple"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{name}.log"
        self._stream = open(self.log_file, "a", encoding="utf-8")

    def _write(self, level: str, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {message}"
        print(line, file=sys.stdout)
        self._stream.write(line + "\n")
        self._stream.flush()

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def close(self) -> None:
        self._stream.close()


_logger: Logger | None = None


def get_logger() -> Logger:
    """Process-level singleton."""
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger
