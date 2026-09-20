"""Environment initialisation (scaf.md section 1; rulings C1/C6: the logic lives inside `python main.py setup`).

Checks the Python version / required packages / GPU, creates the data directories and loads the
configuration. The output format matches the [OK] report of scaf.md L182-187.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "yaml",
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "statsmodels",
    "sentence_transformers",
    "pytest",
]

# directory conventions (rulings C15/C7: data/raw_stories, root-level results/)
DIRS_TO_CREATE = [
    "data/raw_stories",
    "data/annotated",
    "results/raw",
    "results/processed",
    "results/embeddings",
    "logs",
]

MIN_PYTHON = (3, 10)


def check_python() -> bool:
    ok = sys.version_info >= MIN_PYTHON
    print(f"[{'OK' if ok else 'FAIL'}] Python {sys.version.split()[0]} (requires >=3.10)")
    return ok


def check_packages() -> bool:
    ok = True
    for pkg in REQUIRED_PACKAGES:
        present = importlib.util.find_spec(pkg) is not None
        ok = ok and present
        print(f"[{'OK' if present else 'WARN'}] {pkg}")
    return ok


def check_gpu() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[OK] CUDA available: {name} ({vram:.1f} GB VRAM)")
            return True
        print("[WARN] CUDA not available (falling back to CPU; the GPU specification basis is missing)")
        return False
    except ImportError:
        print("[WARN] torch is not installed; skipping the GPU check")
        return False


def create_directories() -> bool:
    for d in DIRS_TO_CREATE:
        Path(d).mkdir(parents=True, exist_ok=True)  # idempotent via exist_ok
    print("[OK] Directory initialized")
    return True


def load_configs() -> bool:
    try:
        from config.config_manager import ConfigManager

        cfg = ConfigManager()
        models = cfg.get("models")
        print(f"[OK] Config loaded ({len(models)} models)")
        return True
    except Exception as exc:  # noqa: BLE001 - setup reports every failure
        print(f"[FAIL] Config loaded: {exc}")
        return False


def initialize_project() -> bool:
    """Setup entry point; returns True when every check passes."""
    results = [
        check_python(),
        check_packages(),
        check_gpu(),
        create_directories(),
        load_configs(),
    ]
    return all(results)
