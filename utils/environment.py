"""环境初始化（scaf.md 1 节；裁决 C1/C6：逻辑并入 `python main.py setup`）。

检查 Python 版本 / 必需依赖 / GPU，创建数据目录，加载配置。
输出格式对齐 scaf.md L182-187 的 [OK] 报告。
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

# 目录约定（裁决 C15/C7：data/raw_stories、根级 results/）
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
    print(f"[{'OK' if ok else 'FAIL'}] Python {sys.version.split()[0]} (要求 >=3.10)")
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
        print("[WARN] CUDA not available（降级 CPU 模式；GPU 规格依据缺失）")
        return False
    except ImportError:
        print("[WARN] torch 未安装，跳过 GPU 检查")
        return False


def create_directories() -> bool:
    for d in DIRS_TO_CREATE:
        Path(d).mkdir(parents=True, exist_ok=True)  # 幂等：exist_ok
    print("[OK] Directory initialized")
    return True


def load_configs() -> bool:
    try:
        from config.config_manager import ConfigManager

        cfg = ConfigManager()
        models = cfg.get("models")
        print(f"[OK] Config loaded ({len(models)} models)")
        return True
    except Exception as exc:  # noqa: BLE001 - setup 报告所有失败
        print(f"[FAIL] Config loaded: {exc}")
        return False


def initialize_project() -> bool:
    """setup 入口；全部检查通过返回 True。"""
    results = [
        check_python(),
        check_packages(),
        check_gpu(),
        create_directories(),
        load_configs(),
    ]
    return all(results)
