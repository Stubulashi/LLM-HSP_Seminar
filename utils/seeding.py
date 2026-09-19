"""随机种子控制（scaf.md Principle 2 的机制实现；机制本身依据缺失，为补充约定）。

job_seed 派生：hash(model|story|repeat|seed_master)，保证同一 job 同配置复跑结果一致。
"""

from __future__ import annotations

import hashlib
import random

import numpy as np


def job_seed(model: str, story: str, repeat: int, seed_master: int) -> int:
    """确定性派生 job 级 seed。"""
    raw = f"{model}|{story}|{repeat}|{seed_master}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def apply_seed(seed: int) -> None:
    """应用 seed 到 random / numpy（torch 在可用时一并设置）。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch 未安装时跳过（模型层依赖，setup 会校验）
