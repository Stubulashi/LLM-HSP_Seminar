"""Random-seed control (the mechanism implementing scaf.md Principle 2; the mechanism itself has no documented basis and is a supplementary convention).

job_seed derivation: hash(model|story|repeat|seed_master), so the same job with the same config reproduces the same result.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np


def job_seed(model: str, story: str, repeat: int, seed_master: int) -> int:
    """Deterministically derive the job-level seed."""
    raw = f"{model}|{story}|{repeat}|{seed_master}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def apply_seed(seed: int) -> None:
    """Apply the seed to random / numpy (torch too when available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # skip when torch is not installed (a model-layer dependency; setup checks it)
