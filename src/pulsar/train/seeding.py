"""Deterministic seeding for reproducible experiments."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """Set all relevant seeds.

    Args:
        seed: integer seed.
        deterministic: if True, force deterministic algorithms in torch
            (may slow down some ops but ensures reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Use deterministic algorithms where possible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # On CPU this is mostly a no-op but sets the flag.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def get_determinism_report() -> dict:
    """Return a small dict of torch / env state relevant to determinism."""
    return {
        "torch_version": torch.__version__,
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
        "num_threads": torch.get_num_threads(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
