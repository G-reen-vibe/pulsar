"""Annealing schedules for Gumbel-softmax temperature."""
from __future__ import annotations

import math


def linear_anneal(progress: float, init: float, final: float) -> float:
    """Linear from init → final as progress goes 0 → 1."""
    progress = max(0.0, min(1.0, progress))
    return init + (final - init) * progress


def cosine_anneal(progress: float, init: float, final: float) -> float:
    """Cosine from init → final as progress goes 0 → 1.

    Smoother than linear — slow start, fast in middle, slow end.
    """
    progress = max(0.0, min(1.0, progress))
    return final + 0.5 * (init - final) * (1 + math.cos(math.pi * progress))


def step_anneal(progress: float, init: float, final: float, num_steps: int = 3) -> float:
    """Step from init → final in num_steps equal jumps."""
    progress = max(0.0, min(1.0, progress))
    step_size = 1.0 / num_steps
    step_idx = min(num_steps, int(progress / step_size))
    return init + (final - init) * (step_idx / num_steps)


SCHEDULES = {
    "linear": linear_anneal,
    "cosine": cosine_anneal,
    "step": step_anneal,
}
