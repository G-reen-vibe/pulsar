"""Metrics: accuracy, loss tracking, time/energy measurement."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Sequence

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Sequence[int] = (1,)) -> dict[str, float]:
    """Top-k accuracy.

    Args:
        logits: (B, num_classes) model outputs.
        targets: (B,) ground truth labels.
        topk: tuple of k values.

    Returns:
        dict {"acc@1": float, "acc@5": float, ...}
    """
    with torch.no_grad():
        max_k = max(topk)
        batch_size = targets.shape[0]
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        results = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results[f"acc@{k}"] = float(correct_k.item()) / batch_size
        return results


class Timer:
    """Context manager + accumulator for measuring wall-clock time."""

    def __init__(self):
        self.total = 0.0
        self.count = 0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.total += time.perf_counter() - self._start
        self.count += 1

    def reset(self):
        self.total = 0.0
        self.count = 0

    @property
    def mean(self) -> float:
        return self.total / max(1, self.count)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, loss: float) -> dict:
    """Standard metric dict."""
    accs = accuracy(logits, targets, topk=(1, 5))
    return {
        "loss": float(loss),
        "acc@1": accs["acc@1"],
        "acc@5": accs["acc@5"],
    }


class MetricHistory:
    """Accumulates metrics over a epoch / phase."""

    def __init__(self):
        self.values: dict[str, list[float]] = defaultdict(list)
        self.total_samples = 0

    def update(self, metrics: dict[str, float], n_samples: int = 1):
        for k, v in metrics.items():
            self.values[k].append(v * n_samples)  # weight by samples
        self.total_samples += n_samples

    def compute(self) -> dict[str, float]:
        """Sample-weighted mean of each metric."""
        return {k: sum(v) / max(1, self.total_samples) for k, v in self.values.items()}

    def reset(self):
        self.values = defaultdict(list)
        self.total_samples = 0
