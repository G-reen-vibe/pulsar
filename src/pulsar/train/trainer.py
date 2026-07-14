"""Trainer: uniform training loop for all model types.

Handles:
- ANN / BNN: standard single-pass forward.
- SNN / Pulsar: forward includes time loop inside the model.
- ANN-to-SNN: trains as ANN, then converts to SNN at eval.
- Gumbel-softmax temperature annealing for Pulsar.
- Multi-seed support: each run is logged with its seed.
- Structured JSON logging per epoch, plus final summary.
- Inference timing (samples/sec) for fair compute comparison.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .seeding import set_seed
from .metrics import accuracy, MetricHistory, Timer
from .annealing import SCHEDULES
from ..models.ann_to_snn import ANNtoSNNMLP, ANNtoSNNCNN


@dataclass
class TrainConfig:
    """Configuration for a single training run.

    Stored alongside results for reproducibility.
    """
    # Experiment identity
    experiment_name: str
    model_name: str
    dataset_name: str
    seed: int

    # Optimization
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # adam | adamw | sgd
    lr_schedule: str = "none"  # none | cosine | step
    warmup_epochs: int = 0

    # Annealing (Pulsar only)
    anneal_schedule: str = "cosine"  # linear | cosine | step
    anneal_warmup_frac: float = 0.1  # fraction of training to keep tau=tau_init

    # Evaluation
    eval_every: int = 1
    eval_batch_size: int = 256

    # ANN-to-SNN specific
    ann_to_snn_T: int = 16  # timesteps at SNN inference

    # Resource
    num_workers: int = 0
    log_every: int = 50  # log to stdout every N batches

    # Misc
    save_dir: str = "results"


@dataclass
class RunResult:
    """Final result of one training run (one seed)."""
    config: dict
    determinism_report: dict
    train_history: list[dict]  # per-epoch
    eval_history: list[dict]   # per-eval
    final_metrics: dict        # final test metrics
    best_metrics: dict         # best test metrics (by acc@1)
    timing: dict               # train_time, eval_time, samples_per_sec
    model_params: int          # total params
    model_size_mb: float       # rough size estimate


def count_params(model: nn.Module) -> tuple[int, float]:
    """Return (total params, size in MB at fp32)."""
    total = sum(p.numel() for p in model.parameters())
    size_mb = total * 4 / (1024 * 1024)
    return total, size_mb


class Trainer:
    """Runs a single training experiment.

    Usage:
        cfg = TrainConfig(...)
        trainer = Trainer(cfg, model_factory, data_factory)
        result = trainer.run()
    """

    def __init__(
        self,
        config: TrainConfig,
        model_factory: Callable[[], nn.Module],
        train_loader: DataLoader,
        test_loader: DataLoader,
        data_info: dict,
    ):
        self.config = config
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_info = data_info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> RunResult:
        cfg = self.config
        set_seed(cfg.seed)
        determinism_report = self._determinism_report()

        # Build model
        model = self.model_factory().to(self.device)
        n_params, size_mb = count_params(model)

        # Optimizer
        opt = self._make_optimizer(model)

        # Loss
        loss_fn = nn.CrossEntropyLoss()

        # Logging
        train_history = []
        eval_history = []
        best_acc = -1.0
        best_metrics = {}
        train_timer = Timer()
        eval_timer = Timer()

        print(f"\n{'='*70}")
        print(f"  Experiment: {cfg.experiment_name}")
        print(f"  Model: {cfg.model_name} | Dataset: {cfg.dataset_name} | Seed: {cfg.seed}")
        print(f"  Params: {n_params:,} ({size_mb:.2f} MB) | Device: {self.device}")
        print(f"{'='*70}")

        # Determine if this is an ANN-to-SNN model — needs special handling
        is_ann_to_snn = isinstance(model, (ANNtoSNNMLP, ANNtoSNNCNN))

        # Determine if model supports Gumbel annealing
        has_anneal = hasattr(model, "anneal_to")
        anneal_fn = SCHEDULES.get(cfg.anneal_schedule, SCHEDULES["cosine"])

        total_steps = cfg.epochs * len(self.train_loader)

        for epoch in range(cfg.epochs):
            # === TRAIN ===
            model.train()
            epoch_metrics = MetricHistory()
            train_timer.__enter__()
            for step, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # Anneal Gumbel temperature (Pulsar only)
                if has_anneal:
                    global_step = epoch * len(self.train_loader) + step
                    progress = global_step / max(1, total_steps)
                    # Warmup: keep tau = tau_init for first warmup_frac
                    if progress < cfg.anneal_warmup_frac:
                        model.set_tau(1.0)  # tau_init is 1.0 in our PulseLayer
                    else:
                        adj_progress = (progress - cfg.anneal_warmup_frac) / (1.0 - cfg.anneal_warmup_frac)
                        # Anneal from 1.0 → 0.1
                        new_tau = anneal_fn(adj_progress, init=1.0, final=0.1)
                        model.set_tau(new_tau)

                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    m = {"loss": float(loss.item())}
                    m.update(accuracy(logits, y, topk=(1, 5)))
                    epoch_metrics.update(m, n_samples=x.shape[0])

                if (step + 1) % cfg.log_every == 0:
                    cur = epoch_metrics.compute()
                    print(f"  epoch {epoch+1}/{cfg.epochs} step {step+1}/{len(self.train_loader)}: "
                          f"loss={cur['loss']:.4f} acc@1={cur['acc@1']:.4f}")
            train_timer.__exit__()

            epoch_summary = epoch_metrics.compute()
            epoch_summary["epoch"] = epoch + 1
            epoch_summary["lr"] = opt.param_groups[0]["lr"]
            train_history.append(epoch_summary)
            print(f"  [epoch {epoch+1}] train: loss={epoch_summary['loss']:.4f} acc@1={epoch_summary['acc@1']:.4f}")

            # === EVAL ===
            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                eval_metrics = self._evaluate(model, loss_fn, is_ann_to_snn, eval_timer)
                eval_metrics["epoch"] = epoch + 1
                eval_history.append(eval_metrics)
                print(f"  [epoch {epoch+1}] test:  loss={eval_metrics['loss']:.4f} acc@1={eval_metrics['acc@1']:.4f} acc@5={eval_metrics['acc@5']:.4f}")

                if eval_metrics["acc@1"] > best_acc:
                    best_acc = eval_metrics["acc@1"]
                    best_metrics = eval_metrics.copy()

        # Final eval
        final = self._evaluate(model, loss_fn, is_ann_to_snn, Timer())

        timing = {
            "train_time_s": train_timer.total,
            "eval_time_s": eval_timer.total,
            "train_samples_per_sec": sum(len(b[1]) for b in self.train_loader) * cfg.epochs / max(1e-9, train_timer.total),
            "eval_samples_per_sec": sum(len(b[1]) for b in self.test_loader) / max(1e-9, eval_timer.total),
        }
        print(f"\n  Final: acc@1={final['acc@1']:.4f} acc@5={final['acc@5']:.4f}")
        print(f"  Best:  acc@1={best_metrics.get('acc@1', 0):.4f}")
        print(f"  Timing: train={timing['train_time_s']:.1f}s eval={timing['eval_time_s']:.1f}s")

        return RunResult(
            config=asdict(cfg),
            determinism_report=determinism_report,
            train_history=train_history,
            eval_history=eval_history,
            final_metrics=final,
            best_metrics=best_metrics,
            timing=timing,
            model_params=n_params,
            model_size_mb=size_mb,
        )

    def _evaluate(
        self, model: nn.Module, loss_fn: nn.Module, is_ann_to_snn: bool, timer: Timer,
    ) -> dict:
        """Evaluate model on test set."""
        cfg = self.config
        model.eval()

        # If ANN-to-SNN, convert now using one batch of test data
        if is_ann_to_snn:
            # Get one batch for calibration
            sample_x, _ = next(iter(self.test_loader))
            sample_x = sample_x.to(self.device)
            with torch.no_grad():
                model.convert(sample_x)

        all_metrics = MetricHistory()
        timer.__enter__()
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                m = {"loss": float(loss.item())}
                m.update(accuracy(logits, y, topk=(1, 5)))
                all_metrics.update(m, n_samples=x.shape[0])
        timer.__exit__()
        return all_metrics.compute()

    def _make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg = self.config
        params = model.parameters()
        if cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"bad optimizer: {cfg.optimizer}")

    def _determinism_report(self) -> dict:
        from .seeding import get_determinism_report
        return get_determinism_report()


def save_result(result: RunResult, path: Path):
    """Save a RunResult to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "config": result.config,
            "determinism_report": result.determinism_report,
            "train_history": result.train_history,
            "eval_history": result.eval_history,
            "final_metrics": result.final_metrics,
            "best_metrics": result.best_metrics,
            "timing": result.timing,
            "model_params": result.model_params,
            "model_size_mb": result.model_size_mb,
        }, f, indent=2)
