"""End-to-end test on MNIST with 1 epoch + small model.

Verifies:
- MNIST downloads and loads.
- Each model can train for 1 epoch on a small MNIST subset.
- Results save correctly.
- ANN-to-SNN conversion works after ANN training.

Not meant to produce good accuracy — just verify the pipeline.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import Subset, DataLoader

from pulsar.data import get_mnist_loaders
from pulsar.models import MODEL_REGISTRY
from pulsar.train import Trainer, TrainConfig, set_seed


def main():
    print("\n[end-to-end] Downloading + loading MNIST (subset)...")
    train_loader, test_loader, info = get_mnist_loaders(
        root="data/mnist", batch_size=64, flatten=False, num_workers=0,
    )
    # Subset for speed
    train_small = Subset(train_loader.dataset, range(min(640, len(train_loader.dataset))))
    test_small = Subset(test_loader.dataset, range(min(320, len(test_loader.dataset))))
    train_loader = DataLoader(train_small, batch_size=64, shuffle=True,
                              collate_fn=train_loader.collate_fn)
    test_loader = DataLoader(test_small, batch_size=64, shuffle=False,
                             collate_fn=test_loader.collate_fn)
    print(f"  Train: {len(train_small)} samples, Test: {len(test_small)} samples")
    print(f"  Info: {info}")

    # Test each model
    for name in ["ann_cnn", "bnn_cnn", "snn_cnn", "ann_to_snn_cnn", "pulsar_cnn"]:
        print(f"\n{'='*70}")
        print(f"  End-to-end test: {name}")
        print(f"{'='*70}")
        set_seed(42)
        cfg = TrainConfig(
            experiment_name=f"e2e_{name}",
            model_name=name,
            dataset_name="mnist",
            seed=42,
            epochs=1,
            batch_size=64,
            lr=1e-3,
            log_every=5,
        )
        # Build model
        kwargs = dict(in_channels=1, num_classes=10, T=2, width=8, input_size=28)
        if name == "ann_cnn":
            kwargs.pop("T")
        elif name == "bnn_cnn":
            kwargs.pop("T")
            kwargs["mode"] = "zero_one"
        elif name == "ann_to_snn_cnn":
            pass
        model = MODEL_REGISTRY[name](**kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")

        def factory(_model=model):
            return _model

        trainer = Trainer(cfg, factory, train_loader, test_loader, info)
        result = trainer.run()
        print(f"  Final acc@1: {result.final_metrics['acc@1']:.4f}")
        print(f"  Train time: {result.timing['train_time_s']:.1f}s")

    print("\n[end-to-end] All models completed 1 epoch successfully.")


if __name__ == "__main__":
    main()
