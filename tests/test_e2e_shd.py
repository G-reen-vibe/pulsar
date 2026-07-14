"""End-to-end test on SHD with each model type.

Verifies that all model types can run on SHD's spike-train input format.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import Subset, DataLoader

from pulsar.data import get_shd_loaders
from pulsar.models import MODEL_REGISTRY
from pulsar.train import Trainer, TrainConfig, set_seed


def main():
    print("\n[end-to-end] Loading SHD...")
    train_loader, test_loader, info = get_shd_loaders(
        root="data/shd", T=8, batch_size=64, num_workers=0,
    )
    # Subset for speed
    train_small = Subset(train_loader.dataset, range(min(640, len(train_loader.dataset))))
    test_small = Subset(test_loader.dataset, range(min(320, len(test_loader.dataset))))
    train_loader = DataLoader(train_small, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_small, batch_size=64, shuffle=False)
    print(f"  Train: {len(train_small)} samples, Test: {len(test_small)} samples")
    print(f"  Info: {info}")

    # Test each model
    for name in ["ann_mlp", "bnn_mlp", "snn_mlp", "ann_to_snn_mlp", "pulsar_mlp"]:
        print(f"\n{'='*70}")
        print(f"  End-to-end SHD test: {name}")
        print(f"{'='*70}")
        set_seed(42)
        cfg = TrainConfig(
            experiment_name=f"e2e_shd_{name}",
            model_name=name,
            dataset_name="shd",
            seed=42,
            epochs=1,
            batch_size=64,
            lr=1e-3,
            log_every=5,
        )
        # Build model — note the different kwargs per model
        common = dict(num_classes=20)
        if name == "ann_mlp":
            model = MODEL_REGISTRY[name](in_features=5600, hidden_dims=[128, 64], **common, dropout=0.1)
        elif name == "bnn_mlp":
            model = MODEL_REGISTRY[name](in_features=5600, hidden_dims=[128, 64], **common, mode="zero_one")
        elif name == "snn_mlp":
            model = MODEL_REGISTRY[name](in_features=700, hidden_dims=[128, 64], **common, T=8, input_is_spike_train=True)
        elif name == "ann_to_snn_mlp":
            model = MODEL_REGISTRY[name](in_features=5600, hidden_dims=[128, 64], **common, T=4)
        elif name == "pulsar_mlp":
            model = MODEL_REGISTRY[name](in_features=700, hidden_dims=[128, 64], **common, T=8, input_is_spike_train=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")

        def factory(_model=model):
            return _model

        trainer = Trainer(cfg, factory, train_loader, test_loader, info)
        result = trainer.run()
        print(f"  Final acc@1: {result.final_metrics['acc@1']:.4f}")
        print(f"  Train time: {result.timing['train_time_s']:.1f}s")

    print("\n[end-to-end] All SHD models completed 1 epoch successfully.")


if __name__ == "__main__":
    main()
