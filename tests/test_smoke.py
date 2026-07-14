"""Smoke test: verify imports + tiny end-to-end run.

This is intentionally tiny — verifies that:
1. All layers / models / data / train modules import.
2. Each model can do a forward pass with the expected input shape.
3. A 1-epoch training run completes on a synthetic dataset.
4. Results are saved to JSON correctly.

Does NOT verify accuracy or correctness of training. That's what real
experiments are for.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pulsar.layers import LIFNeuron, PulseLayer, MembraneNorm, SEWResidual, BinaryActivation
from pulsar.coding import (
    PoissonRateEncoder, LatencyEncoder, ConstantEncoder, LearnedEncoder,
    RateDecoder, LastStepDecoder, AttentionDecoder,
)
from pulsar.models import MODEL_REGISTRY
from pulsar.train import Trainer, TrainConfig, set_seed
from pulsar.data import DATASET_REGISTRY


def test_imports():
    print("[test] imports OK")


def test_layers():
    print("[test] layers...")
    B, F = 4, 16
    x = torch.randn(B, F)
    # LIF
    lif = LIFNeuron(decay=0.9)
    s = lif(x)
    assert s.shape == (B, F), f"LIF shape: {s.shape}"
    assert ((s == 0) | (s == 1)).all(), "LIF output not binary"
    lif.reset_state()
    # PulseLayer
    pl = PulseLayer(tau_init=1.0, tau_min=0.1)
    pl.train()
    s = pl(x)
    assert s.shape == (B, F)
    pl.eval()
    s = pl(x)
    assert ((s == 0) | (s == 1)).all(), "PulseLayer eval not binary"
    # MembraneNorm
    mn = MembraneNorm(F)
    y = mn(x)
    assert y.shape == (B, F)
    # MembraneNorm on 4D
    x4 = torch.randn(2, 8, 4, 4)
    mn4 = MembraneNorm(8)
    y4 = mn4(x4)
    assert y4.shape == (2, 8, 4, 4)
    # BinaryActivation
    bn = BinaryActivation(mode="zero_one")
    bn.train()
    s = bn(x)
    assert s.shape == (B, F)
    print("  layers OK")


def test_encoders_decoders():
    print("[test] encoders/decoders...")
    B, F, T = 4, 16, 8
    x = torch.rand(B, F)
    # Encoders
    for EncCls in [PoissonRateEncoder, LatencyEncoder, ConstantEncoder]:
        enc = EncCls(T=T)
        s = enc(x)
        assert s.shape == (T, B, F), f"{EncCls.__name__}: {s.shape}"
    le = LearnedEncoder(F, F, T=T)
    s = le(x)
    assert s.shape == (T, B, F)
    # Decoders
    spikes = torch.randint(0, 2, (T, B, F)).float()
    for DecCls in [RateDecoder, LastStepDecoder, AttentionDecoder]:
        dec = DecCls(F, 10)
        out = dec(spikes)
        assert out.shape == (B, 10), f"{DecCls.__name__}: {out.shape}"
    print("  encoders/decoders OK")


def test_models_forward():
    print("[test] models forward...")
    B = 4
    # MLP-style on flat input
    x_flat = torch.rand(B, 784)
    for name in ["ann_mlp", "bnn_mlp", "snn_mlp", "ann_to_snn_mlp", "pulsar_mlp"]:
        cls = MODEL_REGISTRY[name]
        try:
            model = cls(in_features=784, hidden_dims=[64, 32], num_classes=10, T=2)
        except TypeError:
            # Some models don't take all args
            model = cls(in_features=784, hidden_dims=[64, 32], num_classes=10)
        model.train()
        out = model(x_flat)
        assert out.shape == (B, 10), f"{name}: {out.shape}"
        print(f"  {name}: OK ({sum(p.numel() for p in model.parameters()):,} params)")

    # CNN-style on image input
    x_img = torch.rand(B, 1, 28, 28)
    for name in ["ann_cnn", "bnn_cnn", "snn_cnn", "ann_to_snn_cnn", "pulsar_cnn"]:
        cls = MODEL_REGISTRY[name]
        try:
            model = cls(in_channels=1, num_classes=10, T=2, width=8, input_size=28)
        except TypeError:
            model = cls(in_channels=1, num_classes=10, width=8, input_size=28)
        model.train()
        out = model(x_img)
        assert out.shape == (B, 10), f"{name}: {out.shape}"
        print(f"  {name}: OK ({sum(p.numel() for p in model.parameters()):,} params)")
    print("  models OK")


def test_ann_to_snn_conversion():
    print("[test] ANN-to-SNN conversion...")
    model = MODEL_REGISTRY["ann_to_snn_mlp"](
        in_features=784, hidden_dims=[64, 32], num_classes=10, T=4,
    )
    model.train()
    x = torch.rand(4, 784)
    # Train mode forward
    out_ann = model(x)
    assert out_ann.shape == (4, 10)
    # Convert
    model.eval()
    model.convert(x)
    # SNN mode forward
    out_snn = model(x)
    assert out_snn.shape == (4, 10)
    print("  ANN-to-SNN conversion OK")


def test_pulse_layer_annealing():
    print("[test] PulseLayer annealing...")
    pl = PulseLayer(tau_init=1.0, tau_min=0.1)
    assert abs(float(pl.tau.item()) - 1.0) < 1e-6
    pl.anneal_to(0.5)
    assert abs(float(pl.tau.item()) - 0.55) < 1e-6, f"tau after 0.5: {pl.tau.item()}"
    pl.anneal_to(1.0)
    assert abs(float(pl.tau.item()) - 0.1) < 1e-6
    pl.set_tau(0.7)
    assert abs(float(pl.tau.item()) - 0.7) < 1e-6
    print("  annealing OK")


def test_pulsar_annealing_in_training():
    print("[test] Pulsar annealing interface...")
    model = MODEL_REGISTRY["pulsar_mlp"](
        in_features=784, hidden_dims=[64, 32], num_classes=10, T=2,
    )
    model.set_tau(0.5)
    model.anneal_to(0.0)
    # All pulse layers should have tau=1.0
    for p in model.pulses:
        assert abs(float(p.tau.item()) - 1.0) < 1e-6
    model.anneal_to(1.0)
    for p in model.pulses:
        assert abs(float(p.tau.item()) - 0.1) < 1e-6
    print("  annealing interface OK")


def test_trainer_smoke():
    """Run a 1-epoch training on a tiny synthetic dataset."""
    print("[test] trainer smoke test (1 epoch, synthetic)...")
    set_seed(42)

    # Synthetic data: 100 samples, 784 features, 10 classes
    n = 100
    X = torch.rand(n, 784)
    Y = torch.randint(0, 10, (n,))

    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(X, Y)
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(ds, batch_size=32, shuffle=False)
    data_info = {"in_features": 784, "num_classes": 10, "input_shape": (784,), "T": 1, "type": "frame"}

    cfg = TrainConfig(
        experiment_name="smoke_test",
        model_name="ann_mlp",
        dataset_name="synthetic",
        seed=42,
        epochs=1,
        batch_size=32,
        lr=1e-3,
        log_every=100,
    )

    def model_factory():
        return MODEL_REGISTRY["ann_mlp"](in_features=784, hidden_dims=[32], num_classes=10)

    trainer = Trainer(cfg, model_factory, train_loader, test_loader, data_info)
    with tempfile.TemporaryDirectory() as tmp:
        result = trainer.run()
        # Verify result
        assert result.final_metrics["acc@1"] > 0.0
        assert len(result.train_history) == 1
        assert len(result.eval_history) == 1
        from pulsar.train.trainer import save_result
        out_path = Path(tmp) / "result.json"
        save_result(result, out_path)
        assert out_path.exists()
        import json
        with open(out_path) as f:
            d = json.load(f)
        assert "config" in d
        assert "final_metrics" in d
    print("  trainer OK")


def test_data_loaders_smoke():
    """Verify dataset registry works (without actually downloading)."""
    print("[test] data registry...")
    # Just verify the registry has expected keys
    assert "mnist" in DATASET_REGISTRY
    assert "fashion_mnist" in DATASET_REGISTRY
    assert "cifar10" in DATASET_REGISTRY
    assert "shd" in DATASET_REGISTRY
    print("  data registry OK")


def main():
    print("\n" + "="*70)
    print("  PULSAR SMOKE TESTS")
    print("="*70)
    test_imports()
    test_layers()
    test_encoders_decoders()
    test_models_forward()
    test_ann_to_snn_conversion()
    test_pulse_layer_annealing()
    test_pulsar_annealing_in_training()
    test_data_loaders_smoke()
    test_trainer_smoke()
    print("\n" + "="*70)
    print("  ALL TESTS PASSED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
