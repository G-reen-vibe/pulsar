"""Quick iteration runner for research rounds.

Usage:
    python experiments/quick.py --name r03_stateful --pulse_stateful

Each flag maps to a PulsarMLP kwarg. Defaults match the baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from pulsar.data import get_shd_loaders
from pulsar.models import MODEL_REGISTRY
from pulsar.train import Trainer, TrainConfig, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--hidden", type=str, default="256,128")
    p.add_argument("--hard", action="store_true", default=True)
    p.add_argument("--no_hard", dest="hard", action="store_false")
    p.add_argument("--pulse_stateful", action="store_true", default=False)
    p.add_argument("--decay_init", type=float, default=0.9)
    p.add_argument("--tau_init", type=float, default=1.0)
    p.add_argument("--tau_min", type=float, default=0.1)
    p.add_argument("--anneal", type=str, default="cosine")
    p.add_argument("--anneal_warmup", type=float, default=0.1)
    p.add_argument("--decoder", type=str, default="attention")
    p.add_argument("--encoder", type=str, default="poisson")
    p.add_argument("--stateful", action="store_true", default=False)
    p.add_argument("--learnable_threshold", action="store_true", default=False)
    p.add_argument("--recurrent", action="store_true", default=False)
    p.add_argument("--norm_type", type=str, default="membrane", choices=["membrane", "batch", "layer", "none"])
    p.add_argument("--num_spike_levels", type=int, default=2)
    p.add_argument("--residual", action="store_true", default=False)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--spike_mode", type=str, default="gumbel", choices=["gumbel", "deterministic"])
    p.add_argument("--spike_l1", type=float, default=0.0)
    p.add_argument("--leaky", type=float, default=0.0)
    p.add_argument("--membrane_readout", action="store_true", default=False)
    p.add_argument("--output_mode", type=str, default="spike", choices=["spike", "gate"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", type=str, default="adam")
    p.add_argument("--dataset", type=str, default="shd")
    p.add_argument("--width", type=int, default=16)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    hidden = [int(x) for x in args.hidden.split(",")]

    print(f"\n{'='*70}\n  Round: {args.name}\n{'='*70}")
    print(f"  Config: hard={args.hard}, pulse_stateful={args.pulse_stateful}, "
          f"decay_init={args.decay_init}, tau=({args.tau_init}→{args.tau_min}), "
          f"anneal={args.anneal}, decoder={args.decoder}")

    # Load data
    if args.dataset == "shd":
        train_loader, test_loader, info = get_shd_loaders(
            root="data/shd", T=args.T, batch_size=args.batch_size, num_workers=0,
        )
        is_shd = True
    elif args.dataset == "mnist":
        from pulsar.data import get_mnist_loaders
        train_loader, test_loader, info = get_mnist_loaders(
            root="data/mnist", batch_size=args.batch_size, num_workers=0, flatten=False,
        )
        is_shd = False
    else:
        raise ValueError(args.dataset)

    results = []
    for seed in seeds:
        set_seed(seed)
        cfg = TrainConfig(
            experiment_name=args.name,
            model_name="pulsar_mlp" if is_shd else "pulsar_cnn",
            dataset_name=args.dataset,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            anneal_schedule=args.anneal,
            anneal_warmup_frac=args.anneal_warmup,
            eval_every=2,
            log_every=100,
            spike_l1=args.spike_l1,
        )

        if is_shd:
            model = MODEL_REGISTRY["pulsar_mlp"](
                in_features=info["in_features"],
                hidden_dims=hidden,
                num_classes=info["num_classes"],
                T=args.T,
                tau_init=args.tau_init,
                tau_min=args.tau_min,
                encoder=args.encoder,
                decoder=args.decoder,
                stateful=args.stateful,
                input_is_spike_train=True,
                hard=args.hard,
                pulse_stateful=args.pulse_stateful,
                decay_init=args.decay_init,
                learnable_threshold=args.learnable_threshold,
                recurrent=args.recurrent,
                norm_type=args.norm_type,
                num_spike_levels=args.num_spike_levels,
                residual=args.residual,
                dropout=args.dropout,
                spike_mode=args.spike_mode,
                leaky=args.leaky,
                membrane_readout=args.membrane_readout,
                output_mode=args.output_mode,
            )
        else:
            model = MODEL_REGISTRY["pulsar_cnn"](
                in_channels=info.get("in_channels", 1),
                num_classes=info["num_classes"],
                T=args.T,
                width=args.width,
                input_size=info.get("input_size", 28),
                tau_init=args.tau_init,
                tau_min=args.tau_min,
                encoder=args.encoder,
                decoder=args.decoder,
                stateful=args.stateful,
                hard=args.hard,
            )

        def factory(_m=model):
            return _m

        trainer = Trainer(cfg, factory, train_loader, test_loader, info)
        t0 = time.time()
        result = trainer.run()
        elapsed = time.time() - t0
        results.append({
            "seed": seed,
            "final_acc@1": result.final_metrics["acc@1"],
            "best_acc@1": result.best_metrics.get("acc@1", 0.0),
            "final_acc@5": result.final_metrics["acc@5"],
            "train_time_s": elapsed,
        })

    # Summary
    print(f"\n{'='*70}\n  SUMMARY: {args.name}\n{'='*70}")
    for r in results:
        print(f"  seed {r['seed']}: final={r['final_acc@1']:.4f} best={r['best_acc@1']:.4f} "
              f"acc@5={r['final_acc@5']:.4f} time={r['train_time_s']:.1f}s")
    if len(results) > 1:
        import statistics
        accs = [r["final_acc@1"] for r in results]
        print(f"  Mean final acc@1: {statistics.mean(accs):.4f} ± "
              f"{statistics.stdev(accs) if len(accs) > 1 else 0:.4f}")

    # Save to results ledger
    ledger = Path("results/ledger.jsonl")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        for r in results:
            entry = {"name": args.name, **r, **vars(args)}
            f.write(json.dumps(entry) + "\n")
    print(f"  Appended to {ledger}")


if __name__ == "__main__":
    main()
