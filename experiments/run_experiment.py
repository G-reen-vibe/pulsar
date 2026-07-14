"""Run a single experiment from a YAML config.

Usage:
    python experiments/run_experiment.py --config configs/shd_pulsar.yaml --seed 42

The config specifies:
    - dataset: which dataset to use
    - model: which model to use
    - model_kwargs: model constructor args
    - train: training hyperparameters (epochs, lr, batch_size, etc.)
    - seeds: list of seeds to run (results are saved per-seed)

Output: results/<experiment_name>/<seed>/result.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# Make 'pulsar' importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pulsar.data import DATASET_REGISTRY
from pulsar.models import MODEL_REGISTRY
from pulsar.train import Trainer, TrainConfig, save_result


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_name: str, model_kwargs: dict, data_info: dict):
    """Instantiate a model. Auto-fills in_features / num_classes / etc. from data_info."""
    cls = MODEL_REGISTRY[model_name]
    kwargs = dict(model_kwargs)
    # Auto-fill common args from data_info
    if "in_features" in kwargs and kwargs["in_features"] is None:
        kwargs["in_features"] = data_info["in_features"]
    if "num_classes" in kwargs and kwargs["num_classes"] is None:
        kwargs["num_classes"] = data_info["num_classes"]
    if "in_channels" in kwargs and kwargs["in_channels"] is None:
        kwargs["in_channels"] = data_info.get("in_channels", 1)
    if "input_size" in kwargs and kwargs["input_size"] is None:
        kwargs["input_size"] = data_info.get("input_size", 32)

    # For MLP models on image data, flatten must be enabled
    # We don't auto-detect this — the config must set data.flatten=true for MLP
    return cls(**kwargs)


def build_data(dataset_name: str, dataset_kwargs: dict):
    """Returns (train_loader, test_loader, info)."""
    fn = DATASET_REGISTRY[dataset_name]
    return fn(**dataset_kwargs)


def run_one(config: dict, seed: int, output_root: Path):
    """Run one experiment with one seed."""
    # Build data
    print(f"\n[setup] Building dataset: {config['dataset']['name']}")
    train_loader, test_loader, data_info = build_data(
        config["dataset"]["name"], config["dataset"].get("kwargs", {})
    )
    print(f"[setup] Data info: {data_info}")

    # Build model
    print(f"[setup] Building model: {config['model']['name']}")
    model_kwargs = config["model"].get("kwargs", {})
    # Use closure to defer model construction (Trainer calls it after set_seed)
    def model_factory():
        return build_model(config["model"]["name"], model_kwargs, data_info)

    # Build train config
    train_cfg_dict = config["train"]
    exp_name = config["experiment_name"]
    cfg = TrainConfig(
        experiment_name=exp_name,
        model_name=config["model"]["name"],
        dataset_name=config["dataset"]["name"],
        seed=seed,
        epochs=train_cfg_dict.get("epochs", 30),
        batch_size=train_cfg_dict.get("batch_size", 128),
        lr=train_cfg_dict.get("lr", 1e-3),
        weight_decay=train_cfg_dict.get("weight_decay", 0.0),
        optimizer=train_cfg_dict.get("optimizer", "adam"),
        lr_schedule=train_cfg_dict.get("lr_schedule", "none"),
        warmup_epochs=train_cfg_dict.get("warmup_epochs", 0),
        anneal_schedule=train_cfg_dict.get("anneal_schedule", "cosine"),
        anneal_warmup_frac=train_cfg_dict.get("anneal_warmup_frac", 0.1),
        eval_every=train_cfg_dict.get("eval_every", 1),
        eval_batch_size=train_cfg_dict.get("eval_batch_size", 256),
        ann_to_snn_T=train_cfg_dict.get("ann_to_snn_T", 16),
        num_workers=train_cfg_dict.get("num_workers", 0),
        log_every=train_cfg_dict.get("log_every", 50),
        save_dir=str(output_root),
    )

    # Run
    trainer = Trainer(cfg, model_factory, train_loader, test_loader, data_info)
    result = trainer.run()

    # Save
    save_dir = output_root / exp_name / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_result(result, save_dir / "result.json")
    # Also save the config
    import json
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"\n[saved] {save_dir / 'result.json'}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds to run (overrides config)")
    parser.add_argument("--output_root", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(args.output_root)

    seeds = args.seeds.split(",") if args.seeds else config.get("seeds", [42])
    seeds = [int(s) for s in seeds]

    results = []
    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"# Running seed {seed}")
        print(f"{'#'*70}")
        t0 = time.time()
        result = run_one(config, seed, output_root)
        elapsed = time.time() - t0
        print(f"\n[seed {seed}] Done in {elapsed:.1f}s. Final acc@1: {result.final_metrics['acc@1']:.4f}")
        results.append((seed, result))

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for seed, r in results:
        print(f"  seed {seed}: final acc@1={r.final_metrics['acc@1']:.4f} best acc@1={r.best_metrics.get('acc@1', 0):.4f}")

    if len(results) > 1:
        import statistics
        accs = [r.final_metrics["acc@1"] for _, r in results]
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        # 95% CI for the mean (small N, so use t-distribution with n-1 dof)
        n = len(accs)
        if n > 1:
            from scipy import stats as scistats
            t_val = scistats.t.ppf(0.975, df=n - 1)
            ci = t_val * std / math.sqrt(n)
        else:
            ci = 0.0
        print(f"\n  Aggregated over {n} seeds:")
        print(f"    mean acc@1 = {mean:.4f} ± {ci:.4f} (95% CI, std={std:.4f})")


if __name__ == "__main__":
    import math  # used in summary
    main()
