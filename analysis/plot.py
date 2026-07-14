"""Plot results: error-bar plots and learning curves.

Usage:
    python analysis/plot.py --results_dir results --output_dir results/plots

Produces:
  - accuracy_bars.png: bar chart of final acc@1 with 95% CI error bars,
    grouped by dataset, one bar per model.
  - learning_curves.png: per-epoch test acc@1, mean ± std across seeds,
    one subplot per dataset, one line per model.
  - timing_bars.png: train time and samples/sec bar charts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Font setup (per repo rules)
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Use the constrained_layout engine exclusively (per repo rules).
LAYOUT_KW = dict(constrained_layout=True)


# Color palette per model type
COLORS = {
    "ann": "#1f77b4",          # blue
    "ann_to_snn": "#9467bd",   # purple
    "bnn": "#8c564b",          # brown
    "snn": "#ff7f0e",          # orange
    "pulsar": "#d62728",       # red (highlight our method)
}


def model_type(name: str) -> str:
    for t in ("ann_to_snn", "pulsar", "snn", "bnn", "ann"):
        if t in name:
            return t
    return "other"


def dataset_of(name: str) -> str:
    for suffix in ("_pulsar", "_ann_to_snn", "_snn", "_bnn", "_ann"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def collect_results(results_dir: Path) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            rf = seed_dir / "result.json"
            if not rf.exists():
                continue
            with open(rf) as f:
                data = json.load(f)
            data["_dataset"] = data.get("config", {}).get("dataset_name", "")
            data["_model_name"] = data.get("config", {}).get("model_name", "")
            data["_model_type"] = model_type(data["_model_name"])
            out[exp_dir.name].append(data)
    return dict(out)


def plot_accuracy_bars(aggregated: dict, output_path: Path):
    """Grouped bar chart: per dataset, one bar per model, error bars = 95% CI."""
    # Group by dataset (using the 'dataset' field from aggregated data)
    by_ds = defaultdict(dict)
    for exp_name, agg in aggregated.items():
        ds = agg.get("dataset", "")
        mt = agg.get("model_type", "")
        by_ds[ds][mt] = agg

    datasets = sorted(by_ds.keys())
    model_types = ["ann", "ann_to_snn", "bnn", "snn", "pulsar"]
    n_datasets = len(datasets)
    n_models = len(model_types)

    fig, ax = plt.subplots(figsize=(max(8, n_datasets * 2.0), 5), **LAYOUT_KW)
    bar_width = 0.8 / n_models
    x = np.arange(n_datasets)
    for i, mt in enumerate(model_types):
        means = []
        cis = []
        for ds in datasets:
            agg = by_ds[ds].get(mt)
            if agg is None:
                means.append(0.0)
                cis.append(0.0)
            else:
                means.append(agg["final_acc@1"]["mean"])
                cis.append(agg["final_acc@1"]["ci_95"])
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=cis, label=mt,
                      color=COLORS[mt], alpha=0.85,
                      edgecolor="black", linewidth=0.5,
                      error_kw={"elinewidth": 1.0, "capsize": 3})
        if mt == "pulsar":
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Final test accuracy (acc@1)")
    ax.set_title("Final test accuracy across models and datasets (mean ± 95% CI)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_learning_curves(raw: dict, output_path: Path):
    """Per-dataset learning curves: test acc@1 vs epoch, mean ± std across seeds."""
    by_ds = defaultdict(dict)
    for exp_name, runs in raw.items():
        # Use dataset and model_type from the result config
        if not runs:
            continue
        ds = runs[0].get("_dataset", "") or runs[0].get("config", {}).get("dataset_name", "")
        mt = runs[0].get("_model_type", "") or model_type(runs[0].get("config", {}).get("model_name", ""))
        by_ds[ds][mt] = runs

    datasets = sorted(by_ds.keys())
    n_ds = len(datasets)
    if n_ds == 0:
        return
    n_cols = min(2, n_ds)
    n_rows = math.ceil(n_ds / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5), **LAYOUT_KW)
    axes = np.atleast_2d(axes).flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for mt in ["ann", "ann_to_snn", "bnn", "snn", "pulsar"]:
            runs = by_ds[ds].get(mt)
            if not runs:
                continue
            per_seed_curves = []
            for r in runs:
                eval_hist = r.get("eval_history", [])
                if not eval_hist:
                    continue
                epochs = [e["epoch"] for e in eval_hist]
                accs = [e["acc@1"] for e in eval_hist]
                per_seed_curves.append((epochs, accs))
            if not per_seed_curves:
                continue
            ref_epochs = per_seed_curves[0][0]
            acc_matrix = np.zeros((len(per_seed_curves), len(ref_epochs)))
            for i, (_, accs) in enumerate(per_seed_curves):
                acc_matrix[i] = accs
            mean = acc_matrix.mean(axis=0)
            std = acc_matrix.std(axis=0, ddof=1) if acc_matrix.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(ref_epochs, mean, label=mt, color=COLORS[mt], linewidth=2)
            ax.fill_between(ref_epochs, mean - std, mean + std, color=COLORS[mt], alpha=0.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test acc@1")
        ax.set_title(ds)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=8)

    for i in range(len(datasets), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Learning curves (mean ± std across seeds)", fontsize=12)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_timing_bars(aggregated: dict, output_path: Path):
    """Grouped bar chart: per dataset, train time, with CI error bars."""
    by_ds = defaultdict(dict)
    for exp_name, agg in aggregated.items():
        ds = agg.get("dataset", "")
        mt = agg.get("model_type", "")
        by_ds[ds][mt] = agg

    datasets = sorted(by_ds.keys())
    model_types = ["ann", "ann_to_snn", "bnn", "snn", "pulsar"]
    n_datasets = len(datasets)
    n_models = len(model_types)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), **LAYOUT_KW)
    bar_width = 0.8 / n_models
    x = np.arange(n_datasets)
    for i, mt in enumerate(model_types):
        times = []
        time_cis = []
        sps = []
        sps_cis = []
        for ds in datasets:
            agg = by_ds[ds].get(mt)
            if agg is None:
                times.append(0.0)
                time_cis.append(0.0)
                sps.append(0.0)
                sps_cis.append(0.0)
            else:
                times.append(agg["train_time_s"]["mean"])
                time_cis.append(agg["train_time_s"]["ci_95"])
                sps.append(agg["train_samples_per_sec"]["mean"])
                sps_cis.append(agg["train_samples_per_sec"]["ci_95"])
        offset = (i - (n_models - 1) / 2) * bar_width
        ax1.bar(x + offset, times, bar_width, yerr=time_cis, label=mt,
                color=COLORS[mt], alpha=0.85, edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 1.0, "capsize": 3})
        ax2.bar(x + offset, sps, bar_width, yerr=sps_cis, label=mt,
                color=COLORS[mt], alpha=0.85, edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 1.0, "capsize": 3})

    for ax, ylabel, title in [(ax1, "Train time (s)", "Total training time"),
                              (ax2, "Samples / sec", "Training throughput")]:
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=8)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = collect_results(results_dir)
    if not raw:
        print(f"ERROR: no results found in {results_dir}")
        sys.exit(1)

    # Aggregate
    from aggregate import aggregate_experiment
    aggregated = {name: aggregate_experiment(results) for name, results in raw.items()}

    print("Generating plots...")
    plot_accuracy_bars(aggregated, output_dir / "accuracy_bars.png")
    plot_learning_curves(raw, output_dir / "learning_curves.png")
    plot_timing_bars(aggregated, output_dir / "timing_bars.png")
    print(f"\nAll plots in {output_dir}/")


if __name__ == "__main__":
    # Make 'aggregate' importable when running this script standalone
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
