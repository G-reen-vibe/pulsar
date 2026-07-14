"""Aggregate results across seeds, compute mean ± CI, run statistical tests.

Usage:
    python analysis/aggregate.py --results_dir results --output results/summary.json

Reads all results/<experiment_name>/seed_*/result.json files and produces:
  - per-experiment summary (mean ± 95% CI across seeds)
  - pairwise comparison table (Pulsar vs each baseline)
  - paired t-test / Wilcoxon signed-rank test on per-seed accuracies
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scistats


def collect_results(results_dir: Path) -> dict[str, list[dict]]:
    """Walk results_dir, return {experiment_name: [result_dict, ...]}.

    Each result dict gets extra keys:
        _experiment: experiment name (dir name)
        _seed: seed dir name
        _dataset: dataset name (from config)
        _model_name: model name (from config)
        _model_type: short model type tag (ann, bnn, snn, ann_to_snn, pulsar)
    """
    out = defaultdict(list)
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            result_file = seed_dir / "result.json"
            if not result_file.exists():
                continue
            with open(result_file) as f:
                data = json.load(f)
            data["_experiment"] = exp_dir.name
            data["_seed"] = seed_dir.name
            data["_dataset"] = data.get("config", {}).get("dataset_name", "")
            data["_model_name"] = data.get("config", {}).get("model_name", "")
            data["_model_type"] = _model_type(data["_model_name"])
            out[exp_dir.name].append(data)
    return dict(out)


def _model_type(model_name: str) -> str:
    """Get short model type tag from full model name."""
    for t in ("ann_to_snn", "pulsar", "snn", "bnn", "ann"):
        if t in model_name:
            return t
    return "other"


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, std, half-width of confidence interval on the mean).

    Uses t-distribution for small samples.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(np.mean(values))
    if n == 1:
        return (m, 0.0, 0.0)
    s = float(np.std(values, ddof=1))
    t_val = float(scistats.t.ppf((1 + confidence) / 2, df=n - 1))
    ci = t_val * s / math.sqrt(n)
    return (m, s, ci)


def aggregate_experiment(results: list[dict]) -> dict:
    """Aggregate a single experiment across all seeds."""
    final_acc1 = [r["final_metrics"]["acc@1"] for r in results]
    final_acc5 = [r["final_metrics"].get("acc@5", 0.0) for r in results]
    best_acc1 = [r["best_metrics"].get("acc@1", 0.0) for r in results]
    train_times = [r["timing"]["train_time_s"] for r in results]
    eval_times = [r["timing"]["eval_time_s"] for r in results]
    train_sps = [r["timing"]["train_samples_per_sec"] for r in results]
    eval_sps = [r["timing"]["eval_samples_per_sec"] for r in results]

    m1, s1, c1 = mean_ci(final_acc1)
    m5, s5, c5 = mean_ci(final_acc5)
    bm1, bs1, bc1 = mean_ci(best_acc1)
    tt_m, tt_s, tt_c = mean_ci(train_times)
    et_m, et_s, et_c = mean_ci(eval_times)
    tsp_m, tsp_s, tsp_c = mean_ci(train_sps)
    esp_m, esp_s, esp_c = mean_ci(eval_sps)

    return {
        "n_seeds": len(results),
        "seeds": [r["config"]["seed"] for r in results],
        "dataset": results[0].get("_dataset", ""),
        "model_name": results[0].get("_model_name", ""),
        "model_type": results[0].get("_model_type", ""),
        "final_acc@1": {"mean": m1, "std": s1, "ci_95": c1, "values": final_acc1},
        "final_acc@5": {"mean": m5, "std": s5, "ci_95": c5, "values": final_acc5},
        "best_acc@1": {"mean": bm1, "std": bs1, "ci_95": bc1, "values": best_acc1},
        "train_time_s": {"mean": tt_m, "std": tt_s, "ci_95": tt_c, "values": train_times},
        "eval_time_s": {"mean": et_m, "std": et_s, "ci_95": et_c, "values": eval_times},
        "train_samples_per_sec": {"mean": tsp_m, "std": tsp_s, "ci_95": tsp_c, "values": train_sps},
        "eval_samples_per_sec": {"mean": esp_m, "std": esp_s, "ci_95": esp_c, "values": eval_sps},
        "model_params": results[0]["model_params"],
        "model_size_mb": results[0]["model_size_mb"],
    }


def pairwise_tests(experiments: dict[str, dict], reference: str = "pulsar") -> list[dict]:
    """For each experiment, run paired t-test and Wilcoxon vs the reference.

    The reference is identified by model_type == reference. For each other
    experiment E on the same dataset, we test H0: mean(acc_E - acc_reference) = 0
    using both paired t-test and Wilcoxon signed-rank test, with seeds as the
    pairing unit (assuming both used the same seeds).
    """
    # Group by dataset
    by_dataset = defaultdict(dict)
    for name, agg in experiments.items():
        ds = agg.get("dataset", "")
        mt = agg.get("model_type", "")
        by_dataset[ds][mt] = (name, agg)

    rows = []
    for ds, exps in by_dataset.items():
        ref = exps.get(reference)
        if ref is None:
            continue
        ref_name, ref_data = ref
        ref_accs = ref_data["final_acc@1"]["values"]
        ref_seeds = ref_data["seeds"]
        for mt, (other_name, other_data) in exps.items():
            if mt == reference:
                continue
            other_accs_by_seed = dict(zip(other_data["seeds"], other_data["final_acc@1"]["values"]))
            ref_accs_by_seed = dict(zip(ref_seeds, ref_accs))
            common_seeds = sorted(set(other_data["seeds"]) & set(ref_seeds))
            if not common_seeds:
                continue
            a = np.array([other_accs_by_seed[s] for s in common_seeds])
            b = np.array([ref_accs_by_seed[s] for s in common_seeds])
            delta = a - b
            try:
                t_stat, p_t = scistats.ttest_rel(a, b)
                t_stat, p_t = float(t_stat), float(p_t)
            except Exception:
                t_stat, p_t = float("nan"), float("nan")
            try:
                if len(common_seeds) >= 5:
                    w_stat, p_w = scistats.wilcoxon(a, b, zero_method="wilcox")
                    w_stat, p_w = float(w_stat), float(p_w)
                else:
                    w_stat, p_w = float("nan"), float("nan")
            except Exception:
                w_stat, p_w = float("nan"), float("nan")

            rows.append({
                "dataset": ds,
                "experiment": other_name,
                "reference": ref_name,
                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "n_seeds_paired": len(common_seeds),
                "paired_t_stat": t_stat,
                "paired_t_pvalue": p_t,
                "wilcoxon_stat": w_stat,
                "wilcoxon_pvalue": p_w,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/summary.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir {results_dir} does not exist")
        sys.exit(1)

    raw = collect_results(results_dir)
    if not raw:
        print(f"ERROR: no results found in {results_dir}")
        sys.exit(1)

    # Aggregate per experiment
    aggregated = {name: aggregate_experiment(results) for name, results in raw.items()}

    # Pairwise tests (Pulsar vs each baseline, per dataset)
    pairwise = pairwise_tests(aggregated, reference="pulsar")

    summary = {
        "experiments": aggregated,
        "pairwise_tests": pairwise,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")

    # Print human-readable summary
    print(f"\n{'='*90}")
    print(f"{'Experiment':<30} {'Final acc@1':>20} {'Best acc@1':>20} {'Train time':>15}")
    print(f"{'':<30} {'mean ± 95% CI':>20} {'mean ± 95% CI':>20} {'mean ± 95% CI':>15}")
    print(f"{'='*90}")
    for name, agg in sorted(aggregated.items()):
        f1 = f"{agg['final_acc@1']['mean']:.4f} ± {agg['final_acc@1']['ci_95']:.4f}"
        b1 = f"{agg['best_acc@1']['mean']:.4f} ± {agg['best_acc@1']['ci_95']:.4f}"
        tt = f"{agg['train_time_s']['mean']:.1f} ± {agg['train_time_s']['ci_95']:.1f}s"
        print(f"{name:<30} {f1:>20} {b1:>20} {tt:>15}")
    print(f"{'='*90}")

    if pairwise:
        print(f"\n{'Pairwise tests (reference = Pulsar)':<60}")
        print(f"{'-'*90}")
        print(f"{'Experiment':<30} {'Δmean':>10} {'t-stat':>10} {'p (t-test)':>12} {'p (Wilcoxon)':>14}")
        print(f"{'-'*90}")
        for row in pairwise:
            print(f"{row['experiment']:<30} {row['delta_mean']:>+10.4f} {row['paired_t_stat']:>10.3f} "
                  f"{row['paired_t_pvalue']:>12.4g} {row['wilcoxon_pvalue']:>14.4g}")
        print(f"{'-'*90}")


if __name__ == "__main__":
    main()
