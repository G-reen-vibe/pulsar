#!/usr/bin/env bash
# Run all experiments (all model × dataset combos, all seeds).
#
# Usage:
#   ./scripts/run_all.sh             # run everything
#   ./scripts/run_all.sh shd         # run only SHD experiments
#   ./scripts/run_all.sh pulsar      # run only Pulsar experiments
#   ./scripts/run_all.sh --no-cifar  # skip CIFAR-10 (slow on CPU)
#
# Results land in results/<experiment_name>/seed_<seed>/result.json

set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
CONFIG_DIR="configs"
RESULTS_DIR="results"
FILTER="$1"

# All config files
CONFIGS=(
  shd_ann shd_ann_to_snn shd_bnn shd_snn shd_pulsar
  mnist_ann mnist_ann_to_snn mnist_bnn mnist_snn mnist_pulsar
  fashion_ann fashion_ann_to_snn fashion_bnn fashion_snn fashion_pulsar
  cifar10_ann cifar10_ann_to_snn cifar10_bnn cifar10_snn cifar10_pulsar
)

for cfg in "${CONFIGS[@]}"; do
  # Filter
  if [[ -n "$FILTER" && "$FILTER" != "--"* && "$cfg" != *"$FILTER"* ]]; then
    continue
  fi
  # Skip CIFAR if requested
  if [[ "$2" == "--no-cifar" || "$FILTER" == "--no-cifar" ]]; then
    if [[ "$cfg" == cifar10* ]]; then continue; fi
  fi

  config_path="$CONFIG_DIR/${cfg}.yaml"
  if [[ ! -f "$config_path" ]]; then
    echo "  SKIP: $config_path does not exist"
    continue
  fi

  echo ""
  echo "================================================================"
  echo "  Running: $cfg"
  echo "================================================================"
  $PYTHON experiments/run_experiment.py --config "$config_path" --output_root "$RESULTS_DIR" || {
    echo "  FAILED: $cfg"
    # Don't exit — keep running other experiments
  }
done

echo ""
echo "================================================================"
echo "  All experiments complete. Aggregating..."
echo "================================================================"
$PYTHON analysis/aggregate.py --results_dir "$RESULTS_DIR" --output "$RESULTS_DIR/summary.json"
$PYTHON analysis/plot.py --results_dir "$RESULTS_DIR" --output_dir "$RESULTS_DIR/plots"
echo ""
echo "Done. See $RESULTS_DIR/summary.json and $RESULTS_DIR/plots/."
