# Pulsar

A from-scratch rethink of spiking neural networks (SNNs) for modern ML.

**Goal:** Make SNNs competitive with SOTA ANNs on standard tasks (vision,
classification, regression) while keeping the discrete-event / sparse-compute
character that makes them interesting.

**Status:** Infrastructure complete and validated. Research iteration not yet
started. See [RESEARCH.md](./RESEARCH.md) for the design space analysis,
candidate directions, and chosen thesis.

## Quick thesis

Treat spikes as **a discrete observation of a continuous latent state**, and
train the latent state end-to-end with **Gumbel-softmax** (annealed
temperature), while keeping inference event-driven and sparse. This bridges
the train/test gap that has plagued SNNs since the beginning.

## What's in this repo

### Core library (`src/pulsar/`)

- `layers/` — building blocks
  - `LIFNeuron` — Leaky Integrate-and-Fire with sigmoid surrogate gradient
    (standard SNN baseline)
  - `PulseLayer` — **our contribution**: Gumbel-softmax spike layer with
    annealable temperature
  - `MembraneNorm` — LayerNorm adapted for pre-spike membrane potentials
  - `SEWResidual` — spike-element-wise residual connection (Fang et al. 2021)
  - `BinaryActivation` — STE-based binarization for BNN baseline
- `coding/` — encoders and decoders
  - Encoders: `PoissonRateEncoder`, `LatencyEncoder`, `ConstantEncoder`,
    `LearnedEncoder`
  - Decoders: `RateDecoder`, `LastStepDecoder`, `AttentionDecoder`
- `models/` — full model implementations (5 model families × 2 architectures)
  - `MLPBaseline` / `CNNBaseline` — vanilla ANN with ReLU (upper bound)
  - `BNNMLP` / `BNNCNN` — binary-activation NN (closest non-SNN discrete
    baseline; isolates what the time dimension buys)
  - `SNNMLP` / `SNNCNN` — surrogate-gradient LIF (standard SNN baseline,
    snntorch-equivalent)
  - `ANNtoSNNMLP` / `ANNtoSNNCNN` — train as ANN, convert to SNN at inference
    (industrial baseline)
  - `PulsarMLP` / `PulsarCNN` — **our model**: Gumbel-softmax spikes +
    membrane norm + attention decoder
- `data/` — dataset loaders
  - `SHDDataset` — Spiking Heidelberg Digits (8157 train / 2264 test, 700
    cochlea channels, 20 classes, neuromorphic audio). Auto-downloads from
    zenkelab.org.
  - MNIST, Fashion-MNIST, CIFAR-10 via torchvision
- `train/` — training infrastructure
  - `Trainer` — uniform loop for all model types (ANN, BNN, SNN, ANN-to-SNN,
    Pulsar). Handles Gumbel temperature annealing, multi-seed, deterministic
    seeding, structured JSON logging.
  - `seeding.py` — `set_seed()` for full reproducibility (Python, NumPy,
    PyTorch, env vars)
  - `metrics.py` — top-k accuracy, sample-weighted epoch aggregation, timers
  - `annealing.py` — linear / cosine / step temperature schedules

### Experiment runner (`experiments/run_experiment.py`)

CLI that takes a YAML config and runs an experiment. Each config specifies:
- Dataset + loader kwargs
- Model + model kwargs
- Training hyperparameters (epochs, LR, optimizer, anneal schedule, etc.)
- List of seeds to run

Output: `results/<experiment_name>/seed_<seed>/result.json` per run, with
full train/eval history, timing, model params, and determinism report.

### Configs (`configs/`)

20 YAML configs covering 5 models × 4 datasets:
- Datasets: SHD, MNIST, Fashion-MNIST, CIFAR-10
- Models: ANN, BNN, SNN, ANN-to-SNN, Pulsar
- 3 seeds per (model, dataset) combo by default

### Analysis (`analysis/`)

- `aggregate.py` — walks `results/`, computes per-experiment mean ± 95% CI
  across seeds, runs pairwise statistical tests (paired t-test + Wilcoxon
  signed-rank) of Pulsar vs each baseline on each dataset.
- `plot.py` — generates three publication-quality plots with error bars:
  - `accuracy_bars.png` — final test accuracy per (model, dataset)
  - `learning_curves.png` — per-epoch test accuracy, mean ± std across seeds
  - `timing_bars.png` — training time and throughput

### Tests (`tests/`)

- `test_smoke.py` — verifies all imports, layer shapes, model forward passes,
  ANN-to-SNN conversion, Gumbel annealing interface, and a 1-epoch training
  run on synthetic data.
- `test_e2e_mnist.py` — verifies each model can train for 1 epoch on a small
  MNIST subset.
- `test_e2e_shd.py` — verifies each model can train for 1 epoch on a small
  SHD subset (tests the spike-train input mode).

### Runner script (`scripts/run_all.sh`)

```bash
./scripts/run_all.sh             # run everything
./scripts/run_all.sh shd         # only SHD experiments
./scripts/run_all.sh --no-cifar  # skip CIFAR-10
```

## How to run

```bash
# Install dependencies (PyTorch CPU build)
pip install -r requirements.txt

# Run smoke tests
python tests/test_smoke.py

# Run one experiment (1 model on 1 dataset, 3 seeds)
python experiments/run_experiment.py --config configs/shd_pulsar.yaml

# Run all experiments, then aggregate + plot
./scripts/run_all.sh

# Just aggregate + plot (after experiments complete)
python analysis/aggregate.py --results_dir results --output results/summary.json
python analysis/plot.py --results_dir results --output_dir results/plots
```

## Environment

This was developed and tested on a CPU-only server (2 cores, 4 GB RAM).
Everything is sized to run on such a machine:
- Models are small (50k–500k params).
- Datasets are small (SHD: 8k samples, MNIST: 60k, CIFAR-10: 60k).
- ImageNet and DVS128 are explicitly out of scope (too large for CPU).
- Timestep counts are minimal (T=4 for image SNNs, T=8 for SHD).

## License

MIT.
