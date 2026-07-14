# Pulsar

A from-scratch rethink of spiking neural networks (SNNs) for modern ML.

**Goal:** Make SNNs competitive with SOTA ANNs on standard tasks (vision, classification, regression) while keeping the discrete-event / sparse-compute character that makes them interesting.

**Status:** Pre-implementation. See [RESEARCH.md](./RESEARCH.md) for the full design space analysis, candidate directions, and chosen thesis.

## Quick thesis

Treat spikes as **a discrete observation of a continuous latent state**, and train the latent state end-to-end with **Gumbel-softmax** (annealed temperature), while keeping inference event-driven and sparse. This bridges the train/test gap that has plagued SNNs since the beginning.

## Repo layout (planned)

```
pulsar/
├── RESEARCH.md          # design space + chosen direction (this is the entry point)
├── src/pulsar/          # core library
│   ├── layers/          # PulseLayer, membrane norm, SEW residual
│   ├── coding/          # learned encoder / decoder
│   ├── models/          # Pulsar-v1 architecture
│   └── train/           # training loop, Gumbel annealing
├── experiments/         # one-off experiments (moonshots)
├── benchmarks/          # SHD, DVS128, CIFAR-10, ImageNet
└── tests/               # unit tests
```

## License

MIT (to be confirmed).
