"""Encoders: convert real-valued inputs into spike trains over T timesteps.

We provide several encoders, all sharing the interface:

    encoder(x) -> spikes  of shape (T, B, *features)

The time-first convention is used so the trainer can iterate `for t in range(T)`
and feed one timestep at a time to stateful neurons.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PoissonRateEncoder(nn.Module):
    """Poisson rate coding. Each pixel is treated as a spike probability.

    For T timesteps, sample spikes ~ Bernoulli(p) where p = x (clipped to [0,1]).

    This is the most common encoding for image → SNN. Stochastic at train,
    stochastic at eval (by default). Set `deterministic_eval=True` to use
    threshold-based spikes at eval (each timestep's spike = (x > 0.5)).

    Args:
        T: number of timesteps.
        deterministic_eval: if True, use threshold instead of sampling at eval.
    """

    def __init__(self, T: int = 8, deterministic_eval: bool = False):
        super().__init__()
        self.T = T
        self.deterministic_eval = deterministic_eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, *features) in [0, 1]. Returns (T, B, *features) in {0, 1}."""
        x = x.clamp(0.0, 1.0)
        # Replicate across time.
        x_rep = x.unsqueeze(0).expand(self.T, *x.shape)
        if self.training or not self.deterministic_eval:
            return torch.bernoulli(x_rep)
        return (x_rep > 0.5).float()


class LatencyEncoder(nn.Module):
    """Latency (time-to-first-spike) encoding.

    Stronger input → earlier spike. Each pixel fires exactly once, at
    timestep t = (1 - x) * T  (clipped). We use a normalized latency:
    - x = 1.0  → spike at t = 0
    - x = 0.0  → spike at t = T-1
    - x = 0.5  → spike at t = T/2

    The output is a (T, B, *features) tensor with at most one spike per pixel.
    """

    def __init__(self, T: int = 8):
        super().__init__()
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, *features) in [0, 1]. Returns (T, B, *features) in {0, 1}."""
        x = x.clamp(0.0, 1.0)
        # spike_time = round((1 - x) * (T - 1))
        spike_time = ((1.0 - x) * (self.T - 1)).round().long()
        spike_time = spike_time.clamp(0, self.T - 1)

        # Build one-hot-in-time spike train.
        T, = (self.T,)
        out = torch.zeros(T, *x.shape, device=x.device, dtype=x.dtype)
        # out[t, b, ...] = 1 where spike_time[b, ...] == t.
        # Vectorize: t_indices = spike_time; we want to scatter.
        # Reshape spike_time to flat, then index.
        flat_idx = spike_time.view(-1)
        # out shape (T, N) where N = prod(x.shape).
        flat_out = out.view(T, -1)
        # For each n, set flat_out[flat_idx[n], n] = 1.
        flat_out[flat_idx, torch.arange(flat_idx.numel(), device=x.device)] = 1.0
        return out.view(T, *x.shape)


class ConstantEncoder(nn.Module):
    """Constant (a.k.a. direct) encoding.

    Each pixel is presented as a constant current at every timestep. This is
    NOT spiking — it makes the SNN effectively a continuous-valued RNN. Useful
    as an ablation to isolate the contribution of spike discretization from
    the contribution of membrane dynamics.

    Output is (T, B, *features) with the same value at each timestep.
    """

    def __init__(self, T: int = 8):
        super().__init__()
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        return x.unsqueeze(0).expand(self.T, *x.shape).contiguous()


class LearnedEncoder(nn.Module):
    """Learnable spike encoder.

    A small (Linear or Conv) network maps the input to a real-valued "drive"
    tensor, which is then either passed through a PulseLayer (for Pulsar) or
    a Bernoulli sample (for SNN baseline). The trainer treats the encoder's
    output as the input current `x_t` for each timestep — the encoder can
    produce a different drive per timestep if desired.

    For simplicity we use a *static* drive (same for all timesteps) by default.
    The `temporal=True` mode adds a learned sinusoidal time modulation so the
    drive varies with t. This is a minimal learned temporal encoder.

    Args:
        in_features: flat input size (for MLP).
        out_features: output drive size.
        T: timesteps.
        temporal: if True, add learnable per-timestep modulation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        T: int = 8,
        temporal: bool = False,
    ):
        super().__init__()
        self.T = T
        self.temporal = temporal
        self.proj = nn.Linear(in_features, out_features)
        if temporal:
            # Per-timestep learnable scale and bias.
            self.t_scale = nn.Parameter(torch.ones(T, out_features))
            self.t_bias = nn.Parameter(torch.zeros(T, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_features). Returns (T, B, out_features) — drive currents."""
        drive = self.proj(x)  # (B, out_features)
        drive = drive.unsqueeze(0).expand(self.T, *drive.shape)  # (T, B, F)
        if self.temporal:
            drive = drive * self.t_scale.unsqueeze(1) + self.t_bias.unsqueeze(1)
        return drive.contiguous()
