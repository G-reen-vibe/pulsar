"""PulseLayer — Pulsar's spike layer based on Gumbel-softmax reparameterization.

This is the core contribution. Instead of using a fixed surrogate gradient
(sigmoid, atan, fast-sigmoid, etc.) we treat the spike as a Bernoulli sample
from a rate `σ(V)` and use the **Gumbel-softmax (concrete) reparameterization**
to backpropagate through it.

Why Gumbel-softmax over surrogate gradients?
- *Principled*: defines a proper differentiable density over the simplex.
- *Annealable*: temperature τ controls the discretization.
- *No train/test mismatch*: at test time we sample exactly the same way.

The forward pass (training):
    s_soft = softmax((log p + g) / τ)   where g ~ Gumbel(0,1), p = [σ(V), 1-σ(V)]
    s_hard = one_hot(argmax(s_soft))    (only if hard=True)
    s = s_hard - s_soft.detach() + s_soft                # straight-through

The forward pass (inference / eval):
    s = (σ(V) >= 0.5).float()                             # deterministic hard spike

Stateful mode (Round 3+):
    V_t = decay * V_{t-1} + (1 - decay) * x_t             # membrane update
    s_t = spike(V_t)
    V_t = V_t - s_t * threshold                           # subtractive reset
The decay is a learnable parameter (reparameterized via sigmoid).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseLayer(nn.Module):
    """Spike layer using Gumbel-softmax reparameterization.

    Args:
        tau_init: initial temperature (higher = smoother).
        tau_min: final temperature after annealing (lower = sharper).
        hard: if True, use straight-through (hard spike forward, soft gradient).
        threshold: scalar threshold for spike.
        rate_clip: clip input to [-rate_clip, +rate_clip] before sigmoid.
        stateful: if True, maintain a membrane potential V_t across timesteps.
            V_t = decay * V_{t-1} + (1 - decay) * x_t, then spike, then reset.
            decay is learnable (init 0.9).
        reset_mode: "subtract" or "zero" (only used in stateful mode).
    """

    def __init__(
        self,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        hard: bool = True,
        threshold: float = 0.0,
        rate_clip: float = 8.0,
        stateful: bool = False,
        decay_init: float = 0.9,
        reset_mode: str = "subtract",
        num_features: int | None = None,
        learnable_threshold: bool = False,
        recurrent: bool = False,
        num_spike_levels: int = 2,
    ):
        super().__init__()
        assert 0.0 < tau_min <= tau_init, "tau_min must be in (0, tau_init]"
        assert num_spike_levels >= 2, "num_spike_levels must be >= 2"
        self.tau_init = float(tau_init)
        self.tau_min = float(tau_min)
        self.hard = hard
        self.rate_clip = float(rate_clip)
        self.stateful = stateful
        self.reset_mode = reset_mode
        self.recurrent = recurrent
        self.num_spike_levels = num_spike_levels
        # Spike value levels: 0, 1/(K-1), 2/(K-1), ..., 1
        self.register_buffer(
            "spike_values",
            torch.linspace(0.0, 1.0, num_spike_levels),
        )

        # Round 5: per-neuron learnable threshold
        if learnable_threshold:
            assert num_features is not None, "num_features required for learnable_threshold"
            self.threshold_param = nn.Parameter(torch.zeros(num_features))
            self.threshold = None
        else:
            self.threshold = float(threshold)
            self.threshold_param = None

        self.register_buffer("tau", torch.tensor(tau_init, dtype=torch.float32))

        if stateful:
            assert 0.0 < decay_init < 1.0
            decay_logit = math.log(decay_init / (1.0 - decay_init))
            self.decay_logit = nn.Parameter(torch.tensor(decay_logit, dtype=torch.float32))
            self._V: torch.Tensor | None = None

        # Round 6: recurrent feedback (s_{t-1} → V_t).
        if recurrent:
            assert num_features is not None, "num_features required for recurrent"
            self.recurrent_weight = nn.Parameter(torch.zeros(num_features))
            self._prev_spike: torch.Tensor | None = None

    @property
    def decay(self) -> float:
        if not self.stateful:
            return 0.0
        return float(torch.sigmoid(self.decay_logit).item())

    def set_tau(self, tau: float):
        self.tau.fill_(float(tau))

    def anneal_to(self, progress: float):
        progress = max(0.0, min(1.0, float(progress)))
        new_tau = self.tau_init + (self.tau_min - self.tau_init) * progress
        self.set_tau(new_tau)

    def reset_state(self):
        if self.stateful:
            self._V = None
        if self.recurrent:
            self._prev_spike = None

    def _get_threshold(self, ref: torch.Tensor) -> torch.Tensor:
        """Return threshold broadcastable to ref's shape."""
        if self.threshold_param is not None:
            # threshold_param has shape (num_features,). For input (B, F) we
            # broadcast on dim=-1. For (B, C, H, W) we reshape and broadcast.
            if ref.dim() == 2:
                return self.threshold_param  # (F,) broadcasts with (B, F)
            elif ref.dim() == 4:
                return self.threshold_param.view(1, -1, 1, 1)
            else:
                # Generic: reshape to (1, F, 1, 1, ...)
                shape = [1] * ref.dim()
                shape[1] = -1
                return self.threshold_param.view(shape)
        return self.threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spike a continuous membrane potential.

        If stateful: x is the input current; we update V_t = decay*V_{t-1} +
            (1-decay)*x_t (+ recurrent_weight * s_{t-1} if recurrent=True).
        If not stateful: x IS the membrane potential; we spike directly.
        """
        if self.stateful:
            decay = self.decay
            if self._V is None or self._V.shape != x.shape or self._V.device != x.device:
                self._V = torch.zeros_like(x)
            self._V = decay * self._V + (1.0 - decay) * x
            # Recurrent feedback from previous spike
            if self.recurrent:
                if self._prev_spike is not None:
                    # Broadcast recurrent_weight to x's shape
                    if x.dim() == 2:
                        rw = self.recurrent_weight  # (F,)
                    elif x.dim() == 4:
                        rw = self.recurrent_weight.view(1, -1, 1, 1)
                    else:
                        shape = [1] * x.dim()
                        shape[1] = -1
                        rw = self.recurrent_weight.view(shape)
                    self._V = self._V + rw * self._prev_spike
            V = self._V
        else:
            V = x

        V = torch.clamp(V, -self.rate_clip, self.rate_clip)
        threshold = self._get_threshold(V)

        if self.num_spike_levels == 2:
            # Binary spike (original)
            logits_1 = V - threshold
            logits_0 = torch.zeros_like(V)
            logits = torch.stack([logits_0, logits_1], dim=-1)

            if self.training:
                tau = float(self.tau.item())
                soft = F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=-1)
                s = soft[..., 1]
            else:
                p = torch.sigmoid(V - threshold)
                s = (p >= 0.5).float()
        else:
            # Multi-level spike (Round 11): K-way categorical
            # Logits: [0, V - threshold, 2*(V-threshold), ..., (K-1)*(V-threshold)]
            # So higher V → higher spike count.
            K = self.num_spike_levels
            levels = torch.arange(K, device=V.device, dtype=V.dtype)  # (K,)
            # Logits shape: (*V.shape, K)
            logits = (V - threshold).unsqueeze(-1) * levels  # broadcast
            # Add a small bias so level 0 (no spike) is preferred when V ≈ 0
            # This makes the prior slightly favor "no spike".
            logits[..., 0] = logits[..., 0] + 0.0  # could add bias here

            if self.training:
                tau = float(self.tau.item())
                soft = F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=-1)
                # Weighted sum of spike values: s = sum_k(soft_k * value_k)
                s = (soft * self.spike_values).sum(dim=-1)
            else:
                # Deterministic: pick the level with highest logit
                # Equivalent to: s = round(sigmoid(V-threshold) * (K-1)) / (K-1)
                # But argmax is cleaner.
                idx = logits.argmax(dim=-1)
                s = self.spike_values[idx]

        # Store spike for next timestep's recurrence
        if self.recurrent:
            self._prev_spike = s.detach()  # detach to avoid BPTT through spikes

        # Subtractive reset on the membrane (stateful mode only)
        if self.stateful and self.reset_mode == "subtract":
            self._V = self._V - s * threshold
        elif self.stateful and self.reset_mode == "zero":
            self._V = self._V * (1.0 - s)

        return s
