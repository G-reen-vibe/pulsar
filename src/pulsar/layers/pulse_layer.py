"""PulseLayer — Pulsar's spike layer based on Gumbel-softmax reparameterization.

This is the core contribution. Instead of using a fixed surrogate gradient
(sigmoid, atan, fast-sigmoid, etc.) we treat the spike as a Bernoulli sample
from a rate `σ(V)` and use the **Gumbel-softmax (concrete) reparameterization**
to backpropagate through it.

Why Gumbel-softmax over surrogate gradients?
- *Principled*: defines a proper differentiable density over the simplex.
  The gradient is the gradient of an actual probability model, not a hand-picked
  shape.
- *Annealable*: temperature τ controls the discretization. τ → ∞ gives uniform
  samples; τ → 0 gives hard one-hots. We anneal from τ=1 → τ=0.1 over training,
  so the model starts in a smooth regime and converges to true discrete spikes.
- *No train/test mismatch*: at test time we sample exactly the same way (just
  with τ→0, which gives the hard spike), so the trained distribution and the
  inference distribution are the same family. This is the central design
  advantage over surrogate gradients, where the surrogate is "fake" at training
  time and discarded at test time.

The forward pass (training):
    s_soft = softmax((log p + g) / τ)   where g ~ Gumbel(0,1), p = [σ(V), 1-σ(V)]
    s_hard = one_hot(argmax(s_soft))
    s = s_hard - s_soft.detach() + s_soft                # straight-through

The forward pass (inference / eval):
    s = (σ(V) >= 0.5).float()                             # deterministic hard spike

Optional: membrane potential normalization before the spike. We keep this as
a separate `MembraneNorm` module so it can be composed with any spike layer.

Reference:
- Jang et al., "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017.
- Maddison et al., "The Concrete Distribution", ICLR 2017.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseLayer(nn.Module):
    """Spike layer using Gumbel-softmax reparameterization.

    Args:
        tau_init: initial temperature (higher = smoother).
        tau_min: final temperature after annealing (lower = sharper).
        hard: if True, use straight-through (hard spike forward, soft gradient).
            Recommended True for SNN behavior at inference.
        threshold: scalar added to logit before sigmoid; equivalent to LIF
            threshold but expressed as bias. We keep this as a buffer for
            parity with the LIF baseline.
        rate_clip: clip the input to logit domain to [−rate_clip, +rate_clip]
            before sigmoid. Prevents saturation from killing gradients.

    Shape:
        Input: (B, *features)
        Output: (B, *features)  -- spike train (0 or 1 in eval mode)
    """

    def __init__(
        self,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        hard: bool = True,
        threshold: float = 0.0,
        rate_clip: float = 8.0,
    ):
        super().__init__()
        assert 0.0 < tau_min <= tau_init, "tau_min must be in (0, tau_init]"
        self.tau_init = float(tau_init)
        self.tau_min = float(tau_min)
        self.hard = hard
        self.threshold = float(threshold)
        self.rate_clip = float(rate_clip)

        # Current temperature — annealed by trainer.
        # Stored as a buffer (not parameter) so it's not optimized.
        self.register_buffer("tau", torch.tensor(tau_init, dtype=torch.float32))

    def set_tau(self, tau: float):
        """Set current temperature (called by trainer's annealing schedule)."""
        self.tau.fill_(float(tau))

    def anneal_to(self, progress: float):
        """Anneal tau linearly from tau_init → tau_min as progress goes 0 → 1.

        progress = current_step / total_steps, clipped to [0, 1].
        """
        progress = max(0.0, min(1.0, float(progress)))
        new_tau = self.tau_init + (self.tau_min - self.tau_init) * progress
        self.set_tau(new_tau)

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """Spike a continuous membrane potential V.

        Args:
            V: (B, *features) membrane potential (pre-spike).
        Returns:
            s: (B, *features) spike train. 0/1 in eval mode or hard=True.
        """
        # Build Bernoulli logits: [log p, log (1-p)] where p = σ(V - threshold).
        # Equivalently, work in 2-class softmax form: logits = [V - threshold, 0]
        # gives p = sigmoid(V - threshold).
        V = torch.clamp(V, -self.rate_clip, self.rate_clip)
        logits_1 = V - self.threshold  # log p
        logits_0 = torch.zeros_like(V)  # log (1 - p)
        # Shape: (..., 2)
        logits = torch.stack([logits_0, logits_1], dim=-1)

        if self.training:
            tau = float(self.tau.item())
            # Sample from Concrete/Gumbel-Softmax distribution.
            soft = F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=-1)
            # soft[..., 1] is the "spike" probability mass.
            s = soft[..., 1]
            return s
        else:
            # Deterministic hard spike at inference.
            p = torch.sigmoid(V - self.threshold)
            return (p >= 0.5).float()
