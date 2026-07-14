"""Leaky Integrate-and-Fire (LIF) neuron with surrogate gradient.

This is the standard SNN building block used by the surrogate-gradient baseline
(snntorch / spikingjelly style). We implement it explicitly so we have a
self-contained, well-understood baseline that does not depend on external SNN
libraries.

Dynamics (per timestep, for input current `x_t`):

    V_t = decay * V_{t-1} + (1 - decay) * x_t        # membrane potential
    s_t = 1 if V_t >= threshold else 0                # spike (forward)
    V_t = V_t - s_t * threshold                       # subtractive reset

The spike function `s_t = Θ(V_t - threshold)` has zero gradient almost
everywhere. We use a sigmoid surrogate (matches snntorch's `sigmoid(beta=...)`):

    ds/dV = (1/β) * σ(-(V - threshold) / β)

Smaller β → sharper surrogate (closer to true spike). Larger β → smoother
gradient.

Key design choices:
- Subtractive reset by default (V_t -= threshold on spike). This is what
  snntorch uses. Alternative is "reset to zero" which loses some information.
- The trainer is responsible for the time loop. Calling `forward(x_t)` runs
  ONE timestep and returns the spike at that step. State is held inside the
  module; call `reset_state()` between samples.
- `learn_decay=True` reparameterizes decay via sigmoid (TA-style, Fang 2021).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SigmoidSurrogate(torch.autograd.Function):
    """Spike function with sigmoid surrogate gradient.

    Forward: s = (V >= threshold).float()
    Backward: ds/dV = (1/beta) * sigmoid(-(V - threshold) / beta)
    """

    @staticmethod
    def forward(ctx, V: torch.Tensor, threshold: float, beta: float) -> torch.Tensor:
        ctx.save_for_backward(V)
        ctx.threshold = threshold
        ctx.beta = beta
        return (V >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (V,) = ctx.saved_tensors
        threshold = ctx.threshold
        beta = ctx.beta
        sg = (1.0 / beta) * torch.sigmoid(-(V - threshold) / beta)
        return grad_output * sg, None, None


def spike_sigmoid(V: torch.Tensor, threshold: float = 1.0, beta: float = 5.0) -> torch.Tensor:
    """Differentiable spike via sigmoid surrogate (functional form)."""
    return SigmoidSurrogate.apply(V, threshold, beta)


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron — single-timestep interface.

    The trainer loops over time and calls this module once per timestep.

    Args:
        decay: membrane potential decay ∈ (0, 1). 1.0 → no leak (pure
            integrator). 0.0 → instantly forgets. Typical: 0.9-0.99.
        threshold: spike threshold.
        beta: surrogate gradient sharpness. Smaller = sharper.
        reset_mode: "subtract" (V -= threshold on spike) or "zero" (V = 0).
        learn_decay: if True, decay is a learnable parameter via sigmoid
            reparameterization (TA-style, Fang et al. 2021).

    Shape conventions:
        Input  (one timestep):  (B, *features)
        Output (one timestep):  (B, *features)  -- spike train at this step
    """

    def __init__(
        self,
        decay: float = 0.95,
        threshold: float = 1.0,
        beta: float = 5.0,
        reset_mode: str = "subtract",
        learn_decay: bool = False,
    ):
        super().__init__()
        assert 0.0 < decay < 1.0, f"decay must be in (0, 1), got {decay}"
        assert reset_mode in ("subtract", "zero"), f"bad reset_mode: {reset_mode}"

        self.threshold = float(threshold)
        self.beta = float(beta)
        self.reset_mode = reset_mode

        if learn_decay:
            decay_logit = math.log(decay / (1.0 - decay))
            self.decay_logit = nn.Parameter(torch.tensor(decay_logit, dtype=torch.float32))
        else:
            self.register_buffer("decay_const", torch.tensor(decay, dtype=torch.float32))

        self._V: torch.Tensor | None = None

    @property
    def decay(self) -> float:
        if hasattr(self, "decay_logit"):
            return float(torch.sigmoid(self.decay_logit).item())
        return float(self.decay_const.item())

    def reset_state(self):
        """Clear membrane potential. Must be called between samples."""
        self._V = None

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """One timestep of LIF dynamics.

        Args:
            x_t: (B, *features) input current at this timestep.
        Returns:
            s_t: (B, *features) spike train at this timestep (0 or 1).
        """
        decay = self.decay
        if self._V is None or self._V.shape != x_t.shape or self._V.device != x_t.device:
            self._V = torch.zeros_like(x_t)

        # Membrane update: V_t = decay * V_{t-1} + (1 - decay) * x_t
        self._V = decay * self._V + (1.0 - decay) * x_t

        # Spike with surrogate gradient
        s = spike_sigmoid(self._V, threshold=self.threshold, beta=self.beta)

        # Reset
        if self.reset_mode == "subtract":
            self._V = self._V - s * self.threshold
        else:  # "zero"
            self._V = self._V * (1.0 - s)

        return s
