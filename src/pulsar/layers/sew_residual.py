"""SEW (Spike-Element-Wise) residual connection.

From Fang et al., "Deep Residual Learning in Spiking Neural Networks" (NeurIPS
2021). The problem with naive residual connections in SNNs: if you add a spike
train x + f(x), the result can have values > 1, which then either saturates
the next layer or requires rescaling.

SEW residual: apply the spike function AFTER the addition, not before.

    out = Spike(x + f(x))     # SEW form
    out = x + f(x)            # naive form (broken)

This is what makes deep SNNs trainable. We implement it as a residual wrapper
that takes a spike-producing sublayer and adds the input spike train via a
spike-element-wise function (we use the same spike layer the rest of the
network uses).

For Pulsar (Gumbel-softmax), the spike function is the PulseLayer itself. For
the SNN baseline, it's the LIF surrogate spike. We pass the spike fn in to
avoid hard-coupling.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class SEWResidual(nn.Module):
    """Spike-Element-Wise residual connection.

    out = spike_fn(x + f(x))

    Args:
        sublayer: f. Takes spikes, returns continuous values (pre-spike).
        spike_fn: the spike function to apply after the addition. This is
            typically the same LIFNeuron / PulseLayer used in the rest of
            the network. Must be the SAME instance (so it shares state).
    """

    def __init__(self, sublayer: nn.Module, spike_fn: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.spike_fn = spike_fn

    def reset_state(self):
        if hasattr(self.spike_fn, "reset_state"):
            self.spike_fn.reset_state()
        if hasattr(self.sublayer, "reset_state"):
            self.sublayer.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a spike train (0/1). sublayer produces continuous pre-spike V.
        V = x + self.sublayer(x)
        return self.spike_fn(V)
