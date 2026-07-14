"""Membrane potential normalization.

SNNs are notoriously unstable to train. The membrane potential `V` can drift
to extreme values (saturation or zero gradient). Conventional LayerNorm
assumes continuous activations; we adapt it for the membrane potential.

This module applies LayerNorm on V (the pre-spike membrane potential). It is
applied BEFORE the spike function, not after, so the spike gets a well-scaled
input regardless of upstream drift.

We also support a learnable per-channel gain (γ) and bias (β), as in standard
LayerNorm.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MembraneNorm(nn.Module):
    """LayerNorm over the channel dimension of a membrane potential tensor.

    Args:
        num_features: size of the channel/feature dimension.
        eps: numerical stability.

    Shape:
        Input: (B, C, *spatial) or (B, C).
        Output: same shape.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # Affine parameters on the channel dim.
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        # Normalize over channel dim (dim=1 for both (B,C,...) and (B,C)).
        # For (B, C) tensors, dims=(1,) normalizes per-sample across channels.
        # For (B, C, H, W), we normalize over (C, H, W) per-sample — this is
        # more like InstanceNorm but matches what works in SNN literature.
        if V.dim() == 2:
            # (B, C) — normalize across C
            mean = V.mean(dim=1, keepdim=True)
            var = V.var(dim=1, keepdim=True, unbiased=False)
            V_norm = (V - mean) / torch.sqrt(var + self.eps)
            return V_norm * self.gamma + self.beta
        elif V.dim() == 4:
            # (B, C, H, W) — normalize across (C, H, W) per-sample, like
            # what works in spiking transformers (membrane-norm per location).
            mean = V.mean(dim=(1, 2, 3), keepdim=True)
            var = V.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
            V_norm = (V - mean) / torch.sqrt(var + self.eps)
            shape = (1, -1, 1, 1)
            return V_norm * self.gamma.view(shape) + self.beta.view(shape)
        else:
            # Fallback: normalize all but batch dim.
            dims = tuple(range(1, V.dim()))
            mean = V.mean(dim=dims, keepdim=True)
            var = V.var(dim=dims, keepdim=True, unbiased=False)
            V_norm = (V - mean) / torch.sqrt(var + self.eps)
            shape = (1, -1) + (1,) * (V.dim() - 2)
            return V_norm * self.gamma.view(shape) + self.beta.view(shape)
