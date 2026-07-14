"""Binary activation for the BNN baseline.

Straight-through estimator (STE) binarization. From Hubara et al., "Binarized
Neural Networks" (NeurIPS 2016) and Rastegari et al., "XNOR-Net" (ECCV 2016).

Forward:  s = sign(V)             # in {−1, +1}
Backward: ds/dV = 1 if |V| ≤ 1 else 0   # clipped identity

For SNN comparison we want spikes in {0, 1} not {−1, +1}. We provide both modes:
- `mode='zero_one'`: s = (V > 0).float()     # BNN as binary spike
- `mode='sign'`:     s = 2*(V > 0) - 1        # standard BNN

BNNs are the closest non-SNN discrete-activation baseline. Comparing against
BNNs lets us isolate what the *time dimension* buys us — if a single-timestep
BNN matches a T-step SNN, the time dimension isn't earning its keep.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BinaryActivation(nn.Module):
    """STE-based binary activation.

    Args:
        mode: "zero_one" or "sign".
        clip_grad: clip straight-through gradient to [-1, 1] (default True).
    """

    def __init__(self, mode: str = "zero_one", clip_grad: bool = True):
        super().__init__()
        assert mode in ("zero_one", "sign"), f"bad mode: {mode}"
        self.mode = mode
        self.clip_grad = clip_grad

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        if self.training:
            # STE: hard binarize forward, identity backward.
            if self.mode == "zero_one":
                s_hard = (V > 0).float()
            else:
                s_hard = (2.0 * (V > 0).float()) - 1.0

            if self.clip_grad:
                # clip_grad identity: 1 if |V| ≤ 1 else 0
                grad_mask = (V.abs() <= 1.0).float()
            else:
                grad_mask = torch.ones_like(V)

            # STE trick: s = s_hard + V - V.detach() so backward passes
            # grad_mask through (we multiply by mask via a small custom op).
            # Simpler form: use V - V.detach() for plain identity, then mask.
            ste_grad = V - V.detach()
            return s_hard + grad_mask * ste_grad
        else:
            # Deterministic at eval.
            if self.mode == "zero_one":
                return (V > 0).float()
            return (2.0 * (V > 0).float()) - 1.0
