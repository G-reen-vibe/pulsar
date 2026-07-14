"""Decoders: aggregate spike trains over time into final logits."""
from __future__ import annotations

import torch
import torch.nn as nn


class RateDecoder(nn.Module):
    """Rate decoder: sum spikes over time, then linear → logits.

    out = W @ (sum_t s_t) + b

    This is the standard readout for rate-coded SNNs. Equivalent to a linear
    layer on the time-averaged spike rate.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (T, B, F). Returns (B, num_classes)."""
        return self.fc(spikes.sum(dim=0))


class LastStepDecoder(nn.Module):
    """Readout from the membrane potential at the last timestep.

    Useful for TTFS-style models where the final state is the most informative.
    For our LIF-based models, we track the membrane potential of the output
    layer and read it at t=T-1.

    Here we provide a generic version: linear on the last timestep's spikes.
    The richer version (read membrane potential) requires the model to expose
    internal state — handled at the model level.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (T, B, F). Returns (B, num_classes)."""
        return self.fc(spikes[-1])


class AttentionDecoder(nn.Module):
    """Attention-based decoder.

    Learnable query attends over the (T, B, F) spike train. This is the
    most expressive decoder and the one we use for Pulsar v1.

    Implementation: single-head attention with one query token.
        Q = learnable parameter (1, 1, F)
        K, V = linear projections of spikes (T, B, F)
        out = softmax(QK^T / sqrt(F)) @ V → (B, F)
        logits = linear(out)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, in_features) * 0.02)
        self.k_proj = nn.Linear(in_features, in_features)
        self.v_proj = nn.Linear(in_features, in_features)
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        assert self.head_dim * num_heads == in_features, "in_features must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (T, B, F). Returns (B, num_classes)."""
        T, B, F = spikes.shape
        # Treat (T, B) as a sequence of T tokens for each of B sequences.
        # Move to (B, T, F) for attention.
        x = spikes.permute(1, 0, 2)  # (B, T, F)
        K = self.k_proj(x)  # (B, T, F)
        V = self.v_proj(x)  # (B, T, F)
        # Expand query to (B, 1, F)
        q = self.query.expand(B, 1, F)

        # Multi-head reshape
        def split_heads(t):
            # (B, L, F) → (B, num_heads, L, head_dim)
            return t.view(t.shape[0], t.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        qh = split_heads(q)  # (B, H, 1, D)
        kh = split_heads(K)  # (B, H, T, D)
        vh = split_heads(V)  # (B, H, T, D)

        attn = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale  # (B, H, 1, T)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vh)  # (B, H, 1, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, 1, F)  # (B, 1, F)
        out = out.squeeze(1)  # (B, F)
        return self.fc(out)
