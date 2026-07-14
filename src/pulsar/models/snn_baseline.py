"""SNN baseline: surrogate-gradient LIF neurons.

This is the standard SNN baseline used in snntorch / spikingjelly. We implement
it ourselves with our own LIFNeuron so the comparison is fair and
self-contained.

Architecture: encoder → [Linear/Conv + LIF] blocks → rate decoder.

The forward pass loops over T timesteps, calling each LIF layer once per step.
State is reset between samples via reset_state().
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import LIFNeuron, MembraneNorm
from ..coding import PoissonRateEncoder, RateDecoder


class SNNMLP(nn.Module):
    """MLP-style SNN with LIF neurons and surrogate gradient training.

    Args:
        in_features: input dimensionality (e.g., 700 for SHD, 784 for MNIST).
        hidden_dims: list of hidden sizes.
        num_classes: output classes.
        T: number of timesteps.
        decay: LIF membrane decay.
        threshold: LIF spike threshold.
        beta: surrogate gradient sharpness.
        encoder: "poisson" or "constant".
        input_is_spike_train: if True, the input is already a spike train of
            shape (B, T, in_features). The encoder is bypassed. Used for SHD.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        num_classes: int,
        T: int = 8,
        decay: float = 0.95,
        threshold: float = 1.0,
        beta: float = 5.0,
        encoder: str = "poisson",
        input_is_spike_train: bool = False,
    ):
        super().__init__()
        self.T = T
        self.in_features = in_features
        self.input_is_spike_train = input_is_spike_train

        if input_is_spike_train:
            self.encoder = None  # input is already spikes
        elif encoder == "poisson":
            self.encoder = PoissonRateEncoder(T=T)
        elif encoder == "constant":
            from ..coding import ConstantEncoder
            self.encoder = ConstantEncoder(T=T)
        else:
            raise ValueError(f"bad encoder: {encoder}")

        # Build MLP layers + LIF neurons
        dims = [in_features] + hidden_dims
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.norms = nn.ModuleList([
            MembraneNorm(d) for d in hidden_dims
        ])
        self.lifs = nn.ModuleList([
            LIFNeuron(decay=decay, threshold=threshold, beta=beta)
            for _ in hidden_dims
        ])
        self.decoder = RateDecoder(hidden_dims[-1], num_classes)

    def reset_state(self):
        for lif in self.lifs:
            lif.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_features) in [0, 1] OR (B, T, in_features) if input_is_spike_train.
        Returns (B, num_classes).
        """
        if self.input_is_spike_train:
            # x is (B, T, F) → permute to (T, B, F)
            spikes = x.permute(1, 0, 2).contiguous()
        else:
            spikes = self.encoder(x)  # (T, B, in_features)
        out_spikes_over_time = []

        self.reset_state()
        for t in range(self.T):
            s = spikes[t]  # (B, in_features)
            for lin, norm, lif in zip(self.linears, self.norms, self.lifs):
                V = lin(s)
                V = norm(V)
                s = lif(V)
            out_spikes_over_time.append(s)  # (B, hidden_dims[-1])

        out = torch.stack(out_spikes_over_time, dim=0)  # (T, B, F)
        return self.decoder(out)


class SNNCNN(nn.Module):
    """CNN-style SNN with LIF neurons.

    Architecture: encoder → 3× (Conv-BN-LIF-MaxPool) → flatten → Linear-LIF → rate decoder.

    Args:
        in_channels: 1 (MNIST) or 3 (CIFAR-10).
        num_classes: output.
        T: timesteps.
        width: base channel width.
        input_size: 28 or 32.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        T: int = 4,
        width: int = 16,
        input_size: int = 32,
        decay: float = 0.95,
        threshold: float = 1.0,
        beta: float = 5.0,
        encoder: str = "poisson",
    ):
        super().__init__()
        self.T = T
        self.input_size = input_size

        if encoder == "poisson":
            self.encoder = PoissonRateEncoder(T=T)
        elif encoder == "constant":
            from ..coding import ConstantEncoder
            self.encoder = ConstantEncoder(T=T)
        else:
            raise ValueError(f"bad encoder: {encoder}")

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.lif1 = LIFNeuron(decay=decay, threshold=threshold, beta=beta)

        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.lif2 = LIFNeuron(decay=decay, threshold=threshold, beta=beta)

        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.lif3 = LIFNeuron(decay=decay, threshold=threshold, beta=beta)

        # After 3 maxpools on input_size: pooled = input_size // 8
        pooled = input_size // 8
        self.fc = nn.Linear(width * 4 * pooled * pooled, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.lif_fc = LIFNeuron(decay=decay, threshold=threshold, beta=beta)

        self.decoder = RateDecoder(128, num_classes)

    def reset_state(self):
        for lif in [self.lif1, self.lif2, self.lif3, self.lif_fc]:
            lif.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) in [0, 1]. Returns (B, num_classes)."""
        spikes = self.encoder(x)  # (T, B, C, H, W)
        out_over_time = []

        self.reset_state()
        for t in range(self.T):
            s = spikes[t]  # (B, C, H, W)
            s = F.max_pool2d(self.lif1(self.bn1(self.conv1(s))), 2)
            s = F.max_pool2d(self.lif2(self.bn2(self.conv2(s))), 2)
            s = F.max_pool2d(self.lif3(self.bn3(self.conv3(s))), 2)
            s = s.flatten(1)
            s = self.lif_fc(self.bn_fc(self.fc(s)))
            out_over_time.append(s)

        out = torch.stack(out_over_time, dim=0)  # (T, B, 128)
        return self.decoder(out)
