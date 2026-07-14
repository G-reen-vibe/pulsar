"""BNN baselines: binary-activation neural networks.

These are the closest non-SNN discrete-activation baseline. BNNs binarize
activations to {0, 1} (or {−1, +1}) but have NO time dimension. Comparing
BNNs vs SNNs vs Pulsar tells us how much the time dimension earns.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import BinaryActivation


class BNNMLP(nn.Module):
    """MLP with binary activations (no time dimension).

    Input is flattened beyond batch dim, so (B, F), (B, T, C), or (B, C, H, W)
    all become (B, prod(*)).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        num_classes: int,
        mode: str = "zero_one",
    ):
        super().__init__()
        dims = [in_features] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(BinaryActivation(mode=mode))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.head(self.features(x))


class BNNCNN(nn.Module):
    """CNN with binary activations (no time dimension)."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 16,
        input_size: int = 32,
        mode: str = "zero_one",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.bin1 = BinaryActivation(mode=mode)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.bin2 = BinaryActivation(mode=mode)
        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.bin3 = BinaryActivation(mode=mode)

        pooled_size = input_size // 8
        self.fc1 = nn.Linear(width * 4 * pooled_size * pooled_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.bin1(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(self.bin2(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(self.bin3(self.bn3(self.conv3(x))), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))  # last hidden layer kept real
        return self.fc2(x)
