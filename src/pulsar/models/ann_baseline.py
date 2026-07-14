"""ANN baselines: vanilla MLP and CNN with ReLU.

These are the upper-bound baselines. If Pulsar or any SNN approaches ANN
accuracy, that's the win condition. The MLP is for SHD/MNIST-flat; the CNN is
for CIFAR-10/MNIST-image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """Vanilla MLP for flat-input tasks (SHD, MNIST-flat, MNIST-image-flattened).

    Input is flattened beyond the batch dimension. So (B, F), (B, T, C), or
    (B, C, H, W) all become (B, prod(*)) before being fed to the MLP.

    Args:
        in_features: input dimensionality (flattened). e.g., 700 for SHD if
            T=1, or 8*700=5600 for SHD with T=8. 784 for MNIST-flat.
        hidden_dims: list of hidden layer sizes.
        num_classes: output classes.
        dropout: dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        dims = [in_features] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, *anything). Returns (B, num_classes)."""
        x = x.flatten(1)
        return self.head(self.features(x))


class CNNBaseline(nn.Module):
    """Small CNN for image classification (MNIST, CIFAR-10).

    Architecture: 2 conv blocks + 2 FC layers. ~150k params on CIFAR-10.
    This is intentionally small — the goal is a *fair comparison* baseline,
    not SOTA on CIFAR-10 (which would need ResNet-50 scale).

    Args:
        in_channels: input channels (1 for MNIST, 3 for CIFAR-10).
        num_classes: output classes.
        width: base channel multiplier (default 16).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 16,
        input_size: int = 32,
    ):
        super().__init__()
        # Two conv blocks: each is Conv-BN-ReLU-MaxPool
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(width * 4)

        # After 3 maxpools on input_size: size // 8
        pooled_size = input_size // 8
        self.fc1 = nn.Linear(width * 4 * pooled_size * pooled_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W). Returns (B, num_classes)."""
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
