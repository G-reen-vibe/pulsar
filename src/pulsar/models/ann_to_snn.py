"""ANN-to-SNN conversion baseline.

Train an ANN, then convert ReLU activations to rate-coded spiking neurons.
At inference, the SNN approximates the ANN by averaging spike counts over T
timesteps.

Conversion rule (Diehl et al. 2015, standard):
- Replace each ReLU with a LIF neuron whose threshold = the max activation
  of that layer seen during a calibration pass.
- Weights of layer i are scaled by threshold_{i+1} / threshold_i so the
  firing rate matches the original ReLU output.
- At inference, run T timesteps and decode by summing (rate code).

This is the practical SNN baseline used in industry. We implement it as a
wrapper around an already-trained ANN: the user trains the ANN, then calls
`convert()` to swap ReLUs for LIFs, then runs `forward()` over T timesteps.

In our framework, this baseline:
1. Trains an ANN normally (handled by the trainer, model = ANN).
2. At evaluation, instantiates an ANNtoSNN wrapper around the trained ANN.
3. Runs T-timestep inference.

So this module is used by the trainer in a special evaluation mode. The
`ANNtoSNNMLP` / `ANNtoSNNCNN` classes here are full self-contained versions
that include an internal ANN, train it, and convert.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import LIFNeuron


class _ReplacedReLU(nn.Module):
    """A ReLU during training, an LIF during converted inference.

    State:
        mode: "ann" or "snn"
        threshold: max activation seen during calibration (set during convert)
        lif: LIF neuron used in SNN mode
    """

    def __init__(self, decay: float = 0.95, threshold: float = 1.0, beta: float = 5.0):
        super().__init__()
        self.mode = "ann"
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32), requires_grad=False)
        self.lif = LIFNeuron(decay=decay, threshold=threshold, beta=beta)

    def reset_state(self):
        self.lif.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "ann":
            return F.relu(x)
        else:
            # In SNN mode, x is pre-thresholding. Scale so threshold maps to 1.
            # Spike when x >= threshold (== LIF with V=x, threshold set).
            # We use threshold as the LIF threshold directly. To avoid
            # modifying weights, we scale x down by threshold first.
            x_scaled = x / (self.threshold + 1e-8)
            # The LIF's threshold is 1.0; spike happens when V >= 1.
            return self.lif(x_scaled)


class ANNtoSNNMLP(nn.Module):
    """ANN-trained, SNN-converted MLP.

    Training: behaves exactly like a normal MLP (ReLU, real-valued).
    Call `convert()` after training to switch to SNN mode.
    In SNN mode, run forward() T times with the same input (rate-encoded) and
    sum the outputs.

    Args:
        in_features, hidden_dims, num_classes: same as MLPBaseline.
        T: timesteps for SNN inference.
        decay, beta: LIF params.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        num_classes: int,
        T: int = 16,
        decay: float = 0.95,
        threshold: float = 1.0,
        beta: float = 5.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.T = T
        self.mode = "ann"  # "ann" or "snn"

        dims = [in_features] + hidden_dims
        layers = []
        self.relus = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            r = _ReplacedReLU(decay=decay, threshold=threshold, beta=beta)
            layers.append(r)
            self.relus.append(r)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def reset_state(self):
        for r in self.relus:
            r.reset_state()

    def convert(self, sample_inputs: torch.Tensor):
        """Switch to SNN mode. Calibrate thresholds on a batch of inputs.

        Args:
            sample_inputs: a batch of typical inputs (B, *). Used to set
                per-layer thresholds = max activation seen.
        """
        self.eval()
        sample_inputs = sample_inputs.flatten(1)  # match forward's flattening
        # First, in ANN mode, capture max activations per layer.
        max_acts = [0.0] * len(self.relus)
        hooks = []

        def make_hook(idx):
            def hook(module, input, output):
                with torch.no_grad():
                    m = input[0].abs().max().item()
                    if m > max_acts[idx]:
                        max_acts[idx] = m
            return hook

        for i, r in enumerate(self.relus):
            hooks.append(r.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            _ = self.features(sample_inputs)

        for h in hooks:
            h.remove()

        # Set thresholds and switch to SNN mode.
        for i, r in enumerate(self.relus):
            t = max(max_acts[i], 1e-6)
            r.threshold.fill_(t)
            r.lif.threshold = t
            r.mode = "snn"
        self.mode = "snn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)  # handle (B, T, F) etc.
        if self.mode == "ann":
            return self.head(self.features(x))
        else:
            # SNN mode: loop T times, accumulate output, average.
            self.reset_state()
            outs = []
            for _ in range(self.T):
                # In SNN mode, input is presented as constant current.
                # (Rate coding with p=x; we approximate by presenting x every step.)
                outs.append(self.head(self.features(x)))
            # Average predictions over time = rate code readout.
            return torch.stack(outs, dim=0).mean(dim=0)


class ANNtoSNNCNN(nn.Module):
    """ANN-trained, SNN-converted CNN. Analogous to ANNtoSNNMLP."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 16,
        input_size: int = 32,
        T: int = 16,
        decay: float = 0.95,
        threshold: float = 1.0,
        beta: float = 5.0,
    ):
        super().__init__()
        self.T = T
        self.mode = "ann"

        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.r1 = _ReplacedReLU(decay=decay, threshold=threshold, beta=beta)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.r2 = _ReplacedReLU(decay=decay, threshold=threshold, beta=beta)
        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.r3 = _ReplacedReLU(decay=decay, threshold=threshold, beta=beta)

        pooled = input_size // 8
        self.fc1 = nn.Linear(width * 4 * pooled * pooled, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.r_fc1 = _ReplacedReLU(decay=decay, threshold=threshold, beta=beta)
        self.fc2 = nn.Linear(128, num_classes)

        self.relus = [self.r1, self.r2, self.r3, self.r_fc1]

    def reset_state(self):
        for r in self.relus:
            r.reset_state()

    def _features(self, x):
        x = F.max_pool2d(self.r1(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(self.r2(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(self.r3(self.bn3(self.conv3(x))), 2)
        x = x.flatten(1)
        x = self.r_fc1(self.bn_fc1(self.fc1(x)))
        return x

    def convert(self, sample_inputs: torch.Tensor):
        self.eval()
        max_acts = [0.0] * len(self.relus)
        hooks = []

        def make_hook(idx):
            def hook(module, input, output):
                with torch.no_grad():
                    m = input[0].abs().max().item()
                    if m > max_acts[idx]:
                        max_acts[idx] = m
            return hook

        for i, r in enumerate(self.relus):
            hooks.append(r.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            _ = self._features(sample_inputs)

        for h in hooks:
            h.remove()

        for i, r in enumerate(self.relus):
            t = max(max_acts[i], 1e-6)
            r.threshold.fill_(t)
            r.lif.threshold = t
            r.mode = "snn"
        self.mode = "snn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "ann":
            return self.fc2(self._features(x))
        else:
            self.reset_state()
            outs = []
            for _ in range(self.T):
                outs.append(self.fc2(self._features(x)))
            return torch.stack(outs, dim=0).mean(dim=0)
