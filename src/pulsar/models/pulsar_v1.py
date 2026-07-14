"""Pulsar v1: Gumbel-softmax spiking neural network.

This is our proposed model. Key differences from the SNN baseline:
- Spike layer is PulseLayer (Gumbel-softmax) instead of LIFNeuron (surrogate
  gradient).
- No membrane dynamics (V_t = decay * V_{t-1} + x_t) by default — the
  PulseLayer is stateless. This is intentional: the time dimension comes
  purely from the encoder, not from membrane leak. We add a learnable
  membrane update as an option (stateful_mode=True) for ablation.
- Attention decoder instead of rate decoder — more expressive, lets the model
  learn WHICH timesteps matter.
- Temperature is annealed by the trainer via `set_tau()` on each PulseLayer.

The forward pass: encoder → [Linear/Conv + MembraneNorm + PulseLayer] blocks
→ attention decoder over the spike train.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import PulseLayer, MembraneNorm
from ..coding import (
    PoissonRateEncoder,
    ConstantEncoder,
    LearnedEncoder,
    AttentionDecoder,
    RateDecoder,
)


class PulsarMLP(nn.Module):
    """Pulsar v1 MLP.

    Args:
        in_features: input dim.
        hidden_dims: list of hidden sizes.
        num_classes: output classes.
        T: timesteps.
        tau_init: initial Gumbel temperature.
        tau_min: final temperature.
        encoder: "poisson", "constant", or "learned".
        decoder: "rate" or "attention".
        stateful: if True, add a learnable membrane leak (alpha) so V_t =
            alpha * V_{t-1} + (1-alpha) * x_t. Default False (stateless).
        input_is_spike_train: if True, input is (B, T, in_features) and
            encoder is bypassed. Used for SHD.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        num_classes: int,
        T: int = 8,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        encoder: str = "poisson",
        decoder: str = "attention",
        stateful: bool = False,
        input_is_spike_train: bool = False,
        hard: bool = True,
        pulse_stateful: bool = False,
        decay_init: float = 0.9,
        learnable_threshold: bool = False,
        recurrent: bool = False,
        norm_type: str = "membrane",
        num_spike_levels: int = 2,
        residual: bool = False,
    ):
        super().__init__()
        self.T = T
        self.in_features = in_features
        self.stateful = stateful
        self.input_is_spike_train = input_is_spike_train
        self.residual = residual

        if input_is_spike_train:
            self.encoder = None
        elif encoder == "poisson":
            self.encoder = PoissonRateEncoder(T=T)
        elif encoder == "constant":
            self.encoder = ConstantEncoder(T=T)
        elif encoder == "learned":
            self.encoder = LearnedEncoder(in_features, in_features, T=T)
        else:
            raise ValueError(f"bad encoder: {encoder}")

        dims = [in_features] + hidden_dims
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        # Round 7: support different norm types
        if norm_type == "membrane":
            self.norms = nn.ModuleList([MembraneNorm(d) for d in hidden_dims])
        elif norm_type == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_dims])
        elif norm_type == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm(d) for d in hidden_dims])
        elif norm_type == "none":
            self.norms = nn.ModuleList([nn.Identity() for d in hidden_dims])
        else:
            raise ValueError(f"bad norm_type: {norm_type}")
        self.pulses = nn.ModuleList([
            PulseLayer(
                tau_init=tau_init, tau_min=tau_min, hard=hard,
                stateful=pulse_stateful, decay_init=decay_init,
                num_features=d, learnable_threshold=learnable_threshold,
                recurrent=recurrent, num_spike_levels=num_spike_levels,
            )
            for d in hidden_dims
        ])

        if stateful:
            # Learnable membrane decay per layer, reparameterized.
            self.alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in hidden_dims
            ])
        # State for stateful mode
        self._V_states = [None] * len(hidden_dims)

        if decoder == "rate":
            self.decoder = RateDecoder(hidden_dims[-1], num_classes)
        elif decoder == "attention":
            self.decoder = AttentionDecoder(hidden_dims[-1], num_classes)
        else:
            raise ValueError(f"bad decoder: {decoder}")

    def set_tau(self, tau: float):
        for p in self.pulses:
            p.set_tau(tau)

    def anneal_to(self, progress: float):
        for p in self.pulses:
            p.anneal_to(progress)

    def reset_state(self):
        self._V_states = [None] * len(self.pulses)
        for p in self.pulses:
            p.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_features) OR (B, T, in_features). Returns (B, num_classes)."""
        if self.input_is_spike_train:
            spikes = x.permute(1, 0, 2).contiguous()  # (T, B, F)
        else:
            spikes = self.encoder(x)
        out_spikes = []

        self.reset_state()
        for t in range(self.T):
            s = spikes[t]
            for i, (lin, norm, pulse) in enumerate(zip(self.linears, self.norms, self.pulses)):
                V = lin(s)
                V = norm(V)
                if self.stateful:
                    alpha = torch.sigmoid(self.alphas[i])
                    if self._V_states[i] is None or self._V_states[i].shape != V.shape:
                        self._V_states[i] = torch.zeros_like(V)
                    self._V_states[i] = alpha * self._V_states[i] + (1 - alpha) * V
                    V = self._V_states[i]
                # Round 13: residual connection (add input spikes to V before spike)
                if self.residual and s.shape == V.shape:
                    V = V + s
                s = pulse(V)
            out_spikes.append(s)

        out = torch.stack(out_spikes, dim=0)  # (T, B, F)
        return self.decoder(out)


class PulsarCNN(nn.Module):
    """Pulsar v1 CNN for image classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        T: int = 4,
        width: int = 16,
        input_size: int = 32,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        encoder: str = "poisson",
        decoder: str = "attention",
        stateful: bool = False,
    ):
        super().__init__()
        self.T = T
        self.input_size = input_size
        self.stateful = stateful

        if encoder == "poisson":
            self.encoder = PoissonRateEncoder(T=T)
        elif encoder == "constant":
            self.encoder = ConstantEncoder(T=T)
        else:
            raise ValueError(f"bad encoder for CNN: {encoder}")

        # 3 conv blocks
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.norm1 = MembraneNorm(width)
        self.pulse1 = PulseLayer(tau_init=tau_init, tau_min=tau_min)

        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1)
        self.norm2 = MembraneNorm(width * 2)
        self.pulse2 = PulseLayer(tau_init=tau_init, tau_min=tau_min)

        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1)
        self.norm3 = MembraneNorm(width * 4)
        self.pulse3 = PulseLayer(tau_init=tau_init, tau_min=tau_min)

        pooled = input_size // 8
        self.fc = nn.Linear(width * 4 * pooled * pooled, 128)
        self.norm_fc = MembraneNorm(128)
        self.pulse_fc = PulseLayer(tau_init=tau_init, tau_min=tau_min, hard=False)

        if stateful:
            self.alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(4)
            ])
        self._V_states = [None] * 4

        if decoder == "rate":
            self.decoder = RateDecoder(128, num_classes)
        elif decoder == "attention":
            self.decoder = AttentionDecoder(128, num_classes)
        else:
            raise ValueError(f"bad decoder: {decoder}")

    def set_tau(self, tau: float):
        for p in [self.pulse1, self.pulse2, self.pulse3, self.pulse_fc]:
            p.set_tau(tau)

    def anneal_to(self, progress: float):
        for p in [self.pulse1, self.pulse2, self.pulse3, self.pulse_fc]:
            p.anneal_to(progress)

    def reset_state(self):
        self._V_states = [None] * 4

    def _step_state(self, i, V):
        if not self.stateful:
            return V
        alpha = torch.sigmoid(self.alphas[i])
        if self._V_states[i] is None or self._V_states[i].shape != V.shape:
            self._V_states[i] = torch.zeros_like(V)
        self._V_states[i] = alpha * self._V_states[i] + (1 - alpha) * V
        return self._V_states[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W). Returns (B, num_classes)."""
        spikes = self.encoder(x)  # (T, B, C, H, W)
        out_over_time = []

        self.reset_state()
        for t in range(self.T):
            s = spikes[t]
            V = self.conv1(s)
            V = self.norm1(V)
            V = self._step_state(0, V)
            s = self.pulse1(V)
            s = F.max_pool2d(s, 2)

            V = self.conv2(s)
            V = self.norm2(V)
            V = self._step_state(1, V)
            s = self.pulse2(V)
            s = F.max_pool2d(s, 2)

            V = self.conv3(s)
            V = self.norm3(V)
            V = self._step_state(2, V)
            s = self.pulse3(V)
            s = F.max_pool2d(s, 2)
            s = s.flatten(1)

            V = self.fc(s)
            V = self.norm_fc(V)
            V = self._step_state(3, V)
            s = self.pulse_fc(V)
            out_over_time.append(s)

        out = torch.stack(out_over_time, dim=0)  # (T, B, 128)
        return self.decoder(out)
