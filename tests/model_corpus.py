#!/usr/bin/env python3
"""Broader local model corpus for Arm direct-fast-lane sweeps.

The goal is not exhaustive fuzzing. These models are intentionally biased toward
real backend risk areas:
  - pooling and residual structures
  - depthwise/grouped conv patterns
  - permute/layernorm layout churn
  - RMSNorm / channel-norm style decompositions
  - decomposed recurrent cells (Elman / GRU / LSTM)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], nn.Module]
    input_shape: tuple[int, ...]
    family: str
    notes: str


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)
        return x * rms * self.weight + self.bias


class ChannelAffineNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, correction=0)
        inv_std = torch.reciprocal(torch.sqrt(var + self.eps))
        return (x - mean) * inv_std * self.weight + self.bias


class ResidualAvgPoolCNN(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.proj = nn.AvgPool2d(kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block(x) + self.proj(x)
        return self.head(x)


class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.dw1 = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
        self.pw1 = nn.Conv2d(8, 16, kernel_size=1)
        self.dw2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 24, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(24, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        x = F.relu(self.pw1(self.dw1(x)))
        x = self.pool(x)
        x = F.relu(self.pw2(self.dw2(x)))
        return self.head(x)


class PermuteLayerNormCNN(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        channels = 12
        self.stem = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.post = nn.Conv2d(channels, channels, kernel_size=1)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, 1e-6)
        x = x.permute(0, 3, 1, 2)
        x = self.post(x)
        return self.head(x)


class RMSNormResidualCNN(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, 12, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.norm1 = RMSNorm2d(12)
        self.norm2 = RMSNorm2d(12)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(12, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.pool(F.relu(x + residual))
        return self.head(x)


class ChannelNormBlockCNN(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.norm = ChannelAffineNorm2d(8)
        self.post = nn.Conv2d(8, 8, kernel_size=1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.relu(self.norm(x))
        x = self.post(x)
        return self.head(x)


class RMSNormBlockCNN(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.norm = RMSNorm2d(8)
        self.post = nn.Conv2d(8, 8, kernel_size=1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.relu(self.norm(x))
        x = self.post(x)
        return self.head(x)


class ReshapeReduceBroadcastCNN(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        channels = 6
        self.stem = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.post = nn.Conv2d(channels, channels, kernel_size=1)
        self.weight = nn.Parameter(torch.ones(channels, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1))
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        n, c, h, w = x.shape
        flat = x.reshape(n, c, h * w)
        mean = flat.mean(dim=2, keepdim=True)
        centered = flat - mean
        var = (centered * centered).mean(dim=2, keepdim=True)
        scaled = centered * torch.rsqrt(var + 1e-6)
        scaled = scaled * self.weight + self.bias
        x = scaled.reshape(n, c, h, w)
        x = self.post(x)
        return self.head(x)


class DecomposedElmanRNN(nn.Module):
    def __init__(self, input_size: int = 16, hidden_size: int = 24, num_classes: int = 5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        for x_t in x.unbind(dim=1):
            h = torch.tanh(self.in_proj(x_t) + self.h_proj(h))
        return self.out_proj(h)


class DecomposedGRU(nn.Module):
    def __init__(self, input_size: int = 16, hidden_size: int = 24, num_classes: int = 5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x_r = nn.Linear(input_size, hidden_size)
        self.h_r = nn.Linear(hidden_size, hidden_size)
        self.x_z = nn.Linear(input_size, hidden_size)
        self.h_z = nn.Linear(hidden_size, hidden_size)
        self.x_n = nn.Linear(input_size, hidden_size)
        self.h_n = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        one = torch.ones_like(h)
        for x_t in x.unbind(dim=1):
            r = torch.sigmoid(self.x_r(x_t) + self.h_r(h))
            z = torch.sigmoid(self.x_z(x_t) + self.h_z(h))
            n = torch.tanh(self.x_n(x_t) + r * self.h_n(h))
            h = (one - z) * n + z * h
        return self.out_proj(h)


class DecomposedLSTM(nn.Module):
    def __init__(self, input_size: int = 16, hidden_size: int = 24, num_classes: int = 5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x_i = nn.Linear(input_size, hidden_size)
        self.h_i = nn.Linear(hidden_size, hidden_size)
        self.x_f = nn.Linear(input_size, hidden_size)
        self.h_f = nn.Linear(hidden_size, hidden_size)
        self.x_g = nn.Linear(input_size, hidden_size)
        self.h_g = nn.Linear(hidden_size, hidden_size)
        self.x_o = nn.Linear(input_size, hidden_size)
        self.h_o = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)
        for x_t in x.unbind(dim=1):
            i = torch.sigmoid(self.x_i(x_t) + self.h_i(h))
            f = torch.sigmoid(self.x_f(x_t) + self.h_f(h))
            g = torch.tanh(self.x_g(x_t) + self.h_g(h))
            o = torch.sigmoid(self.x_o(x_t) + self.h_o(h))
            c = f * c + i * g
            h = o * torch.tanh(c)
        return self.out_proj(h)


EXTRA_MODEL_SPECS: Dict[str, ModelSpec] = {
    "residual_avgpool_cnn": ModelSpec(
        name="residual_avgpool_cnn",
        factory=ResidualAvgPoolCNN,
        input_shape=(1, 1, 28, 28),
        family="cnn",
        notes="Residual block with avg-pool downsampling",
    ),
    "depthwise_separable_cnn": ModelSpec(
        name="depthwise_separable_cnn",
        factory=DepthwiseSeparableCNN,
        input_shape=(1, 1, 28, 28),
        family="cnn",
        notes="Depthwise + pointwise conv stack",
    ),
    "permute_layernorm_cnn": ModelSpec(
        name="permute_layernorm_cnn",
        factory=PermuteLayerNormCNN,
        input_shape=(1, 2, 16, 16),
        family="layout_norm",
        notes="Conv block with NHWC layer_norm round-trip",
    ),
    "rmsnorm_residual_cnn": ModelSpec(
        name="rmsnorm_residual_cnn",
        factory=RMSNormResidualCNN,
        input_shape=(1, 2, 16, 16),
        family="norm",
        notes="Residual CNN using RMSNorm-style decomposition",
    ),
    "channel_norm_block_cnn": ModelSpec(
        name="channel_norm_block_cnn",
        factory=ChannelNormBlockCNN,
        input_shape=(1, 2, 16, 16),
        family="norm",
        notes="Small conv block with channel-wise mean/var norm and affine broadcast",
    ),
    "rmsnorm_block_cnn": ModelSpec(
        name="rmsnorm_block_cnn",
        factory=RMSNormBlockCNN,
        input_shape=(1, 2, 16, 16),
        family="norm",
        notes="Small conv block with RMSNorm-style affine broadcast",
    ),
    "reshape_reduce_broadcast_cnn": ModelSpec(
        name="reshape_reduce_broadcast_cnn",
        factory=ReshapeReduceBroadcastCNN,
        input_shape=(1, 2, 16, 16),
        family="reshape_norm",
        notes="Conv block with reshape -> reduce -> rsqrt -> broadcast affine -> reshape",
    ),
    "elman_rnn_decomposed": ModelSpec(
        name="elman_rnn_decomposed",
        factory=DecomposedElmanRNN,
        input_shape=(1, 8, 16),
        family="recurrent",
        notes="Decomposed Elman cell unrolled over time",
    ),
    "gru_rnn_decomposed": ModelSpec(
        name="gru_rnn_decomposed",
        factory=DecomposedGRU,
        input_shape=(1, 8, 16),
        family="recurrent",
        notes="Decomposed GRU cell with explicit gates",
    ),
    "lstm_rnn_decomposed": ModelSpec(
        name="lstm_rnn_decomposed",
        factory=DecomposedLSTM,
        input_shape=(1, 8, 16),
        family="recurrent",
        notes="Decomposed LSTM cell with explicit gates",
    ),
}


def _base_model_specs() -> Dict[str, ModelSpec]:
    from model import MODELS

    return {
        name: ModelSpec(
            name=name,
            factory=factory,
            input_shape=shape,
            family="baseline",
            notes="Existing local baseline model",
        )
        for name, (factory, shape) in MODELS.items()
    }


def all_model_specs() -> Dict[str, ModelSpec]:
    specs = _base_model_specs()
    specs.update(EXTRA_MODEL_SPECS)
    return specs


def make_example_input(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(*shape)


def get_model_and_example(name: str) -> tuple[nn.Module, torch.Tensor]:
    spec = all_model_specs()[name]
    model = spec.factory().eval()
    return model, make_example_input(spec.input_shape)


def iter_model_specs() -> Iterable[ModelSpec]:
    for name in sorted(all_model_specs()):
        yield all_model_specs()[name]
