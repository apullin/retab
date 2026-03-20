from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.registry import ModelSpec, register_model


class SiluBottleneckCNN(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        channels = 10
        self.stem = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.silu(x)
        x = self.mix(x)
        x = F.relu(x)
        return self.head(x)


class GeluResidualCNN(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        channels = 12
        self.stem = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        residual = x
        x = self.block(x)
        x = F.gelu(x + residual)
        return self.tail(x)


class UnaryChainCNN(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        channels = 8
        self.stem = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.post = nn.Conv2d(channels, channels, kernel_size=1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = torch.exp(torch.clamp(x, min=-3.0, max=3.0))
        x = torch.log(x + 1e-3)
        x = torch.tanh(x)
        x = torch.erf(x)
        x = self.post(x)
        return self.head(x)


register_model(
    ModelSpec(
        name="silu_bottleneck_cnn",
        factory=SiluBottleneckCNN,
        input_shape=(1, 2, 16, 16),
        family="unary_math",
        notes="Conv block with silu activation in the direct fast lane",
    )
)

register_model(
    ModelSpec(
        name="gelu_residual_cnn",
        factory=GeluResidualCNN,
        input_shape=(1, 2, 16, 16),
        family="unary_math",
        notes="Residual CNN with gelu over an NHWC-preserved add",
    )
)

register_model(
    ModelSpec(
        name="unary_chain_cnn",
        factory=UnaryChainCNN,
        input_shape=(1, 1, 28, 28),
        family="unary_math",
        notes="Conv block with exp/log/tanh/erf chain",
    )
)
