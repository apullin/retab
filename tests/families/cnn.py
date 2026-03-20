from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.registry import ModelSpec, register_model


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


register_model(
    ModelSpec(
        name="residual_avgpool_cnn",
        factory=ResidualAvgPoolCNN,
        input_shape=(1, 1, 28, 28),
        family="cnn",
        notes="Residual block with avg-pool downsampling",
    )
)

register_model(
    ModelSpec(
        name="depthwise_separable_cnn",
        factory=DepthwiseSeparableCNN,
        input_shape=(1, 1, 28, 28),
        family="cnn",
        notes="Depthwise + pointwise conv stack",
    )
)
