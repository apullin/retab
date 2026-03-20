from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.registry import ModelSpec, register_model


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


register_model(
    ModelSpec(
        name="permute_layernorm_cnn",
        factory=PermuteLayerNormCNN,
        input_shape=(1, 2, 16, 16),
        family="layout_norm",
        notes="Conv block with NHWC layer_norm round-trip",
    )
)
