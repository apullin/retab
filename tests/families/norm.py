from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.blocks import ChannelAffineNorm2d, RMSNorm2d
from lib.registry import ModelSpec, register_model


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


register_model(
    ModelSpec(
        name="rmsnorm_residual_cnn",
        factory=RMSNormResidualCNN,
        input_shape=(1, 2, 16, 16),
        family="norm",
        notes="Residual CNN using RMSNorm-style decomposition",
    )
)

register_model(
    ModelSpec(
        name="channel_norm_block_cnn",
        factory=ChannelNormBlockCNN,
        input_shape=(1, 2, 16, 16),
        family="norm",
        notes="Small conv block with channel-wise mean/var norm and affine broadcast",
    )
)

register_model(
    ModelSpec(
        name="rmsnorm_block_cnn",
        factory=RMSNormBlockCNN,
        input_shape=(1, 2, 16, 16),
        family="norm",
        notes="Small conv block with RMSNorm-style affine broadcast",
    )
)

register_model(
    ModelSpec(
        name="reshape_reduce_broadcast_cnn",
        factory=ReshapeReduceBroadcastCNN,
        input_shape=(1, 2, 16, 16),
        family="reshape_norm",
        notes="Conv block with reshape -> reduce -> rsqrt -> broadcast affine -> reshape",
    )
)
