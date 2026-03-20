from __future__ import annotations

import torch
import torch.nn as nn


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
