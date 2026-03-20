from __future__ import annotations

import torch
import torch.nn as nn

from lib.registry import ModelSpec, register_model


class RawMatmulModel(nn.Module):
    def __init__(self, features: int = 16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


class RawBmmModel(nn.Module):
    def __init__(self, in_features: int = 8, out_features: int = 10) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rhs = self.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.bmm(x, rhs)


class RawAddmmModel(nn.Module):
    def __init__(self, in_features: int = 14, out_features: int = 9) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randn(in_features, out_features))
        self.register_buffer("bias", torch.randn(3, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x, self.weight)


register_model(
    ModelSpec(
        name="raw_matmul_model",
        factory=RawMatmulModel,
        input_shape=(16,),
        family="matmul",
        notes="1D dot-product matmul surface beyond the linear path",
    )
)

register_model(
    ModelSpec(
        name="raw_bmm_model",
        factory=RawBmmModel,
        input_shape=(2, 5, 8),
        family="matmul",
        notes="Batched matmul lowered through bmm",
    )
)

register_model(
    ModelSpec(
        name="raw_addmm_model",
        factory=RawAddmmModel,
        input_shape=(3, 14),
        family="matmul",
        notes="Raw addmm surface beyond the linear path",
    )
)
