from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.registry import ModelSpec, register_model


class WhereBlendCNN(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.alt = nn.Conv2d(8, 8, kernel_size=1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        residual = self.alt(x)
        x = torch.where(x > 0, x, residual)
        return self.head(x)


class SlicePadCatCNN(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.mix = nn.Conv2d(12, 8, kernel_size=1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        cropped = x[:, :, 1:-1, 1:-1]
        padded = F.pad(cropped, (1, 1, 1, 1))
        x = torch.cat([x, padded], dim=1)
        x = F.relu(self.mix(x))
        return self.head(x)


class GatherIndexSelectCNN(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.mix = nn.Conv2d(8, 8, kernel_size=1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )
        self.register_buffer("indices", torch.tensor([0, 2, 5, 7], dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        flat = x.flatten(2)
        picked = torch.index_select(flat, 2, self.indices)
        gather_idx = self.indices.view(1, 1, -1).repeat(flat.shape[0], flat.shape[1], 1)
        gathered = torch.gather(flat, 2, gather_idx)
        x = (picked + gathered).reshape(flat.shape[0], flat.shape[1], 2, 2)
        x = F.relu(self.mix(x))
        return self.head(x)


register_model(
    ModelSpec(
        name="where_blend_cnn",
        factory=WhereBlendCNN,
        input_shape=(1, 2, 16, 16),
        family="data_movement",
        notes="Conv block with tensor where on a broadcast threshold mask",
    )
)

register_model(
    ModelSpec(
        name="slice_pad_cat_cnn",
        factory=SlicePadCatCNN,
        input_shape=(1, 1, 16, 16),
        family="data_movement",
        notes="Conv block with slice, pad, and cat in the NHWC path",
    )
)

register_model(
    ModelSpec(
        name="gather_index_select_cnn",
        factory=GatherIndexSelectCNN,
        input_shape=(1, 1, 16, 16),
        family="data_movement",
        notes="Conv block with index_select and gather over flattened spatial positions",
    )
)
