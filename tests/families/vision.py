from __future__ import annotations

from torchvision import models  # type: ignore[import-untyped]

from lib.registry import ModelSpec, register_model


register_model(
    ModelSpec(
        name="resnet18",
        factory=lambda: models.resnet18(weights=None),
        input_shape=(1, 3, 224, 224),
        family="vision",
        notes="Torchvision ResNet-18 without pretrained weights",
    )
)

register_model(
    ModelSpec(
        name="mobilenet_v3_small",
        factory=lambda: models.mobilenet_v3_small(weights=None),
        input_shape=(1, 3, 232, 232),
        family="vision",
        notes="Torchvision MobileNetV3-Small without pretrained weights",
    )
)
