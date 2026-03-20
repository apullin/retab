from __future__ import annotations

from model import MODELS

from lib.registry import ModelSpec, register_model


for name, (factory, shape) in MODELS.items():
    register_model(
        ModelSpec(
            name=name,
            factory=factory,
            input_shape=shape,
            family="baseline",
            notes="Existing local baseline model",
        )
    )
