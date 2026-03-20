from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], nn.Module]
    input_shape: tuple[int, ...]
    family: str
    notes: str


_MODEL_SPECS: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    existing = _MODEL_SPECS.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(f"duplicate model spec for {spec.name!r}")
    _MODEL_SPECS[spec.name] = spec
    return spec


def all_model_specs() -> dict[str, ModelSpec]:
    return dict(sorted(_MODEL_SPECS.items()))


def iter_model_specs() -> Iterable[ModelSpec]:
    for name in sorted(_MODEL_SPECS):
        yield _MODEL_SPECS[name]


def make_example_input(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(*shape)


def get_model_and_example(name: str) -> tuple[nn.Module, torch.Tensor]:
    spec = all_model_specs()[name]
    model = spec.factory().eval()
    return model, make_example_input(spec.input_shape)
