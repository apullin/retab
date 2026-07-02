#!/usr/bin/env python3
"""Compatibility shim for the direct lowering model corpus.

Family-specific model definitions live under `tests/families/` and register
themselves through the shared registry in `tests/lib/registry.py`.
"""

from __future__ import annotations

from families import load_all_families
from lib.registry import all_model_specs, get_model_and_example, iter_model_specs


load_all_families()


if __name__ == "__main__":
    for spec in iter_model_specs():
        model, x = get_model_and_example(spec.name)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(
            f"{spec.name:25s}  params={params:,}  "
            f"input={tuple(x.shape)}  output={tuple(y.shape)}"
        )
