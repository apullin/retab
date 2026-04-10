"""Focused tests for control-flow delegate context helpers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import torch


def _load_cf_ctx():
    module_path = Path(__file__).parents[1] / "python" / "retab" / "_cf_ctx.py"
    spec = importlib.util.spec_from_file_location("_retab_cf_ctx", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_delegate_output_contract = _load_cf_ctx()._delegate_output_contract


@dataclass
class _DummyNode:
    name: str
    op: str = "call_function"
    target: object = "dummy.target"
    args: tuple = ()
    meta: dict = field(default_factory=dict)


def test_delegate_output_contract_reinterprets_table_output_layout_shape():
    source = _DummyNode(
        "source",
        target="aten.sigmoid.default",
        meta={
            "val": torch.randn(1, 8, 4, 4),
            "tosa_dim_order": (0, 2, 3, 1),
        },
    )
    table = _DummyNode(
        "table",
        target="tosa.TABLE",
        args=(source,),
        meta={
            "val": torch.randn(1, 8, 4, 4),
            "tosa_dim_order": (0, 2, 3, 1),
        },
    )
    rescale = _DummyNode(
        "table_rescale",
        target="tosa.RESCALE",
        args=(table,),
        meta={
            "val": torch.randn(1, 8, 4, 4),
            "tosa_dim_order": (0, 2, 3, 1),
        },
    )
    output_layout = _DummyNode(
        "sigmoid_output_layout",
        target="tosa.TRANSPOSE",
        args=(rescale, (0, 3, 1, 2)),
        meta={
            "val": torch.randn(1, 4, 8, 4),
            "tosa_dim_order": (0, 2, 3, 1),
        },
    )

    fake_val, dim_order = _delegate_output_contract(output_layout)

    assert tuple(fake_val.shape) == (1, 8, 4, 4)
    assert dim_order == (0, 1, 2, 3)


def test_delegate_output_contract_leaves_normal_output_layout_transpose_unchanged():
    source = _DummyNode(
        "source",
        target="aten.add.Tensor",
        meta={
            "val": torch.randn(1, 8, 4, 4),
            "tosa_dim_order": (0, 2, 3, 1),
        },
    )
    output_layout = _DummyNode(
        "add_output_layout",
        target="tosa.TRANSPOSE",
        args=(source, (0, 3, 1, 2)),
        meta={
            "val": torch.randn(1, 4, 8, 4),
            "tosa_dim_order": (0, 2, 3, 1),
        },
    )

    fake_val, dim_order = _delegate_output_contract(output_layout)

    assert fake_val is output_layout.meta["val"]
    assert dim_order == (0, 2, 3, 1)
