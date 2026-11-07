"""Focused tests for the delegate metadata helper."""

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


_collect_delegate_info_meta = _load_cf_ctx()._collect_delegate_info_meta


@dataclass
class _DummyNode:
    name: str
    op: str = "call_function"
    args: tuple = ()
    meta: dict = field(default_factory=dict)


class _DummyGraph:
    def __init__(self, output):
        self._output = output

    def output_node(self):
        return self._output


@dataclass
class _DummyGraphModule:
    graph: _DummyGraph


def _graph_module(outputs) -> _DummyGraphModule:
    output = _DummyNode("output", op="output", args=(outputs,))
    return _DummyGraphModule(_DummyGraph(output))


def test_delegate_info_meta_lists_every_output_leaf_in_order():
    first = _DummyNode("first", meta={"val": torch.randn(1, 8, 4, 4)})
    second = _DummyNode("second", meta={"val": torch.randn(2)})
    untyped = _DummyNode("untyped")

    outputs = (first, [second, None], (untyped,))
    info = _collect_delegate_info_meta(_graph_module(outputs))

    entries = info["output_meta"]
    assert all(set(entry) == {"fake_val", "node_name"} for entry in entries)
    assert [entry["node_name"] for entry in entries] == ["first", "second", "untyped"]
    assert entries[0]["fake_val"] is first.meta["val"]
    assert entries[1]["fake_val"] is second.meta["val"]
    assert entries[2]["fake_val"] is None

