#!/usr/bin/env python3
"""Parity checks for the diagnostic Rust IR fold_qdq entrypoint.

These tests compare `retab.fold_qdq_ir_fx(...)` against the existing Python
`FoldAndAnnotateQParamsPass` on small graphs first, then one tiny exported
per-channel case.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node


ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ENGINE_ROOT.parent.parent

for path in (ENGINE_ROOT / "python", WORK_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (  # noqa: E402
    FoldAndAnnotateQParamsPass,
)
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402


def _import_retab():
    try:
        import retab
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"retab import unavailable in local env: {exc}")
    return retab


def _annotate_quantized(node: Node) -> None:
    node.meta.setdefault("custom", {})[
        ArmAnnotationInfo.CUSTOM_META_KEY
    ] = ArmAnnotationInfo(quantized=True)


def _count_qdq(gm: GraphModule) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function"
        and ("quantize" in str(node.target) or "dequantize" in str(node.target))
    )


def _canonical_quant_arg(qarg) -> dict[str, object]:
    scale = qarg.scale
    zp = getattr(qarg, "zp", getattr(qarg, "zero_point", None))
    if isinstance(scale, list):
        scale = tuple(float(v) for v in scale)
    else:
        scale = float(scale)
    if isinstance(zp, list):
        zp = tuple(int(v) for v in zp)
    else:
        zp = int(zp)
    return {
        "scale": scale,
        "zp": zp,
        "qmin": int(qarg.qmin),
        "qmax": int(qarg.qmax),
        "dtype": str(qarg.dtype),
        "axis": int(getattr(qarg, "axis", 0)),
        "per_channel": bool(getattr(qarg, "per_channel", False)),
    }


def _canonical_qparam_map(qparams) -> dict[int, dict[str, object]]:
    if not qparams:
        return {}
    return {int(idx): _canonical_quant_arg(qp) for idx, qp in sorted(qparams.items())}


def _call_function_fingerprint(gm: GraphModule) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        rows.append(
            {
                "name": node.name,
                "target": str(node.target),
                "kwargs": {key: str(val) for key, val in sorted(node.kwargs.items())},
                "input_qparams": _canonical_qparam_map(node.meta.get("input_qparams")),
                "output_qparams": _canonical_qparam_map(node.meta.get("output_qparams")),
            }
        )
    return rows


def _find_single_call_function(gm: GraphModule, marker: str) -> Node:
    matches = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and marker in str(node.target)
    ]
    assert len(matches) == 1, f"expected one call_function containing {marker!r}, got {matches}"
    return matches[0]


def _run_python_fold(
    gm: GraphModule, exported_program=None
) -> tuple[bool, GraphModule]:
    result = FoldAndAnnotateQParamsPass(exported_program=exported_program)(gm)
    graph_module = result.graph_module if hasattr(result, "graph_module") else gm
    modified = bool(getattr(result, "modified", True))
    return modified, graph_module


def _run_rust_ir_fold(
    gm: GraphModule, exported_program=None
) -> tuple[bool, GraphModule]:
    retab = _import_retab()
    modified = retab.fold_qdq_ir_fx(gm, exported_program)
    return bool(modified), gm


def _build_simple_dq_add_q_graph() -> GraphModule:
    graph = Graph()
    x = graph.placeholder("x")
    dq = graph.call_function(
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        (x, 0.25, 3, -128, 127, torch.int8),
    )
    add = graph.call_function(exir_ops.edge.aten.add.Tensor, (dq, 1))
    q = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        (add, 0.25, 3, -128, 127, torch.int8),
    )
    graph.output(q)

    gm = GraphModule(nn.Module(), graph)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = torch.randn(2, 3)
        elif node is add:
            _annotate_quantized(node)
    gm.recompile()
    return gm


def _build_sum_dim_intlist_graph() -> GraphModule:
    graph = Graph()
    x = graph.placeholder("x")
    sum_node = graph.call_function(
        exir_ops.edge.aten.sum.dim_IntList,
        (x, [1], False),
        {"dtype": torch.int32},
    )
    q = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        (sum_node, 0.5, 5, -128, 127, torch.int8),
    )
    graph.output(q)

    gm = GraphModule(nn.Module(), graph)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = torch.randn(2, 3)
        elif node is sum_node:
            _annotate_quantized(node)
    gm.recompile()
    return gm


def _build_quantized_conv_export() -> tuple[torch.export.ExportedProgram, GraphModule]:
    pytest.importorskip("torchao.quantization.pt2e.quantize_pt2e")
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    class TinyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = TinyConv().eval()
    example = (torch.randn(1, 3, 8, 8),)
    exported = torch.export.export(model, example, strict=True)
    qmodel = exported.module()

    quantizer = EthosUQuantizer(EthosUCompileSpec(target="ethos-u55-128"))
    quantizer.set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(qmodel, quantizer)
    prepared(*example)
    quantized = convert_pt2e(prepared)
    qep = torch.export.export(quantized, example, strict=True)
    gm = copy.deepcopy(qep.graph_module)

    for node in gm.graph.nodes:
        if node.op == "call_function":
            target = str(node.target)
            if "quantize" not in target and "dequantize" not in target:
                _annotate_quantized(node)
    return qep, gm


def test_fold_qdq_ir_fx_matches_python_simple_dq_add_q_fold():
    base = _build_simple_dq_add_q_graph()

    py_modified, py_gm = _run_python_fold(copy.deepcopy(base))
    rs_modified, rs_gm = _run_rust_ir_fold(copy.deepcopy(base))

    assert py_modified and rs_modified
    assert _count_qdq(py_gm) == 0
    assert _count_qdq(rs_gm) == 0
    assert _call_function_fingerprint(rs_gm) == _call_function_fingerprint(py_gm)


def test_fold_qdq_ir_fx_matches_python_sum_dim_intlist_dtype_fix():
    base = _build_sum_dim_intlist_graph()

    py_modified, py_gm = _run_python_fold(copy.deepcopy(base))
    rs_modified, rs_gm = _run_rust_ir_fold(copy.deepcopy(base))

    assert py_modified and rs_modified
    py_sum = _find_single_call_function(py_gm, "sum.dim_IntList")
    rs_sum = _find_single_call_function(rs_gm, "sum.dim_IntList")
    assert py_sum.kwargs["dtype"] == torch.int8
    assert rs_sum.kwargs["dtype"] == torch.int8
    assert _call_function_fingerprint(rs_gm) == _call_function_fingerprint(py_gm)


def test_fold_qdq_ir_fx_matches_python_per_channel_weight_dq():
    exported_program, base = _build_quantized_conv_export()
    assert any(
        node.op == "call_function" and "dequantize_per_channel" in str(node.target)
        for node in base.graph.nodes
    ), "expected a per-channel weight DQ in the quantized conv graph"

    py_modified, py_gm = _run_python_fold(
        copy.deepcopy(base), exported_program=exported_program
    )
    rs_modified, rs_gm = _run_rust_ir_fold(
        copy.deepcopy(base), exported_program=exported_program
    )

    assert py_modified and rs_modified
    py_conv = _find_single_call_function(py_gm, "conv")
    rs_conv = _find_single_call_function(rs_gm, "conv")
    assert _canonical_qparam_map(rs_conv.meta.get("input_qparams")) == _canonical_qparam_map(
        py_conv.meta.get("input_qparams")
    )
    assert _canonical_qparam_map(rs_conv.meta.get("output_qparams")) == _canonical_qparam_map(
        py_conv.meta.get("output_qparams")
    )

    input_qparams = _canonical_qparam_map(rs_conv.meta.get("input_qparams"))
    assert any(qp["per_channel"] for qp in input_qparams.values())
    assert _count_qdq(rs_gm) == _count_qdq(py_gm)
