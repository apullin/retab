"""RETAB-owned regressions for U55 rank-5 grouped layout handling.

The synthetic model exercises the risky shape grammar behind several historical
layout bugs: repeat/TILE, grouped convolutions, channel-direct normalization,
and view chains across rank-5 grouped dimensions.

Run with `RETAB_RUN_U55_REGRESSIONS=1`; the suite invokes Ethos-U lowering and
is intentionally not part of the default fast unit-test lane.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn


pytestmark = pytest.mark.skipif(
    os.environ.get("RETAB_RUN_U55_REGRESSIONS") != "1",
    reason="set RETAB_RUN_U55_REGRESSIONS=1 to run U55 grouped-layout regressions",
)


@dataclass
class _MiniConfig:
    in_h: int = 20
    in_w: int = 48
    channels: int = 32
    n_out: int = 2


class _ChannelNorm(nn.Module):
    def __init__(self, n_groups: int, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_groups * channels))
        self.bias = nn.Parameter(torch.zeros(n_groups * channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, nc, h, w = x.shape
        xg = x.view(b, self.n_groups, self.channels, h, w)
        mean = xg.mean(dim=2, keepdim=True)
        var = ((xg - mean) ** 2).mean(dim=2, keepdim=True)
        xg = (xg - mean) * torch.rsqrt(var + self.eps)
        x = xg.reshape(b, nc, h, w)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class _MiniRank5GroupedModel(nn.Module):
    def __init__(self, n_groups: int = 2, cfg: _MiniConfig | None = None) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.cfg = cfg or _MiniConfig()
        c = self.cfg.channels
        self.stem = nn.Conv2d(
            n_groups, n_groups * c, (self.cfg.in_h, 1), groups=n_groups
        )
        self.norm = _ChannelNorm(n_groups, c)
        self.dw = nn.Conv2d(
            n_groups * c,
            n_groups * c,
            (1, 7),
            padding="same",
            groups=n_groups * c,
        )
        self.pw1 = nn.Conv2d(n_groups * c, n_groups * c * 2, 1, groups=n_groups)
        self.pw2 = nn.Conv2d(n_groups * c * 2, n_groups * c, 1, groups=n_groups)
        self.head = nn.Linear(n_groups * c, self.cfg.n_out)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        x = window.unsqueeze(1).repeat(1, self.n_groups, 1, 1)
        x = self.norm(self.stem(x))
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw2(torch.nn.functional.gelu(self.pw1(x)))
        x = x + residual
        b, _, h, w = x.shape
        x = x.view(b, self.n_groups, self.cfg.channels, h, w)
        x = x.mean(dim=3).mean(dim=-1)
        x = x.reshape(b, self.n_groups * self.cfg.channels)
        return self.head(x)


def _import_tosa_schema():
    try:
        from tosa.DType import DType
        from tosa.Op import Op
        from tosa.TosaGraph import TosaGraph
        from tosa.TransposeAttribute import TransposeAttribute
    except ModuleNotFoundError as exc:
        pytest.skip(f"TOSA Python schema unavailable: {exc}")
    return DType, Op, TosaGraph, TransposeAttribute


def _parse_tosa(path: Path):
    dtype, op_enum, graph_cls, transpose_attr_cls = _import_tosa_schema()
    op_names = {
        getattr(op_enum(), name): name for name in dir(op_enum()) if name.isupper()
    }
    graph = graph_cls.GetRootAs(path.read_bytes(), 0)
    block = graph.Regions(0).Blocks(0)
    tensors = {}
    for index in range(block.TensorsLength()):
        tensor = block.Tensors(index)
        tensors[tensor.Name().decode()] = {
            "shape": [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())],
            "dtype": tensor.Type(),
        }
    ops = []
    for index in range(block.OperatorsLength()):
        op = block.Operators(index)
        item = {
            "index": index,
            "kind": op_names.get(op.Op(), str(op.Op())),
            "inputs": [op.Inputs(i).decode() for i in range(op.InputsLength())],
            "outputs": [op.Outputs(i).decode() for i in range(op.OutputsLength())],
            "transpose_perm": None,
        }
        if item["kind"] == "TRANSPOSE":
            attr = transpose_attr_cls()
            tbl = op.Attribute()
            attr.Init(tbl.Bytes, tbl.Pos)
            item["transpose_perm"] = [int(v) for v in attr.PermsAsNumpy()]
        ops.append(item)
    return dtype, tensors, ops


def _delegate_count(edge) -> int:
    gm = edge.exported_program().graph_module
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "get_attr" and "lowered" in str(node.target).lower()
    )


@pytest.fixture(scope="module")
def lowered_u55_mini_artifact(tmp_path_factory):
    pytest.importorskip("retab")
    pytest.importorskip("executorch")
    pytest.importorskip("torchao")

    from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from retab.integration.executorch_arm import activated
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    torch.manual_seed(0)
    cfg = _MiniConfig()
    model = _MiniRank5GroupedModel(2, cfg).eval()
    example_inputs = (torch.randn(1, cfg.in_h, cfg.in_w),)

    compile_spec = EthosUCompileSpec(
        target="ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags=["--output-format=raw"],
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())

    exported = torch.export.export(model, example_inputs, strict=True)
    quantized_module = prepare_pt2e(exported.module(), quantizer)
    with torch.no_grad():
        quantized_module(*copy.deepcopy(example_inputs))
    quantized_module = convert_pt2e(quantized_module)
    quantized_ep = torch.export.export(quantized_module, example_inputs, strict=True)

    out_dir = tmp_path_factory.mktemp("u55_rank5_grouped_layout")
    tosa_path = out_dir / "mini_rank5_grouped_layout.tosa"
    old_dump = os.environ.get("RETAB_DUMP_TOSA")
    os.environ["RETAB_DUMP_TOSA"] = str(tosa_path)
    try:
        with activated():
            edge = to_edge_transform_and_lower(
                quantized_ep,
                partitioner=[EthosUPartitioner(compile_spec)],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
    finally:
        if old_dump is None:
            os.environ.pop("RETAB_DUMP_TOSA", None)
        else:
            os.environ["RETAB_DUMP_TOSA"] = old_dump

    pte = edge.to_executorch().buffer
    assert tosa_path.exists(), "RETAB_DUMP_TOSA did not produce a TOSA artifact"
    dtype, tensors, ops = _parse_tosa(tosa_path)
    return {
        "delegate_count": _delegate_count(edge),
        "pte_size": len(pte),
        "dtype": dtype,
        "tensors": tensors,
        "ops": ops,
        "tosa_path": tosa_path,
    }


def test_u55_rank5_grouped_layout_lowers_to_one_delegate(lowered_u55_mini_artifact):
    assert lowered_u55_mini_artifact["delegate_count"] == 1
    assert lowered_u55_mini_artifact["pte_size"] > 0


def test_grouped_conv_slice_outputs_keep_input_dtype(lowered_u55_mini_artifact):
    dtype = lowered_u55_mini_artifact["dtype"]
    tensors = lowered_u55_mini_artifact["tensors"]
    slice_ops = [op for op in lowered_u55_mini_artifact["ops"] if op["kind"] == "SLICE"]
    assert slice_ops, "mini rank-5 grouped model should contain grouped-conv SLICE ops"

    for op in slice_ops:
        input_dtype = tensors[op["inputs"][0]]["dtype"]
        output_dtype = tensors[op["outputs"][0]]["dtype"]
        if input_dtype in {dtype.INT8, dtype.INT16, dtype.INT32}:
            assert output_dtype == input_dtype, (
                f"{op['outputs'][0]} dtype {output_dtype} did not match "
                f"{op['inputs'][0]} dtype {input_dtype}"
            )


def test_repeat_tile_multiples_are_const_shape(lowered_u55_mini_artifact):
    dtype = lowered_u55_mini_artifact["dtype"]
    tensors = lowered_u55_mini_artifact["tensors"]
    ops = lowered_u55_mini_artifact["ops"]
    producer_kind = {
        output: op["kind"] for op in ops for output in op["outputs"]
    }
    tile_ops = [op for op in ops if op["kind"] == "TILE"]
    assert tile_ops, "mini rank-5 grouped model should contain repeat/TILE ops"

    for op in tile_ops:
        multiples = op["inputs"][1]
        assert producer_kind[multiples] == "CONST_SHAPE"
        if multiples in tensors:
            assert tensors[multiples]["dtype"] == dtype.SHAPE


def test_u55_rank5_grouped_layout_pte_stays_compact(lowered_u55_mini_artifact):
    assert lowered_u55_mini_artifact["pte_size"] < 80_000


def test_view_boundary_layout_ping_pong_is_eliminated(lowered_u55_mini_artifact):
    ops = lowered_u55_mini_artifact["ops"]
    ping_pongs = []
    for first, second in zip(ops, ops[1:]):
        if first["kind"] != "TRANSPOSE" or second["kind"] != "TRANSPOSE":
            continue
        if first["outputs"] != second["inputs"]:
            continue
        if not first["outputs"][0].endswith("_output_layout"):
            continue
        if not second["outputs"][0].endswith("_input_layout"):
            continue
        ping_pongs.append((first["index"], first["outputs"][0], second["outputs"][0]))

    assert not ping_pongs
