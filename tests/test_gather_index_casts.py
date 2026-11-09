"""rc4's TopKGather case on TOSA-1.0+INT and Ethos-U85's spec (Linear APU-167).

topk's indices, unsqueezed, expanded and cast to int64, index a gather. TopK is FP-only, so
on an INT spec rc4's partitioner delegates the int32 indices, the int64 cast and the
gather, and its PrepareGatherIndicesPass points the gather back at the int32 indices.
RETAB used to serialize the cast and fail on its INT64 tensor. Now it lowers the delegate
itself, with stock's operators, and the reference model gives the same results for both.
"""

import copy

import numpy as np
import pytest
import torch

pytest.importorskip("executorch.backends.arm.tosa.compile_spec")
pytest.importorskip("retab._native")

from executorch.backends.arm.quantizer import (
    TOSAQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from retab.preprocess import _all_rust_preprocess, preprocess_stats, stock_preprocess
from tosa import DType, Op, TosaGraph
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

try:
    from tosa_tools.tosa_reference_model import tosa_reference_model as reference_model
except ImportError:
    reference_model = pytest.importorskip("tosa_reference_model").tosa_reference_model

pytestmark = [pytest.mark.native, pytest.mark.executorch, pytest.mark.refmodel]

# Generic INT, and the spec EthosUCompileSpec gives Ethos-U85.
SPECS = {"tosa_INT": "TOSA-1.0+INT", "u85_INT": "TOSA-1.0+INT+int16+int4+cf"}


class TopKGather(torch.nn.Module):
    """rc4's TopKGather (backends/arm/test/passes/test_convert_int64_output_ops_to_int32.py)."""

    def forward(self, scores, features):
        indices = torch.topk(scores, 3, dim=1).indices
        expanded = indices.unsqueeze(-1).expand(-1, -1, 2)
        return torch.gather(features, dim=1, index=expanded)


def _delegate(spec, monkeypatch):
    """The edge program and compile spec of the quantized case's one TOSA delegate."""
    compile_spec = TosaCompileSpec(spec)
    torch.manual_seed(0)
    inputs = (torch.randn(2, 8), torch.randn(2, 8, 2))
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    module = prepare_pt2e(torch.export.export(TopKGather(), inputs).module(), quantizer)
    module(*inputs)
    program = torch.export.export(convert_pt2e(module), inputs)

    captured = []
    stock = TOSABackend.__dict__["_preprocess"].__func__

    def capture(edge_program, tosa_spec):
        captured.append((copy.deepcopy(edge_program), tosa_spec))
        return stock(edge_program, tosa_spec)

    monkeypatch.setattr(TOSABackend, "_preprocess", staticmethod(capture))
    to_edge_transform_and_lower(
        program,
        partitioner=[TOSAPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    assert len(captured) == 1
    return captured[0]


def _main_block(data):
    return TosaGraph.TosaGraph.GetRootAs(bytearray(data), 0).Regions(0).Blocks(0)


def _tensors(block):
    return {
        block.Tensors(i).Name().decode(): block.Tensors(i) for i in range(block.TensorsLength())
    }


def _ops(block):
    return sorted(block.Operators(i).Op() for i in range(block.OperatorsLength()))


def _run(data, arrays):
    outputs, status = reference_model.run(data, arrays, initialize_variable_tensor_from_numpy=True)
    assert status == reference_model.GraphStatus.TOSA_VALID
    assert len(outputs) == 1
    return outputs[0]


@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_topk_gather_lowers_on_retab_like_stock(spec, monkeypatch):
    edge_program, compile_spec = _delegate(spec, monkeypatch)
    before = preprocess_stats()
    retab = bytes(_all_rust_preprocess(copy.deepcopy(edge_program), compile_spec).processed_bytes)
    after = preprocess_stats()
    assert after["retab"] == before["retab"] + 1, "RETAB's preprocess never ran"
    assert after["stock_fallback"] == before["stock_fallback"], "the graph fell back to stock"
    stock = bytes(stock_preprocess(copy.deepcopy(edge_program), compile_spec).processed_bytes)

    retab_block, stock_block = _main_block(retab), _main_block(stock)
    assert _ops(retab_block) == _ops(stock_block)
    tensors = _tensors(retab_block)
    allowed = (DType.DType.INT8, DType.DType.INT32, DType.DType.SHAPE)
    assert all(tensor.Type() in allowed for tensor in tensors.values()), "an INT64 tensor"
    gather = next(
        retab_block.Operators(i)
        for i in range(retab_block.OperatorsLength())
        if retab_block.Operators(i).Op() == Op.Op.GATHER
    )
    index = tensors[gather.Inputs(1).decode()]
    assert index.Type() == DType.DType.INT32
    assert index.ShapeAsNumpy().tolist() == [4, 3]

    # The delegate's inputs: the quantized features and the topk's int32 indices.
    signature = []
    for block in (retab_block, stock_block):
        named = _tensors(block)
        signature.append(
            [
                (named[name].Type(), named[name].ShapeAsNumpy().tolist())
                for name in (block.Inputs(i).decode() for i in range(block.InputsLength()))
            ]
        )
    expected = [(DType.DType.INT8, [2, 8, 2]), (DType.DType.INT32, [2, 3])]
    assert signature[0] == signature[1] == expected

    rng = np.random.default_rng(0)
    for _ in range(3):
        features = rng.integers(-128, 128, size=(2, 8, 2), dtype=np.int8)
        indices = rng.integers(0, 8, size=(2, 3), dtype=np.int32)
        result = _run(retab, [features, indices])
        np.testing.assert_array_equal(result, _run(stock, [features, indices]))
        expanded = np.broadcast_to(indices[:, :, None], (2, 3, 2))
        np.testing.assert_array_equal(result, np.take_along_axis(features, expanded, axis=1))
