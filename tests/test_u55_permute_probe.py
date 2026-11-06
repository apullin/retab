"""Explicit mode splits the U55 permutes rc4 splits, by asking rc4's exact check (APU-160).

rc4's DecomposePermuteForU55Pass judges every permute alive at its slot. A batch-2 conv
over a 1x32769 signal gets an NHWC and an NCHW permute from IrRewriteConv that break U55's
worst-case TRANSPOSE rule (2 * 32769 > 2**16) and that Vela rejects, so rc4 splits both
into SLICE -> TRANSPOSE -> CONCAT. Without the probe, explicit mode split only permutes
named on the edge program, left these two whole, and Vela failed with "Non-passthrough
operation (Transpose) could not run on NPU".
"""

import copy
from collections import Counter

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.native, pytest.mark.executorch, pytest.mark.refmodel]

pytest.importorskip("executorch.backends.arm.test.tester.test_pipeline")
pytest.importorskip("ethosu.vela")

CONV_ATEN = "torch.ops.aten.conv2d.default"
CONV_EXIR = "executorch_exir_dialects_edge__ops_aten_convolution_default"
INPUT_SHAPE = (2, 2, 1, 32769)


class BatchedConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, (1, 3), padding=(0, 1))

    def forward(self, x):
        return self.conv(x)


def _ops(tosa: bytes, *, constants: bool = True) -> list[str]:
    from tosa import Op, TosaGraph

    names = {v: k for k, v in vars(Op.Op).items() if not k.startswith("_") and isinstance(v, int)}
    block = TosaGraph.TosaGraph.GetRootAs(bytearray(tosa), 0).Regions(0).Blocks(0)
    ops = [names[block.Operators(i).Op()] for i in range(block.OperatorsLength())]
    return ops if constants else [op for op in ops if op not in ("CONST", "CONST_SHAPE")]


def _reference_outputs(tosa: bytes, inputs: list[np.ndarray]) -> list[np.ndarray]:
    from tosa_reference_model import tosa_reference_model as reference_model

    outputs, status = reference_model.run(
        tosa, [array.copy() for array in inputs], initialize_variable_tensor_from_numpy=True
    )
    assert status == reference_model.GraphStatus.TOSA_VALID
    return outputs


@pytest.fixture
def stock_and_retab(monkeypatch):
    """Lower with RETAB in explicit mode, recording stock's and RETAB's TOSA of each
    delegate (stock runs on a copy of the edge program)."""
    from executorch.backends.arm.tosa.backend import TOSABackend
    from retab.preprocess import _all_rust_preprocess, stock_preprocess

    monkeypatch.setenv("RETAB_LAYOUT", "explicit")
    records = []

    def differential(edge_program, compile_spec):
        stock = stock_preprocess(copy.deepcopy(edge_program), compile_spec)
        retab = _all_rust_preprocess(edge_program, compile_spec)
        records.append((bytes(stock.processed_bytes), bytes(retab.processed_bytes)))
        return retab

    monkeypatch.setattr(TOSABackend, "_preprocess", staticmethod(differential))
    return records


def _assert_split_like_stock(stock: bytes, retab: bytes) -> None:
    """Both conv permutes are split in two along the signal, 32768 + 1, as stock splits
    them: each chunk is sliced, the 32768-wide one transposed and the 1-wide one
    reshaped, and the two are concatenated. The chunks' constants may sit in another
    order, so constants are left out of the sequence."""
    assert _ops(retab, constants=False) == _ops(stock, constants=False)
    counts = Counter(_ops(retab))
    assert (counts["SLICE"], counts["TRANSPOSE"], counts["RESHAPE"], counts["CONCAT"]) == (4, 2, 2, 2)


def test_tosa_u55_conv_permutes_split_like_stock(stock_and_retab):
    """rc4's TOSA pipeline for a U55 subset spec: the harness checks RETAB's TOSA on the
    reference model against the quantized module, and RETAB's outputs equal stock's."""
    from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT

    torch.manual_seed(0)
    pipeline = TosaPipelineINT[tuple[torch.Tensor]](
        BatchedConv(),
        (torch.randn(*INPUT_SHAPE),),
        CONV_ATEN,
        CONV_EXIR,
        tosa_extensions=["u55"],
    )
    pipeline.run()

    [(stock, retab)] = stock_and_retab
    _assert_split_like_stock(stock, retab)
    inputs = [np.random.default_rng(0).integers(-128, 128, INPUT_SHAPE, dtype=np.int8)]
    for got, want in zip(_reference_outputs(retab, inputs), _reference_outputs(stock, inputs), strict=True):
        np.testing.assert_array_equal(got, want)


def test_ethos_u55_vela_compiles_the_split_conv_permutes(stock_and_retab):
    """rc4's Ethos-U55 pipeline up to the Vela compile, which rejected the unsplit
    TRANSPOSEs."""
    from executorch.backends.arm.test.tester.test_pipeline import EthosU55PipelineINT

    torch.manual_seed(0)
    pipeline = EthosU55PipelineINT[tuple[torch.Tensor]](
        BatchedConv(),
        (torch.randn(*INPUT_SHAPE),),
        CONV_ATEN,
        CONV_EXIR,
        run_on_fvp=False,
    )
    pipeline.run()

    [(stock, retab)] = stock_and_retab
    _assert_split_like_stock(stock, retab)
