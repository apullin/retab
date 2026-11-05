"""Regression case: RETAB picks a different layout than stock through LayerNorm.

RETAB is meant to reproduce ExecuTorch's ARM TOSA preprocess, so an artifact that
differs from the stock one is a defect even when both are valid. This pins the
smallest network found that still diverges: Conv -> LayerNorm -> ReLU.

What is the same, and asserted here:
  * the TOSA opcode multiset (206 ops on both paths, reordered only)
  * 40 of the 42 emitted constants

What differs, and is xfailed here:
  the two LayerNorm affine constants. Stock runs the norm body in NCHW and emits
  them as ``[1, 8, 16, 16]``; RETAB keeps NHWC to the delegate boundary and emits
  ``[1, 16, 16, 8]``. The payloads are genuine NCHW->NHWC permutations of each
  other, so both graphs compute the same thing -- it is a layout *choice*, not a
  numerical error. It still matters because it is a divergence from stock, and
  because layout choices feed Vela's scheduling downstream.

A note on why the affine values matter, worth keeping in mind when reproducing:
with torch's default LayerNorm init (weight=1, bias=0) the quantized gamma/beta
are uniform tensors, and the divergence survives all the way through Vela into a
different Ethos-U command stream. With randomized affine params
(``CnnLayerNormRandomAffine``) the TOSA still differs exactly as above, but Vela
normalizes it away and the compiled artifacts come out byte-identical. So TOSA
parity is the stricter and more useful signal; Ethos-U parity can mask this.

Comparison is deliberately name-independent. Tensor names diverge wholesale
between the two paths -- stock names a tensor after the TOSA op that emitted it
(``tosa_conv2d_default``), RETAB after the source ATen node
(``aten_convolution_default``) -- so only opcode multisets and a multiset of
(dtype, shape, payload-hash) over the constants are meaningful.
"""

import hashlib
import unittest
from collections import Counter

import torch
import torch.nn as nn


class CnnLayerNormDefaultInit(nn.Module):
    """LayerNorm left at torch's default init, so gamma/beta quantize to uniform."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.norm = nn.LayerNorm([8, 16, 16])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x)))


class CnnLayerNormRandomAffine(nn.Module):
    """Identical graph, non-uniform affine params. Control for the pair."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.norm = nn.LayerNorm([8, 16, 16])
        with torch.no_grad():
            self.norm.weight.copy_(torch.rand(8, 16, 16) * 2.0 + 0.5)
            self.norm.bias.copy_(torch.randn(8, 16, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x)))


INPUT_SHAPE = (1, 3, 16, 16)


def _strip_guards_fn(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Drop the unused ``_guards_fn`` submodule ``ExportedProgram.module()`` adds.

    The ARM annotation pipeline runs on an ExportPass interpreter, which rejects
    any ``call_module`` node outright. ``_guards_fn`` is a runtime shape-assertion
    hook with no users, so removing it is safe.
    """
    changed = False
    for node in list(gm.graph.nodes):
        if node.op == "call_module" and "_guards_fn" in str(node.target):
            if node.users:
                continue
            gm.graph.erase_node(node)
            changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return gm


def _prepare_quantized(model: nn.Module, seed: int = 0):
    from executorch.backends.arm.quantizer import (
        get_symmetric_quantization_config,
        TOSAQuantizer,
    )
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    torch.manual_seed(seed)
    example = torch.randn(*INPUT_SHAPE)
    exported = torch.export.export(model.eval(), (example,), strict=True)

    compile_spec = TosaCompileSpec("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))

    qmodel = prepare_pt2e(_strip_guards_fn(exported.module()), quantizer)
    with torch.no_grad():
        for _ in range(3):
            qmodel(torch.randn(*INPUT_SHAPE))
    qmodel = convert_pt2e(qmodel)
    return torch.export.export(qmodel, (example,), strict=True), compile_spec


def _lower_to_tosa(program, compile_spec, *, use_retab: bool) -> bytes:
    """Lower once and return the single delegate's TOSA flatbuffer."""
    import copy

    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    def run():
        return to_edge_transform_and_lower(
            copy.deepcopy(program),
            partitioner=[TOSAPartitioner(compile_spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

    if use_retab:
        from retab.integration.executorch_arm import activated

        with activated():
            edge = run()
    else:
        from retab.integration.executorch_arm import uninstall

        uninstall()
        edge = run()

    exported = edge.exported_program()
    blobs = [
        bytes(module.processed_bytes)
        for node in exported.graph.nodes
        if node.op == "get_attr" and "lowered" in str(node.target)
        for module in [getattr(exported.graph_module, node.target, None)]
        if module is not None and hasattr(module, "processed_bytes")
    ]
    if len(blobs) != 1:
        raise AssertionError(f"expected a single delegate, got {len(blobs)}")
    return blobs[0]


def _structure(tosa_bytes: bytes) -> tuple[list[int], "Counter[tuple]"]:
    """Opcode sequence plus a name-independent fingerprint of the constants."""
    from tosa import TosaGraph as TosaGraphFB

    graph = TosaGraphFB.TosaGraph.GetRootAs(bytearray(tosa_bytes), 0)
    block = graph.Regions(0).Blocks(0)

    opcodes = [block.Operators(i).Op() for i in range(block.OperatorsLength())]

    constants = Counter()
    for i in range(block.TensorsLength()):
        tensor = block.Tensors(i)
        if tensor.DataLength() == 0:
            continue
        payload = bytes(tensor.DataAsNumpy())
        constants[
            (
                tensor.Type(),
                tuple(tensor.Shape(j) for j in range(tensor.ShapeLength())),
                hashlib.sha256(payload).hexdigest(),
            )
        ] += 1
    return opcodes, constants


class StockParityLayerNormTest(unittest.TestCase):
    def _lower_both(self, model: nn.Module):
        program, compile_spec = _prepare_quantized(model)
        return (
            _structure(_lower_to_tosa(program, compile_spec, use_retab=False)),
            _structure(_lower_to_tosa(program, compile_spec, use_retab=True)),
        )

    def test_opcode_multiset_matches_stock(self) -> None:
        """The two paths emit the same ops; only their order and layout differ."""
        (stock_ops, _), (retab_ops, _) = self._lower_both(CnnLayerNormDefaultInit())
        self.assertEqual(
            Counter(stock_ops),
            Counter(retab_ops),
            "RETAB emitted a different multiset of TOSA opcodes than stock",
        )

    def test_most_constants_match_stock(self) -> None:
        """Only the LayerNorm affine pair should differ, not the bulk."""
        (_, stock_consts), (_, retab_consts) = self._lower_both(
            CnnLayerNormDefaultInit()
        )
        shared = sum((stock_consts & retab_consts).values())
        self.assertGreaterEqual(
            shared,
            sum(stock_consts.values()) - 2,
            "more than the two LayerNorm affine constants diverged from stock",
        )

    @unittest.expectedFailure
    def test_constants_match_stock(self) -> None:
        """Known divergence: the LayerNorm affine constants are emitted NHWC.

        Stock emits them ``[1, 8, 16, 16]``, RETAB ``[1, 16, 16, 8]``. Remove the
        xfail once RETAB reproduces stock's layout through the norm body.
        """
        (_, stock_consts), (_, retab_consts) = self._lower_both(
            CnnLayerNormDefaultInit()
        )
        self.assertEqual(
            stock_consts,
            retab_consts,
            "RETAB emitted different constant payloads than stock",
        )


if __name__ == "__main__":
    unittest.main()
