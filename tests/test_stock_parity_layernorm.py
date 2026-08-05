"""Stock-parity regression: layout through a LayerNorm that ends at the delegate output.

RETAB is meant to reproduce ExecuTorch's ARM TOSA preprocess, so an artifact that
differs from the stock one is a defect even when both are valid. This is the smallest
network that used to diverge: Conv -> LayerNorm -> ReLU.

Both paths now emit the same 206 TOSA ops (reordered only) and the same 42 constants.

What used to differ was the two LayerNorm affine constants. Stock runs the norm body in
NCHW and emits them as ``[1, 8, 16, 16]``; RETAB used to keep NHWC to the delegate
boundary and emit ``[1, 16, 16, 8]``, the payloads being genuine NCHW->NHWC permutations
of each other. Both graphs computed the same thing -- it was a layout *choice* rather
than a numerical error -- but it was still a divergence from stock, and layout choices
feed Vela's scheduling downstream.

The cause was that RETAB annotated *every* rank>=4 tensor channels-last, whereas stock
keeps the delegate interior logical and only builds channels-last islands around ops
whose TOSA form is defined that way (convolution, pooling, resize). Stock merges an
elementwise run into a neighbouring island only when the run is bounded by a matching
permute pair on every edge, and rejects it outright when it reaches the graph output
(``RemovePermutesAroundElementwiseOps.visit``). Here there is no second layout-sensitive
op after the norm, so stock leaves the whole body -- affine constants included -- in
NCHW. ``LayoutPlan`` in ``src/fx_memory_format_ir.rs`` reproduces that rule, which moves
RETAB's NHWC->NCHW transpose from the delegate output back to just after the conv, where
stock puts it.

A note on why the affine *values* matter, worth keeping in mind when reproducing: with
torch's default LayerNorm init (weight=1, bias=0) the quantized gamma/beta are uniform
tensors, and the divergence used to survive all the way through Vela into a different
Ethos-U command stream. With randomized affine params (``CnnLayerNormRandomAffine``) the
TOSA diverged exactly as above, but Vela normalized it away and the compiled artifacts
came out byte-identical. So TOSA parity is the stricter and more useful signal; Ethos-U
parity can mask this class of defect.

Comparison is deliberately name-independent. Tensor names diverge wholesale between the
two paths -- stock names a tensor after the TOSA op that emitted it
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

    def _assert_matches_stock(self, model: nn.Module) -> None:
        (stock_ops, stock_consts), (retab_ops, retab_consts) = self._lower_both(model)
        self.assertEqual(
            Counter(stock_ops),
            Counter(retab_ops),
            "RETAB emitted a different multiset of TOSA opcodes than stock",
        )
        self.assertEqual(
            stock_consts,
            retab_consts,
            "RETAB emitted different constant payloads than stock",
        )

    def test_matches_stock_with_default_affine_init(self) -> None:
        """Uniform gamma/beta: the case whose divergence reached the command stream."""
        self._assert_matches_stock(CnnLayerNormDefaultInit())

    def test_matches_stock_with_random_affine(self) -> None:
        """Non-uniform gamma/beta, so a wrongly permuted payload cannot hash equal.

        With torch's default init the affine tensors quantize to uniform values, and a
        uniform tensor is its own NCHW->NHWC permutation once the shape is ignored. This
        variant is what makes the payload comparison load-bearing.
        """
        self._assert_matches_stock(CnnLayerNormRandomAffine())


if __name__ == "__main__":
    unittest.main()
