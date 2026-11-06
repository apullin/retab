"""Probe: a rank-changing view feeding a 16-bit TABLE whose operand would be
[N,1,1,C]. Runs rc4's InsertTableOps, then the Arm Down pass, then the Up
pass (arm_pass_manager.py:726-727 order), and prints the graph after each."""
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "layout_replay"))
import gen_layout_replay as G  # noqa: E402

import torch  # noqa: E402
from executorch.backends.arm._passes import (  # noqa: E402
    PropagateViewCopyPermuteDownPass,
    PropagateViewCopyPermuteUpPass,
)
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass  # noqa: E402
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec  # noqa: E402
from executorch.backends.arm.tosa.specification import TosaLoweringContext, TosaSpecification  # noqa: E402

logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
A = G.A


def show(title, gm):
    print(f"--- {title}")
    for node in gm.graph.nodes:
        val = node.meta.get("val")
        shape = list(val.shape) if isinstance(val, torch.Tensor) else None
        dtype = val.dtype if isinstance(val, torch.Tensor) else None
        print(f"  {node.op:13} {node.name:34} {str(node.target)[:40]:40} args={[a.name if isinstance(a, torch.fx.Node) else a for a in node.args] if node.op != 'output' else node.args} {dtype} {shape}")


def build(b, n):
    x = b.input("x", (n, 8), dtype=torch.int16)
    v = b.view(x, [n, 1, 1, 8])
    t = G.table_op(b, A.sigmoid.default, v, G.qargs(1 / 1024, dtype=torch.int16), G.qargs(1 / 32768, dtype=torch.int16))
    b.output([t])


spec = "TOSA-1.0+INT+int16"
for n in (2, 1):
    with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
        b = G.Builder()
        build(b, n)
        ep = b.exported_program()
        gm = ep.graph_module
        gm = InsertTableOpsPass(ep)(gm).graph_module
        show(f"N={n}: after InsertTableOps", gm)
        cs = TosaCompileSpec(spec)
        gm = PropagateViewCopyPermuteDownPass(cs, ep)(gm).graph_module
        show(f"N={n}: after Down", gm)
        gm = PropagateViewCopyPermuteUpPass(cs, ep)(gm).graph_module
        show(f"N={n}: after Up", gm)
        tables = [nd for nd in gm.graph.nodes if nd.target == G.T.TABLE.default]
        for t in tables:
            print(f"N={n}: TABLE operand shape after Down+Up: {list(t.args[0].meta['val'].shape)}")
