"""Prints rc4's graph after InsertTableOps, the Arm Down pass and the Arm Up
pass for probe_t16_extra_transpose.py's minimal case."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import probe_lib as P  # noqa: E402
import torch  # noqa: E402
from executorch.backends.arm._passes import PropagateViewCopyPermuteDownPass, PropagateViewCopyPermuteUpPass
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import TosaLoweringContext, TosaSpecification

G = P.G
A = G.A
Q1 = (G.qargs(1 / 256, dtype=torch.int16), G.qargs(1 / 4096, dtype=torch.int16))
Q2 = (G.qargs(1 / 1024, dtype=torch.int16), G.qargs(1 / 32768, dtype=torch.int16))
Q3 = (G.qargs(1 / 2048, dtype=torch.int16), G.qargs(1 / 2048, dtype=torch.int16))


def show(title, gm):
    print(f"--- {title}")
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name.startswith("b_"):
            continue
        val = node.meta.get("val")
        shape = list(val.shape) if isinstance(val, torch.Tensor) else ""
        args = [a.name if isinstance(a, torch.fx.Node) else a for a in node.args] if node.op != "output" else node.args
        tgt = str(node.target).split(".")[-2] if node.op == "call_function" else node.op
        if node.op == "call_function" and "tosa" in str(node.target):
            tgt = str(node.target).split(":")[0].replace("<EdgeOpOverload: ", "")
        print(f"  {node.name:28} {tgt:22} {args} {shape}")


with TosaLoweringContext(TosaSpecification.create_from_string(G.PROPAGATE_TOSA_SPEC)):
    b = G.Builder()
    x1 = b.input("x1", (2, 1, 3), dtype=torch.int16)
    p9 = b.permute(x1, [1, 2, 0])
    p15 = b.permute(p9, [0, 2, 1])
    t16 = G.table_op(b, A.tanh.default, p15, *Q1)
    p20 = b.permute(t16, [1, 2, 0])
    s21 = G.table_op(b, A.sigmoid.default, p20, *Q2)
    v22 = b.view(p9, [1, 6])
    s23 = G.table_op(b, A.sigmoid.default, v22, *Q3)
    b.output([s21, s23])
    ep = b.exported_program()
    gm = InsertTableOpsPass(ep)(ep.graph_module).graph_module
    show("after InsertTableOps", gm)
    cs = TosaCompileSpec(G.PROPAGATE_TOSA_SPEC)
    gm = PropagateViewCopyPermuteDownPass(cs, ep)(gm).graph_module
    show("after Down", gm)
    gm = PropagateViewCopyPermuteUpPass(cs, ep)(gm).graph_module
    show("after Up", gm)
