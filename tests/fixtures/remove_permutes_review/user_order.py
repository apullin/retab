"""Evidence for remove_permutes_review_tests.rs::a_rewired_user_order_decides_which_region_is_rewritten.

rc4's RemovePermutes iterates node.users in FX insertion order. After a
rewire (as rc4's propagation passes do between fusion calls, without a
retrace) that order differs from graph order, and it decides which of two
regions sharing a boundary permute is rewritten.

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/remove_permutes_review/user_order.py
"""
import logging
import torch
from types import SimpleNamespace
from torch._subclasses.fake_tensor import FakeTensorMode
from executorch.backends.arm._passes import RemovePermutesAroundElementwiseTosaOps
from executorch.backends.arm.tosa.specification import TosaLoweringContext, TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
A = exir_ops.edge.aten
EP = SimpleNamespace(graph_signature=SimpleNamespace(inputs_to_parameters={}, inputs_to_buffers={}, inputs_to_lifted_tensor_constants={}))
mode = FakeTensorMode()

def demo(rewire):
    g = torch.fx.Graph()
    def call(target, *args):
        node = g.create_node("call_function", target, args)
        vals = torch.fx.node.map_arg(args, lambda n: n.meta["val"])
        with mode:
            node.meta["val"] = target(*vals)
        return node
    x = g.placeholder("x"); y = g.placeholder("y")
    with mode:
        x.meta["val"] = torch.empty(4, 4); y.meta["val"] = torch.empty(4, 4)
    p = call(A.permute_copy.default, x, [1, 0])          # P (an involution)
    u1 = call(A.relu.default, y if rewire else p)        # U1
    e1 = call(A.permute_copy.default, u1, [1, 0])        # E1: U1's exit, and an entry of U2
    u2 = call(A.add.Tensor, p, e1)                       # U2
    e2 = call(A.permute_copy.default, u2, [1, 0])        # E2: U2's exit
    g.output((e2,))
    if rewire:
        u1.replace_input_with(y, p)
    order = [n.name for n in g.nodes]
    users = [n.name for n in p.users]
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        result = RemovePermutesAroundElementwiseTosaOps(EP).call(gm)
    permutes = [n for n in result.graph_module.graph.nodes if n.target == A.permute_copy.default]
    print(f"rewire={rewire}: graph order {order}; P.users (FX order) {users}")
    print(f"  modified={result.modified}, {len(permutes)} permute(s) left:")
    print("  " + str(result.graph_module.graph).replace("\n", "\n  "))

demo(False)
demo(True)
