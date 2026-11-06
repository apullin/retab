"""Run small view/permute graphs through rc4's Arm CanonicalizeViewCopyPermutePass.

A case is (inputs, ops, outputs):
  inputs:  [(label, shape)]
  ops:     [(label, kind, input_label, int_list)]  kind in {"permute", "view", "relu"}
  outputs: [label]
The probe records the graph just before the pass's ExportPass retrace, so the
surviving nodes are the original Node objects and keep their labels.
"""

import copy

import torch
from executorch.backends.arm._passes import CanonicalizeViewCopyPermutePass
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir import pass_base
from executorch.exir.dialects._ops import ops as exir_ops
from torch.utils import _pytree as pytree

VIEW = exir_ops.edge.aten.view_copy.default
PERMUTE = exir_ops.edge.aten.permute_copy.default
RELU = exir_ops.edge.aten.relu.default
KINDS = {"permute": PERMUTE, "view": VIEW, "relu": RELU}
SHORT = {PERMUTE: "permute", VIEW: "view", RELU: "relu"}

_captured = {}
_orig_call = pass_base._ExportPassBase.call


def _spy(self, graph_module):
    if isinstance(self, CanonicalizeViewCopyPermutePass):
        _captured["pre"] = _snapshot(graph_module)
    return _orig_call(self, graph_module)


pass_base._ExportPassBase.call = _spy


def _label(node, labels):
    return labels.get(node.name, "NEW:" + node.name)


def _snapshot(graph_module, labels=None):
    rows = []
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            arg = node.args[1] if len(node.args) > 1 else None
            rows.append(
                (
                    node.name,
                    SHORT.get(node.target, str(node.target)),
                    node.args[0].name,
                    list(arg) if arg is not None else None,
                    list(node.meta["val"].shape),
                )
            )
        elif node.op == "output":
            rows.append(("output", "output", [a.name for a in node.args[0]], None, None))
    return rows


def build(inputs, ops, outputs, seed=0):
    torch.manual_seed(seed)
    builder = GraphBuilder()
    proxies = {}
    data = []
    for label, shape in inputs:
        tensor = torch.randn(tuple(shape))
        data.append(tensor)
        proxies[label] = builder.placeholder(label, tensor)
    for label, kind, src, arg in ops:
        args = (proxies[src], list(arg)) if arg is not None else (proxies[src],)
        proxies[label] = builder.call_operator(op=KINDS[kind], args=args)
    builder.output([proxies[label] for label in outputs])
    gm = builder.get_graph_module()
    labels = {proxy.node.name: label for label, proxy in proxies.items()}
    return gm, labels, tuple(data)


def run(inputs, ops, outputs, check_numerics=True):
    gm, labels, data = build(inputs, ops, outputs)
    before = copy.deepcopy(gm)
    _captured.clear()
    result = CanonicalizeViewCopyPermutePass().call(gm)
    pre = _captured.get("pre")
    if pre is None:
        pre = _snapshot(result.graph_module)
    post = _snapshot(result.graph_module)
    if check_numerics:
        before.eval()
        result.graph_module.eval()
        with torch.no_grad():
            a = pytree.tree_flatten(before(*data))[0]
            b = pytree.tree_flatten(result.graph_module(*data))[0]
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
    survivors = []
    out = None
    for name, kind, src, arg, shape in pre:
        if kind == "output":
            out = [labels.get(n, "NEW:" + n) for n in src]
        else:
            survivors.append(
                (labels.get(name, "NEW:" + name), kind, labels.get(src, "NEW:" + src), arg, shape)
            )
    post_ops = [(kind, arg, shape) for _, kind, _, arg, shape in post if kind != "output"]
    pre_ops = [(kind, arg, shape) for _, kind, arg_src, arg, shape in [(r[0], r[1], r[2], r[3], r[4]) for r in pre] if kind != "output"]
    assert pre_ops == post_ops, (pre_ops, post_ops)
    return result.modified, survivors, out


def show(name, inputs, ops, outputs, check_numerics=True):
    modified, survivors, out = run(inputs, ops, outputs, check_numerics)
    print(f"== {name}: modified={modified}")
    for label, kind, src, arg, shape in survivors:
        print(f"   {label} = {kind}({src}, {arg}) -> {shape}")
    print(f"   output: {out}")
    return modified, survivors, out
