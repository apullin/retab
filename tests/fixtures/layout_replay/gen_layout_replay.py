"""Record rc4 layout passes on hand-built edge graphs, for src/layout/replay.rs.

Each fixture holds cases for one ExecuTorch v1.6.0-rc4 pass: an edge-dialect
torch.fx graph ("before") and the graph the real rc4 pass returns ("after").
`replay::check` in src/layout/replay.rs rebuilds every "before" graph as an
IrGraph, runs the RETAB port and compares the result with "after".

Regenerate with the rc4 environment, from a neutral directory so a source
checkout can't shadow the installed package:

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py

Options: `--pass NAME` writes one fixture, `--random N` and `--seed S` change
the random cases, `--out DIR` writes elsewhere, `--case NAME` prints one case.
The output is stable for a given torch, ExecuTorch and CPython `random`
(written with torch 2.15.0, ExecuTorch 1.6.0 and CPython 3.12).

How the graphs are built and run:

- Nodes are created with `Graph.create_node` on `exir_ops.edge` (and
  `exir_ops.backend.tosa`) targets. Each `meta["val"]` is the FakeTensor the
  target returns on its operands' vals under one `FakeTensorMode`, so shapes,
  dtypes and strides (dim orders) are torch's own. Placeholders get
  `torch.empty_strided` vals, which is how channels-last inputs and outputs
  appear.
- Every case is wrapped in a real `ExportedProgram`: user inputs are
  USER_INPUT specs; `kind=param|buffer|constant` placeholders are PARAMETER,
  BUFFER and CONSTANT_TENSOR specs backed by real tensors in `state_dict` or
  `constants`. That is all `is_param_node` reads (`graph_signature.inputs_to_*`).
  The verifier is the stock torch.export one with EdgeOpOverload allowed.
- Passes are constructed as `ArmPassManager._tosa_pipeline` constructs them
  (`arm_pass_manager.py:544,664,725,733,742`) and called the way
  `_ExportedProgramGraphPassAdapter.call` calls them,
  `graph_pass(exported_program.graph_module)`, inside the
  `TosaLoweringContext` the pass manager provides. No DCE runs afterwards:
  dead nodes rc4 leaves are recorded and marked `dead`.

Fixture format (one node per line, in graph order):

    case <name>
    before
    %0 = placeholder x : torch.float32[1, 3, 4, 5] dim_order=[0,2,3,1] q=none
    %1 = aten.relu.default(%0) : torch.float32[1, 3, 4, 5] dim_order=[0,2,3,1] q=out{0}
    output([%1])
    after modified=true
    ...
    end

`%N` is the node's position in its graph (placeholders included). Targets use
RETAB's spelling (`normalize_fx_target` in src/fx_ir.rs). Args: `%N`, ints,
floats (Python repr), `True`/`False`, `None`, `"str"`, dtypes (`torch.int8`)
and `[...]` for lists and tuples; kwargs follow as `name=value`. The value is
`dtype[shape]`. Annotations: `dim_order=` (only when not the identity),
`kind=param|buffer|constant` (constant placeholders), `q=` (`none`, or
`in{keys}` / `out{keys}` for present `input_qparams` / `output_qparams` maps,
`{}` when empty) and `dead` (a call node the output doesn't reach).
`after modified=<bool> same` means the after graph prints exactly like before.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import executorch
import executorch.backends.arm.tosa.dialect  # noqa: F401  registers exir_ops.backend.tosa
import executorch.version
import torch
from executorch.backends.arm._passes import (
    CanonicalizeViewCopyPermutePass,
    FuseViewCopyTransformPass,
    MatchArgRanksPass,
    NormalizeDelegateIOLayoutPass,
    RewriteHighRankSingletonPermutePass,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch._export.verifier import Verifier
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.fx.node import map_arg

A = exir_ops.edge.aten
Q = exir_ops.edge.quantized_decomposed
D = exir_ops.edge.dim_order_ops
T = exir_ops.backend.tosa

SEED = 135
RANDOM_CASES = 300
TOSA_SPEC = "TOSA-1.0+FP+INT"
QARGS = QuantArgs(scale=0.5, zp=0, qmin=-128, qmax=127, dtype=torch.int8)


class ReplayVerifier(Verifier):
    """The stock torch.export verifier, with edge-dialect targets allowed."""

    dialect = "RETAB_LAYOUT_REPLAY"

    def allowed_op_types(self):
        return (OpOverload, HigherOrderOperator, EdgeOpOverload)


# --------------------------------------------------------------------------
# Graph building
# --------------------------------------------------------------------------


def contiguous_strides(shape: Sequence[int], dim_order: Sequence[int]) -> list[int]:
    strides = [0] * len(shape)
    step = 1
    for dim in reversed(dim_order):
        strides[dim] = step
        step *= max(shape[dim], 1)
    return strides


def channels_last_order(rank: int) -> list[int]:
    return [0, *range(2, rank), 1]


class Builder:
    """An edge graph whose nodes carry the FakeTensor their target computes."""

    def __init__(self) -> None:
        self.mode = FakeTensorMode()
        self.graph = torch.fx.Graph()
        self.input_specs: list[InputSpec] = []
        self.state_dict: dict[str, torch.Tensor] = {}
        self.constants: dict[str, torch.Tensor] = {}
        self.first_call: torch.fx.Node | None = None
        self.output_node: torch.fx.Node | None = None

    def _placeholder(self, name, shape, dtype, dim_order) -> torch.fx.Node:
        order = list(range(len(shape))) if dim_order is None else list(dim_order)
        strides = contiguous_strides(shape, order)
        with self.mode:
            val = torch.empty_strided(list(shape), strides, dtype=dtype)
        # Placeholders stay ahead of every call, as in an exported graph.
        if self.first_call is None:
            node = self.graph.placeholder(name)
        else:
            with self.graph.inserting_before(self.first_call):
                node = self.graph.placeholder(name)
        node.meta["val"] = val
        return node

    def input(self, name, shape, dtype=torch.float32, dim_order=None) -> torch.fx.Node:
        node = self._placeholder(name, shape, dtype, dim_order)
        self.input_specs.append(
            InputSpec(InputKind.USER_INPUT, TensorArgument(node.name), None)
        )
        return node

    def constant(
        self, kind, name, shape, dtype=torch.float32, dim_order=None
    ) -> torch.fx.Node:
        node = self._placeholder(name, shape, dtype, dim_order)
        real = torch.empty_strided(
            list(shape),
            contiguous_strides(shape, list(dim_order or range(len(shape)))),
            dtype=dtype,
        ).zero_()
        target = f"c_{node.name}"
        if kind == "param":
            self.state_dict[target] = torch.nn.Parameter(real, requires_grad=False)
            spec = InputSpec(InputKind.PARAMETER, TensorArgument(node.name), target)
        elif kind == "buffer":
            self.state_dict[target] = real
            spec = InputSpec(
                InputKind.BUFFER, TensorArgument(node.name), target, persistent=True
            )
        elif kind == "constant":
            self.constants[target] = real
            spec = InputSpec(
                InputKind.CONSTANT_TENSOR, TensorArgument(node.name), target
            )
        else:
            raise ValueError(kind)
        self.input_specs.append(spec)
        return node

    def call(self, target, *args, **kwargs) -> torch.fx.Node:
        fake_args, fake_kwargs = map_arg((args, kwargs), lambda n: n.meta["val"])
        with self.mode:
            val = target(*fake_args, **fake_kwargs)
        node = self.graph.create_node("call_function", target, args, kwargs)
        node.meta["val"] = val
        if self.first_call is None:
            self.first_call = node
        return node

    # Shorthands for the common layout ops.
    def view(self, x, shape) -> torch.fx.Node:
        return self.call(A.view_copy.default, x, list(shape))

    def permute(self, x, dims) -> torch.fx.Node:
        return self.call(A.permute_copy.default, x, list(dims))

    def output(self, results, container=tuple) -> None:
        self.output_node = self.graph.output(container(results))

    def exported_program(self) -> ExportedProgram:
        """Wrap the graph in an ExportedProgram. Output leaves that aren't
        nodes (None, nested containers) are hidden from the signature check."""
        assert self.output_node is not None
        gm = torch.fx.GraphModule(torch.nn.Module(), self.graph)
        results = self.output_node.args[0]
        leaves = [
            n
            for n in torch.utils._pytree.tree_leaves(results)
            if isinstance(n, torch.fx.Node)
        ]
        flat = isinstance(results, (tuple, list)) and all(
            isinstance(r, torch.fx.Node) for r in results
        )
        if not flat:
            self.output_node.args = (tuple(leaves),)
        signature = ExportGraphSignature(
            input_specs=self.input_specs,
            output_specs=[
                OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(n.name), None)
                for n in leaves
            ],
        )
        ep = ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=signature,
            state_dict=self.state_dict,
            range_constraints={},
            module_call_graph=[],
            constants=self.constants,
            verifiers=[ReplayVerifier],
        )
        if not flat:
            ep.graph_module.graph.output_node().args = (results,)
        return ep


def set_qparams(node, inputs=None, outputs=None) -> None:
    if inputs is not None:
        node.meta["input_qparams"] = {i: QARGS for i in inputs}
    if outputs is not None:
        node.meta["output_qparams"] = {i: QARGS for i in outputs}


# --------------------------------------------------------------------------
# Dumping
# --------------------------------------------------------------------------


def retab_target(target) -> str:
    """RETAB's `normalize_fx_target` (src/fx_ir.rs) applied to `str(target)`."""
    raw = str(target)
    bare = raw.split(": schema")[0].strip()
    if bare.startswith("<") and bare.endswith(">") and ": " in bare:
        return bare[1:-1].split(": ", 1)[1].strip()
    return bare


def fmt_arg(arg, pos) -> str:
    if isinstance(arg, torch.fx.Node):
        return f"%{pos[arg]}"
    if isinstance(arg, bool) or arg is None:
        return str(arg)
    if isinstance(arg, int):
        return str(arg)
    if isinstance(arg, float):
        return repr(arg)
    if isinstance(arg, torch.dtype):
        return str(arg)
    if isinstance(arg, str):
        assert '"' not in arg and "\\" not in arg, arg
        return f'"{arg}"'
    if isinstance(arg, (list, tuple)):
        return "[" + ", ".join(fmt_arg(a, pos) for a in arg) + "]"
    raise TypeError(f"cannot print arg {arg!r} ({type(arg).__name__})")


def fmt_val(val) -> str:
    if not isinstance(val, torch.Tensor):
        raise TypeError(f"expected a tensor val, got {val!r}")
    return f"{val.dtype}[{', '.join(str(int(d)) for d in val.shape)}]"


def fmt_qparams(meta) -> str:
    parts = []
    for key, label in (("input_qparams", "in"), ("output_qparams", "out")):
        if key in meta:
            keys = ",".join(str(k) for k in sorted(meta[key]))
            parts.append(f"{label}{{{keys}}}")
    return "/".join(parts) if parts else "none"


def placeholder_kinds(ep: ExportedProgram) -> dict[str, str]:
    signature = ep.graph_signature
    kinds = {name: "param" for name in signature.inputs_to_parameters}
    kinds.update({name: "buffer" for name in signature.inputs_to_buffers})
    kinds.update(
        {name: "constant" for name in signature.inputs_to_lifted_tensor_constants}
    )
    return kinds


def dump(gm: torch.fx.GraphModule, kinds: dict[str, str]) -> list[str]:
    nodes = list(gm.graph.nodes)
    pos = {node: index for index, node in enumerate(nodes)}
    output = gm.graph.output_node()
    if nodes[-1] is not output:
        raise AssertionError("the output node must be last")
    reached = set()
    stack = [output]
    while stack:
        node = stack.pop()
        if node not in reached:
            reached.add(node)
            stack.extend(node.all_input_nodes)

    lines = []
    for node in nodes:
        if node.op == "output":
            lines.append(f"output({fmt_arg(node.args[0], pos)})")
            continue
        if node.op == "placeholder":
            head = f"placeholder {node.name}"
        elif node.op == "call_function":
            args = [fmt_arg(a, pos) for a in node.args]
            args += [f"{k}={fmt_arg(v, pos)}" for k, v in node.kwargs.items()]
            head = f"{retab_target(node.target)}({', '.join(args)})"
        else:
            raise AssertionError(f"unexpected node op {node.op}")
        val = node.meta["val"]
        notes = []
        dim_order = list(val.dim_order())
        if dim_order != list(range(len(dim_order))):
            notes.append(f"dim_order=[{','.join(map(str, dim_order))}]")
        if node.op == "placeholder" and node.name in kinds:
            notes.append(f"kind={kinds[node.name]}")
        notes.append(f"q={fmt_qparams(node.meta)}")
        if node.op == "call_function" and node not in reached:
            notes.append("dead")
        lines.append(f"%{pos[node]} = {head} : {fmt_val(val)} {' '.join(notes)}")
    return lines


# --------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------


class PassSpec:
    def __init__(self, name, make, sources, structured, random_profile):
        self.name = name
        self.make: Callable[[ExportedProgram], object] = make
        self.sources = sources
        self.structured = structured
        self.random_profile = random_profile


class BuildError(Exception):
    """torch rejected a random graph while it was being built."""


def run_case(spec: PassSpec, build: Callable[[Builder], None]) -> list[str]:
    # TOSA dialect ops need the context to build, and the pass manager runs
    # every pass inside it (`ArmPassManager._tosa_context`).
    with TosaLoweringContext(TosaSpecification.create_from_string(TOSA_SPEC)):
        builder = Builder()
        try:
            build(builder)
            ep = builder.exported_program()
        except Exception as error:
            raise BuildError(f"{type(error).__name__}: {error}") from error
        kinds = placeholder_kinds(ep)
        before = dump(ep.graph_module, kinds)
        graph_pass = spec.make(ep)
        result = graph_pass(ep.graph_module)
    after = dump(result.graph_module, kinds)
    lines = ["before", *before, f"after modified={str(bool(result.modified)).lower()}"]
    if after == before:
        lines[-1] += " same"
    else:
        lines += after
    return lines


# --------------------------------------------------------------------------
# Structured cases
# --------------------------------------------------------------------------

NCHW = (1, 3, 4, 5)


def normalize_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    @case
    def channels_last_input_and_output(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        b.output([b.call(A.relu.default, x)])

    @case
    def channels_last_input_with_qparams(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        set_qparams(x, outputs=[0])
        y = b.input("y", NCHW)
        set_qparams(y, inputs=[0], outputs=[0])
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def contiguous_inputs_are_left_alone(b):
        x = b.input("x", (2, 3, 4, 5))
        b.output([b.call(A.relu.default, x)])

    @case
    def two_channels_last_inputs(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        y = b.input("y", NCHW, dim_order=[0, 2, 3, 1])
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def channels_last_3d_input(b):
        x = b.input("x", (1, 2, 3, 4, 5), dim_order=[0, 2, 3, 4, 1])
        b.output([b.call(A.neg.default, x)])

    @case
    def rank3_custom_dim_order(b):
        x = b.input("x", (2, 3, 4), dim_order=[0, 2, 1])
        b.output([b.call(A.abs.default, x)])

    @case
    def rank2_transposed_input(b):
        x = b.input("x", (3, 4), dim_order=[1, 0])
        b.output([b.call(A.relu.default, x)])

    @case
    def rank6_permuted_input(b):
        x = b.input("x", (2, 3, 2, 1, 2, 3), dim_order=[5, 0, 2, 4, 1, 3])
        b.output([b.call(A.sigmoid.default, x)])

    @case
    def singleton_dims_make_the_dim_order_ambiguous(b):
        x = b.input("x", (1, 3, 1, 1), dim_order=[0, 2, 3, 1])
        y = b.input("y", (1, 1, 4, 1), dim_order=[0, 2, 3, 1])
        b.output([b.call(A.relu.default, x), b.call(A.neg.default, y)])

    @case
    def unused_channels_last_input(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        y = b.input("y", (2, 3))
        del x
        b.output([b.call(A.relu.default, y)])

    @case
    def channels_last_input_returned_directly(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        b.output([x, b.call(A.relu.default, x)])

    @case
    def channels_last_input_used_twice_by_one_user(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        b.output([b.call(A.mul.Tensor, x, x)])

    @case
    def channels_last_input_in_a_list_arg(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        y = b.input("y", NCHW)
        b.output([b.call(A.cat.default, [y, x, y], 1)])

    @case
    def channels_last_int8_and_bool_inputs(b):
        x = b.input("x", NCHW, dtype=torch.int8, dim_order=[0, 2, 3, 1])
        m = b.input("m", NCHW, dtype=torch.bool, dim_order=[0, 2, 3, 1])
        y = b.input("y", NCHW, dtype=torch.int8)
        b.output([b.call(A.where.self, m, x, y)])

    for kind in ("param", "buffer", "constant"):

        def channels_last_constant_is_skipped(b, kind=kind):
            x = b.input("x", NCHW)
            w = b.constant(kind, "w", NCHW, dim_order=[0, 2, 3, 1])
            b.output([b.call(A.add.Tensor, x, w)])

        cases[f"channels_last_{kind}_is_skipped"] = channels_last_constant_is_skipped

    @case
    def channels_last_param_returned_directly(b):
        w = b.constant("param", "w", NCHW, dim_order=[0, 2, 3, 1])
        x = b.input("x", (2, 3))
        b.output([b.call(A.relu.default, x), w])

    @case
    def output_returned_twice(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        r = b.call(A.relu.default, x)
        b.output([r, r])

    @case
    def mixed_outputs(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        y = b.input("y", (2, 3, 4))
        r = b.call(A.relu.default, x)
        n = b.call(A.neg.default, y)
        b.output([r, n, r, y])

    @case
    def output_list_container(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        b.output([b.call(A.relu.default, x), b.call(A.abs.default, x)], container=list)

    @case
    def nested_output_containers(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        r = b.call(A.relu.default, x)
        n = b.call(A.neg.default, x)
        b.output([r, (n, [r])])

    @case
    def output_with_none(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        b.output([b.call(A.relu.default, x), None])

    @case
    def channels_last_output_with_qparams(b):
        x = b.input("x", NCHW)
        c = b.call(D._to_dim_order_copy.default, x, dim_order=[0, 2, 3, 1])
        set_qparams(c, inputs=[0], outputs=[0])
        e = b.call(A.relu.default, x)
        set_qparams(e, inputs=[], outputs=[])
        b.output([c, e])

    @case
    def channels_last_output_from_a_contiguous_input(b):
        x = b.input("x", (2, 3, 4, 5))
        b.output([b.call(D._to_dim_order_copy.default, x, dim_order=[0, 2, 3, 1])])

    @case
    def empty_qparam_maps_are_copied(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        set_qparams(x, inputs=[], outputs=[])
        r = b.call(A.relu.default, x)
        set_qparams(r, inputs=[0], outputs=[])
        b.output([r])

    @case
    def boundary_reshape_is_a_strided_view(b):
        # `input_fake.reshape(boundary_shape)` is a view here, so rc4's
        # placeholder keeps a non-identity dim order.
        x = b.input("x", (5, 2, 2, 1), dim_order=[0, 2, 3, 1])
        b.output([b.call(A.relu.default, x)])

    @case
    def equal_sized_dims_keep_their_strides(b):
        x = b.input("x", (5, 5, 5), dim_order=[0, 2, 1])
        b.output([b.call(A.relu.default, x)])

    @case
    def rank0_and_rank1_inputs(b):
        s = b.input("s", ())
        v = b.input("v", (4,))
        b.output([b.call(A.add.Tensor, v, s)])

    return cases


def match_arg_ranks_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    @case
    def where_docstring_example(b):
        a = b.input("a", (4, 3, 2), dtype=torch.bool)
        x = b.input("x", (2,))
        y = b.input("y", (3, 1))
        b.output([b.call(A.where.self, a, x, y)])

    @case
    def add_with_a_rank1_operand(b):
        x = b.input("x", (1, 2, 3, 4))
        y = b.input("y", (4,))
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def rank0_operand(b):
        x = b.input("x", (2, 3))
        s = b.input("s", ())
        b.output([b.call(A.mul.Tensor, x, s)])

    @case
    def rank6_and_rank0(b):
        x = b.input("x", (1, 2, 1, 3, 1, 2))
        s = b.input("s", ())
        b.output([b.call(A.sub.Tensor, s, x)])

    @case
    def equal_ranks_are_left_alone(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (1, 3))
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def no_target_op(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (3,))
        b.output([b.call(A.cat.default, [b.call(A.relu.default, x), x], 0), y])

    @case
    def scalar_operand_is_not_a_node(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (3,))
        b.output([b.call(A.mul.Tensor, x, 2.0), b.call(A.add.Tensor, x, y)])

    @case
    def repeated_low_rank_arg(b):
        c = b.input("c", (2, 3), dtype=torch.bool)
        x = b.input("x", (3,))
        b.output([b.call(A.where.self, c, x, x)])

    @case
    def repeated_low_rank_arg_three_times(b):
        c = b.input("c", (3,), dtype=torch.bool)
        y = b.input("y", (2, 1, 3))
        w = b.call(A.where.self, c, c, y)
        b.output([b.call(A.where.self, c, y, y), w])

    @case
    def low_rank_bool_condition(b):
        c = b.input("c", (2,), dtype=torch.bool)
        x = b.input("x", (3, 2))
        y = b.input("y", (3, 2))
        b.output([b.call(A.where.self, c, x, y)])

    @case
    def kwarg_operand_is_ignored(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (4,))
        b.output([b.call(A.add.Tensor, x, other=y)])

    @case
    def positional_view_also_replaces_the_kwarg_use(b):
        c = b.input("c", (2, 3), dtype=torch.bool)
        x = b.input("x", (3,))
        b.output([b.call(A.where.self, c, x, other=x)])

    @case
    def two_targets_share_a_low_rank_arg(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (4,))
        b.output([b.call(A.add.Tensor, x, y), b.call(A.mul.Tensor, y, x)])

    @case
    def chain_of_targets(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (4,))
        z = b.input("z", (3, 1))
        a = b.call(A.add.Tensor, x, y)
        b.output([b.call(A.mul.Tensor, z, a)])

    @case
    def qparams_on_the_low_rank_arg(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (4,))
        set_qparams(y, inputs=[0], outputs=[0])
        a = b.call(A.add.Tensor, x, y)
        set_qparams(a, inputs=[0, 1], outputs=[0])
        b.output([a])

    @case
    def channels_last_operands(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        y = b.input("y", (5,))
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def non_target_binary_ops_keep_their_ranks(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (4,))
        m = b.input("m", (2, 3, 4), dtype=torch.bool)
        n = b.input("n", (4,), dtype=torch.bool)
        b.output(
            [
                b.call(A.atan2.default, x, y),
                b.call(A.fmod.Tensor, y, x),
                b.call(A.logical_and.default, m, n),
                b.call(A.add.Tensor, x, x),
            ]
        )

    float_targets = [
        A.add.Tensor,
        A.sub.Tensor,
        A.mul.Tensor,
        A.div.Tensor,
        A.pow.Tensor_Tensor,
        A.remainder.Tensor,
        A.maximum.default,
        A.minimum.default,
        A.eq.Tensor,
        A.gt.Tensor,
        A.ge.Tensor,
        A.lt.Tensor,
        A.le.Tensor,
        A.ne.Tensor,
    ]
    int_targets = [
        A.bitwise_right_shift.Tensor,
        A.bitwise_left_shift.Tensor,
        A.bitwise_and.Tensor,
        A.bitwise_xor.Tensor,
        A.bitwise_or.Tensor,
    ]
    for target in float_targets + int_targets:
        dtype = torch.int32 if target in int_targets else torch.float32

        def each_target(b, target=target, dtype=dtype):
            x = b.input("x", (2, 3, 4), dtype=dtype)
            y = b.input("y", (3, 1), dtype=dtype)
            b.output([b.call(target, y, x)])

        cases[f"target_{retab_target(target)}"] = each_target

    @case
    def target_aten_div_Tensor_mode(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (4,))
        b.output([b.call(A.div.Tensor_mode, x, y, rounding_mode="floor")])

    tosa_targets = {
        "ADD": (torch.float32, {}),
        "SUB": (torch.int32, {}),
        "MUL": (torch.int32, {"shift": 0}),
        "MAXIMUM": (torch.float32, {}),
        "MINIMUM": (torch.int32, {}),
        "POW": (torch.float32, {}),
        "EQUAL": (torch.float32, {}),
        "GREATER": (torch.int32, {}),
        "GREATER_EQUAL": (torch.float32, {}),
        "BITWISE_AND": (torch.int32, {}),
        "BITWISE_OR": (torch.int32, {}),
        "BITWISE_XOR": (torch.int32, {}),
        "LOGICAL_AND": (torch.bool, {}),
        "LOGICAL_OR": (torch.bool, {}),
        "LOGICAL_XOR": (torch.bool, {}),
        "LOGICAL_LEFT_SHIFT": (torch.int32, {}),
        "ARITHMETIC_RIGHT_SHIFT": (torch.int32, {"round": False}),
    }
    for name, (dtype, kwargs) in tosa_targets.items():

        def each_tosa_target(b, name=name, dtype=dtype, kwargs=kwargs):
            x = b.input("x", (2, 3, 4), dtype=dtype)
            y = b.input("y", (4,), dtype=dtype)
            target = getattr(T, name).default
            b.output([b.call(target, x, y, **kwargs)])

        cases[f"target_tosa_{name}"] = each_tosa_target

    return cases


def high_rank_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    @case
    def rank5_with_a_singleton(b):
        x = b.input("x", (1, 2, 3, 4, 5))
        b.output([b.permute(x, [0, 4, 1, 2, 3])])

    @case
    def identity_reduced_permutation_leaves_two_views(b):
        x = b.input("x", (2, 1, 3, 1, 4))
        b.output([b.permute(x, [1, 0, 3, 2, 4])])

    @case
    def negative_dims(b):
        x = b.input("x", (1, 2, 3, 4, 5))
        b.output([b.permute(x, [0, -1, 1, -3, 3])])

    @case
    def rank4_is_left_alone(b):
        x = b.input("x", (1, 2, 3, 4))
        b.output([b.permute(x, [0, 3, 1, 2])])

    @case
    def no_singleton_is_left_alone(b):
        x = b.input("x", (2, 3, 2, 3, 2))
        b.output([b.permute(x, [0, 4, 1, 2, 3])])

    @case
    def reduced_rank_above_4_is_left_alone(b):
        x = b.input("x", (1, 2, 3, 2, 3, 2))
        b.output([b.permute(x, [0, 5, 1, 2, 3, 4])])

    @case
    def all_singleton_input(b):
        x = b.input("x", (1, 1, 1, 1, 1))
        b.output([b.permute(x, [4, 3, 2, 1, 0])])

    @case
    def rank6_with_two_singletons(b):
        x = b.input("x", (2, 1, 3, 1, 4, 5))
        b.output([b.permute(x, [5, 1, 0, 3, 4, 2])])

    @case
    def rank6_reduced_to_rank1(b):
        x = b.input("x", (1, 1, 7, 1, 1, 1))
        b.output([b.permute(x, [5, 2, 4, 3, 1, 0])])

    @case
    def aten_permute_default_target(b):
        x = b.input("x", (1, 2, 3, 4, 5))
        b.output([b.call(A.permute.default, x, [0, 4, 1, 2, 3])])

    @case
    def dims_as_a_kwarg_are_left_alone(b):
        x = b.input("x", (1, 2, 3, 4, 5))
        b.output([b.call(A.permute_copy.default, x, dims=[0, 4, 1, 2, 3])])

    @case
    def qparams_reach_every_new_node(b):
        x = b.input("x", (1, 2, 3, 4, 5), dtype=torch.int8)
        p = b.permute(x, [0, 4, 1, 2, 3])
        set_qparams(p, inputs=[0], outputs=[0])
        q = b.permute(x, [0, 1, 2, 4, 3])
        set_qparams(q, inputs=[], outputs=[])
        b.output([p, q])

    @case
    def chained_permutes(b):
        x = b.input("x", (1, 2, 3, 4, 5))
        p = b.permute(x, [0, 4, 1, 2, 3])
        b.output([b.permute(p, [4, 3, 2, 1, 0])])

    @case
    def dead_permute_is_rewritten(b):
        x = b.input("x", (1, 2, 3, 4, 5))
        b.permute(x, [0, 4, 1, 2, 3])
        b.output([b.call(A.relu.default, x)])

    @case
    def no_permute(b):
        x = b.input("x", (1, 2, 3, 4, 5))
        b.output([b.view(x, [2, 3, 4, 5])])

    @case
    def permute_between_unary_ops_and_returned_twice(b):
        x = b.input("x", (2, 1, 3, 1, 4))
        r = b.call(A.relu.default, x)
        p = b.permute(r, [4, 1, 0, 3, 2])
        n = b.call(A.neg.default, p)
        b.output([n, p, p])

    @case
    def channels_last_input_permute(b):
        x = b.input("x", (1, 2, 1, 3, 4), dim_order=[0, 2, 3, 4, 1])
        b.output([b.permute(x, [0, 2, 3, 4, 1])])

    return cases


UNARY_OPS = [
    (A.alias_copy.default, {}),
    (A.clone.default, {}),
    (D._clone_dim_order.default, {}),
    (A._to_copy.default, {"dtype": torch.float16}),
    (D._to_dim_order_copy.default, {"dtype": torch.float16}),
    (A.abs.default, {}),
    (A.clamp.default, {"min": -1.0, "max": 6.0}),
    (A.ceil.default, {}),
    (A.floor.default, {}),
    (A.neg.default, {}),
    (A.relu.default, {}),
    (A.round.default, {}),
    (A.sigmoid.default, {}),
    (A.silu.default, {}),
    (A.sqrt.default, {}),
    (A.tanh.default, {}),
    (A.sign.default, {}),
    (A.reciprocal.default, {}),
    (A.gelu.default, {"approximate": "tanh"}),
    (A.rsqrt.default, {}),
    (A.exp.default, {}),
    (A.log.default, {}),
]


def unary(b, op_kwargs, x):
    op, kwargs = op_kwargs
    if op is A.clamp.default:
        return b.call(op, x, kwargs["min"], kwargs["max"])
    return b.call(op, x, **kwargs)


def fuse_view_copy_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    @case
    def docstring_example(b):
        x = b.input("x", (2, 3, 4))
        v0 = b.view(x, [6, 4])
        v1 = b.view(v0, [24])
        a = b.call(A.abs.default, v1)
        v2 = b.view(a, [4, 6])
        s = b.call(A.sqrt.default, v2)
        v3 = b.view(s, [2, 12])
        b.output([b.call(A.neg.default, v3), b.call(A.relu.default, v3)])

    @case
    def a_fork_stops_the_chain(b):
        x = b.input("x", (2, 3))
        v0 = b.view(x, [6])
        r = b.call(A.relu.default, v0)
        n = b.call(A.neg.default, v0)
        b.output([b.view(r, [3, 2]), n])

    @case
    def a_view_with_several_users_is_still_bypassed(b):
        x = b.input("x", (2, 3))
        v0 = b.view(x, [6])
        r = b.call(A.relu.default, v0)
        v1 = b.view(r, [3, 2])
        b.output([b.call(A.neg.default, v1), b.call(A.abs.default, v1)])

    @case
    def unary_ops_after_the_last_view_keep_their_shape(b):
        x = b.input("x", (2, 3))
        v1 = b.view(b.view(x, [6]), [3, 2])
        r = b.call(A.relu.default, v1)
        b.output([b.call(A.add.Tensor, r, r)])

    @case
    def nothing_fuses_so_dead_code_stays(b):
        x = b.input("x", (2, 3))
        v0 = b.view(x, [6])
        a = b.call(A.add.Tensor, v0, v0)
        b.call(A.neg.default, x)
        b.output([b.view(a, [3, 2])])

    @case
    def a_fusion_removes_unrelated_dead_code(b):
        x = b.input("x", (2, 3))
        b.call(A.neg.default, x)
        v1 = b.view(b.view(x, [6]), [3, 2])
        b.view(x, [1, 6])
        b.output([v1])

    @case
    def no_view(b):
        x = b.input("x", (2, 3))
        b.output([b.call(A.relu.default, x)])

    for op_kwargs in UNARY_OPS:

        def through_each_unary_op(b, op_kwargs=op_kwargs):
            x = b.input("x", (2, 3, 4))
            v = b.view(x, [6, 4])
            u = unary(b, op_kwargs, v)
            b.output([b.view(u, [4, 6])])

        cases[f"through_{retab_target(op_kwargs[0])}"] = through_each_unary_op

    @case
    def quantize_dequantize_chain(b):
        x = b.input("x", (2, 3, 4))
        v = b.view(x, [24])
        q = b.call(Q.quantize_per_tensor.default, v, 0.5, 0, -128, 127, torch.int8)
        d = b.call(Q.dequantize_per_tensor.default, q, 0.5, 0, -128, 127, torch.int8)
        b.output([b.view(d, [4, 6])])

    @case
    def final_view_with_an_inferred_dim(b):
        x = b.input("x", (2, 3, 4))
        v = b.view(x, [6, 4])
        r = b.call(A.relu.default, v)
        b.output([b.view(r, [-1, 8])])

    @case
    def first_view_with_an_inferred_dim(b):
        x = b.input("x", (2, 3, 4))
        v = b.view(x, [-1, 4])
        r = b.call(A.relu.default, v)
        b.output([b.view(r, [3, 8])])

    @case
    def final_view_to_rank0(b):
        x = b.input("x", (1, 1))
        v = b.view(x, [1])
        r = b.call(A.relu.default, v)
        b.output([b.view(r, [])])

    @case
    def view_to_rank0_and_back(b):
        x = b.input("x", (1,))
        v = b.view(x, [])
        r = b.call(A.relu.default, v)
        b.output([b.view(r, [1, 1, 1])])

    @case
    def chain_ends_at_the_output(b):
        x = b.input("x", (2, 6))
        v = b.view(b.call(A.abs.default, b.view(x, [12])), [3, 4])
        b.output([v, v])

    @case
    def a_dead_user_stops_the_chain(b):
        x = b.input("x", (2, 3))
        v0 = b.view(x, [6])
        b.call(A.neg.default, v0)
        b.output([b.view(b.call(A.relu.default, v0), [3, 2])])

    @case
    def two_independent_chains(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (4, 5))
        a = b.view(b.call(A.relu.default, b.view(x, [6])), [3, 2])
        c = b.view(b.view(y, [20]), [2, 10])
        b.output([a, c])

    @case
    def fork_in_the_middle_of_the_chain(b):
        x = b.input("x", (2, 3))
        v0 = b.view(x, [6])
        r = b.call(A.relu.default, v0)
        v1 = b.view(r, [3, 2])
        n = b.call(A.neg.default, r)
        b.output([v1, n])

    @case
    def channels_last_input_view_chain(b):
        x = b.input("x", NCHW, dim_order=[0, 2, 3, 1])
        v = b.view(x, [3, 20])
        r = b.call(A.relu.default, v)
        b.output([b.view(r, [60])])

    @case
    def qparams_are_kept_on_retained_nodes(b):
        x = b.input("x", (2, 3))
        v0 = b.view(x, [6])
        set_qparams(v0, inputs=[0], outputs=[0])
        r = b.call(A.relu.default, v0)
        set_qparams(r, inputs=[], outputs=[0])
        v1 = b.view(r, [3, 2])
        set_qparams(v1, outputs=[0])
        b.output([v1])

    @case
    def long_chain(b):
        x = b.input("x", (2, 3, 4))
        h = b.view(x, [24])
        for i, shape in enumerate(([4, 6], [2, 2, 6], [1, 24], [8, 3])):
            h = unary(b, UNARY_OPS[(5 + 3 * i) % len(UNARY_OPS)], h)
            h = b.view(h, shape)
        b.output([b.call(A.relu.default, h)])

    return cases


def canonicalize_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    @case
    def direct_permute_chain(b):
        x = b.input("x", (2, 3, 4, 5))
        b.output([b.permute(b.permute(x, [0, 2, 3, 1]), [0, 3, 1, 2])])

    @case
    def singleton_permute_to_view(b):
        x = b.input("x", (1, 1, 1, 4))
        b.output([b.permute(x, [0, 3, 1, 2])])

    @case
    def keeps_equal_sized_non_singleton_permute(b):
        x = b.input("x", (2, 1, 2))
        b.output([b.permute(x, [2, 1, 0])])

    @case
    def pixel_shuffle_view_permute_chain(b):
        x = b.input("x", (1, 2, 2, 8))
        p1 = b.permute(x, [0, 3, 1, 2])
        v1 = b.view(p1, [1, 2, 2, 2, 2, 2])
        p2 = b.permute(v1, [0, 1, 4, 2, 5, 3])
        v2 = b.view(p2, [1, 2, 4, 4])
        b.output([b.permute(v2, [0, 2, 3, 1])])

    @case
    def direct_output_axis_permute(b):
        x = b.input("x", (2, 12, 5))
        v1 = b.view(x, [2, 3, 4, 5])
        p1 = b.permute(v1, [2, 0, 1, 3])
        b.output([b.view(p1, [4, 6, 5])])

    @case
    def reordered_view_with_grouped_permute(b):
        x = b.input("x", (2, 3, 4))
        v1 = b.view(x, [3, 2, 4])
        p1 = b.permute(v1, [2, 0, 1])
        b.output([b.view(p1, [4, 6])])

    @case
    def moves_view_before_permute(b):
        x = b.input("x", (2, 3, 4))
        p1 = b.permute(x, [2, 0, 1])
        v1 = b.view(p1, [4, 6])
        b.output([b.permute(v1, [1, 0])])

    @case
    def moves_permute_before_view(b):
        x = b.input("x", (1, 2, 10, 10))
        v1 = b.view(x, [1, 2, 5, 2, 5, 2])
        p1 = b.permute(v1, [3, 5, 0, 1, 2, 4])
        v2 = b.view(p1, [4, 2, 5, 5])
        b.output([b.permute(v2, [0, 2, 3, 1])])

    @case
    def sinks_singleton_view_below_permute(b):
        x = b.input("x", (2, 8, 64))
        v1 = b.view(x, [2, 8, 1, 64])
        b.output([b.permute(v1, [0, 2, 3, 1])])

    @case
    def follows_interleaved_chain_users(b):
        x = b.input("x", (4, 2, 4))
        y = b.input("y", (2, 3))
        p1 = b.permute(x, [1, 0, 2])
        unrelated = b.view(y, [3, 2])
        v1 = b.view(p1, [1, 2, 4, 4])
        p2 = b.permute(v1, [0, 2, 1, 3])
        b.output([p2, unrelated])

    @case
    def reordered_view_rejects_separating_permute(b):
        x = b.input("x", (2, 3, 4))
        v1 = b.view(x, [3, 2, 4])
        b.output([b.permute(v1, [0, 2, 1])])

    @case
    def does_not_cross_multi_user_chain_node(b):
        x = b.input("x", (2, 3, 4))
        p1 = b.permute(x, [0, 2, 1])
        p2 = b.permute(p1, [0, 2, 1])
        v1 = b.view(p1, [2, 12])
        b.output([p2, v1])

    @case
    def unsupported_start_view_does_not_block_suffix(b):
        x = b.input("x", (2, 3))
        v1 = b.view(x, [3, 2])
        b.output([b.permute(b.permute(v1, [1, 0]), [1, 0])])

    @case
    def unsupported_end_view_does_not_block_prefix(b):
        x = b.input("x", (2, 3))
        p2 = b.permute(b.permute(x, [1, 0]), [1, 0])
        b.output([b.view(p2, [3, 2])])

    @case
    def identity_view_and_permute(b):
        x = b.input("x", (2, 3, 4))
        v = b.view(x, [2, 3, 4])
        p = b.permute(v, [0, 1, 2])
        b.output([b.call(A.relu.default, p)])

    @case
    def negative_dims_are_normalized(b):
        x = b.input("x", (2, 3, 4))
        b.output([b.permute(x, [-1, 0, -2])])

    @case
    def consecutive_views_fuse(b):
        x = b.input("x", (2, 3, 4))
        b.output([b.view(b.view(b.view(x, [6, 4]), [24]), [4, 3, 2])])

    @case
    def views_with_inferred_dims(b):
        x = b.input("x", (2, 3, 4))
        v = b.view(x, [-1, 4])
        p = b.permute(v, [1, 0])
        b.output([b.view(p, [4, -1])])

    @case
    def chain_through_unary_op_is_two_chains(b):
        x = b.input("x", (2, 3, 4))
        p = b.permute(x, [2, 0, 1])
        r = b.call(A.relu.default, p)
        b.output([b.permute(r, [1, 2, 0])])

    @case
    def qparams_on_chain_nodes(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int8)
        p1 = b.permute(x, [0, 2, 1])
        set_qparams(p1, inputs=[0], outputs=[0])
        p2 = b.permute(p1, [1, 0, 2])
        set_qparams(p2, inputs=[0], outputs=[0])
        b.output([p2])

    return cases


# --------------------------------------------------------------------------
# Random cases
# --------------------------------------------------------------------------

UNARY_FLOAT = [
    A.relu.default,
    A.abs.default,
    A.neg.default,
    A.sigmoid.default,
    A.tanh.default,
    A.exp.default,
]
BINARY_FLOAT = [
    A.add.Tensor,
    A.sub.Tensor,
    A.mul.Tensor,
    A.div.Tensor,
    A.maximum.default,
    A.minimum.default,
]
COMPARE = [A.eq.Tensor, A.gt.Tensor, A.ge.Tensor, A.lt.Tensor, A.le.Tensor, A.ne.Tensor]
NON_TARGET_BINARY = [A.atan2.default, A.fmod.Tensor, A.hypot.default]


def prime_factors(n: int) -> list[int]:
    factors, p = [], 2
    while n > 1:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1
    return factors


class RandomGraph:
    """A seeded random edge graph over one profile's op mix."""

    def __init__(self, rng: random.Random, profile: str, builder: Builder) -> None:
        self.rng = rng
        self.profile = profile
        self.b = builder
        self.pool: list[torch.fx.Node] = []
        self.inputs = 0

    # Shapes ---------------------------------------------------------------

    def shape(self, rank: int, max_numel: int = 240) -> list[int]:
        rng = self.rng
        while True:
            dims = [1 if rng.random() < 0.3 else rng.randint(2, 5) for _ in range(rank)]
            if math.prod(dims) <= max_numel:
                return dims

    def view_shape(self, shape: Sequence[int]) -> list[int]:
        rng = self.rng
        factors = prime_factors(math.prod(shape))
        rng.shuffle(factors)
        buckets = rng.randint(1, min(4, max(1, len(factors)))) if factors else 0
        dims = [1] * buckets
        for f in factors:
            dims[rng.randrange(buckets)] *= f
        while len(dims) < 6 and rng.random() < 0.3:
            dims.insert(rng.randint(0, len(dims)), 1)
        if not dims and rng.random() < 0.5:
            dims = [1] * rng.randint(1, 3)
        if dims and rng.random() < 0.15:
            dims[rng.randrange(len(dims))] = -1
        return dims

    def permutation(self, rank: int) -> list[int]:
        dims = list(range(rank))
        self.rng.shuffle(dims)
        if self.rng.random() < 0.3:
            dims = [d - rank if self.rng.random() < 0.5 else d for d in dims]
        return dims

    def broadcast_partner(self, shape: Sequence[int]) -> list[int]:
        rng = self.rng
        rank = rng.randint(0, len(shape))
        tail = list(shape[len(shape) - rank :]) if rank else []
        partner = [1 if rng.random() < 0.3 else d for d in tail]
        if rng.random() < 0.1 and len(partner) < 6:
            partner.insert(0, rng.randint(1, 3))
        return partner

    def dim_order(self, rank: int) -> list[int] | None:
        rng = self.rng
        if rank < 2 or rng.random() < 0.4:
            return None
        if rank in (4, 5) and rng.random() < 0.6:
            return channels_last_order(rank)
        order = list(range(rank))
        rng.shuffle(order)
        return order

    # Nodes ------------------------------------------------------------------

    def new_input(self, shape, dtype=torch.float32, dim_order=None) -> torch.fx.Node:
        rng = self.rng
        name = f"x{self.inputs}"
        self.inputs += 1
        if self.profile == "normalize" and rng.random() < 0.2:
            kind = rng.choice(["param", "buffer", "constant"])
            node = self.b.constant(kind, name, shape, dtype, dim_order)
        else:
            node = self.b.input(name, shape, dtype, dim_order)
        if rng.random() < 0.15:
            set_qparams(node, inputs=rng.choice([None, [], [0]]), outputs=[0])
        return node

    def pick(self, dtype=torch.float32) -> torch.fx.Node:
        candidates = [n for n in self.pool if n.meta["val"].dtype == dtype]
        if not candidates or self.rng.random() < 0.15:
            rank = self.rng.randint(0, 6)
            node = self.new_input(self.shape(rank), dtype, self.dim_order(rank))
            self.pool.append(node)
            return node
        # Prefer recent values so chains form.
        return candidates[-1 - min(int(self.rng.expovariate(0.7)), len(candidates) - 1)]

    def operand_for(self, other: torch.fx.Node, dtype=torch.float32) -> torch.fx.Node:
        shape = list(other.meta["val"].shape)
        for node in reversed(self.pool):
            val = node.meta["val"]
            if node is not other and val.dtype == dtype and self.rng.random() < 0.3:
                try:
                    torch.broadcast_shapes(tuple(val.shape), tuple(shape))
                    return node
                except RuntimeError:
                    pass
        partner = self.broadcast_partner(shape)
        node = self.new_input(partner, dtype, self.dim_order(len(partner)))
        self.pool.append(node)
        return node

    def emit(self, node: torch.fx.Node) -> torch.fx.Node:
        self.pool.append(node)
        if self.rng.random() < 0.08:
            set_qparams(node, inputs=self.rng.choice([[], [0], [0, 1]]), outputs=[0])
        return node

    def step(self, weights: dict[str, float]) -> None:
        rng = self.rng
        kind = rng.choices(list(weights), list(weights.values()))[0]
        b = self.b
        if kind == "unary":
            x = self.pick()
            self.emit(b.call(rng.choice(UNARY_FLOAT), x))
        elif kind == "fuse_unary":
            x = self.pick()
            op_kwargs = rng.choice(UNARY_OPS)
            if op_kwargs[0] in (A._to_copy.default, D._to_dim_order_copy.default):
                op_kwargs = (op_kwargs[0], {"dtype": torch.float32})
            self.emit(unary(b, op_kwargs, x))
        elif kind == "binary":
            x = self.pick()
            y = self.operand_for(x)
            args = (x, y) if rng.random() < 0.5 else (y, x)
            if rng.random() < 0.1:
                args = (x, x)
            self.emit(b.call(rng.choice(BINARY_FLOAT), *args))
        elif kind == "compare_where":
            x = self.pick()
            y = self.operand_for(x)
            cond = b.call(rng.choice(COMPARE), x, y)
            other = self.operand_for(cond if rng.random() < 0.5 else x)
            if rng.random() < 0.15:
                other = x
            self.emit(b.call(A.where.self, cond, x, other))
        elif kind == "non_target":
            x = self.pick()
            y = self.operand_for(x)
            self.emit(b.call(rng.choice(NON_TARGET_BINARY), x, y))
        elif kind == "scalar":
            x = self.pick()
            self.emit(b.call(rng.choice([A.mul.Tensor, A.add.Tensor]), x, 2.0))
        elif kind == "tosa":
            x = self.pick(torch.int32)
            y = self.operand_for(x, torch.int32)
            name = rng.choice(["ADD", "SUB", "MAXIMUM", "BITWISE_AND", "GREATER"])
            self.emit(b.call(getattr(T, name).default, x, y))
        elif kind == "permute":
            x = self.pick()
            rank = x.meta["val"].dim()
            target = A.permute.default if rng.random() < 0.05 else A.permute_copy.default
            self.emit(b.call(target, x, self.permutation(rank)))
        elif kind == "high_rank_permute":
            rank = rng.randint(5, 6)
            shape = self.shape(rank, max_numel=360)
            for _ in range(rng.randint(1, 2)):
                shape[rng.randrange(rank)] = 1
            x = self.new_input(shape, dim_order=self.dim_order(rank))
            self.pool.append(x)
            if rng.random() < 0.5:
                x = b.call(rng.choice(UNARY_FLOAT), x)
            p = b.call(A.permute_copy.default, x, self.permutation(rank))
            if rng.random() < 0.2:
                set_qparams(p, inputs=[0], outputs=[0])
            self.pool.append(p)
        elif kind == "view":
            x = self.pick()
            self.emit(b.view(x, self.view_shape(list(x.meta["val"].shape))))
        elif kind == "dead":
            x = self.pick()
            b.call(rng.choice(UNARY_FLOAT), x)
        else:
            raise ValueError(kind)

    def finish(self) -> None:
        """Return every computed value nothing uses, so only the `dead` step
        leaves dead code, plus some inner values, repeats and inputs."""
        rng = self.rng
        computed = [n for n in self.pool if n.op == "call_function"]
        outputs = [n for n in computed if not n.users] or [rng.choice(self.pool)]
        rng.shuffle(outputs)
        if computed and rng.random() < 0.25:
            outputs.append(rng.choice(computed))
        if rng.random() < 0.2:
            outputs.append(rng.choice(outputs))
        if rng.random() < 0.15:
            outputs.append(rng.choice(self.pool))
        if self.profile == "normalize" and rng.random() < 0.4:
            # A channels-last result, the way export records one.
            source = rng.choice(outputs)
            val = source.meta["val"]
            if val.dim() == 4 and val.dtype == torch.float32:
                order = channels_last_order(4)
                outputs.append(
                    self.b.call(D._to_dim_order_copy.default, source, dim_order=order)
                )
        self.b.output(outputs)


PROFILES = {
    "normalize": {"unary": 3, "binary": 2, "permute": 1, "view": 1, "dead": 0.3},
    "match_arg_ranks": {
        "binary": 4,
        "compare_where": 2,
        "unary": 1,
        "view": 1,
        "non_target": 0.5,
        "scalar": 0.4,
        "tosa": 0.6,
    },
    "high_rank": {"high_rank_permute": 3, "permute": 1, "unary": 1, "view": 1, "dead": 0.2},
    "fuse_view_copy": {"view": 4, "fuse_unary": 3, "binary": 1, "dead": 0.4, "permute": 0.3},
    "canonicalize": {"view": 4, "permute": 4, "unary": 1, "binary": 0.5, "dead": 0.2},
}


def random_case(profile: str, seed: int) -> Callable[[Builder], None]:
    def build(b: Builder) -> None:
        rng = random.Random(seed)
        graph = RandomGraph(rng, profile, b)
        for _ in range(rng.randint(1, 3)):
            rank = rng.randint(0, 6)
            graph.pool.append(graph.new_input(graph.shape(rank), dim_order=graph.dim_order(rank)))
        for _ in range(rng.randint(1, 9)):
            graph.step(PROFILES[profile])
        graph.finish()

    return build


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

PASSES = [
    PassSpec(
        "normalize_delegate_io",
        lambda ep: NormalizeDelegateIOLayoutPass(ep),
        [
            "backends/arm/_passes/normalize_delegate_io_layout_pass.py",
            "backends/arm/_passes/arm_pass_utils.py",
        ],
        normalize_cases,
        "normalize",
    ),
    PassSpec(
        "match_arg_ranks",
        lambda ep: MatchArgRanksPass(ep),
        ["backends/arm/_passes/match_arg_ranks_pass.py"],
        match_arg_ranks_cases,
        "match_arg_ranks",
    ),
    PassSpec(
        "high_rank_singleton_permute",
        lambda ep: RewriteHighRankSingletonPermutePass(),
        ["backends/arm/_passes/rewrite_high_rank_singleton_permute_pass.py"],
        high_rank_cases,
        "high_rank",
    ),
    PassSpec(
        "fuse_view_copy",
        lambda ep: FuseViewCopyTransformPass(),
        [
            "backends/transforms/fuse_view_copy.py",
            "backends/arm/_passes/fuse_view_copy_transform_pass.py",
        ],
        fuse_view_copy_cases,
        "fuse_view_copy",
    ),
    PassSpec(
        "canonicalize_view_copy_permute",
        lambda ep: CanonicalizeViewCopyPermutePass(),
        [
            "backends/transforms/canonicalize_view_copy_permute_pass.py",
            "backends/arm/_passes/canonicalize_view_copy_permute_pass.py",
            "backends/transforms/permute_view_meta.py",
            "backends/transforms/dim_maps.py",
        ],
        canonicalize_cases,
        "canonicalize",
    ),
]
COMMON_SOURCES = ["backends/arm/_passes/arm_pass.py", "exir/pass_base.py"]


def source_sha256(relative: str) -> str:
    # `executorch` is a namespace package; `executorch.version` sits at its root.
    root = Path(executorch.version.__file__).resolve().parent
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


def header(
    spec: PassSpec, seed: int, random_cases: int, skipped: int, rc4_pass: str
) -> list[str]:
    git = getattr(executorch.version, "git_version", "unknown")
    lines = [
        f"# rc4 {rc4_pass} before/after graphs for src/layout/replay.rs.",
        "# Generated by tests/fixtures/layout_replay/gen_layout_replay.py; don't edit by hand.",
        "# Regenerate with the rc4 environment, from a neutral directory:",
        "#   cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py",
        f"# executorch {executorch.version.__version__} (git {git}), torch {torch.__version__},"
        f" python {sys.version.split()[0]}",
        f"# seed {seed}: {random_cases} random graphs, {skipped} of which torch rejected;"
        " format: see the generator's docstring.",
    ]
    for relative in spec.sources + COMMON_SOURCES:
        lines.append(f"# rc4-source sha256 {source_sha256(relative)} {relative}")
    return lines


def generate(spec: PassSpec, seed: int, random_cases: int) -> tuple[list[str], int]:
    cases = dict(spec.structured())
    for index in range(random_cases):
        cases[f"random_{index}"] = random_case(spec.random_profile, seed * 100_000 + index)
    lines: list[str] = []
    skipped = 0
    for name, build in cases.items():
        try:
            body = run_case(spec, build)
        except BuildError as error:
            if not name.startswith("random_"):
                raise
            skipped += 1
            print(f"{spec.name}/{name}: skipped, {error}"[:240], file=sys.stderr)
            continue
        lines += [f"case {name}", *body, "end"]
    return lines, skipped


def main() -> None:
    # FakeTensor logs a traceback for every random graph torch rejects.
    logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pass", dest="only", choices=[p.name for p in PASSES])
    parser.add_argument("--random", type=int, default=RANDOM_CASES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--case", help="print one case of --pass instead of writing")
    args = parser.parse_args()

    for spec in PASSES:
        if args.only and spec.name != args.only:
            continue
        rc4_pass = type(spec.make(None)).__name__
        if args.case:
            cases = dict(spec.structured())
            if args.case.startswith("random_"):
                index = int(args.case.removeprefix("random_"))
                cases[args.case] = random_case(spec.random_profile, args.seed * 100_000 + index)
            print("\n".join(run_case(spec, cases[args.case])))
            continue
        lines, skipped = generate(spec, args.seed, args.random)
        head = header(spec, args.seed, args.random, skipped, rc4_pass)
        text = "\n".join(head + lines) + "\n"
        path = args.out / f"{spec.name}.txt"
        path.write_text(text)
        count = sum(1 for line in lines if line.startswith("case "))
        print(f"{path.name}: {count} cases ({skipped} random graphs skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
