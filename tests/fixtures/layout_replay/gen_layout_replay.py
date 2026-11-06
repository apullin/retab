"""Record rc4 layout passes on hand-built edge graphs, for src/layout/replay.rs.

Each fixture holds cases for one ExecuTorch v1.6.0-rc4 pass: an edge-dialect
torch.fx graph ("before") and the graph the real rc4 pass returns ("after").
`replay::check` in src/layout/replay.rs rebuilds every "before" graph as an
IrGraph, runs the RETAB port and compares the result with "after".

Regenerate with the rc4 environment, from a neutral directory so a source
checkout can't shadow the installed package:

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py

Options: `--pass NAME` writes one fixture (repeat it for several), `--random N`
and `--seed S` change the random cases, `--out DIR` writes elsewhere, `--case
NAME` prints one case (of `--pass`). The output is stable for a given torch, ExecuTorch and CPython
`random` (written with torch 2.15.0, torchao 0.18.0, ExecuTorch 1.6.0 and
CPython 3.12).

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

Pipeline-captured fixtures (APU-135 P2)
---------------------------------------

The P2 fixtures (`rewrite_conv*.txt`, `rewrite_max_pool2d.txt`,
`remove_getitem.txt`, `rewrite_upsample.txt`, `rewrite_avg_pool2d*.txt`) are
not hand-built: small nn.Modules are quantized with rc4's TOSAQuantizer (INT
profiles) and lowered with `to_edge_transform_and_lower` and the
TOSAPartitioner, and capture passes registered with
`register_pass_insertions_before/after` (arm_pass_manager.py:278-296) record
the graph the real ArmPassManager pipeline hands to the target pass and the
graph it gets back. `modified` is whether the pass changed the graph (the
hooks can't see the flag the pass returned). No captured graph holds dead code
(rc4's P2 passes erase what they replace, and the passes before them clean up
after themselves; an ExportPass retrace alone would keep a dead node), so the
replay compares these graphs whole, dead nodes included, and p2_replay_tests
pins that none appears. Every case seeds torch with `SEED` before it builds
the model and its inputs, so the output is stable for a given torch, torchao
and ExecuTorch.

- `rewrite_conv.txt`: RewriteConvPass on INT8 (per-tensor `_pt`, per-channel
  `_pc`) and A8W4 (`_a8w4`) models; `rewrite_conv_a16w8.txt` and
  `rewrite_conv_a16w8_u55.txt`: A16W8 models on TOSA-1.0+INT+int16 and on the
  U55 subset (`TOSA-1.0+INT+int16+int4+u55`).
- `remove_getitem.txt`: RemoveGetItemPass (arm_pass_manager.py:605) on the
  max-pool models, which shows that rc4's MAX_POOL2D keeps the input qparams
  of `max_pool2d_with_indices` and drops the getitem's output qparams.
- `rewrite_max_pool2d.txt`: RewriteMaxPool2dPass. RETAB has no
  RemoveGetItemPass and keeps `max_pool2d_with_indices` + `getitem(0)`, so
  each case also comes as `<case>_with_indices`: the same rc4 before graph
  with RemoveGetItem undone (the pool becomes `max_pool2d_with_indices` with
  the edge args and qparam maps it had before RemoveGetItem, followed by a
  getitem with the getitem's maps), and the same after graph. The edge args
  are the ones RETAB keeps, with trailing defaults omitted; rc4's
  DecomposeMaxPool2dPass spells them out (`0, 1, False`) before
  RewriteMaxPool2dPass, which fills in the same defaults for either form.
- `rewrite_upsample.txt`: RewriteUpsamplePass.
- `rewrite_avg_pool2d.txt`: RewriteAvgPool2dPass alone (the zero points are
  still `aten.full` nodes); `rewrite_avg_pool2d_aot.txt`: the graph before
  RewriteAvgPool2dPass against the graph after the ComputeConstantOpsAOTPass
  that follows it (arm_pass_manager.py:698-699), where the zero points are
  `<full>_pre_computed` placeholders quantized with the pool's output qparams.
- `rewrite_conv_rc4_tests.txt`: rc4's own RewriteConvPass tests
  (backends/arm/test/passes/test_rewrite_conv_pass.py) that call the pass
  directly on an edge program instead of lowering through the pipeline. Each
  case builds the program one of those tests builds (an FP program, or a
  VGF-quantized one after the test's `_run_pre_rewrite_passes`) and calls
  RewriteConvPass the way the test does, so the Rust ports of the tests can
  assert the test's facts on RETAB's result and replay rc4's whole graph.
  `modified` is the flag the pass returned.

These files start with a `format 2` line and record more per node:

- `q=` holds each qparam's value, `{key:value,...}`: a per-tensor value is
  `dtype(scale,zp,qmin,qmax)` and a per-channel one
  `dtype([scales],[zps],qmin,qmax,axis=A)`, with floats in Python repr and no
  spaces, e.g. `q=in{0:torch.int8(0.0176,10,-128,127)}/out{}`.
- `data=hex:<bytes>` on every constant placeholder (and `get_attr` node): the
  tensor's contiguous little-endian bytes in its own dtype, or
  `data=sha256:<digest>` of those bytes above 4 KiB. The generator refuses a
  before graph with a sha256 constant, so every replay imports real data.
- `special=INT48|INT4|SHAPE|...`: the node's `tosa_special_dtype`.
- `no_fuse`: the node's `do_not_fuse_duplicate` flag is set.
- vals that aren't tensors print as tuples, `(torch.int8[1, 3, 4, 4],
  torch.int64[1, 3, 4, 4])` for `max_pool2d_with_indices` and `(4, 2, 4, 2)`
  for CONST_SHAPE.
- `get_attr NAME` nodes print like placeholders; devices, layouts and memory
  formats print as `device:cpu`, `layout:torch.strided` and
  `memory_format:torch.contiguous_format`.
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
# Pipeline-captured fixtures (APU-135 P2)
# --------------------------------------------------------------------------

FORMAT = 2
DATA_HEX_LIMIT = 4096
NN = torch.nn
FN = torch.nn.functional


class Ref:
    """A node reference in a captured graph: the node's position."""

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index


class DNode:
    """One node of a captured graph, with everything a fixture line prints."""

    def __init__(self, op: str, name: str) -> None:
        self.op = op
        self.name = name
        self.target = ""
        self.args: list = []
        self.kwargs: dict = {}
        self.val = ""
        self.shape: list[int] | None = None
        self.dim_order: list[int] | None = None
        self.kind: str | None = None
        self.data: str | None = None
        self.special: str | None = None
        self.no_fuse = False
        self.qin: dict | None = None
        self.qout: dict | None = None

    def copy(self) -> "DNode":
        node = DNode(self.op, self.name)
        node.__dict__.update(self.__dict__)
        node.args = list(self.args)
        node.kwargs = dict(self.kwargs)
        return node


class Captured:
    """A captured graph: its nodes and the output node's argument."""

    def __init__(self, nodes: list[DNode], output) -> None:
        self.nodes = nodes
        self.output = output

    def lines(self) -> list[str]:
        return format_captured(self)


def fmt_val2(val) -> str:
    if isinstance(val, torch.Tensor):
        return fmt_val(val)
    if isinstance(val, (list, tuple)):
        return "(" + ", ".join(fmt_val2(item) for item in val) + ")"
    if isinstance(val, (int, torch.SymInt)) and not isinstance(val, bool):
        return str(int(val))
    raise TypeError(f"cannot print val {val!r} ({type(val).__name__})")


def fmt_arg2(arg) -> str:
    if isinstance(arg, Ref):
        return f"%{arg.index}"
    if isinstance(arg, torch.device):
        return f"device:{arg}"
    if isinstance(arg, torch.layout):
        return f"layout:{arg}"
    if isinstance(arg, torch.memory_format):
        return f"memory_format:{arg}"
    if isinstance(arg, (list, tuple)):
        return "[" + ", ".join(fmt_arg2(a) for a in arg) + "]"
    return fmt_arg(arg, {})


def fmt_quant_args(qargs) -> str:
    dtype = str(qargs.dtype)
    if qargs.per_channel:
        if not isinstance(qargs.scale, (list, tuple)) or not isinstance(
            qargs.zp, (list, tuple)
        ):
            raise TypeError(f"per-channel qparams without lists: {qargs}")
        scales = ",".join(repr(float(s)) for s in qargs.scale)
        zps = ",".join(str(int(z)) for z in qargs.zp)
        return (
            f"{dtype}([{scales}],[{zps}],{int(qargs.qmin)},{int(qargs.qmax)},"
            f"axis={int(qargs.axis)})"
        )
    text = f"{dtype}({float(qargs.scale)!r},{int(qargs.zp)},{int(qargs.qmin)},{int(qargs.qmax)}"
    if int(qargs.axis) != 0:
        text += f",axis={int(qargs.axis)}"
    return text + ")"


def fmt_qmap(qmap: dict) -> str:
    return "{" + ",".join(f"{k}:{fmt_quant_args(qmap[k])}" for k in sorted(qmap)) + "}"


def fmt_qparams2(node: DNode) -> str:
    parts = []
    if node.qin is not None:
        parts.append(f"in{fmt_qmap(node.qin)}")
    if node.qout is not None:
        parts.append(f"out{fmt_qmap(node.qout)}")
    return "/".join(parts) if parts else "none"


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def encode_data(tensor: torch.Tensor) -> str:
    raw = tensor_bytes(tensor)
    if len(raw) <= DATA_HEX_LIMIT:
        return "hex:" + raw.hex()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def special_dtype_name(value) -> str:
    return getattr(value, "name", None) or str(value)


def capture_graph(gm: torch.fx.GraphModule, ep: ExportedProgram) -> Captured:
    """`gm` as DNodes: targets, args, vals, qparams and constant data."""
    from executorch.backends.arm.tosa.mapping import TosaSpecialDtype

    nodes = list(gm.graph.nodes)
    output = gm.graph.output_node()
    if nodes[-1] is not output:
        raise AssertionError("the output node must be last")
    pos = {node: index for index, node in enumerate(nodes)}

    def convert(arg):
        if isinstance(arg, torch.fx.Node):
            return Ref(pos[arg])
        if isinstance(arg, (list, tuple)):
            return [convert(a) for a in arg]
        if isinstance(arg, dict):
            raise TypeError(f"cannot capture a dict argument {arg!r}")
        return arg

    signature = ep.graph_signature
    bindings = {}
    for mapping, kind in (
        (signature.inputs_to_parameters, "param"),
        (signature.inputs_to_buffers, "buffer"),
        (signature.inputs_to_lifted_tensor_constants, "constant"),
    ):
        for name, target in mapping.items():
            bindings[name] = (kind, target)

    def constant_tensor(target: str):
        if target in ep.state_dict:
            return ep.state_dict[target]
        if target in ep.constants:
            return ep.constants[target]
        return None

    captured = []
    for node in nodes[:-1]:
        dnode = DNode(node.op, node.name)
        tensor = None
        if node.op == "placeholder":
            if node.name in bindings:
                dnode.kind, target = bindings[node.name]
                tensor = constant_tensor(target)
                if tensor is None:
                    raise AssertionError(f"no data for constant placeholder {node.name}")
        elif node.op == "get_attr":
            tensor = gm
            for part in str(node.target).split("."):
                tensor = getattr(tensor, part)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"get_attr {node.name} is not a tensor")
        elif node.op == "call_function":
            dnode.target = retab_target(node.target)
            dnode.args = convert(list(node.args))
            dnode.kwargs = {k: convert(v) for k, v in node.kwargs.items()}
        else:
            raise AssertionError(f"unexpected node op {node.op}")
        if "val" not in node.meta:
            raise AssertionError(f"{node.name} has no val")
        val = node.meta["val"]
        dnode.val = fmt_val2(val)
        if isinstance(val, torch.Tensor):
            dnode.shape = [int(d) for d in val.shape]
            order = list(val.dim_order())
            if order != list(range(len(order))):
                dnode.dim_order = order
        if tensor is not None:
            if not isinstance(val, torch.Tensor) or (
                tensor.dtype != val.dtype or list(tensor.shape) != list(val.shape)
            ):
                raise AssertionError(f"{node.name}: data {tensor.dtype}{list(tensor.shape)} "
                                     f"doesn't match its val {dnode.val}")
            dnode.data = encode_data(tensor)
        special = node.meta.get(TosaSpecialDtype.meta_key())
        if special is not None:
            dnode.special = special_dtype_name(special)
        dnode.no_fuse = bool(node.meta.get("do_not_fuse_duplicate", False))
        for key, attr in (("input_qparams", "qin"), ("output_qparams", "qout")):
            if key in node.meta and node.meta[key] is not None:
                setattr(dnode, attr, dict(node.meta[key]))
        captured.append(dnode)
    return Captured(captured, convert(output.args[0]))


def captured_refs(arg, out: list[int]) -> None:
    if isinstance(arg, Ref):
        out.append(arg.index)
    elif isinstance(arg, list):
        for item in arg:
            captured_refs(item, out)


def format_captured(graph: Captured) -> list[str]:
    reached = set()
    stack: list[int] = []
    captured_refs(graph.output, stack)
    while stack:
        index = stack.pop()
        if index in reached:
            continue
        reached.add(index)
        node = graph.nodes[index]
        captured_refs(node.args, stack)
        captured_refs(list(node.kwargs.values()), stack)
    lines = []
    for index, node in enumerate(graph.nodes):
        if node.op in ("placeholder", "get_attr"):
            head = f"{node.op} {node.name}"
        else:
            args = [fmt_arg2(a) for a in node.args]
            args += [f"{k}={fmt_arg2(v)}" for k, v in node.kwargs.items()]
            head = f"{node.target}({', '.join(args)})"
        notes = []
        if node.dim_order is not None:
            notes.append(f"dim_order=[{','.join(map(str, node.dim_order))}]")
        if node.kind is not None:
            notes.append(f"kind={node.kind}")
        if node.data is not None:
            notes.append(f"data={node.data}")
        if node.special is not None:
            notes.append(f"special={node.special}")
        if node.no_fuse:
            notes.append("no_fuse")
        notes.append(f"q={fmt_qparams2(node)}")
        if node.op == "call_function" and index not in reached:
            notes.append("dead")
        lines.append(f"%{index} = {head} : {node.val} {' '.join(notes)}")
    lines.append(f"output({fmt_arg2(graph.output)})")
    return lines


def undo_remove_getitem(graph: Captured, source: Captured) -> Captured:
    """`graph` with each `max_pool2d` turned back into the
    `max_pool2d_with_indices` + `getitem(0)` pair RETAB keeps. The pair takes
    its qparam maps from `source`, the graph RemoveGetItemPass received, pool
    for pool in graph order, and the pool keeps `graph`'s input but takes the
    rest of its args and its kwargs from `source`: the edge args, which RETAB
    keeps up to IrRewriteMaxPool2d (IrNormalizeMaxPool2dInputRank only lifts
    the input, IrDecomposeMaxPoolDilation skips undilated pools and
    IrSizeAdjustInput only replaces the input), while rc4's
    DecomposeMaxPool2dPass spells the omitted defaults out (`0, 1, False`)."""
    pairs = []
    for index, node in enumerate(source.nodes):
        if node.target != "aten.max_pool2d_with_indices.default":
            continue
        users = [
            user
            for user in source.nodes
            if user.target == "<built-in function getitem>"
            and isinstance(user.args[0], Ref)
            and user.args[0].index == index
        ]
        if len(users) != 1 or users[0].args[1] != 0:
            raise AssertionError("expected one getitem(0) per max_pool2d_with_indices")
        pairs.append((node, users[0]))
    pools = [i for i, node in enumerate(graph.nodes) if node.target == "aten.max_pool2d.default"]
    if len(pools) != len(pairs):
        raise AssertionError(f"{len(pools)} max_pool2d nodes, {len(pairs)} source pairs")
    new_nodes: list[DNode] = []
    remap: dict[int, int] = {}

    def renumber(arg):
        if isinstance(arg, Ref):
            return Ref(remap[arg.index])
        if isinstance(arg, list):
            return [renumber(a) for a in arg]
        return arg

    # A node only reads earlier nodes, so its references are mapped already.
    for index, node in enumerate(graph.nodes):
        node = node.copy()
        node.args = renumber(node.args)
        node.kwargs = {k: renumber(v) for k, v in node.kwargs.items()}
        if index not in pools:
            remap[index] = len(new_nodes)
            new_nodes.append(node)
            continue
        pool_source, getitem_source = pairs[pools.index(index)]
        edge_args = pool_source.args[1:]
        edge_refs: list[int] = []
        captured_refs([edge_args, list(pool_source.kwargs.values())], edge_refs)
        if edge_refs:
            raise AssertionError(f"{pool_source.name}: a node among the pool's edge args")
        getitem = DNode("call_function", f"{node.name}_getitem")
        getitem.target = "<built-in function getitem>"
        getitem.args = [Ref(len(new_nodes)), 0]
        getitem.val = node.val
        getitem.shape = node.shape
        getitem.dim_order = node.dim_order
        getitem.qin = getitem_source.qin
        getitem.qout = getitem_source.qout
        pool = node
        pool.target = "aten.max_pool2d_with_indices.default"
        pool.args = [node.args[0], *edge_args]
        pool.kwargs = dict(pool_source.kwargs)
        pool.val = f"({getitem.val}, torch.int64[{', '.join(map(str, getitem.shape))}])"
        pool.shape = None
        pool.dim_order = None
        pool.qin = pool_source.qin
        pool.qout = pool_source.qout
        new_nodes.append(pool)
        remap[index] = len(new_nodes)
        new_nodes.append(getitem)
    return Captured(new_nodes, renumber(graph.output))


class Recorder:
    """What the capture passes saw during one lowering: `(hook, phase,
    Captured)` in pipeline order."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, Captured]] = []

    def take(self) -> list[tuple[str, str, Captured]]:
        records, self.records = self.records, []
        return records


RECORDER = Recorder()


def make_capture_pass(hook: str, phase: str):
    from executorch.backends.arm._passes.arm_pass import ArmPass
    from executorch.exir.pass_base import PassResult

    class CapturePass(ArmPass):
        _passes_required_after: set = set()

        def __init__(self) -> None:
            super().__init__()
            self.hook = hook
            self.phase = phase
            # `_ExportedProgramGraphPassAdapter.call` sets this for the call.
            self.exported_program = None

        def call(self, graph_module):
            RECORDER.records.append(
                (self.hook, self.phase, capture_graph(graph_module, self.exported_program))
            )
            return PassResult(graph_module, False)

    CapturePass.__name__ = f"Capture_{phase}_{hook}"
    return CapturePass()


def capture_hooks():
    """`(hook name, rc4 pass class)` for every pass the P2 fixtures capture."""
    from executorch.backends.arm._passes import (
        ComputeConstantOpsAOTPass,
        RemoveGetItemPass,
        RewriteAvgPool2dPass,
        RewriteConvPass,
        RewriteMaxPool2dPass,
        RewriteUpsamplePass,
    )

    return [
        ("remove_getitem", RemoveGetItemPass),
        ("rewrite_avg_pool2d", RewriteAvgPool2dPass),
        ("compute_constant_ops_aot", ComputeConstantOpsAOTPass),
        ("rewrite_upsample", RewriteUpsamplePass),
        ("rewrite_max_pool2d", RewriteMaxPool2dPass),
        ("rewrite_conv", RewriteConvPass),
    ]


# Models ---------------------------------------------------------------------


class Conv2dM(NN.Module):
    def __init__(self, cin=3, cout=4, k=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = NN.Conv2d(
            cin, cout, k, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class Conv1dM(NN.Module):
    def __init__(self, cin=4, cout=6, k=3, stride=2, padding=1, groups=1, bias=True):
        super().__init__()
        self.conv = NN.Conv1d(cin, cout, k, stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Conv3dM(NN.Module):
    def __init__(self, cin=3, cout=5, k=3, stride=(1, 2, 2), padding=1, bias=True):
        super().__init__()
        self.conv = NN.Conv3d(cin, cout, k, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class TConv2dM(NN.Module):
    def __init__(self, cin=3, cout=4, k=3, stride=2, padding=1, output_padding=1, bias=True):
        super().__init__()
        self.conv = NN.ConvTranspose2d(
            cin, cout, k, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class LinearM(NN.Module):
    def __init__(self, cin=8, cout=4):
        super().__init__()
        self.fc = NN.Linear(cin, cout)

    def forward(self, x):
        return self.fc(x)


class ConvActM(NN.Module):
    """A conv followed by an activation: relu (fused into the conv's output
    qparams) or a hardtanh whose clamp stays."""

    def __init__(self, act, lo=-1.0, hi=1.0):
        super().__init__()
        self.conv = NN.Conv2d(3, 4, 3, padding=1)
        self.act, self.lo, self.hi = act, lo, hi

    def forward(self, x):
        y = self.conv(x)
        return FN.relu(y) if self.act == "relu" else FN.hardtanh(y, self.lo, self.hi)


class TwoConvsNoBias(NN.Module):
    def __init__(self):
        super().__init__()
        self.c1 = NN.Conv2d(3, 4, 3, padding=1, bias=False)
        self.c2 = NN.Conv2d(4, 5, 1, bias=False)

    def forward(self, x):
        return self.c2(self.c1(x))


class TwoConvsOneInput(NN.Module):
    def __init__(self):
        super().__init__()
        self.a = NN.Conv2d(3, 4, 3, padding=1)
        self.b = NN.Conv2d(3, 4, 1)

    def forward(self, x):
        return self.a(x) + self.b(x)


class DWConvsModule(NN.Module):
    """rc4's DWConvsModule (test_dw_convs_with_shared_weights.py): one
    depthwise conv module used twice. In INT nothing is shared by the time
    RewriteConvPass runs: the quantizer freezes one quantized weight per conv
    (even for one nn.Parameter read by two F.conv2d calls on the same input),
    so the convs read two placeholders with equal data. RewriteConvPass's
    shared-weight handling (design §2.6, "the second lands before the first")
    is reachable only in FP, which RETAB doesn't lower; rc4's pass test runs
    this module through RewriteConvPass alone in FP
    (test_rewrite_conv_pass.py, test_rewrite_conv_tosa_FP)."""

    def __init__(self):
        super().__init__()
        conv = NN.Conv2d(6, 6, kernel_size=(2, 2), groups=6)
        self.seq = NN.ModuleList([conv, NN.ReLU(), conv])

    def forward(self, x):
        for module in self.seq:
            x = module(x)
        return x


class CumsumM(NN.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=1)


# rc4's test_rewrite_conv_pass.py A16W8 models.


class A16W8MixedConsumerChain(NN.Module):
    def __init__(self, following):
        super().__init__()
        self.expand = NN.Conv2d(4, 8, 1)
        self.depthwise = NN.Conv2d(8, 8, 3, padding=1, groups=8)
        self.project = NN.Conv2d(8, 4, 1)
        self.following = following

    def forward(self, x):
        expanded = FN.relu(self.expand(x))
        filtered = FN.relu(self.depthwise(expanded))
        projected = self.project(filtered)
        residual = projected + x
        return self.following(projected) + residual


class A16W8Int32Consumer(NN.Module):
    def __init__(self):
        super().__init__()
        self.conv = NN.Conv2d(4, 4, 1)

    def forward(self, x):
        return self.conv(x) + x


class A16W8Conv1dInt32Consumer(NN.Module):
    def __init__(self):
        super().__init__()
        self.conv = NN.Conv1d(4, 4, 1)

    def forward(self, x):
        return self.conv(x) + x


class A16W8Conv1dSharedConsumers(NN.Module):
    def __init__(self):
        super().__init__()
        self.conv = NN.Conv1d(4, 4, 1)

    def forward(self, x):
        y = self.conv(x)
        return y - y.amax(dim=2, keepdim=True)


class A16W8MixedConsumer(NN.Module):
    def __init__(self, producer, following):
        super().__init__()
        self.producer = producer
        self.following = following

    def forward(self, x):
        produced = self.producer(x)
        return self.following(produced) + produced + x


class A16W8PermutedInt32Consumer(NN.Module):
    def __init__(self):
        super().__init__()
        self.producer = NN.Conv2d(4, 4, 1)
        self.following = NN.Conv2d(4, 4, 1)

    def forward(self, x):
        produced = self.producer(x)
        permuted = produced.permute(0, 1, 3, 2)
        return self.following(produced) + permuted + x


class ConvAddSub(NN.Module):
    def __init__(self):
        super().__init__()
        self.conv = NN.Conv2d(4, 4, 1)

    def forward(self, x):
        y = self.conv(x)
        return y + x, y - x


class PoolM(NN.Module):
    def __init__(self, pool, conv=False):
        super().__init__()
        self.conv = NN.Conv2d(3, 3, 3, padding=1) if conv else None
        self.pool = pool

    def forward(self, x):
        if self.conv is not None:
            x = FN.relu(self.conv(x))
        return self.pool(x)


class UpsampleM(NN.Module):
    def __init__(self, mode, size=None, scale=None, align_corners=None):
        super().__init__()
        self.mode, self.size, self.scale, self.align_corners = mode, size, scale, align_corners

    def forward(self, x):
        return FN.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners,
        )


def quant_config(name: str):
    """`(TOSA spec, quantization config)` for a quantization label."""
    from executorch.backends.arm.quantizer.arm_quantizer import (
        get_symmetric_a16w8_quantization_config,
        get_symmetric_a8w4_quantization_config,
        get_symmetric_quantization_config,
    )

    if name == "pt":
        return "TOSA-1.0+INT", get_symmetric_quantization_config(is_per_channel=False)
    if name == "pc":
        return "TOSA-1.0+INT", get_symmetric_quantization_config(is_per_channel=True)
    if name == "a8w4":
        return "TOSA-1.0+INT+int4", get_symmetric_a8w4_quantization_config(
            is_per_channel=True, is_qat=False
        )
    if name == "a16w8":
        return "TOSA-1.0+INT+int16", get_symmetric_a16w8_quantization_config()
    if name == "a16w8_u55":
        return "TOSA-1.0+INT+int16+int4+u55", get_symmetric_a16w8_quantization_config()
    raise ValueError(name)


def shape_input(*shape):
    return lambda: (torch.randn(*shape),)


def float32(value: float) -> float:
    """`value` rounded to float32, as a quantized export can carry a scale
    factor (1.6 becomes 1.600000023841858; rewrite_upsample.py:101-108)."""
    return float(torch.tensor(value, dtype=torch.float32))


# `name -> (model factory, input factory)`.
CONV_MODELS = {
    "conv2d": (lambda: Conv2dM(), shape_input(1, 3, 8, 8)),
    "conv2d_nobias": (lambda: Conv2dM(bias=False), shape_input(1, 3, 8, 8)),
    "conv2d_s2p1": (lambda: Conv2dM(stride=2, padding=1), shape_input(1, 3, 8, 8)),
    "conv2d_s3p2": (lambda: Conv2dM(stride=3, padding=2), shape_input(1, 3, 10, 11)),
    # The stride leaves a remainder larger than the padding, so
    # SizeAdjustInputPass slices the input (to 7x7) ahead of RewriteConvPass.
    "conv2d_s2p0": (lambda: Conv2dM(stride=2, padding=0), shape_input(1, 3, 8, 8)),
    "conv2d_1x1_s2": (lambda: Conv2dM(k=1, stride=2, padding=0), shape_input(1, 3, 8, 8)),
    "conv2d_dilation2": (lambda: Conv2dM(padding=2, dilation=2), shape_input(1, 3, 8, 8)),
    "conv2d_relu": (lambda: ConvActM("relu"), shape_input(1, 3, 8, 8)),
    "conv2d_hardtanh": (lambda: ConvActM("hardtanh"), shape_input(1, 3, 8, 8)),
    "conv2d_relu6": (lambda: ConvActM("hardtanh", 0.0, 6.0), shape_input(1, 3, 8, 8)),
    "dw2d": (lambda: Conv2dM(4, 8, groups=4), shape_input(1, 4, 8, 8)),
    "dw2d_nobias": (lambda: Conv2dM(4, 8, groups=4, bias=False), shape_input(1, 4, 8, 8)),
    "dw2d_kx1": (
        lambda: Conv2dM(4, 8, k=(3, 1), padding=(1, 0), groups=4),
        shape_input(1, 4, 8, 8),
    ),
    "grouped2d": (lambda: Conv2dM(4, 6, groups=2), shape_input(1, 4, 8, 8)),
    "tconv2d": (lambda: TConv2dM(), shape_input(1, 3, 6, 6)),
    "tconv2d_nobias": (lambda: TConv2dM(bias=False), shape_input(1, 3, 6, 6)),
    "conv1d": (lambda: Conv1dM(), shape_input(1, 4, 9)),
    "conv1d_nobias": (lambda: Conv1dM(bias=False), shape_input(1, 4, 9)),
    "conv1d_dw": (lambda: Conv1dM(4, 8, groups=4), shape_input(1, 4, 9)),
    # Sliced to length 7 like conv2d_s2p0; RewriteConvPass emits slice ->
    # permute -> view -> CONV2D (design C7).
    "conv1d_s2p0": (lambda: Conv1dM(padding=0), shape_input(1, 4, 8)),
    "conv3d": (lambda: Conv3dM(), shape_input(1, 3, 4, 6, 6)),
    "linear": (lambda: LinearM(), shape_input(2, 8)),
    "linear_rank3": (lambda: LinearM(8, 5), shape_input(2, 3, 8)),
    "linear_in1": (lambda: LinearM(1, 4), shape_input(2, 1)),
    "conv2d_cin1": (lambda: Conv2dM(1, 4), shape_input(1, 1, 8, 8)),
    "conv1d_cin1": (lambda: Conv1dM(1, 4, stride=1), shape_input(1, 1, 9)),
    "conv3d_cin1": (lambda: Conv3dM(1, 4, stride=1), shape_input(1, 1, 4, 6, 6)),
    "two_convs_no_bias": (lambda: TwoConvsNoBias(), shape_input(1, 3, 8, 8)),
    "two_convs_one_input": (lambda: TwoConvsOneInput(), shape_input(1, 3, 8, 8)),
    "dw_reused_module": (lambda: DWConvsModule(), shape_input(1, 6, 24, 24)),
    "cumsum": (lambda: CumsumM(), shape_input(1, 4, 5)),
}

A16W8_MODELS = {
    "conv2d": CONV_MODELS["conv2d"],
    "dw2d": CONV_MODELS["dw2d"],
    "tconv2d": CONV_MODELS["tconv2d"],
    "conv3d": CONV_MODELS["conv3d"],
    "conv1d": CONV_MODELS["conv1d"],
    "linear": CONV_MODELS["linear"],
    "conv_add": (lambda: A16W8Int32Consumer(), shape_input(1, 4, 8, 8)),
    "conv1d_add": (lambda: A16W8Conv1dInt32Consumer(), shape_input(1, 4, 8)),
    "conv1d_shared_consumers": (lambda: A16W8Conv1dSharedConsumers(), shape_input(1, 4, 8)),
    "conv_add_sub": (lambda: ConvAddSub(), shape_input(1, 4, 8, 8)),
    "permuted_int32": (lambda: A16W8PermutedInt32Consumer(), shape_input(1, 4, 8, 8)),
    # test_rewrite_conv_a16w8_mixed_consumers_restore_int16's six models.
    "mixed_chain_conv": (
        lambda: A16W8MixedConsumerChain(NN.Conv2d(4, 4, 1)),
        shape_input(1, 4, 8, 8),
    ),
    "mixed_chain_dw": (
        lambda: A16W8MixedConsumerChain(NN.Conv2d(4, 4, 3, padding=1, groups=4)),
        shape_input(1, 4, 8, 8),
    ),
    "mixed_chain_avg_pool": (
        lambda: A16W8MixedConsumerChain(NN.AvgPool2d(3, stride=1, padding=1)),
        shape_input(1, 4, 8, 8),
    ),
    "mixed_chain_max_pool": (
        lambda: A16W8MixedConsumerChain(NN.MaxPool2d(3, stride=1, padding=1)),
        shape_input(1, 4, 8, 8),
    ),
    "mixed_conv3d": (
        lambda: A16W8MixedConsumer(NN.Conv3d(4, 4, 3, padding=1), NN.Conv3d(4, 4, 1)),
        shape_input(1, 4, 4, 8, 8),
    ),
    "mixed_tconv": (
        lambda: A16W8MixedConsumer(NN.ConvTranspose2d(4, 4, 3, padding=1), NN.Conv2d(4, 4, 1)),
        shape_input(1, 4, 8, 8),
    ),
}

MAX_POOL_MODELS = {
    "k3s2p1": (lambda: PoolM(NN.MaxPool2d(3, 2, 1)), shape_input(1, 3, 8, 8)),
    "k2s2": (lambda: PoolM(NN.MaxPool2d(2, 2)), shape_input(1, 3, 8, 8)),
    "k3s1p1": (lambda: PoolM(NN.MaxPool2d(3, 1, 1)), shape_input(1, 3, 8, 8)),
    "ceil_mode": (lambda: PoolM(NN.MaxPool2d(3, 2, 0, ceil_mode=True)), shape_input(1, 3, 8, 8)),
    "rect": (lambda: PoolM(NN.MaxPool2d((3, 2), (2, 1), (1, 0))), shape_input(1, 3, 8, 7)),
    "rank3": (lambda: PoolM(NN.MaxPool2d(2, 2)), shape_input(3, 8, 8)),
    "after_conv": (lambda: PoolM(NN.MaxPool2d(2, 2), conv=True), shape_input(1, 3, 8, 8)),
}

AVG_POOL_MODELS = {
    "k3s2p1": (
        lambda: PoolM(NN.AvgPool2d(3, 2, 1, count_include_pad=False)),
        shape_input(1, 3, 8, 8),
    ),
    "k3s2p1_include_pad": (lambda: PoolM(NN.AvgPool2d(3, 2, 1)), shape_input(1, 3, 8, 8)),
    "k2s2": (lambda: PoolM(NN.AvgPool2d(2, 2)), shape_input(1, 3, 8, 8)),
    "ceil_mode": (lambda: PoolM(NN.AvgPool2d(3, 2, 0, ceil_mode=True)), shape_input(1, 3, 8, 8)),
    "rect": (lambda: PoolM(NN.AvgPool2d((3, 2), (2, 1), (1, 0))), shape_input(1, 3, 8, 7)),
    "adaptive": (lambda: PoolM(NN.AdaptiveAvgPool2d((2, 2))), shape_input(1, 3, 8, 8)),
    "global": (lambda: PoolM(NN.AvgPool2d(8)), shape_input(1, 3, 8, 8)),
    "after_conv": (lambda: PoolM(NN.AvgPool2d(2, 2), conv=True), shape_input(1, 3, 8, 8)),
}

UPSAMPLE_MODELS = {
    "nearest_size_5to8": (lambda: UpsampleM("nearest", size=(8, 8)), shape_input(1, 3, 5, 5)),
    "nearest_scale2": (lambda: UpsampleM("nearest", scale=2.0), shape_input(1, 3, 4, 4)),
    "nearest_scale1_5": (lambda: UpsampleM("nearest", scale=1.5), shape_input(1, 3, 4, 4)),
    "nearest_scale1_6": (lambda: UpsampleM("nearest", scale=1.6), shape_input(1, 3, 5, 5)),
    "nearest_scale2_5": (lambda: UpsampleM("nearest", scale=2.5), shape_input(1, 3, 4, 4)),
    "nearest_scale_third": (lambda: UpsampleM("nearest", scale=1 / 3), shape_input(1, 3, 9, 9)),
    # Float32-rounded factors: only nsimplify's 1e-6 tolerance recovers 8/5
    # and 1/3 from them (design risk 6). 1.6 and 1/3 above are doubles within
    # 1e-16 of their rationals.
    "nearest_scale1_6_f32": (
        lambda: UpsampleM("nearest", scale=float32(1.6)),
        shape_input(1, 3, 5, 5),
    ),
    "nearest_scale_third_f32": (
        lambda: UpsampleM("nearest", scale=float32(1 / 3)),
        shape_input(1, 3, 9, 9),
    ),
    "nearest_scale2x3": (lambda: UpsampleM("nearest", scale=(2.0, 3.0)), shape_input(1, 3, 4, 4)),
    "bilinear_scale2": (
        lambda: UpsampleM("bilinear", scale=2.0, align_corners=False),
        shape_input(1, 3, 4, 4),
    ),
    "bilinear_scale2_align_corners": (
        lambda: UpsampleM("bilinear", scale=2.0, align_corners=True),
        shape_input(1, 3, 4, 4),
    ),
    "bilinear_size_5to9_align_corners": (
        lambda: UpsampleM("bilinear", size=(9, 9), align_corners=True),
        shape_input(1, 3, 5, 5),
    ),
    "bilinear_size_5to8": (
        lambda: UpsampleM("bilinear", size=(8, 8), align_corners=False),
        shape_input(1, 3, 5, 5),
    ),
    "bilinear_scale1_5": (
        lambda: UpsampleM("bilinear", scale=1.5, align_corners=False),
        shape_input(1, 3, 4, 4),
    ),
}


class ModelCase:
    def __init__(self, name: str, models: dict, model: str, quant: str) -> None:
        self.name = name
        self.make, self.inputs = models[model]
        self.quant = quant


def model_cases(prefix: str, models: dict, quants: Sequence[str], only=None) -> list[ModelCase]:
    names = list(models) if only is None else list(only)
    return [
        ModelCase(f"{prefix}{name}_{quant}", models, name, quant)
        for name in names
        for quant in quants
    ]


class PipelineSpec:
    """One P2 fixture: the hook whose before graph opens each case, the hook
    whose after graph closes it, the cases and the rc4 sources they depend on."""

    def __init__(self, name, rc4_pass, before, after, cases, sources, with_indices=False):
        self.name = name
        self.rc4_pass = rc4_pass
        self.before = before
        self.after = after
        self.cases: Callable[[], list[ModelCase]] = cases
        self.sources = sources
        self.with_indices = with_indices


PIPELINE_COMMON_SOURCES = [
    "backends/arm/_passes/arm_pass_manager.py",
    "backends/arm/_passes/arm_pass_utils.py",
    "backends/arm/_passes/fold_qdq_with_annotated_qparams_pass.py",
    "backends/arm/_passes/size_adjust_input_pass.py",
    "backends/arm/quantizer/arm_quantizer.py",
    "backends/arm/quantizer/quantization_annotator.py",
]
CONV_SOURCES = [
    "backends/arm/_passes/rewrite_conv_pass.py",
    "backends/arm/_passes/decompose_grouped_conv_pass.py",
    "backends/arm/_passes/decompose_linear_pass.py",
    "backends/arm/_passes/insert_rescales_pass.py",
    "backends/arm/tosa/dialect/ops/conv2d.py",
    "backends/arm/tosa/dialect/ops/conv3d.py",
    "backends/arm/tosa/dialect/ops/depthwise_conv2d.py",
    "backends/arm/tosa/dialect/ops/transpose_conv2d.py",
]
MAX_POOL_SOURCES = [
    "backends/arm/_passes/rewrite_max_pool2d_pass.py",
    "backends/arm/_passes/remove_getitem_pass.py",
    "backends/transforms/remove_getitem_op.py",
    # They reshape the pool between RemoveGetItemPass and RewriteMaxPool2dPass
    # (arm_pass_manager.py:692-693).
    "backends/arm/_passes/normalize_max_pool2d_input_rank_pass.py",
    "backends/arm/_passes/decompose_maxpool2d_with_dilation_pass.py",
    "backends/arm/tosa/dialect/ops/max_pool2d.py",
]
AVG_POOL_SOURCES = [
    "backends/arm/_passes/rewrite_avg_pool2d_pass.py",
    "backends/arm/_passes/fuse_constant_ops_pass.py",
    "backends/arm/tosa/dialect/ops/avg_pool2d.py",
]
UPSAMPLE_SOURCES = [
    "backends/arm/_passes/rewrite_upsample.py",
    "backends/arm/tosa/dialect/ops/resize.py",
    "backends/arm/tosa/dialect/ops/shape_ops.py",
]


def conv_cases():
    return (
        model_cases("", CONV_MODELS, ["pt", "pc"])
        + model_cases("", CONV_MODELS, ["a8w4"], only=["conv2d", "dw2d", "conv1d", "linear"])
    )


def max_pool_cases():
    return model_cases("max_pool_", MAX_POOL_MODELS, ["pc"]) + model_cases(
        "max_pool_", MAX_POOL_MODELS, ["a16w8"], only=["k3s2p1"]
    )


def avg_pool_cases():
    return model_cases("avg_pool_", AVG_POOL_MODELS, ["pc"]) + model_cases(
        "avg_pool_", AVG_POOL_MODELS, ["a16w8"], only=["k3s2p1"]
    )


def upsample_cases():
    return model_cases("", UPSAMPLE_MODELS, ["pc"]) + model_cases(
        "", UPSAMPLE_MODELS, ["a16w8"], only=["nearest_scale2", "bilinear_scale2"]
    )


PIPELINE_SPECS = [
    PipelineSpec(
        "rewrite_conv",
        "RewriteConvPass",
        "rewrite_conv",
        "rewrite_conv",
        conv_cases,
        CONV_SOURCES,
    ),
    PipelineSpec(
        "rewrite_conv_a16w8",
        "RewriteConvPass",
        "rewrite_conv",
        "rewrite_conv",
        lambda: model_cases("", A16W8_MODELS, ["a16w8"]),
        CONV_SOURCES,
    ),
    PipelineSpec(
        "rewrite_conv_a16w8_u55",
        "RewriteConvPass",
        "rewrite_conv",
        "rewrite_conv",
        # The U55 subset has no CONV3D.
        lambda: model_cases(
            "",
            A16W8_MODELS,
            ["a16w8_u55"],
            only=[name for name in A16W8_MODELS if name not in ("conv3d", "mixed_conv3d")],
        ),
        CONV_SOURCES,
    ),
    PipelineSpec(
        "remove_getitem",
        "RemoveGetItemPass",
        "remove_getitem",
        "remove_getitem",
        max_pool_cases,
        MAX_POOL_SOURCES,
    ),
    PipelineSpec(
        "rewrite_max_pool2d",
        "RewriteMaxPool2dPass",
        "rewrite_max_pool2d",
        "rewrite_max_pool2d",
        max_pool_cases,
        MAX_POOL_SOURCES,
        with_indices=True,
    ),
    PipelineSpec(
        "rewrite_upsample",
        "RewriteUpsamplePass",
        "rewrite_upsample",
        "rewrite_upsample",
        upsample_cases,
        UPSAMPLE_SOURCES,
    ),
    PipelineSpec(
        "rewrite_avg_pool2d",
        "RewriteAvgPool2dPass",
        "rewrite_avg_pool2d",
        "rewrite_avg_pool2d",
        avg_pool_cases,
        AVG_POOL_SOURCES,
    ),
    PipelineSpec(
        "rewrite_avg_pool2d_aot",
        "RewriteAvgPool2dPass then ComputeConstantOpsAOTPass",
        "rewrite_avg_pool2d",
        "compute_constant_ops_aot",
        avg_pool_cases,
        AVG_POOL_SOURCES,
    ),
]


class LoweringCache:
    """Each model case is lowered once; its captures serve every fixture."""

    def __init__(self) -> None:
        self.records: dict[str, list[tuple[str, str, Captured]]] = {}

    def get(self, case: ModelCase) -> list[tuple[str, str, Captured]]:
        if case.name not in self.records:
            self.records[case.name] = lower_and_capture(case)
        return self.records[case.name]


def lower_and_capture(case: ModelCase) -> list[tuple[str, str, Captured]]:
    from executorch.backends.arm._passes.arm_pass_manager import (
        clear_registered_pass_insertions,
        register_pass_insertions_after,
        register_pass_insertions_before,
    )
    from executorch.backends.arm.quantizer import TOSAQuantizer
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    spec, config = quant_config(case.quant)
    torch.manual_seed(SEED)
    model = case.make().eval()
    inputs = case.inputs()
    compile_spec = TosaCompileSpec(spec)
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(config)
    exported = torch.export.export(model, inputs)
    with torch.no_grad():
        quantized = quantizer._quantize_with_submodules(
            exported.module(check_guards=False), [inputs]
        )
    exported = torch.export.export(quantized, inputs)
    clear_registered_pass_insertions()
    for hook, pass_type in capture_hooks():
        register_pass_insertions_before(pass_type, [make_capture_pass(hook, "before")])
        register_pass_insertions_after(pass_type, [make_capture_pass(hook, "after")])
    RECORDER.take()
    try:
        to_edge_transform_and_lower(
            exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[TOSAPartitioner(compile_spec)],
        )
    finally:
        clear_registered_pass_insertions()
    return RECORDER.take()


def captures(records, hook: str, phase: str) -> list[Captured]:
    return [graph for h, p, graph in records if h == hook and p == phase]


def pipeline_case_bodies(spec: PipelineSpec, case: ModelCase, cache: LoweringCache):
    """`(case name, lines)` for one model case; one per pipeline run that
    reached the hooks (a model with several partitions runs it several
    times), plus the `_with_indices` variants."""
    records = cache.get(case)
    befores = captures(records, spec.before, "before")
    afters = captures(records, spec.after, "after")
    if not befores or len(befores) != len(afters):
        raise AssertionError(
            f"{spec.name}/{case.name}: {len(befores)} before and {len(afters)} after "
            "captures (was the model partitioned?)"
        )
    sources = captures(records, "remove_getitem", "before") if spec.with_indices else []
    bodies = []
    for run, (before, after) in enumerate(zip(befores, afters)):
        name = case.name if len(befores) == 1 else f"{case.name}_run{run}"
        for node in before.nodes:
            if node.data is not None and node.data.startswith("sha256:"):
                raise AssertionError(
                    f"{spec.name}/{name}: before-graph constant {node.name} is over "
                    f"{DATA_HEX_LIMIT} bytes, so a replay can't import it"
                )
        variants = [(name, before)]
        if spec.with_indices:
            if len(sources) != len(befores):
                raise AssertionError(f"{name}: no RemoveGetItemPass capture to undo")
            variants.append((f"{name}_with_indices", undo_remove_getitem(before, sources[run])))
        after_lines = after.lines()
        # The variants share rc4's flag: whether the pass changed its graph.
        modified = before.lines() != after_lines
        for variant, graph in variants:
            before_lines = graph.lines()
            lines = ["before", *before_lines, f"after modified={str(modified).lower()}"]
            if after_lines == before_lines:
                lines[-1] += " same"
            else:
                lines += after_lines
            bodies.append((variant, lines))
    return bodies


def pipeline_header(spec: PipelineSpec, count: int) -> list[str]:
    import torchao

    git = getattr(executorch.version, "git_version", "unknown")
    lines = [
        f"# rc4 {spec.rc4_pass} before/after graphs, captured in ArmPassManager's pipeline,"
        " for src/layout/replay.rs.",
        "# Generated by tests/fixtures/layout_replay/gen_layout_replay.py; don't edit by hand.",
        "# Regenerate with the rc4 environment, from a neutral directory:",
        "#   cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py"
        f" --pass {spec.name}",
        f"# executorch {executorch.version.__version__} (git {git}), torch {torch.__version__},"
        f" torchao {torchao.__version__}, python {sys.version.split()[0]}",
        f"# seed {SEED}: {count} cases; format {FORMAT}: see the generator's docstring.",
    ]
    for relative in spec.sources + PIPELINE_COMMON_SOURCES + COMMON_SOURCES:
        lines.append(f"# rc4-source sha256 {source_sha256(relative)} {relative}")
    return lines


def generate_pipeline(spec: PipelineSpec, cache: LoweringCache) -> list[str]:
    lines = [f"format {FORMAT}"]
    for case in spec.cases():
        for name, body in pipeline_case_bodies(spec, case, cache):
            lines += [f"case {name}", *body, "end"]
    return lines


# --------------------------------------------------------------------------
# rc4's own RewriteConvPass tests (APU-135 P2 S3)
# --------------------------------------------------------------------------
#
# The installed package doesn't ship backends/arm/test/passes, so the models
# and helpers those tests use are copied here (TinyConvReluCat, ConvModule,
# Conv1dBiasModule, `_export_quantized`, `_run_pre_rewrite_passes`); the
# DWConvsModule above is rc4's. Every case seeds torch with SEED before it
# builds the model and its inputs: the structure is the test's, the data is
# this generator's.


class TinyConvReluCat(NN.Module):
    """test_rewrite_conv_pass.py's TinyConvReluCat."""

    def __init__(self, conv1_bias: bool = True) -> None:
        super().__init__()
        self.conv1 = NN.Conv2d(4, 4, 3, padding=1, bias=conv1_bias)
        self.conv2 = NN.Conv2d(8, 4, 1)
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-0.1, 0.1)

    def forward(self, x, y):
        relu_out = torch.nn.functional.relu(self.conv1(x))
        merged = torch.cat((relu_out, y), dim=1)
        return self.conv2(merged)


class ConvModule(NN.Module):
    """test_rewrite_conv_pass.py's ConvModule: k3 s3 p0, 3 -> 16."""

    def __init__(self):
        super().__init__()
        self.conv = NN.Conv2d(3, 16, kernel_size=3, stride=3, padding=0)

    def forward(self, x):
        return self.conv(x)


class Conv1dBiasModule(NN.Module):
    """test_rewrite_conv_pass.py's Conv1dBiasModule."""

    def __init__(self, depthwise: bool = False) -> None:
        super().__init__()
        groups = 4 if depthwise else 1
        out_channels = 8 if depthwise else 6
        self.conv = NN.Conv1d(4, out_channels, kernel_size=3, padding=1, groups=groups, bias=True)

    def forward(self, x):
        return self.conv(x)


def rc4_test_fp_program(model, inputs):
    """`to_edge(export(model, inputs)).exported_program()`, as the FP tests
    (and PassPipeline's to_edge stage) build it."""
    from executorch.exir import to_edge

    ep = to_edge(torch.export.export(model.eval(), inputs)).exported_program()
    return ep, ep.graph_module


def rc4_test_vgf_program(model):
    """The VGF tests' `_export_quantized` (VgfQuantizer on TOSA-1.0+INT+FP,
    per-channel, symmetric [-127, 127]), `to_edge` without IR validity checks
    and `_run_pre_rewrite_passes` (test_rewrite_conv_pass.py:166-224)."""
    from executorch.backends.arm._passes import (
        ConvertToClampPass,
        FoldAndAnnotateQParamsPass,
        FuseQuantizedActivationPass,
        QuantizeClampArgumentsPass,
    )
    from executorch.backends.arm.quantizer.arm_quantizer import (
        get_symmetric_quantization_config,
        VgfQuantizer,
    )
    from executorch.backends.arm.vgf import VgfCompileSpec
    from executorch.exir import EdgeCompileConfig, to_edge

    inputs = (torch.rand(1, 4, 16, 16), torch.rand(1, 4, 16, 16) - 0.065)
    quantizer = VgfQuantizer(VgfCompileSpec("TOSA-1.0+INT+FP"))
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=True, act_qmin=-127, act_qmax=127, weight_qmin=-127, weight_qmax=127
        )
    )
    exported = torch.export.export(model.eval(), inputs).module(check_guards=False)
    quantized = quantizer._quantize_with_submodules(exported, [inputs])
    ep = to_edge(
        torch.export.export(quantized, inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    gm = ep.graph_module
    for pass_ in (
        FuseQuantizedActivationPass(),
        ConvertToClampPass(),
        FoldAndAnnotateQParamsPass(ep),
        QuantizeClampArgumentsPass(),
    ):
        gm = pass_(gm).graph_module
    return ep, gm


class Rc4TestCase:
    """One rc4 test's RewriteConvPass call: the program it runs the pass on
    and the TOSA spec of the lowering context it calls it in (None: none)."""

    def __init__(self, name: str, test: str, build, spec: str | None) -> None:
        self.name = name
        self.test = test
        self.build = build
        self.spec = spec


def rc4_test_cases() -> list[Rc4TestCase]:
    return [
        Rc4TestCase(
            "tosa_fp_dw_convs_shared_weights",
            "test_rewrite_conv_tosa_FP (:438)",
            lambda: rc4_test_fp_program(DWConvsModule(), (torch.randn(1, 6, 24, 24),)),
            "TOSA-1.0+FP",
        ),
        Rc4TestCase(
            "no_convolution",
            "test_rewrite_conv_without_convolution_does_not_require_context (:555)",
            lambda: rc4_test_fp_program(NN.Identity(), (torch.randn(1, 4),)),
            None,
        ),
        Rc4TestCase(
            "conv1d_atomic_rank3_layout_boundaries",
            "test_rewrite_conv1d_emits_atomic_rank3_layout_boundaries[False] (:828)",
            lambda: rc4_test_fp_program(Conv1dBiasModule(False), (torch.randn(1, 4, 8),)),
            "TOSA-1.0+INT+FP",
        ),
        Rc4TestCase(
            "conv1d_dw_atomic_rank3_layout_boundaries",
            "test_rewrite_conv1d_emits_atomic_rank3_layout_boundaries[True] (:828)",
            lambda: rc4_test_fp_program(Conv1dBiasModule(True), (torch.randn(1, 4, 8),)),
            "TOSA-1.0+INT+FP",
        ),
        Rc4TestCase(
            "vgf_quant_no_bias_conv_relu_cat",
            "test_rewrite_conv_vgf_quant_infers_quantized_bias_dtype_from_inputs (:911)",
            lambda: rc4_test_vgf_program(TinyConvReluCat(conv1_bias=False)),
            "TOSA-1.0+INT+FP",
        ),
        Rc4TestCase(
            # The test's dynamic H and W are multiples of three; RETAB's IR is
            # static, so the example input's shape stands for them.
            "static_padding_remainder_zero",
            "test_rewrite_conv_dynamic_keeps_static_padding_when_symbolic_remainder_is_zero"
            " (:975), static",
            lambda: rc4_test_fp_program(ConvModule(), (torch.randn(1, 3, 9, 12),)),
            "TOSA-1.0+FP",
        ),
    ]


RC4_TEST_FIXTURE = "rewrite_conv_rc4_tests"
RC4_TEST_SOURCES = CONV_SOURCES


def rc4_test_case_lines(case: Rc4TestCase) -> list[str]:
    from executorch.backends.arm._passes.rewrite_conv_pass import RewriteConvPass

    torch.manual_seed(SEED)
    ep, gm = case.build()
    before = capture_graph(gm, ep)
    for node in before.nodes:
        if node.data is not None and node.data.startswith("sha256:"):
            raise AssertionError(f"{case.name}: before-graph constant {node.name} is too large")
    rewrite = RewriteConvPass(ep)
    if case.spec is None:
        result = rewrite(gm)
    else:
        with TosaLoweringContext(TosaSpecification.create_from_string(case.spec)):
            result = rewrite(gm)
    after = capture_graph(result.graph_module, ep)
    before_lines, after_lines = before.lines(), after.lines()
    lines = ["before", *before_lines, f"after modified={str(bool(result.modified)).lower()}"]
    if after_lines == before_lines:
        lines[-1] += " same"
    else:
        lines += after_lines
    return [f"case {case.name}", f"# {case.test}", *lines, "end"]


def generate_rc4_tests() -> list[str]:
    import torchao

    cases = rc4_test_cases()
    git = getattr(executorch.version, "git_version", "unknown")
    lines = [
        "# rc4 RewriteConvPass before/after graphs from rc4's own pass tests"
        " (backends/arm/test/passes/test_rewrite_conv_pass.py), for src/layout/replay.rs.",
        "# Generated by tests/fixtures/layout_replay/gen_layout_replay.py; don't edit by hand.",
        "# Regenerate with the rc4 environment, from a neutral directory:",
        "#   cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/layout_replay/gen_layout_replay.py"
        f" --pass {RC4_TEST_FIXTURE}",
        f"# executorch {executorch.version.__version__} (git {git}), torch {torch.__version__},"
        f" torchao {torchao.__version__}, python {sys.version.split()[0]}",
        f"# seed {SEED}: {len(cases)} cases; format {FORMAT}: see the generator's docstring.",
    ]
    for relative in RC4_TEST_SOURCES + PIPELINE_COMMON_SOURCES + COMMON_SOURCES:
        lines.append(f"# rc4-source sha256 {source_sha256(relative)} {relative}")
    lines.append(f"format {FORMAT}")
    for case in cases:
        lines += rc4_test_case_lines(case)
    return lines


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
    parser.add_argument(
        "--pass",
        dest="only",
        action="append",
        choices=[p.name for p in PASSES + PIPELINE_SPECS] + [RC4_TEST_FIXTURE],
        help="write this fixture (repeatable; default: all)",
    )
    parser.add_argument("--random", type=int, default=RANDOM_CASES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--case", help="print one case of --pass instead of writing")
    args = parser.parse_args()

    for spec in PASSES:
        if args.only and spec.name not in args.only:
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

    cache = LoweringCache()
    for spec in PIPELINE_SPECS:
        if args.only and spec.name not in args.only:
            continue
        if args.case:
            matches = [
                c
                for c in spec.cases()
                if args.case == c.name or args.case.startswith(c.name + "_")
            ]
            case = max(matches, key=lambda c: len(c.name), default=None)
            if case is None:
                parser.error(f"no case {args.case} in {spec.name}")
            for name, body in pipeline_case_bodies(spec, case, cache):
                if name == args.case:
                    print("\n".join([f"case {name}", *body, "end"]))
            continue
        lines = generate_pipeline(spec, cache)
        count = sum(1 for line in lines if line.startswith("case "))
        text = "\n".join(pipeline_header(spec, count) + lines) + "\n"
        path = args.out / f"{spec.name}.txt"
        path.write_text(text)
        print(f"{path.name}: {count} cases", file=sys.stderr)

    if not args.only or RC4_TEST_FIXTURE in args.only:
        if args.case:
            matches = [case for case in rc4_test_cases() if case.name == args.case]
            if matches:
                print("\n".join(rc4_test_case_lines(matches[0])))
        else:
            lines = generate_rc4_tests()
            path = args.out / f"{RC4_TEST_FIXTURE}.txt"
            path.write_text("\n".join(lines) + "\n")
            count = sum(1 for line in lines if line.startswith("case "))
            print(f"{path.name}: {count} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
