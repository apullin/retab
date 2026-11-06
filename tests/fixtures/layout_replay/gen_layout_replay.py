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
- FuseDuplicateUsers also runs where rc4 already has TABLE ops (the
  `fuse_duplicate_users_driver` and `_late` fixtures, see `TableStagePass`):
  the real InsertTableOpsPass (and FuseEqualPlaceholdersPass) bring the graph
  to that stage first, and the TABLE nodes are turned back into the table ops
  RETAB still holds, so the fixture shows rc4's decisions in RETAB's form.
  The propagation passes run the same way (`PropagationPass`), with the TABLE
  values they changed; where rc4 leaves a 16-bit TABLE and its RESCALE apart,
  the table op holds the TABLE's int32 value and the RESCALE stays a node
  (RETAB's split form, `layout::pending`). `_without_program` fixtures build
  them without the exported program, as rc4's graph-level tests do, and the
  `_down_up` fixture runs the Down pass and then the Up pass, as phase 6 does.

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
`kind=param|buffer|constant` (constant placeholders), `target=` (a placeholder
whose target is not its name), `q=` (`none`, or `in{keys}` / `out{keys}` for
present `input_qparams` / `output_qparams` maps, `{}` when empty), `qp=` (the
value of one map entry that isn't `QARGS`, as
`qp=out0:<scale>:<zp>:<qmin>:<qmax>:<dtype>`), `special=`
(the `TosaSpecialDtype` member in `meta["tosa_special_dtype"]`), `nofuse`
(`meta["do_not_fuse_duplicate"]` is set) and `dead` (a call node the output
doesn't reach). `after modified=<bool> same` means the after graph prints
exactly like before; `after raises=<ExceptionType>` means the rc4 pass raised
(the port must then panic), and the comment line after it (`# rc4: ...`) holds
the start of its message. The P1 fixtures predate `target=`, `qp=`,
`special=`, `nofuse` and `raises`, but none of their graphs needs them.

A `touch %N %M ..` line between the before graph and `after` lists nodes whose
args were re-set (`node.args = node.args`), in that order, before the pass ran.
torch.fx moves an updated node to the end of every input's `users`
(`Node._update_args_kwargs`), so touches put users out of graph order, as the
rewires rc4's propagation passes make between their calls of the fusion pass
do. The port must replay them in its `LayoutCtx` (`LayoutCtx::set_args`).

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
  spaces, e.g. `q=in{0:torch.int8(0.0176,10,-128,127)}/out{}`. There is no
  `qp=`.
- `data=hex:<bytes>` on every constant placeholder (and `get_attr` node): the
  tensor's contiguous little-endian bytes in its own dtype, or
  `data=sha256:<digest>` of those bytes above 4 KiB. The generator refuses a
  before graph with a sha256 constant, so every replay imports real data.
- vals that aren't tensors print as tuples, `(torch.int8[1, 3, 4, 4],
  torch.int64[1, 3, 4, 4])` for `max_pool2d_with_indices` and `(4, 2, 4, 2)`
  for CONST_SHAPE.
- `get_attr NAME` nodes print like placeholders; devices, layouts and memory
  formats print as `device:cpu`, `layout:torch.strided` and
  `memory_format:torch.contiguous_format`.

They print two annotations of the hand-built fixtures their own way, before
`q=`: `special=INT48|INT4|SHAPE|...`, the node's `tosa_special_dtype`, and
`no_fuse` (not `nofuse`), the `do_not_fuse_duplicate` flag. They record no
placeholder targets.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import executorch
import executorch.backends.arm.tosa.dialect  # noqa: F401  registers exir_ops.backend.tosa
import executorch.backends.transforms.channels_last_ops  # noqa: F401  registers exir_ops.edge.channels_last
import executorch.version
import torch
from executorch.backends.arm._passes import (
    PropagateViewCopyPermuteDownPass,
    PropagateViewCopyPermuteUpPass,
    CanonicalizeViewCopyPermutePass,
    FuseDuplicateUsersPass,
    FuseIdenticalInputTransformsPass,
    FuseViewCopyTransformPass,
    MatchArgRanksPass,
    MoveDataMovementOpsToSmallerDtypePass,
    NormalizeDelegateIOLayoutPass,
    RemovePermutesAroundElementwiseTosaOps,
    RewriteHighRankSingletonPermutePass,
)
from executorch.backends.arm._passes.fuse_equal_placeholders_pass import (
    FuseEqualPlaceholdersPass,
)
from executorch.backends.arm._passes.fuse_duplicate_users_pass import (
    TOSA_EXCLUDED_TARGETS,
)
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass, TableOps
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.transforms.fuse_duplicate_users_pass import (
    DO_NOT_FUSE_DUPLICATE_META_KEY,
    FuseDuplicateUsersPass as SharedFuseDuplicateUsersPass,
)
from executorch.backends.transforms.fuse_identical_input_transforms_pass import (
    FuseIdenticalInputTransformsPass as SharedFuseIdenticalInputTransformsPass,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult
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
        # Nodes to touch before the pass runs (see the `touch` line).
        self.touches: list[torch.fx.Node] = []

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

    def touch(self, node) -> None:
        """Re-set `node`'s args before the pass runs, which moves it to the end
        of its inputs' users."""
        self.touches.append(node)

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


def fmt_qvalues(meta) -> list[str]:
    """`qp=` notes for the qparam entries that aren't `QARGS`."""
    notes = []
    for key, label in (("input_qparams", "in"), ("output_qparams", "out")):
        for index, args in sorted(meta.get(key, {}).items()):
            if args == QARGS:
                continue
            if args.per_channel or isinstance(args.scale, list):
                raise TypeError(f"cannot print per-channel qparams {args}")
            notes.append(
                f"qp={label}{index}:{float(args.scale)!r}:{args.zp}:{args.qmin}:{args.qmax}:{args.dtype}"
            )
    return notes


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
        if node.op == "placeholder" and node.target != node.name:
            notes.append(f"target={node.target}")
        notes.append(f"q={fmt_qparams(node.meta)}")
        notes.extend(fmt_qvalues(node.meta))
        special = node.meta.get(TosaSpecialDtype.meta_key())
        if special is not None:
            notes.append(f"special={special.name}")
        if node.meta.get(DO_NOT_FUSE_DUPLICATE_META_KEY, False):
            notes.append("nofuse")
        if node.op == "call_function" and node not in reached:
            notes.append("dead")
        lines.append(f"%{pos[node]} = {head} : {fmt_val(val)} {' '.join(notes)}")
    return lines


# --------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------


class PassSpec:
    def __init__(
        self, name, make, sources, structured, random_profile, random_cases=None, tosa_spec=None
    ):
        self.name = name
        self.make: Callable[[ExportedProgram], object] = make
        self.sources = sources
        self.structured = structured
        self.random_profile = random_profile
        # More random graphs than RANDOM_CASES: where torch rejects enough of
        # them that fewer would be left, or where few graphs show what the
        # fixture is about.
        self.random_cases = RANDOM_CASES if random_cases is None else random_cases
        # 16-bit TABLE ops need the int16 extension.
        self.tosa_spec = TOSA_SPEC if tosa_spec is None else tosa_spec


class BuildError(Exception):
    """torch rejected a random graph while it was being built."""


def raised(error: Exception) -> list[str]:
    """The `after raises=` line for an exception rc4's pass raised, and the
    comment after it. The message goes in the comment, which the parser skips:
    tests that check the port fails for rc4's reason read it from the text."""
    message = " ".join(str(error).split())[:300]
    return [f"after raises={type(error).__name__}", f"# rc4: {message}"]


def touch(gm: torch.fx.GraphModule, touches: Sequence[torch.fx.Node]) -> list[str]:
    """Touch `touches` in order and return the `touch` line, if any."""
    if not touches:
        return []
    position = {node: index for index, node in enumerate(gm.graph.nodes)}
    for node in touches:
        node.args = node.args
    return ["touch " + " ".join(f"%{position[node]}" for node in touches)]


def run_case(spec: PassSpec, build: Callable[[Builder], None]) -> list[str]:
    # TOSA dialect ops need the context to build, and the pass manager runs
    # every pass inside it (`ArmPassManager._tosa_context`).
    with TosaLoweringContext(TosaSpecification.create_from_string(spec.tosa_spec)):
        builder = Builder()
        try:
            build(builder)
            ep = builder.exported_program()
        except Exception as error:
            raise BuildError(f"{type(error).__name__}: {error}") from error
        kinds = placeholder_kinds(ep)
        before = dump(ep.graph_module, kinds)
        touches = touch(ep.graph_module, builder.touches)
        graph_pass = spec.make(ep)
        try:
            result = graph_pass(ep.graph_module)
        except Exception as error:  # noqa: BLE001 - recorded for the port to match
            return ["before", *before, *touches, *raised(error)]
    after = dump(result.graph_module, kinds)
    lines = ["before", *before, *touches, f"after modified={str(bool(result.modified)).lower()}"]
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
# APU-160: FuseDuplicateUsers, FuseIdenticalInputTransforms and
# MoveDataMovementOpsToSmallerDtype
# --------------------------------------------------------------------------


def rescale(b, x, dtype, scales=(0.5,), in_zp=0, out_zp=0, **kwargs):
    return b.call(T.RESCALE.default, x, dtype, list(scales), in_zp, out_zp, **kwargs)


def set_special(node, special=TosaSpecialDtype.INT48) -> None:
    node.meta[TosaSpecialDtype.meta_key()] = special


def mark_no_fuse(node) -> None:
    node.meta[DO_NOT_FUSE_DUPLICATE_META_KEY] = True


def avg_pool(b, x):
    """`torch.nn.AvgPool2d(1)` as edge lowers it (test_fuse_duplicate_users_pass.py)."""
    return b.call(A.avg_pool2d.default, x, [1, 1], [1, 1], [0, 0], False, True)


def fuse_duplicate_users_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    @case
    def docstring_example(b):
        x = b.input("x", (2, 3))
        bias = b.input("bias", (2, 3))
        y = b.call(A.relu.default, x)
        b.output([b.call(A.add.Tensor, y, bias), b.call(A.add.Tensor, y, bias)])

    @case
    def three_avg_pools(b):
        x = b.input("x", (1, 1, 1, 1))
        s = b.call(A.add.Tensor, avg_pool(b, x), avg_pool(b, x))
        b.output([b.call(A.add.Tensor, s, avg_pool(b, x))])

    @case
    def avg_pool_chains(b):
        x = b.input("x", (1, 1, 1, 1))
        first = avg_pool(b, avg_pool(b, x))
        second = avg_pool(b, avg_pool(b, x))
        third = avg_pool(b, avg_pool(b, x))
        s = b.call(A.add.Tensor, first, second)
        b.output([b.call(A.add.Tensor, s, third)])

    @case
    def users_not_in_node_order(b):
        x = b.input("x", (1,))
        y = b.input("y", (1,))
        later = b.call(A.add.Tensor, x, y)
        with b.graph.inserting_before(later):
            earlier = b.call(A.add.Tensor, x, y)
            consumer = b.call(A.neg.default, earlier)
        b.first_call = earlier
        b.output([consumer, later])

    @case
    def do_not_fuse_marker_on_the_first(b):
        x = b.input("x", (1,))
        y = b.input("y", (1,))
        first = b.call(A.add.Tensor, x, y)
        mark_no_fuse(first)
        b.output([first, b.call(A.add.Tensor, x, y)])

    @case
    def do_not_fuse_marker_on_the_second(b):
        x = b.input("x", (1,))
        y = b.input("y", (1,))
        first = b.call(A.add.Tensor, x, y)
        second = b.call(A.add.Tensor, x, y)
        mark_no_fuse(second)
        third = b.call(A.add.Tensor, x, y)
        b.output([first, second, third])

    @case
    def erased_node_still_queued(b):
        x = b.input("x", (1,))
        n0 = b.call(A.neg.default, x)
        n1 = b.call(A.neg.default, x)
        b.output([b.call(A.relu.default, n0), b.call(A.relu.default, n1)])

    @case
    def identical_rescale_users_are_kept(b):
        x = b.input("x", (1,), dtype=torch.int8)
        b.output([rescale(b, x, torch.int32, [1.0], 16, 0), rescale(b, x, torch.int32, [1.0], 16, 0)])

    @case
    def users_of_kept_rescales_still_fuse(b):
        x = b.input("x", (2,), dtype=torch.int8)
        r = rescale(b, x, torch.int32)
        b.output([b.call(A.neg.default, r), b.call(A.neg.default, r), rescale(b, x, torch.int32)])

    @case
    def scalar_args_compare_by_value(b):
        x = b.input("x", (2,))
        b.output(
            [
                b.call(A.mul.Tensor, x, 2),
                b.call(A.mul.Tensor, x, 2.0),
                b.call(A.mul.Tensor, x, True),
                b.call(A.mul.Tensor, x, 1),
                b.call(A.mul.Tensor, x, 1.5),
                b.call(A.mul.Tensor, x, 0.0),
                b.call(A.mul.Tensor, x, -0.0),
                b.call(A.mul.Tensor, x, False),
            ]
        )

    @case
    def separate_nan_scalars_never_fuse(b):
        x = b.input("x", (2,))
        b.output([b.call(A.add.Tensor, x, float("nan")), b.call(A.add.Tensor, x, float("nan"))])

    @case
    def one_shared_nan_object_fuses(b):
        x = b.input("x", (2,))
        nan = float("nan")
        b.output([b.call(A.add.Tensor, x, nan), b.call(A.add.Tensor, x, nan)])

    @case
    def kwargs_take_part(b):
        x = b.input("x", (2,))
        y = b.input("y", (2,))
        b.output(
            [
                b.call(A.add.Tensor, x, y, alpha=2),
                b.call(A.add.Tensor, x, y, alpha=2.0),
                b.call(A.add.Tensor, x, y),
                b.call(A.add.Tensor, x, y, alpha=1),
            ]
        )

    @case
    def argument_order_matters(b):
        x = b.input("x", (2,))
        y = b.input("y", (2,))
        b.output([b.call(A.add.Tensor, x, y), b.call(A.add.Tensor, y, x)])

    @case
    def different_targets_same_args(b):
        x = b.input("x", (2,))
        y = b.input("y", (2,))
        b.output([b.call(A.add.Tensor, x, y), b.call(A.sub.Tensor, x, y), b.call(A.mul.Tensor, x, y)])

    @case
    def a_single_user_is_skipped(b):
        x = b.input("x", (2,))
        b.output([b.call(A.neg.default, x)])

    @case
    def duplicates_returned_together(b):
        x = b.input("x", (2, 3))
        b.output([b.call(A.relu.default, x), b.call(A.relu.default, x), x])

    @case
    def list_args_with_nodes(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3))
        b.output(
            [
                b.call(A.cat.default, [x, y], 0),
                b.call(A.cat.default, [x, y], 0),
                b.call(A.cat.default, [x, y], 1),
                b.call(A.cat.default, [y, x], 0),
            ]
        )

    @case
    def a_fusion_exposes_list_users(b):
        x = b.input("x", (2, 3))
        n0 = b.call(A.neg.default, x)
        n1 = b.call(A.neg.default, x)
        b.output([b.call(A.cat.default, [n0, x], 1), b.call(A.cat.default, [n1, x], 1)])

    @case
    def a_fusion_rewires_a_later_group_member(b):
        x = b.input("x", (2,))
        a0 = b.call(A.add.Tensor, x, x)
        a1 = b.call(A.add.Tensor, x, x)
        c0 = b.call(A.add.Tensor, x, a0)
        c1 = b.call(A.add.Tensor, x, a1)
        b.output([c0, c1])

    @case
    def representative_is_earliest_after_a_cascade(b):
        x = b.input("x", (2,))
        n0 = b.call(A.neg.default, x)
        later_user = b.call(A.relu.default, n0)
        n1 = b.call(A.neg.default, x)
        earlier_user = b.call(A.relu.default, n1)
        b.output([earlier_user, later_user])

    @case
    def dim_order_copies(b):
        x = b.input("x", (2, 3))
        b.output(
            [
                b.call(D._clone_dim_order.default, x),
                b.call(D._clone_dim_order.default, x),
                b.call(D._to_dim_order_copy.default, x, dtype=torch.float16),
                b.call(D._to_dim_order_copy.default, x, dtype=torch.float16),
                b.call(D._to_dim_order_copy.default, x, dtype=torch.float32),
            ]
        )

    @case
    def tosa_ops_other_than_rescale_fuse(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        y = b.input("y", (2, 3), dtype=torch.int32)
        b.output([b.call(T.ADD.default, x, y), b.call(T.ADD.default, x, y), b.call(T.SUB.default, x, y)])

    @case
    def dead_duplicates_fuse_too(b):
        x = b.input("x", (2,))
        live = b.call(A.neg.default, x)
        b.call(A.neg.default, x)
        b.output([live])

    @case
    def views_and_permutes(b):
        x = b.input("x", (2, 3, 4))
        b.output(
            [
                b.view(x, [6, 4]),
                b.view(x, [6, 4]),
                b.view(x, [-1, 4]),
                b.permute(x, [0, 2, 1]),
                b.permute(x, [0, 2, 1]),
                b.permute(x, [0, -1, 1]),
            ]
        )

    @case
    def qparams_do_not_take_part(b):
        x = b.input("x", (2,), dtype=torch.int8)
        first = b.call(A.neg.default, x)
        set_qparams(first, inputs=[0], outputs=[0])
        b.output([first, b.call(A.neg.default, x)])

    return cases


# FuseDuplicateUsers where rc4 already has TABLE ops ---------------------

# Meta keys the table-stage composite uses to find its way back; never dumped.
_TABLE_OF = "retab_replay_table_of"
_TABLE_RESCALE = "retab_replay_table_rescale"


def qargs(scale, zp=0, dtype=torch.int8, qmin=None, qmax=None) -> QuantArgs:
    info = torch.iinfo(dtype)
    return QuantArgs(
        scale=scale,
        zp=zp,
        qmin=info.min if qmin is None else qmin,
        qmax=info.max if qmax is None else qmax,
        dtype=dtype,
    )


def table_op(b, target, x, qin, qout, *args, **kwargs):
    """A quantized table op as RETAB holds it where rc4 has a TABLE
    (`layout::pending`): one input and one output qparam, and the value
    InsertTableOpsPass's TABLE (or its RESCALE) returns, `x`'s shape in the
    output qparam's dtype, so rc4's retraces leave downstream values alone.
    The op is never executed on integers, which torch rejects for some
    (leaky_relu with a float slope)."""
    node = b.graph.create_node("call_function", target, (x, *args), kwargs)
    if b.first_call is None:
        b.first_call = node
    with b.mode:
        node.meta["val"] = torch.empty(tuple(x.meta["val"].shape), dtype=qout.dtype)
    node.meta["input_qparams"] = {0: qin}
    node.meta["output_qparams"] = {0: qout}
    return node


def _is_table_op(node) -> bool:
    """InsertTableOpsPass's test (insert_table_ops.py:359-366)."""
    return (
        node.op == "call_function"
        and node.target in set(TableOps.included_ops())
        and len(node.meta.get("input_qparams", {})) > 0
        and len(node.meta.get("output_qparams", {})) > 0
    )


class TableStagePass:
    """FuseDuplicateUsers at a stage where rc4 already has TABLE ops.

    InsertTableOpsPass (arm_pass_manager.py:708), and FuseEqualPlaceholdersPass
    (`:755`) when `merge_constants` holds, bring the graph to the stage; then
    `make_fdu()` runs as its rc4 caller constructs it, and every TABLE (with
    the RESCALE a 16-bit table gets) becomes the table op it replaced again,
    reading the TABLE's input, so the fixture shows rc4's decisions in the
    form RETAB keeps (`layout::pending`). Where rc4 fused 16-bit TABLEs, each
    of their RESCALEs becomes its own table op, which RETAB never fuses
    either; only `modified` tells. `modified` is FuseDuplicateUsers' own.
    """

    def __init__(self, ep, make_fdu, merge_constants):
        self.ep = ep
        self.make_fdu = make_fdu
        self.merge_constants = merge_constants
        merge = "FuseEqualPlaceholdersPass + " if merge_constants else ""
        self.rc4_name = f"InsertTableOpsPass + {merge}FuseDuplicateUsersPass"

    def __call__(self, gm):
        originals = {node.name: node for node in gm.graph.nodes if _is_table_op(node)}
        names = {node.name for node in gm.graph.nodes}
        gm = InsertTableOpsPass(self.ep)(gm).graph_module
        for node in gm.graph.nodes:
            if node.name in names:
                continue
            if node.target == T.TABLE.default:
                constant = node.args[1].name
                node.meta[_TABLE_OF] = constant.removeprefix("b_").removesuffix("_table_constant")
            elif node.target == T.RESCALE.default:
                (table,) = node.all_input_nodes
                node.meta[_TABLE_OF] = table.meta[_TABLE_OF]
                node.meta[_TABLE_RESCALE] = True
        if self.merge_constants:
            gm = FuseEqualPlaceholdersPass(self.ep)(gm).graph_module
        result = self.make_fdu()(gm)
        gm = result.graph_module
        graph = gm.graph
        for table in [n for n in graph.nodes if n.target == T.TABLE.default]:
            rescales = [user for user in table.users if user.meta.get(_TABLE_RESCALE)]
            for node in rescales or [table]:
                original = originals[node.meta[_TABLE_OF]]
                with graph.inserting_before(node):
                    restored = graph.create_node(
                        "call_function",
                        original.target,
                        (table.args[0], *original.args[1:]),
                        dict(original.kwargs),
                    )
                restored.meta = dict(original.meta)
                node.replace_all_uses_with(restored)
                graph.erase_node(node)
            if rescales and not table.users:
                graph.erase_node(table)
        for node in list(graph.nodes):
            if node.op == "placeholder" and node.name.startswith("b_") and not node.users:
                graph.erase_node(node)
        return PassResult(gm, result.modified)


# The table ops the table-stage cases use, with their extra arguments; elu
# reads rc4-only meta (insert_table_ops.py:128-134) and is left out.
TABLE_TARGETS = [
    (A.sigmoid.default, (), {}),
    (A.tanh.default, (), {}),
    (A.exp.default, (), {}),
    (A.ceil.default, (), {}),
    (A.floor.default, (), {}),
    (A.round.default, (), {}),
    (A.rsqrt.default, (), {}),
    (A.hardswish.default, (), {}),
    (A.gelu.default, (), {}),
    (A.gelu.default, (), {"approximate": "tanh"}),
    (A.leaky_relu.default, (0.125,), {}),
    (A.pow.Tensor_Scalar, (2,), {}),
    (A.pow.Tensor_Scalar, (3,), {}),
    (A.softplus.default, (1.0, 20.0), {}),
]
# Int8 qparams the random table cases draw from: equal draws give equal
# tables, and the scale-1 and saturating ones give tables that coincide.
TABLE_QPARAMS = [
    qargs(0.5),
    qargs(0.25),
    qargs(1.0),
    qargs(0.5, 3),
    qargs(0.5, qmin=-127),
    qargs(1 / 256, -128),
]


def table_stage_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    def two(b, first, second, target=A.sigmoid.default, *args):
        x = b.input("x", (2, 3), dtype=torch.int8)
        b.output([table_op(b, target, x, *first, *args), table_op(b, target, x, *second, *args)])

    @case
    def equal_tables(b):
        two(b, (qargs(0.5), qargs(0.5)), (qargs(0.5), qargs(0.5)))

    @case
    def output_scales_differ(b):
        two(b, (qargs(0.5), qargs(0.5)), (qargs(0.5), qargs(0.25)))

    @case
    def input_zero_points_differ(b):
        two(b, (qargs(0.5), qargs(0.5)), (qargs(0.5, 3), qargs(0.5)))

    @case
    def saturating_tables_coincide(b):
        two(b, (qargs(0.5), qargs(1.0)), (qargs(0.5, qmin=-127), qargs(1.0)))

    @case
    def ceil_and_floor_of_integer_codes_coincide(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        ceil = table_op(b, A.ceil.default, x, qargs(1.0), qargs(1.0))
        floor = table_op(b, A.floor.default, x, qargs(1.0), qargs(1.0))
        b.output([floor, ceil])

    @case
    def ceil_and_floor_of_half_codes_differ(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        ceil = table_op(b, A.ceil.default, x, qargs(0.5), qargs(1.0))
        floor = table_op(b, A.floor.default, x, qargs(0.5), qargs(1.0))
        b.output([ceil, floor])

    @case
    def tables_of_different_inputs(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        y = b.input("y", (2, 3), dtype=torch.int8)
        b.output(
            [
                table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5)),
                table_op(b, A.sigmoid.default, y, qargs(0.5), qargs(0.5)),
            ]
        )

    @case
    def three_tables_two_equal(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        b.output(
            [
                table_op(b, A.tanh.default, x, qargs(0.5), qargs(0.25)),
                table_op(b, A.tanh.default, x, qargs(0.5), qargs(1 / 256, -128)),
                table_op(b, A.tanh.default, x, qargs(0.5), qargs(0.25)),
            ]
        )

    @case
    def users_of_one_table_fuse(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        t = table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5))
        b.output([b.call(A.neg.default, t), b.call(A.neg.default, t)])

    @case
    def a_fusion_exposes_equal_tables(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        n0 = b.call(A.neg.default, x)
        n1 = b.call(A.neg.default, x)
        b.output(
            [
                table_op(b, A.exp.default, n0, qargs(0.5), qargs(0.5)),
                table_op(b, A.exp.default, n1, qargs(0.5), qargs(0.5)),
            ]
        )

    @case
    def equal_tables_expose_equal_users(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        t0 = table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5))
        t1 = table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5))
        b.output([b.call(A.abs.default, t0), b.call(A.abs.default, t1)])

    @case
    def a_marked_table(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        first = table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5))
        mark_no_fuse(first)
        b.output([first, table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5))])

    @case
    def a_table_and_a_float_op(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        b.output(
            [
                table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5)),
                b.call(A.sigmoid.default, x),
                b.call(A.sigmoid.default, x),
            ]
        )

    @case
    def table_arguments(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        q = (qargs(0.5), qargs(0.5))
        b.output(
            [
                table_op(b, A.pow.Tensor_Scalar, x, *q, 2),
                table_op(b, A.pow.Tensor_Scalar, x, *q, 2),
                table_op(b, A.pow.Tensor_Scalar, x, *q, 3),
                table_op(b, A.leaky_relu.default, x, *q, 0.125),
                table_op(b, A.leaky_relu.default, x, *q, 0.25),
                table_op(b, A.gelu.default, x, *q),
                table_op(b, A.gelu.default, x, *q, approximate="tanh"),
                table_op(b, A.gelu.default, x, *q, approximate="none"),
                table_op(b, A.softplus.default, x, *q, 1.0, 20.0),
                table_op(b, A.softplus.default, x, *q, 2.0, 20.0),
            ]
        )

    @case
    def a_table_of_a_table(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        inner = [table_op(b, A.exp.default, x, qargs(0.25), qargs(0.5)) for _ in range(2)]
        b.output([table_op(b, A.tanh.default, t, qargs(0.5), qargs(0.5)) for t in inner])

    @case
    def rescale_users_of_tables(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        t0 = table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5))
        t1 = table_op(b, A.sigmoid.default, x, qargs(0.5), qargs(0.5))
        b.output([rescale(b, t0, torch.int32), rescale(b, t1, torch.int32)])

    @case
    def int16_tables_keep_one_rescale_each(b):
        x = b.input("x", (2, 3), dtype=torch.int16)
        q = (qargs(1 / 1024, dtype=torch.int16), qargs(1 / 32768, dtype=torch.int16))
        b.output([table_op(b, A.sigmoid.default, x, *q), table_op(b, A.sigmoid.default, x, *q)])

    @case
    def int16_tables_that_differ(b):
        x = b.input("x", (2, 3), dtype=torch.int16)
        b.output(
            [
                table_op(b, A.tanh.default, x, qargs(1 / 1024, dtype=torch.int16), qargs(1 / 32768, dtype=torch.int16)),
                table_op(b, A.tanh.default, x, qargs(1 / 512, dtype=torch.int16), qargs(1 / 32768, dtype=torch.int16)),
            ]
        )

    return cases


def fuse_identical_input_transforms_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    def binary(b, x_shape, y_shape, x_transforms, y_transforms, op=A.add.Tensor, dtype=torch.float32):
        x = b.input("x", x_shape, dtype=dtype)
        y = b.input("y", y_shape, dtype=dtype)
        for target, arg in x_transforms:
            x = b.call(target, x, list(arg))
        for target, arg in y_transforms:
            y = b.call(target, y, list(arg))
        b.output([b.call(op, x, y)])

    V = A.view_copy.default
    P = A.permute_copy.default

    # rc4's test_fuse_identical_input_transforms_pass.py graphs.
    cases["sinks_view"] = lambda b: binary(b, (6,), (6,), [(V, [2, 3])], [(V, [2, 3])])
    cases["sinks_permute"] = lambda b: binary(
        b, (2, 3, 4), (2, 3, 4), [(P, [0, 2, 1])], [(P, [0, 2, 1])]
    )

    @case
    def sinks_where_input_permutes(b):
        c = b.input("condition", (2, 3, 4), dtype=torch.bool)
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        b.output([b.call(A.where.self, b.permute(c, [0, 2, 1]), b.permute(x, [0, 2, 1]), b.permute(y, [0, 2, 1]))])

    @case
    def sinks_cat_input_views(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        b.output([b.call(A.cat.default, [b.view(x, [6, 4]), b.view(y, [6, 4])], 1)])

    cases["rejects_cat_mixed_transforms"] = lambda b: concat(b, (2, 3, 4), (2, 3, 4), (V, [2, 3, 4]), (P, [0, 1, 2]), 0)
    cases["rejects_cat_different_permutes"] = lambda b: concat(
        b, (2, 3, 4), (2, 4, 3), (P, [0, 1, 2]), (P, [0, 2, 1]), 0
    )

    def concat(b, x_shape, y_shape, x_transform, y_transform, dim):
        x = b.input("x", x_shape)
        y = b.input("y", y_shape)
        tx = b.call(x_transform[0], x, list(x_transform[1]))
        ty = b.call(y_transform[0], y, list(y_transform[1]))
        b.output([b.call(A.cat.default, [tx, ty], dim)])

    @case
    def rejects_shared_binary_input_transform(b):
        x = b.input("x", (6,))
        y = b.input("y", (6,))
        vx = b.view(x, [2, 3])
        b.output([b.call(A.add.Tensor, vx, b.view(y, [2, 3])), vx])

    @case
    def rejects_shared_cat_input_transform(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        vx = b.view(x, [6, 4])
        b.output([b.call(A.cat.default, [vx, b.view(y, [6, 4])], 1), vx])

    cases["rejects_different_views"] = lambda b: binary(b, (6,), (6,), [(V, [2, 3])], [(V, [1, 2, 3])])
    cases["rejects_different_permutes"] = lambda b: binary(
        b, (2, 1, 3), (2, 1, 3), [(P, [0, 2, 1])], [(P, [0, 1, 2])]
    )
    cases["sinks_view_permute_chain"] = lambda b: binary(
        b, (6, 4), (6, 4), [(V, [2, 3, 4]), (P, [0, 2, 1])], [(V, [2, 3, 4]), (P, [0, 2, 1])]
    )
    cases["sinks_permute_view_chain"] = lambda b: binary(
        b, (2, 3, 4), (2, 3, 4), [(P, [0, 2, 1]), (V, [2, 2, 6])], [(P, [0, 2, 1]), (V, [2, 2, 6])]
    )
    for index, (xs, ys, vs) in enumerate([((1, 6), (6,), (2, 3)), ((1, 1, 6), (1, 6), (3, 2))]):
        cases[f"sinks_views_through_rank_broadcasts_{index}"] = (
            lambda b, xs=xs, ys=ys, vs=vs: binary(b, xs, ys, [(V, vs)], [(V, vs)])
        )
    for index, (xs, ys, vs) in enumerate(
        [((1, 6), (6, 1), (2, 3)), ((1, 1, 6), (1, 6, 1), (3, 2)), ((6,), (6, 1), (3, 2))]
    ):
        cases[f"rejects_views_through_expanding_broadcast_{index}"] = (
            lambda b, xs=xs, ys=ys, vs=vs: binary(b, xs, ys, [(V, vs)], [(V, vs)])
        )
    for index, (xs, ys) in enumerate([((1, 2, 3), (4, 2, 3)), ((4, 1, 3), (1, 2, 3)), ((4, 2, 1), (1, 1, 3))]):
        cases[f"sinks_permutes_through_broadcasts_{index}"] = (
            lambda b, xs=xs, ys=ys: binary(b, xs, ys, [(P, [0, 2, 1])], [(P, [0, 2, 1])])
        )

    @case
    def cat_rejects_different_view_inputs(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (1, 6, 4))
        b.output([b.call(A.cat.default, [b.view(x, [6, 4]), b.view(y, [6, 4])], 1)])

    # Branches of `_sink_identical_input_transforms` and its helpers.
    @case
    def non_target_op_is_skipped(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        b.output([b.call(A.maximum.default, b.permute(x, [0, 2, 1]), b.permute(y, [0, 2, 1]))])

    @case
    def one_transform_used_twice_is_one_input(b):
        x = b.input("x", (2, 3, 4))
        p = b.permute(x, [0, 2, 1])
        b.output([b.call(A.add.Tensor, p, p)])

    @case
    def scalar_operand_is_one_input(b):
        x = b.input("x", (2, 3, 4))
        b.output([b.call(A.mul.Tensor, b.permute(x, [0, 2, 1]), 2.0)])

    @case
    def no_transform_input(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3))
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def view_and_permute_differ(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (3, 2))
        b.output([b.call(A.add.Tensor, b.view(x, [3, 2]), b.permute(y, [0, 1]))])

    @case
    def layout_invariant_operand(b):
        x = b.input("x", (2, 3, 4))
        ones = b.input("ones", (1, 1, 1))
        b.output([b.call(A.mul.Tensor, b.permute(x, [0, 2, 1]), ones)])

    @case
    def layout_invariant_first_operand(b):
        x = b.input("x", (2, 3, 4))
        ones = b.input("ones", (1, 1, 1))
        b.output([b.call(A.sub.Tensor, ones, b.permute(x, [2, 0, 1]))])

    @case
    def layout_invariant_rank_must_match(b):
        x = b.input("x", (2, 3, 4))
        ones = b.input("ones", (1, 1))
        b.output([b.call(A.add.Tensor, b.permute(x, [0, 2, 1]), ones)])

    @case
    def non_unit_operand_blocks(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (1, 4, 1))
        b.output([b.call(A.add.Tensor, b.permute(x, [0, 2, 1]), y)])

    @case
    def layout_invariant_view_operand(b):
        x = b.input("x", (1, 6))
        ones = b.input("ones", (1, 1))
        b.output([b.call(A.add.Tensor, b.view(x, [2, 3]), ones)])

    @case
    def layout_invariant_where_condition(b):
        c = b.input("c", (1, 1, 1), dtype=torch.bool)
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        b.output([b.call(A.where.self, c, b.permute(x, [1, 0, 2]), b.permute(y, [1, 0, 2]))])

    @case
    def cat_rejects_a_layout_invariant_input(b):
        x = b.input("x", (1, 1, 1))
        ones = b.input("ones", (1, 1, 1))
        b.output([b.call(A.cat.default, [b.permute(x, [0, 2, 1]), ones], 0)])

    @case
    def view_producers_that_do_not_broadcast(b):
        x = b.input("x", (6,))
        y = b.input("y", (2, 3))
        b.output([b.call(A.add.Tensor, b.view(x, [2, 3]), b.view(y, [2, 3]))])

    @case
    def view_with_an_inferred_dim(b):
        x = b.input("x", (2, 6))
        y = b.input("y", (2, 6))
        b.output([b.call(A.add.Tensor, b.view(x, [-1, 3]), b.view(y, [-1, 3]))])

    @case
    def negative_permute_dims(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        b.output([b.call(A.mul.Tensor, b.permute(x, [-1, 0, -2]), b.permute(y, [-1, 0, -2]))])

    @case
    def comparison_output_is_bool(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        b.output([b.call(A.gt.Tensor, b.permute(x, [2, 1, 0]), b.permute(y, [2, 1, 0]))])

    @case
    def integer_ops(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        y = b.input("y", (2, 3), dtype=torch.int32)
        px, py = b.permute(x, [1, 0]), b.permute(y, [1, 0])
        vx, vy = b.view(x, [6]), b.view(y, [6])
        b.output(
            [
                b.call(A.bitwise_and.Tensor, px, py),
                b.call(A.bitwise_left_shift.Tensor, vx, vy),
            ]
        )

    @case
    def kwargs_are_kept(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3))
        b.output([b.call(A.sub.Tensor, b.permute(x, [1, 0]), b.permute(y, [1, 0]), alpha=2.0)])

    @case
    def users_read_the_new_transform(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        a = b.call(A.add.Tensor, b.permute(x, [0, 2, 1]), b.permute(y, [0, 2, 1]))
        b.output([b.call(A.relu.default, a), b.call(A.neg.default, a), a])

    @case
    def sunk_transforms_meet_again(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3))
        z = b.input("z", (2, 3))
        w = b.input("w", (2, 3))
        a = b.call(A.add.Tensor, b.permute(x, [1, 0]), b.permute(y, [1, 0]))
        c = b.call(A.mul.Tensor, b.permute(z, [1, 0]), b.permute(w, [1, 0]))
        b.output([b.call(A.sub.Tensor, a, c)])

    @case
    def qparams_follow_the_transform(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        y = b.input("y", (2, 3), dtype=torch.int8)
        px, py = b.permute(x, [1, 0]), b.permute(y, [1, 0])
        set_qparams(px, inputs=[0], outputs=[0])
        a = b.call(A.add.Tensor, px, py)
        set_qparams(a, inputs=[0, 1], outputs=[0])
        b.output([a])

    @case
    def special_dtype_follows_the_transform(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        y = b.input("y", (2, 3), dtype=torch.int32)
        px, py = b.permute(x, [1, 0]), b.permute(y, [1, 0])
        set_special(px)
        b.output([b.call(A.add.Tensor, px, py)])

    @case
    def cat_permutes_map_the_dim(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 5, 4))
        b.output([b.call(A.cat.default, [b.permute(x, [0, 2, 1]), b.permute(y, [0, 2, 1])], 2)])

    @case
    def cat_dim_as_kwarg(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 5, 4))
        b.output([b.call(A.cat.default, [b.permute(x, [0, 2, 1]), b.permute(y, [0, 2, 1])], dim=-1)])

    @case
    def cat_without_dim(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (5, 3, 4))
        b.output([b.call(A.cat.default, [b.permute(x, [0, 2, 1]), b.permute(y, [0, 2, 1])])])

    @case
    def cat_three_inputs(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 1))
        z = b.input("z", (2, 4))
        b.output([b.call(A.cat.default, [b.permute(x, [1, 0]), b.permute(y, [1, 0]), b.permute(z, [1, 0])], 0)])

    @case
    def cat_repeated_input(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3))
        px = b.permute(x, [1, 0])
        b.output([b.call(A.cat.default, [px, b.permute(y, [1, 0]), px], 0)])

    @case
    def cat_producers_differ_off_the_dim(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (3, 3))
        b.output([b.call(A.cat.default, [b.view(x, [6]), b.view(y, [9])], 0)])

    @case
    def cat_view_dim_maps_to_two_dims(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        b.output([b.call(A.cat.default, [b.view(x, [6, 4]), b.view(y, [6, 4])], 0)])

    @case
    def cat_view_that_splits_a_dim(b):
        x = b.input("x", (6, 4))
        y = b.input("y", (6, 4))
        b.output([b.call(A.cat.default, [b.view(x, [2, 3, 4]), b.view(y, [2, 3, 4])], 2)])

    @case
    def cat_view_with_singletons(b):
        x = b.input("x", (3, 4))
        y = b.input("y", (3, 4))
        b.output([b.call(A.cat.default, [b.view(x, [1, 3, 4]), b.view(y, [1, 3, 4])], 0)])

    @case
    def cat_views_numel_mismatch(b):
        x = b.input("x", (2, 4))
        y = b.input("y", (2, 2, 2))
        b.output([b.call(A.cat.default, [b.view(x, [8]), b.view(y, [8])], 0)])

    @case
    def concatenate_target(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3))
        b.output([b.call(A.concatenate.default, [b.permute(x, [1, 0]), b.permute(y, [1, 0])], 1)])

    @case
    def dead_code_goes_when_modified(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3))
        b.call(A.neg.default, x)
        b.output([b.call(A.add.Tensor, b.permute(x, [1, 0]), b.permute(y, [1, 0]))])

    @case
    def dead_code_stays_when_unmodified(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3))
        b.call(A.neg.default, x)
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def rank0_views(b):
        x = b.input("x", (1,))
        y = b.input("y", (1,))
        b.output([b.call(A.add.Tensor, b.view(x, []), b.view(y, []))])

    # NormalizeTransformInputPlaceholdersPass.
    @case
    def transform_named_placeholder_target_is_normalized(b):
        x = b.input("aten_view_copy_default_2", (2, 3))
        x.target = "aten_view_copy_default"
        y = b.input("y", (2, 3))
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def permute_named_placeholder_target_is_normalized(b):
        x = b.input("aten_permute_copy_default_1", (2, 3))
        x.target = "x"
        y = b.input("y", (2, 3))
        b.output([b.call(A.sub.Tensor, x, y)])

    @case
    def target_alone_can_look_like_a_transform(b):
        x = b.input("x", (2, 3))
        x.target = "aten_view_copy_default_7"
        y = b.input("y", (2, 3))
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def matching_names_are_left_alone(b):
        x = b.input("aten_view_copy_default", (2, 3))
        y = b.input("y", (2, 3))
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def non_transform_names_are_left_alone(b):
        x = b.input("custom_view_copy_default", (2, 3))
        x.target = "custom_view_copy_default_base"
        y = b.input("y", (2, 3))
        b.output([b.call(A.add.Tensor, x, y)])

    @case
    def normalization_needs_a_target_op(b):
        x = b.input("aten_view_copy_default_2", (2, 3))
        x.target = "aten_view_copy_default"
        b.call(A.neg.default, x)
        b.output([b.call(A.relu.default, x)])

    @case
    def normalization_alone_removes_dead_code(b):
        x = b.input("aten_view_copy_default_2", (2, 3))
        x.target = "aten_view_copy_default"
        y = b.input("y", (2, 3))
        b.call(A.neg.default, x)
        b.output([b.call(A.add.Tensor, x, y)])

    return cases


def move_data_movement_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    # rc4's test_move_data_movement_ops_to_smaller_dtype_pass.py graphs.
    @case
    def moves_permute_after_narrowing_rescale(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int32)
        b.output([rescale(b, b.permute(x, [0, 2, 3, 1]), torch.int8, [1.0], 0, 0)])

    @case
    def moves_permute_before_widening_rescale(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int8)
        b.output([b.permute(rescale(b, x, torch.int32, [1.0], 0, 0), [0, 2, 3, 1])])

    @case
    def keeps_per_channel_rescale_order(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int32)
        b.output([rescale(b, b.permute(x, [0, 2, 3, 1]), torch.int8, [1.0, 2.0], 0, 0)])

    @case
    def splits_merged_permute_at_distinct_narrowing_rescales(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int32)
        p = b.permute(x, [0, 2, 3, 1])
        b.output([rescale(b, p, torch.int8, [1.0]), rescale(b, p, torch.int8, [2.0])])

    # Branches.
    @case
    def clone_moves_after_narrowing_rescale(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        b.output([rescale(b, b.call(D._clone_dim_order.default, x), torch.int8)])

    @case
    def clone_moves_before_widening_rescale(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        b.output([b.call(D._clone_dim_order.default, rescale(b, x, torch.int32))])

    @case
    def to_dim_order_copy_moves(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        c = b.call(D._to_dim_order_copy.default, x, dtype=torch.int32)
        b.output([rescale(b, c, torch.int16)])

    @case
    def to_dim_order_copy_keeps_its_dtype_kwarg(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        r = rescale(b, x, torch.int32)
        b.output([b.call(D._to_dim_order_copy.default, r, dtype=torch.int32)])

    @case
    def stale_clone_moves_back_and_forth(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        c = b.call(D._clone_dim_order.default, x)
        b.output([rescale(b, rescale(b, c, torch.int8), torch.int16)])

    @case
    def stale_clone_under_a_permute(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int32)
        c = b.call(D._clone_dim_order.default, x)
        p = b.permute(c, [0, 2, 3, 1])
        b.output([rescale(b, rescale(b, p, torch.int8), torch.int16)])

    @case
    def equal_widths_stay(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        b.output([rescale(b, b.permute(x, [1, 0]), torch.int32)])

    @case
    def float_permutes_stay(b):
        x = b.input("x", (2, 3))
        y = b.input("y", (2, 3), dtype=torch.int8)
        b.output([b.permute(x, [1, 0]), rescale(b, y, torch.int32)])

    @case
    def int16_steps(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int16)
        p = b.permute(rescale(b, x, torch.int32), [2, 0, 1])
        b.output([rescale(b, b.permute(p, [1, 2, 0]), torch.int8)])

    @case
    def between_widening_and_narrowing_rescales(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        p = b.permute(rescale(b, x, torch.int32), [1, 0])
        b.output([rescale(b, p, torch.int8)])

    @case
    def permute_chain_before_a_narrowing_rescale(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int32)
        p1 = b.permute(x, [0, 2, 3, 1])
        p2 = b.permute(p1, [0, 3, 1, 2])
        b.output([rescale(b, p2, torch.int8)])

    @case
    def permute_chain_after_a_widening_rescale(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int8)
        p1 = b.permute(rescale(b, x, torch.int32), [0, 2, 3, 1])
        b.output([b.permute(p1, [0, 3, 1, 2])])

    @case
    def split_keeps_the_original_for_other_users(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        p = b.permute(x, [1, 0])
        b.output([rescale(b, p, torch.int8), b.call(A.neg.default, p)])

    @case
    def split_with_the_narrowing_rescale_second(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        p = b.permute(x, [1, 0])
        b.output([b.call(A.neg.default, p), rescale(b, p, torch.int8)])

    @case
    def split_three_ways(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int32)
        p = b.permute(x, [2, 0, 1])
        b.output([rescale(b, p, torch.int8, [1.0]), rescale(b, p, torch.int16, [2.0]), rescale(b, p, torch.int8, [3.0])])

    @case
    def per_channel_user_does_not_count(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        p = b.permute(x, [1, 0])
        b.output([rescale(b, p, torch.int8, [1.0, 2.0, 3.0]), rescale(b, p, torch.int8, [0.5])])

    @case
    def special_dtype_mismatch_blocks(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        p = b.permute(x, [1, 0])
        set_special(p)
        b.output([rescale(b, p, torch.int8)])

    @case
    def matching_special_dtypes_allow(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        p = b.permute(x, [1, 0])
        set_special(p)
        r = rescale(b, p, torch.int8)
        set_special(r)
        b.output([r])

    @case
    def special_dtype_on_the_producer(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        set_special(x)
        r = rescale(b, x, torch.int32)
        b.output([b.permute(r, [1, 0])])

    @case
    def rescale_with_unsigned_input(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        r = rescale(b, x, torch.int32, [0.5], 128, 0, input_unsigned=True)
        b.output([b.permute(r, [1, 0])])

    @case
    def widening_rescale_with_a_later_other_user(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int8)
        r = rescale(b, x, torch.int32)
        p = b.permute(r, [0, 2, 3, 1])
        b.output([p, b.call(A.neg.default, r)])

    @case
    def widening_rescale_with_an_earlier_other_user(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int8)
        r = rescale(b, x, torch.int32)
        n = b.call(A.neg.default, r)
        b.output([b.permute(r, [0, 2, 3, 1]), n])

    @case
    def rescale_users_follow_the_move(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        r = rescale(b, b.permute(x, [1, 0]), torch.int8)
        b.output([b.call(A.neg.default, r), b.call(A.abs.default, r)])

    @case
    def qparams_stay_with_their_nodes(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        p = b.permute(x, [1, 0])
        set_qparams(p, inputs=[0], outputs=[0])
        r = rescale(b, p, torch.int8)
        set_qparams(r, outputs=[0])
        b.output([r])

    @case
    def casting_copy_moves_below_a_rescale(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        c = b.call(D._to_dim_order_copy.default, x, dtype=torch.int32)
        b.output([b.call(A.neg.default, rescale(b, c, torch.int16))])

    @case
    def casting_copy_feeds_a_true_division(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        c = b.call(D._to_dim_order_copy.default, x, dtype=torch.int32)
        r = rescale(b, c, torch.int16)
        b.output([b.call(A.div.Tensor, r, r), c])

    @case
    def add_reads_a_rewired_rescale(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int8)
        y = b.input("y", (1, 2, 3, 4), dtype=torch.int32)
        r = rescale(b, x, torch.int32)
        p = b.permute(r, [0, 2, 3, 1])
        b.output([p, b.call(A.add.Tensor, r, y)])

    @case
    def add_reads_a_rewired_rescale_of_the_same_shape(b):
        x = b.input("x", (1, 2, 2, 2), dtype=torch.int8)
        y = b.input("y", (1, 2, 2, 2), dtype=torch.int32)
        r = rescale(b, x, torch.int32)
        p = b.permute(r, [0, 2, 3, 1])
        b.output([p, b.call(A.add.Tensor, r, y)])

    @case
    def a_rewired_user_broadcasts_to_a_new_shape(b):
        x = b.input("x", (2, 1), dtype=torch.int8)
        y = b.input("y", (2, 1), dtype=torch.int32)
        r = rescale(b, x, torch.int32)
        p = b.permute(r, [1, 0])
        s = b.call(A.add.Tensor, r, y)
        b.output([p, b.call(A.neg.default, s), b.call(A.sum.dim_IntList, s, [0], True)])

    @case
    def a_rewired_user_without_a_shape_rule(b):
        x = b.input("x", (1, 2, 3, 4), dtype=torch.int8)
        r = rescale(b, x, torch.int32)
        p = b.permute(r, [0, 2, 3, 1])
        b.output([p, b.call(A.flip.default, r, [0])])

    @case
    def exp_after_narrowing_rescale(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        r = rescale(b, b.permute(x, [1, 0]), torch.int8)
        b.output([b.call(A.exp.default, r)])

    @case
    def sqrt_after_widening_rescale(b):
        x = b.input("x", (2, 3), dtype=torch.int8)
        p = b.permute(rescale(b, x, torch.int32), [1, 0])
        b.output([b.call(A.sqrt.default, p)])

    @case
    def float_results_after_a_sound_move(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        r = rescale(b, b.permute(x, [1, 0]), torch.int8)
        b.output(
            [
                b.call(A.div.Tensor, r, r),
                b.call(A.mul.Tensor, r, 0.5),
                b.call(A.gt.Tensor, r, r),
                b.call(A.rsqrt.default, r),
            ]
        )

    @case
    def casting_copy_feeds_float_results(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        c = b.call(D._to_dim_order_copy.default, x, dtype=torch.int32)
        r = rescale(b, c, torch.int16)
        b.output(
            [
                b.call(A.exp.default, r),
                b.call(A.mul.Tensor, r, 0.5),
                b.call(A.add.Tensor, r, r),
                b.call(A.ne.Tensor, r, r),
                b.call(A.sum.dim_IntList, r, [0], True),
                b.call(A.cat.default, [r, r], 0),
            ]
        )

    @case
    def independent_moves_in_order(b):
        x = b.input("x", (2, 3), dtype=torch.int32)
        y = b.input("y", (3, 4), dtype=torch.int8)
        first = rescale(b, b.permute(x, [1, 0]), torch.int8)
        second = b.permute(rescale(b, y, torch.int16), [1, 0])
        b.output([first, second])

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
FIIT_FLOAT_OPS = [
    A.add.Tensor,
    A.sub.Tensor,
    A.mul.Tensor,
    A.div.Tensor,
    A.remainder.Tensor,
    A.eq.Tensor,
    A.gt.Tensor,
    A.ge.Tensor,
    A.lt.Tensor,
    A.le.Tensor,
    A.ne.Tensor,
    A.maximum.default,
]
FIIT_INT_OPS = [
    A.bitwise_and.Tensor,
    A.bitwise_or.Tensor,
    A.bitwise_xor.Tensor,
    A.bitwise_left_shift.Tensor,
    A.bitwise_right_shift.Tensor,
    A.add.Tensor,
]


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
        elif kind.startswith("dup_") or kind == "mark":
            self.duplicate_step(kind)
        elif kind.startswith("fiit_"):
            self.identical_transforms_step(kind)
        elif kind.startswith("mdm_"):
            self.data_movement_step(kind)
        elif kind.startswith("table"):
            self.table_step(kind)
        else:
            raise ValueError(kind)

    # FuseDuplicateUsers ---------------------------------------------------

    def duplicate_step(self, kind: str) -> None:
        rng, b = self.rng, self.b
        copies = rng.randint(2, 3)
        if kind == "dup_unary":
            x = self.pick()
            op = rng.choice([*UNARY_FLOAT, D._clone_dim_order.default])
            for _ in range(copies):
                self.emit(b.call(op, x))
        elif kind == "dup_binary":
            x = self.pick()
            y = self.operand_for(x)
            op = rng.choice([A.add.Tensor, A.sub.Tensor, A.mul.Tensor, A.maximum.default])
            for _ in range(copies):
                args = (x, y) if rng.random() < 0.8 else (y, x)
                kwargs = {}
                if op in (A.add.Tensor, A.sub.Tensor) and rng.random() < 0.3:
                    kwargs["alpha"] = rng.choice([1, 1.0, 2, 2.0, True])
                self.emit(b.call(op, *args, **kwargs))
        elif kind == "dup_scalar":
            x = self.pick()
            op = rng.choice([A.mul.Tensor, A.add.Tensor])
            for _ in range(copies):
                scalar = rng.choice([2, 2.0, 1, 1.0, True, 0.5, 0.0, -0.0, False])
                self.emit(b.call(op, x, scalar))
        elif kind == "dup_rescale":
            x = self.pick(torch.int8)
            for _ in range(copies):
                r = rescale(b, x, torch.int32, [rng.choice([0.5, 0.25])])
                self.pool.append(r)
                if rng.random() < 0.5:
                    self.emit(b.call(A.neg.default, r))
        elif kind == "dup_cat":
            x = self.pick()
            rank = x.meta["val"].dim()
            if rank == 0:
                return
            other = x if rng.random() < 0.5 else self.emit(b.call(A.neg.default, x))
            dim = rng.randrange(-rank, rank)
            for _ in range(copies):
                parts = [x, other] if rng.random() < 0.8 else [other, x]
                self.emit(b.call(A.cat.default, parts, dim))
        elif kind == "dup_view":
            x = self.pick()
            shape = self.view_shape(list(x.meta["val"].shape))
            for _ in range(copies):
                self.emit(b.view(x, shape))
        elif kind == "dup_permute":
            x = self.pick()
            dims = self.permutation(x.meta["val"].dim())
            for _ in range(copies):
                self.emit(b.permute(x, dims))
        elif kind == "dup_chain":
            x = self.pick()
            ops = [rng.choice(UNARY_FLOAT) for _ in range(rng.randint(2, 3))]
            for _ in range(2):
                value = x
                for op in ops:
                    value = b.call(op, value)
                self.emit(value)
        elif kind == "mark":
            computed = [n for n in self.pool if n.op == "call_function"]
            if computed:
                mark_no_fuse(rng.choice(computed))
        else:
            raise ValueError(kind)

    # FuseDuplicateUsers where rc4 has TABLE ops --------------------------

    def table_step(self, kind: str) -> None:
        rng, b = self.rng, self.b

        def draw():
            return rng.choice(TABLE_QPARAMS), rng.choice(TABLE_QPARAMS)

        if kind == "table":
            x = self.pick(torch.int8)
            target, args, kwargs = rng.choice(TABLE_TARGETS)
            self.pool.append(table_op(b, target, x, *draw(), *args, **kwargs))
        elif kind == "table_dup":
            x = self.pick(torch.int8)
            target, args, kwargs = rng.choice(TABLE_TARGETS)
            first = draw()
            for _ in range(rng.randint(2, 3)):
                q = first if rng.random() < 0.6 else draw()
                if rng.random() < 0.2:
                    target, args, kwargs = rng.choice(TABLE_TARGETS)
                self.pool.append(table_op(b, target, x, *q, *args, **kwargs))
        elif kind == "table_user":
            tables = [n for n in self.pool if _is_table_op(n)]
            if not tables:
                return
            t = rng.choice(tables)
            if rng.random() < 0.3:
                other = self.operand_for(t, torch.int8)
                for _ in range(rng.randint(1, 2)):
                    self.pool.append(b.call(A.add.Tensor, t, other))
            else:
                op = rng.choice([A.neg.default, A.abs.default, A.relu.default])
                for _ in range(rng.randint(1, 2)):
                    self.pool.append(b.call(op, t))
        elif kind == "table_int_dup":
            x = self.pick(torch.int8)
            op = rng.choice([A.neg.default, A.abs.default])
            for _ in range(rng.randint(2, 3)):
                self.pool.append(b.call(op, x))
        else:
            raise ValueError(kind)

    # FuseIdenticalInputTransforms ---------------------------------------

    def transform_for(self, x):
        shape = list(x.meta["val"].shape)
        if self.rng.random() < 0.5:
            return A.permute_copy.default, self.permutation(len(shape))
        return A.view_copy.default, self.view_shape(shape)

    def transform_partner(self, x, transform, dtype=torch.float32):
        """An input the same transform applies to: usually x's shape, else a
        same-rank broadcast partner (permute) or another shape with x's
        element count (view)."""
        rng = self.rng
        shape = list(x.meta["val"].shape)
        if rng.random() < 0.7 or not shape:
            partner = shape
        elif transform[0] is A.permute_copy.default:
            partner = [1 if rng.random() < 0.4 else d for d in shape]
        else:
            partner = [d for d in self.view_shape(shape) if d != -1]
            if math.prod(partner) != math.prod(shape):
                partner = shape
        node = self.new_input(partner, dtype, self.dim_order(len(partner)))
        self.pool.append(node)
        return node

    def apply(self, transform, x):
        return self.b.call(transform[0], x, list(transform[1]))

    def identical_transforms_step(self, kind: str) -> None:
        rng, b = self.rng, self.b
        if kind in ("fiit_binary", "fiit_int", "fiit_shared"):
            dtype = torch.int32 if kind == "fiit_int" else torch.float32
            x = self.pick(dtype)
            transform = self.transform_for(x)
            y = self.transform_partner(x, transform, dtype)
            tx, ty = self.apply(transform, x), self.apply(transform, y)
            ops = FIIT_INT_OPS if kind == "fiit_int" else FIIT_FLOAT_OPS
            args = (tx, ty) if rng.random() < 0.5 else (ty, tx)
            self.emit(b.call(rng.choice(ops), *args))
            if kind == "fiit_shared":
                self.emit(b.call(A.neg.default, rng.choice([tx, ty])))
        elif kind == "fiit_invariant":
            x = self.pick()
            transform = self.transform_for(x)
            tx = self.apply(transform, x)
            rank = tx.meta["val"].dim()
            roll = rng.random()
            if roll < 0.7:
                shape = [1] * rank
            elif roll < 0.85:
                shape = [1] * max(0, rank + rng.choice([-1, 1]))
            else:
                shape = [1 if rng.random() < 0.5 else d for d in tx.meta["val"].shape]
            ones = self.new_input(shape)
            self.pool.append(ones)
            args = (tx, ones) if rng.random() < 0.6 else (ones, tx)
            self.emit(b.call(rng.choice(FIIT_FLOAT_OPS), *args))
        elif kind == "fiit_where":
            x = self.pick()
            transform = self.transform_for(x)
            y = self.transform_partner(x, transform)
            if rng.random() < 0.7:
                c = self.new_input(list(x.meta["val"].shape), torch.bool)
                condition = self.apply(transform, c)
            else:
                condition = self.new_input([1] * len(transform[1]), torch.bool)
            self.emit(b.call(A.where.self, condition, self.apply(transform, x), self.apply(transform, y)))
        elif kind == "fiit_cat":
            x = self.pick()
            shape = list(x.meta["val"].shape)
            transform = self.transform_for(x)
            tx = self.apply(transform, x)
            out_rank = tx.meta["val"].dim()
            if out_rank == 0:
                return
            dim = rng.randrange(-out_rank, out_rank)
            partner = list(shape)
            if transform[0] is A.permute_copy.default and rng.random() < 0.8:
                source = transform[1][dim % out_rank] % out_rank
                partner[source] = rng.randint(1, 4)
            y = self.new_input(partner, dim_order=self.dim_order(len(partner)))
            self.pool.append(y)
            parts = [tx, self.apply(transform, y)]
            if rng.random() < 0.2:
                z = self.new_input(partner)
                parts.append(self.apply(transform, z))
            style = rng.random()
            if style < 0.7:
                node = b.call(A.cat.default, parts, dim)
            elif style < 0.85:
                node = b.call(A.cat.default, parts, dim=dim)
            else:
                node = b.call(A.cat.default, parts)
            self.emit(node)
        elif kind == "fiit_chain":
            x = self.pick()
            first = self.transform_for(x)
            y = self.transform_partner(x, first)
            tx, ty = self.apply(first, x), self.apply(first, y)
            second = self.transform_for(tx)
            self.emit(b.call(rng.choice(FIIT_FLOAT_OPS), self.apply(second, tx), self.apply(second, ty)))
        elif kind == "fiit_mixed":
            x = self.pick()
            transform = self.transform_for(x)
            y = self.transform_partner(x, transform)
            other = self.transform_for(y)
            self.emit(b.call(rng.choice(FIIT_FLOAT_OPS), self.apply(transform, x), self.apply(other, y)))
        else:
            raise ValueError(kind)

    # MoveDataMovementOpsToSmallerDtype -----------------------------------

    def data_movement_step(self, kind: str) -> None:
        rng, b = self.rng, self.b
        int_dtypes = [torch.int8, torch.int16, torch.int32]
        if kind == "mdm_chain":
            value = self.pick(rng.choice(int_dtypes))
            for _ in range(rng.randint(2, 5)):
                step = rng.choices(["rescale", "permute", "clone", "to_copy"], [4, 4, 1, 1])[0]
                if step == "rescale":
                    scales = [0.5] if rng.random() < 0.85 else [0.5, 0.25]
                    value = rescale(b, value, rng.choice(int_dtypes), scales)
                elif step == "permute":
                    value = b.permute(value, self.permutation(value.meta["val"].dim()))
                elif step == "clone":
                    value = b.call(D._clone_dim_order.default, value)
                elif rng.random() < 0.5:
                    value = b.call(D._to_dim_order_copy.default, value, dtype=value.meta["val"].dtype)
                else:
                    value = b.call(D._to_dim_order_copy.default, value)
                if rng.random() < 0.15:
                    self.pool.append(value)
            self.emit(value)
        elif kind == "mdm_fanout":
            value = self.pick(rng.choice(int_dtypes))
            p = b.permute(value, self.permutation(value.meta["val"].dim()))
            for _ in range(rng.randint(2, 3)):
                if rng.random() < 0.75:
                    scales = [rng.choice([0.5, 0.25])] if rng.random() < 0.85 else [0.5, 0.25]
                    self.emit(rescale(b, p, rng.choice(int_dtypes), scales))
                else:
                    self.emit(b.call(A.neg.default, p))
        elif kind == "mdm_special":
            computed = [n for n in self.pool if n.op == "call_function"]
            if computed:
                set_special(rng.choice(computed))
        elif kind == "mdm_shared_rescale":
            # A RESCALE that widens, read by a permute and by later users,
            # which rc4 rewires to the permuted value when the permute moves.
            narrow = rng.choice([torch.int8, torch.int16])
            value = self.pick(narrow)
            r = rescale(b, value, torch.int32)
            self.pool.append(b.permute(r, self.permutation(r.meta["val"].dim())))
            for _ in range(rng.randint(1, 2)):
                if rng.random() < 0.5:
                    partner = self.operand_for(r, torch.int32)
                    user = b.call(rng.choice([A.add.Tensor, A.mul.Tensor, A.maximum.default]), r, partner)
                else:
                    user = b.call(rng.choice([A.neg.default, A.abs.default, A.exp.default, A.sqrt.default]), r)
                self.pool.append(user)
                if rng.random() < 0.4:
                    self.pool.append(b.call(rng.choice([A.neg.default, A.relu.default]), user))
        elif kind == "mdm_float_user":
            # Float and bool results of integer values, often moved ones.
            candidates = [n for n in self.pool if n.meta["val"].dtype in int_dtypes]
            if not candidates:
                return
            value = candidates[-1 - min(int(rng.expovariate(0.7)), len(candidates) - 1)]
            op = rng.choice(["exp", "sqrt", "rsqrt", "sigmoid", "div", "mul_float", "gt", "sum"])
            if op == "div":
                user = b.call(A.div.Tensor, value, value)
            elif op == "mul_float":
                user = b.call(A.mul.Tensor, value, 0.5)
            elif op == "gt":
                user = b.call(A.gt.Tensor, value, value)
            elif op == "sum":
                rank = value.meta["val"].dim()
                if rank == 0:
                    return
                user = b.call(A.sum.dim_IntList, value, [rng.randrange(rank)], True)
            else:
                user = b.call(getattr(A, op).default, value)
            self.pool.append(user)
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
    "fuse_duplicate_users": {
        "dup_unary": 2,
        "dup_binary": 2,
        "dup_scalar": 1,
        "dup_rescale": 0.7,
        "dup_cat": 0.6,
        "dup_view": 0.6,
        "dup_permute": 0.6,
        "dup_chain": 1,
        "mark": 0.3,
        "unary": 1,
        "binary": 1,
        "dead": 0.3,
    },
    "fuse_identical_input_transforms": {
        "fiit_binary": 4,
        "fiit_int": 0.7,
        "fiit_invariant": 1.2,
        "fiit_where": 0.8,
        "fiit_cat": 2,
        "fiit_chain": 1,
        "fiit_shared": 0.6,
        "fiit_mixed": 0.8,
        "unary": 0.7,
        "dead": 0.3,
    },
    "move_data_movement": {
        "mdm_chain": 4,
        "mdm_fanout": 1.5,
        "mdm_special": 0.4,
        "mdm_shared_rescale": 1.2,
        "mdm_float_user": 1.2,
        "dead": 0.2,
    },
    "table_stage": {
        "table": 2,
        "table_dup": 3,
        "table_user": 1.5,
        "table_int_dup": 1,
        "mark": 0.3,
        "dead": 0.3,
    },
}


def random_case(profile: str, seed: int) -> Callable[[Builder], None]:
    if profile == "remove_permutes":
        return remove_permutes_random_case(seed)
    if profile == "remove_permutes_user_order":
        return remove_permutes_user_order_random_case(seed)
    if profile == "propagate":
        return propagation_random_case(seed)

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
    # rc4's MaxPool2dWithoutStride (test_rewrite_max_pool2d_pass.py): export
    # omits the stride, which RewriteMaxPool2dPass reads as the kernel.
    "no_stride": (lambda: PoolM(lambda x: FN.max_pool2d(x, 3)), shape_input(1, 3, 9, 9)),
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
    # rc4's AvgPool2dWithEmptyStride (test_rewrite_avg_pool2d_pass.py): export
    # omits `stride=[]`, which RewriteAvgPool2dPass reads as the kernel.
    "no_stride": (
        lambda: PoolM(lambda x: FN.avg_pool2d(x, kernel_size=[2, 3], stride=[])),
        shape_input(1, 3, 8, 9),
    ),
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
    # test_get_resize_parameters_1d_nearest_matches_pytorch_scale_factor's 6 -> 9.
    "nearest_size_6to9": (lambda: UpsampleM("nearest", size=(9, 9)), shape_input(1, 3, 6, 6)),
    # Downscales: nearest by an explicit factor, bilinear by size.
    "nearest_scale_half": (lambda: UpsampleM("nearest", scale=0.5), shape_input(1, 3, 8, 8)),
    "bilinear_size_8to3": (
        lambda: UpsampleM("bilinear", size=(3, 3), align_corners=False),
        shape_input(1, 3, 8, 8),
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


class Conv2dBiasModule(NN.Module):
    """test_rewrite_conv_pass.py's Conv2dBiasModule."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = NN.Conv2d(4, 6, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv2dBiasModule(NN.Module):
    """test_rewrite_conv_pass.py's DepthwiseConv2dBiasModule."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = NN.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=True)

    def forward(self, x):
        return self.conv(x)


class Conv3dBiasModule(NN.Module):
    """test_rewrite_conv_pass.py's Conv3dBiasModule."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = NN.Conv3d(3, 5, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class TransposeConv2dBiasModule(NN.Module):
    """test_rewrite_conv_pass.py's TransposeConv2dBiasModule."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = NN.ConvTranspose2d(
            3, 4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
        )

    def forward(self, x):
        return self.conv(x)


# `_compile_spec_int16().tosa_spec`, the A16W8 tests' default lowering context
# (test_rewrite_conv_pass.py:170-171, :285), and the U55 subset
# test_rewrite_conv_a16w8_u55_separates_distinct_output_rescales passes (:655).
A16W8_TEST_SPEC = "TOSA-1.0+INT+FP+int16"
A16W8_TEST_U55_SPEC = "TOSA-1.0+INT+int16+int4+u55"


def rc4_test_a16w8_program(model, inputs, spec: str, insert_int32_rescales: bool = True):
    """The A16W8 tests' program (test_rewrite_conv_pass.py:194-293):
    `_export_quantized_a16w8` (VgfQuantizer on TOSA-1.0+INT+FP+int16 with
    `get_symmetric_a16w8_quantization_config()`), `to_edge` without IR validity
    checks and `_run_pre_rewrite_passes`, then, as `_rewrite_a16w8_convs` does,
    InsertRescaleInt32Pass in the lowering context of `spec`, the TOSA spec
    RewriteConvPass then runs in. The int16-bias tests (:933-972) call
    RewriteConvPass without InsertRescaleInt32Pass."""
    from executorch.backends.arm._passes import (
        ConvertToClampPass,
        FoldAndAnnotateQParamsPass,
        FuseQuantizedActivationPass,
        InsertRescaleInt32Pass,
        QuantizeClampArgumentsPass,
    )
    from executorch.backends.arm.quantizer.arm_quantizer import (
        get_symmetric_a16w8_quantization_config,
        VgfQuantizer,
    )
    from executorch.backends.arm.vgf import VgfCompileSpec
    from executorch.exir import EdgeCompileConfig, to_edge

    exported = torch.export.export(model.eval(), inputs).module(check_guards=False)
    quantizer = VgfQuantizer(VgfCompileSpec("TOSA-1.0+INT+FP+int16"))
    quantizer.set_global(get_symmetric_a16w8_quantization_config())
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
    if insert_int32_rescales:
        with TosaLoweringContext(TosaSpecification.create_from_string(spec)):
            gm = InsertRescaleInt32Pass()(gm).graph_module
    return ep, gm


def rc4_test_a16w8_cases() -> list[Rc4TestCase]:
    """The A16W8 tests' RewriteConvPass calls (test_rewrite_conv_pass.py:446-972)."""

    def a16w8(name, test, model, shape, spec=A16W8_TEST_SPEC, insert_int32_rescales=True):
        return Rc4TestCase(
            name,
            test,
            lambda: rc4_test_a16w8_program(
                model(), (torch.randn(*shape),), spec, insert_int32_rescales
            ),
            spec,
        )

    mixed = "test_rewrite_conv_a16w8_mixed_consumers_restore_int16"
    u55 = "test_rewrite_conv_a16w8_u55_separates_distinct_output_rescales (:650)"
    bias = "test_rewrite_conv_int16_bias_lowers_to_single_tosa_conv"
    return [
        a16w8(
            "a16w8_mixed_consumers_conv",
            f"{mixed}[conv] (:446); {u55}, generic",
            lambda: A16W8MixedConsumerChain(NN.Conv2d(4, 4, 1)),
            (1, 4, 8, 8),
        ),
        a16w8(
            "a16w8_mixed_consumers_conv_u55",
            f"{u55}, U55",
            lambda: A16W8MixedConsumerChain(NN.Conv2d(4, 4, 1)),
            (1, 4, 8, 8),
            spec=A16W8_TEST_U55_SPEC,
        ),
        a16w8(
            "a16w8_mixed_consumers_depthwise_conv",
            f"{mixed}[depthwise_conv] (:446)",
            lambda: A16W8MixedConsumerChain(NN.Conv2d(4, 4, 3, padding=1, groups=4)),
            (1, 4, 8, 8),
        ),
        a16w8(
            "a16w8_mixed_consumers_avg_pool",
            f"{mixed}[avg_pool] (:446)",
            lambda: A16W8MixedConsumerChain(NN.AvgPool2d(3, stride=1, padding=1)),
            (1, 4, 8, 8),
        ),
        a16w8(
            "a16w8_mixed_consumers_max_pool",
            f"{mixed}[max_pool] (:446)",
            lambda: A16W8MixedConsumerChain(NN.MaxPool2d(3, stride=1, padding=1)),
            (1, 4, 8, 8),
        ),
        a16w8(
            "a16w8_mixed_consumers_conv3d",
            f"{mixed}[conv3d] (:446)",
            lambda: A16W8MixedConsumer(NN.Conv3d(4, 4, 3, padding=1), NN.Conv3d(4, 4, 1)),
            (1, 4, 4, 8, 8),
        ),
        a16w8(
            "a16w8_mixed_consumers_transpose_conv2d",
            f"{mixed}[transpose_conv2d] (:446)",
            lambda: A16W8MixedConsumer(NN.ConvTranspose2d(4, 4, 3, padding=1), NN.Conv2d(4, 4, 1)),
            (1, 4, 8, 8),
        ),
        a16w8(
            "a16w8_preserves_int32_for_int32_consumers",
            "test_rewrite_conv_a16w8_preserves_int32_for_int32_consumers (:690)",
            A16W8Int32Consumer,
            (1, 4, 8, 8),
        ),
        a16w8(
            "conv1d_a16w8_narrows_instead_of_forking_int32",
            "test_rewrite_conv1d_a16w8_narrows_instead_of_forking_int32 (:723)",
            A16W8Conv1dInt32Consumer,
            (1, 4, 8),
        ),
        a16w8(
            "conv1d_a16w8_shares_one_layout_boundary",
            "test_rewrite_conv1d_a16w8_shares_one_layout_boundary (:757)",
            A16W8Conv1dSharedConsumers,
            (1, 4, 8),
        ),
        a16w8(
            "a16w8_preserves_int32_after_permute",
            "test_rewrite_conv_a16w8_preserves_int32_after_permute (:786)",
            A16W8PermutedInt32Consumer,
            (1, 4, 8, 8),
        ),
        a16w8(
            "int16_bias_conv2d",
            f"{bias}[Conv2dBiasModule] (:933)",
            Conv2dBiasModule,
            (1, 4, 8, 8),
            insert_int32_rescales=False,
        ),
        a16w8(
            "int16_bias_depthwise_conv2d",
            f"{bias}[DepthwiseConv2dBiasModule] (:933)",
            DepthwiseConv2dBiasModule,
            (1, 4, 8, 8),
            insert_int32_rescales=False,
        ),
        a16w8(
            "int16_bias_conv3d",
            f"{bias}[Conv3dBiasModule] (:933)",
            Conv3dBiasModule,
            (1, 3, 6, 6, 6),
            insert_int32_rescales=False,
        ),
        a16w8(
            "int16_bias_transpose_conv2d",
            f"{bias}[TransposeConv2dBiasModule] (:933)",
            TransposeConv2dBiasModule,
            (1, 3, 6, 6),
            insert_int32_rescales=False,
        ),
    ]


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
    ] + rc4_test_a16w8_cases()


RC4_TEST_FIXTURE = "rewrite_conv_rc4_tests"
# The A16W8 cases' U55 step signs RESCALE-to-permute heads with
# `build_node_signature`.
RC4_TEST_SOURCES = CONV_SOURCES + ["backends/transforms/fuse_duplicate_users_pass.py"]


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
# RemovePermutesAroundElementwiseTosaOps
# --------------------------------------------------------------------------


class _GraphOrderedEdges(set):
    """A set of node pairs, iterated in graph order of its pairs.

    rc4 iterates `edges_in` and `constant_edges_in` in an order that depends
    on object addresses, and the order can change the result: where the
    compensating permutes for a constant feeding several region nodes land
    (each goes directly after the constant), and what a region node reads when
    one of its bypassed permutes feeds another (`replace_input_with` replaces
    every use, so bypassing the downstream permute first lets the upstream
    bypass rewrite both). Pinning both to graph order makes the fixture
    reproducible; the RETAB port iterates in the same order.
    """

    def __iter__(self):
        edges = list(set.__iter__(self))
        if not edges:
            return iter(edges)
        position = {node: index for index, node in enumerate(edges[0][0].graph.nodes)}
        return iter(sorted(edges, key=lambda edge: (position[edge[0]], position[edge[1]])))


class _PinnedSubgraph(RemovePermutesAroundElementwiseOps.Subgraph):
    """rc4's Subgraph with `edges_in` and `constant_edges_in` pinned,
    including the set the TOSA subclass assigns
    (remove_permutes_around_elementwise_tosa_ops.py:53)."""

    def __setattr__(self, name, value):
        if name in ("edges_in", "constant_edges_in"):
            value = _GraphOrderedEdges(value)
        super().__setattr__(name, value)


RemovePermutesAroundElementwiseOps.Subgraph = _PinnedSubgraph

TO_NHWC = [0, 2, 3, 1]
TO_NCHW = [0, 3, 1, 2]


def inverse_permutation(perm: Sequence[int]) -> list[int]:
    rank = len(perm)
    normalized = [d % rank for d in perm] if rank else []
    return [normalized.index(i) for i in range(rank)]


def interleave(b, x, dim, scale, head="unsqueeze"):
    """repeat_interleave(scale, dim) as unsqueeze -> expand_copy -> view_copy."""
    shape = list(x.meta["val"].shape)
    unsqueezed = list(shape)
    unsqueezed.insert(dim + 1, 1)
    if head == "unsqueeze":
        u = b.call(A.unsqueeze_copy.default, x, dim + 1)
    else:
        u = b.view(x, unsqueezed)
    expanded = list(unsqueezed)
    expanded[dim + 1] = scale
    e = b.call(A.expand_copy.default, u, expanded)
    merged = list(shape)
    merged[dim] *= scale
    return b.view(e, merged)


def remove_permutes_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    X4 = (2, 3, 4, 5)

    @case
    def no_permute(b):
        x = b.input("x", X4)
        b.output([b.call(A.relu.default, x)])

    @case
    def round_trip_around_a_unary_op(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, b.permute(x, TO_NHWC))
        b.output([b.permute(r, TO_NCHW)])

    @case
    def channels_last_dialect_permutes(b):
        x = b.input("x", X4)
        CL = exir_ops.edge.channels_last.permute_copy.default
        r = b.call(A.hardtanh.default, b.call(CL, x, TO_NHWC))
        b.output([b.call(CL, r, TO_NCHW)])

    @case
    def aten_permute_default_is_not_a_permute(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, b.call(A.permute.default, x, TO_NHWC))
        b.output([b.call(A.permute.default, r, TO_NCHW)])

    @case
    def diamond_region(b):
        x = b.input("x", X4)
        a = b.call(A.abs.default, b.permute(x, TO_NHWC))
        n = b.call(A.neg.default, a)
        s = b.call(A.sigmoid.default, a)
        b.output([b.permute(b.call(A.add.Tensor, n, s), TO_NCHW)])

    @case
    def two_entry_permutes_with_the_same_permutation(b):
        x = b.input("x", X4)
        y = b.input("y", X4)
        a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), b.permute(y, TO_NHWC))
        b.output([b.permute(a, TO_NCHW)])

    @case
    def negative_dims_in_the_permutes(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, b.permute(x, [0, -2, -1, 1]))
        b.output([b.permute(r, [0, -1, 1, 2])])

    @case
    def entry_permute_with_a_user_outside_the_region(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_NHWC)
        r = b.call(A.relu.default, p)
        b.output([b.permute(r, TO_NCHW), p])

    @case
    def mismatched_entry_permute_fails(b):
        x = b.input("x", (2, 3, 4, 4))
        y = b.input("y", (2, 4, 3, 4))
        a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), b.permute(y, [0, 3, 1, 2]))
        b.output([b.permute(a, TO_NCHW)])

    @case
    def non_inverse_exit_is_composed(b):
        x = b.input("x", (2, 3, 4))
        m = b.call(A.mul.Tensor, *[b.permute(x, [1, 2, 0])] * 2)
        b.output([b.permute(m, [1, 0, 2])])

    @case
    def non_inverse_exit_with_dims_as_a_kwarg(b):
        x = b.input("x", (2, 3, 4))
        m = b.call(A.mul.Tensor, *[b.permute(x, [1, 2, 0])] * 2)
        b.output([b.call(A.permute_copy.default, m, dims=[1, 0, 2])])

    @case
    def non_inverse_exit_that_removes_nothing_is_skipped(b):
        x = b.input("x", (2, 3, 4))
        p = b.permute(x, [1, 2, 0])
        m = b.call(A.neg.default, p)
        b.output([b.permute(m, [1, 0, 2]), p])

    @case
    def matching_and_composed_exits(b):
        x = b.input("x", (2, 3, 4))
        m = b.call(A.relu.default, b.permute(x, [1, 2, 0]))
        b.output([b.permute(m, [2, 0, 1]), b.permute(m, [1, 0, 2])])

    @case
    def region_reaching_the_output_fails(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, b.permute(x, TO_NHWC))
        b.output([b.permute(r, TO_NCHW), r])

    @case
    def region_feeding_a_non_permutable_op_fails(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, b.permute(x, TO_NHWC))
        c = b.call(A.cumsum.default, r, 1)
        b.output([b.permute(r, TO_NCHW), c])

    @case
    def broadcast_raising_the_rank_fails(b):
        x = b.input("x", (1, 8))
        ones = b.call(A.full.default, [1, 1, 1], 1.0)
        a = b.call(A.add.Tensor, b.permute(x, [1, 0]), ones)
        s = b.call(A.slice_copy.Tensor, a, 2, 0, 1)
        b.output([b.view(s, [8])])

    @case
    def numel1_inputs_stay_wired(b):
        x = b.input("x", X4)
        c = b.constant("param", "c", (1, 1, 1, 1))
        d = b.input("d", (1,))
        m = b.call(A.mul.Tensor, b.permute(x, TO_NHWC), c)
        a = b.call(A.add.Tensor, m, d)
        b.output([b.permute(a, TO_NCHW)])

    @case
    def full_constant_gets_a_permute(b):
        x = b.input("x", X4)
        f = b.call(A.full.default, [2, 4, 5, 3], 0.5)
        a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), f)
        b.output([b.permute(a, TO_NCHW)])

    @case
    def signature_constants_get_permutes(b):
        x = b.input("x", X4)
        w = b.constant("param", "w", (2, 4, 5, 3))
        y = b.input("y", (3,))
        v = b.constant("buffer", "v", (1, 4, 5, 3))
        k = b.constant("constant", "k", (2, 1, 1, 3))
        a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), w)
        m = b.call(A.mul.Tensor, a, v)
        s = b.call(A.sub.Tensor, m, k)
        b.output([b.permute(s, TO_NCHW), y])

    @case
    def prefixed_user_input_counts_as_a_constant(b):
        x = b.input("x", X4)
        w = b.input("p_weight", (2, 4, 5, 3))
        a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), w)
        b.output([b.permute(a, TO_NCHW)])

    @case
    def plain_user_input_in_the_region_fails(b):
        x = b.input("x", X4)
        w = b.input("w", (2, 4, 5, 3))
        a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), w)
        b.output([b.permute(a, TO_NCHW)])

    @case
    def constant_feeding_two_region_nodes(b):
        x = b.input("x", X4)
        w = b.constant("param", "w", (2, 4, 5, 3))
        m = b.call(A.mul.Tensor, b.permute(x, TO_NHWC), w)
        s = b.call(A.sub.Tensor, m, w)
        b.output([b.permute(s, TO_NCHW)])

    @case
    def lower_rank_constant_gets_a_view(b):
        x = b.input("x", (2, 5, 4))
        c = b.constant("param", "c", (1, 5))
        a = b.call(A.add.Tensor, b.permute(x, [0, 2, 1]), c)
        b.output([b.permute(a, [0, 2, 1])])

    @case
    def lower_rank_constant_with_a_moved_dim_raises(b):
        x = b.input("x", X4)
        c = b.constant("buffer", "c", (5, 3))
        a = b.call(A.mul.Tensor, b.permute(x, TO_NHWC), c)
        b.output([b.permute(a, TO_NCHW)])

    @case
    def keepdim_reductions_remap_their_dims(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_NHWC)
        mx = b.call(A.amax.default, p, [1], True)
        mn = b.call(A.amin.default, p, -1, True)
        me = b.call(A.mean.dim, p, [1, 2], True)
        su = b.call(A.sum.dim_IntList, p, [-1], keepdim=True)
        t = b.call(A.sub.Tensor, p, mx)
        t = b.call(A.add.Tensor, t, mn)
        t = b.call(A.mul.Tensor, t, me)
        t = b.call(A.sub.Tensor, t, su)
        b.output([b.permute(t, TO_NCHW)])

    @case
    def reduction_without_keepdim_fails(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_NHWC)
        m = b.call(A.mean.dim, p, [3])
        b.output([b.permute(b.call(A.relu.default, p), TO_NCHW), m])

    @case
    def amax_with_the_default_dim_gets_a_dim_kwarg(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_NHWC)
        m = b.call(A.amax.default, p, keepdim=True)
        b.output([b.permute(b.call(A.sub.Tensor, p, m), TO_NCHW)])

    @case
    def cat_dims_remapped(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_NHWC)
        c0 = b.call(A.cat.default, [p, p], 3)
        c1 = b.call(A.cat.default, [p, p], dim=-1)
        c2 = b.call(A.cat.default, [p, p])
        b.output([b.permute(c0, TO_NCHW), b.permute(c1, TO_NCHW), b.permute(c2, TO_NCHW)])

    @case
    def cat_with_a_second_entry_permute(b):
        x = b.input("x", X4)
        y = b.input("y", (2, 1, 4, 5))
        c = b.call(A.cat.default, [b.permute(x, TO_NHWC), b.permute(y, TO_NHWC)], 3)
        b.output([b.permute(c, TO_NCHW)])

    @case
    def slice_dims_remapped(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_NHWC)
        s0 = b.call(A.slice_copy.Tensor, p, 3, 0, 2)
        s1 = b.call(A.slice_copy.Tensor, s0, dim=1, start=1, end=3)
        s2 = b.call(A.slice_copy.Tensor, s1, start=0, end=1)
        b.output([b.permute(s2, TO_NCHW)])

    @case
    def constant_pad_nd_remapped(b):
        x = b.input("x", (1, 6, 6, 3))
        p = b.permute(x, TO_NCHW)
        pad = b.call(A.constant_pad_nd.default, p, [0, 0, 0, 0, 0, 1], 0.0)
        b.output([b.permute(pad, TO_NHWC)])

    @case
    def constant_pad_nd_on_two_dims(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_NHWC)
        pad = b.call(A.constant_pad_nd.default, p, [1, 2, 0, 1])
        b.output([b.permute(pad, TO_NCHW)])

    @case
    def aten_pad_constant_mode_remapped(b):
        x = b.input("x", (1, 6, 6, 3))
        p = b.permute(x, TO_NCHW)
        pad = b.call(A.pad.default, p, [0, 0, 0, 0, 0, 1], "constant", 0.0)
        b.output([b.permute(pad, TO_NHWC)])

    @case
    def aten_pad_reflect_mode_fails(b):
        x = b.input("x", (1, 6, 3))
        p = b.permute(x, [0, 2, 1])
        pad = b.call(A.pad.default, p, [1, 1], "reflect")
        b.output([b.permute(pad, [0, 2, 1])])

    @case
    def unsqueeze_and_squeeze_views(b):
        x = b.input("x", (1, 12, 4))
        v1 = b.view(b.permute(x, [0, 2, 1]), [1, 4, 1, 12])
        m = b.call(A.mul.Tensor, v1, v1)
        v2 = b.view(m, [1, 4, 12])
        b.output([b.permute(v2, [0, 2, 1])])

    @case
    def squeeze_view_into_a_lower_rank_exit(b):
        x = b.input("x", (1, 1, 4, 6))
        v = b.view(b.permute(x, TO_NCHW), [1, 6, 4])
        h = b.call(A.hardtanh.default, v)
        b.output([b.permute(h, [0, 2, 1])])

    @case
    def squeeze_of_a_moved_unit_dim(b):
        x = b.input("x", (5, 3, 1, 4))
        v = b.view(b.permute(x, [2, 0, 1, 3]), [5, 3, 4])
        s = b.call(A.slice_copy.Tensor, v, 0, 0, 3)
        m = b.call(A.mul.Tensor, s, s)
        b.output([b.permute(m, [1, 0, 2])])

    @case
    def view_adding_and_removing_two_unit_dims(b):
        x = b.input("x", (1, 4, 6))
        v1 = b.view(b.permute(x, [0, 2, 1]), [1, 1, 1, 6, 4])
        m = b.call(A.mul.Tensor, v1, v1)
        v2 = b.view(m, [1, 6, 4])
        b.output([b.permute(v2, [0, 2, 1])])

    @case
    def non_squeeze_view_fails(b):
        x = b.input("x", (1, 6, 8))
        v = b.view(b.permute(x, [0, 2, 1]), [1, 4, 12])
        m = b.call(A.mul.Tensor, v, v)
        b.output([b.permute(m, [0, 2, 1])])

    @case
    def singleton_view_boundary(b):
        x = b.input("x", (1, 8, 4))
        c = b.call(A.clamp.default, b.permute(x, [0, 2, 1]), 0.0, None)
        v = b.view(c, [1, 4, 8, 1])
        b.output([b.permute(v, TO_NHWC)])

    @case
    def shared_singleton_view_is_not_a_boundary(b):
        x = b.input("x", (1, 4, 8))
        v = b.view(b.permute(x, [0, 2, 1]), [1, 8, 4, 1])
        b.output([b.permute(v, TO_NHWC), b.permute(v, TO_NCHW)])

    @case
    def upstream_view_reached_at_another_rank_fails(b):
        x = b.input("x", (1, 12, 4))
        f = b.call(A.full.default, [4, 12], 1.0)
        v = b.view(f, [1, 4, 12])
        a = b.call(A.add.Tensor, b.permute(x, [0, 2, 1]), v)
        b.output([b.permute(a, [0, 2, 1])])

    @case
    def upstream_squeeze_view_reached_at_another_rank_fails(b):
        x = b.input("x", (4, 8))
        y = b.input("y", (8, 4, 1))
        a = b.call(A.add.Tensor, b.permute(x, [1, 0]), b.view(y, [8, 4]))
        b.output([b.permute(a, [1, 0])])

    @case
    def sink_view_below_the_region(b):
        x = b.input("x", (1, 4, 1, 1))
        direct = b.input("direct", (1, 8, 4))
        p = b.permute(x, TO_NHWC)
        v = b.view(b.call(A.mul.Tensor, p, p), [1, 1, 4])
        b.output([b.call(A.sub.Tensor, direct, v)])

    @case
    def sink_view_below_the_region_splitting_its_dim(b):
        x = b.input("x", (1, 4, 1, 1))
        p = b.permute(x, TO_NHWC)
        b.output([b.view(b.call(A.mul.Tensor, p, p), [1, 2, 2])])

    @case
    def sink_view_into_the_region_is_remapped(b):
        x = b.input("x", (1, 8, 4))
        g = b.input("g", (1, 1, 1, 4))
        skip = b.input("skip", (1, 8, 4))
        c = b.call(A.clamp.default, b.permute(x, [0, 2, 1]), 0.0, None)
        gate = b.view(g, [1, 4, 1])
        m = b.call(A.mul.Tensor, c, gate)
        r = b.call(A.add.Tensor, m, b.permute(skip, [0, 2, 1]))
        b.output([b.permute(b.view(c, [1, 4, 8, 1]), TO_NHWC), b.permute(r, [0, 2, 1])])

    @case
    def splitting_sink_view_into_the_region_fails(b):
        x = b.input("x", (1, 4, 2))
        s = b.input("s", (1, 1, 1, 8))
        m = b.call(A.mul.Tensor, b.permute(x, [0, 2, 1]), b.view(s, [1, 2, 4]))
        b.output([b.permute(m, [0, 2, 1])])

    @case
    def sink_view_with_two_users_fails(b):
        x = b.input("x", (1, 8, 4))
        g = b.input("g", (1, 1, 1, 4))
        gate = b.view(g, [1, 4, 1])
        m = b.call(A.mul.Tensor, b.permute(x, [0, 2, 1]), gate)
        b.output([b.permute(m, [0, 2, 1]), b.call(A.neg.default, gate)])

    @case
    def keyword_tensor_input(b):
        x = b.input("x", (1, 4, 1, 1))
        direct = b.input("direct", (1, 8, 4))
        p = b.permute(x, TO_NHWC)
        v = b.view(b.call(A.mul.Tensor, p, p), [1, 1, 4])
        b.output([b.call(A.sub.Tensor, v, other=direct)])

    @case
    def interleave_with_an_unsqueeze_head(b):
        x = b.input("x", (1, 16, 20, 1))
        v = interleave(b, b.permute(x, TO_NCHW), 3, 2)
        b.output([b.permute(v, TO_NHWC)])

    @case
    def interleave_with_a_view_head(b):
        x = b.input("x", (1, 16, 20, 3))
        v = interleave(b, b.permute(x, TO_NCHW), 2, 4, head="view")
        b.output([b.permute(v, TO_NHWC)])

    @case
    def interleave_with_keyword_arguments(b):
        x = b.input("x", (1, 16, 20, 1))
        p = b.permute(x, TO_NCHW)
        u = b.call(A.unsqueeze_copy.default, p, dim=4)
        e = b.call(A.expand_copy.default, u, size=[1, 1, 16, 20, 2])
        v = b.call(A.view_copy.default, e, size=[1, 1, 16, 40])
        b.output([b.permute(v, TO_NHWC)])

    @case
    def interleave_composes_with_elementwise_ops(b):
        x = b.input("x", (1, 16, 20, 1))
        p = b.permute(x, TO_NCHW)
        v = interleave(b, b.call(A.mul.Tensor, p, p), 3, 2)
        b.output([b.permute(b.call(A.hardtanh.default, v), TO_NHWC)])

    @case
    def interleave_after_a_nop_stretch(b):
        x = b.input("x", (1, 16, 20, 1))
        u0 = b.call(A.unsqueeze_copy.default, b.permute(x, TO_NCHW), 3)
        v0 = b.view(u0, [1, 1, 16, 20])
        b.output([b.permute(interleave(b, v0, 3, 2), TO_NHWC)])

    @case
    def interleave_with_an_extra_expand_user_fails(b):
        x = b.input("x", (1, 16, 20, 1))
        u = b.call(A.unsqueeze_copy.default, b.permute(x, TO_NCHW), 4)
        e = b.call(A.expand_copy.default, u, [1, 1, 16, 20, 2])
        v = b.view(e, [1, 1, 16, 40])
        b.output([b.permute(v, TO_NHWC), b.call(A.mul.Tensor, e, e)])

    @case
    def interleave_with_a_non_merging_view_fails(b):
        x = b.input("x", (1, 16, 20, 1))
        u = b.call(A.unsqueeze_copy.default, b.permute(x, TO_NCHW), 4)
        e = b.call(A.expand_copy.default, u, [1, 1, 16, 20, 2])
        b.output([b.view(e, [1, 16, 20, 2])])

    @case
    def per_tensor_and_per_channel_rescales(b):
        x = b.input("x", (1, 3, 4, 5), dtype=torch.int32)
        p = b.permute(x, TO_NHWC)
        r0 = b.call(T.RESCALE.default, p, torch.int8, [0.5], 0, 0)
        r1 = b.call(T.RESCALE.default, r0, torch.int32, [0.5, 0.25, 0.125], 0, 0)
        b.output([b.permute(r1, TO_NCHW)])

    @case
    def quantize_dequantize_region(b):
        x = b.input("x", X4)
        q = b.call(Q.quantize_per_tensor.default, b.permute(x, TO_NHWC), 0.5, 0, -128, 127, torch.int8)
        d = b.call(Q.dequantize_per_tensor.default, q, 0.5, 0, -128, 127, torch.int8)
        b.output([b.permute(d, TO_NCHW)])

    @case
    def hardswish_is_permutable_as_a_table_op(b):
        x = b.input("x", X4)
        h = b.call(A.hardswish.default, b.permute(x, TO_NHWC))
        b.output([b.permute(h, TO_NCHW)])

    @case
    def pending_table_region(b):
        x = b.input("x", X4)
        s = b.call(A.sigmoid.default, b.permute(x, TO_NHWC))
        set_qparams(s, inputs=[0], outputs=[0])
        b.output([b.permute(s, TO_NCHW)])

    @case
    def table_constant_edge_is_dropped(b):
        x = b.input("x", (1, 3, 4, 5), dtype=torch.int8)
        table = b.constant("buffer", "b_table_constant", (256,), dtype=torch.int8)
        t = b.call(T.TABLE.default, b.permute(x, TO_NHWC), table)
        b.output([b.permute(t, TO_NCHW)])

    @case
    def table_in_a_rank0_region_fails(b):
        x = b.input("x", (), dtype=torch.int8)
        table = b.constant("buffer", "b_table_constant", (256,), dtype=torch.int8)
        t = b.call(T.TABLE.default, b.permute(x, []), table)
        b.output([b.permute(t, [])])

    @case
    def pending_table_in_a_rank0_region(b):
        x = b.input("x", ())
        s = b.call(A.sigmoid.default, b.permute(x, []))
        set_qparams(s, inputs=[0], outputs=[0])
        b.output([b.permute(s, [])])

    @case
    def rank0_region(b):
        x = b.input("x", ())
        b.output([b.permute(b.call(A.relu.default, b.permute(x, [])), [])])

    @case
    def stale_shared_boundary_is_skipped(b):
        x = b.input("x", (1, 6, 6, 4))
        w = b.input("p_layer_norm_weight", (1, 1, 1, 4))
        p = b.permute(x, [0, 3, 1, 2])
        m1 = b.call(A.mul.Tensor, p, p)
        shared = b.permute(m1, TO_NHWC)
        m2 = b.call(A.mul.Tensor, shared, w)
        b.output([b.permute(m2, [0, 3, 1, 2])])

    @case
    def permute_bypassed_by_one_region_is_kept_by_another(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 4, 3))
        pa = b.permute(x, [0, 2, 1])
        t = b.call(A.tanh.default, b.permute(y, [0, 2, 1]))
        pb = b.permute(t, [0, 2, 1])
        m = b.call(A.mul.Tensor, pa, pb)
        b.output([b.permute(m, [0, 2, 1])])

    @case
    def chained_regions_fold_one_per_call(b):
        x = b.input("x", (2, 3, 4))
        pa = b.permute(x, [1, 2, 0])
        pb = b.permute(b.call(A.mul.Tensor, pa, pa), [1, 0, 2])
        pc = b.permute(b.call(A.add.Tensor, pb, pb), [0, 2, 1])
        b.output([pc])

    @case
    def exit_permute_returned_twice_and_used(b):
        x = b.input("x", X4)
        q = b.permute(b.call(A.relu.default, b.permute(x, TO_NHWC)), TO_NCHW)
        b.output([q, b.call(A.neg.default, q), q])

    @case
    def deep_region(b):
        x = b.input("x", (1, 2, 3))
        node = b.permute(x, [0, 2, 1])
        for _ in range(300):
            node = b.call(A.hardtanh.default, node)
        b.output([b.permute(node, [0, 2, 1])])

    @case
    def unrelated_dead_code_goes_when_modified(b):
        x = b.input("x", X4)
        b.call(A.neg.default, x)
        r = b.call(A.relu.default, b.permute(x, TO_NHWC))
        b.output([b.permute(r, TO_NCHW)])

    @case
    def dead_code_stays_when_nothing_changes(b):
        x = b.input("x", X4)
        b.call(A.neg.default, x)
        b.output([b.call(A.relu.default, b.permute(x, TO_NHWC))])

    @case
    def permutes_into_a_shared_region_from_both_sides(b):
        x = b.input("x", X4)
        y = b.input("y", (2, 4, 5, 3))
        a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), y)
        b.output([b.permute(a, TO_NCHW), b.permute(b.call(A.neg.default, a), TO_NCHW)])

    @case
    def two_regions_off_one_permute(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_NHWC)
        r = b.call(A.relu.default, p)
        n = b.call(A.neg.default, p)
        b.output([b.permute(r, TO_NCHW), b.permute(n, TO_NCHW)])

    @case
    def chained_entry_permutes_into_one_node(b):
        x = b.input("x", (2, 3, 3))
        p1 = b.permute(x, [0, 2, 1])
        p2 = b.permute(p1, [0, 2, 1])
        s = b.call(A.sub.Tensor, p2, p1)
        b.output([b.permute(b.call(A.exp.default, s), [0, 2, 1])])

    @case
    def constant_read_directly_and_through_an_entry_permute(b):
        c = b.constant("param", "c", (2, 3, 3))
        a = b.call(A.add.Tensor, b.permute(c, [0, 2, 1]), c)
        b.output([b.permute(b.call(A.relu.default, a), [0, 2, 1])])

    @case
    def sink_view_into_the_region_at_a_lower_rank_fails(b):
        x = b.input("x", (1, 8, 4))
        g = b.input("g", (4,))
        m = b.call(A.mul.Tensor, b.permute(x, [0, 2, 1]), b.view(g, [4, 1]))
        b.output([b.permute(m, [0, 2, 1])])

    @case
    def squeeze_view_with_an_exit_permute(b):
        x = b.input("x", (1, 6, 4))
        v = b.view(b.permute(x, [0, 2, 1]), [4, 6])
        b.output([b.permute(v, [1, 0])])

    @case
    def permute_into_a_splitting_view(b):
        x = b.input("x", (2, 3, 4))
        b.output([b.view(b.permute(x, [1, 0, 2]), [3, 2, 2, 2])])

    @case
    def expand_adding_a_leading_dim_is_not_an_interleave(b):
        x = b.input("x", (1, 4, 3))
        u = b.call(A.unsqueeze_copy.default, b.permute(x, [0, 2, 1]), 2)
        e = b.call(A.expand_copy.default, u, [2, 1, 3, 2, 4])
        b.output([e])

    @case
    def zero_scale_expand_is_not_an_interleave(b):
        x = b.input("x", (1, 4, 3))
        u = b.call(A.unsqueeze_copy.default, b.permute(x, [0, 2, 1]), 2)
        e = b.call(A.expand_copy.default, u, [1, 3, 0, 4])
        b.output([b.view(e, [1, 0, 4])])

    @case
    def expand_followed_by_a_non_view_is_not_an_interleave(b):
        x = b.input("x", (1, 4, 3))
        u = b.call(A.unsqueeze_copy.default, b.permute(x, [0, 2, 1]), 2)
        e = b.call(A.expand_copy.default, u, [1, 3, 2, 4])
        b.output([b.call(A.relu.default, e)])

    @case
    def constant_pad_nd_with_the_pad_as_a_kwarg_fails(b):
        x = b.input("x", (1, 6, 6, 3))
        pad = b.call(A.constant_pad_nd.default, b.permute(x, TO_NCHW), pad=[1, 1])
        b.output([b.permute(pad, TO_NHWC)])

    @case
    def region_with_a_qparam_permute_exit(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, b.permute(x, TO_NHWC))
        q = b.permute(r, [0, 2, 1, 3])
        set_qparams(q, inputs=[0], outputs=[0])
        b.output([q])

    return cases


class RemovePermutesRandom:
    """Seeded random graphs shaped for RemovePermutesAroundElementwiseOps:
    regions of elementwise and dim-carrying ops between a permute and,
    usually, its inverse, with constants, numel-1 operands, sink views,
    rank-changing views, interleaves, TOSA ops and blockers mixed in."""

    FLOAT_UNARY = [
        A.relu.default,
        A.abs.default,
        A.neg.default,
        A.sigmoid.default,
        A.tanh.default,
        A.exp.default,
        A.hardtanh.default,
        A.hardswish.default,
        A.erf.default,
        A.leaky_relu.default,
    ]
    INT_UNARY = [A.abs.default, A.neg.default]
    BINARY = [A.add.Tensor, A.sub.Tensor, A.mul.Tensor, A.maximum.default, A.minimum.default]

    def __init__(self, rng: random.Random, b: Builder) -> None:
        self.rng = rng
        self.b = b
        self.inputs = 0
        self.perm: dict[torch.fx.Node, list[int]] = {}
        self.exits: list[torch.fx.Node] = []
        self.values: list[torch.fx.Node] = []

    # Helpers ---------------------------------------------------------------

    def name(self, prefix="x") -> str:
        self.inputs += 1
        return f"{prefix}{self.inputs}"

    def shape(self, rank: int) -> list[int]:
        rng = self.rng
        while True:
            dims = [1 if rng.random() < 0.25 else rng.randint(2, 5) for _ in range(rank)]
            if math.prod(dims) <= 240:
                return dims

    def try_call(self, target, *args, **kwargs):
        try:
            return self.b.call(target, *args, **kwargs)
        except Exception:  # noqa: BLE001  torch rejected the op; skip the step
            return None

    def new_input(self, shape, dtype=torch.float32):
        rng = self.rng
        roll = rng.random()
        if roll < 0.15:
            kind = rng.choice(["param", "buffer", "constant"])
            node = self.b.constant(kind, self.name("w"), shape, dtype)
        elif roll < 0.2:
            node = self.b.input(self.name(rng.choice(["p_", "b_", "c_"])), shape, dtype)
        else:
            node = self.b.input(self.name(), shape, dtype)
        return node

    def operand(self, like: torch.fx.Node) -> torch.fx.Node | float:
        """A second operand for a binary op on the region value `like`."""
        rng = self.rng
        shape = list(like.meta["val"].shape)
        dtype = like.meta["val"].dtype
        rank = len(shape)
        roll = rng.random()
        if roll < 0.25:
            same = [v for v in self.values if list(v.meta["val"].shape) == shape and v.meta["val"].dtype == dtype]
            if same:
                return rng.choice(same)
        if roll < 0.45:
            # A constant of the region's rank, possibly broadcasting.
            const_shape = [1 if rng.random() < 0.3 else d for d in shape]
            if rng.random() < 0.3:
                return self.try_call(A.full.default, const_shape, 0.5, dtype=dtype) or 2.0
            kind = rng.choice(["param", "buffer", "constant"])
            return self.b.constant(kind, self.name("w"), const_shape, dtype)
        if roll < 0.52:
            # A lower-rank constant.
            low = rng.randint(1, max(1, rank - 1)) if rank > 1 else 1
            const_shape = shape[rank - low :] if rank else [1]
            return self.b.constant("param", self.name("w"), const_shape, dtype)
        if roll < 0.62:
            return self.new_input([1] * rng.randint(0, rank), dtype)
        if roll < 0.68 and dtype == torch.float32:
            return 2.0
        if roll < 0.78 and like in self.perm and rank:
            # Another permuted input with the same permutation (or not).
            perm = self.perm[like]
            if rng.random() < 0.25:
                perm = list(range(rank))
                rng.shuffle(perm)
            source_shape = [0] * rank
            for i, d in enumerate(perm):
                source_shape[d] = shape[i]
            return self.b.permute(self.new_input(source_shape, dtype), perm)
        if roll < 0.86 and rank:
            # A sink view into the region: one non-unit dim at most.
            non_unit = [i for i, d in enumerate(shape) if d != 1]
            target = [1] * rank
            if non_unit:
                i = rng.choice(non_unit)
                target[i] = shape[i]
            source = self.new_input([math.prod(target)] if rng.random() < 0.5 else [1, math.prod(target)], dtype)
            return self.try_call(A.view_copy.default, source, target) or 2.0
        return self.new_input(shape, dtype)

    def keep(self, node, perm):
        if node is not None:
            self.values.append(node)
            if perm is not None:
                self.perm[node] = perm
        return node

    def pick(self) -> torch.fx.Node:
        values = self.values
        return values[-1 - min(int(self.rng.expovariate(0.8)), len(values) - 1)]

    # Steps -----------------------------------------------------------------

    def step(self) -> None:
        rng = self.rng
        v = self.pick()
        val = v.meta["val"]
        shape = list(val.shape)
        rank = len(shape)
        is_float = val.dtype == torch.float32
        perm = self.perm.get(v)
        kind = rng.choices(
            [
                "unary",
                "binary",
                "reduction",
                "cat",
                "slice",
                "pad",
                "rescale",
                "qdq",
                "table",
                "unsqueeze",
                "squeeze",
                "interleave",
                "inner_permute",
                "blocker",
            ],
            [5, 6, 2, 1.5, 1.5, 1, 1, 0.7, 1, 1.2, 1.2, 0.6, 0.6, 0.4],
        )[0]
        b = self.b
        if kind == "unary":
            target = rng.choice(self.FLOAT_UNARY if is_float else self.INT_UNARY)
            self.keep(self.try_call(target, v), perm)
        elif kind == "binary":
            other = self.operand(v)
            args = (v, other) if rng.random() < 0.7 or not isinstance(other, torch.fx.Node) else (other, v)
            self.keep(self.try_call(rng.choice(self.BINARY), *args), perm)
        elif kind == "reduction" and rank:
            target = rng.choice([A.amax.default, A.amin.default, A.mean.dim, A.sum.dim_IntList])
            if not is_float and target is A.mean.dim:
                return
            dims = rng.sample(range(rank), rng.randint(1, rank))
            dims = [d - rank if rng.random() < 0.3 else d for d in dims]
            dim = dims[0] if len(dims) == 1 and rng.random() < 0.4 and target in (A.amax.default, A.amin.default) else dims
            keepdim = rng.random() < 0.85
            if rng.random() < 0.5:
                node = self.try_call(target, v, dim, keepdim)
            else:
                node = self.try_call(target, v, dim, keepdim=keepdim)
            if node is not None and keepdim:
                self.keep(self.try_call(A.sub.Tensor, v, node), perm)
            elif node is not None:
                self.exits.append(node)
        elif kind == "cat" and rank:
            dim = rng.randrange(rank)
            dim = dim - rank if rng.random() < 0.3 else dim
            others = [o for o in self.values if list(o.meta["val"].shape) == shape and o.meta["val"].dtype == val.dtype]
            tensors = [v, rng.choice(others)]
            roll = rng.random()
            if roll < 0.5:
                node = self.try_call(A.cat.default, tensors, dim)
            elif roll < 0.8:
                node = self.try_call(A.cat.default, tensors, dim=dim)
            else:
                node = self.try_call(A.cat.default, tensors)
            self.keep(node, perm)
        elif kind == "slice" and rank:
            dims = [i for i, d in enumerate(shape) if d >= 2]
            if not dims:
                return
            dim = rng.choice(dims)
            end = rng.randint(1, shape[dim])
            if rng.random() < 0.6:
                node = self.try_call(A.slice_copy.Tensor, v, dim, 0, end)
            else:
                node = self.try_call(A.slice_copy.Tensor, v, dim=dim, start=0, end=end)
            self.keep(node, perm)
        elif kind == "pad" and rank and is_float:
            pairs = rng.randint(1, rank)
            pad = []
            for _ in range(pairs):
                pad += [rng.randint(0, 2), rng.randint(0, 1)]
            roll = rng.random()
            if roll < 0.5:
                node = self.try_call(A.constant_pad_nd.default, v, pad, 0.0)
            elif roll < 0.85:
                node = self.try_call(A.pad.default, v, pad, "constant", 0.0)
            else:
                node = self.try_call(A.pad.default, v, pad[:2], "reflect")
            self.keep(node, perm)
        elif kind == "rescale" and not is_float:
            if rng.random() < 0.8 or not rank:
                scales = [0.5]
            else:
                scales = [0.5] * shape[-1]
            dtype = torch.int8 if val.dtype == torch.int32 else torch.int32
            self.keep(self.try_call(T.RESCALE.default, v, dtype, scales, 0, 0), perm)
        elif kind == "qdq" and is_float:
            q = self.try_call(Q.quantize_per_tensor.default, v, 0.5, 0, -128, 127, torch.int8)
            if q is not None:
                self.keep(self.try_call(Q.dequantize_per_tensor.default, q, 0.5, 0, -128, 127, torch.int8), perm)
        elif kind == "table" and is_float and rank:
            node = self.try_call(rng.choice([A.sigmoid.default, A.tanh.default, A.exp.default]), v)
            if node is not None:
                set_qparams(node, inputs=[0], outputs=[0])
            self.keep(node, perm)
        elif kind == "unsqueeze" and rank < 5:
            positions = sorted(rng.sample(range(rank + 2), rng.randint(1, 2)))
            out = list(shape)
            for p in positions:
                out.insert(min(p, len(out)), 1)
            node = self.try_call(A.view_copy.default, v, out)
            self.keep(node, adapt_permutation(perm, shape, out) if perm else None)
        elif kind == "squeeze" and 1 in shape:
            ones = [i for i, d in enumerate(shape) if d == 1]
            drop = set(rng.sample(ones, rng.randint(1, len(ones))))
            out = [d for i, d in enumerate(shape) if i not in drop]
            node = self.try_call(A.view_copy.default, v, out)
            self.keep(node, adapt_permutation(perm, shape, out) if perm else None)
        elif kind == "interleave" and rank:
            dim = rng.randrange(rank)
            try:
                node = interleave(b, v, dim, rng.randint(1, 3), head=rng.choice(["unsqueeze", "view"]))
            except Exception:  # noqa: BLE001
                node = None
            self.keep(node, perm)
        elif kind == "inner_permute" and rank:
            inner = list(range(rank))
            rng.shuffle(inner)
            node = b.permute(v, inner)
            self.keep(node, inner)
        elif kind == "blocker" and rank:
            self.exits.append(self.try_call(A.cumsum.default, v, rng.randrange(rank)))

    def region(self) -> None:
        rng = self.rng
        rank = rng.choices([0, 1, 2, 3, 4, 5], [0.3, 0.6, 1.5, 3, 4, 1])[0]
        dtype = torch.int32 if rng.random() < 0.15 else torch.float32
        perm = list(range(rank))
        rng.shuffle(perm)
        source = self.new_input(self.shape(rank), dtype)
        written = [d - rank if rng.random() < 0.2 else d for d in perm]
        entry = self.b.permute(source, written)
        self.values = [entry]
        self.perm = {entry: perm}
        for _ in range(rng.randint(1, 7)):
            self.step()
        for _ in range(rng.randint(1, 2)):
            self.exit(self.pick())

    def exit(self, v) -> None:
        rng = self.rng
        val = v.meta["val"]
        rank = val.dim()
        perm = self.perm.get(v)
        roll = rng.random()
        if roll < 0.6 and perm is not None:
            self.exits.append(self.b.permute(v, inverse_permutation(perm)))
        elif roll < 0.8:
            other = list(range(rank))
            rng.shuffle(other)
            self.exits.append(self.b.permute(v, other))
        elif roll < 0.9 and sum(d != 1 for d in val.shape) <= 1:
            self.exits.append(self.try_call(A.view_copy.default, v, [val.numel()]))
        else:
            self.exits.append(v)

    def finish(self) -> None:
        rng = self.rng
        outputs = [n for n in self.exits if n is not None and not n.users]
        dangling = [n for n in self.b.graph.nodes if n.op == "call_function" and not n.users]
        outputs += [n for n in dangling if n not in outputs and rng.random() < 0.2]
        if not outputs:
            outputs = [self.values[-1]]
        if rng.random() < 0.15:
            outputs.append(rng.choice(self.values))
        if rng.random() < 0.1:
            self.b.call(A.neg.default, rng.choice(self.values))  # dead code
        self.b.output(outputs)


def adapt_permutation(perm, in_shape, out_shape):
    """rc4's `_adapt_permute_across_view` for a squeeze/unsqueeze view, so the
    generator can close a region across a rank change."""

    def extra_ones(longer, shorter):
        extra, j = [], 0
        for i, d in enumerate(longer):
            if j < len(shorter) and d == shorter[j]:
                j += 1
                continue
            if d != 1:
                return None
            extra.append(i)
        return extra if j == len(shorter) else None

    if len(out_shape) > len(in_shape):
        positions = extra_ones(out_shape, in_shape)
        if positions is None:
            return None
        new = list(perm)
        for index in positions:
            new = [x + 1 if x >= index else x for x in new]
            new.insert(index, index)
        return new
    if len(in_shape) > len(out_shape):
        positions = extra_ones(in_shape, out_shape)
        if positions is None:
            return None
        removed = {perm[p] for p in positions}
        return [x - sum(1 for r in removed if r < x) for x in perm if x not in removed]
    return None


def remove_permutes_random_case(seed: int) -> Callable[[Builder], None]:
    def build(b: Builder) -> None:
        rng = random.Random(seed)
        graph = RemovePermutesRandom(rng, b)
        for _ in range(rng.randint(1, 3)):
            graph.region()
        graph.finish()

    return build


# --------------------------------------------------------------------------
# RemovePermutesAroundElementwiseTosaOps on users out of graph order
# --------------------------------------------------------------------------


def shared_boundary(b: Builder, touched=()) -> None:
    """P = permute(x) feeds U1 = relu(P), whose exit E1 is also an entry of
    U2 = add(P, E1): the regions from U1 and U2 share E1, so the one the
    sweep meets first among P's users is rewritten. `touched` names nodes to
    touch, in order: "u1", "u2"."""
    x = b.input("x", (4, 4))
    b.input("y", (4, 4))
    p = b.permute(x, [1, 0])
    u1 = b.call(A.relu.default, p)
    e1 = b.permute(u1, [1, 0])
    u2 = b.call(A.add.Tensor, p, e1)
    b.output([b.permute(u2, [1, 0])])
    for name in touched:
        b.touch({"u1": u1, "u2": u2}[name])


def remove_permutes_user_order_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    @case
    def users_in_graph_order(b):
        shared_boundary(b)

    @case
    def touched_first_user_goes_last(b):
        # P.users = [U2, U1], as after U1 is rewired to read P.
        shared_boundary(b, ["u1"])

    @case
    def touching_users_in_graph_order_keeps_it(b):
        shared_boundary(b, ["u1", "u2"])

    @case
    def touching_the_last_user_changes_nothing(b):
        shared_boundary(b, ["u2"])

    @case
    def three_users_of_one_permute(b):
        """Three regions from P's users, two of which share an exit."""
        x = b.input("x", (2, 3, 4))
        p = b.permute(x, [0, 2, 1])
        u1 = b.call(A.neg.default, p)
        e1 = b.permute(u1, [0, 2, 1])
        u2 = b.call(A.mul.Tensor, p, b.permute(e1, [0, 2, 1]))
        u3 = b.call(A.abs.default, p)
        b.output([b.permute(u2, [0, 2, 1]), b.permute(u3, [0, 2, 1]), e1])
        b.touch(u1)
        b.touch(u3)

    @case
    def region_user_order_inside_the_search(b):
        """A region node whose users are visited in touched order: one user
        leads to a non-matching exit, the other to the inverse."""
        x = b.input("x", (1, 3, 4, 5))
        p = b.permute(x, TO_NHWC)
        r = b.call(A.relu.default, p)
        a = b.call(A.neg.default, r)
        c = b.call(A.abs.default, r)
        b.output([b.permute(a, TO_NCHW), b.permute(c, [0, 3, 2, 1])])
        b.touch(a)

    return cases


# Self-inverse permutations: with one of these every exit permute is also an
# entry permute, as in `shared_boundary`.
INVOLUTIONS = {
    2: [[1, 0]],
    3: [[0, 2, 1], [2, 1, 0], [1, 0, 2]],
    4: [[0, 1, 3, 2], [1, 0, 2, 3], [0, 3, 2, 1], [2, 1, 0, 3], [1, 0, 3, 2], [3, 2, 1, 0]],
}


class SharedBoundaryRandom:
    """Seeded random graphs of small regions that share entry and exit
    permutes: one self-inverse permutation P and a shape P maps to itself, so
    a permute by P can enter any region and every region's exit can enter a
    later one. Most regions read the first entry permute, so its users start
    regions that compete for the same exits: which one the sweep meets first,
    and so which one is rewritten, depends on the order of its users."""

    UNARY = [A.relu.default, A.neg.default, A.abs.default, A.sigmoid.default]
    BINARY = [A.add.Tensor, A.mul.Tensor, A.sub.Tensor]

    def __init__(self, rng: random.Random, b: Builder) -> None:
        self.rng = rng
        self.b = b
        self.perm = rng.choice(INVOLUTIONS[rng.choice([2, 3, 4])])
        self.shape = [0] * len(self.perm)
        for i, j in enumerate(self.perm):
            if not self.shape[i]:
                self.shape[i] = self.shape[j] = rng.randint(2, 4)
        self.count = 0
        self.first = self.b.permute(self.new_input(), self.perm)
        self.exits: list[torch.fx.Node] = []
        self.open_ends: list[torch.fx.Node] = []

    def new_input(self) -> torch.fx.Node:
        self.count += 1
        if self.rng.random() < 0.15:
            return self.b.constant("param", f"w{self.count}", self.shape)
        return self.b.input(f"x{self.count}", self.shape)

    def entry(self) -> torch.fx.Node:
        rng = self.rng
        roll = rng.random()
        if roll < 0.45 or not self.exits:
            return self.first if roll < 0.85 else self.b.permute(self.new_input(), self.perm)
        return rng.choice(self.exits)

    def region(self) -> None:
        rng = self.rng
        value = self.entry()
        if rng.random() < 0.5:
            value = self.b.call(rng.choice(self.BINARY), value, self.entry())
        for _ in range(rng.randint(0, 2)):
            if rng.random() < 0.6:
                value = self.b.call(rng.choice(self.UNARY), value)
            else:
                other = self.entry() if rng.random() < 0.7 else self.new_input()
                value = self.b.call(rng.choice(self.BINARY), value, other)
        if value.target != A.permute_copy.default and rng.random() < 0.85:
            self.exits.append(self.b.permute(value, self.perm))
        else:
            self.open_ends.append(value)

    def finish(self) -> None:
        outputs = [n for n in self.exits if not n.users] + self.open_ends
        self.b.output(outputs or [self.first])


def remove_permutes_user_order_random_case(seed: int) -> Callable[[Builder], None]:
    """An even seed takes the `remove_permutes` random graph (same seed, same
    graph), an odd one a `SharedBoundaryRandom` graph; then a random subset of
    the call nodes is touched in a random order."""

    def build(b: Builder) -> None:
        if seed % 2 == 0:
            remove_permutes_random_case(seed)(b)
        else:
            rng = random.Random(seed)
            graph = SharedBoundaryRandom(rng, b)
            for _ in range(rng.randint(2, 6)):
                graph.region()
            graph.finish()
        rng = random.Random(seed ^ 0x5EED)
        calls = [node for node in b.graph.nodes if node.op == "call_function"]
        for node in rng.sample(calls, rng.randint(1, len(calls))):
            b.touch(node)

    return build


# --------------------------------------------------------------------------
# PropagateViewCopyPermuteDownPass / PropagateViewCopyPermuteUpPass
# --------------------------------------------------------------------------

# 16-bit TABLE ops need the int16 extension.
PROPAGATE_TOSA_SPEC = "TOSA-1.0+FP+INT+int16"
_PROPAGATION_TABLE_OF = "retab_replay_propagation_table_of"
_PROPAGATION_TABLE_RESCALE = "retab_replay_propagation_table_rescale"


_PROPAGATION_ORIGINAL = "retab_replay_propagation_original"


def insert_table_ops(ep, gm):
    """The real InsertTableOpsPass (`arm_pass_manager.py:708`), with each
    TABLE it makes marked with the name of the table op it replaced and each
    RESCALE it makes with its TABLE's mark. Returns the graph module and the
    table ops by name. The pass's retrace renames nodes, so a RESCALE the
    graph already had can take the name of a new one: new nodes are told apart
    by a marker set on every node first, which the retrace carries over (it
    copies each node's meta), not by name."""
    originals = {node.name: node for node in gm.graph.nodes if _is_table_op(node)}
    if not originals:
        return gm, originals
    for node in gm.graph.nodes:
        node.meta[_PROPAGATION_ORIGINAL] = True
    gm = InsertTableOpsPass(ep)(gm).graph_module
    new_nodes = [node for node in gm.graph.nodes if not node.meta.pop(_PROPAGATION_ORIGINAL, False)]
    for node in new_nodes:
        if node.target == T.TABLE.default:
            constant = node.args[1].name
            node.meta[_PROPAGATION_TABLE_OF] = constant.removeprefix("b_").removesuffix("_table_constant")
    for node in new_nodes:
        if node.target == T.RESCALE.default:
            (table,) = node.all_input_nodes
            node.meta[_PROPAGATION_TABLE_OF] = table.meta[_PROPAGATION_TABLE_OF]
            node.meta[_PROPAGATION_TABLE_RESCALE] = True
    return gm, originals


class PropagationPass:
    """rc4's Arm propagation pass, constructed as `arm_pass_manager.py:726-727`
    constructs it (`PropagateViewCopyPermuteDownPass(compile_spec,
    exported_program)`), on a graph whose quantized table ops are TABLE ops.

    The real InsertTableOpsPass (`arm_pass_manager.py:708`) brings the graph
    to that stage first (`insert_table_ops`), and afterwards every TABLE
    becomes the table op it replaced again, reading the TABLE's input
    (`restore_pending_tables`), so the fixture shows rc4's decisions in the
    form RETAB keeps (`layout::pending`). `modified` is the propagation pass's
    own. `pass_classes` lists the passes to run in turn: one, or the Down and
    the Up pass as phase 6 runs them.
    """

    def __init__(self, ep, *pass_classes, compile_spec=True, with_program=True):
        self.ep = ep
        self.pass_classes = pass_classes
        self.compile_spec = TosaCompileSpec(PROPAGATE_TOSA_SPEC) if compile_spec else None
        self.with_program = with_program
        names = " + ".join(f"{cls.__module__.split('.')[1]} {cls.__name__}" for cls in pass_classes)
        self.rc4_name = f"InsertTableOpsPass + {names}"

    def __call__(self, gm):
        gm, originals = insert_table_ops(self.ep, gm)
        modified = False
        for pass_cls in self.pass_classes:
            if self.with_program:
                graph_pass = pass_cls(self.compile_spec, self.ep)
            else:
                graph_pass = pass_cls()
            result = graph_pass(gm)
            gm = result.graph_module
            modified |= result.modified
        if originals:
            restore_pending_tables(gm, originals)
        return PassResult(gm, modified)


def restore_pending_tables(gm, originals) -> None:
    """Turn every TABLE back into the table op it replaced, reading the
    TABLE's input. Where a 16-bit TABLE's RESCALE still reads only the TABLE
    and is its only user, the table op takes the RESCALE's place and value;
    where rc4 left the two apart (a view parked between them), the table op
    holds the TABLE's int32 value and the RESCALE stays a node, which is
    RETAB's split form (`layout::pending::split_table_rescales`)."""
    graph = gm.graph
    rescale_of = {
        node.meta[_PROPAGATION_TABLE_OF]: node
        for node in graph.nodes
        if node.target == T.RESCALE.default and node.meta.get(_PROPAGATION_TABLE_RESCALE)
    }
    for table in [n for n in graph.nodes if n.target == T.TABLE.default]:
        if _PROPAGATION_TABLE_OF not in table.meta:
            continue
        original = originals[table.meta[_PROPAGATION_TABLE_OF]]
        rescale = rescale_of.get(table.meta[_PROPAGATION_TABLE_OF])
        merged = rescale is not None and list(table.users) == [rescale] and rescale.all_input_nodes == [table]
        output = rescale if merged else table
        with graph.inserting_before(output):
            restored = graph.create_node(
                "call_function",
                original.target,
                (table.args[0], *original.args[1:]),
                dict(original.kwargs),
            )
        restored.meta = dict(original.meta)
        restored.meta["val"] = output.meta["val"]
        output.replace_all_uses_with(restored)
        graph.erase_node(output)
        if merged:
            graph.erase_node(table)
    for rescale in rescale_of.values():
        rescale.meta.pop(_PROPAGATION_TABLE_OF, None)
        rescale.meta.pop(_PROPAGATION_TABLE_RESCALE, None)
    for node in list(graph.nodes):
        if node.op == "placeholder" and node.name.startswith("b_") and not node.users:
            graph.erase_node(node)


class PropagationRunError(Exception):
    pass


def propagation_cases():
    cases = {}

    def case(fn):
        cases[fn.__name__] = fn
        return fn

    X4 = (1, 2, 3, 4)
    TO_LAST = [0, 2, 3, 1]
    TO_FIRST = [0, 3, 1, 2]

    def unaries(b, x, ops=(A.relu.default, A.neg.default)):
        for op in ops:
            x = b.call(op, x)
        return x

    def int8_table(b, x, target=A.sigmoid.default, *args):
        return table_op(b, target, x, qargs(0.5), qargs(0.25), *args)

    def int16_table(b, x, target=A.sigmoid.default):
        return table_op(b, target, x, qargs(1 / 1024, dtype=torch.int16), qargs(1 / 32768, dtype=torch.int16))

    # Moves across pointwise ops ---------------------------------------

    @case
    def permute_then_unary_chain(b):
        x = b.input("x", X4)
        b.output([unaries(b, b.permute(x, TO_LAST))])

    @case
    def unary_chain_then_permute(b):
        x = b.input("x", X4)
        b.output([b.permute(unaries(b, x), TO_LAST)])

    @case
    def view_then_unary_chain(b):
        x = b.input("x", (2, 3, 4))
        b.output([unaries(b, b.view(x, [2, 12]))])

    @case
    def unary_chain_then_view(b):
        x = b.input("x", (2, 3, 4))
        b.output([b.view(unaries(b, x), [2, 12])])

    @case
    def view_with_an_inferred_dim(b):
        x = b.input("x", (2, 3, 4))
        b.output([b.view(unaries(b, b.view(x, [-1, 4])), [6, 2, 2])])

    @case
    def negative_permute_dims(b):
        x = b.input("x", X4)
        b.output([b.permute(unaries(b, b.permute(x, [0, -2, -1, 1])), [0, -1, 1, 2])])

    @case
    def rank0_view_and_permute(b):
        x = b.input("x", ())
        v = b.view(b.call(A.relu.default, b.permute(x, [])), [1])
        b.output([b.call(A.neg.default, v)])

    @case
    def permute_returned_directly(b):
        x = b.input("x", X4)
        b.output([b.permute(x, TO_LAST)])

    @case
    def permute_of_a_placeholder_used_twice(b):
        x = b.input("x", X4)
        b.output([b.call(A.relu.default, b.permute(x, TO_LAST)), b.call(A.neg.default, x)])

    @case
    def shared_producer_stops_up(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, x)
        b.output([b.permute(b.call(A.neg.default, r), TO_LAST), b.call(A.abs.default, r)])

    @case
    def transparent_copies(b):
        x = b.input("x", X4)
        c = b.call(D._clone_dim_order.default, b.permute(x, TO_LAST))
        t = b.call(D._to_dim_order_copy.default, c, dtype=torch.float32)
        b.output([b.permute(b.call(A.relu.default, t), TO_FIRST)])

    @case
    def rank_changing_view_stops_at_a_transparent_copy(b):
        x = b.input("x", (2, 3, 4))
        c = b.call(D._clone_dim_order.default, b.view(x, [6, 4]))
        b.output([b.view(b.call(D._clone_dim_order.default, b.call(A.relu.default, c)), [24])])

    @case
    def rank_preserving_view_crosses_a_transparent_copy(b):
        x = b.input("x", (2, 3, 4))
        c = b.call(D._clone_dim_order.default, b.view(x, [3, 2, 4]))
        b.output([b.view(b.call(D._clone_dim_order.default, c), [4, 3, 2])])

    @case
    def aten_view_default_crosses_pointwise_ops(b):
        x = b.input("x", (2, 3, 4))
        v = b.call(A.view.default, x, [6, 4])
        b.output([b.call(A.view.default, unaries(b, v), [24])])

    @case
    def aten_view_default_meets_a_slice(b):
        x = b.input("x", (2, 3, 4))
        s = b.call(A.slice_copy.Tensor, x, 0, 0, 1)
        b.output([b.call(A.view.default, s, [12])])

    @case
    def dtype_changing_pointwise_ops(b):
        x = b.input("x", X4)
        y = b.input("y", X4)
        p = b.permute(x, TO_LAST)
        g = b.call(A.gt.Tensor, p, b.permute(y, TO_LAST))
        b.output([b.permute(b.call(A.logical_not.default, g), TO_FIRST)])

    # Binary ops -------------------------------------------------------

    @case
    def up_crosses_a_binary_op_with_a_layout_invariant_operand(b):
        x = b.input("x", X4)
        ones = b.input("ones", (1, 1, 1, 1))
        b.output([b.permute(b.call(A.add.Tensor, x, ones), TO_LAST)])

    @case
    def up_stops_at_a_binary_op_with_two_layout_operands(b):
        x = b.input("x", X4)
        y = b.input("y", X4)
        b.output([b.permute(b.call(A.add.Tensor, x, y), TO_LAST)])

    @case
    def up_crosses_a_rank_raising_broadcast(b):
        x = b.input("x", (3, 4))
        ones = b.input("ones", (1, 1, 1, 1))
        b.output([b.permute(b.call(A.mul.Tensor, x, ones), [0, 1, 3, 2])])

    @case
    def up_rank_raising_broadcast_that_moves_a_new_axis(b):
        x = b.input("x", (3, 4))
        ones = b.input("ones", (1, 1, 1, 1))
        b.output([b.permute(b.call(A.mul.Tensor, x, ones), [1, 0, 3, 2])])

    @case
    def down_stops_at_a_binary_op_with_a_scalar_like_operand(b):
        x = b.input("x", X4)
        ones = b.input("ones", (1, 1))
        b.output([b.call(A.add.Tensor, b.permute(x, TO_LAST), ones)])

    @case
    def identical_permutes_into_a_binary_op(b):
        x = b.input("x", X4)
        y = b.input("y", X4)
        b.output([b.call(A.add.Tensor, b.permute(x, TO_LAST), b.permute(y, TO_LAST))])

    @case
    def identical_views_into_a_cat(b):
        x = b.input("x", (2, 3, 4))
        y = b.input("y", (2, 3, 4))
        b.output([b.call(A.cat.default, [b.view(x, [6, 4]), b.view(y, [6, 4])], 1)])

    @case
    def where_with_a_broadcast_condition(b):
        c = b.input("c", (1, 1, 1), dtype=torch.bool)
        x = b.input("x", (1, 3, 5))
        z = b.input("z", (1, 1, 1))
        b.output([b.permute(b.call(A.where.self, c, x, z), [0, 2, 1])])

    @case
    def binary_with_itself(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        b.output([b.call(A.mul.Tensor, p, p)])

    # Reductions -------------------------------------------------------

    @case
    def permute_crosses_a_keepdim_mean(b):
        x = b.input("x", X4)
        m = b.call(A.mean.dim, b.permute(x, TO_LAST), [1, -1], True)
        b.output([b.call(A.relu.default, m)])

    @case
    def keepdim_mean_then_permute(b):
        x = b.input("x", X4)
        b.output([b.permute(b.call(A.mean.dim, x, [2], True), TO_LAST)])

    @case
    def mean_with_an_int_dim(b):
        x = b.input("x", X4)
        b.output([b.permute(b.call(A.mean.dim, b.permute(x, TO_LAST), -1, True), TO_FIRST)])

    @case
    def mean_without_keepdim_raises(b):
        x = b.input("x", X4)
        b.output([b.call(A.mean.dim, b.permute(x, TO_LAST), [1])])

    @case
    def mean_with_keepdim_false_raises(b):
        x = b.input("x", X4)
        b.output([b.permute(b.call(A.mean.dim, x, [1], False), [0, 2, 1])])

    @case
    def keepdim_as_a_kwarg(b):
        x = b.input("x", X4)
        b.output([b.permute(b.call(A.mean.dim, x, [1], keepdim=True), TO_LAST)])

    @case
    def sum_is_not_a_tosa_reduction(b):
        x = b.input("x", X4)
        s = b.call(A.sum.dim_IntList, b.permute(x, TO_LAST), [3], True)
        b.output([b.permute(s, TO_FIRST)])

    @case
    def view_moves_before_a_mean_with_a_split_dim(b):
        x = b.input("x", (6, 4))
        b.output([b.view(b.call(A.mean.dim, x, [0], True), [1, 1, 4])])

    @case
    def view_stays_after_a_mean_it_cannot_map(b):
        x = b.input("x", (1, 50, 10, 20))
        b.output([b.view(b.call(A.mean.dim, x, [-1], True), [1, 50, 10])])

    @case
    def view_moves_after_a_mean(b):
        x = b.input("x", (2, 3, 4))
        m = b.call(A.mean.dim, b.view(x, [2, 12]), [1], True)
        b.output([b.call(A.relu.default, m)])

    @case
    def view_crosses_a_mean_over_whole_groups(b):
        x = b.input("x", (4, 6))
        m = b.call(A.mean.dim, b.view(x, [2, 2, 6]), [0, 1], True)
        b.output([b.call(A.relu.default, m)])

    # Slices -----------------------------------------------------------

    @case
    def permute_crosses_a_slice(b):
        x = b.input("x", (1, 4, 3, 3))
        s = b.call(A.slice_copy.Tensor, b.permute(x, TO_LAST), 3, 0, 2)
        b.output([b.call(A.relu.default, s)])

    @case
    def slice_then_permute(b):
        x = b.input("x", (3, 2, 8, 16))
        s = b.call(A.slice_copy.Tensor, x, 0, 0, 1)
        b.output([b.permute(s, [0, 3, 1, 2])])

    @case
    def slice_then_permute_then_view(b):
        x = b.input("x", (3, 2, 8, 16))
        s = b.call(A.slice_copy.Tensor, x, 0, 0, 1)
        b.output([b.view(b.permute(s, [0, 3, 1, 2]), [1, 32, 8])])

    @case
    def view_moves_before_a_unit_slice(b):
        x = b.input("x", (5, 2, 3, 4, 6))
        s = b.call(A.slice_copy.Tensor, x, 2, 1, 2)
        b.output([b.view(s, [5, 2, 24])])

    @case
    def unit_slice_before_a_reordering_view(b):
        x = b.input("x", (1, 3, 1, 7))
        s = b.call(A.slice_copy.Tensor, x, 3, 2, 3)
        b.output([b.view(s, [1, 1, 1, 3])])

    @case
    def view_with_a_mapped_slice_dim(b):
        x = b.input("x", (4, 3, 5))
        s = b.call(A.slice_copy.Tensor, x, 2, 1, 4)
        b.output([b.view(s, [12, 3])])

    @case
    def split_dim_view_stops_at_a_slice(b):
        x = b.input("x", (6, 4))
        s = b.call(A.slice_copy.Tensor, x, 0, 0, 4)
        b.output([b.view(s, [2, 2, 4])])

    @case
    def rank_changing_view_stops_at_a_slice(b):
        x = b.input("x", (1, 2, 3))
        s = b.call(A.slice_copy.Tensor, x, 0, 0, 1)
        b.output([b.view(s, [2, 3])])

    @case
    def slice_with_a_step(b):
        x = b.input("x", (4, 6))
        s = b.call(A.slice_copy.Tensor, x, 1, 0, 6, 2)
        b.output([b.permute(s, [1, 0])])

    @case
    def slice_with_open_bounds(b):
        x = b.input("x", (4, 6))
        s = b.call(A.slice_copy.Tensor, x, 1, None, 3)
        b.output([b.view(s, [12])])

    @case
    def view_moves_after_a_slice(b):
        x = b.input("x", (2, 3, 4))
        s = b.call(A.slice_copy.Tensor, b.view(x, [6, 4]), 1, 1, 3)
        b.output([b.call(A.relu.default, s)])

    # Cat fan-outs -----------------------------------------------------

    @case
    def noop_permute_after_a_cat_splits(b):
        x = b.input("x", (1, 1, 3, 4))
        y = b.input("y", (1, 1, 3, 4))
        b.output([b.permute(b.call(A.cat.default, [x, y], 1), TO_LAST)])

    @case
    def two_equal_permutes_after_a_cat(b):
        x = b.input("x", (1, 1, 3, 4))
        y = b.input("y", (1, 1, 3, 4))
        c = b.call(A.cat.default, [x, y], 1)
        b.output([b.permute(c, TO_LAST), b.call(A.relu.default, b.permute(c, TO_LAST))])

    @case
    def different_permutes_after_a_cat(b):
        x = b.input("x", (1, 1, 3, 4))
        y = b.input("y", (1, 1, 3, 4))
        c = b.call(A.cat.default, [x, y], 1)
        b.output([b.permute(c, TO_LAST), b.permute(c, [0, 2, 1, 3])])

    @case
    def cat_with_a_non_permute_user(b):
        x = b.input("x", (1, 1, 3, 4))
        y = b.input("y", (1, 1, 3, 4))
        c = b.call(A.cat.default, [x, y], 1)
        b.output([b.permute(c, TO_LAST), b.call(A.relu.default, c)])

    @case
    def permute_after_a_cat_moving_real_dims(b):
        x = b.input("x", (1, 2, 3, 4))
        y = b.input("y", (1, 2, 3, 4))
        b.output([b.permute(b.call(A.cat.default, [x, y], 1), TO_LAST)])

    @case
    def cat_without_a_dim(b):
        x = b.input("x", (1, 1, 3, 4))
        y = b.input("y", (2, 1, 3, 4))
        b.output([b.permute(b.call(A.cat.default, [x, y]), [1, 0, 2, 3])])

    @case
    def cat_dim_as_a_kwarg(b):
        x = b.input("x", (1, 1, 3, 4))
        y = b.input("y", (1, 1, 3, 4))
        b.output([b.permute(b.call(A.cat.default, [x, y], dim=1), TO_LAST)])

    @case
    def cat_with_a_repeated_input(b):
        x = b.input("x", (1, 1, 3, 4))
        b.output([b.permute(b.call(A.cat.default, [x, x], 1), TO_LAST)])

    @case
    def cat_inputs_with_qparams(b):
        x = b.input("x", (1, 1, 3, 4))
        y = b.call(A.relu.default, b.input("y", (1, 1, 3, 4)))
        set_qparams(y, inputs=[0], outputs=[0])
        p = b.permute(b.call(A.cat.default, [x, y], 1), TO_LAST)
        set_qparams(p, outputs=[0])
        b.output([p])

    # Forks moving down ------------------------------------------------

    @case
    def fork_into_two_unary_ops(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        b.output([b.call(A.relu.default, p), b.call(A.neg.default, p)])

    @case
    def fork_branches_rejoin(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        b.output([b.call(A.add.Tensor, b.call(A.abs.default, p), b.call(A.neg.default, p))])

    @case
    def fork_mixed_rejoin_and_diverge(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        left, right, other = (b.call(op, p) for op in (A.abs.default, A.neg.default, A.relu.default))
        b.output([b.call(A.add.Tensor, left, right), other])

    @case
    def fork_after_crossed_nodes(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, b.permute(x, TO_LAST))
        b.output([b.call(A.abs.default, r), b.call(A.neg.default, r)])

    @case
    def fork_into_slices(b):
        x = b.input("x", (1, 4, 3, 3))
        p = b.permute(x, TO_LAST)
        left = b.view(b.call(A.slice_copy.Tensor, p, 3, 0, 2), [1, 9, 2])
        right = b.view(b.call(A.slice_copy.Tensor, p, 3, 2, 4), [1, 9, 2])
        b.output([b.permute(left, [0, 2, 1]), b.permute(right, [0, 2, 1])])

    @case
    def fork_into_a_keepdim_mean(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        b.output([b.call(A.mean.dim, p, [1], True), b.call(A.relu.default, p)])

    @case
    def fork_with_a_non_elementwise_branch(b):
        x = b.input("x", (2, 3))
        w = b.input("w", (2, 2))
        p = b.permute(x, [1, 0])
        b.output([b.call(A.add.Tensor, b.call(A.relu.default, p), b.call(A.mm.default, p, w))])

    @case
    def fork_branch_with_a_layout_invariant_operand(b):
        x = b.input("x", X4)
        ones = b.input("ones", (1, 1, 1, 1))
        p = b.permute(x, TO_LAST)
        b.output([b.call(A.add.Tensor, p, ones), b.call(A.neg.default, p)])

    @case
    def fork_branch_reading_another_branch(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        a = b.call(A.abs.default, p)
        b.output([b.call(A.add.Tensor, p, a)])

    @case
    def fork_branch_changes_rank(b):
        x = b.input("x", (2, 3))
        p = b.permute(x, [1, 0])
        b.output([b.call(A.relu.default, p), b.call(A.add.Tensor, p, b.input("y", (4, 3, 2)))])

    @case
    def view_stops_at_a_fork(b):
        x = b.input("x", (2, 3, 4))
        v = b.view(x, [6, 4])
        b.output([b.call(A.relu.default, v), b.call(A.neg.default, v)])

    @case
    def fork_branch_with_qparams(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        r = b.call(A.relu.default, p)
        set_qparams(r, inputs=[0], outputs=[0])
        b.output([r, b.call(A.neg.default, p)])

    @case
    def fork_into_a_cat_with_an_untracked_input(b):
        x = b.input("x", X4)
        y = b.input("y", (1, 3, 4, 2))
        p = b.permute(x, TO_LAST)
        b.output([b.call(A.cat.default, [b.call(A.relu.default, p), b.call(A.neg.default, p), y], 3)])

    # Pointwise regions moving down -----------------------------------

    @case
    def single_exit_pointwise_region(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        b.output([b.call(A.sub.Tensor, p, b.call(A.floor.default, p))])

    @case
    def pointwise_region_with_two_exits(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        f = b.call(A.floor.default, p)
        b.output([f, b.call(A.sub.Tensor, p, f)])

    @case
    def pointwise_region_with_an_outside_operand(b):
        x = b.input("x", X4)
        y = b.input("y", (1, 3, 4, 2))
        p = b.permute(x, TO_LAST)
        f = b.call(A.floor.default, p)
        b.output([b.call(A.sub.Tensor, b.call(A.add.Tensor, p, y), f)])

    @case
    def pointwise_region_with_a_layout_invariant_operand(b):
        x = b.input("x", X4)
        ones = b.input("ones", (1, 1, 1, 1))
        p = b.permute(x, TO_LAST)
        f = b.call(A.floor.default, p)
        b.output([b.call(A.sub.Tensor, b.call(A.mul.Tensor, p, ones), f)])

    @case
    def elementwise_fork_into_a_where(b):
        x = b.input("x", X4)
        scale = b.input("scale", (1, 1, 1, 1))
        bias = b.input("bias", (1, 1, 1, 1))
        s = b.call(A.mul.Tensor, b.permute(x, TO_LAST), scale)
        cond = b.call(A.ge.Tensor, s, bias)
        fl = b.call(A.floor.default, b.call(A.add.Tensor, s, bias))
        ce = b.call(A.ceil.default, b.call(A.sub.Tensor, s, bias))
        b.output([b.call(A.where.self, cond, fl, ce)])

    @case
    def region_reaching_a_reduction(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        f = b.call(A.floor.default, p)
        b.output([b.call(A.mean.dim, b.call(A.sub.Tensor, p, f), [1], True)])

    def advance_then_region_move(b, view):
        # rc4's Down walk crosses floor and the mean, rewriting the mean's
        # dims, then fails to split the fork after it and falls back to a
        # region move from the permute, which keeps the rewritten dims: the
        # mean's users get a new shape, which rc4's next re-trace re-derives
        # (and raises on where a view no longer fits).
        x = b.input("x", (4, 2, 1, 3))
        ones = b.input("ones", (1, 1, 1, 1))
        other = b.input("other", (1, 3, 1, 1))
        p = b.permute(x, [1, 3, 2, 0])
        m = b.call(A.mean.dim, b.call(A.floor.default, p), [0, 2, -1], True)
        w = b.call(A.where.self, b.call(A.gt.Tensor, m, ones), m, other)
        c = b.call(A.ceil.default, w)
        b.output([b.view(c, [3]) if view else c])

    @case
    def advance_then_region_move_breaks_a_view(b):
        advance_then_region_move(b, view=True)

    @case
    def advance_then_region_move_reshapes_users(b):
        advance_then_region_move(b, view=False)

    @case
    def channel_split_after_clamp(b):
        # The same with a slice, the shape RewriteConvPass leaves around a
        # channel split between two convolutions: rc4's walk rewrites the
        # slice's dim from 1 to 3, the region move keeps it, and the slice
        # cuts W instead of C; both outputs change shape.
        y = b.input("y", (1, 3, 5, 8))
        s = b.call(A.slice_copy.Tensor, b.call(A.clamp.default, b.permute(y, TO_FIRST), 0.0, 6.0), 1, 0, 4)
        b.output([b.permute(s, TO_LAST), b.call(A.neg.default, s)])

    @case
    def sweep_rewiring_reorders_users_for_remove_permutes(b):
        # The Down sweeps rewire nodes, which moves them to the end of their
        # inputs' users, and the fusion round runs RemovePermutes before any
        # re-trace, so it meets the users in that order: rc4 ends with one
        # permute here, and with two if the users were in graph order.
        x = b.input("x0", (3, 3))
        p1 = b.permute(x, [1, 0])
        p2 = b.permute(p1, [1, 0])
        n4 = b.call(A.neg.default, b.view(p2, [3, 3]))
        s6 = b.call(A.sub.Tensor, b.permute(n4, [1, 0]), p2)
        p9 = b.permute(b.permute(b.permute(s6, [1, 0]), [1, 0]), [1, 0])
        m11 = b.call(A.mul.Tensor, p9, b.view(p9, [3, 3]))
        b.output([p1, b.permute(b.call(A.sub.Tensor, m11, p1), [1, 0])])

    @case
    def region_with_qparams_on_the_permute(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        set_qparams(p, inputs=[0], outputs=[0])
        b.output([b.call(A.sub.Tensor, p, b.call(A.ceil.default, p))])

    # Dead code ----------------------------------------------------------

    @case
    def dead_user_of_a_permute(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        b.call(A.relu.default, p)
        b.output([x])

    @case
    def dead_transparent_user_of_a_permute(b):
        # RemovePermutes leaves a dim-order copy alone, so Down walks into it
        # and finds no user: rc4 asserts (`:200-205`).
        x = b.input("x", X4)
        b.call(D._clone_dim_order.default, b.permute(x, TO_LAST))
        b.output([x])

    @case
    def dead_user_beside_a_live_one(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        b.call(A.relu.default, p)
        b.output([b.call(A.neg.default, p)])

    @case
    def dead_user_of_the_permutes_input(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, x)
        b.call(A.abs.default, r)
        b.output([b.permute(r, TO_LAST)])

    @case
    def dead_permute(b):
        x = b.input("x", X4)
        b.call(A.relu.default, b.permute(x, TO_LAST))
        b.output([b.call(A.neg.default, x)])

    # Fusions ------------------------------------------------------------

    @case
    def inverse_permutes_around_a_chain(b):
        x = b.input("x", X4)
        b.output([b.permute(unaries(b, b.permute(x, TO_LAST)), TO_FIRST)])

    @case
    def consecutive_permutes(b):
        x = b.input("x", (1, 2, 3))
        b.output([b.permute(b.permute(x, [0, 2, 1]), [0, 2, 1])])

    @case
    def duplicate_permutes_of_one_producer(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, x)
        b.output([b.call(A.add.Tensor, b.permute(r, TO_LAST), b.permute(r, TO_LAST))])

    @case
    def fusion_enables_further_propagation(b):
        x = b.input("x", X4)
        r = b.call(A.relu.default, x)
        n = b.call(A.neg.default, r)
        b.output([b.call(A.add.Tensor, b.permute(n, TO_LAST), b.permute(r, TO_LAST))])

    @case
    def duplicate_permutes_at_a_placeholder(b):
        x = b.input("x", (1, 4, 3, 3))
        left = b.call(A.slice_copy.Tensor, x, 1, 0, 2)
        right = b.call(A.slice_copy.Tensor, x, 1, 2, 4)
        b.output([b.permute(left, TO_LAST), b.permute(right, TO_LAST)])

    @case
    def matching_chains_across_a_slice_fanout(b):
        x = b.input("x", (1, 4, 3, 3))
        left = b.view(b.call(A.slice_copy.Tensor, x, 1, 0, 2), [1, 2, 9])
        right = b.view(b.call(A.slice_copy.Tensor, x, 1, 2, 4), [1, 2, 9])
        b.output([b.permute(left, [0, 2, 1]), b.permute(right, [0, 2, 1])])

    @case
    def singleton_permutes_fused_horizontally(b):
        x = b.input("x", (1, 1, 1, 1))
        y = b.input("y", (1, 1, 1, 1))
        b.output([b.call(A.add.Tensor, b.permute(x, TO_LAST), b.permute(y, TO_LAST))])

    @case
    def sunk_view_before_a_rank_reducing_permute(b):
        x = b.input("x", (2, 8, 1, 32))
        y = b.input("y", (2, 8, 1, 32))
        a = b.call(A.add.Tensor, b.view(x, [2, 8, 32]), b.view(y, [2, 8, 32]))
        b.output([b.permute(b.view(a, [2, 8, 32]), [0, 2, 1])])

    @case
    def mixed_permute_and_view_chain(b):
        x = b.input("x", (2, 3, 4))
        p = b.permute(x, [1, 0, 2])
        v = b.view(p, [3, 8])
        t = b.call(A.relu.default, v)
        b.output([b.permute(b.view(t, [3, 2, 4]), [1, 0, 2])])

    # TOSA: RESCALE, TABLE, INT48, SCATTER -------------------------------

    @case
    def per_tensor_rescale_is_crossed(b):
        x = b.input("x", X4, dtype=torch.int32)
        b.output([b.permute(rescale(b, x, torch.int8, [1.0]), TO_LAST)])

    @case
    def permute_sinks_through_rescales(b):
        x = b.input("x", X4, dtype=torch.int8)
        r = rescale(b, b.permute(x, TO_LAST), torch.int32)
        b.output([rescale(b, b.call(A.abs.default, r), torch.int8)])

    @case
    def per_channel_rescale_blocks(b):
        x = b.input("x", (2, 3, 4, 5), dtype=torch.int32)
        b.output([b.permute(rescale(b, x, torch.int8, [1.0] * 5), TO_FIRST)])

    @case
    def rank_changing_view_stops_at_a_rescale(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int8)
        b.output([b.view(rescale(b, b.call(A.abs.default, x), torch.int32), [6, 4])])

    @case
    def rank_preserving_view_crosses_a_rescale(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int8)
        b.output([b.view(rescale(b, x, torch.int32), [3, 2, 4])])

    @case
    def rescale_changing_the_special_dtype(b):
        x = b.input("x", (1, 1, 1, 15), dtype=torch.int32)
        set_special(x)
        b.output([b.view(rescale(b, x, torch.int32, [1.0]), [15])])

    @case
    def int48_permute_does_not_move(b):
        x = b.input("x", X4, dtype=torch.int32)
        set_special(x)
        r = b.call(A.relu.default, x)
        set_special(r)
        p = b.permute(r, TO_LAST)
        set_special(p)
        b.output([p])

    @case
    def permute_stops_before_an_int48_input(b):
        x = b.input("x", (2, 3, 5), dtype=torch.int32)
        set_special(x)
        b.output([b.permute(rescale(b, x, torch.int8, [0.25]), [1, 0, 2])])

    @case
    def int48_value_stops_a_sinking_permute(b):
        x = b.input("x", X4, dtype=torch.int32)
        r = b.call(A.abs.default, b.permute(x, TO_LAST))
        set_special(r)
        b.output([b.call(A.neg.default, r)])

    @case
    def int48_only_blocks_permutes(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int32)
        set_special(x)
        b.output([b.view(b.call(A.abs.default, x), [3, 2, 4])])

    @case
    def shared_rescale_producer_stops_up(b):
        x = b.input("x", (10, 80, 16), dtype=torch.int8)
        y = b.input("y", (10, 80, 16), dtype=torch.int32)
        r = rescale(b, x, torch.int32, [1.0])
        b.output([b.permute(r, [1, 0, 2]), b.call(A.add.Tensor, r, y)])

    @case
    def int8_table_between_permutes(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int8)
        t = int8_table(b, b.permute(x, [1, 0, 2]))
        b.output([b.permute(t, [1, 0, 2])])

    @case
    def permute_then_int8_table_chain(b):
        x = b.input("x", X4, dtype=torch.int8)
        t = int8_table(b, b.call(A.abs.default, b.permute(x, TO_LAST)))
        b.output([b.call(A.neg.default, t)])

    @case
    def int8_table_then_permute(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int8)
        b.output([b.permute(int8_table(b, b.call(A.abs.default, x)), [1, 0, 2])])

    @case
    def view_and_permute_around_a_table(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int8)
        t = int8_table(b, b.view(b.permute(x, [1, 0, 2]), [3, 8]))
        b.output([b.permute(b.view(t, [3, 2, 4]), [1, 0, 2])])

    @case
    def view_stays_before_an_unsafe_table_operand(b):
        x = b.input("x", (990, 1, 1, 64), dtype=torch.int8)
        b.output([int8_table(b, b.view(x, [99, 10, 64]))])

    @case
    def view_stays_after_an_unsafe_table_operand(b):
        x = b.input("x", (99, 10, 64), dtype=torch.int8)
        b.output([b.view(int8_table(b, x), [990, 1, 1, 64])])

    @case
    def view_sinks_through_a_safe_table(b):
        x = b.input("x", (1, 1, 1, 64), dtype=torch.int8)
        b.output([int8_table(b, b.view(x, [1, 64]))])

    @case
    def hardswish_table(b):
        x = b.input("x", X4, dtype=torch.int8)
        t = int8_table(b, b.permute(x, TO_LAST), A.hardswish.default)
        b.output([b.call(A.neg.default, t)])

    @case
    def table_in_a_pointwise_region(b):
        x = b.input("x", X4, dtype=torch.int8)
        p = b.permute(x, TO_LAST)
        b.output([b.call(A.sub.Tensor, p, int8_table(b, p))])

    @case
    def table_as_a_fork_branch(b):
        x = b.input("x", X4, dtype=torch.int8)
        p = b.permute(x, TO_LAST)
        b.output([int8_table(b, p), b.call(A.neg.default, p)])

    @case
    def table_with_a_layout_invariant_input(b):
        x = b.input("x", (1, 1, 1, 1), dtype=torch.int8)
        b.output([b.permute(int8_table(b, x), TO_LAST)])

    @case
    def int16_table_between_permutes(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int16)
        t = int16_table(b, b.permute(x, [1, 0, 2]))
        b.output([b.permute(t, [1, 0, 2])])

    @case
    def permute_sinks_through_an_int16_table(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int16)
        b.output([b.call(A.neg.default, int16_table(b, b.permute(x, [1, 0, 2])))])

    @case
    def rank_changing_view_stops_at_an_int16_table(b):
        x = b.input("x", (2, 3, 4), dtype=torch.int16)
        b.output([b.view(int16_table(b, b.call(A.abs.default, x)), [6, 4])])

    def view_into_an_int16_table(b, n):
        # Down sinks the view past the TABLE (which then reads x's [n, 8])
        # and stops it at the RESCALE, which a rank-changing view can't
        # cross; RETAB keeps the table split there (`layout::pending`).
        # Moving up, the TABLE guard keeps the view below the TABLE when it
        # would read [n, 1, 1, 8] with n > 1 (MLBEDSW-11805).
        x = b.input("x", (n, 8), dtype=torch.int16)
        b.output([int16_table(b, b.view(x, [n, 1, 1, 8]))])

    @case
    def view_parks_between_an_int16_table_and_its_rescale(b):
        view_into_an_int16_table(b, 2)

    @case
    def view_parks_before_a_safe_int16_table_rescale(b):
        view_into_an_int16_table(b, 1)

    @case
    def permute_parks_between_an_int16_table_and_its_rescale(b):
        # Moving up, the permute crosses the RESCALE but not the TABLE, which
        # would read the permute's [2, 1, 1, 8].
        x = b.input("x", (2, 8, 1, 1), dtype=torch.int16)
        b.output([b.permute(int16_table(b, x), [0, 2, 3, 1])])

    @case
    def int16_tables_behind_a_shared_permute(b):
        # The view's walk down parks it in its table, so the shared permute
        # keeps one user and its chain cancels.
        x = b.input("x1", (2, 1, 3), dtype=torch.int16)
        p = b.permute(x, [1, 2, 0])
        t = table_op(b, A.tanh.default, b.permute(p, [0, 2, 1]), qargs(1 / 256, dtype=torch.int16), qargs(1 / 4096, dtype=torch.int16))
        s1 = int16_table(b, b.permute(t, [1, 2, 0]))
        s2 = table_op(b, A.sigmoid.default, b.view(p, [1, 6]), qargs(1 / 2048, dtype=torch.int16), qargs(1 / 2048, dtype=torch.int16))
        b.output([s1, s2])

    @case
    def scatter_user_pins_a_view(b):
        x = b.input("x", (3, 16, 2, 8))
        indices = b.input("indices", (1, 4), dtype=torch.int32)
        data = b.input("data", (1, 4, 8))
        s = b.call(A.slice_copy.Tensor, x, 0, 0, 1)
        b.output([b.call(T.SCATTER.default, b.view(s, [1, 32, 8]), indices, data)])

    @case
    def scatter_input_after_a_sinking_permute(b):
        x = b.input("x", (1, 8, 32))
        indices = b.input("indices", (1, 4), dtype=torch.int32)
        data = b.input("data", (1, 4, 8))
        p = b.permute(b.call(A.relu.default, x), [0, 2, 1])
        b.output([b.call(T.SCATTER.default, b.call(A.neg.default, p), indices, data)])

    # Qparams ride along -----------------------------------------------

    @case
    def qparams_on_moved_nodes(b):
        x = b.input("x", X4)
        p = b.permute(x, TO_LAST)
        set_qparams(p, inputs=[0], outputs=[0])
        r = b.call(A.relu.default, p)
        set_qparams(r, inputs=[], outputs=[0])
        b.output([r])

    @case
    def constants_feed_the_chain(b):
        x = b.input("x", X4)
        w = b.constant("param", "w", (1, 3, 4, 2))
        b.output([b.call(A.add.Tensor, b.permute(x, TO_LAST), w)])

    return cases


# Steps per random propagation graph; a fuzzing session can ask for larger
# graphs with RETAB_REPLAY_PROPAGATION_STEPS=low,high (the committed fixtures
# use the default).
PROPAGATION_STEPS = tuple(
    int(bound) for bound in os.environ.get("RETAB_REPLAY_PROPAGATION_STEPS", "2,10").split(",")
)


def propagation_random_case(seed: int) -> Callable[[Builder], None]:
    def build(b: Builder) -> None:
        rng = random.Random(seed)
        graph = PropagationGraph(rng, b)
        for _ in range(rng.randint(1, 3)):
            graph.fresh_float()
        for _ in range(rng.randint(*PROPAGATION_STEPS)):
            graph.step()
        graph.finish()

    return build


PROPAGATION_UNARY = [
    A.relu.default,
    A.abs.default,
    A.neg.default,
    A.floor.default,
    A.ceil.default,
    A.sqrt.default,
]
PROPAGATION_BINARY = [A.add.Tensor, A.sub.Tensor, A.mul.Tensor, A.maximum.default]
# 16-bit table ops and (input, output) qparams the random graphs draw: most
# pairs need a left shift, so InsertTableOpsPass adds a RESCALE.
TABLE16_TARGETS = [(A.sigmoid.default, (), {}), (A.tanh.default, (), {}), (A.exp.default, (), {})]
TABLE16_QPARAMS = [
    (qargs(1 / 1024, dtype=torch.int16), qargs(1 / 32768, dtype=torch.int16)),
    (qargs(1 / 256, dtype=torch.int16), qargs(1 / 4096, dtype=torch.int16)),
    (qargs(1 / 2048, dtype=torch.int16), qargs(1 / 2048, dtype=torch.int16)),
]


class PropagationGraph:
    """Seeded random graphs for the propagation passes: chains of pointwise
    ops, keepdim means, slices, cats and forks with permutes and views among
    them, int8 and int16 RESCALE and TABLE islands (a 16-bit TABLE is followed
    by a RESCALE), INT48 data and SCATTER users."""

    WEIGHTS = {
        "permute": 3,
        "view": 2,
        "unary": 3,
        "transparent": 0.8,
        "binary": 2,
        "where": 0.5,
        "mean": 1,
        "sum": 0.3,
        "slice": 1.2,
        "cat": 0.8,
        "fork": 1.2,
        "region": 0.6,
        "chain": 1.5,
        "pre_chain": 1,
        "same_transforms": 0.8,
        "int": 0.8,
        "table": 0.7,
        "table16": 0.7,
        "scatter": 0.15,
        "mm": 0.25,
        "dead": 0.12,
        "qparams": 0.5,
        "slice_view": 0.8,
        "view_slice": 0.8,
        "mean_view": 0.6,
        "view_mean": 0.6,
        "permute_mean": 0.6,
    }

    def __init__(self, rng: random.Random, b: Builder) -> None:
        self.rng = rng
        self.b = b
        self.pool: list[torch.fx.Node] = []
        self.inputs = 0

    def random_slice(self, x):
        rng = self.rng
        shape = list(x.meta["val"].shape)
        dim = rng.randrange(len(shape))
        size = shape[dim]
        if rng.random() < 0.45:
            start = rng.randrange(size)
            end = start + 1
        else:
            start = rng.randrange(size)
            end = rng.randint(start + 1, size)
        return self.b.call(A.slice_copy.Tensor, x, dim - len(shape) if rng.random() < 0.15 else dim, start, end)

    def random_keepdim_mean(self, x):
        rank = x.meta["val"].dim()
        dims = sorted(self.rng.sample(range(rank), self.rng.randint(1, rank)))
        if self.rng.random() < 0.2:
            dims = [d - rank for d in dims]
        return self.b.call(A.mean.dim, x, dims, True)

    def shape(self, rank: int, max_numel: int = 180) -> list[int]:
        while True:
            dims = [1 if self.rng.random() < 0.3 else self.rng.randint(2, 4) for _ in range(rank)]
            if math.prod(dims) <= max_numel:
                return dims

    def permutation(self, rank: int) -> list[int]:
        dims = list(range(rank))
        self.rng.shuffle(dims)
        if self.rng.random() < 0.2:
            dims = [d - rank if self.rng.random() < 0.5 else d for d in dims]
        return dims

    def view_shape(self, shape) -> list[int]:
        rng = self.rng
        factors = prime_factors(math.prod(shape))
        rng.shuffle(factors)
        buckets = rng.randint(1, min(4, max(1, len(factors)))) if factors else 0
        dims = [1] * buckets
        for f in factors:
            dims[rng.randrange(buckets)] *= f
        while len(dims) < 5 and rng.random() < 0.25:
            dims.insert(rng.randint(0, len(dims)), 1)
        if not dims and rng.random() < 0.5:
            dims = [1] * rng.randint(1, 2)
        if dims and rng.random() < 0.1:
            dims[rng.randrange(len(dims))] = -1
        return dims

    def new_input(self, shape, dtype=torch.float32) -> torch.fx.Node:
        name = f"x{self.inputs}"
        self.inputs += 1
        node = self.b.input(name, shape, dtype)
        if self.rng.random() < 0.1:
            set_qparams(node, inputs=self.rng.choice([None, [0]]), outputs=[0])
        return node

    def fresh_float(self, rank=None) -> torch.fx.Node:
        rank = self.rng.randint(0, 5) if rank is None else rank
        node = self.new_input(self.shape(rank))
        self.pool.append(node)
        return node

    def pick(self, dtype=torch.float32) -> torch.fx.Node:
        candidates = [n for n in self.pool if n.meta["val"].dtype == dtype]
        if not candidates or self.rng.random() < 0.12:
            node = self.new_input(self.shape(self.rng.randint(0, 5)), dtype)
            self.pool.append(node)
            return node
        return candidates[-1 - min(int(self.rng.expovariate(0.8)), len(candidates) - 1)]

    def keep(self, node):
        self.pool.append(node)
        return node

    def partner(self, x) -> torch.fx.Node:
        """A second operand for `x`: same shape, all-ones, broadcast or lower rank."""
        rng = self.rng
        shape = list(x.meta["val"].shape)
        roll = rng.random()
        if roll < 0.35:
            other = shape
        elif roll < 0.6:
            other = [1] * len(shape)
        elif roll < 0.8:
            other = [1 if rng.random() < 0.5 else d for d in shape]
        else:
            keep = rng.randint(0, len(shape))
            other = [1 if rng.random() < 0.4 else d for d in shape[len(shape) - keep :]]
        return self.keep(self.new_input(other, x.meta["val"].dtype))

    def unary(self, x):
        return self.b.call(self.rng.choice(PROPAGATION_UNARY), x)

    def transform(self, x):
        shape = list(x.meta["val"].shape)
        if self.rng.random() < 0.6:
            return self.b.permute(x, self.permutation(len(shape)))
        return self.b.view(x, self.view_shape(shape))

    def maybe_qparams(self, node):
        # Never on a table op: with both maps it would be a pending table.
        if self.rng.random() < 0.08 and node.target not in set(TableOps.included_ops()):
            set_qparams(node, inputs=self.rng.choice([[], [0]]), outputs=[0])
        return node

    def step(self) -> None:
        rng, b = self.rng, self.b
        kind = rng.choices(list(self.WEIGHTS), list(self.WEIGHTS.values()))[0]
        if kind == "permute":
            x = self.pick()
            self.keep(self.maybe_qparams(b.permute(x, self.permutation(x.meta["val"].dim()))))
        elif kind == "view":
            x = self.pick()
            self.keep(self.maybe_qparams(b.view(x, self.view_shape(list(x.meta["val"].shape)))))
        elif kind == "unary":
            self.keep(self.maybe_qparams(self.unary(self.pick())))
        elif kind == "transparent":
            x = self.pick()
            if rng.random() < 0.5:
                self.keep(b.call(D._clone_dim_order.default, x))
            else:
                self.keep(b.call(D._to_dim_order_copy.default, x, dtype=torch.float32))
        elif kind == "binary":
            x = self.pick()
            y = self.partner(x)
            args = (x, y) if rng.random() < 0.6 else (y, x)
            self.keep(self.maybe_qparams(b.call(rng.choice(PROPAGATION_BINARY), *args)))
        elif kind == "where":
            x = self.pick()
            y = self.partner(x)
            cond = b.call(A.gt.Tensor, x, y)
            self.keep(b.call(A.where.self, cond, x, self.partner(x)))
        elif kind in ("mean", "sum"):
            x = self.pick()
            rank = x.meta["val"].dim()
            if rank == 0:
                return
            dims = sorted(rng.sample(range(rank), rng.randint(1, rank)))
            dims = [d - rank if rng.random() < 0.2 else d for d in dims]
            keepdim = rng.random() >= 0.03
            target = A.mean.dim if kind == "mean" else A.sum.dim_IntList
            if rng.random() < 0.15 and len(dims) == 1:
                dims = dims[0]
            self.keep(b.call(target, x, dims, keepdim))
        elif kind == "slice":
            x = self.pick()
            shape = list(x.meta["val"].shape)
            if not shape:
                return
            dim = rng.randrange(len(shape))
            size = shape[dim]
            if rng.random() < 0.4:
                start = rng.randrange(size)
                end = start + 1
            else:
                start = rng.randrange(size)
                end = rng.randint(start + 1, size)
            args = [x, dim - len(shape) if rng.random() < 0.15 else dim, start, end]
            if rng.random() < 0.1:
                args.append(rng.choice([1, 2]))
            self.keep(b.call(A.slice_copy.Tensor, *args))
        elif kind == "cat":
            x = self.pick()
            shape = list(x.meta["val"].shape)
            if not shape:
                return
            dim = rng.randrange(len(shape))
            other_shape = list(shape)
            other_shape[dim] = rng.randint(1, 3)
            parts = [x, self.keep(self.new_input(other_shape))]
            if rng.random() < 0.3:
                parts.append(self.keep(self.new_input(other_shape)))
            rng.shuffle(parts)
            c = self.keep(b.call(A.cat.default, parts, dim))
            if rng.random() < 0.5:
                self.keep(b.permute(c, self.permutation(len(shape))))
        elif kind == "fork":
            x = self.pick()
            source = self.transform(x) if rng.random() < 0.6 else x
            for _ in range(rng.randint(2, 3)):
                branch = rng.random()
                shape = list(source.meta["val"].shape)
                if branch < 0.55 or not shape:
                    self.keep(self.unary(source))
                elif branch < 0.75:
                    dim = rng.randrange(len(shape))
                    self.keep(b.call(A.slice_copy.Tensor, source, dim, 0, max(1, shape[dim] - 1)))
                elif branch < 0.9:
                    self.keep(b.call(A.mean.dim, source, [rng.randrange(len(shape))], True))
                else:
                    self.keep(b.call(A.add.Tensor, source, self.partner(source)))
        elif kind == "region":
            x = self.pick()
            p = b.permute(x, self.permutation(x.meta["val"].dim()))
            f = b.call(rng.choice([A.floor.default, A.ceil.default, A.abs.default]), p)
            node = b.call(A.sub.Tensor, p, f)
            if rng.random() < 0.3:
                self.keep(f)
            self.keep(node)
        elif kind == "chain":
            x = self.pick()
            rank = x.meta["val"].dim()
            perm = self.permutation(rank)
            value = b.permute(x, perm)
            for _ in range(rng.randint(1, 3)):
                value = self.unary(value)
            if rng.random() < 0.6:
                inverse = [0] * rank
                for i, d in enumerate(perm):
                    inverse[d % rank] = i
                value = b.permute(value, inverse)
            else:
                value = self.transform(value)
            self.keep(value)
        elif kind == "pre_chain":
            value = self.pick()
            for _ in range(rng.randint(1, 3)):
                value = self.unary(value)
            self.keep(self.transform(value))
        elif kind == "same_transforms":
            x = self.pick()
            shape = list(x.meta["val"].shape)
            y = self.keep(self.new_input(shape))
            if rng.random() < 0.5:
                perm = self.permutation(len(shape))
                tx, ty = b.permute(x, perm), b.permute(y, perm)
            else:
                view = self.view_shape(shape)
                tx, ty = b.view(x, view), b.view(y, view)
            self.keep(b.call(rng.choice(PROPAGATION_BINARY), tx, ty))
        elif kind == "int":
            narrow = rng.choice([torch.int8, torch.int16])
            x = self.pick(torch.int8) if narrow == torch.int8 and rng.random() < 0.5 else self.new_input(self.shape(rng.randint(1, 4)), narrow)
            value = x
            for _ in range(rng.randint(2, 4)):
                step = rng.choices(["rescale", "permute", "view", "abs"], [3, 3, 1, 1])[0]
                dtype = value.meta["val"].dtype
                if step == "rescale":
                    out = torch.int32 if dtype != torch.int32 else rng.choice([torch.int8, torch.int16])
                    scales = [0.5] if rng.random() < 0.85 else [0.5] * max(1, value.meta["val"].shape[-1] if value.meta["val"].dim() else 1)
                    value = rescale(b, value, out, scales)
                elif step == "permute":
                    value = b.permute(value, self.permutation(value.meta["val"].dim()))
                elif step == "view":
                    value = b.view(value, self.view_shape(list(value.meta["val"].shape)))
                else:
                    value = b.call(A.abs.default, value)
                if rng.random() < 0.15:
                    set_special(value)
            self.keep(value)
        elif kind == "table":
            x = self.pick(torch.int8)
            if rng.random() < 0.5:
                x = b.permute(x, self.permutation(x.meta["val"].dim()))
            target, args, kwargs = rng.choice(TABLE_TARGETS)
            t = table_op(b, target, x, rng.choice(TABLE_QPARAMS), rng.choice(TABLE_QPARAMS), *args, **kwargs)
            self.keep(t)
            if rng.random() < 0.5:
                self.keep(self.transform(t))
        elif kind == "table16":
            x = self.pick(torch.int16)
            roll = rng.random()
            if roll < 0.35:
                x = b.permute(x, self.permutation(x.meta["val"].dim()))
            elif roll < 0.7:
                x = b.view(x, self.view_shape(list(x.meta["val"].shape)))
            target, args, kwargs = rng.choice(TABLE16_TARGETS)
            qin, qout = rng.choice(TABLE16_QPARAMS)
            t = table_op(b, target, x, qin, qout, *args, **kwargs)
            self.keep(t)
            if rng.random() < 0.5:
                self.keep(self.transform(t))
        elif kind == "scatter":
            x = self.pick()
            shape = list(x.meta["val"].shape)
            if len(shape) != 3:
                return
            n, _, c = shape
            indices = self.new_input([n, 2], torch.int32)
            data = self.new_input([n, 2, c])
            self.keep(b.call(T.SCATTER.default, x, indices, data))
        elif kind == "mm":
            x = self.pick()
            shape = list(x.meta["val"].shape)
            if len(shape) != 2:
                return
            w = self.new_input([shape[1], rng.randint(1, 3)])
            self.keep(b.call(A.mm.default, x, w))
        elif kind == "dead":
            self.unary(self.pick())
        elif kind in ("slice_view", "view_slice", "mean_view", "view_mean", "permute_mean"):
            x = self.pick()
            if x.meta["val"].dim() == 0:
                return
            if kind == "slice_view":
                s = self.random_slice(x)
                self.keep(b.view(s, self.view_shape(list(s.meta["val"].shape))))
            elif kind == "view_slice":
                v = b.view(x, self.view_shape(list(x.meta["val"].shape)))
                if v.meta["val"].dim():
                    self.keep(self.random_slice(v))
            elif kind == "mean_view":
                m = self.random_keepdim_mean(x)
                self.keep(b.view(m, self.view_shape(list(m.meta["val"].shape))))
            elif kind == "view_mean":
                v = b.view(x, self.view_shape(list(x.meta["val"].shape)))
                if v.meta["val"].dim():
                    self.keep(self.random_keepdim_mean(v))
            else:
                rank = x.meta["val"].dim()
                if rng.random() < 0.5:
                    self.keep(self.random_keepdim_mean(b.permute(x, self.permutation(rank))))
                else:
                    self.keep(b.permute(self.random_keepdim_mean(x), self.permutation(rank)))
        elif kind == "qparams":
            computed = [n for n in self.pool if n.op == "call_function" and not _is_table_op(n)]
            if computed:
                node = rng.choice(computed)
                if node.target not in set(TableOps.included_ops()):
                    set_qparams(node, inputs=[0], outputs=[0])
        else:
            raise ValueError(kind)

    def finish(self) -> None:
        rng = self.rng
        computed = [n for n in self.pool if n.op == "call_function"]
        outputs = [n for n in computed if not n.users] or [rng.choice(self.pool)]
        rng.shuffle(outputs)
        if computed and rng.random() < 0.2:
            outputs.append(rng.choice(computed))
        self.b.output(outputs)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

PROPAGATION_SOURCES = [
    "backends/transforms/propagate_view_copy_permute_pass.py",
    "backends/arm/_passes/propagate_view_copy_permute_pass.py",
    "backends/transforms/dim_maps.py",
    "backends/transforms/permute_view_meta.py",
    "backends/transforms/canonicalize_view_copy_permute_pass.py",
    "backends/transforms/fuse_duplicate_users_pass.py",
    "backends/transforms/fuse_identical_input_transforms_pass.py",
    "backends/transforms/remove_permutes_around_elementwise_ops.py",
    "backends/arm/_passes/remove_permutes_around_elementwise_tosa_ops.py",
    "backends/arm/_passes/insert_table_ops.py",
]

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
        "fuse_duplicate_users",
        lambda ep: FuseDuplicateUsersPass(),
        [
            "backends/transforms/fuse_duplicate_users_pass.py",
            "backends/arm/_passes/fuse_duplicate_users_pass.py",
        ],
        fuse_duplicate_users_cases,
        "fuse_duplicate_users",
        random_cases=360,
    ),
    # The propagation driver's fuse_horizontal at phase 6
    # (propagate_view_copy_permute_pass.py:569-576, with the TOSA exclusions of
    # backends/arm/_passes/propagate_view_copy_permute_pass.py:37-38), after
    # InsertTableOpsPass (arm_pass_manager.py:708).
    PassSpec(
        "fuse_duplicate_users_driver",
        lambda ep: TableStagePass(
            ep, lambda: SharedFuseDuplicateUsersPass(TOSA_EXCLUDED_TARGETS), merge_constants=False
        ),
        [
            "backends/transforms/fuse_duplicate_users_pass.py",
            "backends/arm/_passes/fuse_duplicate_users_pass.py",
            "backends/arm/_passes/insert_table_ops.py",
            "backends/transforms/propagate_view_copy_permute_pass.py",
            "backends/arm/_passes/propagate_view_copy_permute_pass.py",
        ],
        table_stage_cases,
        "table_stage",
        random_cases=360,
        tosa_spec="TOSA-1.0+FP+INT+int16",
    ),
    # FuseDuplicateUsersPass() at arm_pass_manager.py:765, after
    # FuseEqualPlaceholdersPass (`:755`).
    PassSpec(
        "fuse_duplicate_users_late",
        lambda ep: TableStagePass(ep, FuseDuplicateUsersPass, merge_constants=True),
        [
            "backends/transforms/fuse_duplicate_users_pass.py",
            "backends/arm/_passes/fuse_duplicate_users_pass.py",
            "backends/arm/_passes/insert_table_ops.py",
            "backends/arm/_passes/fuse_equal_placeholders_pass.py",
        ],
        table_stage_cases,
        "table_stage",
        random_cases=360,
        tosa_spec="TOSA-1.0+FP+INT+int16",
    ),
    PassSpec(
        "fuse_identical_input_transforms",
        lambda ep: FuseIdenticalInputTransformsPass(ep),
        [
            "backends/transforms/fuse_identical_input_transforms_pass.py",
            "backends/arm/_passes/fuse_identical_input_transforms_pass.py",
            "backends/transforms/permute_view_meta.py",
            "backends/transforms/dim_maps.py",
        ],
        fuse_identical_input_transforms_cases,
        "fuse_identical_input_transforms",
        random_cases=400,
    ),
    # The propagation driver's construction (propagate_view_copy_permute_pass.py:
    # 921-928): the shared pass, Arm's permute targets, no exported program and
    # no ArmOpTargetedPass gate. Random graphs never hold transform-named
    # placeholders, the only thing the two constructions disagree on.
    PassSpec(
        "fuse_identical_input_transforms_driver",
        lambda ep: SharedFuseIdenticalInputTransformsPass(
            permute_targets=frozenset({A.permute_copy.default})
        ),
        [
            "backends/transforms/fuse_identical_input_transforms_pass.py",
            "backends/transforms/propagate_view_copy_permute_pass.py",
            "backends/transforms/permute_view_meta.py",
            "backends/transforms/dim_maps.py",
        ],
        fuse_identical_input_transforms_cases,
        "fuse_identical_input_transforms",
        random_cases=0,
    ),
    PassSpec(
        "move_data_movement_ops_to_smaller_dtype",
        lambda ep: MoveDataMovementOpsToSmallerDtypePass(),
        [
            "backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py",
            "backends/transforms/permute_view_meta.py",
            "backends/arm/tosa/dialect/ops/conversion_ops.py",
        ],
        move_data_movement_cases,
        "move_data_movement",
        random_cases=360,
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
    PassSpec(
        "remove_permutes_around_elementwise",
        # `TosaPropagationOverrides.make_fusion_pass`
        # (backends/arm/_passes/propagate_view_copy_permute_pass.py:90-93).
        lambda ep: RemovePermutesAroundElementwiseTosaOps(ep),
        [
            "backends/transforms/remove_permutes_around_elementwise_ops.py",
            "backends/arm/_passes/remove_permutes_around_elementwise_tosa_ops.py",
            "backends/transforms/permute_pass_utils.py",
            "backends/transforms/channels_last_layout.py",
            "backends/arm/_passes/insert_table_ops.py",
            "backends/arm/_passes/propagate_view_copy_permute_pass.py",
        ],
        remove_permutes_cases,
        "remove_permutes",
    ),
    PassSpec(
        "remove_permutes_user_order",
        lambda ep: RemovePermutesAroundElementwiseTosaOps(ep),
        [
            "backends/transforms/remove_permutes_around_elementwise_ops.py",
            "backends/arm/_passes/remove_permutes_around_elementwise_tosa_ops.py",
            "backends/transforms/permute_pass_utils.py",
            "backends/transforms/channels_last_layout.py",
            "backends/arm/_passes/insert_table_ops.py",
            "backends/arm/_passes/propagate_view_copy_permute_pass.py",
        ],
        remove_permutes_user_order_cases,
        "remove_permutes_user_order",
        # About one random graph in ten depends on its user order.
        random_cases=2 * RANDOM_CASES,
    ),
    # `PropagateViewCopyPermuteDownPass(self.compile_spec, exported_program)`
    # and the Up pass (arm_pass_manager.py:726-727), after InsertTableOpsPass
    # (`:708`); see `PropagationPass`.
    PassSpec(
        "propagate_view_copy_permute_down",
        lambda ep: PropagationPass(ep, PropagateViewCopyPermuteDownPass),
        PROPAGATION_SOURCES,
        propagation_cases,
        "propagate",
        random_cases=420,
        tosa_spec=PROPAGATE_TOSA_SPEC,
    ),
    PassSpec(
        "propagate_view_copy_permute_up",
        lambda ep: PropagationPass(ep, PropagateViewCopyPermuteUpPass),
        PROPAGATION_SOURCES,
        propagation_cases,
        "propagate",
        random_cases=420,
        tosa_spec=PROPAGATE_TOSA_SPEC,
    ),
    # The Down pass, then the Up pass, as phase 6 runs them: the Up pass
    # meets what Down left, such as a view parked between a 16-bit TABLE and
    # its RESCALE.
    PassSpec(
        "propagate_view_copy_permute_down_up",
        lambda ep: PropagationPass(ep, PropagateViewCopyPermuteDownPass, PropagateViewCopyPermuteUpPass),
        PROPAGATION_SOURCES,
        propagation_cases,
        "propagate",
        random_cases=420,
        tosa_spec=PROPAGATE_TOSA_SPEC,
    ),
    # The same passes built without arguments, as rc4's graph-level tests
    # build them: no fusion pass, so the walks meet every permute
    # RemovePermutes would have cancelled first.
    PassSpec(
        "propagate_view_copy_permute_down_without_program",
        lambda ep: PropagationPass(ep, PropagateViewCopyPermuteDownPass, with_program=False),
        PROPAGATION_SOURCES,
        propagation_cases,
        "propagate",
        random_cases=420,
        tosa_spec=PROPAGATE_TOSA_SPEC,
    ),
    PassSpec(
        "propagate_view_copy_permute_up_without_program",
        lambda ep: PropagationPass(ep, PropagateViewCopyPermuteUpPass, with_program=False),
        PROPAGATION_SOURCES,
        propagation_cases,
        "propagate",
        random_cases=420,
        tosa_spec=PROPAGATE_TOSA_SPEC,
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
    parser.add_argument(
        "--random", type=int, help=f"random graphs per pass (default {RANDOM_CASES} or the pass's own)"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--case", help="print one case of --pass instead of writing")
    args = parser.parse_args()

    for spec in PASSES:
        if args.only and spec.name not in args.only:
            continue
        made = spec.make(None)
        rc4_pass = getattr(made, "rc4_name", type(made).__name__)
        if args.case:
            cases = dict(spec.structured())
            if args.case.startswith("random_"):
                index = int(args.case.removeprefix("random_"))
                cases[args.case] = random_case(spec.random_profile, args.seed * 100_000 + index)
            print("\n".join(run_case(spec, cases[args.case])))
            continue
        random_cases = args.random if args.random is not None else spec.random_cases
        lines, skipped = generate(spec, args.seed, random_cases)
        head = header(spec, args.seed, random_cases, skipped, rc4_pass)
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
