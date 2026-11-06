"""Write rc4 CanonicalizeViewCopyPermutePass results on seeded graphs for
src/layout/canonicalize_rc4_tests.rs.

Regenerate with a Python that has ExecuTorch v1.6.0-rc4 installed, from a neutral
directory so a source checkout can't shadow the installed package:

    cd /tmp && python -P <retab>/tests/fixtures/canonicalize_rc4_cases.py \
        > <retab>/tests/fixtures/canonicalize_rc4_cases.txt

Each case is an edge graph of placeholders, view_copy, permute_copy and relu
nodes. The script runs Arm's configuration of the pass
(backends/arm/_passes/canonicalize_view_copy_permute_pass.py, retrace included)
and records the live call nodes it leaves, in graph order. rc4's retrace renames
call nodes after their targets, so every node carries a `label` meta key, which
the retrace copies with the rest of the meta: a label in the result names the
input node that the result node is, as the Rust port's names do. Each result is
also run on random data against the original graph, and the script stops if rc4
ever changes what the graph computes.

The cases are the static graphs of backends/arm/test/passes/
test_canonicalize_view_copy_permute_pass.py, chains built around the swap
vectors of test_view_permute_swap.py and test_permute_view_swap.py, and random
graphs and long linear chains from `random.Random(SEED)` (stable for a given
CPython `random`; written with 3.12).

Format, one token list per line:

    case N
    input LABEL SHAPE DTYPE
    op LABEL KIND INPUT ARGS        KIND is view, permute or relu
    output LABEL...
    expect modified|unmodified
    node LABEL KIND INPUT ARGS SHAPE DTYPE   the live call nodes after the pass
    output LABEL...                    the output node's args after the pass
    end

SHAPE and ARGS are comma-separated ints; `-` is an empty list and `_` no
argument (relu). An empty `op` list is what RETAB's FX import turns into
IrArg::None.
"""

import copy
import hashlib
import random
import sys

import executorch.backends.arm._passes.canonicalize_view_copy_permute_pass as arm_pass
import executorch.backends.transforms.canonicalize_view_copy_permute_pass as base_pass
import executorch.backends.transforms.dim_maps as dim_maps
import executorch.backends.transforms.permute_view_meta as permute_view_meta
import torch
import torch.fx as fx
from executorch.backends.arm._passes import CanonicalizeViewCopyPermutePass
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode

SEED = 154
RANDOM_CASES = 500
LONG_CHAIN_CASES = 80
MAX_RANK = 6
DIMS = [1, 1, 1, 2, 2, 3, 4, 5, 6]
RANKS = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]

VIEW = exir_ops.edge.aten.view_copy.default
PERMUTE = exir_ops.edge.aten.permute_copy.default
RELU = exir_ops.edge.aten.relu.default
TARGETS = {"view": VIEW, "permute": PERMUTE, "relu": RELU}
KINDS = {target: kind for kind, target in TARGETS.items()}
DTYPES = {"f32": torch.float32, "i8": torch.int8}
DTYPE_NAMES = {dtype: name for name, dtype in DTYPES.items()}

rng = random.Random(SEED)


def numel(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def prime_factors(n):
    factors, divisor = [], 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    if n > 1:
        factors.append(n)
    return factors


def ints(values):
    return ",".join(str(value) for value in values) if values else "-"


class Case:
    def __init__(self):
        self.inputs = []  # (label, shape, dtype)
        self.ops = []  # (label, kind, input, args)
        self.outputs = []
        self.shapes = {}

    def input(self, shape, dtype="f32"):
        label = f"x{len(self.inputs)}"
        self.inputs.append((label, list(shape), dtype))
        self.shapes[label] = list(shape)
        return label

    def op(self, kind, source, args=None):
        label = f"n{len(self.ops)}"
        shape = self.shapes[source]
        if kind == "view":
            args = list(args)
            out = list(args)
            if -1 in out:
                known = numel(dim for dim in out if dim != -1)
                out[out.index(-1)] = numel(shape) // known
            assert numel(out) == numel(shape), (shape, args)
        elif kind == "permute":
            args = list(args)
            out = [shape[dim] for dim in args]
        else:
            out = list(shape)
        self.ops.append((label, kind, source, args))
        self.shapes[label] = out
        return label


def build(case):
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    graph = fx.Graph()
    nodes = {}
    for label, shape, dtype in case.inputs:
        node = graph.placeholder(label)
        node.meta["val"] = mode.from_tensor(torch.empty(shape, dtype=DTYPES[dtype]))
        node.meta["label"] = label
        nodes[label] = node
    for label, kind, source, args in case.ops:
        call_args = (nodes[source],) if args is None else (nodes[source], list(args))
        node = graph.call_function(TARGETS[kind], call_args)
        with mode:
            node.meta["val"] = TARGETS[kind](
                *[arg.meta["val"] if isinstance(arg, fx.Node) else arg for arg in call_args]
            )
        node.meta["label"] = label
        nodes[label] = node
    graph.output(tuple(nodes[label] for label in case.outputs))
    return fx.GraphModule(torch.nn.Module(), graph)


def random_inputs(case):
    tensors = []
    for _, shape, dtype in case.inputs:
        if dtype == "i8":
            tensors.append(torch.randint(-100, 100, shape, dtype=torch.int8))
        else:
            tensors.append(torch.randn(shape))
    return tensors


def run_case(index, case):
    graph_module = build(case)
    original = copy.deepcopy(graph_module)
    result = CanonicalizeViewCopyPermutePass().call(graph_module)
    inputs = random_inputs(case)
    expected = original(*inputs)
    actual = result.graph_module(*inputs)
    if len(expected) != len(actual) or not all(
        torch.equal(lhs, rhs) for lhs, rhs in zip(expected, actual)
    ):
        raise SystemExit(f"case {index}: rc4 changed what the graph computes")

    lines = [f"case {index}"]
    lines += [f"input {label} {ints(shape)} {dtype}" for label, shape, dtype in case.inputs]
    for label, kind, source, args in case.ops:
        lines.append(f"op {label} {kind} {source} {'_' if args is None else ints(args)}")
    lines.append("output " + " ".join(case.outputs))
    lines.append("expect " + ("modified" if result.modified else "unmodified"))
    for node in result.graph_module.graph.nodes:
        if node.op == "call_function":
            kind = KINDS[node.target]
            args = "_" if kind == "relu" else ints(list(node.args[1]))
            source = node.args[0].meta["label"]
            shape = ints(list(node.meta["val"].shape))
            dtype = DTYPE_NAMES[node.meta["val"].dtype]
            lines.append(f"node {node.meta['label']} {kind} {source} {args} {shape} {dtype}")
        elif node.op == "output":
            lines.append("output " + " ".join(arg.meta["label"] for arg in node.args[0]))
    lines.append("end")
    return lines


# --- Upstream graphs -------------------------------------------------------


def upstream_canonicalize_cases():
    """The static graphs of test_canonicalize_view_copy_permute_pass.py."""
    cases = []

    def chain(shape, steps, extra=None):
        case = Case()
        current = case.input(shape)
        for kind, args in steps:
            current = case.op(kind, current, args)
        case.outputs.append(current)
        if extra:
            extra(case)
        cases.append(case)
        return case

    chain([2, 3, 4, 5], [("permute", [0, 2, 3, 1]), ("permute", [0, 3, 1, 2])])
    chain([1, 1, 1, 4], [("permute", [0, 3, 1, 2])])
    chain([2, 1, 2], [("permute", [2, 1, 0])])
    chain(
        [1, 2, 2, 8],
        [
            ("permute", [0, 3, 1, 2]),
            ("view", [1, 2, 2, 2, 2, 2]),
            ("permute", [0, 1, 4, 2, 5, 3]),
            ("view", [1, 2, 4, 4]),
            ("permute", [0, 2, 3, 1]),
        ],
    )
    chain([2, 12, 5], [("view", [2, 3, 4, 5]), ("permute", [2, 0, 1, 3]), ("view", [4, 6, 5])])
    chain([2, 3, 4], [("view", [3, 2, 4]), ("permute", [2, 0, 1]), ("view", [4, 6])])
    chain([2, 3, 4], [("permute", [2, 0, 1]), ("view", [4, 6]), ("permute", [1, 0])])
    chain(
        [1, 2, 10, 10],
        [
            ("view", [1, 2, 5, 2, 5, 2]),
            ("permute", [3, 5, 0, 1, 2, 4]),
            ("view", [4, 2, 5, 5]),
            ("permute", [0, 2, 3, 1]),
        ],
    )
    chain([2, 8, 64], [("view", [2, 8, 1, 64]), ("permute", [0, 2, 3, 1])])

    # test_canonicalize_follows_interleaved_chain_users
    case = Case()
    x, y = case.input([4, 2, 4]), case.input([2, 3])
    p1 = case.op("permute", x, [1, 0, 2])
    unrelated = case.op("view", y, [3, 2])
    v1 = case.op("view", p1, [1, 2, 4, 4])
    p2 = case.op("permute", v1, [0, 2, 1, 3])
    case.outputs += [p2, unrelated]
    cases.append(case)

    chain([2, 3, 4], [("view", [3, 2, 4]), ("permute", [0, 2, 1])])

    # test_canonicalize_does_not_cross_multi_user_chain_node
    case = Case()
    x = case.input([2, 3, 4])
    p1 = case.op("permute", x, [0, 2, 1])
    p2 = case.op("permute", p1, [0, 2, 1])
    v1 = case.op("view", p1, [2, 12])
    case.outputs += [p2, v1]
    cases.append(case)

    chain([2, 3], [("view", [3, 2]), ("permute", [1, 0]), ("permute", [1, 0])])
    chain([2, 3], [("permute", [1, 0]), ("permute", [1, 0]), ("view", [3, 2])])
    # The _view_permute_swap vectors as lone pairs (the first is a sink).
    chain([2, 3, 4, 5, 6, 7, 8, 9], [("view", [1, 2, 3, 4, 5, 6, 7, 8, 9]), ("permute", [0, 8, 7, 6, 5, 4, 3, 2, 1])])
    chain([4], [("view", [2, 2]), ("permute", [1, 0])])
    return cases


VIEW_PERMUTE_VECTORS = [
    ([2, 3, 4], [2, 3, 2, 2], [0, 2, 3, 1]),
    ([2, 3, 4], [2, 3, 2, 2], [2, 3, 0, 1]),
    ([2, 3, 4], [2, 12], [1, 0]),
    ([2, 3, 4], [6, 4], [1, 0]),
    ([2, 3, 4], [6, 4], [-1, 0]),
    ([2, 3, 4], [1, 6, 4], [0, 2, 1]),
    ([2, 3, 4], [1, 6, 4], [2, 0, 1]),
    ([2, 3, 4], [1, 6, 4], [2, 1, 0]),
    ([2, 3, 4], [2, 3, 2, 2], [1, 2, 3, 0]),
    ([2, 3, 4, 5], [2, 3, 20], [0, 2, 1]),
    ([2, 3, 4, 5], [2, 3, 20], [2, 0, 1]),
    ([2, 3, 4, 5], [6, 4, 5], [1, 2, 0]),
    ([2, 3, 4, 5], [24, 5], [1, 0]),
    ([1, 2, 3, 4], [2, 12], [1, 0]),
    ([2, 1, 3, 4], [6, 4], [1, 0]),
    ([2, 3, 1, 4], [2, 3, 2, 2], [0, 2, 3, 1]),
    ([2, 2, 3], [2, 2, 3], [2, 0, 1]),
    ([2, 2, 3], [2, 2, 3], [1, 2, 0]),
    ([2, 2, 2, 3], [8, 3], [1, 0]),
    ([2, 2, 2, 3], [2, 2, 2, 3], [3, 0, 1, 2]),
    ([2, 3, 4], [2, 4, 3], [0, 2, 1]),
    ([4], [2, 2], [1, 0]),
]

PERMUTE_VIEW_VECTORS = [
    ([2, 3, 4], [2, 0, 1], [4, 6]),
    ([2, 3, 4], [1, 2, 0], [12, 2]),
    ([2, 3, 4], [0, 2, 1], [2, 2, 2, 3]),
    ([2, 3, 4], [-1, 0, 1], [4, 6]),
    ([2, 3, 4], [2, 0, 1], [1, 4, 6]),
    ([2, 3, 4], [2, 0, 1], [4, 1, 6]),
    ([2, 3, 4], [2, 0, 1], [4, 6, 1]),
    ([2, 3, 4], [2, 0, 1], [2, 2, 2, 3]),
    ([2, 3, 4], [1, 2, 0], [3, 2, 2, 2]),
    ([2, 3, 4, 5], [0, 2, 3, 1], [2, 20, 3]),
    ([2, 3, 4, 5], [2, 3, 0, 1], [20, 2, 3]),
    ([2, 3, 4, 5], [2, 3, 0, 1], [4, 5, 6]),
    ([2, 3, 4, 5], [3, 0, 1, 2], [5, 24]),
    ([1, 2, 3, 4], [2, 3, 0, 1], [12, 2]),
    ([2, 1, 3, 4], [1, 3, 0, 2], [4, 6]),
    ([2, 3, 1, 4], [2, 0, 3, 1], [2, 2, 2, 3]),
    ([2, 2, 3], [2, 0, 1], [3, 2, 2]),
    ([2, 2, 3], [1, 2, 0], [2, 3, 2]),
    ([2, 2, 2, 3], [3, 0, 1, 2], [3, 8]),
    ([2, 2, 2, 3], [3, 0, 1, 2], [3, 2, 2, 2]),
    ([2, 3, 4], [1, 0, 2], [3, 8]),
    ([2, 3, 4], [1, 0, 2], [6, 4]),
    ([2, 3, 4], [2, 1, 0], [4, 6]),
    ([2, 3, 4], [0, 2, 1], [8, 3]),
    ([2, 3, 4, 5], [3, 0, 1, 2], [10, 12]),
    ([2, 3, 4, 5], [1, 3, 0, 2], [15, 8]),
]


def swap_vector_cases():
    """Each swap vector as a lone pair, then inside three-op chains."""
    cases = []
    for shape, view_shape, dims in VIEW_PERMUTE_VECTORS:
        for variant in ("pair", "flatten", "lead_permute", "relu_fork"):
            case = Case()
            if variant == "lead_permute":
                reverse = list(reversed(range(len(shape))))
                current = case.op("permute", case.input(list(reversed(shape))), reverse)
            else:
                current = case.input(shape)
            current = case.op("permute", case.op("view", current, view_shape), dims)
            if variant == "flatten":
                current = case.op("view", current, [numel(shape)])
            elif variant == "relu_fork":
                case.outputs.append(case.op("relu", current))
                current = case.op("permute", current, random_permutation(len(view_shape)))
            case.outputs.append(current)
            cases.append(case)
    for shape, dims, view_shape in PERMUTE_VIEW_VECTORS:
        for variant in ("pair", "permute", "lead_view"):
            case = Case()
            if variant == "lead_view":
                current = case.op("view", case.input(list(reversed(shape))), shape)
            else:
                current = case.input(shape)
            current = case.op("view", case.op("permute", current, dims), view_shape)
            if variant == "permute":
                current = case.op("permute", current, random_permutation(len(view_shape)))
            case.outputs.append(current)
            cases.append(case)
    return cases


# --- Random graphs ---------------------------------------------------------


def random_permutation(rank):
    dims = list(range(rank))
    rng.shuffle(dims)
    return dims


def random_permute(shape):
    rank = len(shape)
    strategy = rng.choices(
        ["identity", "random", "singleton", "swap", "reverse", "rotate"],
        weights=[2, 6, 3, 2, 1, 2],
    )[0]
    dims = list(range(rank))
    if strategy == "random":
        dims = random_permutation(rank)
    elif strategy == "singleton":
        units = [axis for axis in dims if shape[axis] == 1]
        dims = [axis for axis in dims if shape[axis] != 1]
        for axis in units:
            dims.insert(rng.randint(0, len(dims)), axis)
    elif strategy == "swap" and rank >= 2:
        i, j = rng.sample(range(rank), 2)
        dims[i], dims[j] = dims[j], dims[i]
    elif strategy == "reverse":
        dims.reverse()
    elif strategy == "rotate" and rank >= 2:
        if rng.random() < 0.5:
            dims = [0] + dims[2:] + [1]
        else:
            dims = [0, rank - 1] + dims[1:-1]
    if rank and rng.random() < 0.25:
        dims = [dim - rank if rng.random() < 0.5 else dim for dim in dims]
    return dims


def random_view(shape):
    n = numel(shape)
    target = list(shape)
    strategy = rng.choices(
        ["identity", "insert", "remove", "split", "merge", "refactor", "reorder", "flatten"],
        weights=[1, 3, 2, 3, 3, 3, 2, 1],
    )[0]
    if strategy == "insert" or (strategy == "remove" and 1 not in shape):
        for _ in range(rng.choice([1, 1, 2])):
            target.insert(rng.randint(0, len(target)), 1)
    elif strategy == "remove":
        units = [axis for axis, dim in enumerate(shape) if dim == 1]
        drop = set(rng.sample(units, rng.randint(1, len(units))))
        target = [dim for axis, dim in enumerate(shape) if axis not in drop]
    elif strategy == "split":
        composite = [axis for axis, dim in enumerate(shape) if len(prime_factors(dim)) > 1]
        if composite:
            axis = rng.choice(composite)
            primes = prime_factors(shape[axis])
            cut = rng.randint(1, len(primes) - 1)
            first = numel(primes[:cut])
            pair = [first, shape[axis] // first]
            if rng.random() < 0.5:
                pair.reverse()
            target = shape[:axis] + pair + shape[axis + 1 :]
    elif strategy == "merge" and len(shape) >= 2:
        axis = rng.randint(0, len(shape) - 2)
        target = shape[:axis] + [shape[axis] * shape[axis + 1]] + shape[axis + 2 :]
    elif strategy == "refactor":
        primes = prime_factors(n)
        if rng.random() < 0.4:
            rng.shuffle(primes)
        target = []
        for prime in primes:
            if target and rng.random() < 0.5:
                target[-1] *= prime
            else:
                target.append(prime)
        for _ in range(rng.choice([0, 0, 1, 2])):
            target.insert(rng.randint(0, len(target)), 1)
    elif strategy == "reorder":
        target = rng.sample(shape, len(shape))
    elif strategy == "flatten":
        target = [n] if n != 1 or rng.random() < 0.5 else []
    while len(target) > MAX_RANK:
        axis = rng.randint(0, len(target) - 2)
        target = target[:axis] + [target[axis] * target[axis + 1]] + target[axis + 2 :]
    if target and rng.random() < 0.2:
        target[rng.randrange(len(target))] = -1
    return target


def random_case():
    case = Case()
    dtype = "i8" if rng.random() < 0.15 else "f32"
    values = []
    for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
        values.append(case.input([rng.choice(DIMS) for _ in range(rng.choice(RANKS))], dtype))
    for _ in range(rng.choice([1, 1, 2, 2, 3])):
        current = rng.choice(values)
        for _ in range(rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 6])):
            draw = rng.random()
            shape = case.shapes[current]
            if draw < 0.06:
                current = case.op("relu", current)
            elif draw < 0.53:
                current = case.op("permute", current, random_permute(shape))
            else:
                current = case.op("view", current, random_view(shape))
            values.append(current)
            if rng.random() < 0.08:
                case.outputs.append(current)
        if rng.random() < 0.9:
            case.outputs.append(current)
    if not case.outputs:
        case.outputs.append(values[-1])
    if rng.random() < 0.1:
        case.outputs.append(rng.choice(case.outputs))
    return case


def long_chain_case():
    """One linear chain of 7-12 views and permutes, where the swap loop runs
    many rounds and picks the first swappable pair each time."""
    case = Case()
    current = case.input([rng.choice([1, 2, 2, 3, 4, 6]) for _ in range(rng.choice([2, 3, 3, 4]))])
    for step in range(rng.randint(7, 12)):
        shape = case.shapes[current]
        if step % 2 or rng.random() < 0.2:
            current = case.op("permute", current, random_permute(shape))
        else:
            current = case.op("view", current, random_view(shape))
    case.outputs.append(current)
    return case


def main():
    cases = upstream_canonicalize_cases() + swap_vector_cases()
    cases += [random_case() for _ in range(RANDOM_CASES)]
    cases += [long_chain_case() for _ in range(LONG_CHAIN_CASES)]
    lines = []
    for index, case in enumerate(cases):
        lines += run_case(index, case)
    header = [
        "# rc4 CanonicalizeViewCopyPermutePass (Arm configuration) results; regenerate",
        f"# with canonicalize_rc4_cases.py (seed {SEED}, torch {torch.__version__}).",
    ]
    for module, path in [
        (base_pass, "backends/transforms/canonicalize_view_copy_permute_pass.py"),
        (arm_pass, "backends/arm/_passes/canonicalize_view_copy_permute_pass.py"),
        (dim_maps, "backends/transforms/dim_maps.py"),
        (permute_view_meta, "backends/transforms/permute_view_meta.py"),
    ]:
        with open(module.__file__, "rb") as source:
            header.append(f"# {path} sha256 {hashlib.sha256(source.read()).hexdigest()}")
    header.append("# See the script's docstring for the format.")
    sys.stdout.write("\n".join(header + lines) + "\n")


if __name__ == "__main__":
    main()
