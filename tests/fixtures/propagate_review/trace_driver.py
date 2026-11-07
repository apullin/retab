"""Trace rc4's Arm Down pass the way `propagate_review_trace` traces the port
(`layout::propagate::TRACE`): the graph, with each node's users in torch.fx's
order, after each sweep and before each fusion pass, nodes named by position.
rc4's TABLE constants are left out and its TABLEs print without them, so the
two logs line up apart from argument spelling.

Usage, from a neutral directory with the rc4 environment:

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/propagate_review/trace_driver.py motif 71 m1141
    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/propagate_review/trace_driver.py const 3 const_190
    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/propagate_review/trace_driver.py random 7 random_544

`motif`, `const` and `random` rebuild a case of user_order_motif.py,
fuzz_constants.py and the replay generator's random propagation graphs from
its seed and index; record the same case with the matching script and run
`RETAB_REVIEW_DIR=<dir> RETAB_TRACE_CASE=<case> cargo test --lib propagate_review_trace -- --ignored --nocapture`
for the port's log.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_lib as P  # noqa: E402
import torch  # noqa: E402
from executorch.backends.arm._passes import PropagateViewCopyPermuteDownPass  # noqa: E402
from executorch.backends.arm.tosa.specification import TosaLoweringContext, TosaSpecification  # noqa: E402
from executorch.backends.transforms import canonicalize_view_copy_permute_pass as canonicalize  # noqa: E402
from executorch.backends.transforms import fuse_identical_input_transforms_pass as fiit  # noqa: E402
from executorch.backends.transforms import propagate_view_copy_permute_pass as shared  # noqa: E402
from executorch.backends.transforms import remove_permutes_around_elementwise_ops as remove_permutes  # noqa: E402

G = P.G


def script_namespace(name, cut):
    """The module-level definitions of a review script, without its driver."""
    source = (HERE / name).read_text().split(cut)[0]
    namespace = {"__name__": name, "__file__": str(HERE / name)}
    exec(compile(source, name, "exec"), namespace)
    return namespace


def builder(kind, seed, case):
    index = int(case.rsplit("_", 1)[-1].lstrip("m"))
    if kind == "motif":
        return script_namespace("user_order_motif.py", "count, seed = int(sys.argv[1])")["make"](seed * 100_000 + index)
    if kind == "const":
        return script_namespace("fuzz_constants.py", "which, count, seed = sys.argv[1]")["build_case"](seed * 100_000 + index)
    if kind == "random":
        return G.propagation_random_case(seed * 100_000 + index)
    raise SystemExit(f"unknown kind {kind}")


def target(node):
    if node.op != "call_function":
        return node.op
    text = str(node.target).split(": schema")[0].strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].split(": ", 1)[1]
    return text


def describe(gm):
    nodes = [
        n
        for n in gm.graph.nodes
        if not (n.op == "placeholder" and n.name.startswith("b_") and n.name.endswith("_table_constant"))
    ]
    position = {n: i for i, n in enumerate(nodes)}
    lines = []
    for n in nodes:
        if n.op == "output":
            continue
        args = []
        if n.op == "call_function":
            for a in n.args:
                if isinstance(a, torch.fx.Node):
                    if a in position:
                        args.append(f"%{position[a]}")
                else:
                    args.append(repr(a))
        users = [f"%{position[u]}" for u in n.users if u.op != "output"]
        if any(u.op == "output" for u in n.users):
            users.append("out")
        lines.append(f"  %{position[n]} {target(n)}({', '.join(args)}) users={users}")
    return "\n".join(lines)


LOG = []


def install_hooks():
    original_iteration = shared.PropagateViewCopyPermutePass._propagate_iteration

    def iteration(self, gm):
        modified = original_iteration(self, gm)
        LOG.append(f"after sweep modified={modified}\n{describe(gm)}")
        return modified

    shared.PropagateViewCopyPermutePass._propagate_iteration = iteration
    for cls, name in [
        (fiit.FuseIdenticalInputTransformsPass, "fiit"),
        (remove_permutes.RemovePermutesAroundElementwiseOps, "remove_permutes"),
        (canonicalize.CanonicalizeViewCopyPermutePass, "canonicalize"),
    ]:
        original = cls.call

        def call(self, gm, original=original, name=name):
            LOG.append(f"before {name}\n{describe(gm)}")
            return original(self, gm)

        cls.call = call


def main():
    kind, seed, case = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    build = builder(kind, seed, case)
    install_hooks()
    with TosaLoweringContext(TosaSpecification.create_from_string(G.PROPAGATE_TOSA_SPEC)):
        b = G.Builder()
        build(b)
        ep = b.exported_program()
        G.PropagationPass(ep, PropagateViewCopyPermuteDownPass)(ep.graph_module)
    print("\n".join(LOG))


main()
