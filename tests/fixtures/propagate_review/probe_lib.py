"""Shared helpers for the propagation review probes (src/layout/propagate_review_tests.rs).

They run rc4's Arm propagation passes, built as arm_pass_manager.py:726-727
builds them (with the exported program), on graphs made with the replay
generator's builder: the Down pass, the Up pass, or both in sequence
(`DownThenUp`), and write fixture-format cases to `$RETAB_REVIEW_DIR`
(default /tmp/retab_review) as down.txt, up.txt and down_up.txt, which
`propagate_review_external` and `propagate_review_order_only` replay.

Run with the rc4 environment from a neutral directory, for example:

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/propagate_review/fuzz_t16.py down_up 2000 5
"""
import logging
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "layout_replay"))
import gen_layout_replay as G  # noqa: E402

import torch  # noqa: E402
from executorch.backends.arm._passes import (  # noqa: E402
    PropagateViewCopyPermuteDownPass,
    PropagateViewCopyPermuteUpPass,
)
from executorch.exir.pass_base import PassResult  # noqa: E402

logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)


class DownThenUp:
    """InsertTableOps, the Arm Down pass, the Arm Up pass, then the TABLEs
    restored to RETAB's form, as G.PropagationPass does for one pass."""

    rc4_name = "InsertTableOpsPass + Down + Up"

    def __init__(self, ep):
        self.ep = ep

    def __call__(self, gm):
        down = G.PropagationPass(self.ep, PropagateViewCopyPermuteDownPass)
        # Run both passes between one InsertTableOps and one restore.
        originals = {node.name: node for node in gm.graph.nodes if G._is_table_op(node)}
        if originals:
            names = {node.name for node in gm.graph.nodes}
            gm = G.InsertTableOpsPass(self.ep)(gm).graph_module
            for node in gm.graph.nodes:
                if node.name in names:
                    continue
                if node.target == G.T.TABLE.default:
                    constant = node.args[1].name
                    node.meta[G._PROPAGATION_TABLE_OF] = constant.removeprefix("b_").removesuffix("_table_constant")
                elif node.target == G.T.RESCALE.default:
                    (table,) = node.all_input_nodes
                    node.meta[G._PROPAGATION_TABLE_OF] = table.meta[G._PROPAGATION_TABLE_OF]
                    node.meta[G._PROPAGATION_TABLE_RESCALE] = True
        modified = False
        for cls in (PropagateViewCopyPermuteDownPass, PropagateViewCopyPermuteUpPass):
            result = cls(down.compile_spec, self.ep)(gm)
            gm = result.graph_module
            modified |= result.modified
        if originals:
            G.restore_pending_tables(gm, originals)
        return PassResult(gm, modified)


SPECS = {
    "down": G.PassSpec("down", lambda ep: G.PropagationPass(ep, PropagateViewCopyPermuteDownPass), [], None, None, tosa_spec=G.PROPAGATE_TOSA_SPEC),
    "up": G.PassSpec("up", lambda ep: G.PropagationPass(ep, PropagateViewCopyPermuteUpPass), [], None, None, tosa_spec=G.PROPAGATE_TOSA_SPEC),
    "down_up": G.PassSpec("down_up", lambda ep: DownThenUp(ep), [], None, None, tosa_spec=G.PROPAGATE_TOSA_SPEC),
}


def out_path(which):
    directory = os.environ.get("RETAB_REVIEW_DIR", "/tmp/retab_review")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{which}.txt")


def emit(cases, which, out):
    lines = []
    for name, build in cases.items():
        try:
            body = G.run_case(SPECS[which], build)
        except G.BuildError as error:
            print(f"{which}/{name}: rc4 result not representable: {error}", file=sys.stderr)
            continue
        lines += [f"case {name}", *body, "end"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {sum(1 for l in lines if l.startswith('case '))} cases", file=sys.stderr)
