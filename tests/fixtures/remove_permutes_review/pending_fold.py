"""Evidence for the review note on constants folded before phase 6.

rc4 folds constant-only subgraphs (FuseConstantArgsPass) before phase 6;
RETAB records the fold as a lowering effect and keeps the nodes. Compare
RemovePermutesAroundElementwiseTosaOps on rc4's folded graph with the same
pass on the unfolded graph the RETAB port sees (until APU-135 P2 S0a materializes
the folds in the IR).

    cd /tmp && <venv>/bin/python -P <retab>/tests/fixtures/remove_permutes_review/pending_fold.py
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layout_replay"))
import gen_layout_replay as G
import torch
from executorch.backends.arm._passes import FuseConstantArgsPass, RemovePermutesAroundElementwiseTosaOps
from executorch.backends.arm.tosa.specification import TosaLoweringContext, TosaSpecification
logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
A = G.A

def e1(b):
    """A rank-raising view of a 2-d parameter feeds an NHWC region."""
    w = b.constant("param", "p_pos", (4, 5))
    x = b.input("x", (1, 3, 4, 5))
    p = b.permute(x, [0, 2, 3, 1])                 # [1,4,5,3]
    wv = b.view(w, [1, 4, 5, 1])
    m = b.call(A.mul.Tensor, p, wv)
    b.output([b.permute(m, [0, 3, 1, 2])])

def e2(b):
    """A permute of a parameter feeds the region next to the entry permute."""
    w = b.constant("param", "p_bias", (1, 8, 4))
    x = b.input("x", (1, 8, 4))
    xp = b.permute(x, [0, 2, 1])
    bp = b.permute(w, [0, 2, 1])
    r = b.call(A.relu.default, b.call(A.add.Tensor, xp, bp))
    b.output([b.permute(r, [0, 2, 1])])

def e3(b):
    """A product of two parameters feeds the region."""
    s = b.constant("param", "p_scale", (1, 4, 5, 3))
    g = b.constant("param", "p_gain", (1, 4, 5, 3))
    x = b.input("x", (1, 3, 4, 5))
    p = b.permute(x, [0, 2, 3, 1])
    k = b.call(A.mul.Tensor, s, g)
    b.output([b.permute(b.call(A.add.Tensor, p, k), [0, 3, 1, 2])])

def run(build, fold):
    with TosaLoweringContext(TosaSpecification.create_from_string(G.TOSA_SPEC)):
        bld = G.Builder(); build(bld); ep = bld.exported_program()
        if fold:
            res = FuseConstantArgsPass(ep)(ep.graph_module)
            from executorch.exir._program_utils import _get_updated_graph_signature
            ep._graph_module = res.graph_module
            ep._graph_signature = _get_updated_graph_signature(ep.graph_signature, res.graph_module)
        kinds = G.placeholder_kinds(ep)
        before = G.dump(ep.graph_module, kinds)
        res = RemovePermutesAroundElementwiseTosaOps(ep).call(ep.graph_module)
        after = G.dump(res.graph_module, G.placeholder_kinds(ep))
    return before, res.modified, after

for name, build in [("e1", e1), ("e2", e2), ("e3", e3)]:
    for fold in (True, False):
        before, modified, after = run(build, fold)
        label = "rc4 pipeline (folded by FuseConstantArgsPass)" if fold else "unfolded graph (what the RETAB port sees)"
        print(f"=== {name}: {label}\n  before:")
        print("\n".join("    " + l for l in before))
        print(f"  after modified={modified}:")
        print("\n".join("    " + l for l in after))
