"""Which driver sub-pass makes user_order_motif.py's m1141 and m3299 (seed 71)
depend on FX user order? Each run re-sorts every node's users into graph order
right before one sub-pass. FuseIdenticalInputTransforms runs just before
RemovePermutes in each fusion round and retraces only when it changes the
graph, so a sort before it reaches RemovePermutes too; a sort before
Canonicalize, which runs after RemovePermutes, changes nothing."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
src = open(str(__import__("pathlib").Path(__file__).resolve().parent / "user_order_motif.py")).read().split("count, seed = int(sys.argv[1])")[0]
ns = {"__name__": "motif", "__file__": __file__}
exec(compile(src, "user_order_motif", "exec"), ns)
S = ns["S"]
G = S.G
from executorch.backends.transforms import remove_permutes_around_elementwise_ops as rp
from executorch.backends.transforms import canonicalize_view_copy_permute_pass as cv
from executorch.backends.transforms import fuse_identical_input_transforms_pass as fi
from executorch.backends.transforms import propagate_view_copy_permute_pass as shared

# Undo the generic hooks; install per-sub-pass hooks instead.
S.MODE["sort"] = False
WHICH = {"name": None}
orig = {
    "remove_permutes": rp.RemovePermutesAroundElementwiseOps.call,
    "canonicalize": cv.CanonicalizeViewCopyPermutePass.call,
    "fiit": fi.FuseIdenticalInputTransformsPass.call,
}

def hook(name):
    def call(self, gm):
        if WHICH["name"] == name:
            S.sort_users(gm)
        return orig[name](self, gm)
    return call

rp.RemovePermutesAroundElementwiseOps.call = hook("remove_permutes")
cv.CanonicalizeViewCopyPermutePass.call = hook("canonicalize")
fi.FuseIdenticalInputTransformsPass.call = hook("fiit")

for case in ("m1141", "m3299"):
    build = ns["make"](71 * 100_000 + int(case[1:]))
    WHICH["name"] = None
    base = S.run(build, S.PropagateViewCopyPermuteDownPass, False)
    for name in orig:
        WHICH["name"] = name
        out = S.run(build, S.PropagateViewCopyPermuteDownPass, False)
        n_perm = out.count("permute_copy")
        print(f"{case}: users sorted before {name:16}: {'changes' if out != base else 'same   '} the result ({n_perm} permutes; as is: {base.count('permute_copy')})")
