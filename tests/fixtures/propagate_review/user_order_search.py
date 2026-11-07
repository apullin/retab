"""Search for graphs where rc4's propagation driver depends on FX user order.

Runs each graph through the Arm pass twice: as is, and with every node's
users re-sorted into graph order before each fusion sub-pass (which is what
the port assumes). Prints the cases whose results differ."""
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import probe_lib as P  # noqa: E402
import torch  # noqa: E402
from executorch.backends.transforms import propagate_view_copy_permute_pass as shared  # noqa: E402
from executorch.backends.arm._passes import PropagateViewCopyPermuteDownPass, PropagateViewCopyPermuteUpPass  # noqa: E402
from executorch.backends.arm.tosa.specification import TosaLoweringContext, TosaSpecification  # noqa: E402

G = P.G
A = G.A
MODE = {"sort": False}

orig_h = shared.PropagateViewCopyPermutePass.fuse_horizontal
orig_v = shared.PropagateViewCopyPermutePass.fuse_vertical
UP_H = shared.PropagateViewCopyPermuteUpPass.fuse_horizontal
DOWN_H = shared.PropagateViewCopyPermuteDownPass.fuse_horizontal


def sort_users(gm):
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    for n in gm.graph.nodes:
        items = sorted(n.users.items(), key=lambda kv: order[kv[0]])
        n.users.clear()
        n.users.update(items)


def wrap(fn):
    def inner(self, gm):
        if MODE["sort"]:
            sort_users(gm)
        return fn(self, gm)
    return inner


shared.PropagateViewCopyPermuteUpPass.fuse_horizontal = wrap(UP_H)
shared.PropagateViewCopyPermuteDownPass.fuse_horizontal = wrap(DOWN_H)
shared.PropagateViewCopyPermutePass.fuse_vertical = wrap(orig_v)


def run(build, cls, sort):
    MODE["sort"] = sort
    with TosaLoweringContext(TosaSpecification.create_from_string(G.PROPAGATE_TOSA_SPEC)):
        b = G.Builder()
        build(b)
        ep = b.exported_program()
        kinds = G.placeholder_kinds(ep)
        try:
            gp = G.PropagationPass(ep, cls)
            res = gp(ep.graph_module)
            return "\n".join(G.dump(res.graph_module, kinds))
        except Exception as e:  # noqa: BLE001
            return f"raises {type(e).__name__}"


def search(builders, label):
    found = []
    for name, build in builders:
        for cls in (PropagateViewCopyPermuteDownPass, PropagateViewCopyPermuteUpPass):
            try:
                a = run(build, cls, False)
                b2 = run(build, cls, True)
            except G.BuildError:
                continue
            except Exception:  # noqa: BLE001
                continue
            if a != b2:
                found.append((name, cls.__name__))
                print(f"{label} {name} {cls.__name__}: results depend on user order", flush=True)
    return found


if __name__ == "__main__":
    count, seed = int(sys.argv[1]), int(sys.argv[2])
    builders = [(f"random_{i}", G.propagation_random_case(seed * 100_000 + i)) for i in range(count)]
    found = search(builders, "random")
    print(f"done: {len(found)} order-dependent of {count}")
