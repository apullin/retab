"""Probe: the propagation driver hands RemovePermutes users in FX insertion
order, which the Down sweep's rewiring leaves out of graph order (the
minimized form of a motif search case). Prints whether rc4's Down result
changes when every node's users are re-sorted into graph order before each
fusion sub-pass, then records the case for the port's replay. Usage:
`probe_user_order_min.py` (writes $RETAB_REVIEW_DIR/down.txt, down_up.txt)."""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import user_order_search as S  # noqa: E402

G = S.G
A = G.A
P = S.P


def driver_user_order(b):
    x = b.input("x0", (3, 3))
    p1 = b.permute(x, [1, 0])
    p2 = b.permute(p1, [1, 0])
    v3 = b.view(p2, [3, 3])
    n4 = b.call(A.neg.default, v3)
    p5 = b.permute(n4, [1, 0])
    s6 = b.call(A.sub.Tensor, p5, p2)
    p7 = b.permute(s6, [1, 0])
    p8 = b.permute(p7, [1, 0])
    p9 = b.permute(p8, [1, 0])
    v10 = b.view(p9, [3, 3])
    m11 = b.call(A.mul.Tensor, p9, v10)
    s12 = b.call(A.sub.Tensor, m11, p1)
    p13 = b.permute(s12, [1, 0])
    b.output([p1, p13])


for sort in (False, True):
    out = S.run(driver_user_order, S.PropagateViewCopyPermuteDownPass, sort)
    label = "users re-sorted into graph order" if sort else "as rc4 runs"
    print(f"--- Down, {label}: {out.count('permute_copy')} permutes\n{out}")
S.MODE["sort"] = False  # record what rc4 actually does
cases = {"driver_user_order": driver_user_order}
P.emit(cases, "down", P.out_path("down"))
P.emit(cases, "down_up", P.out_path("down_up"))
