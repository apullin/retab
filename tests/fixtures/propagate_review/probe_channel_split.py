"""Probe: a channel split after an NCHW permute and a clamp, one branch
re-permuted to NHWC: Down's advance rewrites the slice's dim, the fork
can't be split (a permute branch isn't swappable), and the region move from
the original permute leaves the slice cutting the wrong axis."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import probe_lib as P  # noqa: E402
import torch  # noqa: E402
G = P.G
A = G.A


def build(b):
    y = b.input("y", (1, 3, 5, 8))                 # NHWC activations
    p = b.permute(y, [0, 3, 1, 2])                 # NCHW [1, 8, 3, 5]
    c = b.call(A.clamp.default, p, 0.0, 6.0)
    s = b.call(A.slice_copy.Tensor, c, 1, 0, 4)    # first 4 channels [1, 4, 3, 5]
    a = b.permute(s, [0, 2, 3, 1])                 # back to NHWC for the next conv
    n = b.call(A.neg.default, s)
    b.output([a, n])


P.emit({"channel_split_after_clamp": build}, "down", P.out_path("down"))
print(open(P.out_path("down")).read())
