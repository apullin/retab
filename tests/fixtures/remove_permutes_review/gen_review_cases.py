"""Review cases for RemovePermutesAroundElementwiseTosaOps (APU-158 review).

Hand-built graphs aimed at paths the main replay fixture reaches rarely or not
at all: slices with negative bounds and steps, reductions with int and None
dims, a constant read at two ranks or by two regions, involution permutes that
are both an entry and an exit, interleaves with scale 1, constants inside a cat
list, a permuted where condition, tensor kwargs, rank-0 constants, lower-rank
`full` constants and diamonds of squeeze views. Each case runs rc4's real pass
through the replay generator's `run_case` (same pass construction and edge-set
pinning as tests/fixtures/layout_replay/gen_layout_replay.py) and is written in
the replay format to remove_permutes_review.txt next to this script. Like
the remove-permutes replay fixtures, the cases record rc4 with the release/1.6
backports of #24315 and of #24469's mixed-rank broadcast guard, so generate
them the same way (see the replay generator's docstring):

    cd /tmp && PYTHONPATH=<checkout>/src <venv>/bin/python -P <retab>/tests/fixtures/remove_permutes_review/gen_review_cases.py
"""
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "layout_replay"))
import gen_layout_replay as G  # noqa: E402
import torch  # noqa: E402

logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
A, Q, T = G.A, G.Q, G.T
TO_NHWC, TO_NCHW = G.TO_NHWC, G.TO_NCHW
cases = {}
def case(fn):
    cases[fn.__name__] = fn
    return fn

@case
def slice_negative_bounds_and_step(b):
    x = b.input("x", (2, 9, 4, 5))
    p = b.permute(x, TO_NHWC)              # [2,4,5,9]
    s = b.call(A.slice_copy.Tensor, p, 3, -7, -1, 2)
    s = b.call(A.slice_copy.Tensor, s, 1, 1, 9223372036854775807)
    s = b.call(A.slice_copy.Tensor, s, -2, 3, 1)   # empty slice
    b.output([b.permute(s, TO_NCHW)])

@case
def sum_with_dim_none_keepdim(b):
    x = b.input("x", (2, 3, 4, 5))
    p = b.permute(x, TO_NHWC)
    s = b.call(A.sum.dim_IntList, p, None, True)
    b.output([b.permute(b.call(A.sub.Tensor, p, s), TO_NCHW)])

@case
def sum_with_empty_dim_keepdim(b):
    x = b.input("x", (2, 3, 4, 5))
    p = b.permute(x, TO_NHWC)
    s = b.call(A.sum.dim_IntList, p, [], True)
    b.output([b.permute(b.call(A.sub.Tensor, p, s), TO_NCHW)])

@case
def mean_with_int_dim(b):
    x = b.input("x", (2, 3, 4, 5))
    p = b.permute(x, TO_NHWC)
    m = b.call(A.mean.dim, p, -1, True)
    b.output([b.permute(b.call(A.sub.Tensor, p, m), TO_NCHW)])

@case
def constant_feeding_nodes_at_two_ranks(b):
    x = b.input("x", (1, 4, 6))
    c = b.constant("param", "c", (1, 6, 4))
    p = b.permute(x, [0, 2, 1])             # [1,6,4]
    a = b.call(A.add.Tensor, p, c)
    v = b.view(a, [6, 4])                    # squeeze leading 1
    c2 = b.call(A.mul.Tensor, v, b.view(c, [6, 4]))
    b.output([b.permute(c2, [1, 0])])

@case
def constant_twice_in_one_node_and_once_in_another(b):
    x = b.input("x", (2, 3, 4, 5))
    c = b.constant("buffer", "c", (2, 4, 5, 3))
    p = b.permute(x, TO_NHWC)
    a = b.call(A.add.Tensor, c, c)
    m = b.call(A.mul.Tensor, p, a)
    s = b.call(A.sub.Tensor, m, c)
    b.output([b.permute(s, TO_NCHW)])

@case
def constant_shared_by_two_regions(b):
    x = b.input("x", (2, 3, 4, 5))
    y = b.input("y", (2, 3, 4, 5))
    c = b.constant("param", "c", (2, 4, 5, 3))
    r1 = b.call(A.add.Tensor, b.permute(x, TO_NHWC), c)
    r2 = b.call(A.mul.Tensor, b.permute(y, TO_NHWC), c)
    b.output([b.permute(r1, TO_NCHW), b.permute(r2, TO_NCHW)])

@case
def involution_entry_is_also_an_exit(b):
    x = b.input("x", (2, 3, 4))
    p = b.permute(x, [0, 2, 1])
    r = b.call(A.relu.default, p)
    q = b.permute(r, [0, 2, 1])
    n = b.call(A.neg.default, q)
    b.output([b.permute(n, [0, 2, 1])])

@case
def involution_permute_inside_one_region(b):
    x = b.input("x", (2, 3, 4))
    p = b.permute(x, [0, 2, 1])
    r = b.call(A.relu.default, p)
    q = b.permute(r, [0, 2, 1])
    a = b.call(A.add.Tensor, q, b.permute(r, [0, 2, 1]))
    b.output([b.permute(a, [0, 2, 1])])

@case
def interleave_with_scale_one(b):
    x = b.input("x", (1, 16, 20, 3))
    v = G.interleave(b, b.permute(x, TO_NCHW), 2, 1)
    b.output([b.permute(v, TO_NHWC)])

@case
def interleave_with_scale_one_view_head(b):
    x = b.input("x", (1, 16, 20, 3))
    v = G.interleave(b, b.permute(x, TO_NCHW), 1, 1, head="view")
    b.output([b.permute(b.call(A.relu.default, v), TO_NHWC)])

@case
def cat_with_a_constant_in_the_list(b):
    x = b.input("x", (2, 3, 4, 5))
    c = b.constant("param", "c", (2, 4, 5, 3))
    p = b.permute(x, TO_NHWC)
    k = b.call(A.cat.default, [p, c, p], 2)
    b.output([b.permute(k, TO_NCHW)])

@case
def cat_with_a_lower_rank_constant(b):
    x = b.input("x", (2, 3, 4, 5))
    c = b.constant("param", "c", (4, 5, 3))
    p = b.permute(x, TO_NHWC)
    a = b.call(A.add.Tensor, p, c)
    k = b.call(A.cat.default, [a, p], -1)
    b.output([b.permute(k, TO_NCHW)])

@case
def where_with_a_permuted_condition(b):
    x = b.input("x", (2, 3, 4, 5))
    y = b.input("y", (2, 3, 4, 5))
    c = b.constant("constant", "c", (2, 4, 5, 3))
    px, py = b.permute(x, TO_NHWC), b.permute(y, TO_NHWC)
    g = b.call(A.gt.Tensor, px, py)
    w = b.call(A.where.self, g, px, c)
    b.output([b.permute(w, TO_NCHW)])

@case
def negative_pad_crops(b):
    x = b.input("x", (1, 6, 6, 3))
    p = b.permute(x, TO_NCHW)
    pad = b.call(A.constant_pad_nd.default, p, [-1, 2, 0, -2], 0.0)
    b.output([b.permute(pad, TO_NHWC)])

@case
def amax_with_int_dim(b):
    x = b.input("x", (2, 3, 4, 5))
    p = b.permute(x, TO_NHWC)
    m = b.call(A.amax.default, p, -3, True)
    b.output([b.permute(b.call(A.sub.Tensor, p, m), TO_NCHW)])

@case
def clamp_tensor_with_tensor_kwargs(b):
    x = b.input("x", (2, 3, 4, 5))
    lo = b.input("lo", (2, 3, 4, 5))
    hi = b.constant("param", "hi", (2, 4, 5, 3))
    p = b.permute(x, TO_NHWC)
    c = b.call(A.clamp.Tensor, p, max=hi, min=b.permute(lo, TO_NHWC))
    b.output([b.permute(c, TO_NCHW)])

@case
def rank0_constant_in_rank0_region(b):
    x = b.input("x", ())
    c = b.constant("param", "c", ())
    a = b.call(A.add.Tensor, b.permute(x, []), c)
    b.output([b.permute(a, [])])

@case
def full_lower_rank_constant(b):
    x = b.input("x", (2, 3, 4, 5))
    f = b.call(A.full.default, [5, 3], 0.25)
    a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), f)
    b.output([b.permute(a, TO_NCHW)])

@case
def full_lower_rank_constant_kept_dims(b):
    x = b.input("x", (2, 3, 4, 5))
    f = b.call(A.full.default, [1, 3], 0.25)
    a = b.call(A.add.Tensor, b.permute(x, TO_NHWC), f)
    b.output([b.permute(a, TO_NCHW)])

@case
def numel1_region_node_feeding_a_view(b):
    x = b.input("x", (1, 1, 1, 1))
    p = b.permute(x, TO_NHWC)
    r = b.call(A.relu.default, p)
    v = b.view(r, [1])
    b.output([b.permute(r, TO_NCHW), v])

@case
def region_node_with_many_exit_permutes(b):
    x = b.input("x", (2, 3, 4))
    r = b.call(A.relu.default, b.permute(x, [1, 2, 0]))
    outs = [b.permute(r, [2, 0, 1]), b.permute(r, [0, 2, 1]), b.permute(r, [2, 0, 1]), b.permute(r, [1, 0, 2])]
    b.output(outs)

@case
def two_entries_chained_and_constant(b):
    x = b.input("x", (2, 3, 3))
    c = b.constant("param", "c", (2, 3, 3))
    p1 = b.permute(x, [0, 2, 1])
    p2 = b.permute(p1, [0, 2, 1])
    s = b.call(A.add.Tensor, b.call(A.sub.Tensor, p2, p1), c)
    b.output([b.permute(s, [0, 2, 1])])

@case
def view_seeded_region(b):
    x = b.input("x", (1, 4, 6))
    p = b.permute(x, [0, 2, 1])            # [1,6,4]
    v = b.view(p, [6, 4])
    r = b.call(A.relu.default, v)
    v2 = b.view(r, [1, 6, 4])
    b.output([b.permute(v2, [0, 2, 1]), b.permute(r, [1, 0])])

@case
def diamond_with_views_on_both_paths(b):
    x = b.input("x", (1, 4, 1, 6))
    p = b.permute(x, [0, 3, 2, 1])          # [1,6,1,4]
    a = b.call(A.abs.default, p)
    v1 = b.view(a, [1, 6, 4])               # squeeze dim 2
    v2 = b.view(a, [6, 1, 4])               # squeeze dim 0
    w1 = b.view(v1, [1, 6, 1, 4])
    w2 = b.view(v2, [1, 6, 1, 4])
    m = b.call(A.add.Tensor, w1, w2)
    b.output([b.permute(m, [0, 3, 2, 1])])

@case
def diamond_with_mismatched_views(b):
    x = b.input("x", (1, 4, 1, 6))
    p = b.permute(x, [0, 3, 2, 1])          # [1,6,1,4]
    a = b.call(A.abs.default, p)
    v1 = b.view(a, [6, 1, 4])               # drop dim 0
    v2 = b.view(a, [1, 6, 4])               # drop dim 2
    m = b.call(A.add.Tensor, v1, b.view(v2, [6, 1, 4]))
    b.output([b.permute(m, [2, 0, 1])])

@case
def qdq_with_tensor_kwargs_dtype(b):
    x = b.input("x", (2, 3, 4, 5))
    q = b.call(Q.quantize_per_tensor.default, b.permute(x, TO_NHWC), 0.5, 0, -128, 127, dtype=torch.int8)
    d = b.call(Q.dequantize_per_tensor.default, q, 0.5, 0, -128, 127, torch.int8, out_dtype=torch.float32)
    b.output([b.permute(d, TO_NCHW)])

@case
def rescale_of_a_permuted_int32_with_per_channel_scales(b):
    x = b.input("x", (1, 3, 4, 5), dtype=torch.int32)
    r = b.call(T.RESCALE.default, b.permute(x, TO_NHWC), torch.int8, [0.5, 0.25, 0.125], 0, 0)
    b.output([b.permute(r, TO_NCHW)])

@case
def sink_view_below_feeding_a_conv_like_op(b):
    x = b.input("x", (1, 4, 1, 1))
    w = b.constant("param", "w", (4, 3))
    p = b.permute(x, TO_NHWC)
    v = b.view(b.call(A.mul.Tensor, p, p), [1, 4])
    b.output([b.call(A.mm.default, v, w)])

spec = [p for p in G.PASSES if p.name == "remove_permutes_around_elementwise"][0]


def main() -> None:
    lines = [
        "# rc4 RemovePermutesAroundElementwiseTosaOps on the APU-158 review cases.",
        "# Generated by tests/fixtures/remove_permutes_review/gen_review_cases.py; don't edit by hand.",
        *G.source_checkout_line(),
    ]
    for name, build in cases.items():
        lines += [f"case {name}", *G.run_case(spec, build), "end"]
    out = HERE / "remove_permutes_review.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
