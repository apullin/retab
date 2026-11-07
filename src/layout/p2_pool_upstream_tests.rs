//! Ports of rc4's own tests for the pool and upsample rewrites (APU-135 P2
//! S7): `backends/arm/test/passes/test_rewrite_avg_pool2d_pass.py`,
//! `test_rewrite_max_pool2d_pass.py` and `test_rewrite_upsample_pass.py`.
//!
//! rc4's pass tests export a float module, run the pass (with RemoveGetItem
//! and DecomposeMaxPool2d before the max-pool rewrite) and count ops. The
//! ports start from the edge graph rc4's PassPipeline hands the pass, in
//! RETAB's form (`max_pool2d_with_indices` + `getitem`, which RETAB keeps),
//! and check rc4's whole result: the graphs were confirmed by running the
//! upstream tests in .venv-et16 with each pass dumping the graph it received
//! and returned. rc4's op counts hold as a consequence.
//!
//! Not ported, because they need symbolic shapes, which RETAB never lowers:
//! `test_rewrite_max_pool2d_tosa_dynamic_shape` and its adaptive variant
//! (MAX_POOL2D_ADAPTIVE is TOSA 1.1 + shape, design F1), and
//! `test_get_resize_parameters_1d_rejects_non_constant_symbolic_ratio`. The
//! symbolic constant-ratio test is ported at the symbol's hint. rc4's op tests
//! (`test_avg_pool2d.py`, `test_max_pool.py`, `test_upsample_*.py`) count TOSA
//! ops only on FP8 pipelines, which RETAB doesn't lower; their INT cases are
//! covered by the op-suite run, as RETAB's pipeline runs these rewrites.

use crate::fx_ir::{IrArg, IrNode, IrPass, NodeMeta, NodeOp};
use crate::layout::rewrite_avg_pool2d::IrRewriteAvgPool2dPass;
use crate::layout::rewrite_max_pool2d::IrRewriteMaxPool2dPass;
use crate::layout::rewrite_upsample::{resize_parameters_1d, ResizeMode};
use crate::layout::testing::{ints, node, LayoutGraph, PERMUTE, TOSA_AVG_POOL2D, TOSA_MAX_POOL2D};

const F32: &str = "torch.float32";
const FULL: &str = "aten.full.default";

/// `IrArg` has no `PartialEq`; its derived `Debug` is exact.
fn assert_args(node: &IrNode, expected: &[IrArg]) {
    assert_eq!(
        format!("{:?}", node.args),
        format!("{expected:?}"),
        "{}",
        node.name
    );
}

/// The live calls in graph order.
fn calls(g: &LayoutGraph) -> Vec<&IrNode> {
    g.graph()
        .topological_live_node_ids()
        .into_iter()
        .map(|id| g.graph().node(id))
        .filter(|node| node.op == NodeOp::CallFunction)
        .collect()
}

fn count(g: &LayoutGraph, target: &str) -> usize {
    g.count(target)
}

// ---------------------------------------------------------------------------
// test_rewrite_avg_pool2d_pass.py
// ---------------------------------------------------------------------------

/// `x -> avg_pool2d(x, args...)` as export gives it, trailing defaults
/// omitted: rc4's float modules.
fn avg_pool_module(input: &[i64], args: Vec<IrArg>, output: &[i64]) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", input);
    let mut all_args = vec![node(x)];
    all_args.extend(args);
    let pool = g.add_unrefreshed("aten.avg_pool2d.default", all_args);
    g.graph_mut().node_mut(pool).meta = NodeMeta::tensor(output.to_vec(), F32);
    g.output(&[pool]);
    g
}

/// rc4's result for a float pool: two float32 zero points of 0, the permutes
/// and AVG_POOL2D with a float32 accumulator.
fn assert_avg_pool_result(
    g: &LayoutGraph,
    kernel: [i64; 2],
    stride: [i64; 2],
    nhwc: &[i64],
    pooled: &[i64],
) {
    // `ops_after_pass`: one AVG_POOL2D, two permutes; avg_pool2d is gone.
    assert_eq!(count(g, TOSA_AVG_POOL2D), 1);
    assert_eq!(count(g, PERMUTE), 2);
    assert_eq!(count(g, "aten.avg_pool2d.default"), 0);
    let nodes = calls(g);
    let targets: Vec<&str> = nodes.iter().map(|node| node.target.as_str()).collect();
    assert_eq!(targets, [FULL, FULL, PERMUTE, TOSA_AVG_POOL2D, PERMUTE]);
    let [zp_in, zp_out, pre, pool, post] = nodes[..] else {
        unreachable!()
    };
    for zp in [zp_in, zp_out] {
        assert_args(zp, &[ints(&[1]), IrArg::Int(0)]);
        assert_eq!(
            format!("{:?}", zp.kwarg("dtype")),
            format!("{:?}", Some(IrArg::Dtype(F32.into())))
        );
    }
    assert_args(pre, &[node(g.id("x")), ints(&[0, 2, 3, 1])]);
    assert_eq!(pre.meta.val_shape(), Some(nhwc));
    assert_args(
        pool,
        &[
            node(pre.id),
            node(zp_in.id),
            node(zp_out.id),
            ints(&kernel),
            ints(&stride),
            ints(&[0, 0, 0, 0]),
            IrArg::Dtype(F32.into()),
        ],
    );
    assert_eq!(pool.meta.val_shape(), Some(pooled));
    assert_eq!(pool.meta.val_dtype(), Some(F32));
    assert_args(post, &[node(pool.id), ints(&[0, 3, 1, 2])]);
}

/// `test_rewrite_avg_pool2d_tosa[avg_pool2d_with_stride]`.
#[test]
fn rewrite_avg_pool2d_with_stride() {
    let mut g = avg_pool_module(
        &[1, 3, 8, 8],
        vec![ints(&[2, 2]), ints(&[2, 2])],
        &[1, 3, 4, 4],
    );
    assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
    assert_avg_pool_result(&g, [2, 2], [2, 2], &[1, 8, 8, 3], &[1, 4, 4, 3]);
}

/// `test_rewrite_avg_pool2d_tosa[avg_pool2d_without_stride]`: the stride is
/// the kernel.
#[test]
fn rewrite_avg_pool2d_without_stride() {
    let mut g = avg_pool_module(&[1, 3, 9, 9], vec![ints(&[3, 3])], &[1, 3, 3, 3]);
    assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
    assert_avg_pool_result(&g, [3, 3], [3, 3], &[1, 9, 9, 3], &[1, 3, 3, 3]);
}

/// `test_rewrite_avg_pool2d_tosa[avg_pool2d_list_kernel]`.
#[test]
fn rewrite_avg_pool2d_list_kernel() {
    let mut g = avg_pool_module(&[1, 3, 8, 9], vec![ints(&[2, 3])], &[1, 3, 4, 3]);
    assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
    assert_avg_pool_result(&g, [2, 3], [2, 3], &[1, 8, 9, 3], &[1, 4, 3, 3]);
}

/// `test_rewrite_avg_pool2d_tosa_empty_stride_uses_kernel_size`:
/// `F.avg_pool2d(x, kernel_size=[2, 3], stride=[])` exports with the stride
/// omitted, and AVG_POOL2D's stride (`args[4]`) is the kernel.
#[test]
fn rewrite_avg_pool2d_empty_stride_uses_kernel_size() {
    let mut g = avg_pool_module(&[1, 3, 8, 9], vec![ints(&[2, 3])], &[1, 3, 4, 3]);
    assert!(IrRewriteAvgPool2dPass.run(g.graph_mut()));
    assert_avg_pool_result(&g, [2, 3], [2, 3], &[1, 8, 9, 3], &[1, 4, 3, 3]);
    let pool = g.graph().node(
        g.graph()
            .live_call_function_ids()
            .into_iter()
            .find(|&id| g.graph().node(id).target == TOSA_AVG_POOL2D)
            .unwrap(),
    );
    assert_eq!(pool.arg_int_list(4), Some(vec![2, 3]));
}

// ---------------------------------------------------------------------------
// test_rewrite_max_pool2d_pass.py
// ---------------------------------------------------------------------------

/// `x -> max_pool2d_with_indices(x, args...) -> getitem(.., 0)` as export
/// gives it: RETAB's form of the graph RemoveGetItemPass receives.
fn max_pool_module(input: &[i64], args: Vec<IrArg>, output: &[i64]) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let x = g.input("x", input);
    let mut all_args = vec![node(x)];
    all_args.extend(args);
    let pool = g.add_unrefreshed("aten.max_pool2d_with_indices.default", all_args);
    let getitem = g.add_unrefreshed(
        "<built-in function getitem>",
        vec![node(pool), IrArg::Int(0)],
    );
    g.graph_mut().node_mut(getitem).meta = NodeMeta::tensor(output.to_vec(), F32);
    g.output(&[getitem]);
    g
}

/// rc4's result after RemoveGetItem, DecomposeMaxPool2d and
/// RewriteMaxPool2d: permute, MAX_POOL2D, permute.
fn assert_max_pool_result(
    g: &LayoutGraph,
    kernel: [i64; 2],
    stride: [i64; 2],
    nhwc: &[i64],
    pooled: &[i64],
) {
    // `ops_after_pass`: one MAX_POOL2D, and no pool or getitem left.
    assert_eq!(count(g, TOSA_MAX_POOL2D), 1);
    let nodes = calls(g);
    let targets: Vec<&str> = nodes.iter().map(|node| node.target.as_str()).collect();
    assert_eq!(targets, [PERMUTE, TOSA_MAX_POOL2D, PERMUTE]);
    let [pre, pool, post] = nodes[..] else {
        unreachable!()
    };
    assert_args(pre, &[node(g.id("x")), ints(&[0, 2, 3, 1])]);
    assert_eq!(pre.meta.val_shape(), Some(nhwc));
    assert_args(
        pool,
        &[
            node(pre.id),
            ints(&kernel),
            ints(&stride),
            ints(&[0, 0, 0, 0]),
        ],
    );
    assert_eq!(pool.meta.val_shape(), Some(pooled));
    assert_eq!(pool.meta.val_dtype(), Some(F32));
    assert_args(post, &[node(pool.id), ints(&[0, 3, 1, 2])]);
}

/// `test_rewrite_max_pool2d_tosa[max_pool2d_with_stride]`.
#[test]
fn rewrite_max_pool2d_with_stride() {
    let mut g = max_pool_module(
        &[1, 3, 8, 8],
        vec![ints(&[2, 2]), ints(&[2, 2])],
        &[1, 3, 4, 4],
    );
    assert!(IrRewriteMaxPool2dPass.run(g.graph_mut()));
    assert_max_pool_result(&g, [2, 2], [2, 2], &[1, 8, 8, 3], &[1, 4, 4, 3]);
}

/// `test_rewrite_max_pool2d_tosa[max_pool2d_without_stride]`.
#[test]
fn rewrite_max_pool2d_without_stride() {
    let mut g = max_pool_module(&[1, 3, 9, 9], vec![ints(&[3, 3])], &[1, 3, 3, 3]);
    assert!(IrRewriteMaxPool2dPass.run(g.graph_mut()));
    assert_max_pool_result(&g, [3, 3], [3, 3], &[1, 9, 9, 3], &[1, 3, 3, 3]);
}

/// `test_rewrite_max_pool2d_tosa[max_pool2d_list_kernel]`.
#[test]
fn rewrite_max_pool2d_list_kernel() {
    let mut g = max_pool_module(&[1, 3, 8, 9], vec![ints(&[2, 3])], &[1, 3, 4, 3]);
    assert!(IrRewriteMaxPool2dPass.run(g.graph_mut()));
    assert_max_pool_result(&g, [2, 3], [2, 3], &[1, 8, 9, 3], &[1, 4, 3, 3]);
}

/// `test_rewrite_max_pool2d_tosa[max_pool2d_empty_stride]` and
/// `test_rewrite_max_pool2d_tosa_empty_stride_uses_kernel_size`:
/// `F.max_pool2d(x, kernel_size=[2, 2], stride=[])` exports with the stride
/// omitted, and MAX_POOL2D's stride (`args[2]`) is the kernel.
#[test]
fn rewrite_max_pool2d_empty_stride_uses_kernel_size() {
    let mut g = max_pool_module(&[1, 3, 8, 8], vec![ints(&[2, 2])], &[1, 3, 4, 4]);
    assert!(IrRewriteMaxPool2dPass.run(g.graph_mut()));
    assert_max_pool_result(&g, [2, 2], [2, 2], &[1, 8, 8, 3], &[1, 4, 4, 3]);
    let pool = calls(&g)[1];
    assert_eq!(pool.arg_int_list(2), Some(vec![2, 2]));
}

// ---------------------------------------------------------------------------
// test_rewrite_upsample_pass.py
// ---------------------------------------------------------------------------

/// `test_get_resize_parameters_1d_supports_symbolic_shapes_with_constant_ratio`
/// at the symbol's hint: input 4, output 8, rc4's default bilinear mode.
#[test]
fn resize_parameters_for_a_constant_ratio() {
    assert_eq!(
        resize_parameters_1d(4, 8, false, ResizeMode::Bilinear, None),
        Ok([4, 2, -1, 1])
    );
}

/// `test_get_resize_parameters_1d_nearest_matches_pytorch_scale_factor`:
/// TOSA nearest samples `floor((out * scale_d + offset + scale_n / 2) /
/// scale_n)`, which the negative half-scale offset makes PyTorch's
/// `floor(out * input / output)`: 0 0 1 2 2 3 4 4 5 for 6 -> 9.
#[test]
fn nearest_resize_parameters_match_pytorchs_sampling() {
    let [scale_n, scale_d, offset, border] =
        resize_parameters_1d(6, 9, false, ResizeMode::Nearest, None).unwrap();
    assert_eq!([scale_n, scale_d, offset, border], [6, 4, -3, -1]);
    let tosa: Vec<i64> = (0..9)
        .map(|out| (out * scale_d + offset + scale_n / 2).div_euclid(scale_n))
        .collect();
    let pytorch: Vec<i64> = (0..9).map(|out| out * 6 / 9).collect();
    assert_eq!(tosa, [0, 0, 1, 2, 2, 3, 4, 4, 5]);
    assert_eq!(tosa, pytorch);
}

/// `test_get_resize_parameters_1d_nearest_uses_explicit_scale_factor`: 4 * 1.6
/// rounds down to 6, but PyTorch samples with 1.6, and so does TOSA. The
/// float32 value of 1.6 maps to the same compact rational.
#[test]
fn nearest_resize_parameters_use_the_explicit_scale_factor() {
    for factor in [1.6, f64::from(1.6f32)] {
        assert_eq!(
            resize_parameters_1d(4, 6, false, ResizeMode::Nearest, Some(factor)),
            Ok([16, 10, -8, -6]),
            "{factor}"
        );
    }
}

/// `test_get_resize_parameters_1d_nearest_preserves_integer_upscale_encoding`:
/// exact integer upscales keep the half-pixel encoding, which Vela lowers to
/// U55's 2x, 4x and 8x nearest-neighbour implementation.
#[test]
fn nearest_integer_upscales_keep_the_half_pixel_encoding() {
    for (output, expected) in [
        (6, [4, 2, -1, 1]),
        (12, [8, 2, -3, 3]),
        (24, [16, 2, -7, 7]),
    ] {
        assert_eq!(
            resize_parameters_1d(3, output, false, ResizeMode::Nearest, None),
            Ok(expected),
            "3 -> {output}"
        );
    }
}
