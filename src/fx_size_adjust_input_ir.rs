//! Pure Rust IrPass: adjust conv/pool input sizes to yield exact integer outputs.
//!
//! In explicit layout mode (APU-135 P2; design §4.2-P, F5; critique B12) the
//! pass slices as rc4's `SizeAdjustInputPass`
//! (`backends/arm/_passes/size_adjust_input_pass.py`) does, where dim-order
//! mode differs:
//!
//! - A convolution input of rank 3 or more is adjusted (`get_slices_convolution`,
//!   `:199-231`): explicit mode keeps conv1d at rank 3, while dim-order mode
//!   meets conv1d only after `IrConv1dUnsqueeze`. Pool inputs stay rank 4.
//! - Each slice has its sliced val, which the explicit rewrites read for the
//!   convolution's input permute and pad remainder; dim-order mode leaves the
//!   input's unsliced val (F5).
//! - Each slice is built from the node it feeds, with that node's qparam maps
//!   emptied (`create_node(..., from_node=node)`, `:396-401`; `layout::meta`).
//!   Without qparams, its val holds its real dtype: the dtype rc4 gives the
//!   consumer's input, which the consumer's input qparam records
//!   (`layout::dtype::operand_dtype`). Dim-order mode copies the input's meta
//!   and stamps the consumer's input qparam on both maps, which its serializer
//!   needs to type the slice.
//!
//! Explicit mode is gated on the layout mode until the default-mode corpus
//! shows whether the first two changes are byte-identical there; the third
//! must stay gated.

use crate::fx_conv_args::ConvArgs;
use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::fx_meta::TensorFacts;
use crate::fx_target::{size_adjust_target_kind, PoolSizeAdjustKind, SizeAdjustTargetKind};
use crate::layout::shape::slice_shape;
use crate::layout::{dtype, meta};
use crate::pipeline_config::LayoutMode;
use crate::runner_logging::log_line;

fn pair_arg(args: &[IrArg], idx: usize, default: Vec<i64>) -> Vec<i64> {
    match args.get(idx) {
        Some(IrArg::List(items)) => items.iter().filter_map(|a| a.as_int()).collect(),
        Some(IrArg::Int(v)) => vec![*v; default.len()],
        _ => default,
    }
}

fn pooling_size_adjust_allowed(kind: PoolSizeAdjustKind, node: &crate::fx_ir::IrNode) -> bool {
    match kind {
        PoolSizeAdjustKind::Max => !node.arg_bool_or(5, false),
        PoolSizeAdjustKind::Avg => {
            let ceil_mode = node.arg_bool_or(4, false);
            let count_include_pad = node.arg_bool_or(5, true);
            let divisor_override = node.arg(6).is_some_and(|arg| !matches!(arg, IrArg::None));
            let kernel_area = node.arg_int_pair_or_scalar(1).map(|(h, w)| h * w);
            let divisor_matches_kernel =
                node.arg_int(6).is_some() && node.arg_int(6) == kernel_area;
            let pad = pair_arg(&node.args, 3, vec![0, 0]);
            let no_padding = pad.iter().all(|&value| value == 0);
            !ceil_mode
                && (!count_include_pad || no_padding)
                && (!divisor_override || divisor_matches_kernel)
        }
    }
}

/// The layout mode decides how slices are built; see the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct IrSizeAdjustInputPass {
    pub layout: LayoutMode,
}

/// rc4's slices for `consumer` (`size_adjust_input_pass.py:377-403`): one
/// `slice_copy(x, dim, 0, end)` per adjusted axis, chained from `input`,
/// placed before the consumer. Returns the last slice, or `None` if a slice
/// shape cannot be derived.
fn insert_rc4_slices(
    graph: &mut IrGraph,
    consumer: NodeId,
    input: NodeId,
    input_shape: &[i64],
    slices: &[(i64, i64, i64)],
) -> Option<NodeId> {
    let dtype = dtype::operand_dtype(graph, consumer, 0)
        .unwrap_or_else(|| graph.node_dtype_or_float32(input).to_owned());
    let mut shape = input_shape.to_vec();
    let mut shapes = Vec::with_capacity(slices.len());
    for &(dim, start, end) in slices {
        shape = slice_shape(&shape, dim, Some(start), Some(end), 1).ok()?;
        shapes.push(shape.clone());
    }
    let source = graph.node(consumer).meta.clone();
    let name = graph.node(consumer).name.clone();
    let mut current = input;
    let mut cursor = graph.rewrite_cursor_after(consumer, input);
    for (&(dim, start, end), shape) in slices.iter().zip(shapes) {
        current = cursor.slice_copy(
            format!("{name}_adj_d{dim}"),
            current,
            dim,
            start,
            end,
            None,
            meta::from_node(&source, TensorFacts::new(shape, dtype.clone())),
        );
    }
    Some(current)
}

impl IrPass for IrSizeAdjustInputPass {
    fn name(&self) -> &str {
        "IrSizeAdjustInput"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let explicit = self.layout == LayoutMode::Explicit;
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            // Skip nodes marked with skip_size_adjust
            if graph.nodes[id].meta.extra_bool("skip_size_adjust") == Some(true) {
                continue;
            }
            let Some(target_kind) = size_adjust_target_kind(&graph.nodes[id].target) else {
                continue;
            };

            let input_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => continue,
            };
            let min_rank = match target_kind {
                SizeAdjustTargetKind::Conv if explicit => 3,
                _ => 4,
            };
            let input_shape = match graph.node_shape(input_id) {
                Some(s) if s.len() >= min_rank => s.to_vec(),
                _ => continue,
            };

            let node_name = graph.nodes[id].name.clone();

            let slices = match target_kind {
                SizeAdjustTargetKind::Conv => {
                    let Some(conv) = ConvArgs::parse(&graph.nodes[id]) else {
                        continue;
                    };
                    if conv.transposed {
                        continue;
                    }
                    let weight_shape = graph
                        .node_shape(conv.weight)
                        .map(|s| s.to_vec())
                        .unwrap_or_default();
                    if weight_shape.len() < input_shape.len() {
                        continue;
                    }
                    let spatial_rank = input_shape.len() - 2;
                    if spatial_rank == 0 {
                        continue;
                    }

                    let stride = &conv.stride;
                    let raw_pad = &conv.padding;
                    let dilation = &conv.dilation;
                    let pad_per_dim: Vec<(i64, i64)> = if raw_pad.len() == spatial_rank * 2 {
                        (0..spatial_rank)
                            .map(|i| (raw_pad[i * 2], raw_pad[i * 2 + 1]))
                            .collect()
                    } else if raw_pad.len() == spatial_rank {
                        raw_pad.iter().map(|&p| (p, p)).collect()
                    } else {
                        vec![(0, 0); spatial_rank]
                    };

                    let mut slices = Vec::new();
                    for i in 0..spatial_rank {
                        let dim = i + 2;
                        let s = stride.get(i).copied().unwrap_or(1);
                        let (pad_before, pad_after) = pad_per_dim.get(i).copied().unwrap_or((0, 0));
                        let total_pad = pad_before + pad_after;
                        let d = dilation.get(i).copied().unwrap_or(1);
                        let k = weight_shape.get(dim).copied().unwrap_or(1);
                        let input_len = input_shape.get(dim).copied().unwrap_or(1);
                        let remainder = (input_len + total_pad - d * (k - 1) - 1) % s;
                        if remainder > pad_before {
                            let adjustment = remainder - pad_before;
                            slices.push((dim as i64, 0i64, input_len - adjustment));
                        }
                    }
                    slices
                }
                SizeAdjustTargetKind::Pool(pool_kind) => {
                    if !pooling_size_adjust_allowed(pool_kind, &graph.nodes[id]) {
                        continue;
                    }
                    let kernel = pair_arg(&graph.nodes[id].args, 1, vec![2, 2]);
                    let stride = pair_arg(&graph.nodes[id].args, 2, kernel.clone());
                    let pad = pair_arg(&graph.nodes[id].args, 3, vec![0, 0]);
                    let mut slices = Vec::new();
                    for i in 0..2 {
                        let dim = i + 2;
                        let k = kernel.get(i).copied().unwrap_or(2);
                        let s = stride.get(i).copied().unwrap_or(k);
                        let p = pad.get(i).copied().unwrap_or(0);
                        let input_len = input_shape.get(dim).copied().unwrap_or(1);
                        let remainder = (input_len + 2 * p - k) % s;
                        if remainder > p {
                            let adjustment = remainder - p;
                            slices.push((dim as i64, 0i64, input_len - adjustment));
                        }
                    }
                    slices
                }
            };

            if slices.is_empty() {
                continue;
            }

            if explicit {
                let Some(current) = insert_rc4_slices(graph, id, input_id, &input_shape, &slices)
                else {
                    continue;
                };
                graph.set_arg(id, 0, IrArg::Node(current));
                modified = true;
                continue;
            }

            // Create slice nodes and update the conv/pool's input arg.
            // The slice inherits meta from a (possibly layout-neutral) input node
            // that may carry no output_qparams — which leaves it FP32 at
            // serialization while the consumer expects int8. Stamp the consumer's
            // input quantization onto each slice so it stays in the input domain.
            let consumer_input_q = graph.nodes[id]
                .meta
                .input_qparams()
                .and_then(|m| m.get(&0))
                .cloned();
            let mut current = input_id;
            let slice_base_meta = graph.nodes[input_id]
                .meta
                .cloned_with_same_dtype_or_float32(input_shape.clone());
            let mut cursor = graph.rewrite_cursor_after(id, input_id);
            for (dim, start, end) in &slices {
                let mut meta = slice_base_meta.clone();
                if let Some(ref q) = consumer_input_q {
                    meta.set_both_qparams(q.clone());
                }
                let slice_id = cursor.slice_copy(
                    format!("{}_adj_d{}", node_name, dim),
                    current,
                    *dim,
                    *start,
                    *end,
                    None,
                    // Match the FX pass: copy the original input metadata onto
                    // the slice node and let the final fake-tensor refresh
                    // reconcile the exact sliced shape.
                    meta,
                );
                current = slice_id;
            }

            // Update conv/pool's first arg to point to the sliced input
            graph.set_arg(id, 0, IrArg::Node(current));
            if graph.nodes[id].name.ends_with("_conv2d") {
                graph.anchor_after(id, current);
            }
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrSizeAdjustInput] adjusted conv/pool input sizes"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{i64_list_arg, IrNode, PipelineStep, QParamMap};
    use crate::pipeline_config::PipelineConfig;
    use crate::test_ir::{f32_call, f32_placeholder, output, placeholder, qparam};

    #[test]
    fn convolution_height_and_width_trims_fuse_after_layout_conversion() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 3, 8, 8]);
        let weight = f32_placeholder(&mut graph, "weight", [4, 3, 1, 1]);
        let conv = f32_call(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(x),
                IrArg::Node(weight),
                IrArg::None,
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[0, 0]),
                i64_list_arg(&[1, 1]),
                IrArg::Bool(false),
                i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            [1, 4, 4, 4],
        );
        output(&mut graph, conv);
        assert!(IrSizeAdjustInputPass::default().run(&mut graph));
        assert_eq!(
            graph
                .live_nodes()
                .filter(|node| node.target == "aten.slice_copy.Tensor")
                .count(),
            2
        );
        crate::fx_memory_format_ir::IrToTosaMemoryFormatPass.run(&mut graph);
        assert!(crate::fx_rewrite_slice_ir::IrRewriteSlicePass.run(&mut graph));
        assert!(crate::fx_fuse_consecutive_slices_ir::IrFuseConsecutiveSlicesPass.run(&mut graph));
        let slice = graph.node(graph.node(conv).arg_node(0).unwrap());
        assert_eq!(slice.target, "backend.tosa.SLICE.default");
        assert_eq!(slice.arg_int_list(1), Some(vec![0, 0, 0, 0]));
        assert_eq!(slice.arg_int_list(2), Some(vec![1, 3, 7, 7]));
        assert_eq!(
            graph
                .live_nodes()
                .filter(|node| node.target == "backend.tosa.SLICE.default")
                .count(),
            1
        );
    }

    #[test]
    fn regression_transposed_convolution_keeps_seven_by_seven_input() {
        let mut graph = IrGraph::empty();
        let input = f32_placeholder(&mut graph, "x", [1, 1, 7, 7]);
        let weight = f32_placeholder(&mut graph, "weight", [1, 1, 2, 2]);
        let conv = f32_call(
            &mut graph,
            "conv_transpose",
            "aten.convolution.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::None,
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[0, 0]),
                i64_list_arg(&[1, 1]),
                IrArg::Bool(true),
                i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            [1, 1, 14, 14],
        );
        output(&mut graph, conv);

        assert!(!IrSizeAdjustInputPass::default().run(&mut graph));
        assert_eq!(graph.node(conv).arg_node(0), Some(input));
        assert_eq!(graph.node_shape(input), Some([1, 1, 7, 7].as_slice()));
        assert_eq!(graph.live_call_function_ids(), vec![conv]);
    }

    #[test]
    fn regression_avg_pool_with_kernel_area_divisor_slices_unused_edges() {
        let mut graph = IrGraph::empty();
        let input = f32_placeholder(&mut graph, "x", [1, 1, 5, 5]);
        let pool = f32_call(
            &mut graph,
            "pool",
            "aten.avg_pool2d.default",
            vec![
                IrArg::Node(input),
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[0, 0]),
                IrArg::Bool(false),
                IrArg::Bool(true),
                IrArg::Int(4),
            ],
            [1, 1, 2, 2],
        );
        output(&mut graph, pool);
        assert!(!crate::fx_decompose_avgpool_pad_ir::IrDecomposeAvgPoolPadPass.run(&mut graph));

        assert!(IrSizeAdjustInputPass::default().run(&mut graph));
        let width_slice = graph.node(pool).arg_node(0).unwrap();
        assert_eq!(graph.node(width_slice).target, "aten.slice_copy.Tensor");
        assert_eq!(graph.node(width_slice).arg_int(1), Some(3));
        assert_eq!(graph.node(width_slice).arg_int(3), Some(4));
        let height_slice = graph.node(width_slice).arg_node(0).unwrap();
        assert_eq!(graph.node(height_slice).target, "aten.slice_copy.Tensor");
        assert_eq!(graph.node(height_slice).arg_int(1), Some(2));
        assert_eq!(graph.node(height_slice).arg_int(3), Some(4));
        assert_eq!(graph.node(height_slice).arg_node(0), Some(input));
    }

    // Explicit layout mode: rc4's SizeAdjustInputPass (APU-135 P2).

    const EXPLICIT: IrSizeAdjustInputPass = IrSizeAdjustInputPass {
        layout: LayoutMode::Explicit,
    };
    const DIM_ORDER: IrSizeAdjustInputPass = IrSizeAdjustInputPass {
        layout: LayoutMode::DimOrder,
    };
    const SLICE: &str = "aten.slice_copy.Tensor";
    const I8: &str = "torch.int8";
    const I32: &str = "torch.int32";
    const F32: &str = "torch.float32";

    /// The graphs rc4 leaves after SizeAdjustInputPass, dumped from its
    /// pipeline (`to_edge_transform_and_lower` with the pass patched to print
    /// its result) for quantized `x -> relu -> conv` and `-> max_pool2d`
    /// models, int8 per-channel. RETAB's IR holds the relu, which rc4 has
    /// turned into a clamp, in float32 with int8 qparams (design F2), and the
    /// weight and bias in their storage dtypes.
    fn quantized_relu(graph: &mut IrGraph, shape: &[i64]) -> NodeId {
        let x = placeholder(graph, "x", shape.to_vec(), I8);
        let clamp = f32_call(
            graph,
            "clamp",
            "aten.clamp.default",
            vec![IrArg::Node(x), IrArg::Int(-33), IrArg::None],
            shape.to_vec(),
        );
        graph
            .node_mut(clamp)
            .meta
            .set_both_qparams(qparam(0.02, -33, I8));
        clamp
    }

    fn quantized_conv(
        input_shape: &[i64],
        weight_shape: &[i64],
        stride: &[i64],
        pad: &[i64],
        output_shape: &[i64],
    ) -> (IrGraph, NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let clamp = quantized_relu(&mut graph, input_shape);
        let weight = placeholder(&mut graph, "weight", weight_shape.to_vec(), I8);
        let bias = placeholder(&mut graph, "bias", vec![weight_shape[0]], I32);
        let spatial = stride.len();
        let conv = f32_call(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(clamp),
                IrArg::Node(weight),
                IrArg::Node(bias),
                i64_list_arg(stride),
                i64_list_arg(pad),
                i64_list_arg(&vec![1; spatial]),
                IrArg::Bool(false),
                i64_list_arg(&vec![0; spatial]),
                IrArg::Int(1),
            ],
            output_shape.to_vec(),
        );
        graph.node_mut(conv).meta.set_input_qparams_from([
            (0, qparam(0.02, -33, I8)),
            (1, qparam(0.01, 0, I8)),
            (2, qparam(0.0002, 0, I32)),
        ]);
        graph
            .node_mut(conv)
            .meta
            .set_single_output_qparam(qparam(0.05, 4, I8));
        output(&mut graph, conv);
        (graph, clamp, conv)
    }

    fn max_pool(
        graph: &mut IrGraph,
        input: NodeId,
        kernel: i64,
        stride: i64,
        pad: i64,
        shape: &[i64],
    ) -> NodeId {
        f32_call(
            graph,
            "pool",
            "aten.max_pool2d.default",
            vec![
                IrArg::Node(input),
                i64_list_arg(&[kernel, kernel]),
                i64_list_arg(&[stride, stride]),
                i64_list_arg(&[pad, pad]),
                IrArg::Int(1),
                IrArg::Bool(false),
            ],
            shape.to_vec(),
        )
    }

    /// The slices feeding `consumer`, first to last.
    fn slice_chain(graph: &IrGraph, consumer: NodeId) -> Vec<&IrNode> {
        let mut chain = Vec::new();
        let mut current = graph.node(consumer).arg_node(0);
        while let Some(node) = current.map(|id| graph.node(id)) {
            if node.target != SLICE {
                break;
            }
            chain.push(node);
            current = node.arg_node(0);
        }
        chain.reverse();
        chain
    }

    /// `(dim, start, end)`, the val's shape and dtype, and the qparam maps'
    /// sizes (`None` when absent).
    #[allow(clippy::type_complexity)]
    fn describe(
        node: &IrNode,
    ) -> (
        (i64, i64, i64),
        Vec<i64>,
        String,
        Option<usize>,
        Option<usize>,
    ) {
        let size = |map: Option<&QParamMap>| map.map(|map| map.iter().count());
        (
            (
                node.arg_int(1).unwrap(),
                node.arg_int(2).unwrap(),
                node.arg_int(3).unwrap(),
            ),
            node.meta.val_shape().unwrap().to_vec(),
            node.meta.val_dtype().unwrap().to_owned(),
            size(node.meta.input_qparams()),
            size(node.meta.output_qparams()),
        )
    }

    /// rc4: `aten_slice_copy_tensor = slice_copy(clamp, 2, 0, 8) :: int8[1, 3, 8]
    /// iq={} oq={}`. Dim-order mode leaves rank-3 convolutions alone.
    #[test]
    fn explicit_mode_slices_a_conv1d_input_like_rc4() {
        let conv1d = || quantized_conv(&[1, 3, 10], &[4, 3, 2], &[3], &[0], &[1, 4, 3]);
        assert!(!DIM_ORDER.run(&mut conv1d().0));

        let (mut graph, clamp, conv) = conv1d();
        assert!(EXPLICIT.run(&mut graph));
        let chain = slice_chain(&graph, conv);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].arg_node(0), Some(clamp));
        assert_eq!(chain[0].args.len(), 4);
        assert_eq!(
            describe(chain[0]),
            ((2, 0, 8), vec![1, 3, 8], I8.to_owned(), Some(0), Some(0))
        );
        // The consumer keeps its meta.
        assert_eq!(
            graph
                .node(conv)
                .meta
                .input_qparams()
                .map(|map| map.iter().count()),
            Some(3)
        );
    }

    /// rc4: `slice_copy(clamp, 2, 0, 8) :: int8[1, 3, 8, 10] iq={} oq={}`; the
    /// width needs no slice. Dim-order mode keeps its own slice: the unsliced
    /// val, and the convolution's input qparam on both maps.
    #[test]
    fn explicit_mode_slices_a_conv2d_input_like_rc4() {
        let conv2d = || {
            quantized_conv(
                &[1, 3, 9, 10],
                &[4, 3, 3, 3],
                &[3, 3],
                &[1, 1],
                &[1, 4, 3, 4],
            )
        };

        let (mut explicit, clamp, conv) = conv2d();
        assert!(EXPLICIT.run(&mut explicit));
        let chain = slice_chain(&explicit, conv);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].arg_node(0), Some(clamp));
        assert_eq!(
            describe(chain[0]),
            (
                (2, 0, 8),
                vec![1, 3, 8, 10],
                I8.to_owned(),
                Some(0),
                Some(0)
            )
        );

        let (mut dim_order, _, conv) = conv2d();
        assert!(DIM_ORDER.run(&mut dim_order));
        let chain = slice_chain(&dim_order, conv);
        assert_eq!(chain.len(), 1);
        assert_eq!(
            describe(chain[0]),
            (
                (2, 0, 8),
                vec![1, 3, 9, 10],
                F32.to_owned(),
                Some(1),
                Some(1)
            )
        );
        assert_eq!(
            chain[0].meta.output_qparam(),
            dim_order.node(conv).meta.input_qparam()
        );
    }

    /// rc4: two slices, `int8[1, 3, 8, 9]` then `int8[1, 3, 8, 8]`, both with
    /// empty maps; the max pool has input qparams only after RemoveGetItemPass.
    #[test]
    fn explicit_mode_slices_a_max_pool_input_like_rc4() {
        let mut graph = IrGraph::empty();
        let clamp = quantized_relu(&mut graph, &[1, 3, 9, 9]);
        let pool = max_pool(&mut graph, clamp, 3, 3, 1, &[1, 3, 3, 3]);
        graph
            .node_mut(pool)
            .meta
            .set_single_input_qparam(qparam(0.02, -33, I8));
        graph
            .node_mut(pool)
            .meta
            .set_output_qparams(QParamMap::new());
        output(&mut graph, pool);

        assert!(EXPLICIT.run(&mut graph));
        let slices: Vec<_> = slice_chain(&graph, pool)
            .into_iter()
            .map(describe)
            .collect();
        assert_eq!(
            slices,
            [
                ((2, 0, 8), vec![1, 3, 8, 9], I8.to_owned(), Some(0), Some(0)),
                ((3, 0, 8), vec![1, 3, 8, 8], I8.to_owned(), Some(0), Some(0)),
            ]
        );
        // The vals are what a layout refresh derives, so later explicit passes
        // that refresh a slice leave it as it is.
        for slice in slice_chain(&graph, pool) {
            let refreshed = crate::layout::shape::refreshed_meta(&graph, slice.id).unwrap();
            assert_eq!(refreshed.tensor_facts(), slice.meta.tensor_facts());
        }
    }

    /// rc4's `_needs_truncation` (`test_size_adjust_input_pass.py:60-61`).
    fn needs_truncation(input: i64, kernel: i64, stride: i64, pad: i64) -> bool {
        (input + 2 * pad - kernel).rem_euclid(stride) > pad
    }

    fn float_conv(
        input_shape: &[i64],
        kernel: i64,
        stride: i64,
        pad: i64,
        transposed: bool,
    ) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", input_shape.to_vec());
        let weight = f32_placeholder(&mut graph, "weight", [16, 3, kernel, kernel]);
        let bias = f32_placeholder(&mut graph, "bias", [16]);
        let conv = f32_call(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(x),
                IrArg::Node(weight),
                IrArg::Node(bias),
                i64_list_arg(&[stride, stride]),
                i64_list_arg(&[pad, pad]),
                i64_list_arg(&[1, 1]),
                IrArg::Bool(transposed),
                i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            [1, 16, 3, 3],
        );
        output(&mut graph, conv);
        (graph, conv)
    }

    /// Ports of rc4's static `test_size_adjust_input_static_conv2d`,
    /// `..._no_adjustment_needed` and `..._skips_transpose_conv2d`
    /// (`backends/arm/test/passes/test_size_adjust_input_pass.py:64-154`) on
    /// float graphs: the convolution's input val no longer needs truncation,
    /// and the slices have no qparam maps because the convolution has none.
    #[test]
    fn explicit_mode_passes_rc4s_static_convolution_tests() {
        let (mut graph, conv) = float_conv(&[1, 3, 9, 9], 3, 3, 1, false);
        assert!(needs_truncation(9, 3, 3, 1));
        assert!(EXPLICIT.run(&mut graph));
        let input = graph.node(conv).arg_node(0).unwrap();
        let shape = graph.node_shape(input).unwrap();
        assert_eq!(shape, [1, 3, 8, 8]);
        assert!(!needs_truncation(shape[2], 3, 3, 1) && !needs_truncation(shape[3], 3, 3, 1));
        for slice in slice_chain(&graph, conv) {
            assert_eq!(slice.meta.val_dtype(), Some(F32));
            assert!(!slice.meta.has_qparam_metadata());
            assert_eq!(slice.meta.val_dim_order(), Some(&[0, 1, 2, 3][..]));
        }

        let (mut graph, _) = float_conv(&[1, 3, 9, 9], 3, 1, 1, false);
        assert!(!EXPLICIT.run(&mut graph));
        let (mut graph, conv) = float_conv(&[1, 3, 16, 16], 3, 2, 0, true);
        assert!(!EXPLICIT.run(&mut graph));
        assert!(slice_chain(&graph, conv).is_empty());
    }

    /// Ports of rc4's `test_size_adjust_input_static_pooling` and
    /// `..._no_adjustment_needed` (`:269-342`).
    #[test]
    fn explicit_mode_passes_rc4s_static_pooling_tests() {
        for (stride, sliced) in [(3, true), (1, false)] {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", [1, 3, 9, 9]);
            let pool = max_pool(&mut graph, x, 3, stride, 1, &[1, 3, 3, 3]);
            output(&mut graph, pool);
            assert_eq!(needs_truncation(9, 3, stride, 1), sliced);
            assert_eq!(EXPLICIT.run(&mut graph), sliced);
            let input = graph.node(pool).arg_node(0).unwrap();
            let shape = graph.node_shape(input).unwrap();
            assert!(
                !needs_truncation(shape[2], 3, stride, 1)
                    && !needs_truncation(shape[3], 3, stride, 1)
            );
        }
    }

    /// rc4 expands a one-element stride, padding or dilation around the
    /// channel before it computes the slices (`expand_around_channel`,
    /// `arm_pass_utils.py:135-145`, called at `size_adjust_input_pass.py:211-215`).
    /// `F.conv2d(x, w, stride=[3], padding=[1])` exports as
    /// `convolution(x, w, None, [3], [1], [1, 1], False, [0, 0], 1)`. rc4's
    /// SizeAdjustInputPass, run in .venv-et16 on that edge graph, slices a
    /// `[1, 3, 9, 9]` input to `[1, 3, 8, 9]` then `[1, 3, 8, 8]`, and leaves a
    /// `[1, 3, 10, 10]` input alone. RETAB reads the missing width stride as 1
    /// and both pads as 0, so it slices nothing on 9x9 and cuts a visited row
    /// on 10x10 (the conv would then have 3 output rows, not 4).
    #[test]
    #[ignore = "BUG: explicit-mode slicing does not expand one-element conv stride/padding lists as rc4's expand_around_channel does"]
    fn explicit_mode_expands_one_element_conv_params_like_rc4() {
        let build = |input: [i64; 4], out: [i64; 4]| {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", input);
            let weight = f32_placeholder(&mut graph, "weight", [4, 3, 3, 3]);
            let conv = f32_call(
                &mut graph,
                "conv",
                "aten.convolution.default",
                vec![
                    IrArg::Node(x),
                    IrArg::Node(weight),
                    IrArg::None,
                    i64_list_arg(&[3]),
                    i64_list_arg(&[1]),
                    i64_list_arg(&[1, 1]),
                    IrArg::Bool(false),
                    i64_list_arg(&[0, 0]),
                    IrArg::Int(1),
                ],
                out,
            );
            output(&mut graph, conv);
            (graph, conv)
        };
        let slices = |graph: &IrGraph, conv: NodeId| -> Vec<((i64, i64, i64), Vec<i64>)> {
            slice_chain(graph, conv)
                .into_iter()
                .map(|slice| {
                    let (bounds, shape, ..) = describe(slice);
                    (bounds, shape)
                })
                .collect()
        };

        let (mut graph, conv) = build([1, 3, 9, 9], [1, 4, 3, 3]);
        let adjusted = EXPLICIT.run(&mut graph);
        assert_eq!(
            slices(&graph, conv),
            [((2, 0, 8), vec![1, 3, 8, 9]), ((3, 0, 8), vec![1, 3, 8, 8])]
        );
        assert!(adjusted);

        let (mut graph, conv) = build([1, 3, 10, 10], [1, 4, 4, 4]);
        assert!(!EXPLICIT.run(&mut graph));
        assert!(slices(&graph, conv).is_empty());
    }

    /// rc4 inserts the slices directly before the node they feed
    /// (`graph.inserting_before(node)`, `size_adjust_input_pass.py:377`).
    #[test]
    fn explicit_slices_sit_directly_before_their_consumer() {
        let mut graph = IrGraph::empty();
        let clamp = quantized_relu(&mut graph, &[1, 3, 9, 9]);
        let other = f32_call(
            &mut graph,
            "other",
            "aten.neg.default",
            vec![IrArg::Node(clamp)],
            [1, 3, 9, 9],
        );
        let pool = max_pool(&mut graph, clamp, 3, 3, 1, &[1, 3, 3, 3]);
        let out = output(&mut graph, pool);
        graph.set_args(
            out,
            vec![IrArg::List(vec![IrArg::Node(pool), IrArg::Node(other)])],
        );

        assert!(EXPLICIT.run(&mut graph));
        let order: Vec<&str> = graph
            .topological_live_node_ids()
            .into_iter()
            .map(|id| graph.node(id).name.as_str())
            .collect();
        assert_eq!(
            order,
            [
                "x",
                "clamp",
                "other",
                "pool_adj_d2",
                "pool_adj_d3",
                "pool",
                "output"
            ]
        );
    }

    /// The main and region pipelines build the pass for their layout mode.
    #[test]
    fn pipelines_slice_as_their_layout_mode_says() {
        let pass_in = |steps: Vec<PipelineStep>| {
            steps
                .into_iter()
                .find_map(|step| match step {
                    PipelineStep::Ir(pass) if pass.name() == "IrSizeAdjustInput" => Some(pass),
                    _ => None,
                })
                .expect("the pipeline adjusts input sizes")
        };
        for layout in [LayoutMode::DimOrder, LayoutMode::Explicit] {
            let config = PipelineConfig {
                layout,
                ..PipelineConfig::from_tosa_spec("TOSA-1.0+INT")
            };
            for steps in [
                crate::fx_pipeline::build_main_pipeline(&config),
                crate::fx_pipeline::build_ir_cond_submodule_pipeline(&config),
            ] {
                let (mut graph, _, conv) =
                    quantized_conv(&[1, 3, 10], &[4, 3, 2], &[3], &[0], &[1, 4, 3]);
                let adjusted = pass_in(steps).run(&mut graph);
                assert_eq!(adjusted, layout == LayoutMode::Explicit, "{layout:?}");
                assert_eq!(slice_chain(&graph, conv).len(), usize::from(adjusted));
            }
        }
    }
}
