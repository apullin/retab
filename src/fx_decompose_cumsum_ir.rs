//! Pure Rust IrPass version of FxDecomposeCumsumPass.
//!
//! Decomposes cumsum into: view_copy -> full -> convolution -> slice_copy -> view_copy.
//! This mirrors the Arm backend's cumsum lowering: reshape to 4D, convolve with
//! an all-ones kernel (cumulative sum via convolution), slice off padding, reshape back.

use crate::fx_ir::{i64_list_arg, wrap_dim, IrArg, IrGraph, IrPass, NodeId, QParam};
use crate::fx_target::is_cumsum_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeCumsumPass;

impl IrPass for IrDecomposeCumsumPass {
    fn name(&self) -> &str {
        "IrDecomposeCumsum"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            is_cumsum_target(&graph.nodes[id].target) && decompose_cumsum(graph, id)
        });

        if modified {
            log_line(format_args!("[IrDecomposeCumsum] decomposed cumsum ops"));
        }
        modified
    }
}

fn decompose_cumsum(graph: &mut IrGraph, node_id: NodeId) -> bool {
    // Extract input node and dim argument
    let input_id = match graph.nodes[node_id].first_node_arg() {
        Some(id) => id,
        None => return false,
    };

    let dim_raw = match graph.nodes[node_id].args.get(1) {
        Some(IrArg::Int(d)) => *d,
        _ => return false,
    };

    let original_shape = match graph.node_shape_vec(input_id) {
        Some(s) => s,
        None => return false,
    };
    if original_shape.is_empty() {
        return false;
    }

    let input_meta = graph.nodes[input_id].meta.clone();
    let input_stamp = input_meta.tensor_stamp_same_dtype_or_float32();

    let Some(dim) = wrap_dim(dim_raw, original_shape.len()) else {
        return false;
    };

    let pre_cumsum_dim: i64 = if dim > 0 {
        original_shape[..dim].iter().product()
    } else {
        1
    };
    let cumsum_dim = original_shape[dim];
    let post_cumsum_dim: i64 = if dim < original_shape.len() - 1 {
        original_shape[dim + 1..].iter().product()
    } else {
        1
    };

    // Trivial case: cumsum_dim <= 1 is identity
    if cumsum_dim <= 1 {
        graph.replace_and_erase(node_id, input_id);
        return true;
    }

    let node_name = graph.nodes[node_id].name.clone();
    let node_meta = graph.nodes[node_id].meta.clone();

    let conv_shape = vec![pre_cumsum_dim, 1, cumsum_dim, post_cumsum_dim];
    let weight_shape = vec![1_i64, 1, cumsum_dim, 1];

    // 1. view_copy(input, conv_shape) — reshape to 4D
    let view1_meta = input_stamp.fresh_tensor(conv_shape.as_slice());
    let view1_id = graph.add_anchored_view_copy(
        node_id,
        format!("{}_view_pre", node_name),
        input_id,
        &conv_shape,
        view1_meta,
    );

    // 2. full(weight_shape, 1.0) — all-ones convolution kernel
    let weight_meta = input_stamp.fresh_tensor(weight_shape.as_slice());
    let weight_id = graph.add_anchored_call_function(
        node_id,
        format!("{}_weight", node_name),
        "aten.full.default",
        vec![i64_list_arg(&weight_shape), IrArg::Float(1.0)],
        weight_meta,
    );

    // 3. convolution(view1, weight, None, [1,1], [cumsum_dim-1, 0], [1,1], false, [0], 1)
    //    The padding on dim 2 causes the conv to compute prefix sums.
    //    Output shape: [pre, 1, cumsum_dim + (cumsum_dim-1), post] = [pre, 1, 2*cumsum_dim-1, post]
    let conv_out_h = cumsum_dim + (cumsum_dim - 1); // padded dim
    let conv_out_shape = vec![pre_cumsum_dim, 1, conv_out_h, post_cumsum_dim];
    let mut conv_meta = input_stamp.fresh_tensor(conv_out_shape.as_slice());

    // Quantization fixup: propagate input qparams, add weight qparam at index 1
    if let Some(orig_iqp) = node_meta.input_qparams() {
        if !orig_iqp.is_empty() {
            let mut conv_iqp = orig_iqp.clone();
            // Weight qparam: scale=1.0, zp=0, int8 range
            conv_iqp.insert(
                1,
                QParam::per_tensor(1.0, 0, -128, 127, "torch.int8".into()),
            );
            conv_meta.set_input_qparams(conv_iqp);
            if let Some(output_qparam) = node_meta.output_qparam().cloned() {
                conv_meta.set_single_output_qparam(output_qparam);
            }

            graph
                .node_mut(weight_id)
                .meta
                .set_single_output_qparam(QParam::per_tensor(
                    1.0,
                    0,
                    -128,
                    127,
                    "torch.int8".into(),
                ));
        }
    }

    let conv_id = graph.add_anchored_call_function(
        node_id,
        format!("{}_conv", node_name),
        "aten.convolution.default",
        vec![
            IrArg::Node(view1_id),
            IrArg::Node(weight_id),
            IrArg::None,                                                  // bias
            IrArg::List(vec![IrArg::Int(1), IrArg::Int(1)]),              // stride
            IrArg::List(vec![IrArg::Int(cumsum_dim - 1), IrArg::Int(0)]), // padding
            IrArg::List(vec![IrArg::Int(1), IrArg::Int(1)]),              // dilation
            IrArg::Bool(false),                                           // transposed
            IrArg::List(vec![IrArg::Int(0)]),                             // output_padding
            IrArg::Int(1),                                                // groups
        ],
        conv_meta,
    );

    // 4. slice_copy(conv, dim=2, start=0, end=cumsum_dim) — trim padding
    let slice_shape = vec![pre_cumsum_dim, 1, cumsum_dim, post_cumsum_dim];
    let slice_meta = input_stamp.fresh_tensor(slice_shape.as_slice());
    let slice_id = graph.add_anchored_slice_copy(
        node_id,
        format!("{}_slice", node_name),
        conv_id,
        2,
        0,
        cumsum_dim,
        None,
        slice_meta,
    );

    // 5. view_copy(slice, original_shape) — reshape back
    let view2_meta = node_meta.cloned_with_dtype_of_or_float32(original_shape.clone(), &input_meta);
    graph.replace_with_view_copy(
        node_id,
        format!("{}_view_post", node_name),
        slice_id,
        &original_shape,
        view2_meta,
    );
    true
}
