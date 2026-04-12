//! Pure Rust IrPass version of FxDecomposeCumsumPass.
//!
//! Decomposes cumsum into: view_copy -> full -> convolution -> slice_copy -> view_copy.
//! This mirrors the Arm backend's cumsum lowering: reshape to 4D, convolve with
//! an all-ones kernel (cumulative sum via convolution), slice off padding, reshape back.

use std::collections::HashMap;

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp, QParam};
use crate::runner_logging::log_line;

pub struct IrDecomposeCumsumPass;

impl IrPass for IrDecomposeCumsumPass {
    fn name(&self) -> &str {
        "IrDecomposeCumsum"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("cumsum") {
                continue;
            }

            if decompose_cumsum(graph, id) {
                modified = true;
            }
        }

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

    let original_shape = match graph.nodes[input_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    if original_shape.is_empty() {
        return false;
    }

    let input_dtype = graph.nodes[input_id]
        .meta
        .val_dtype()
        .unwrap_or("torch.float32")
        .to_string();

    let rank = original_shape.len() as i64;
    let dim = dim_raw.rem_euclid(rank) as usize;

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
    let view1_meta = make_meta(&conv_shape, &input_dtype);
    let view1_id = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_view_pre", node_name),
        "aten.view_copy.default".to_string(),
        vec![
            IrArg::Node(input_id),
            IrArg::List(conv_shape.iter().map(|&d| IrArg::Int(d)).collect()),
        ],
        view1_meta,
    );
    graph.anchor_to(view1_id, node_id);

    // 2. full(weight_shape, 1.0) — all-ones convolution kernel
    let weight_meta = make_meta(&weight_shape, &input_dtype);
    let weight_id = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_weight", node_name),
        "aten.full.default".to_string(),
        vec![
            IrArg::List(weight_shape.iter().map(|&d| IrArg::Int(d)).collect()),
            IrArg::Float(1.0),
        ],
        weight_meta,
    );
    graph.anchor_to(weight_id, node_id);

    // 3. convolution(view1, weight, None, [1,1], [cumsum_dim-1, 0], [1,1], false, [0], 1)
    //    The padding on dim 2 causes the conv to compute prefix sums.
    //    Output shape: [pre, 1, cumsum_dim + (cumsum_dim-1), post] = [pre, 1, 2*cumsum_dim-1, post]
    let conv_out_h = cumsum_dim + (cumsum_dim - 1); // padded dim
    let conv_out_shape = vec![pre_cumsum_dim, 1, conv_out_h, post_cumsum_dim];
    let mut conv_meta = make_meta(&conv_out_shape, &input_dtype);

    // Quantization fixup: propagate input qparams, add weight qparam at index 1
    if let Some(ref orig_iqp) = node_meta.input_qparams {
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

            // Set weight output qparam
            let mut weight_oqp = HashMap::new();
            weight_oqp.insert(
                0,
                QParam::per_tensor(1.0, 0, -128, 127, "torch.int8".into()),
            );
            graph
                .node_mut(weight_id)
                .meta
                .set_output_qparams(weight_oqp);
        }
    }

    let conv_id = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_conv", node_name),
        "aten.convolution.default".to_string(),
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
    graph.anchor_to(conv_id, node_id);

    // 4. slice_copy(conv, dim=2, start=0, end=cumsum_dim) — trim padding
    let slice_shape = vec![pre_cumsum_dim, 1, cumsum_dim, post_cumsum_dim];
    let slice_meta = make_meta(&slice_shape, &input_dtype);
    let slice_id = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_slice", node_name),
        "aten.slice_copy.Tensor".to_string(),
        vec![
            IrArg::Node(conv_id),
            IrArg::Int(2),          // dim
            IrArg::Int(0),          // start
            IrArg::Int(cumsum_dim), // end
        ],
        slice_meta,
    );
    graph.anchor_to(slice_id, node_id);

    // 5. view_copy(slice, original_shape) — reshape back
    let mut view2_meta = node_meta;
    view2_meta.set_val(FakeTensorMeta {
        shape: original_shape.clone(),
        dtype: input_dtype,
    });
    let view2_id = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_view_post", node_name),
        "aten.view_copy.default".to_string(),
        vec![
            IrArg::Node(slice_id),
            IrArg::List(original_shape.iter().map(|&d| IrArg::Int(d)).collect()),
        ],
        view2_meta,
    );

    graph.replace_and_erase(node_id, view2_id);
    true
}

/// Build a minimal NodeMeta with shape and dtype set.
fn make_meta(shape: &[i64], dtype: &str) -> NodeMeta {
    let mut meta = NodeMeta::default();
    meta.set_val(FakeTensorMeta {
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
    });
    meta
}
