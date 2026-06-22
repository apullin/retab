//! Pure Rust IrPass version of FxConv1dUnsqueezePass.
//!
//! TOSA has no native 1D convolution. This pass rewrites conv1d into:
//!   view_copy(input[..., W] -> [..., W, 1])
//!   view_copy(weight[..., KW] -> [..., KW, 1])
//!   convolution(... stride=[s,1], pad=[p,0], dilation=[d,1])
//!   view_copy(output[..., W_out, 1] -> [..., W_out])

use crate::fx_conv_args::ConvArgs;
use crate::fx_ir::{
    i64_list_arg as i64_list, IrArg, IrGraph, IrPass, IrRewriteCursor, NodeId, NodeMeta,
};
use crate::runner_logging::log_line;

/// True for conv1d.default or convolution.default with 1D stride.
fn is_conv1d(graph: &IrGraph, id: NodeId) -> bool {
    ConvArgs::parse(&graph.nodes[id]).is_some_and(|args| args.is_conv1d_form())
}

/// Build a view_copy node that reshapes `input_id` to `new_shape`.
fn add_view_copy(
    cursor: &mut IrRewriteCursor<'_>,
    name_prefix: &str,
    suffix: &str,
    base_meta: &NodeMeta,
    input_id: NodeId,
    new_shape: &[i64],
) -> NodeId {
    let meta = base_meta
        .tensor_stamp_same_dtype_or_float32()
        .empty_qparam_maps(new_shape.to_vec());
    cursor.view_copy(
        format!("{}_{}", name_prefix, suffix),
        input_id,
        new_shape,
        meta,
    )
}

/// Unsqueeze a conv1d node into conv2d form on the IrGraph.
fn unsqueeze_conv1d(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    let node_name = node.name.clone();
    let base_meta = node.meta.clone();
    let Some(conv) = ConvArgs::parse(node) else {
        return false;
    };
    if !conv.is_conv1d_form() {
        return false;
    }

    // Get input and weight shapes.
    let x_shape = match graph.node_shape_vec(conv.input) {
        Some(s) => s,
        None => return false,
    };
    let x_meta = match graph.node_dtype(conv.input) {
        Some(_) => graph.nodes[conv.input].meta.clone(),
        None => return false,
    };
    let w_meta = match graph.node_dtype(conv.weight) {
        Some(_) => graph.nodes[conv.weight].meta.clone(),
        None => return false,
    };
    let w_shape = match graph.node_shape_vec(conv.weight) {
        Some(s) => s,
        None => return false,
    };

    // Build 2D conv parameters: append trailing dimension.
    let mut stride_2d = conv.stride.clone();
    stride_2d.push(1);
    let mut pad_2d = conv.padding.clone();
    pad_2d.push(0);
    let mut dilation_2d = conv.dilation.clone();
    dilation_2d.push(1);
    let mut output_padding_2d = conv.output_padding.clone();
    output_padding_2d.push(0);

    // Compute conv2d output shape: take the original output shape and insert a trailing 1.
    let orig_out_shape = match graph.node_shape_vec(id) {
        Some(s) => s,
        None => return false,
    };
    if graph.node_dtype(id).is_none() {
        return false;
    }
    let mut conv2d_shape = orig_out_shape.clone();
    conv2d_shape.push(1);

    let conv2d_meta = base_meta.cloned_with_same_dtype_or_float32(conv2d_shape.to_vec());
    let mut cursor = graph.rewrite_cursor(id);

    // Unsqueeze input: [..., W] -> [..., W, 1]
    let mut x_4d = x_shape.clone();
    x_4d.push(1);
    let x_unsqueezed = add_view_copy(
        &mut cursor,
        &node_name,
        "input_unsqueeze",
        &x_meta,
        conv.input,
        &x_4d,
    );

    // Unsqueeze weight: [..., KW] -> [..., KW, 1]
    let mut w_4d = w_shape.clone();
    w_4d.push(1);
    let w_unsqueezed = add_view_copy(
        &mut cursor,
        &node_name,
        "weight_unsqueeze",
        &w_meta,
        conv.weight,
        &w_4d,
    );

    let conv2d = cursor.call_function(
        format!("{}_conv2d", node_name),
        "aten.convolution.default",
        vec![
            IrArg::Node(x_unsqueezed),
            IrArg::Node(w_unsqueezed),
            conv.bias_arg,
            i64_list(&stride_2d),
            i64_list(&pad_2d),
            i64_list(&dilation_2d),
            IrArg::Bool(conv.transposed),
            i64_list(&output_padding_2d),
            IrArg::Int(conv.groups),
        ],
        conv2d_meta,
    );

    // Squeeze output: [..., W_out, 1] -> [..., W_out]
    let squeezed = add_view_copy(
        &mut cursor,
        &node_name,
        "output_squeeze",
        &base_meta,
        conv2d,
        &orig_out_shape,
    );

    cursor.replace_anchor_with(squeezed);
    true
}

pub struct IrConv1dUnsqueezePass;

impl IrPass for IrConv1dUnsqueezePass {
    fn name(&self) -> &str {
        "IrConv1dUnsqueeze"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            is_conv1d(graph, id) && unsqueeze_conv1d(graph, id)
        });

        if modified {
            log_line(format_args!(
                "[IrConv1dUnsqueeze] rewrote conv1d -> unsqueeze+conv2d+squeeze"
            ));
        }
        modified
    }
}
