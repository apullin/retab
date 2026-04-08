//! Pure Rust IrPass version of FxConv1dUnsqueezePass.
//!
//! TOSA has no native 1D convolution. This pass rewrites conv1d into:
//!   view_copy(input[..., W] -> [..., W, 1])
//!   view_copy(weight[..., KW] -> [..., KW, 1])
//!   convolution(... stride=[s,1], pad=[p,0], dilation=[d,1])
//!   view_copy(output[..., W_out, 1] -> [..., W_out])

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

/// True for conv1d.default or convolution.default with 1D stride.
fn is_conv1d(graph: &IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    let target = &node.target;

    if target.contains("conv1d.default") {
        return true;
    }

    if target.contains("convolution.default") {
        // Check stride arg (index 3): if it's a list of length 1, it's 1D.
        if let Some(IrArg::List(items)) = node.args.get(3) {
            return items.len() == 1;
        }
    }

    false
}

/// Build a view_copy node that reshapes `input_id` to `new_shape`.
fn add_view_copy(
    graph: &mut IrGraph,
    name_prefix: &str,
    suffix: &str,
    input_id: NodeId,
    new_shape: &[i64],
    dtype: &str,
    base_meta: &NodeMeta,
) -> NodeId {
    let mut meta = base_meta.clone();
    meta.set_val(FakeTensorMeta {
        shape: new_shape.to_vec(),
        dtype: dtype.to_string(),
    });
    graph.add_node(
        NodeOp::CallFunction,
        format!("{}_{}", name_prefix, suffix),
        "aten.view_copy.default".to_string(),
        vec![IrArg::Node(input_id), i64_list(new_shape)],
        meta,
    )
}

/// Unsqueeze a conv1d node into conv2d form on the IrGraph.
fn unsqueeze_conv1d(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    let target = node.target.clone();
    let node_name = node.name.clone();
    let args = node.args.clone();
    let base_meta = node.meta.clone();

    let is_conv1d_default = target.contains("conv1d.default");

    // Parse arguments depending on variant.
    let (x_id, w_id, bias_arg, stride, pad, dilation, transposed, output_padding, groups);

    if is_conv1d_default {
        // conv1d.default: (input, weight, [bias], [stride], [padding], [dilation], [groups])
        if args.len() < 2 {
            return false;
        }
        x_id = match args[0].as_node() {
            Some(id) => id,
            None => return false,
        };
        w_id = match args[1].as_node() {
            Some(id) => id,
            None => return false,
        };
        bias_arg = match args.get(2) {
            Some(IrArg::None) | None => IrArg::None,
            Some(IrArg::Node(nid)) => IrArg::Node(*nid),
            _ => IrArg::None,
        };
        stride = args.get(3).and_then(|a| a.as_int_list()).unwrap_or_else(|| vec![1]);
        pad = args.get(4).and_then(|a| a.as_int_list()).unwrap_or_else(|| vec![0]);
        dilation = args.get(5).and_then(|a| a.as_int_list()).unwrap_or_else(|| vec![1]);
        groups = args.get(6).and_then(|a| a.as_int()).unwrap_or(1);
        transposed = false;
        output_padding = vec![0];
    } else {
        // convolution.default: 9 args
        //   (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
        if args.len() < 9 {
            return false;
        }
        x_id = match args[0].as_node() {
            Some(id) => id,
            None => return false,
        };
        w_id = match args[1].as_node() {
            Some(id) => id,
            None => return false,
        };
        bias_arg = match &args[2] {
            IrArg::None => IrArg::None,
            IrArg::Node(nid) => IrArg::Node(*nid),
            _ => IrArg::None,
        };
        stride = match args[3].as_int_list() {
            Some(v) => v,
            None => return false,
        };
        pad = match args[4].as_int_list() {
            Some(v) => v,
            None => return false,
        };
        dilation = match args[5].as_int_list() {
            Some(v) => v,
            None => return false,
        };
        transposed = args[6].as_bool().unwrap_or(false);
        output_padding = match args[7].as_int_list() {
            Some(v) => v,
            None => return false,
        };
        groups = match args[8].as_int() {
            Some(v) => v,
            None => return false,
        };
    }

    // Only handle 1D (stride length == 1).
    if stride.len() != 1 {
        return false;
    }

    // Get input and weight shapes.
    let x_shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let x_dtype = match graph.nodes[x_id].meta.val_dtype() {
        Some(d) => d.to_string(),
        None => return false,
    };
    let w_shape = match graph.nodes[w_id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };

    // Unsqueeze input: [..., W] -> [..., W, 1]
    let mut x_4d = x_shape.clone();
    x_4d.push(1);
    let x_unsqueezed = add_view_copy(
        graph,
        &node_name,
        "input_unsqueeze",
        x_id,
        &x_4d,
        &x_dtype,
        &base_meta,
    );

    // Unsqueeze weight: [..., KW] -> [..., KW, 1]
    let mut w_4d = w_shape.clone();
    w_4d.push(1);
    let w_unsqueezed = add_view_copy(
        graph,
        &node_name,
        "weight_unsqueeze",
        w_id,
        &w_4d,
        &x_dtype,
        &base_meta,
    );

    // Build 2D conv parameters: append trailing dimension.
    let mut stride_2d = stride;
    stride_2d.push(1);
    let mut pad_2d = pad;
    pad_2d.push(0);
    let mut dilation_2d = dilation;
    dilation_2d.push(1);
    let mut output_padding_2d = output_padding;
    output_padding_2d.push(0);

    // Compute conv2d output shape: take the original output shape and insert a trailing 1.
    let orig_out_shape = match graph.nodes[id].meta.val_shape() {
        Some(s) => s.to_vec(),
        None => return false,
    };
    let orig_out_dtype = match graph.nodes[id].meta.val_dtype() {
        Some(d) => d.to_string(),
        None => return false,
    };
    let mut conv2d_shape = orig_out_shape.clone();
    conv2d_shape.push(1);

    let mut conv2d_meta = base_meta.clone();
    conv2d_meta.set_val(FakeTensorMeta {
        shape: conv2d_shape.to_vec(),
        dtype: orig_out_dtype.clone(),
    });

    let conv2d = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_conv2d", node_name),
        "aten.convolution.default".to_string(),
        vec![
            IrArg::Node(x_unsqueezed),
            IrArg::Node(w_unsqueezed),
            bias_arg,
            i64_list(&stride_2d),
            i64_list(&pad_2d),
            i64_list(&dilation_2d),
            IrArg::Bool(transposed),
            i64_list(&output_padding_2d),
            IrArg::Int(groups),
        ],
        conv2d_meta,
    );
    graph.anchor_to(conv2d, id);

    // Squeeze output: [..., W_out, 1] -> [..., W_out]
    let squeezed = add_view_copy(
        graph,
        &node_name,
        "output_squeeze",
        conv2d,
        &orig_out_shape,
        &orig_out_dtype,
        &base_meta,
    );

    graph.replace_and_erase(id, squeezed);
    true
}

pub struct IrConv1dUnsqueezePass;

impl IrPass for IrConv1dUnsqueezePass {
    fn name(&self) -> &str {
        "IrConv1dUnsqueeze"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !is_conv1d(graph, id) {
                continue;
            }
            if unsqueeze_conv1d(graph, id) {
                modified = true;
            }
        }

        if modified {
            log_line(format_args!("[IrConv1dUnsqueeze] rewrote conv1d -> unsqueeze+conv2d+squeeze"));
        }
        modified
    }
}
