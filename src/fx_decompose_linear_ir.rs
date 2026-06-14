//! Pure Rust IrPass version of FxDecomposeLinearPass.
//!
//! Decomposes `aten.linear.default(x, weight, bias)` into:
//!   1. Reshape input to 4D:  [..., Ci] -> [N, Ci, 1, 1]
//!   2. Reshape weight to 4D: [Co, Ci] -> [Co, Ci, 1, 1]
//!   3. Conv2d with 1x1 kernel (aten.convolution.default)
//!   4. Reshape output back:  [N, Co, 1, 1] -> [..., Co]

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId};
use crate::runner_logging::log_line;

pub struct IrDecomposeLinearPass;

impl IrPass for IrDecomposeLinearPass {
    fn name(&self) -> &str {
        "IrDecomposeLinear"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !graph.nodes[id].is_linear() {
                continue;
            }

            if graph.nodes[id].args.len() < 2 {
                continue;
            }

            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => continue,
            };
            let weights_id = match graph.nodes[id].args[1].as_node() {
                Some(nid) => nid,
                None => continue,
            };
            let bias_id: Option<NodeId> = if graph.nodes[id].args.len() > 2 {
                match &graph.nodes[id].args[2] {
                    IrArg::None => None,
                    IrArg::Node(nid) => Some(*nid),
                    _ => None,
                }
            } else {
                None
            };

            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let weights_shape = match graph.nodes[weights_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let output_shape = match graph.nodes[id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let output_dtype = graph.nodes[id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();
            let input_dtype = graph.nodes[input_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();
            let weight_dtype = graph.nodes[weights_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();

            // Compute batch dimension: product of all dims except last.
            let ci = input_shape[input_shape.len() - 1];
            let batches: i64 = input_shape[..input_shape.len() - 1].iter().product();
            let co = weights_shape[0];

            let node_name = graph.nodes[id].name.clone();

            // 1. Reshape input to 4D: [..., Ci] -> [batches, Ci, 1, 1]
            //    Propagate input qparams to the reshaped node.
            let input_4d_shape = [batches, ci, 1, 1];
            let mut input_4d_meta = graph.nodes[input_id]
                .meta
                .cloned_with_val(input_4d_shape.to_vec(), input_dtype);
            input_4d_meta.set_empty_qparam_maps();
            let input_reshaped = graph.add_anchored_call_function(
                id,
                format!("{}_input_view", node_name),
                "aten.view_copy.default",
                vec![IrArg::Node(input_id), i64_list(&input_4d_shape)],
                input_4d_meta,
            );

            // 2. Reshape weights to 4D: [Co, Ci] -> [Co, Ci, 1, 1]
            //    Propagate weight qparams (including per-channel) to the reshaped node.
            let weights_4d_shape = [co, weights_shape[1], 1, 1];
            let mut weights_4d_meta = graph.nodes[weights_id]
                .meta
                .cloned_with_val(weights_4d_shape.to_vec(), weight_dtype);
            weights_4d_meta.set_empty_qparam_maps();
            let weights_reshaped = graph.add_anchored_call_function(
                id,
                format!("{}_weight_view", node_name),
                "aten.view_copy.default",
                vec![IrArg::Node(weights_id), i64_list(&weights_4d_shape)],
                weights_4d_meta,
            );

            // 3. Conv2d with 1x1 kernel
            //    convolution(input, weight, bias, stride, padding, dilation,
            //                transposed, output_padding, groups)
            let conv_out_shape = [batches, co, 1, 1];
            let conv_meta = graph.nodes[id]
                .meta
                .cloned_with_val(conv_out_shape.to_vec(), &output_dtype)
                .with_qparams_from(&graph.nodes[id].meta);
            let bias_arg = match bias_id {
                Some(nid) => IrArg::Node(nid),
                None => IrArg::None,
            };
            let conv = graph.add_anchored_call_function(
                id,
                format!("{}_conv", node_name),
                "aten.convolution.default",
                vec![
                    IrArg::Node(input_reshaped),
                    IrArg::Node(weights_reshaped),
                    bias_arg,
                    i64_list(&[1, 1]),  // stride
                    i64_list(&[0, 0]),  // padding
                    i64_list(&[1, 1]),  // dilation
                    IrArg::Bool(false), // transposed
                    i64_list(&[0, 0]),  // output_padding
                    IrArg::Int(1),      // groups
                ],
                conv_meta,
            );

            // 4. Reshape output back to original shape
            let mut out_meta = graph.nodes[id]
                .meta
                .cloned_with_val(output_shape.clone(), output_dtype);
            out_meta.set_empty_qparam_maps();
            graph.replace_with_call_function(
                id,
                format!("{}_output_view", node_name),
                "aten.view_copy.default",
                vec![IrArg::Node(conv), i64_list(&output_shape)],
                out_meta,
            );
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeLinear] decomposed linear ops to conv2d"
            ));
        }
        modified
    }
}
