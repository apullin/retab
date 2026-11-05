//! Port of rc4's DecomposeLinearPass: a Linear becomes a 1x1 conv with the rows in
//! the width.
//!
//! `aten.linear.default(x, weight, bias)` becomes
//!   1. view x:          [..., Ci] -> [1, 1, rows, Ci]
//!   2. permute(0,3,1,2) -> [1, Ci, 1, rows]
//!   3. view weight:     [Co, Ci] -> [Co, Ci, 1, 1]
//!   4. 1x1 conv        -> [1, Co, 1, rows]
//!   5. permute(0,2,3,1) -> [1, 1, rows, Co]
//!   6. view            -> [..., Co]
//!
//! In NHWC the rows are the width, so Vela never sees a batch of `rows`. The layout
//! pass turns both permutes into no-ops (see `fx_memory_format_ir` layout islands).

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

            let input_shape = match graph.node_shape_vec(input_id) {
                Some(s) => s,
                None => continue,
            };
            let weights_shape = match graph.node_shape_vec(weights_id) {
                Some(s) => s,
                None => continue,
            };
            let output_shape = match graph.node_shape_vec(id) {
                Some(s) => s,
                None => continue,
            };
            // Compute batch dimension: product of all dims except last.
            let ci = input_shape[input_shape.len() - 1];
            let batches: i64 = input_shape[..input_shape.len() - 1].iter().product();
            let co = weights_shape[0];

            let node_name = graph.nodes[id].name.clone();

            let input_rows_shape = [1, 1, batches, ci];
            let input_4d_shape = [1, ci, 1, batches];
            let input_base_meta = graph.nodes[input_id].meta.clone();
            let input_stamp = input_base_meta.tensor_stamp_same_dtype_or_float32();
            let input_rows_meta = input_stamp.empty_qparam_maps(input_rows_shape);
            let input_4d_meta = input_stamp.empty_qparam_maps(input_4d_shape);

            // 2. Reshape weights to 4D: [Co, Ci] -> [Co, Ci, 1, 1]
            //    Propagate weight qparams (including per-channel) to the reshaped node.
            let weights_4d_shape = [co, weights_shape[1], 1, 1];
            let weights_base_meta = graph.nodes[weights_id].meta.clone();
            let weights_4d_meta = weights_base_meta
                .tensor_stamp_same_dtype_or_float32()
                .empty_qparam_maps(weights_4d_shape);

            // 3. Conv2d with 1x1 kernel
            //    convolution(input, weight, bias, stride, padding, dilation,
            //                transposed, output_padding, groups)
            let conv_out_shape = [1, co, 1, batches];
            let output_rows_shape = [1, 1, batches, co];
            let output_base_meta = graph.nodes[id].meta.clone();
            let output_meta = output_base_meta.tensor_stamp_same_dtype_or_float32();
            let conv_meta = output_meta.preserve_qparams(conv_out_shape);
            let bias_arg = match bias_id {
                Some(nid) => IrArg::Node(nid),
                None => IrArg::None,
            };

            let mut cursor = graph.rewrite_cursor(id);

            let input_rows = cursor.view_copy(
                format!("{}_input_view", node_name),
                input_id,
                &input_rows_shape,
                input_rows_meta,
            );
            let input_reshaped = cursor.permute_copy(
                format!("{}_input_permute", node_name),
                input_rows,
                &[0, 3, 1, 2],
                input_4d_meta,
            );

            let weights_reshaped = cursor.view_copy(
                format!("{}_weight_view", node_name),
                weights_id,
                &weights_4d_shape,
                weights_4d_meta,
            );

            let conv = cursor.call_function(
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

            let output_rows = cursor.permute_copy(
                format!("{}_output_permute", node_name),
                conv,
                &[0, 2, 3, 1],
                output_meta.empty_qparam_maps(output_rows_shape),
            );
            let out_meta = output_meta.empty_qparam_maps(output_shape.clone());
            let result = cursor.view_copy(
                format!("{}_output_view", node_name),
                output_rows,
                &output_shape,
                out_meta,
            );
            cursor.replace_anchor_with(result);
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
