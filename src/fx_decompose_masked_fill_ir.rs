//! Pure Rust IrPass version of FxDecomposeMaskedFillPass.
//!
//! Decomposes `masked_fill` into `full_like(x, value)` + `where(mask, fill, x)`.
//!
//! Runs AFTER `FxFoldAndAnnotateQParamsPass` so that:
//!   - The `masked_fill` node already has `output_qparams`.
//!   - The fill value scalar can be quantized to match the tensor data type.
//!   - The created `full_like` constant is quantized at the same scale/zp as `x`.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

pub struct IrDecomposeMaskedFillPass;

impl IrPass for IrDecomposeMaskedFillPass {
    fn name(&self) -> &str {
        "IrDecomposeMaskedFill"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("masked_fill") {
                continue;
            }
            if graph.nodes[id].args.len() < 3 {
                continue;
            }

            // Extract args: x (input tensor), mask, value (scalar)
            let x_id = match &graph.nodes[id].args[0] {
                IrArg::Node(nid) => *nid,
                _ => continue,
            };
            let mask_id = match &graph.nodes[id].args[1] {
                IrArg::Node(nid) => *nid,
                _ => continue,
            };
            let value_arg = graph.nodes[id].args[2].clone();

            let node_name = graph.nodes[id].name.clone();
            let node_meta = graph.nodes[id].meta.clone();

            // Create full_like(x, value) — fills a tensor shaped like x with the scalar value
            let full_like_id = graph.add_node(
                NodeOp::CallFunction,
                format!("{node_name}_full_like"),
                "aten.full_like.default".to_string(),
                vec![IrArg::Node(x_id), value_arg],
                node_meta.clone(),
            );
            graph.anchor_to(full_like_id, id);

            // Create where(mask, fill, x) — True positions get fill_value, False keep x
            let where_id = graph.add_node(
                NodeOp::CallFunction,
                format!("{node_name}_where"),
                "aten.where.self".to_string(),
                vec![
                    IrArg::Node(mask_id),
                    IrArg::Node(full_like_id),
                    IrArg::Node(x_id),
                ],
                node_meta,
            );

            graph.replace_and_erase(id, where_id);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeMaskedFill] decomposed masked_fill ops"
            ));
        }
        modified
    }
}
