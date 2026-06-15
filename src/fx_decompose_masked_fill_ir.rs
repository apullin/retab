//! Pure Rust IrPass version of FxDecomposeMaskedFillPass.
//!
//! Decomposes `masked_fill` into `full_like(x, value)` + `where(mask, fill, x)`.
//!
//! Runs AFTER `FxFoldAndAnnotateQParamsPass` so that:
//!   - The `masked_fill` node already has `output_qparams`.
//!   - The fill value scalar can be quantized to match the tensor data type.
//!   - The created `full_like` constant is quantized at the same scale/zp as `x`.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeMeta};
use crate::fx_target::is_masked_fill_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeMaskedFillPass;

fn tensorized_scalar_meta(meta: &NodeMeta) -> NodeMeta {
    let mut new_meta = meta.clone();
    new_meta.set_empty_input_qparams();
    new_meta
}

fn where_meta(meta: &NodeMeta) -> NodeMeta {
    let mut new_meta = meta.clone();
    match meta.output_qparam().cloned() {
        Some(qp) => new_meta.set_input_qparams_from([(1, qp.clone()), (2, qp)]),
        None => new_meta.set_empty_input_qparams(),
    }
    new_meta
}

impl IrPass for IrDecomposeMaskedFillPass {
    fn name(&self) -> &str {
        "IrDecomposeMaskedFill"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if !is_masked_fill_target(&graph.nodes[id].target) {
                return false;
            }
            if graph.nodes[id].args.len() < 3 {
                return false;
            }

            // Extract args: x (input tensor), mask, value (scalar)
            let x_id = match &graph.nodes[id].args[0] {
                IrArg::Node(nid) => *nid,
                _ => return false,
            };
            let mask_id = match &graph.nodes[id].args[1] {
                IrArg::Node(nid) => *nid,
                _ => return false,
            };
            let value_arg = graph.nodes[id].args[2].clone();

            let node_name = graph.nodes[id].name.clone();
            let node_meta = graph.nodes[id].meta.clone();
            let mut cursor = graph.rewrite_cursor(id);

            // Create full_like(x, value) — fills a tensor shaped like x with the scalar value
            let full_like_id = cursor.call_function(
                format!("{node_name}_full_like"),
                "aten.full_like.default",
                vec![IrArg::Node(x_id), value_arg],
                tensorized_scalar_meta(&node_meta),
            );

            // Create where(mask, fill, x) — True positions get fill_value, False keep x
            let where_id = cursor.call_function(
                format!("{node_name}_where"),
                "aten.where.self",
                vec![
                    IrArg::Node(mask_id),
                    IrArg::Node(full_like_id),
                    IrArg::Node(x_id),
                ],
                where_meta(&node_meta),
            );
            cursor.replace_anchor_with(where_id);
            true
        });

        if modified {
            log_line(format_args!(
                "[IrDecomposeMaskedFill] decomposed masked_fill ops"
            ));
        }
        modified
    }
}
