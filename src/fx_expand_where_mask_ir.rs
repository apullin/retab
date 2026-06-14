//! Pure Rust IrPass version of FxExpandWhereMaskPass.
//!
//! Expands lower-rank boolean masks feeding `where` to the full output shape.
//! Materializes the broadcast explicitly as `view_copy + expand_copy` so that
//! downstream TOSA SELECT emission (which is stricter than PyTorch broadcasting)
//! gets identically-shaped operands.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::fx_target::is_where_target;
use crate::runner_logging::log_line;

pub struct IrExpandWhereMaskPass;

impl IrPass for IrExpandWhereMaskPass {
    fn name(&self) -> &str {
        "IrExpandWhereMask"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if !is_where_target(&graph.nodes[id].target) {
                return false;
            }
            if graph.nodes[id].args.len() < 3 {
                return false;
            }

            // arg[0] = mask, arg[1] = true_val, arg[2] = false_val
            let mask_id = match graph.nodes[id].args[0].as_node() {
                Some(mid) => mid,
                None => return false,
            };

            let output_shape = match graph.nodes[id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => return false,
            };
            let mask_shape = match graph.nodes[mask_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => return false,
            };

            if output_shape.is_empty() || mask_shape.is_empty() {
                return false;
            }
            if mask_shape.len() > output_shape.len() {
                return false;
            }

            // Pad mask shape with leading 1s to match output rank
            let mut padded_mask_shape = vec![1i64; output_shape.len() - mask_shape.len()];
            padded_mask_shape.extend(mask_shape.iter().copied());

            // Check broadcastability
            let broadcastable = padded_mask_shape
                .iter()
                .zip(output_shape.iter())
                .all(|(mask_dim, out_dim)| *mask_dim == 1 || *mask_dim == *out_dim);
            if !broadcastable || padded_mask_shape == output_shape {
                return false;
            }

            // Build meta for the view_copy node (padded mask shape, same dtype as mask)
            let mask_dtype = graph.nodes[mask_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.bool")
                .to_string();
            let view_meta = graph.nodes[mask_id]
                .meta
                .cloned_with_val(padded_mask_shape.clone(), &mask_dtype);

            // Insert view_copy: reshape mask to padded shape
            let view_id = graph.add_anchored_call_function(
                id,
                format!("{}_view_copy", graph.nodes[id].name),
                "aten.view_copy.default",
                vec![
                    IrArg::Node(mask_id),
                    IrArg::List(padded_mask_shape.iter().map(|&d| IrArg::Int(d)).collect()),
                ],
                view_meta,
            );

            // Build meta for the expand_copy node (full output shape, same dtype)
            let expand_meta = graph.nodes[mask_id]
                .meta
                .cloned_with_val(output_shape.clone(), mask_dtype);

            // Insert expand_copy: broadcast to full output shape
            let expand_id = graph.add_anchored_call_function(
                id,
                format!("{}_expand_copy", graph.nodes[id].name),
                "aten.expand_copy.default",
                vec![
                    IrArg::Node(view_id),
                    IrArg::List(output_shape.iter().map(|&d| IrArg::Int(d)).collect()),
                ],
                expand_meta,
            );

            // Clone the remaining args (true_val, false_val) from the original where node
            let true_arg = graph.nodes[id].args[1].clone();
            let false_arg = graph.nodes[id].args[2].clone();

            // Insert new where.self with the expanded mask
            let new_where_id = graph.add_anchored_call_function(
                id,
                format!("{}_where", graph.nodes[id].name),
                "aten.where.self",
                vec![IrArg::Node(expand_id), true_arg, false_arg],
                graph.nodes[id].meta.clone(),
            );

            graph.replace_and_erase(id, new_where_id);
            true
        });

        if modified {
            log_line(format_args!(
                "[IrExpandWhereMask] expanded where masks to match output shape"
            ));
        }
        modified
    }
}
