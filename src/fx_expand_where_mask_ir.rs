//! Pure Rust IrPass version of FxExpandWhereMaskPass.
//!
//! Expands lower-rank boolean masks feeding `where` to the full output shape.
//! Materializes the broadcast explicitly as `view_copy + expand_copy` so that
//! downstream TOSA SELECT emission (which is stricter than PyTorch broadcasting)
//! gets identically-shaped operands.

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

fn is_where(target: &str) -> bool {
    target.contains("where.self") || target.contains("aten_where")
}

pub struct IrExpandWhereMaskPass;

impl IrPass for IrExpandWhereMaskPass {
    fn name(&self) -> &str {
        "IrExpandWhereMask"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !is_where(&graph.nodes[id].target) {
                continue;
            }
            if graph.nodes[id].args.len() < 3 {
                continue;
            }

            // arg[0] = mask, arg[1] = true_val, arg[2] = false_val
            let mask_id = match graph.nodes[id].args[0].as_node() {
                Some(mid) => mid,
                None => continue,
            };

            let output_shape = match graph.nodes[id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let mask_shape = match graph.nodes[mask_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };

            if output_shape.is_empty() || mask_shape.is_empty() {
                continue;
            }
            if mask_shape.len() > output_shape.len() {
                continue;
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
                continue;
            }

            // Build meta for the view_copy node (padded mask shape, same dtype as mask)
            let mask_dtype = graph.nodes[mask_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.bool")
                .to_string();
            let mut view_meta = graph.nodes[mask_id].meta.clone();
            view_meta.set_val(FakeTensorMeta {
                shape: padded_mask_shape.clone(),
                dtype: mask_dtype.clone(),
            });

            // Insert view_copy: reshape mask to padded shape
            let view_id = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_view_copy", graph.nodes[id].name),
                "aten.view_copy.default".to_string(),
                vec![
                    IrArg::Node(mask_id),
                    IrArg::List(padded_mask_shape.iter().map(|&d| IrArg::Int(d)).collect()),
                ],
                view_meta,
            );
            // Anchor before the original where node so topological order is preserved
            graph.anchor_to(view_id, id);

            // Build meta for the expand_copy node (full output shape, same dtype)
            let mut expand_meta = graph.nodes[mask_id].meta.clone();
            expand_meta.set_val(FakeTensorMeta {
                shape: output_shape.clone(),
                dtype: mask_dtype,
            });

            // Insert expand_copy: broadcast to full output shape
            let expand_id = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_expand_copy", graph.nodes[id].name),
                "aten.expand_copy.default".to_string(),
                vec![
                    IrArg::Node(view_id),
                    IrArg::List(output_shape.iter().map(|&d| IrArg::Int(d)).collect()),
                ],
                expand_meta,
            );
            // Anchor before the original where node so topological order is preserved
            graph.anchor_to(expand_id, id);

            // Clone the remaining args (true_val, false_val) from the original where node
            let true_arg = graph.nodes[id].args[1].clone();
            let false_arg = graph.nodes[id].args[2].clone();

            // Insert new where.self with the expanded mask
            let new_where_id = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_where", graph.nodes[id].name),
                "aten.where.self".to_string(),
                vec![IrArg::Node(expand_id), true_arg, false_arg],
                graph.nodes[id].meta.clone(),
            );
            graph.anchor_to(new_where_id, id);

            graph.replace_and_erase(id, new_where_id);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrExpandWhereMask] expanded where masks to match output shape"
            ));
        }
        modified
    }
}
