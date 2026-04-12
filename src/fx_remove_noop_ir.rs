//! Pure Rust IrPass version of FxRemoveNoopPass.
//!
//! Removes identity-like nodes from the graph: clone, _clone_dim_order,
//! _to_dim_order_copy, _to_copy, alias_copy, copy, and detach_copy where
//! input and output dtypes match.
//!
//! For `aten.copy.default`, the pass returns args[1] (the source) since
//! copy(dst, src) copies src into dst. For all others, args[0] is the
//! pass-through value.

use crate::fx_ir::{IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

/// Target strings that are candidates for noop removal.
const NOOP_TARGETS: &[&str] = &[
    "aten.clone.default",
    "_clone_dim_order.default",
    "_to_dim_order_copy.default",
    "aten._to_copy.default",
    "alias_copy.default",
    "copy.default",
    "detach_copy.default",
];

fn is_noop_target(target: &str) -> bool {
    NOOP_TARGETS.iter().any(|t| target.contains(t))
}

fn is_copy_target(target: &str) -> bool {
    // Specifically aten.copy.default (not alias_copy or detach_copy)
    target.contains("copy.default")
        && !target.contains("alias_copy")
        && !target.contains("detach_copy")
        && !target.contains("dim_order_copy")
}

pub struct IrRemoveNoopPass;

impl IrPass for IrRemoveNoopPass {
    fn name(&self) -> &str {
        "IrRemoveNoop"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }

            let target = &graph.nodes[id].target;
            if !is_noop_target(target) {
                continue;
            }

            if graph.nodes[id].args.is_empty() {
                continue;
            }

            // The input node is always args[0] (used for dtype check).
            let input_id: NodeId = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => continue,
            };

            // Determine the replacement: copy uses args[1] (src), others use args[0]
            let replacement_id: NodeId = if is_copy_target(&graph.nodes[id].target) {
                match graph.nodes[id].args.get(1).and_then(|a| a.as_node()) {
                    Some(nid) => nid,
                    None => continue,
                }
            } else {
                input_id
            };

            // Check dtype match: input dtype must equal output dtype.
            let input_dtype = graph.nodes[input_id].meta.val_dtype();
            if input_dtype.is_none() {
                continue;
            }
            if graph.nodes[id].target.contains("_to_dim_order_copy")
                || graph.nodes[id].target.contains("aten._to_copy")
            {
                let output_dtype = graph.nodes[id].meta.val_dtype();
                if input_dtype != output_dtype {
                    continue;
                }
            }

            log_line(format_args!(
                "[IrRemoveNoop] removing '{}' (target={})",
                graph.nodes[id].name, graph.nodes[id].target
            ));

            graph.replace_and_erase(id, replacement_id);
            modified = true;
        }

        if modified {
            log_line(format_args!("[IrRemoveNoop] removed noop identity nodes"));
        }
        modified
    }
}
