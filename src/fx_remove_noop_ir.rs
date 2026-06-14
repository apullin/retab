//! Pure Rust IrPass version of FxRemoveNoopPass.
//!
//! Removes identity-like nodes from the graph: clone, _clone_dim_order,
//! _to_dim_order_copy, _to_copy, alias_copy, copy, and detach_copy where
//! input and output dtypes match.
//!
//! For `aten.copy.default`, the pass returns args[1] (the source) since
//! copy(dst, src) copies src into dst. For all others, args[0] is the
//! pass-through value.

use crate::fx_ir::{IrGraph, IrPass, NodeId};
use crate::fx_target::{noop_target, NoopTarget};
use crate::runner_logging::log_line;

pub struct IrRemoveNoopPass;

impl IrPass for IrRemoveNoopPass {
    fn name(&self) -> &str {
        "IrRemoveNoop"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let Some(noop) = noop_target(&graph.nodes[id].target) else {
                return false;
            };

            if graph.nodes[id].args.is_empty() {
                return false;
            }

            // The input node is always args[0] (used for dtype check).
            let input_id: NodeId = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => return false,
            };

            // Determine the replacement: copy uses args[1] (src), others use args[0]
            let replacement_id: NodeId = match noop {
                NoopTarget::CopySourceArg => match graph.nodes[id].arg_node(1) {
                    Some(nid) => nid,
                    None => return false,
                },
                NoopTarget::PassthroughFirstArg | NoopTarget::DtypeCheckedPassthrough => input_id,
            };

            // Check dtype match: input dtype must equal output dtype.
            let input_dtype = graph.node_dtype(input_id);
            if input_dtype.is_none() {
                return false;
            }
            if noop == NoopTarget::DtypeCheckedPassthrough {
                let output_dtype = graph.node_dtype(id);
                if input_dtype != output_dtype {
                    return false;
                }
            }

            log_line(format_args!(
                "[IrRemoveNoop] removing '{}' (target={})",
                graph.nodes[id].name, graph.nodes[id].target
            ));

            graph.replace_and_erase(id, replacement_id);
            true
        });

        if modified {
            log_line(format_args!("[IrRemoveNoop] removed noop identity nodes"));
        }
        modified
    }
}
