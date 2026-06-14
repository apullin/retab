//! Pure Rust IrPass version of FxConvertFullLikeToFullPass.
//!
//! Converts `aten.full_like.default` to `aten.full.default` so the TOSA
//! serializer can emit a CONST tensor.
//!
//!   full_like(x, fill_value) -> full(x.shape, fill_value)
//!
//! The meta (including FakeTensorMeta with shape/dtype) is cloned from the
//! original node, which preserves dtype information through the graph.

use crate::fx_ir::{IrArg, IrGraph, IrPass};
use crate::fx_target::is_full_like_target;
use crate::runner_logging::log_line;

pub struct IrConvertFullLikePass;

impl IrPass for IrConvertFullLikePass {
    fn name(&self) -> &str {
        "IrConvertFullLike"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if !is_full_like_target(&graph.nodes[id].target) {
                return false;
            }

            if graph.nodes[id].args.len() < 2 {
                return false;
            }

            // Extract input node ID and fill value from args
            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => return false,
            };
            let fill_value = graph.nodes[id].args[1].clone();

            // Get input shape from the input node's meta
            let input_shape = match graph.node_shape_vec(input_id) {
                Some(s) => s,
                None => return false,
            };

            // Build args: full(shape_list, fill_value)
            let shape_arg = IrArg::List(input_shape.iter().map(|&d| IrArg::Int(d)).collect());

            graph.replace_with_call_function(
                id,
                format!("{}_full", graph.nodes[id].name),
                "aten.full.default".to_string(),
                vec![shape_arg, fill_value],
                graph.nodes[id].meta.clone(),
            );
            true
        });

        if modified {
            log_line(format_args!(
                "[IrConvertFullLike] converted full_like -> full"
            ));
        }
        modified
    }
}
