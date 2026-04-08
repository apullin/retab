//! Pure Rust IrPass version of FxConvertFullLikeToFullPass.
//!
//! Converts `aten.full_like.default` to `aten.full.default` so the TOSA
//! serializer can emit a CONST tensor.
//!
//!   full_like(x, fill_value) -> full(x.shape, fill_value)
//!
//! The meta (including FakeTensorMeta with shape/dtype) is cloned from the
//! original node, which preserves dtype information through the graph.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

pub struct IrConvertFullLikePass;

impl IrPass for IrConvertFullLikePass {
    fn name(&self) -> &str {
        "IrConvertFullLike"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }

            if !graph.nodes[id].target.contains("full_like") {
                continue;
            }

            if graph.nodes[id].args.len() < 2 {
                continue;
            }

            // Extract input node ID and fill value from args
            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => continue,
            };
            let fill_value = graph.nodes[id].args[1].clone();

            // Get input shape from the input node's meta
            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };

            // Build args: full(shape_list, fill_value)
            let shape_arg = IrArg::List(
                input_shape.iter().map(|&d| IrArg::Int(d)).collect(),
            );

            let new_id = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_full", graph.nodes[id].name),
                "aten.full.default".to_string(),
                vec![shape_arg, fill_value],
                graph.nodes[id].meta.clone(),
            );
            graph.replace_and_erase(id, new_id);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrConvertFullLike] converted full_like -> full"
            ));
        }
        modified
    }
}
