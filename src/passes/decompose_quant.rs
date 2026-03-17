use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;

/// Remove any remaining QuantizePerTensor / DequantizePerTensor /
/// DequantizePerChannel nodes that were not handled by the fold_qdq pass
/// (edge cases, e.g., around graph inputs/outputs).
pub struct DecomposeQuantNodesPass;

impl Pass for DecomposeQuantNodesPass {
    fn name(&self) -> &str {
        "DecomposeQuantNodes"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let target = match graph.node(node_id) {
                Some(n) => n.target.clone(),
                None => continue,
            };

            match target {
                OpTarget::QuantizePerTensor
                | OpTarget::DequantizePerTensor
                | OpTarget::DequantizePerChannel => {
                    graph.bypass_node(node_id);
                    modified = true;
                }
                _ => {}
            }
        }

        modified
    }
}
