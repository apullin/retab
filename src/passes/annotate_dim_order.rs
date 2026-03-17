use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;
use smallvec::SmallVec;

/// Set `tosa_dim_order` metadata on output nodes based on their rank.
/// - 4D tensors get [0,1,2,3] (NCHW identity; the memory_format pass changes
///   this later for NHWC ops).
/// - Lower-dimensional tensors get an identity permutation of appropriate length.
pub struct AnnotateOutputDimOrderPass;

impl Pass for AnnotateOutputDimOrderPass {
    fn name(&self) -> &str {
        "AnnotateOutputDimOrder"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let node = match graph.node(node_id) {
                Some(n) => n,
                None => continue,
            };

            // Skip nodes that already have a dim order set
            if node.meta.tosa_dim_order.is_some() {
                continue;
            }

            let rank = node.meta.val.shape_len as usize;
            if rank == 0 {
                continue;
            }

            // Identity permutation
            let dim_order: SmallVec<[u8; 6]> = (0..rank as u8).collect();

            if let Some(node) = graph.node_mut(node_id) {
                node.meta.tosa_dim_order = Some(dim_order);
                modified = true;
            }
        }

        modified
    }
}
