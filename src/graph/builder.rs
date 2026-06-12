use super::Graph;
use crate::graph::ir::*;
#[cfg(test)]
use smallvec::SmallVec;

impl Graph {
    // -----------------------------------------------------------------------
    // Node creation
    // -----------------------------------------------------------------------

    /// Create a new node and append it to the end of the graph order.
    #[cfg(test)]
    pub fn create_node(
        &mut self,
        op: OpType,
        target: OpTarget,
        args: SmallVec<[Arg; 4]>,
        kwargs: Vec<(String, Arg)>,
        meta: NodeMeta,
        name: String,
    ) -> NodeId {
        let node = Node {
            op,
            target,
            args,
            kwargs,
            meta,
            users: SmallVec::new(),
            name: name.clone(),
        };
        let id = self.nodes.insert(node);
        self.order.push(id);
        self.node_names.push((id, name));

        // Register this node as a user of all its inputs.
        let input_ids = Self::collect_input_ids(&self.nodes[id]);
        for input_id in input_ids {
            if let Some(input_node) = self.nodes.get_mut(input_id) {
                if !input_node.users.contains(&id) {
                    input_node.users.push(id);
                }
            }
        }

        id
    }

    /// Return all node IDs in insertion order.
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.order.clone()
    }
}
