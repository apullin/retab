pub mod ir;

use ir::{Node, NodeId};
use slotmap::SlotMap;
#[cfg(test)]
use smallvec::SmallVec;

use crate::pipeline_config::LayoutMode;

/// The core graph IR. Nodes live in a SlotMap arena and are referenced by
/// stable `NodeId` keys. The graph tracks insertion order via `order`, input
/// placeholders via `inputs`, and the single output node via `output`.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: SlotMap<NodeId, Node>,
    /// Insertion-order list of all node IDs.
    pub order: Vec<NodeId>,
    /// Placeholder node IDs that represent graph inputs.
    pub inputs: Vec<NodeId>,
    /// The output node, if set.
    pub output: Option<NodeId>,
    /// Human-readable names for debugging.
    pub node_names: Vec<(NodeId, String)>,
    /// The layout model the graph was lowered with. Ingest sets it from the stashed
    /// pipeline config, before finalize, for the parent and every region. In `Explicit`
    /// mode a node's val is its TOSA shape, so ingest and the serializer switch off
    /// their implicit-layout logic (APU-135 P2 design §4.2).
    pub layout: LayoutMode,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            order: Vec::new(),
            inputs: Vec::new(),
            output: None,
            node_names: Vec::new(),
            layout: LayoutMode::DimOrder,
        }
    }

    /// Whether the graph was lowered in explicit layout mode.
    pub fn is_explicit_layout(&self) -> bool {
        self.layout == LayoutMode::Explicit
    }

    /// Get a reference to a node by its ID.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get a mutable reference to a node by its ID.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    /// Return all node IDs in insertion order.
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.order.clone()
    }

    /// Create a new node and append it to the end of the graph order.
    #[cfg(test)]
    pub fn create_node(
        &mut self,
        op: ir::OpType,
        target: ir::OpTarget,
        args: SmallVec<[ir::Arg; 4]>,
        kwargs: Vec<(String, ir::Arg)>,
        meta: ir::NodeMeta,
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

        for input_id in Self::collect_input_ids(&self.nodes[id]) {
            if let Some(input_node) = self.nodes.get_mut(input_id) {
                if !input_node.users.contains(&id) {
                    input_node.users.push(id);
                }
            }
        }

        id
    }

    /// Collect all NodeId references from a node's args and kwargs.
    pub fn collect_input_ids(node: &Node) -> smallvec::SmallVec<[NodeId; 8]> {
        let mut ids = smallvec::SmallVec::new();
        for arg in &node.args {
            ids.extend(arg.node_refs());
        }
        for (_, arg) in &node.kwargs {
            ids.extend(arg.node_refs());
        }
        ids
    }

    /// Recompute all user lists from scratch by walking args.
    /// This fixes any stale edges left by passes that don't maintain users correctly.
    pub fn rebuild_user_lists(&mut self) {
        // Clear all user lists
        for (_, node) in self.nodes.iter_mut() {
            node.users.clear();
        }

        // Rebuild by walking all args
        let all_ids: Vec<NodeId> = self.order.clone();
        for &node_id in &all_ids {
            let input_ids: smallvec::SmallVec<[NodeId; 8]> = {
                let Some(node) = self.nodes.get(node_id) else {
                    continue;
                };
                Self::collect_input_ids(node)
            };
            for input_id in input_ids {
                if let Some(input_node) = self.nodes.get_mut(input_id) {
                    if !input_node.users.contains(&node_id) {
                        input_node.users.push(node_id);
                    }
                }
            }
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
