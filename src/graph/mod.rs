pub mod builder;
pub mod ir;
pub mod topo;

use ir::{Node, NodeId};
use slotmap::SlotMap;
use smallvec::SmallVec;

/// The core graph IR. Nodes live in a SlotMap arena and are referenced by
/// stable `NodeId` keys. The graph tracks insertion order via `order`, input
/// placeholders via `inputs`, and the single output node via `output`.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: SlotMap<NodeId, Node>,
    /// Insertion-order list of all node IDs (maintained by builder methods).
    pub order: Vec<NodeId>,
    /// Placeholder node IDs that represent graph inputs.
    pub inputs: Vec<NodeId>,
    /// The output node, if set.
    pub output: Option<NodeId>,
    /// Human-readable names for debugging.
    pub node_names: Vec<(NodeId, String)>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            order: Vec::new(),
            inputs: Vec::new(),
            output: None,
            node_names: Vec::new(),
        }
    }

    /// Get a reference to a node by its ID.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get a mutable reference to a node by its ID.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    /// Collect all NodeId references from a node's args and kwargs.
    fn collect_input_ids(node: &Node) -> SmallVec<[NodeId; 8]> {
        let mut ids = SmallVec::new();
        for arg in &node.args {
            ids.extend(arg.node_refs());
        }
        for (_, arg) in &node.kwargs {
            ids.extend(arg.node_refs());
        }
        ids
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
