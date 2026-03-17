use super::Graph;
use crate::graph::ir::*;
use smallvec::SmallVec;

impl Graph {
    // -----------------------------------------------------------------------
    // Node creation
    // -----------------------------------------------------------------------

    /// Create a new node and append it to the end of the graph order.
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

    // -----------------------------------------------------------------------
    // Node removal
    // -----------------------------------------------------------------------

    /// Remove a node from the graph. Cleans up user lists of its inputs.
    pub fn erase_node(&mut self, id: NodeId) {
        if let Some(node) = self.nodes.remove(id) {
            // Remove from order.
            self.order.retain(|&nid| nid != id);
            self.node_names.retain(|(nid, _)| *nid != id);

            // Remove this node from the user lists of all its inputs.
            let input_ids = Self::collect_input_ids(&node);
            for input_id in input_ids {
                if let Some(input_node) = self.nodes.get_mut(input_id) {
                    input_node.users.retain(|uid| *uid != id);
                }
            }

            // Remove this node from the user lists of any node that lists it.
            // (Handles kwargs/args pointing at this node from other nodes.)
            let all_ids: Vec<NodeId> = self.nodes.keys().collect();
            for nid in all_ids {
                if let Some(n) = self.nodes.get_mut(nid) {
                    n.users.retain(|uid| *uid != id);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Rewiring
    // -----------------------------------------------------------------------

    /// Replace every reference to `old_id` with `new_id` across all nodes.
    pub fn replace_all_uses(&mut self, old_id: NodeId, new_id: NodeId) {
        let all_ids: Vec<NodeId> = self.nodes.keys().collect();
        for nid in all_ids {
            let node = &mut self.nodes[nid];
            let mut changed = false;

            // Rewrite args.
            for arg in &mut node.args {
                if Self::replace_id_in_arg(arg, old_id, new_id) {
                    changed = true;
                }
            }
            // Rewrite kwargs.
            for (_, arg) in &mut node.kwargs {
                if Self::replace_id_in_arg(arg, old_id, new_id) {
                    changed = true;
                }
            }

            // Update user lists.
            if changed {
                node.users.retain(|uid| *uid != old_id);
            }
        }

        // Transfer users from old node to new node.
        if let Some(old_node) = self.nodes.get(old_id) {
            let old_users: SmallVec<[NodeId; 4]> = old_node.users.clone();
            if let Some(new_node) = self.nodes.get_mut(new_id) {
                for uid in old_users {
                    if !new_node.users.contains(&uid) {
                        new_node.users.push(uid);
                    }
                }
            }
            // Clear old node's users.
            if let Some(old_node) = self.nodes.get_mut(old_id) {
                old_node.users.clear();
            }
        }
    }

    /// Bypass a node: rewire its first input directly to all its users, then
    /// erase it. Useful for removing DQ nodes.
    pub fn bypass_node(&mut self, id: NodeId) {
        let first_input = self.nodes.get(id).and_then(|node| {
            node.args.first().and_then(|arg| match arg {
                Arg::Node(nid) => Some(*nid),
                _ => None,
            })
        });

        if let Some(input_id) = first_input {
            self.replace_all_uses(id, input_id);
            self.erase_node(id);
        }
    }

    // -----------------------------------------------------------------------
    // Insertion helpers
    // -----------------------------------------------------------------------

    /// Insert a new node immediately before `ref_id` in the graph order.
    pub fn insert_before(
        &mut self,
        ref_id: NodeId,
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

        if let Some(pos) = self.order.iter().position(|&nid| nid == ref_id) {
            self.order.insert(pos, id);
        } else {
            self.order.push(id);
        }
        self.node_names.push((id, name));

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

    /// Insert a new node immediately after `ref_id` in the graph order.
    pub fn insert_after(
        &mut self,
        ref_id: NodeId,
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

        if let Some(pos) = self.order.iter().position(|&nid| nid == ref_id) {
            self.order.insert(pos + 1, id);
        } else {
            self.order.push(id);
        }
        self.node_names.push((id, name));

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

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Find all nodes with the given target.
    pub fn nodes_by_target(&self, target: &OpTarget) -> Vec<NodeId> {
        self.order
            .iter()
            .copied()
            .filter(|&id| {
                self.nodes
                    .get(id)
                    .map(|n| &n.target == target)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Return all node IDs in insertion order.
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.order.clone()
    }

    /// Iterate over all nodes in order, yielding (NodeId, &Node).
    pub fn iter_nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.order
            .iter()
            .filter_map(move |&id| self.nodes.get(id).map(|n| (id, n)))
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Replace `old_id` with `new_id` inside a single Arg. Returns true if
    /// anything changed.
    fn replace_id_in_arg(arg: &mut Arg, old_id: NodeId, new_id: NodeId) -> bool {
        match arg {
            Arg::Node(id) if *id == old_id => {
                *id = new_id;
                true
            }
            Arg::NodeList(ids) => {
                let mut changed = false;
                for id in ids.iter_mut() {
                    if *id == old_id {
                        *id = new_id;
                        changed = true;
                    }
                }
                changed
            }
            Arg::OptionalNode(Some(id)) if *id == old_id => {
                *id = new_id;
                true
            }
            _ => false,
        }
    }
}
