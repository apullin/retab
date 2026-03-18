use super::Graph;
use crate::graph::ir::NodeId;
use std::collections::{HashMap, HashSet, VecDeque};

impl Graph {
    /// Return a topologically sorted list of node IDs. Uses Kahn's algorithm.
    /// Falls back to insertion order for any nodes not reachable via edges.
    pub fn topological_sort(&self) -> Vec<NodeId> {
        let active_ids: Vec<NodeId> = self.order.iter().copied().collect();
        if active_ids.is_empty() {
            return Vec::new();
        }

        // Build in-degree map.
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for &id in &active_ids {
            in_degree.entry(id).or_insert(0);
        }

        for &id in &active_ids {
            if let Some(node) = self.nodes.get(id) {
                let input_ids = Self::collect_input_ids(node);
                for input_id in input_ids {
                    if self.nodes.contains_key(input_id) {
                        *in_degree.entry(id).or_insert(0) += 1;
                    }
                }
            }
        }

        // Kahn's algorithm.
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for &id in &active_ids {
            if in_degree[&id] == 0 {
                queue.push_back(id);
            }
        }

        let mut sorted = Vec::with_capacity(active_ids.len());
        let mut visited: HashSet<NodeId> = HashSet::with_capacity(active_ids.len());
        while let Some(id) = queue.pop_front() {
            if visited.contains(&id) {
                continue; // Already processed — skip (handles stale user edges)
            }
            visited.insert(id);
            sorted.push(id);
            if let Some(node) = self.nodes.get(id) {
                for &user_id in &node.users {
                    if visited.contains(&user_id) {
                        continue; // Already sorted
                    }
                    if let Some(deg) = in_degree.get_mut(&user_id) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(user_id);
                        }
                    }
                }
            }
        }

        // If there are nodes not covered (cycle or disconnected), append them
        // in insertion order. Use HashSet for O(1) lookup instead of O(N).
        if sorted.len() < active_ids.len() {
            let sorted_set: HashSet<NodeId> = sorted.iter().copied().collect();
            let missing = active_ids.len() - sorted.len();
            eprintln!(
                "[topo_sort] WARNING: {} of {} nodes not in topo order (cycles or stale edges)",
                missing, active_ids.len()
            );
            for &id in &active_ids {
                if !sorted_set.contains(&id) {
                    sorted.push(id);
                }
            }
        }

        sorted
    }
}
