//! Pure Rust IrPass version of FxFuseDuplicateUsersPass.
//!
//! Deduplicates sibling users of the same producer that have identical
//! (op, target, args) signatures. Only fuses direct sibling users,
//! not globally.

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

/// Build a signature string for a node based on its target and args.
fn node_signature(graph: &IrGraph, id: NodeId) -> String {
    let node = &graph.nodes[id];
    let mut sig = node.target.clone();
    sig.push('|');
    for (i, arg) in node.args.iter().enumerate() {
        if i > 0 {
            sig.push(',');
        }
        sig.push_str(&arg_signature(arg));
    }
    sig
}

fn arg_signature(arg: &IrArg) -> String {
    match arg {
        IrArg::Node(id) => format!("N{id}"),
        IrArg::Int(v) => format!("i{v}"),
        IrArg::Float(v) => format!("f{v}"),
        IrArg::Bool(v) => format!("b{v}"),
        IrArg::Str(s) => format!("s{s}"),
        IrArg::Dtype(s) => format!("d{s}"),
        IrArg::List(items) => {
            let inner: Vec<String> = items.iter().map(arg_signature).collect();
            format!("[{}]", inner.join(","))
        }
        IrArg::None => "None".into(),
    }
}

/// Find all alive users of a node (nodes that reference it in their args).
fn find_users(graph: &IrGraph, producer_id: NodeId) -> Vec<NodeId> {
    graph
        .nodes
        .iter()
        .filter(|n| n.alive && n.op == NodeOp::CallFunction)
        .filter(|n| {
            n.args
                .iter()
                .any(|a| matches!(a, IrArg::Node(id) if *id == producer_id))
        })
        .map(|n| n.id)
        .collect()
}

pub struct IrFuseDuplicateUsersPass;

impl IrPass for IrFuseDuplicateUsersPass {
    fn name(&self) -> &str {
        "IrFuseDuplicateUsers"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        // May need multiple iterations since fusing can expose more duplicates
        loop {
            let mut changed_this_round = false;
            let ids: Vec<NodeId> = (0..graph.nodes.len())
                .filter(|&id| graph.nodes[id].alive)
                .collect();

            for &producer_id in &ids {
                let users = find_users(graph, producer_id);
                if users.len() < 2 {
                    continue;
                }

                // Group users by target
                let mut by_target: HashMap<String, Vec<NodeId>> = HashMap::new();
                for &uid in &users {
                    by_target
                        .entry(graph.nodes[uid].target.clone())
                        .or_default()
                        .push(uid);
                }

                for (_target, group) in &by_target {
                    if group.len() < 2 {
                        continue;
                    }

                    // Within same-target group, find duplicates by full signature
                    let mut sig_to_rep: HashMap<String, NodeId> = HashMap::new();
                    for &uid in group {
                        let sig = node_signature(graph, uid);
                        if let Some(&rep) = sig_to_rep.get(&sig) {
                            if rep != uid {
                                graph.replace_and_erase(uid, rep);
                                changed_this_round = true;
                                modified = true;
                            }
                        } else {
                            sig_to_rep.insert(sig, uid);
                        }
                    }
                }
            }

            if !changed_this_round {
                break;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrFuseDuplicateUsers] fused duplicate sibling users"
            ));
        }
        modified
    }
}
