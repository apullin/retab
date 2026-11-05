//! Pure Rust IrPass version of FxFuseDuplicateUsersPass.
//!
//! Deduplicates sibling users of the same producer that have identical
//! (op, target, args, kwargs) signatures. Only fuses direct sibling users,
//! not globally.

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::runner_logging::log_line;

/// Build a signature string for a node based on its op, target, args and kwargs.
fn node_signature(graph: &IrGraph, id: NodeId) -> String {
    let node = &graph.nodes[id];
    let mut sig = format!("{:?}|{}", node.op, node.target);
    sig.push('|');
    for (i, arg) in node.args.iter().enumerate() {
        if i > 0 {
            sig.push(',');
        }
        sig.push_str(&arg_signature(arg));
    }
    sig.push('|');
    let mut kwargs: Vec<_> = node.kwargs.iter().collect();
    kwargs.sort_by_key(|(key, _)| *key);
    for (key, value) in kwargs {
        sig.push_str(&format!("{key:?}={};", arg_signature(value)));
    }
    sig
}

fn arg_signature(arg: &IrArg) -> String {
    match arg {
        IrArg::Node(id) => format!("N{id}"),
        IrArg::Int(v) => format!("i{v}"),
        IrArg::Float(v) => format!("f{v}"),
        IrArg::Bool(v) => format!("b{v}"),
        IrArg::Str(s) => format!("s{s:?}"),
        IrArg::Dtype(s) => format!("d{s:?}"),
        IrArg::List(items) => {
            let inner: Vec<String> = items.iter().map(arg_signature).collect();
            format!("[{}]", inner.join(","))
        }
        IrArg::None => "None".into(),
    }
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
            let ids = graph.live_node_ids();

            for &producer_id in &ids {
                let users = graph.live_call_function_users(producer_id);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{dtype_kwarg, NodeMeta};
    use crate::test_ir::{f32_call, f32_placeholder, output};

    #[test]
    fn regression_siblings_with_different_kwargs_remain_distinct() {
        for (target, key, left_value, right_value) in [
            (
                "dim_order_ops._to_dim_order_copy.default",
                "dtype",
                IrArg::Dtype("torch.int32".into()),
                IrArg::Dtype("torch.float32".into()),
            ),
            (
                "aten.clone.default",
                "memory_format",
                IrArg::Str("torch.contiguous_format".into()),
                IrArg::Str("torch.preserve_format".into()),
            ),
            (
                "aten.div.Tensor_mode",
                "rounding_mode",
                IrArg::Str("floor".into()),
                IrArg::Str("trunc".into()),
            ),
        ] {
            let mut graph = IrGraph::empty();
            let input = f32_placeholder(&mut graph, "x", [2]);
            let mut args = vec![IrArg::Node(input)];
            if target == "aten.div.Tensor_mode" {
                let divisor = f32_placeholder(&mut graph, "divisor", [2]);
                args.push(IrArg::Node(divisor));
            }
            let left = f32_call(&mut graph, "left", target, args.clone(), [2]);
            let right = f32_call(&mut graph, "right", target, args, [2]);
            if let IrArg::Dtype(dtype) = &left_value {
                graph.node_mut(left).meta = NodeMeta::tensor([2], dtype);
            }
            graph.set_kwarg(left, key, left_value);
            graph.set_kwarg(right, key, right_value);
            let sum = f32_call(
                &mut graph,
                "sum",
                "aten.add.Tensor",
                vec![IrArg::Node(left), IrArg::Node(right)],
                [2],
            );
            output(&mut graph, sum);

            assert!(!IrFuseDuplicateUsersPass.run(&mut graph), "{target}");
            assert!(graph.node(left).alive && graph.node(right).alive);
            assert_eq!(graph.node(sum).arg_node(0), Some(left));
            assert_eq!(graph.node(sum).arg_node(1), Some(right));
        }
    }

    #[test]
    fn identical_kwargs_fuse_independent_of_insertion_order() {
        let mut graph = IrGraph::empty();
        let input = f32_placeholder(&mut graph, "x", [2]);
        let left = f32_call(
            &mut graph,
            "left",
            "aten._to_copy.default",
            vec![IrArg::Node(input)],
            [2],
        );
        let right = f32_call(
            &mut graph,
            "right",
            "aten._to_copy.default",
            vec![IrArg::Node(input)],
            [2],
        );
        graph.node_mut(left).kwargs = dtype_kwarg("torch.float32");
        graph.set_kwarg(left, "non_blocking", IrArg::Bool(false));
        graph.set_kwarg(right, "non_blocking", IrArg::Bool(false));
        graph.set_kwarg(right, "dtype", IrArg::Dtype("torch.float32".into()));
        let sum = f32_call(
            &mut graph,
            "sum",
            "aten.add.Tensor",
            vec![IrArg::Node(left), IrArg::Node(right)],
            [2],
        );
        output(&mut graph, sum);

        assert!(IrFuseDuplicateUsersPass.run(&mut graph));
        assert_eq!(graph.node(sum).arg_node(0), graph.node(sum).arg_node(1));
        assert_ne!(graph.node(left).alive, graph.node(right).alive);
    }
}
