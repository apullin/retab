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
    node_signature_from(graph, id, 0)
}

/// [`node_signature`] without the positional args before
/// `positional_arg_start`: rc4's `build_node_signature(node,
/// positional_arg_start=...)` (`backends/transforms/fuse_duplicate_users_pass.py:69-83`),
/// which RewriteConvPass uses to compare RESCALE-to-permute heads apart from
/// their inputs. Node args sign as their ids, as rc4's do as their names, and
/// the args kept include trailing positional flags.
pub(crate) fn node_signature_from(
    graph: &IrGraph,
    id: NodeId,
    positional_arg_start: usize,
) -> String {
    let node = &graph.nodes[id];
    let mut sig = format!("{:?}|{}", node.op, node.target);
    sig.push('|');
    for (i, arg) in node.args.iter().skip(positional_arg_start).enumerate() {
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
            let ids = graph.topological_live_node_ids();
            let mut node_order = vec![usize::MAX; graph.nodes.len()];
            for (position, &id) in ids.iter().enumerate() {
                node_order[id] = position;
            }

            for &producer_id in &ids {
                if !graph.node(producer_id).alive {
                    continue;
                }
                let mut users = graph.live_call_function_users(producer_id);
                users.retain(|&id| {
                    let node = graph.node(id);
                    !node.is_rescale()
                        && node.meta.extra_bool("do_not_fuse_duplicate") != Some(true)
                });
                users.sort_by_key(|&id| node_order[id]);
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

                for group in by_target.values() {
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
    use crate::fx_ir::{dtype_kwarg, i64_list_arg, MetaValue, NodeMeta};
    use crate::test_ir::{call, f32_call, f32_placeholder, output, placeholder};

    #[test]
    fn fuses_duplicate_pools_and_pool_chains() {
        for depth in [1, 2] {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", [1, 1, 1, 1]);
            let mut branches = Vec::new();
            for branch in 0..3 {
                let mut current = x;
                for layer in 0..depth {
                    current = f32_call(
                        &mut graph,
                        &format!("pool{branch}_{layer}"),
                        "aten.avg_pool2d.default",
                        vec![
                            IrArg::Node(current),
                            i64_list_arg(&[1, 1]),
                            i64_list_arg(&[1, 1]),
                            i64_list_arg(&[0, 0]),
                            IrArg::Bool(false),
                            IrArg::Bool(true),
                        ],
                        [1, 1, 1, 1],
                    );
                }
                branches.push(IrArg::Node(current));
            }
            let out = output(&mut graph, x);
            graph.set_args(out, vec![IrArg::List(branches)]);
            assert!(IrFuseDuplicateUsersPass.run(&mut graph));
            assert_eq!(
                graph
                    .live_nodes()
                    .filter(|node| node.target == "aten.avg_pool2d.default")
                    .count(),
                depth
            );
            let IrArg::List(outputs) = &graph.node(out).args[0] else {
                panic!("output list")
            };
            assert_eq!(outputs[0].as_node(), outputs[1].as_node());
            assert_eq!(outputs[0].as_node(), outputs[2].as_node());
        }
    }

    fn users_not_in_node_order() -> (IrGraph, NodeId, NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2]);
        let y = f32_placeholder(&mut graph, "y", [2]);
        let later = f32_call(
            &mut graph,
            "later",
            "aten.add.Tensor",
            vec![IrArg::Node(x), IrArg::Node(y)],
            [2],
        );
        let earlier = f32_call(
            &mut graph,
            "earlier",
            "aten.add.Tensor",
            vec![IrArg::Node(x), IrArg::Node(y)],
            [2],
        );
        graph.anchor_to(earlier, later);
        let consumer = f32_call(
            &mut graph,
            "consumer",
            "aten.neg.default",
            vec![IrArg::Node(earlier)],
            [2],
        );
        graph.anchor_to(consumer, later);
        let out = output(&mut graph, consumer);
        graph.set_args(
            out,
            vec![IrArg::List(vec![IrArg::Node(consumer), IrArg::Node(later)])],
        );
        (graph, earlier, later, consumer)
    }

    #[test]
    fn representative_follows_graph_order_instead_of_user_insertion_order() {
        let (mut graph, earlier, later, consumer) = users_not_in_node_order();
        assert!(earlier > later);
        assert!(IrFuseDuplicateUsersPass.run(&mut graph));
        assert!(graph.node(earlier).alive);
        assert!(!graph.node(later).alive);
        assert_eq!(graph.node(consumer).arg_node(0), Some(earlier));
        let order = graph.topological_live_node_ids();
        assert!(
            order.iter().position(|&id| id == earlier)
                < order.iter().position(|&id| id == consumer)
        );
    }

    #[test]
    fn honors_do_not_fuse_marker_on_either_duplicate() {
        for mark_earlier in [false, true] {
            let (mut graph, earlier, later, _) = users_not_in_node_order();
            graph
                .node_mut(if mark_earlier { earlier } else { later })
                .meta
                .extras
                .insert("do_not_fuse_duplicate".into(), MetaValue::Bool(true));
            assert!(!IrFuseDuplicateUsersPass.run(&mut graph));
            assert!(graph.node(earlier).alive && graph.node(later).alive);
        }
    }

    #[test]
    fn skips_erased_queued_nodes_when_fusion_exposes_more_duplicates() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2]);
        let first_neg = f32_call(
            &mut graph,
            "first_neg",
            "aten.neg.default",
            vec![IrArg::Node(x)],
            [2],
        );
        let second_neg = f32_call(
            &mut graph,
            "second_neg",
            "aten.neg.default",
            vec![IrArg::Node(x)],
            [2],
        );
        let first_relu = f32_call(
            &mut graph,
            "first_relu",
            "aten.relu.default",
            vec![IrArg::Node(first_neg)],
            [2],
        );
        let second_relu = f32_call(
            &mut graph,
            "second_relu",
            "aten.relu.default",
            vec![IrArg::Node(second_neg)],
            [2],
        );
        let out = output(&mut graph, first_relu);
        graph.set_args(
            out,
            vec![IrArg::List(vec![
                IrArg::Node(first_relu),
                IrArg::Node(second_relu),
            ])],
        );
        assert!(IrFuseDuplicateUsersPass.run(&mut graph));
        assert!(graph.node(first_neg).alive && graph.node(first_relu).alive);
        assert!(!graph.node(second_neg).alive && !graph.node(second_relu).alive);
        assert_eq!(graph.node(first_relu).arg_node(0), Some(first_neg));
    }

    #[test]
    fn keeps_identical_rescale_users() {
        for target in ["backend.tosa.RESCALE.default", "tosa.RESCALE.default"] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [1], "torch.int8");
            let args = vec![
                IrArg::Node(x),
                IrArg::Dtype("torch.int32".into()),
                IrArg::List(vec![IrArg::Float(1.0)]),
                IrArg::Int(16),
                IrArg::Int(0),
            ];
            let first = call(
                &mut graph,
                "first",
                target,
                args.clone(),
                [1],
                "torch.int32",
            );
            let second = call(&mut graph, "second", target, args, [1], "torch.int32");
            let out = output(&mut graph, first);
            graph.set_args(
                out,
                vec![IrArg::List(vec![IrArg::Node(first), IrArg::Node(second)])],
            );
            assert!(!IrFuseDuplicateUsersPass.run(&mut graph));
            assert!(graph.node(first).alive && graph.node(second).alive);
        }
    }

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
