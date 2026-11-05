use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeOp};
use crate::fx_target::is_view_like_target;

pub struct IrFuseCascadedViewsPass;
pub struct IrFuseDuplicateViewsPass;

impl IrPass for IrFuseCascadedViewsPass {
    fn name(&self) -> &str {
        "IrFuseCascadedViews"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !graph.nodes[id].alive || !is_view_like_target(&graph.nodes[id].target) {
                continue;
            }
            while let Some(parent) = graph.nodes[id].first_node_arg() {
                if !graph.nodes[parent].alive
                    || graph.nodes[parent].op != NodeOp::CallFunction
                    || !is_view_like_target(&graph.nodes[parent].target)
                {
                    break;
                }
                let Some(input) = graph.nodes[parent].first_node_arg() else {
                    break;
                };
                if graph.nodes[id].target == "aten.flatten.using_ints" {
                    let Some(shape) = graph.node_shape_vec(id) else {
                        break;
                    };
                    graph.nodes[id].target = "aten.view_copy.default".into();
                    graph.set_args(id, vec![IrArg::Node(input), i64_list_arg(&shape)]);
                } else {
                    graph.set_arg(id, 0, IrArg::Node(input));
                }
                // rc4 bypasses each view user independently; other users keep the parent.
                if !graph.has_live_users(parent) {
                    graph.erase(parent);
                }
                modified = true;
            }
        }
        modified
    }
}

impl IrPass for IrFuseDuplicateViewsPass {
    fn name(&self) -> &str {
        "IrFuseDuplicateViews"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut representatives = HashMap::new();
        for id in graph.live_call_function_ids() {
            let node = graph.node(id);
            if node.target != "aten.view_copy.default"
                || node.meta.input_qparams().is_some_and(|q| !q.is_empty())
                || node.meta.output_qparams().is_some_and(|q| !q.is_empty())
                || node.meta.extra_str("tosa_special_dtype").is_some()
                || graph
                    .live_user_ids(id)
                    .iter()
                    .any(|&user| graph.node(user).op == NodeOp::Output)
            {
                continue;
            }
            let (Some(input), Some(shape)) = (node.first_node_arg(), graph.node_shape_vec(id))
            else {
                continue;
            };
            let key = (
                input,
                shape,
                node.meta.cloned_tosa_dim_order(),
                node.meta.val_dtype().map(str::to_owned),
            );
            if let Some(&representative) = representatives.get(&key) {
                for user in graph.live_user_ids(id) {
                    graph.replace_input_use(user, id, representative);
                }
                graph.replace_and_erase(id, representative);
                modified = true;
            } else {
                representatives.insert(key, id);
            }
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{f32_call, f32_placeholder, output};

    #[test]
    fn fuses_single_user_chain_and_preserves_final_shape() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 32, 7, 7]);
        let flatten = f32_call(
            &mut graph,
            "flatten",
            "aten.view_copy.default",
            vec![IrArg::Node(x), i64_list_arg(&[1, 1568])],
            [1, 1568],
        );
        let view = f32_call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(flatten), i64_list_arg(&[1, 1568, 1, 1])],
            [1, 1568, 1, 1],
        );
        let relu = f32_call(
            &mut graph,
            "relu",
            "aten.relu.default",
            vec![IrArg::Node(view)],
            [1, 1568, 1, 1],
        );
        output(&mut graph, relu);

        assert!(IrFuseCascadedViewsPass.run(&mut graph));
        assert!(!graph.node(flatten).alive);
        assert_eq!(graph.node(view).first_node_arg(), Some(x));
        assert_eq!(graph.node_shape(view), Some([1, 1568, 1, 1].as_slice()));
        assert_eq!(
            graph.node(view).args[1].as_int_list(),
            Some(vec![1, 1568, 1, 1])
        );
        assert_eq!(graph.node(relu).first_node_arg(), Some(view));
        assert!(!IrFuseCascadedViewsPass.run(&mut graph));
    }

    #[test]
    fn bypasses_multi_user_view_without_changing_other_branch() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 1, 16]);
        let squeeze = f32_call(
            &mut graph,
            "squeeze",
            "aten.view_copy.default",
            vec![IrArg::Node(x), i64_list_arg(&[1, 16])],
            [1, 16],
        );
        let relu = f32_call(
            &mut graph,
            "relu",
            "aten.relu.default",
            vec![IrArg::Node(squeeze)],
            [1, 16],
        );
        let view = f32_call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(squeeze), i64_list_arg(&[1, 16, 1, 1])],
            [1, 16, 1, 1],
        );
        output(&mut graph, view);

        assert!(IrFuseCascadedViewsPass.run(&mut graph));
        assert!(graph.node(squeeze).alive);
        assert_eq!(graph.node_shape(squeeze), Some([1, 16].as_slice()));
        assert_eq!(graph.node(relu).first_node_arg(), Some(squeeze));
        assert_eq!(graph.node(view).first_node_arg(), Some(x));
        assert_eq!(graph.live_user_ids(squeeze), vec![relu]);
    }

    #[test]
    fn fuses_long_chain_ending_in_graph_output() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 3, 4]);
        let mut input = x;
        let mut views = Vec::new();
        for (i, shape) in [vec![6, 4], vec![24], vec![1, 24]].into_iter().enumerate() {
            input = f32_call(
                &mut graph,
                &format!("view{i}"),
                "aten.view_copy.default",
                vec![IrArg::Node(input), i64_list_arg(&shape)],
                shape,
            );
            views.push(input);
        }
        let out = output(&mut graph, input);

        assert!(IrFuseCascadedViewsPass.run(&mut graph));
        assert!(views[..2].iter().all(|&id| !graph.node(id).alive));
        assert_eq!(graph.node(input).first_node_arg(), Some(x));
        assert_eq!(graph.node_shape(input), Some([1, 24].as_slice()));
        assert_eq!(graph.node(out).first_node_arg(), Some(input));
    }

    #[test]
    fn uses_resolved_flatten_shape_when_bypassing_a_view() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 3, 4]);
        let view = f32_call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(x), i64_list_arg(&[6, 2, 2])],
            [6, 2, 2],
        );
        let flatten = f32_call(
            &mut graph,
            "flatten",
            "aten.flatten.using_ints",
            vec![IrArg::Node(view), IrArg::Int(1), IrArg::Int(2)],
            [6, 4],
        );
        output(&mut graph, flatten);

        assert!(IrFuseCascadedViewsPass.run(&mut graph));
        assert!(!graph.node(view).alive);
        assert_eq!(graph.node(flatten).target, "aten.view_copy.default");
        assert_eq!(graph.node(flatten).first_node_arg(), Some(x));
        assert_eq!(graph.node(flatten).args[1].as_int_list(), Some(vec![6, 4]));
    }

    #[test]
    fn deduplicates_gate_views_without_fusing_their_rescale_users() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 24]);
        let mut views = Vec::new();
        let mut rescales = Vec::new();
        for gate in 0..3 {
            let view = f32_call(
                &mut graph,
                &format!("view{gate}"),
                "aten.view_copy.default",
                vec![IrArg::Node(x), i64_list_arg(&[1, 24, 1, 1])],
                [1, 24, 1, 1],
            );
            graph.nodes[view].meta.set_tosa_dim_order(vec![0, 2, 3, 1]);
            views.push(view);
            rescales.push(f32_call(
                &mut graph,
                &format!("rescale{gate}"),
                "backend.tosa.RESCALE.default",
                vec![IrArg::Node(view), IrArg::Float(0.25)],
                [1, 24, 1, 1],
            ));
        }
        graph.add_node(
            NodeOp::Output,
            "output".into(),
            "output".into(),
            vec![IrArg::List(rescales.iter().copied().map(IrArg::Node).collect())],
            crate::fx_ir::NodeMeta::default(),
        );

        assert!(IrFuseDuplicateViewsPass.run(&mut graph));
        assert!(graph.node(views[0]).alive);
        assert!(views[1..].iter().all(|&view| !graph.node(view).alive));
        for rescale in rescales {
            assert!(graph.node(rescale).alive);
            assert_eq!(graph.node(rescale).first_node_arg(), Some(views[0]));
        }
        assert!(!IrFuseDuplicateViewsPass.run(&mut graph));
    }

    #[test]
    fn preserves_duplicate_views_that_are_graph_outputs() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 24]);
        let mut views = Vec::new();
        for branch in 0..2 {
            views.push(f32_call(
                &mut graph,
                &format!("view{branch}"),
                "aten.view_copy.default",
                vec![IrArg::Node(x), i64_list_arg(&[1, 24, 1, 1])],
                [1, 24, 1, 1],
            ));
        }
        let relu = f32_call(
            &mut graph,
            "relu",
            "aten.relu.default",
            vec![IrArg::Node(views[0])],
            [1, 24, 1, 1],
        );
        graph.add_node(
            NodeOp::Output,
            "output".into(),
            "output".into(),
            vec![IrArg::List(vec![IrArg::Node(relu), IrArg::Node(views[1])])],
            crate::fx_ir::NodeMeta::default(),
        );

        assert!(!IrFuseDuplicateViewsPass.run(&mut graph));
        assert!(views.iter().all(|&view| graph.node(view).alive));
        assert_eq!(graph.node(relu).first_node_arg(), Some(views[0]));
    }
}
