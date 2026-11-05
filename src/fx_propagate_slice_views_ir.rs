use std::collections::HashMap;

use crate::fx_ir::{normalize_dim, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_target::is_view_like_target;

pub struct IrPropagateSliceViewsPass;

fn physical_shape(graph: &IrGraph, id: NodeId) -> Option<Vec<i64>> {
    let shape = graph.node_shape(id)?;
    match graph.nodes[id].meta.tosa_dim_order() {
        Some(order) if order.len() == shape.len() => {
            order.iter().map(|&axis| shape.get(axis).copied()).collect()
        }
        Some(_) => None,
        None => Some(shape.to_vec()),
    }
}

fn singleton_insertion_axes(input: &[i64], output: &[i64]) -> Option<Vec<usize>> {
    if output.len() <= input.len() {
        return None;
    }
    let mut axes = Vec::with_capacity(input.len());
    for (axis, &dim) in output.iter().enumerate() {
        if input.get(axes.len()) == Some(&dim) {
            axes.push(axis);
        } else if dim != 1 {
            return None;
        }
    }
    (axes.len() == input.len()).then_some(axes)
}

impl IrPass for IrPropagateSliceViewsPass {
    fn name(&self) -> &str {
        "IrPropagateSliceViews"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let mut shared_views: HashMap<(NodeId, Vec<i64>), NodeId> = HashMap::new();
        for slice in graph.live_call_function_ids() {
            if graph.nodes[slice].target != "aten.slice_copy.Tensor"
                || graph.nodes[slice].args.len() < 4
                || graph.nodes[slice].arg_or_kwarg_int(4, "step").unwrap_or(1) != 1
            {
                continue;
            }
            let Some(input) = graph.nodes[slice].first_node_arg() else {
                continue;
            };
            let views = graph.live_user_ids(slice);
            if views.is_empty()
                || views.iter().any(|&view| {
                    graph.nodes[view].op != NodeOp::CallFunction
                        || !is_view_like_target(&graph.nodes[view].target)
                        || graph.nodes[view].first_node_arg() != Some(slice)
                })
            {
                continue;
            }
            let Some(output_shape) = physical_shape(graph, views[0]) else {
                continue;
            };
            if output_shape.iter().any(|&dim| dim <= 0)
                || views.iter().any(|&view| {
                    physical_shape(graph, view).as_ref() != Some(&output_shape)
                        || graph.node_dtype(view) != graph.node_dtype(slice)
                })
            {
                continue;
            }
            let (Some(input_shape), Some(slice_shape)) =
                (physical_shape(graph, input), physical_shape(graph, slice))
            else {
                continue;
            };
            if input_shape.len() != slice_shape.len()
                || graph.nodes[input].meta.tosa_dim_order()
                    != graph.nodes[slice].meta.tosa_dim_order()
            {
                continue;
            }
            let Some(dim) = graph.nodes[slice]
                .arg_or_kwarg_int(1, "dim")
                .and_then(|dim| normalize_dim(dim, input_shape.len()))
            else {
                continue;
            };
            let dim = graph.nodes[slice]
                .meta
                .tosa_dim_order()
                .and_then(|order| order.iter().position(|&axis| axis == dim))
                .unwrap_or(dim);
            if input_shape.iter().zip(&slice_shape).enumerate().any(
                |(axis, (&input_size, &slice_size))| {
                    input_size <= 0 || (axis != dim && input_size != slice_size)
                },
            ) {
                continue;
            }
            let Some(axes) = singleton_insertion_axes(&slice_shape, &output_shape) else {
                continue;
            };
            let mut lifted_shape = output_shape.clone();
            for (&axis, &size) in axes.iter().zip(&input_shape) {
                lifted_shape[axis] = size;
            }
            let key = (input, lifted_shape.clone());
            let lifted_view = match shared_views.get(&key) {
                Some(&view) => view,
                None => {
                    // Work in physical shapes after layout annotation, as rc4's slice/view swap does.
                    let mut meta = graph.nodes[views[0]]
                        .meta
                        .cloned_with_same_dtype_or_float32(lifted_shape.clone());
                    meta.set_tosa_dim_order((0..lifted_shape.len()).collect());
                    let name = format!("{}_input_view", graph.nodes[slice].name);
                    let view = graph.rewrite_cursor_after(slice, input).view_copy(
                        name,
                        input,
                        &lifted_shape,
                        meta,
                    );
                    shared_views.insert(key, view);
                    view
                }
            };
            graph.set_arg(slice, 0, IrArg::Node(lifted_view));
            graph.set_arg(slice, 1, IrArg::Int(axes[dim] as i64));
            graph.nodes[slice].meta = graph.nodes[slice]
                .meta
                .cloned_with_same_dtype_or_float32(output_shape.clone());
            graph.nodes[slice]
                .meta
                .set_tosa_dim_order((0..output_shape.len()).collect());
            for view in views {
                for user in graph.live_user_ids(view) {
                    graph.replace_input_use(user, view, slice);
                }
                graph.replace_and_erase(view, slice);
            }
            modified = true;
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{i64_list_arg, NodeMeta};
    use crate::test_ir::{f32_call, f32_placeholder, output};

    #[test]
    fn shares_sequence_view_across_time_steps_and_projection_users() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 8, 16]);
        let mut slices = Vec::new();
        let mut views = Vec::new();
        for step in 0..8 {
            let slice = f32_call(
                &mut graph,
                &format!("slice{step}"),
                "aten.slice_copy.Tensor",
                vec![
                    IrArg::Node(x),
                    IrArg::Int(-2),
                    IrArg::Int(step),
                    IrArg::Int(step + 1),
                    IrArg::Int(1),
                ],
                [1, 1, 16],
            );
            slices.push(slice);
            for gate in 0..2 {
                let view = f32_call(
                    &mut graph,
                    &format!("view{step}_{gate}"),
                    "aten.view_copy.default",
                    vec![IrArg::Node(slice), i64_list_arg(&[1, 16, 1, 1])],
                    [1, 16, 1, 1],
                );
                graph.nodes[view].meta.set_tosa_dim_order(vec![0, 2, 3, 1]);
                views.push(view);
            }
        }
        let out = graph.add_node(
            NodeOp::Output,
            "output".into(),
            "output".into(),
            vec![IrArg::List(
                views.iter().copied().map(IrArg::Node).collect(),
            )],
            NodeMeta::default(),
        );

        assert!(IrPropagateSliceViewsPass.run(&mut graph));
        let lifted = graph.node(slices[0]).first_node_arg().unwrap();
        assert_eq!(graph.node(lifted).first_node_arg(), Some(x));
        assert_eq!(graph.node_shape(lifted), Some([1, 8, 1, 16].as_slice()));
        assert_eq!(
            graph.node(lifted).meta.tosa_dim_order(),
            Some([0, 1, 2, 3].as_slice())
        );
        for (step, &slice) in slices.iter().enumerate() {
            assert_eq!(graph.node(slice).first_node_arg(), Some(lifted));
            assert_eq!(graph.node(slice).args[1].as_int(), Some(1));
            assert_eq!(graph.node(slice).args[2].as_int(), Some(step as i64));
            assert_eq!(graph.node_shape(slice), Some([1, 1, 1, 16].as_slice()));
            assert!(graph.node(out).args[0].contains_node(slice));
        }
        assert!(views.iter().all(|&view| !graph.node(view).alive));
        assert_eq!(graph.live_node_count(), 11);
        assert!(!IrPropagateSliceViewsPass.run(&mut graph));
    }

    #[test]
    fn preserves_slice_with_a_user_that_needs_its_original_shape() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 8, 16]);
        let slice = f32_call(
            &mut graph,
            "slice",
            "aten.slice_copy.Tensor",
            vec![IrArg::Node(x), IrArg::Int(1), IrArg::Int(0), IrArg::Int(1)],
            [1, 1, 16],
        );
        let view = f32_call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(slice), i64_list_arg(&[1, 1, 1, 16])],
            [1, 1, 1, 16],
        );
        let relu = f32_call(
            &mut graph,
            "relu",
            "aten.relu.default",
            vec![IrArg::Node(slice)],
            [1, 1, 16],
        );
        output(&mut graph, view);
        output(&mut graph, relu);

        assert!(!IrPropagateSliceViewsPass.run(&mut graph));
        assert_eq!(graph.node_shape(slice), Some([1, 1, 16].as_slice()));
        assert_eq!(graph.node(view).first_node_arg(), Some(slice));
        assert_eq!(graph.node(relu).first_node_arg(), Some(slice));
    }

    #[test]
    fn leaves_views_that_repack_non_singleton_dimensions_alone() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 4, 8]);
        let slice = f32_call(
            &mut graph,
            "slice",
            "aten.slice_copy.Tensor",
            vec![IrArg::Node(x), IrArg::Int(1), IrArg::Int(0), IrArg::Int(2)],
            [1, 2, 8],
        );
        let view = f32_call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(slice), i64_list_arg(&[1, 1, 4, 4])],
            [1, 1, 4, 4],
        );
        output(&mut graph, view);

        assert!(!IrPropagateSliceViewsPass.run(&mut graph));
        assert_eq!(graph.node(view).first_node_arg(), Some(slice));
        assert_eq!(graph.node(slice).first_node_arg(), Some(x));
    }

    #[test]
    fn final_fusion_collapses_a_view_exposed_by_slice_propagation() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 128]);
        let sequence = f32_call(
            &mut graph,
            "sequence",
            "aten.view_copy.default",
            vec![IrArg::Node(x), i64_list_arg(&[1, 8, 16])],
            [1, 8, 16],
        );
        let slice = f32_call(
            &mut graph,
            "slice",
            "aten.slice_copy.Tensor",
            vec![IrArg::Node(sequence), IrArg::Int(1), IrArg::Int(0), IrArg::Int(1)],
            [1, 1, 16],
        );
        let view = f32_call(
            &mut graph,
            "view",
            "aten.view_copy.default",
            vec![IrArg::Node(slice), i64_list_arg(&[1, 1, 1, 16])],
            [1, 1, 1, 16],
        );
        output(&mut graph, view);

        assert!(IrPropagateSliceViewsPass.run(&mut graph));
        assert!(crate::fx_fuse_cascaded_views_ir::IrFuseCascadedViewsPass.run(&mut graph));
        let lifted = graph.node(slice).first_node_arg().unwrap();
        assert_eq!(graph.node(lifted).first_node_arg(), Some(x));
        assert_eq!(graph.node_shape(lifted), Some([1, 8, 1, 16].as_slice()));
        assert!(!graph.node(sequence).alive);
        assert!(!graph.node(view).alive);
    }
}
