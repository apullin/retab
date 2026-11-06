//! Pure Rust IrPass version of FxRemoveNoopPass.
//!
//! Removes identity-like nodes from the graph: clone, _clone_dim_order,
//! _to_dim_order_copy, _to_copy, alias_copy, copy, and detach_copy where
//! input and output dtypes match.
//!
//! For `aten.copy.default`, cast and broadcast args[1] (the source) to the
//! destination's dtype and shape. For all others, args[0] is the pass-through
//! value.

use crate::fx_fuse_consecutive_slices_ir::static_slice;
use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_target::{noop_target, NoopTarget};
use crate::runner_logging::log_line;

pub struct IrRemoveNoopPass;

fn is_identity_resize(graph: &IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    let scale_index = match node.target.as_str() {
        "aten.upsample_bilinear2d.vec" => 3,
        "aten.upsample_nearest2d.vec" => 2,
        _ => return false,
    };
    let scales = node.arg(scale_index).unwrap_or(&IrArg::None);
    let identity_scale = match scales {
        IrArg::Int(_) | IrArg::Float(_) => scales.as_numeric_f64() == Some(1.0),
        IrArg::List(values) => {
            values.len() == 2
                && values
                    .iter()
                    .all(|scale| scale.as_numeric_f64() == Some(1.0))
        }
        _ => false,
    };
    if identity_scale {
        return true;
    }
    let Some(input_shape) = node.arg_node(0).and_then(|input| graph.node_shape(input)) else {
        return false;
    };
    let output_spatial = node.arg(1).and_then(|arg| graph.constant_shape(arg));
    matches!(scales, IrArg::None)
        && input_shape.len() >= 2
        && output_spatial.as_deref() == Some(&input_shape[2..])
}

fn partition_would_be_empty_without_resize(graph: &IrGraph) -> bool {
    let mut has_resize = false;
    for node in graph.live_nodes() {
        if node.op != NodeOp::CallFunction || node.is_q_or_dq() {
            continue;
        }
        if is_identity_resize(graph, node.id) {
            has_resize = true;
            continue;
        }
        if matches!(
            node.target.as_str(),
            "aten.cat.default" | "aten.concatenate.default"
        )
            && matches!(node.arg(0), Some(IrArg::List(inputs)) if inputs.len() == 1) {
                continue;
            }
        match noop_target(&node.target) {
            Some(NoopTarget::PassthroughFirstArg) => continue,
            Some(NoopTarget::DtypeCheckedPassthrough) => {
                let input_dtype = node.arg_node(0).and_then(|id| graph.node_dtype(id));
                let output_dtype = node
                    .kwarg("dtype")
                    .and_then(|arg| match arg {
                        IrArg::Dtype(dtype) => Some(dtype.as_str()),
                        _ => None,
                    })
                    .or(input_dtype);
                if input_dtype.is_some() && input_dtype == output_dtype {
                    continue;
                }
            }
            Some(NoopTarget::CopySourceArg) => {
                if let Some((destination, source)) = node.arg_node(0).zip(node.arg_node(1)) {
                    if graph.node_shape(destination).is_some()
                        && graph.node_shape(destination) == graph.node_shape(source)
                        && graph.node_dtype(destination) == graph.node_dtype(source)
                    {
                        continue;
                    }
                }
            }
            _ => {}
        }
        return false;
    }
    has_resize
}

fn is_identity_view(graph: &IrGraph, id: NodeId, input: NodeId) -> bool {
    let node = graph.node(id);
    if !matches!(
        node.target.as_str(),
        "aten.view_copy.default"
            | "aten.view.default"
            | "aten.reshape.default"
            | "aten._unsafe_view.default"
            | "backend.tosa.RESHAPE.default"
            | "tosa.RESHAPE.default"
    ) {
        return false;
    }
    let Some(input_shape) = graph.node_shape(input) else {
        return false;
    };
    if graph.node_shape(id) != Some(input_shape) {
        return false;
    }
    if let Some((input_order, output_order)) = graph
        .node(input)
        .meta
        .tosa_dim_order()
        .zip(node.meta.tosa_dim_order())
    {
        if input_order != output_order {
            return false;
        }
    }
    let Some(mut shape) = node.arg(1).and_then(|arg| graph.constant_shape(arg)) else {
        return false;
    };
    if let Some(inferred) = shape.iter().position(|&dim| dim == -1) {
        let Some(known) = shape
            .iter()
            .filter(|&&dim| dim != -1)
            .try_fold(1i64, |product, &dim| {
                (dim >= 0).then(|| product.checked_mul(dim)).flatten()
            })
        else {
            return false;
        };
        let Some(numel) = input_shape
            .iter()
            .try_fold(1i64, |product, &dim| product.checked_mul(dim))
        else {
            return false;
        };
        if shape.iter().filter(|&&dim| dim == -1).count() != 1 || known == 0 || numel % known != 0 {
            return false;
        }
        shape[inferred] = numel / known;
    }
    shape == input_shape
}

/// A `_to_copy`/`_to_dim_order_copy` whose val dtypes disagree only because
/// q/dq folding left a stale float output dtype: the input dtype matches the
/// node's own output qparam, shapes agree, and every live user declares a
/// quantized input for the slots referencing this node, so rewiring to the
/// input is value-preserving. This mirrors stock, where the copy survives
/// every pass but the serializer emits nothing for it.
fn is_quantized_noop_copy(graph: &IrGraph, id: NodeId, input_id: NodeId) -> bool {
    let Some(input_dtype) = graph.node_dtype(input_id).map(str::to_string) else {
        return false;
    };
    let node = &graph.nodes[id];
    let Some(qp) = node.meta.output_qparam() else {
        return false;
    };
    if qp.is_per_channel() || qp.dtype_str != input_dtype {
        return false;
    }
    if graph.node_shape_vec(id) != graph.node_shape_vec(input_id) {
        return false;
    }
    for user_id in graph.live_user_ids(id) {
        let user = graph.node(user_id);
        for (index, arg) in user.args.iter().enumerate() {
            if arg.as_node() == Some(id) && user.meta.input_qparam_at(index).is_none() {
                return false;
            }
        }
        for arg in user.kwargs.values() {
            if arg_contains_node(arg, id) {
                return false;
            }
        }
    }
    true
}

fn arg_contains_node(arg: &IrArg, id: NodeId) -> bool {
    match arg {
        IrArg::Node(node_id) => *node_id == id,
        IrArg::List(values) => values.iter().any(|arg| arg_contains_node(arg, id)),
        _ => false,
    }
}

impl IrPass for IrRemoveNoopPass {
    fn name(&self) -> &str {
        "IrRemoveNoop"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        if partition_would_be_empty_without_resize(graph) {
            return false;
        }
        let mut modified = graph.rewrite_live_call_functions(|graph, id| {
            let node = graph.node(id);
            if matches!(
                node.target.as_str(),
                "aten.cat.default" | "aten.concatenate.default"
            ) {
                if let Some(IrArg::List(inputs)) = node.arg(0) {
                    if inputs.len() == 1 {
                        if let Some(input) = inputs[0].as_node() {
                            graph.replace_and_erase(id, input);
                            return true;
                        }
                    }
                }
                return false;
            }
            if let Some(input) = node.arg_node(0) {
                let zero_pad = match node.target.as_str() {
                    "backend.tosa.PAD.default" | "tosa.PAD.default" => node
                        .arg(1)
                        .and_then(|arg| graph.constant_shape(arg))
                        .is_some_and(|pads| {
                            graph
                                .node_shape(input)
                                .is_some_and(|shape| pads.len() == 2 * shape.len())
                                && pads.iter().all(|&pad| pad == 0)
                        }),
                    "aten.constant_pad_nd.default" | "aten.pad.default" => {
                        node.arg_int_list(1).is_some_and(|pads| {
                            graph.node_shape(input).is_some_and(|shape| {
                                pads.len() % 2 == 0 && pads.len() <= 2 * shape.len()
                            }) && pads.iter().all(|&pad| pad == 0)
                        })
                    }
                    _ => false,
                };
                let full_slice = static_slice(graph, id).is_some_and(|slice| {
                    graph.logical_input_shape(id, input).is_some_and(|shape| {
                        slice.start.len() == shape.len()
                            && slice.start.iter().all(|&start| start == 0)
                            && slice.size == shape
                    })
                });
                if zero_pad
                    || full_slice
                    || is_identity_resize(graph, id)
                    || is_identity_view(graph, id, input)
                {
                    graph.replace_and_erase(id, input);
                    return true;
                }
            }
            let Some(noop) = noop_target(&graph.nodes[id].target) else {
                return false;
            };

            if graph.nodes[id].args.is_empty() {
                return false;
            }

            // The input node is always args[0] (used for dtype check).
            let input_id: NodeId = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => return false,
            };

            // Check dtype match: input dtype must equal output dtype.
            let input_dtype = graph.node_dtype(input_id);
            let Some(input_dtype) = input_dtype.map(str::to_string) else {
                return false;
            };

            if noop == NoopTarget::CopySourceArg {
                let Some(source_id) = graph.nodes[id].arg_node(1) else {
                    return false;
                };
                if graph.node_dtype(source_id).is_none() {
                    return false;
                }
                let Some(source_shape) = graph.node_shape_vec(source_id) else {
                    return false;
                };
                let Some(output_shape) = graph.node_shape_vec(input_id) else {
                    return false;
                };
                if source_shape.len() > output_shape.len() {
                    return false;
                }
                let rank_padding = output_shape.len() - source_shape.len();
                let mut aligned_shape = vec![1; rank_padding];
                aligned_shape.extend_from_slice(&source_shape);
                if aligned_shape
                    .iter()
                    .zip(&output_shape)
                    .any(|(&source, &output)| source != 1 && source != output)
                {
                    return false;
                }

                let name = graph.nodes[id].name.clone();
                let meta = graph.nodes[id].meta.clone();
                let mut cursor = graph.rewrite_cursor(id);
                let mut source = cursor.dtype_cast_if_needed(
                    format!("{name}_cast"),
                    source_id,
                    &input_dtype,
                    meta.cloned_with_val(source_shape.clone(), &input_dtype),
                );
                if source_shape != output_shape {
                    if rank_padding != 0 {
                        source = cursor.view_copy(
                            format!("{name}_view"),
                            source,
                            &aligned_shape,
                            meta.cloned_with_val(aligned_shape.clone(), &input_dtype),
                        );
                    }
                    let multiples: Vec<i64> = aligned_shape
                        .iter()
                        .zip(&output_shape)
                        .map(|(&source, &output)| if source == 1 { output } else { 1 })
                        .collect();
                    source = cursor.call_function(
                        format!("{name}_repeat"),
                        "aten.repeat.default",
                        vec![IrArg::Node(source), i64_list_arg(&multiples)],
                        meta.cloned_with_val(output_shape, &input_dtype),
                    );
                }
                cursor.replace_anchor_with(source);
                return true;
            }

            if noop == NoopTarget::DtypeCheckedPassthrough {
                let output_dtype = graph.node_dtype(id);
                if Some(input_dtype.as_str()) != output_dtype
                    && !is_quantized_noop_copy(graph, id, input_id)
                {
                    return false;
                }
            }

            log_line(format_args!(
                "[IrRemoveNoop] removing '{}' (target={})",
                graph.nodes[id].name, graph.nodes[id].target
            ));

            graph.replace_and_erase(id, input_id);
            true
        });
        modified |= graph.eliminate_dead_code();

        if modified {
            log_line(format_args!("[IrRemoveNoop] removed noop identity nodes"));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{IrArg, NodeMeta};
    use crate::test_ir::{call, call_default, output, placeholder};

    fn data_layout_graph(
        target: &str,
        values: &[&[i64]],
        shape: &[i64],
    ) -> (IrGraph, NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], "torch.float32");
        let mut args = vec![IrArg::Node(x)];
        for (index, values) in values.iter().enumerate() {
            let constant = call_default(
                &mut graph,
                &format!("shape{index}"),
                "backend.tosa.CONST_SHAPE.default",
                vec![i64_list_arg(values)],
            );
            args.push(IrArg::Node(constant));
        }
        let node = call(&mut graph, "result", target, args, shape, "torch.float32");
        output(&mut graph, node);
        (graph, x, node)
    }

    #[test]
    fn removes_single_input_concats_and_keeps_multi_input_concats() {
        for target in ["aten.cat.default", "aten.concatenate.default"] {
            for single in [false, true] {
                let mut graph = IrGraph::empty();
                let x = placeholder(&mut graph, "x", [2, 3], "torch.float32");
                let members = vec![IrArg::Node(x); if single { 1 } else { 2 }];
                let cat = call(
                    &mut graph,
                    "cat",
                    target,
                    vec![IrArg::List(members), IrArg::Int(0)],
                    [if single { 2 } else { 4 }, 3],
                    "torch.float32",
                );
                let out = output(&mut graph, cat);
                assert_eq!(IrRemoveNoopPass.run(&mut graph), single);
                assert_eq!(graph.node(cat).alive, !single);
                assert_eq!(
                    graph.node(out).arg_node(0),
                    Some(if single { x } else { cat })
                );
            }
        }
    }

    #[test]
    fn removes_zero_pad_and_unused_shape_constant() {
        let (mut graph, x, pad) =
            data_layout_graph("backend.tosa.PAD.default", &[&[0, 0, 0, 0]], &[2, 3]);
        assert!(IrRemoveNoopPass.run(&mut graph));
        assert!(!graph.node(pad).alive);
        assert!(graph.live_call_function_ids().is_empty());
        assert_eq!(graph.output_node().unwrap().arg_node(0), Some(x));
    }

    #[test]
    fn keeps_nonzero_and_malformed_tosa_padding() {
        for pads in [vec![0, 0, 1, 0], vec![], vec![0], vec![0, 0]] {
            let (mut graph, _, pad) =
                data_layout_graph("backend.tosa.PAD.default", &[&pads], &[2, 3]);
            assert!(!IrRemoveNoopPass.run(&mut graph));
            assert!(graph.node(pad).alive);
        }
    }

    #[test]
    fn removes_full_slice_and_unused_shape_constants() {
        let (mut graph, x, slice) =
            data_layout_graph("backend.tosa.SLICE.default", &[&[0, 0], &[2, 3]], &[2, 3]);
        assert!(IrRemoveNoopPass.run(&mut graph));
        assert!(!graph.node(slice).alive);
        assert!(graph.live_call_function_ids().is_empty());
        assert_eq!(graph.output_node().unwrap().arg_node(0), Some(x));
    }

    #[test]
    fn keeps_partial_and_nonconstant_slices() {
        let (mut graph, _, slice) =
            data_layout_graph("backend.tosa.SLICE.default", &[&[1, 0], &[1, 3]], &[1, 3]);
        assert!(!IrRemoveNoopPass.run(&mut graph));
        assert!(graph.node(slice).alive);
        let (mut graph, _, slice) =
            data_layout_graph("backend.tosa.SLICE.default", &[&[0, 0], &[2, 3]], &[2, 3]);
        let dynamic = graph.add_node(
            NodeOp::Placeholder,
            "dynamic".into(),
            "dynamic".into(),
            Vec::new(),
            NodeMeta::default(),
        );
        graph.set_arg(slice, 1, IrArg::Node(dynamic));
        IrRemoveNoopPass.run(&mut graph);
        assert!(graph.node(slice).alive);
    }

    #[test]
    fn removes_normalized_full_aten_slice_but_keeps_empty_slice() {
        for (start, end, shape, remove) in [
            (0, i64::MAX, [2, 3], true),
            (-3, 3, [2, 3], true),
            (1, 1, [2, 0], false),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2, 3], "torch.float32");
            let slice = call(
                &mut graph,
                "slice",
                "aten.slice_copy.Tensor",
                vec![
                    IrArg::Node(x),
                    IrArg::Int(1),
                    IrArg::Int(start),
                    IrArg::Int(end),
                ],
                shape,
                "torch.float32",
            );
            output(&mut graph, slice);
            assert_eq!(IrRemoveNoopPass.run(&mut graph), remove);
            assert_eq!(graph.node(slice).alive, !remove);
        }
    }

    #[test]
    fn removes_identity_views_and_reshapes_including_inferred_dimensions() {
        for target in [
            "aten.view_copy.default",
            "aten.reshape.default",
            "backend.tosa.RESHAPE.default",
        ] {
            for shape in [[2, 3], [-1, 3]] {
                let mut graph = IrGraph::empty();
                let x = placeholder(&mut graph, "x", [2, 3], "torch.float32");
                let view = call(
                    &mut graph,
                    "view",
                    target,
                    vec![IrArg::Node(x), i64_list_arg(&shape)],
                    [2, 3],
                    "torch.float32",
                );
                let out = output(&mut graph, view);
                assert!(IrRemoveNoopPass.run(&mut graph));
                assert!(!graph.node(view).alive);
                assert_eq!(graph.node(out).arg_node(0), Some(x));
            }
        }
    }

    #[test]
    fn keeps_nonidentity_reshape_chains_and_stale_view_metadata() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], "torch.float32");
        let first = call(
            &mut graph,
            "first",
            "aten.view_copy.default",
            vec![IrArg::Node(x), i64_list_arg(&[6])],
            [6],
            "torch.float32",
        );
        let second = call(
            &mut graph,
            "second",
            "aten.view_copy.default",
            vec![IrArg::Node(first), i64_list_arg(&[3, 2])],
            [3, 2],
            "torch.float32",
        );
        output(&mut graph, second);
        assert!(!IrRemoveNoopPass.run(&mut graph));
        assert!(graph.node(first).alive && graph.node(second).alive);
        graph.node_mut(first).meta = NodeMeta::tensor([2, 3], "torch.float32");
        assert!(!IrRemoveNoopPass.run(&mut graph));
        assert!(graph.node(first).alive);
    }

    #[test]
    fn identity_resize_is_kept_when_partition_would_be_empty() {
        for (target, bilinear) in [
            ("aten.upsample_bilinear2d.vec", true),
            ("aten.upsample_nearest2d.vec", false),
        ] {
            for alias in [false, true] {
                let mut graph = IrGraph::empty();
                let x = placeholder(&mut graph, "x", [1, 4, 8, 8], "torch.float32");
                let mut args = vec![IrArg::Node(x), IrArg::None];
                if bilinear {
                    args.push(IrArg::Bool(false));
                }
                args.push(IrArg::List(vec![IrArg::Float(1.0), IrArg::Float(1.0)]));
                let resize = call(
                    &mut graph,
                    "resize",
                    target,
                    args,
                    [1, 4, 8, 8],
                    "torch.float32",
                );
                let result = if alias {
                    call(
                        &mut graph,
                        "alias",
                        "aten.alias_copy.default",
                        vec![IrArg::Node(resize)],
                        [1, 4, 8, 8],
                        "torch.float32",
                    )
                } else {
                    resize
                };
                output(&mut graph, result);
                assert!(!IrRemoveNoopPass.run(&mut graph));
                assert!(graph.node(resize).alive && graph.node(result).alive);
            }
        }
    }

    #[test]
    fn identity_resize_is_removed_when_compute_survives() {
        for (target, bilinear) in [
            ("aten.upsample_bilinear2d.vec", true),
            ("aten.upsample_nearest2d.vec", false),
        ] {
            for explicit_size in [false, true] {
                let mut graph = IrGraph::empty();
                let x = placeholder(&mut graph, "x", [1, 4, 8, 8], "torch.float32");
                let mut args = vec![
                    IrArg::Node(x),
                    if explicit_size {
                        i64_list_arg(&[8, 8])
                    } else {
                        IrArg::None
                    },
                ];
                if bilinear {
                    args.push(IrArg::Bool(false));
                }
                args.push(if explicit_size {
                    IrArg::None
                } else {
                    IrArg::Float(1.0)
                });
                let resize = call(
                    &mut graph,
                    "resize",
                    target,
                    args,
                    [1, 4, 8, 8],
                    "torch.float32",
                );
                let add = call(
                    &mut graph,
                    "add",
                    "aten.add.Tensor",
                    vec![IrArg::Node(resize), IrArg::Node(x)],
                    [1, 4, 8, 8],
                    "torch.float32",
                );
                output(&mut graph, add);
                assert!(IrRemoveNoopPass.run(&mut graph));
                assert!(!graph.node(resize).alive);
                assert_eq!(graph.node(add).arg_node(0), Some(x));
            }
        }
    }

    #[test]
    fn resize_shape_equality_does_not_override_nonidentity_or_dynamic_parameters() {
        for dynamic in [false, true] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [1, 4, 3, 3], "torch.float32");
            let size = if dynamic {
                IrArg::Node(placeholder(&mut graph, "dynamic_size", [2], "torch.int32"))
            } else {
                IrArg::None
            };
            let scale = if dynamic {
                IrArg::None
            } else {
                IrArg::List(vec![IrArg::Float(1.1), IrArg::Float(1.1)])
            };
            let resize = call(
                &mut graph,
                "resize",
                "aten.upsample_bilinear2d.vec",
                vec![IrArg::Node(x), size, IrArg::Bool(false), scale],
                [1, 4, 3, 3],
                "torch.float32",
            );
            let add = call(
                &mut graph,
                "add",
                "aten.add.Tensor",
                vec![IrArg::Node(resize), IrArg::Node(x)],
                [1, 4, 3, 3],
                "torch.float32",
            );
            output(&mut graph, add);
            assert!(!IrRemoveNoopPass.run(&mut graph));
            assert!(graph.node(resize).alive);
        }
    }

    #[test]
    fn noop_pad_removal_exposes_duplicate_users_with_unique_outputs() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], "torch.float32");
        let y = placeholder(&mut graph, "y", [2, 3], "torch.float32");
        let padding = call_default(
            &mut graph,
            "padding",
            "backend.tosa.CONST_SHAPE.default",
            vec![i64_list_arg(&[0, 0, 0, 0])],
        );
        let pad = call(
            &mut graph,
            "pad",
            "backend.tosa.PAD.default",
            vec![IrArg::Node(x), IrArg::Node(padding)],
            [2, 3],
            "torch.float32",
        );
        let mut adds = Vec::new();
        for (name, input) in [("first", pad), ("second", x)] {
            adds.push(call(
                &mut graph,
                name,
                "aten.add.Tensor",
                vec![IrArg::Node(input), IrArg::Node(y)],
                [2, 3],
                "torch.float32",
            ));
        }
        let out = output(&mut graph, adds[0]);
        graph.set_args(
            out,
            vec![IrArg::List(adds.into_iter().map(IrArg::Node).collect())],
        );
        assert!(IrRemoveNoopPass.run(&mut graph));
        assert!(crate::fx_fuse_duplicate_users_ir::IrFuseDuplicateUsersPass.run(&mut graph));
        assert!(
            crate::fx_ensure_unique_output_nodes_ir::IrEnsureUniqueOutputNodesPass.run(&mut graph)
        );
        assert!(!graph.node(pad).alive && !graph.node(padding).alive);
        let IrArg::List(outputs) = &graph.node(out).args[0] else {
            panic!("output list")
        };
        let first = graph.node(outputs[0].as_node().unwrap());
        let second = graph.node(outputs[1].as_node().unwrap());
        assert_ne!(first.id, second.id);
        assert_eq!(first.target, "backend.tosa.IDENTITY.default");
        assert_eq!(second.target, "backend.tosa.IDENTITY.default");
        assert_eq!(first.arg_node(0), second.arg_node(0));
        let add = graph.node(first.arg_node(0).unwrap());
        assert_eq!(add.target, "aten.add.Tensor");
        assert_eq!(add.arg_node(0), Some(x));
        assert_eq!(graph.live_call_function_ids().len(), 3);
    }

    #[test]
    fn eliminates_graph_wide_dead_chains_even_without_removable_noops() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], "torch.float32");
        let unused_input = placeholder(&mut graph, "unused", [2, 3], "torch.float32");
        let first = call(
            &mut graph,
            "dead_neg",
            "aten.neg.default",
            vec![IrArg::Node(unused_input)],
            [2, 3],
            "torch.float32",
        );
        let second = call(
            &mut graph,
            "dead_relu",
            "aten.relu.default",
            vec![IrArg::Node(first)],
            [2, 3],
            "torch.float32",
        );
        output(&mut graph, x);
        assert!(IrRemoveNoopPass.run(&mut graph));
        assert!(!graph.node(first).alive && !graph.node(second).alive);
        assert!(graph.node(x).alive && graph.node(unused_input).alive);
        assert!(!IrRemoveNoopPass.run(&mut graph));
    }

    #[test]
    fn regression_copy_casts_source_to_destination_dtype() {
        for (source_dtype, destination_dtype) in [
            ("torch.int32", "torch.float32"),
            ("torch.float32", "torch.int32"),
        ] {
            let mut graph = IrGraph::empty();
            let dst = placeholder(&mut graph, "dst", [2, 3], destination_dtype);
            let src = placeholder(&mut graph, "src", [2, 3], source_dtype);
            let copy = call(
                &mut graph,
                "copy",
                "aten.copy.default",
                vec![IrArg::Node(dst), IrArg::Node(src)],
                [2, 3],
                destination_dtype,
            );
            let out = output(&mut graph, copy);

            assert!(IrRemoveNoopPass.run(&mut graph));
            let cast = graph.node(out).arg_node(0).unwrap();
            assert_eq!(graph.node(cast).target, "aten._to_copy.default");
            assert_eq!(graph.node(cast).arg_node(0), Some(src));
            assert_eq!(graph.node_dtype(cast), Some(destination_dtype));
            assert_eq!(graph.node_shape(cast), Some([2, 3].as_slice()));
            assert!(matches!(
                graph.node(cast).kwargs.get("dtype"),
                Some(IrArg::Dtype(dtype)) if dtype == destination_dtype
            ));
            assert!(!graph.node(copy).alive);
        }
    }

    #[test]
    fn regression_copy_broadcasts_source_to_destination_shape() {
        for (source_shape, aligned_shape, multiples) in [
            (vec![3], vec![1, 3], vec![2, 1]),
            (vec![1, 3], vec![1, 3], vec![2, 1]),
            (vec![2, 1], vec![2, 1], vec![1, 3]),
            (vec![], vec![1, 1], vec![2, 3]),
        ] {
            let mut graph = IrGraph::empty();
            let dst = placeholder(&mut graph, "dst", [2, 3], "torch.float32");
            let src = placeholder(&mut graph, "src", source_shape.clone(), "torch.float32");
            let copy = call(
                &mut graph,
                "copy",
                "aten.copy.default",
                vec![IrArg::Node(dst), IrArg::Node(src)],
                [2, 3],
                "torch.float32",
            );
            let out = output(&mut graph, copy);

            assert!(IrRemoveNoopPass.run(&mut graph));
            let repeat = graph.node(out).arg_node(0).unwrap();
            assert_eq!(graph.node(repeat).target, "aten.repeat.default");
            assert_eq!(graph.node(repeat).arg_int_list(1), Some(multiples));
            assert_eq!(graph.node_shape(repeat), Some([2, 3].as_slice()));
            let aligned = graph.node(repeat).arg_node(0).unwrap();
            assert_eq!(graph.node_shape(aligned), Some(aligned_shape.as_slice()));
            if source_shape.len() < 2 {
                assert_eq!(graph.node(aligned).target, "aten.view_copy.default");
                assert_eq!(graph.node(aligned).arg_node(0), Some(src));
            } else {
                assert_eq!(aligned, src);
            }
        }
    }

    #[test]
    fn regression_copy_casts_before_rank_padding_and_repeat() {
        let mut graph = IrGraph::empty();
        let dst = placeholder(&mut graph, "dst", [2, 3], "torch.float32");
        let src = placeholder(&mut graph, "src", [3], "torch.int32");
        let copy = call(
            &mut graph,
            "copy",
            "aten.copy.default",
            vec![IrArg::Node(dst), IrArg::Node(src)],
            [2, 3],
            "torch.float32",
        );
        let out = output(&mut graph, copy);

        assert!(IrRemoveNoopPass.run(&mut graph));
        let repeat = graph.node(out).arg_node(0).unwrap();
        assert_eq!(graph.node(repeat).target, "aten.repeat.default");
        let view = graph.node(repeat).arg_node(0).unwrap();
        assert_eq!(graph.node(view).target, "aten.view_copy.default");
        let cast = graph.node(view).arg_node(0).unwrap();
        assert_eq!(graph.node(cast).target, "aten._to_copy.default");
        assert_eq!(graph.node(cast).arg_node(0), Some(src));
        for (id, shape) in [(cast, vec![3]), (view, vec![1, 3]), (repeat, vec![2, 3])] {
            assert_eq!(graph.node_dtype(id), Some("torch.float32"));
            assert_eq!(graph.node_shape(id), Some(shape.as_slice()));
        }
    }

    #[test]
    fn identity_copy_returns_source() {
        let mut graph = IrGraph::empty();
        let dst = placeholder(&mut graph, "dst", [2, 3], "torch.float32");
        let src = placeholder(&mut graph, "src", [2, 3], "torch.float32");
        let copy = call(
            &mut graph,
            "copy",
            "aten.copy.default",
            vec![IrArg::Node(dst), IrArg::Node(src)],
            [2, 3],
            "torch.float32",
        );
        let out = output(&mut graph, copy);

        assert!(IrRemoveNoopPass.run(&mut graph));
        assert_eq!(graph.node(out).arg_node(0), Some(src));
    }

    fn quantized_copy_graph(user_qparam: bool) -> (IrGraph, usize, usize, usize) {
        use crate::fx_ir::QParam;

        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x", [4, 8], "torch.int8");
        let copy = call(
            &mut graph,
            "copy",
            "dim_order_ops._to_dim_order_copy.default",
            vec![IrArg::Node(input)],
            [4, 8],
            "torch.float32",
        );
        graph
            .node_mut(copy)
            .meta
            .set_single_output_qparam(QParam::per_tensor(
                1.0,
                0,
                -128,
                127,
                "torch.int8".to_string(),
            ));
        let user = call(
            &mut graph,
            "rescale",
            "backend.tosa.RESCALE.default",
            vec![IrArg::Node(copy)],
            [4, 8],
            "torch.int8",
        );
        if user_qparam {
            graph
                .node_mut(user)
                .meta
                .set_single_input_qparam(QParam::per_tensor(
                    0.02,
                    0,
                    -128,
                    127,
                    "torch.int8".to_string(),
                ));
        }
        let out = output(&mut graph, user);
        (graph, input, copy, out)
    }

    #[test]
    fn quantized_noop_copy_returns_quantized_input() {
        let (mut graph, input, copy, out) = quantized_copy_graph(true);

        assert!(IrRemoveNoopPass.run(&mut graph));
        assert!(!graph.node(copy).alive);
        let rescale = graph.node(out).arg_node(0).unwrap();
        assert_eq!(graph.node(rescale).target, "backend.tosa.RESCALE.default");
        assert_eq!(graph.node(rescale).arg_node(0), Some(input));
    }

    #[test]
    fn quantized_copy_kept_without_quantized_users() {
        let (mut graph, _, copy, _) = quantized_copy_graph(false);

        assert!(!IrRemoveNoopPass.run(&mut graph));
        assert!(graph.node(copy).alive);
    }

    #[test]
    fn regression_copy_float_to_int_truncates_source_before_casting() {
        let mut graph = IrGraph::empty();
        let dst = placeholder(&mut graph, "dst", [2, 3], "torch.int32");
        let src = placeholder(&mut graph, "src", [3], "torch.float32");
        let copy = call(
            &mut graph,
            "copy",
            "aten.copy.default",
            vec![IrArg::Node(dst), IrArg::Node(src)],
            [2, 3],
            "torch.int32",
        );
        let out = output(&mut graph, copy);

        assert!(crate::fx_decorate_fp_int_cast_ir::IrDecorateFpToIntCastPass.run(&mut graph));
        let truncated = graph.node(copy).arg_node(1).unwrap();
        assert_eq!(graph.node(truncated).target, "aten.where.self");
        for (arg, target) in [(1, "aten.floor.default"), (2, "aten.ceil.default")] {
            let rounding = graph.node(truncated).arg_node(arg).unwrap();
            assert_eq!(graph.node(rounding).target, target);
            assert_eq!(graph.node(rounding).arg_node(0), Some(src));
            assert_eq!(graph.node_shape(rounding), Some([3].as_slice()));
        }
        assert_eq!(graph.node(copy).arg_node(0), Some(dst));

        assert!(IrRemoveNoopPass.run(&mut graph));
        let repeat = graph.node(out).arg_node(0).unwrap();
        let view = graph.node(repeat).arg_node(0).unwrap();
        let cast = graph.node(view).arg_node(0).unwrap();
        assert_eq!(graph.node(cast).target, "aten._to_copy.default");
        assert_eq!(graph.node(cast).arg_node(0), Some(truncated));
    }
}
