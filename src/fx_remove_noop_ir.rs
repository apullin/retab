//! Pure Rust IrPass version of FxRemoveNoopPass.
//!
//! Removes identity-like nodes from the graph: clone, _clone_dim_order,
//! _to_dim_order_copy, _to_copy, alias_copy, copy, and detach_copy where
//! input and output dtypes match.
//!
//! For `aten.copy.default`, cast and broadcast args[1] (the source) to the
//! destination's dtype and shape. For all others, args[0] is the pass-through
//! value.

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId};
use crate::fx_target::{noop_target, NoopTarget};
use crate::runner_logging::log_line;

pub struct IrRemoveNoopPass;

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
        let modified = graph.rewrite_live_call_functions(|graph, id| {
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

        if modified {
            log_line(format_args!("[IrRemoveNoop] removed noop identity nodes"));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::IrArg;
    use crate::test_ir::{call, output, placeholder};

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

    #[test]
    fn copy_lowering_uses_logical_shapes_before_memory_format_conversion() {
        let mut graph = IrGraph::empty();
        let dst = placeholder(&mut graph, "dst", [1, 2, 3, 4], "torch.float32");
        let src = placeholder(&mut graph, "src", [1, 2, 3, 4], "torch.float32");
        let copy = call(
            &mut graph,
            "copy",
            "aten.copy.default",
            vec![IrArg::Node(dst), IrArg::Node(src)],
            [1, 2, 3, 4],
            "torch.float32",
        );
        let sum = call(
            &mut graph,
            "sum",
            "aten.add.Tensor",
            vec![IrArg::Node(copy), IrArg::Node(src)],
            [1, 2, 3, 4],
            "torch.float32",
        );
        output(&mut graph, sum);

        assert!(IrRemoveNoopPass.run(&mut graph));
        assert!(crate::fx_memory_format_ir::IrToTosaMemoryFormatPass.run(&mut graph));

        assert!(!graph.node(copy).alive);
        assert_eq!(graph.node(sum).arg_node(0), graph.node(sum).arg_node(1));
        // Elementwise-only graphs stay logical (NCHW): no boundary transpose.
        let input = graph.node(sum).arg_node(0).unwrap();
        assert_eq!(graph.node(input).name, "src");
        assert_eq!(
            graph
                .node(sum)
                .meta
                .tosa_dim_order()
                .map(|order| order.to_vec()),
            Some(vec![0, 1, 2, 3])
        );
    }
}
