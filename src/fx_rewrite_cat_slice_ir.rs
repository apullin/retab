use crate::fx_fuse_consecutive_concats_ir::{cat_dim, cat_inputs};
use crate::fx_ir::{normalize_dim, IrArg, IrGraph, IrNode, IrPass, NodeId};

pub struct IrRewriteCatSlicePass;

struct Overlap {
    input: NodeId,
    start: i64,
    end: i64,
    size: i64,
}

fn slice_range(node: &IrNode, rank: usize, dim_size: i64) -> Option<(usize, i64, i64)> {
    if node.target != "aten.slice_copy.Tensor"
        || node.args.len() < 4
        || node.arg(4).is_some_and(|arg| arg.as_int() != Some(1))
    {
        return None;
    }
    let dim = normalize_dim(node.arg_int(1)?, rank)?;
    let normalize =
        |bound: i64| if bound < 0 { bound + dim_size } else { bound }.clamp(0, dim_size);
    let start = normalize(node.arg_int(2)?);
    let end = normalize(node.arg_int(3)?);
    (start < end).then_some((dim, start, end))
}

fn overlapping_inputs(
    graph: &IrGraph,
    inputs: &[NodeId],
    dim: usize,
    rank: usize,
    start: i64,
    end: i64,
) -> Option<Vec<Overlap>> {
    let mut overlaps = Vec::new();
    let mut offset = 0i64;
    for &input in inputs {
        let shape = graph.logical_input_shape(input)?;
        if shape.len() != rank {
            return None;
        }
        let size = shape[dim];
        let input_end = offset.checked_add(size)?;
        let overlap_start = start.max(offset);
        let overlap_end = end.min(input_end);
        if overlap_start < overlap_end {
            overlaps.push(Overlap {
                input,
                start: overlap_start - offset,
                end: overlap_end - offset,
                size,
            });
        }
        offset = input_end;
    }
    Some(overlaps)
}

impl IrPass for IrRewriteCatSlicePass {
    fn name(&self) -> &str {
        "IrRewriteCatSlice"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for source in graph.topological_live_node_ids() {
            let node = graph.node(source);
            let (Some(inputs), Some(raw_dim), Some(shape)) =
                (cat_inputs(node), cat_dim(node), graph.node_shape(source))
            else {
                continue;
            };
            let rank = shape.len();
            let Some(dim) = normalize_dim(raw_dim, rank) else {
                continue;
            };
            let dim_size = shape[dim];
            let input_qparam = node.meta.input_qparam().cloned();
            let users = graph.live_user_ids(source);
            if users.is_empty() {
                continue;
            }
            let mut replacements = Vec::new();
            let mut op_count = 0;
            for &user in &users {
                let user_node = graph.node(user);
                let Some((slice_dim, start, end)) = slice_range(user_node, rank, dim_size) else {
                    break;
                };
                if user_node.arg_node(0) != Some(source) || slice_dim != dim {
                    break;
                }
                let Some(overlaps) = overlapping_inputs(graph, &inputs, dim, rank, start, end)
                    .filter(|overlaps| !overlaps.is_empty())
                else {
                    break;
                };
                op_count += overlaps
                    .iter()
                    .filter(|overlap| {
                        !(overlap.start == 0
                            && overlap.end == overlap.size
                            && (overlaps.len() > 1 || input_qparam.is_none()))
                    })
                    .count()
                    + usize::from(overlaps.len() > 1);
                replacements.push((user, overlaps));
            }
            if replacements.len() != users.len() || op_count > 1 + users.len() {
                continue;
            }
            for (user, overlaps) in replacements {
                let meta = graph.node(user).meta.clone();
                let name = graph.node(user).name.clone();
                let mut replacement_inputs = Vec::new();
                let mut cursor = graph.rewrite_cursor(user);
                for (index, overlap) in overlaps.iter().enumerate() {
                    let input = if overlap.start == 0
                        && overlap.end == overlap.size
                        && (overlaps.len() > 1 || input_qparam.is_none())
                    {
                        overlap.input
                    } else {
                        let mut shape = cursor.graph().logical_input_shape(overlap.input).unwrap();
                        shape[dim] = overlap.end - overlap.start;
                        let mut slice_meta = meta.cloned_with_same_dtype_or_float32(shape);
                        if let Some(qparam) = &input_qparam {
                            slice_meta.set_single_input_qparam(qparam.clone());
                        }
                        cursor.slice_copy(
                            format!("{name}_input{index}"),
                            overlap.input,
                            dim as i64,
                            overlap.start,
                            overlap.end,
                            Some(1),
                            slice_meta,
                        )
                    };
                    replacement_inputs.push(input);
                }
                let replacement = if replacement_inputs.len() == 1 {
                    replacement_inputs[0]
                } else {
                    let mut cat_meta = meta;
                    if let Some(qparam) = &input_qparam {
                        cat_meta.set_single_input_qparam(qparam.clone());
                    }
                    cursor.call_function(
                        format!("{name}_cat"),
                        "aten.cat.default",
                        vec![
                            IrArg::List(replacement_inputs.into_iter().map(IrArg::Node).collect()),
                            IrArg::Int(dim as i64),
                        ],
                        cat_meta,
                    )
                };
                cursor.replace_anchor_with(replacement);
            }
            modified = true;
        }
        if modified {
            graph.eliminate_dead_code();
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_fuse_consecutive_concats_ir::IrFuseConsecutiveConcatsPass;
    use crate::fx_ir::QParamMap;
    use crate::test_ir::{f32_call, f32_placeholder, output, qparam};

    fn concat_slice_graph(
        channels: &[i64],
        ranges: &[(i64, i64)],
        inserted: bool,
        step: bool,
    ) -> (IrGraph, Vec<NodeId>, NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let inputs: Vec<_> = channels
            .iter()
            .enumerate()
            .map(|(i, &channels)| {
                f32_placeholder(&mut graph, &format!("input{i}"), [1, channels, 8, 8])
            })
            .collect();
        let total = channels.iter().sum::<i64>();
        let source = f32_call(
            &mut graph,
            "source",
            "aten.cat.default",
            vec![
                IrArg::List(inputs.iter().copied().map(IrArg::Node).collect()),
                IrArg::Int(1),
            ],
            [1, total, 8, 8],
        );
        let slices: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(i, &(start, end))| {
                let mut args = vec![
                    IrArg::Node(source),
                    IrArg::Int(1),
                    IrArg::Int(start),
                    IrArg::Int(end),
                ];
                if step {
                    args.push(IrArg::Int(1));
                }
                f32_call(
                    &mut graph,
                    &format!("slice{i}"),
                    "aten.slice_copy.Tensor",
                    args,
                    [1, end.min(total) - start, 8, 8],
                )
            })
            .collect();
        let result = if inserted {
            let inserted = f32_placeholder(&mut graph, "inserted", [1, 1, 8, 8]);
            let mut args = vec![IrArg::Node(slices[0]), IrArg::Node(inserted)];
            args.extend(slices[1..].iter().copied().map(IrArg::Node));
            f32_call(
                &mut graph,
                "result",
                "aten.cat.default",
                vec![IrArg::List(args), IrArg::Int(1)],
                [1, total + 1, 8, 8],
            )
        } else {
            slices[0]
        };
        output(&mut graph, result);
        (graph, inputs, source, result)
    }

    fn run(graph: &mut IrGraph) -> bool {
        let rewritten = IrRewriteCatSlicePass.run(graph);
        IrFuseConsecutiveConcatsPass.run(graph) || rewritten
    }

    #[test]
    fn inserts_on_boundary_without_changing_outer_qparams() {
        let (mut graph, inputs, source, result) =
            concat_slice_graph(&[3, 4, 2], &[(0, 7), (7, 9)], true, true);
        let qparams = QParamMap::from_pairs(
            (0..3).map(|i| (i, qparam(0.25 * (i + 1) as f64, 3, "torch.int8"))),
        );
        graph
            .node_mut(result)
            .meta
            .set_input_qparams(qparams.clone());
        assert!(run(&mut graph));
        assert!(!graph.node(source).alive);
        assert_eq!(graph.live_call_function_ids().len(), 2);
        let result_inputs = cat_inputs(graph.node(result)).unwrap();
        assert_eq!(
            cat_inputs(graph.node(result_inputs[0])),
            Some(inputs[..2].to_vec())
        );
        assert_eq!(graph.node(result_inputs[1]).name, "inserted");
        assert_eq!(result_inputs[2], inputs[2]);
        assert_eq!(
            graph.node(result).meta.sorted_input_qparams(),
            qparams.cloned_pairs()
        );
    }

    #[test]
    fn rejects_multi_user_operation_growth() {
        for ranges in [[(0, 5), (5, 9)], [(0, 5), (6, 9)]] {
            let (mut graph, _, source, result) =
                concat_slice_graph(&[3, 4, 2], &ranges, true, true);
            assert!(!run(&mut graph));
            assert!(graph.node(source).alive && graph.node(result).alive);
        }
    }

    #[test]
    fn rewrites_partial_coverage() {
        let (mut graph, inputs, source, result) =
            concat_slice_graph(&[3, 4], &[(0, 3)], true, true);
        assert!(run(&mut graph));
        assert!(!graph.node(source).alive);
        assert_eq!(graph.live_call_function_ids(), vec![result]);
        let result_inputs = cat_inputs(graph.node(result)).unwrap();
        assert_eq!(result_inputs[0], inputs[0]);
        assert_eq!(graph.node(result_inputs[1]).name, "inserted");
    }

    #[test]
    fn rejects_empty_slice() {
        let (mut graph, _, source, result) = concat_slice_graph(&[0, 0], &[(0, 0)], false, true);
        assert!(!run(&mut graph));
        assert!(graph.node(source).alive && graph.node(result).alive);
    }

    #[test]
    fn rejects_shared_concat() {
        let (mut graph, _, source, result) = concat_slice_graph(&[3, 4], &[(0, 3)], false, true);
        let out = graph.output_node().unwrap().id;
        graph.set_args(
            out,
            vec![IrArg::List(vec![IrArg::Node(source), IrArg::Node(result)])],
        );
        assert!(!IrRewriteCatSlicePass.run(&mut graph));
    }

    #[test]
    fn rejects_three_piece_single_user_growth() {
        let (mut graph, _, source, result) = concat_slice_graph(&[3, 3, 3], &[(1, 8)], false, true);
        assert!(!IrRewriteCatSlicePass.run(&mut graph));
        assert!(graph.node(source).alive && graph.node(result).alive);
    }

    #[test]
    fn rewrites_quantized_full_coverage() {
        let (mut graph, inputs, source, _) = concat_slice_graph(&[3, 4], &[(0, 7)], false, true);
        let qp = qparam(0.25, 3, "torch.int8");
        graph
            .node_mut(source)
            .meta
            .set_single_input_qparam(qp.clone());
        assert!(IrRewriteCatSlicePass.run(&mut graph));
        let cat = graph.node(graph.output_node().unwrap().arg_node(0).unwrap());
        assert_eq!(cat_inputs(cat), Some(inputs));
        assert_eq!(cat.meta.input_qparam(), Some(&qp));
        assert_eq!(graph.live_call_function_ids(), vec![cat.id]);
    }

    #[test]
    fn accepts_implicit_step_and_clamps_end_to_concat_size() {
        let (mut graph, inputs, _, result) =
            concat_slice_graph(&[3, 4], &[(0, 3), (3, i64::MAX)], true, false);
        assert!(run(&mut graph));
        assert_eq!(graph.live_call_function_ids(), vec![result]);
        let result_inputs = cat_inputs(graph.node(result)).unwrap();
        assert_eq!(result_inputs[0], inputs[0]);
        assert_eq!(graph.node(result_inputs[1]).name, "inserted");
        assert_eq!(result_inputs[2], inputs[1]);
    }

    #[test]
    fn preserves_source_input_and_slice_output_qparams() {
        let (mut graph, inputs, source, sliced) =
            concat_slice_graph(&[3, 4], &[(0, 5)], false, true);
        let cat_qp = qparam(0.25, 3, "torch.int8");
        let slice_qp = qparam(0.5, -7, "torch.int8");
        graph
            .node_mut(source)
            .meta
            .set_single_input_qparam(cat_qp.clone());
        graph
            .node_mut(sliced)
            .meta
            .set_single_input_qparam(qparam(0.75, 1, "torch.int8"));
        graph
            .node_mut(sliced)
            .meta
            .set_single_output_qparam(slice_qp.clone());
        assert!(IrRewriteCatSlicePass.run(&mut graph));
        let cat = graph.node(graph.output_node().unwrap().arg_node(0).unwrap());
        let replacements = cat_inputs(cat).unwrap();
        assert_eq!(replacements[0], inputs[0]);
        let slice = graph.node(replacements[1]);
        assert_eq!(slice.target, "aten.slice_copy.Tensor");
        assert_eq!(slice.arg_node(0), Some(inputs[1]));
        assert_eq!(
            (
                slice.arg_int(1),
                slice.arg_int(2),
                slice.arg_int(3),
                slice.arg_int(4)
            ),
            (Some(1), Some(0), Some(2), Some(1))
        );
        assert_eq!(graph.node_shape(slice.id), Some([1, 2, 8, 8].as_slice()));
        for node in [slice, cat] {
            assert_eq!(node.meta.input_qparam(), Some(&cat_qp));
            assert_eq!(node.meta.output_qparam(), Some(&slice_qp));
        }
    }
}
