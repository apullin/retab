//! Pure Rust IrPass: slice_scatter -> slice+cat (step == 1)
//! or arange+permute+index_put (step > 1).

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass};
use crate::fx_target::is_slice_scatter_target;
use crate::runner_logging::log_line;

fn permuted_shape(shape: &[i64], perm: &[i64]) -> Vec<i64> {
    perm.iter()
        .filter_map(|&idx| shape.get(idx as usize).copied())
        .collect()
}

fn fixup_start(start: Option<i64>, dim_size: i64) -> i64 {
    let start = start.unwrap_or(0);
    // PyTorch slice semantics: a negative index is shifted by dim_size ONCE then
    // clamped (not modulo-wrapped), so start < -dim_size clamps to 0.
    let start = if start < 0 { start + dim_size } else { start };
    start.clamp(0, dim_size)
}

fn fixup_end(end: Option<i64>, dim_size: i64) -> i64 {
    let end = end.unwrap_or(dim_size);
    let end = if end < 0 { end + dim_size } else { end };
    end.clamp(0, dim_size)
}

pub struct IrDecomposeSliceScatterPass;

impl IrPass for IrDecomposeSliceScatterPass {
    fn name(&self) -> &str {
        "IrDecomposeSliceScatter"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_slice_scatter_target(&graph.nodes[id].target) {
                continue;
            }

            let args = graph.nodes[id].args.clone();
            let kwargs = graph.nodes[id].kwargs.clone();
            let Some(input_id) = args.first().and_then(IrArg::as_node) else {
                continue;
            };
            let Some(src_id) = args.get(1).and_then(IrArg::as_node) else {
                continue;
            };
            let dim = args
                .get(2)
                .and_then(IrArg::as_int)
                .or_else(|| kwargs.get("dim").and_then(IrArg::as_int));
            let start = args
                .get(3)
                .and_then(IrArg::as_int)
                .or_else(|| kwargs.get("start").and_then(IrArg::as_int));
            let end = args
                .get(4)
                .and_then(IrArg::as_int)
                .or_else(|| kwargs.get("end").and_then(IrArg::as_int));
            let step = args
                .get(5)
                .and_then(IrArg::as_int)
                .or_else(|| kwargs.get("step").and_then(IrArg::as_int))
                .unwrap_or(1);

            let Some(dim) = dim else {
                continue;
            };
            if step <= 0 {
                continue;
            }

            let Some(input_shape) = graph.nodes[input_id].meta.val_shape().map(|s| s.to_vec())
            else {
                continue;
            };
            let Some(src_shape) = graph.nodes[src_id].meta.val_shape().map(|s| s.to_vec()) else {
                continue;
            };
            if input_shape.is_empty() {
                continue;
            }

            let rank = input_shape.len() as i64;
            let dim_norm = dim.rem_euclid(rank) as usize;
            let dim_size = input_shape[dim_norm];
            let start_i = fixup_start(start, dim_size);
            let end_i = fixup_end(end, dim_size);
            if end_i <= start_i {
                graph.replace_and_erase(id, input_id);
                modified = true;
                continue;
            }

            let width = ((end_i - start_i - 1) / step) + 1;
            let mut slice_shape = input_shape.clone();
            slice_shape[dim_norm] = width;
            if src_shape != slice_shape {
                continue;
            }

            let name = graph.nodes[id].name.clone();
            let meta = graph.nodes[id].meta.clone();
            let dtype = graph.nodes[input_id]
                .meta
                .val_dtype_or_float32()
                .to_string();

            let result = if step == 1 {
                // Only cat non-empty slices: TOSA cannot allocate a zero-dim tensor,
                // so an empty prefix (start_i == 0) or empty suffix (end_i == dim_size)
                // must be omitted rather than become a degenerate slice/cat input.
                let mut cat_inputs: Vec<IrArg> = Vec::new();

                if start_i > 0 {
                    let mut prefix_shape = input_shape.clone();
                    prefix_shape[dim_norm] = start_i;
                    let prefix = graph.add_anchored_call_function(
                        id,
                        format!("{}_prefix", name),
                        "aten.slice_copy.Tensor".to_string(),
                        vec![
                            IrArg::Node(input_id),
                            IrArg::Int(dim_norm as i64),
                            IrArg::Int(0),
                            IrArg::Int(start_i),
                            IrArg::Int(1),
                        ],
                        meta.cloned_with_val(prefix_shape, &dtype),
                    );
                    cat_inputs.push(IrArg::Node(prefix));
                }

                cat_inputs.push(IrArg::Node(src_id));

                if end_i < dim_size {
                    let mut suffix_shape = input_shape.clone();
                    suffix_shape[dim_norm] = dim_size - end_i;
                    let suffix = graph.add_anchored_call_function(
                        id,
                        format!("{}_suffix", name),
                        "aten.slice_copy.Tensor".to_string(),
                        vec![
                            IrArg::Node(input_id),
                            IrArg::Int(dim_norm as i64),
                            IrArg::Int(end_i),
                            IrArg::Int(dim_size),
                            IrArg::Int(1),
                        ],
                        meta.cloned_with_val(suffix_shape, &dtype),
                    );
                    cat_inputs.push(IrArg::Node(suffix));
                }

                if cat_inputs.len() == 1 {
                    // Whole tensor scattered (empty prefix and suffix): result is src.
                    src_id
                } else {
                    graph.add_anchored_call_function(
                        id,
                        format!("{}_cat", name),
                        "aten.cat.default".to_string(),
                        vec![IrArg::List(cat_inputs), IrArg::Int(dim_norm as i64)],
                        meta.cloned_with_val(input_shape.clone(), &dtype),
                    )
                }
            } else {
                let arange = graph.add_anchored_arange(
                    id,
                    format!("{}_indices", name),
                    start_i,
                    end_i,
                    step,
                    "torch.int32",
                    meta.cloned_with_val(vec![width], "torch.int32"),
                );

                let mut input_cur = input_id;
                let mut src_cur = src_id;
                let current_input_shape = if dim_norm != 0 {
                    let perm: Vec<i64> = std::iter::once(dim_norm as i64)
                        .chain(
                            (0..input_shape.len())
                                .filter_map(|idx| (idx != dim_norm).then_some(idx as i64)),
                        )
                        .collect();
                    let mut inv_perm = vec![0i64; input_shape.len()];
                    for (idx, &axis) in perm.iter().enumerate() {
                        inv_perm[axis as usize] = idx as i64;
                    }

                    let input_perm = graph.add_anchored_call_function(
                        id,
                        format!("{}_input_perm", name),
                        "aten.permute_copy.default".to_string(),
                        vec![IrArg::Node(input_id), i64_list(&perm)],
                        meta.cloned_with_val(permuted_shape(&input_shape, &perm), &dtype),
                    );
                    input_cur = input_perm;

                    let src_perm = graph.add_anchored_call_function(
                        id,
                        format!("{}_src_perm", name),
                        "aten.permute_copy.default".to_string(),
                        vec![IrArg::Node(src_id), i64_list(&perm)],
                        meta.cloned_with_val(permuted_shape(&src_shape, &perm), &dtype),
                    );
                    src_cur = src_perm;

                    let updated_perm = graph.add_anchored_call_function(
                        id,
                        format!("{}_index_put", name),
                        "aten.index_put.default".to_string(),
                        vec![
                            IrArg::Node(input_cur),
                            IrArg::List(vec![IrArg::Node(arange)]),
                            IrArg::Node(src_cur),
                            IrArg::Bool(false),
                        ],
                        meta.cloned_with_val(permuted_shape(&input_shape, &perm), &dtype),
                    );

                    let updated = graph.add_anchored_call_function(
                        id,
                        format!("{}_result_perm", name),
                        "aten.permute_copy.default".to_string(),
                        vec![IrArg::Node(updated_perm), i64_list(&inv_perm)],
                        meta.cloned_with_val(input_shape.clone(), &dtype),
                    );
                    graph.replace_and_erase(id, updated);
                    modified = true;
                    continue;
                } else {
                    input_shape.clone()
                };

                let updated = graph.add_anchored_call_function(
                    id,
                    format!("{}_index_put", name),
                    "aten.index_put.default".to_string(),
                    vec![
                        IrArg::Node(input_cur),
                        IrArg::List(vec![IrArg::Node(arange)]),
                        IrArg::Node(src_cur),
                        IrArg::Bool(false),
                    ],
                    meta.cloned_with_val(current_input_shape, &dtype),
                );
                updated
            };

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeSliceScatter] decomposed slice_scatter ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{NodeMeta, NodeOp};
    use crate::fx_target::{is_cat_target, is_index_put_default_target};

    #[test]
    fn decompose_slice_scatter_step1_rewrites_to_cat() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor([5, 9], "torch.float32"),
        );
        let src = graph.add_node(
            NodeOp::Placeholder,
            "src".to_string(),
            "src".to_string(),
            vec![],
            NodeMeta::tensor([5, 5], "torch.float32"),
        );
        let node = graph.add_node(
            NodeOp::CallFunction,
            "slice_scatter".to_string(),
            "aten.slice_scatter.default".to_string(),
            vec![
                IrArg::Node(x),
                IrArg::Node(src),
                IrArg::Int(1),
                IrArg::Int(2),
                IrArg::Int(7),
                IrArg::Int(1),
            ],
            NodeMeta::tensor([5, 9], "torch.float32"),
        );

        assert!(IrDecomposeSliceScatterPass.run(&mut graph));
        assert!(!graph.nodes[node].alive);
        assert!(graph
            .nodes
            .iter()
            .any(|n| n.alive && is_cat_target(&n.target)));
    }

    #[test]
    fn decompose_slice_scatter_stepn_rewrites_to_index_put() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            vec![],
            NodeMeta::tensor([2, 4, 6], "torch.float32"),
        );
        let src = graph.add_node(
            NodeOp::Placeholder,
            "src".to_string(),
            "src".to_string(),
            vec![],
            NodeMeta::tensor([2, 4, 2], "torch.float32"),
        );
        let node = graph.add_node(
            NodeOp::CallFunction,
            "slice_scatter".to_string(),
            "aten.slice_scatter.default".to_string(),
            vec![
                IrArg::Node(x),
                IrArg::Node(src),
                IrArg::Int(2),
                IrArg::Int(1),
                IrArg::Int(5),
                IrArg::Int(2),
            ],
            NodeMeta::tensor([2, 4, 6], "torch.float32"),
        );

        assert!(IrDecomposeSliceScatterPass.run(&mut graph));
        assert!(!graph.nodes[node].alive);
        assert!(graph
            .nodes
            .iter()
            .any(|n| n.alive && is_index_put_default_target(&n.target)));
    }
}
