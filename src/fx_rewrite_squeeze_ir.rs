//! Pure Rust IrPass version of FxRewriteSqueezePass.
//!
//! Rewrites squeeze_copy/unsqueeze_copy → view_copy,
//! transpose_copy → permute_copy.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::fx_target::{is_dim_transpose_target, is_squeeze_target, is_unsqueeze_target};
use crate::runner_logging::log_line;

fn compute_squeeze_shape(input_shape: &[i64], dims: &[i64], rank: usize) -> Vec<i64> {
    if dims.is_empty() {
        return input_shape.iter().copied().filter(|&d| d != 1).collect();
    }
    let mut result = Vec::with_capacity(input_shape.len());
    for (i, &dim_size) in input_shape.iter().enumerate() {
        let should_remove = dims.iter().any(|&d| {
            let normalized = if d < 0 { d + rank as i64 } else { d };
            normalized == i as i64 && dim_size == 1
        });
        if !should_remove {
            result.push(dim_size);
        }
    }
    result
}

fn compute_unsqueeze_shape(input_shape: &[i64], dim: i64, rank: usize) -> Vec<i64> {
    let out_rank = rank + 1;
    let normalized = if dim < 0 { dim + out_rank as i64 } else { dim };
    let pos = normalized as usize;
    let mut result = Vec::with_capacity(input_shape.len() + 1);
    for (i, &dim_size) in input_shape.iter().enumerate() {
        if i == pos {
            result.push(1);
        }
        result.push(dim_size);
    }
    if pos >= input_shape.len() {
        result.push(1);
    }
    result
}

pub struct IrRewriteSqueezePass;

impl IrPass for IrRewriteSqueezePass {
    fn name(&self) -> &str {
        "IrRewriteSqueeze"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }

            let target = &graph.nodes[id].target;
            let is_sq = is_squeeze_target(target);
            let is_usq = is_unsqueeze_target(target);
            let is_tr = is_dim_transpose_target(target);
            if !is_sq && !is_usq && !is_tr {
                continue;
            }

            let input_id = match graph.nodes[id].first_node_arg() {
                Some(id) => id,
                None => continue,
            };
            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let input_rank = input_shape.len();

            if is_sq {
                let dims: Vec<i64> = if graph.nodes[id].args.len() > 1 {
                    match &graph.nodes[id].args[1] {
                        IrArg::Int(d) => vec![*d],
                        IrArg::List(items) => items.iter().filter_map(|a| a.as_int()).collect(),
                        _ => Vec::new(),
                    }
                } else {
                    Vec::new()
                };
                let output_shape = compute_squeeze_shape(&input_shape, &dims, input_rank);

                // Create view_copy: replace squeeze with view_copy(input, output_shape)
                let new_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_view_copy", graph.nodes[id].name),
                    "aten.view_copy.default".to_string(),
                    vec![
                        IrArg::Node(input_id),
                        IrArg::List(output_shape.iter().map(|&d| IrArg::Int(d)).collect()),
                    ],
                    graph.nodes[id].meta.clone(),
                );
                graph.replace_and_erase(id, new_id);
                modified = true;
            } else if is_usq {
                let dim = match graph.nodes[id].args.get(1) {
                    Some(IrArg::Int(d)) => *d,
                    _ => continue,
                };
                let output_shape = compute_unsqueeze_shape(&input_shape, dim, input_rank);

                let new_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_view_copy", graph.nodes[id].name),
                    "aten.view_copy.default".to_string(),
                    vec![
                        IrArg::Node(input_id),
                        IrArg::List(output_shape.iter().map(|&d| IrArg::Int(d)).collect()),
                    ],
                    graph.nodes[id].meta.clone(),
                );
                graph.replace_and_erase(id, new_id);
                modified = true;
            } else if is_tr {
                if graph.nodes[id].args.len() < 3 {
                    continue;
                }
                let dim0 = match graph.nodes[id].args[1].as_int() {
                    Some(d) => d,
                    None => continue,
                };
                let dim1 = match graph.nodes[id].args[2].as_int() {
                    Some(d) => d,
                    None => continue,
                };
                let rank = input_rank as i64;
                let d0 = (if dim0 < 0 { dim0 + rank } else { dim0 }) as usize;
                let d1 = (if dim1 < 0 { dim1 + rank } else { dim1 }) as usize;
                let mut perm: Vec<i64> = (0..input_rank as i64).collect();
                perm.swap(d0, d1);

                let new_id = graph.add_node(
                    NodeOp::CallFunction,
                    format!("{}_permute_copy", graph.nodes[id].name),
                    "aten.permute_copy.default".to_string(),
                    vec![
                        IrArg::Node(input_id),
                        IrArg::List(perm.iter().map(|&d| IrArg::Int(d)).collect()),
                    ],
                    graph.nodes[id].meta.clone(),
                );
                graph.replace_and_erase(id, new_id);
                modified = true;
            }
        }

        if modified {
            log_line(format_args!(
                "[IrRewriteSqueeze] rewrote squeeze/unsqueeze/transpose ops"
            ));
        }
        modified
    }
}
