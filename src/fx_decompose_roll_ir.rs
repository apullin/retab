//! IrPass port of ExecuTorch's DecomposeRollPass (v1.6.0-rc4).
//!
//! A static `aten.roll` becomes slices and concatenation. For each `(shift, dim)` pair,
//! with `shift` normalized modulo the dim size `n` and applied sequentially:
//!   result = cat(slice(result, dim, n - shift, n), slice(result, dim, 0, n - shift), dim)
//! Rolls without explicit dims (flattened rolls) are not static decompositions and are
//! left alone, as upstream's `get_static_roll_parameters` does.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId};
use crate::runner_logging::log_line;

pub struct IrDecomposeRollPass;

/// Upstream `get_static_roll_parameters`: normalized `(shift, dim, size)` per pair.
fn static_roll_parameters(
    shape: &[i64],
    shifts: &[i64],
    dims: &[i64],
) -> Option<Vec<(i64, usize, i64)>> {
    if shape.is_empty() || shape.iter().any(|&size| size <= 0) {
        return None;
    }
    if shifts.is_empty() || shifts.len() != dims.len() {
        return None;
    }
    let rank = shape.len() as i64;
    shifts
        .iter()
        .zip(dims)
        .map(|(&shift, &dim)| {
            if !(-rank..rank).contains(&dim) {
                return None;
            }
            let dim = dim.rem_euclid(rank) as usize;
            let size = shape[dim];
            Some((shift.rem_euclid(size), dim, size))
        })
        .collect()
}

fn decompose_one(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if node.target != "aten.roll.default" {
        return false;
    }
    let Some(input_id) = node.first_node_arg() else {
        return false;
    };
    let shifts = node
        .arg_int_list(1)
        .or_else(|| node.arg_int(1).map(|s| vec![s]));
    let dims = node
        .arg_int_list(2)
        .or_else(|| node.arg_int(2).map(|d| vec![d]));
    let (Some(shifts), Some(dims), Some(shape)) = (shifts, dims, graph.node_shape_vec(input_id))
    else {
        return false;
    };
    let Some(parameters) = static_roll_parameters(&shape, &shifts, &dims) else {
        return false;
    };

    let name = node.name.clone();
    let meta = node.meta.clone();
    let mut cursor = graph.rewrite_cursor(id);
    let mut result = input_id;
    for (step, (shift, dim, size)) in parameters.into_iter().enumerate() {
        if shift == 0 {
            continue;
        }
        let slice =
            |cursor: &mut crate::fx_ir::IrRewriteCursor<'_>, tag: &str, source, start, end| {
                let mut slice_shape = shape.clone();
                slice_shape[dim] = end - start;
                cursor.slice_copy(
                    format!("{name}_roll{step}_{tag}"),
                    source,
                    dim as i64,
                    start,
                    end,
                    None,
                    meta.cloned_with_same_dtype_or_float32(slice_shape),
                )
            };
        let tail = slice(&mut cursor, "tail", result, size - shift, size);
        let head = slice(&mut cursor, "head", result, 0, size - shift);
        result = cursor.call_function(
            format!("{name}_roll{step}_cat"),
            "aten.cat.default",
            vec![
                IrArg::List(vec![IrArg::Node(tail), IrArg::Node(head)]),
                IrArg::Int(dim as i64),
            ],
            meta.clone(),
        );
    }
    cursor.replace_anchor_with(result);
    true
}

impl IrPass for IrDecomposeRollPass {
    fn name(&self) -> &str {
        "IrDecomposeRoll"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(decompose_one);
        if modified {
            log_line(format_args!(
                "[IrDecomposeRoll] decomposed static rolls into slice + cat"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::i64_list_arg;
    use crate::test_ir::{call, output, placeholder};

    #[test]
    fn normalizes_shifts_and_dims_like_upstream() {
        assert_eq!(
            static_roll_parameters(&[2, 5], &[-1, 7], &[1, -1]),
            Some(vec![(4, 1, 5), (2, 1, 5)])
        );
        assert_eq!(static_roll_parameters(&[2, 5], &[1], &[]), None);
        assert_eq!(static_roll_parameters(&[2, 5], &[1], &[2]), None);
    }

    #[test]
    fn decomposes_roll_into_tail_head_concat() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", &[2, 5], "torch.int8");
        let roll = call(
            &mut graph,
            "roll",
            "aten.roll.default",
            vec![IrArg::Node(x), i64_list_arg(&[2]), i64_list_arg(&[1])],
            &[2, 5],
            "torch.int8",
        );
        output(&mut graph, roll);
        assert!(IrDecomposeRollPass.run(&mut graph));
        assert!(!graph.nodes[roll].alive);
        let cat = graph.output_node().unwrap().first_node_arg().unwrap();
        assert_eq!(graph.nodes[cat].target, "aten.cat.default");
        let IrArg::List(parts) = &graph.nodes[cat].args[0] else {
            panic!("cat inputs");
        };
        let tail = parts[0].as_node().unwrap();
        assert_eq!(graph.nodes[tail].arg_int(2), Some(3));
        assert_eq!(graph.nodes[tail].arg_int(3), Some(5));
        assert_eq!(graph.node_shape_vec(tail), Some(vec![2, 2]));
    }
}
