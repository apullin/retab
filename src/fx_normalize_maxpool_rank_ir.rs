//! IrPass port of ExecuTorch's NormalizeMaxPool2dInputRankPass (v1.6.0-rc4).
//!
//! TOSA MAX_POOL2D needs a rank-4 input, so an unbatched `[C, H, W]` max_pool2d gets a
//! leading unit dim: unsqueeze(0) -> max_pool2d -> squeeze(0). RETAB also sees the
//! tuple-valued `max_pool2d_with_indices`, whose tensor lives on its `getitem(0)`;
//! there the squeeze follows the getitem. Pools whose indices output is used are left
//! alone.

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass, NodeId};
use crate::runner_logging::log_line;

pub struct IrNormalizeMaxPool2dInputRankPass;

fn with_leading_unit(shape: &[i64]) -> Vec<i64> {
    std::iter::once(1).chain(shape.iter().copied()).collect()
}

/// Restamp `value` at rank 4, then insert `squeeze_copy.dims(value, [0])` after it and
/// move its users onto the squeeze.
fn squeeze_after(graph: &mut IrGraph, value: NodeId, rank3_shape: Vec<i64>) {
    let rank4 = graph.nodes[value]
        .meta
        .cloned_with_same_dtype_or_float32(with_leading_unit(&rank3_shape));
    let squeeze_meta = graph.nodes[value]
        .meta
        .cloned_with_same_dtype_or_float32(rank3_shape);
    graph.nodes[value].meta = rank4;
    let name = format!("{}_squeeze", graph.nodes[value].name);
    let squeeze = {
        let mut cursor = graph.rewrite_cursor_after(value, value);
        cursor.call_function(
            name,
            "aten.squeeze_copy.dims",
            vec![IrArg::Node(value), i64_list_arg(&[0])],
            squeeze_meta,
        )
    };
    graph.replace_live_uses_except(value, squeeze, squeeze);
}

fn normalize_one(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    let with_indices = node.target == "aten.max_pool2d_with_indices.default";
    if node.target != "aten.max_pool2d.default" && !with_indices {
        return false;
    }
    let Some(input_id) = node.first_node_arg() else {
        return false;
    };
    let Some(input_shape) = graph.node_shape_vec(input_id) else {
        return false;
    };
    if input_shape.len() != 3 {
        return false;
    }
    // The tensors to restamp at rank 4 and squeeze back: the pool itself, or the
    // getitem(0) users of the tuple-valued variant.
    let tensor_outputs: Vec<(NodeId, Vec<i64>)> = if with_indices {
        let users = graph.live_getitem_users(id);
        if users.is_empty() || users.iter().any(|&(_, index)| index != 0) {
            return false;
        }
        let mut outputs = Vec::new();
        for (getitem, _) in users {
            let Some(shape) = graph.node_shape_vec(getitem) else {
                return false;
            };
            outputs.push((getitem, shape));
        }
        outputs
    } else {
        let Some(shape) = graph.node_shape_vec(id) else {
            return false;
        };
        vec![(id, shape)]
    };

    let name = graph.nodes[id].name.clone();
    let unsqueeze_meta = graph.nodes[input_id]
        .meta
        .cloned_with_same_dtype_or_float32(with_leading_unit(&input_shape));
    let unsqueeze = {
        let mut cursor = graph.rewrite_cursor(id);
        cursor.call_function(
            format!("{name}_unsqueeze"),
            "aten.unsqueeze_copy.default",
            vec![IrArg::Node(input_id), IrArg::Int(0)],
            unsqueeze_meta,
        )
    };
    graph.set_arg(id, 0, IrArg::Node(unsqueeze));
    for (output, shape) in tensor_outputs {
        squeeze_after(graph, output, shape);
    }
    true
}

impl IrPass for IrNormalizeMaxPool2dInputRankPass {
    fn name(&self) -> &str {
        "IrNormalizeMaxPool2dInputRank"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(normalize_one);
        if modified {
            log_line(format_args!(
                "[IrNormalizeMaxPool2dInputRank] lifted rank-3 max_pool2d inputs to rank 4"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call, output, placeholder};

    #[test]
    fn lifts_rank3_max_pool_with_indices_and_squeezes_its_getitem() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", &[3, 9, 11], "torch.int8");
        let pool = call(
            &mut graph,
            "pool",
            "aten.max_pool2d_with_indices.default",
            vec![IrArg::Node(x), i64_list_arg(&[3, 3]), i64_list_arg(&[2, 2])],
            &[],
            "torch.int8",
        );
        let item = call(
            &mut graph,
            "item",
            "<built-in function getitem>",
            vec![IrArg::Node(pool), IrArg::Int(0)],
            &[3, 4, 5],
            "torch.int8",
        );
        output(&mut graph, item);

        assert!(IrNormalizeMaxPool2dInputRankPass.run(&mut graph));
        let unsqueeze = graph.nodes[pool].first_node_arg().unwrap();
        assert_eq!(graph.node_shape_vec(unsqueeze), Some(vec![1, 3, 9, 11]));
        assert_eq!(graph.node_shape_vec(item), Some(vec![1, 3, 4, 5]));
        let out = graph.output_node().unwrap().first_node_arg().unwrap();
        assert_eq!(graph.nodes[out].target, "aten.squeeze_copy.dims");
        assert_eq!(graph.node_shape_vec(out), Some(vec![3, 4, 5]));
        assert_eq!(graph.nodes[out].first_node_arg(), Some(item));
    }

    #[test]
    fn leaves_rank4_pools_alone() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", &[1, 3, 9, 11], "torch.int8");
        let pool = call(
            &mut graph,
            "pool",
            "aten.max_pool2d.default",
            vec![IrArg::Node(x), i64_list_arg(&[3, 3])],
            &[1, 3, 4, 5],
            "torch.int8",
        );
        output(&mut graph, pool);
        assert!(!IrNormalizeMaxPool2dInputRankPass.run(&mut graph));
    }
}
