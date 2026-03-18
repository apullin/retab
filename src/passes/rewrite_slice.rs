use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::TensorStore;
use smallvec::smallvec;

/// Rewrite ATen Slice to TOSA Slice.
///
/// ATen slice args: [input, dim, start, end, step?]
/// TOSA slice takes: input + start_indices + sizes as kwargs.
pub struct RewriteSlicePass;

impl Pass for RewriteSlicePass {
    fn name(&self) -> &str {
        "RewriteSlice"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let target = match graph.node(node_id) {
                Some(n) => n.target.clone(),
                None => continue,
            };

            if target == OpTarget::Slice {
                if rewrite_slice(graph, node_id) {
                    modified = true;
                }
            }
        }

        modified
    }
}

fn rewrite_slice(graph: &mut Graph, node_id: NodeId) -> bool {
    let (input_id, dim, start, end, input_shape) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        let input_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };

        let dim = match node.args.get(1) {
            Some(Arg::Int(d)) => *d,
            _ => 0,
        };

        let start = match node.args.get(2) {
            Some(Arg::Int(s)) => *s,
            Some(Arg::None) => 0,
            _ => 0,
        };

        let end = match node.args.get(3) {
            Some(Arg::Int(e)) => *e,
            Some(Arg::None) => i64::MAX,
            _ => i64::MAX,
        };

        let input_shape = graph
            .node(input_id)
            .map(|n| n.meta.val.shape_slice().to_vec())
            .unwrap_or_default();

        (input_id, dim, start, end, input_shape)
    };

    if input_shape.is_empty() {
        return false;
    }

    let rank = input_shape.len();
    let dim = if dim < 0 {
        (dim + rank as i64) as usize
    } else {
        dim as usize
    };

    if dim >= rank {
        return false;
    }

    let dim_size = input_shape[dim];

    // Fixup start: handle negative and out-of-bounds
    let start = if start < 0 {
        (start % dim_size).max(0)
    } else if start >= dim_size {
        dim_size - 1
    } else {
        start
    };

    // Fixup end: clamp to dim_size
    let end = if end < 0 {
        (end % dim_size).max(0)
    } else {
        end.min(dim_size)
    };

    // Build start_indices and sizes
    let mut start_list = vec![0i64; rank];
    start_list[dim] = start;

    // Output shape: same as input except sliced dim
    let output_shape = graph
        .node(node_id)
        .map(|n| n.meta.val.shape_slice().to_vec())
        .unwrap_or_else(|| {
            let mut s = input_shape.clone();
            s[dim] = end - start;
            s
        });

    let start_sv: smallvec::SmallVec<[i64; 6]> = start_list.iter().copied().collect();
    let size_sv: smallvec::SmallVec<[i64; 6]> = output_shape.iter().copied().collect();

    if let Some(node) = graph.node_mut(node_id) {
        node.target = OpTarget::TosaSlice;
        node.args = smallvec![Arg::Node(input_id)];
        node.kwargs = vec![
            ("start".into(), Arg::IntList(start_sv)),
            ("size".into(), Arg::IntList(size_sv)),
        ];
    }

    shape::propagate_shape(graph, node_id);

    true
}
