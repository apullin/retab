use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::TensorStore;
use smallvec::smallvec;

/// Rewrite ATen Pad / constant_pad_nd to TOSA Pad.
///
/// PyTorch padding format is pairs from innermost to outermost dimension:
///   (W_left, W_right, H_top, H_bottom, ...)
/// TOSA padding is pairs from outermost to innermost:
///   (N_before, N_after, C_before, C_after, H_before, H_after, W_before, W_after)
pub struct RewritePadPass;

impl Pass for RewritePadPass {
    fn name(&self) -> &str {
        "RewritePad"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let target = match graph.node(node_id) {
                Some(n) => n.target.clone(),
                None => continue,
            };

            if target == OpTarget::Pad {
                if rewrite_pad(graph, node_id) {
                    modified = true;
                }
            }
        }

        modified
    }
}

fn rewrite_pad(graph: &mut Graph, node_id: NodeId) -> bool {
    let (input_id, pytorch_pad, pad_value, input_rank) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        let input_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };

        let pytorch_pad: Vec<i64> = match node.args.get(1) {
            Some(Arg::IntList(v)) => v.to_vec(),
            _ => return false,
        };

        let pad_value = match node.args.get(2) {
            Some(Arg::Int(v)) => *v,
            Some(Arg::Float(v)) => *v as i64,
            _ => 0,
        };

        let input_rank = graph
            .node(input_id)
            .map(|n| n.meta.val.shape_len as usize)
            .unwrap_or(0);

        (input_id, pytorch_pad, pad_value, input_rank)
    };

    if input_rank == 0 {
        return false;
    }

    // Convert PyTorch padding (innermost-first pairs) to TOSA (outermost-first pairs).
    // PyTorch: (left, right, top, bottom, ...)
    // We need to reverse the pair order and zero-pad leading dims.
    let num_pad_pairs = pytorch_pad.len() / 2;
    let mut tosa_pad = vec![0i64; input_rank * 2];

    // Fill from the innermost dimension backwards
    for i in 0..num_pad_pairs {
        let dim_idx = input_rank - 1 - i;
        if dim_idx < input_rank {
            tosa_pad[dim_idx * 2] = pytorch_pad[i * 2];
            tosa_pad[dim_idx * 2 + 1] = pytorch_pad[i * 2 + 1];
        }
    }

    let tosa_pad_sv: smallvec::SmallVec<[i64; 6]> = tosa_pad.iter().copied().collect();

    if let Some(node) = graph.node_mut(node_id) {
        node.target = OpTarget::TosaPad;
        node.args = smallvec![Arg::Node(input_id)];
        node.kwargs = vec![
            ("padding".into(), Arg::IntList(tosa_pad_sv)),
            ("pad_value".into(), Arg::Int(pad_value)),
        ];
    }

    shape::propagate_shape(graph, node_id);

    true
}
