use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;

/// Convert `FullLike` nodes to `Full` by copying the shape from the input
/// tensor and discarding the input reference.
///
/// FullLike(input, fill_value, ...) -> Full(input_shape, fill_value, ...)
pub struct ConvertFullLikeToFullPass;

impl Pass for ConvertFullLikeToFullPass {
    fn name(&self) -> &str {
        "ConvertFullLikeToFull"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::FullLike);
        if targets.is_empty() {
            return false;
        }

        for fl_id in targets {
            convert_one(graph, fl_id);
        }
        true
    }
}

fn convert_one(graph: &mut Graph, fl_id: NodeId) {
    // Get the input shape and fill value from the FullLike node.
    let (input_shape, fill_value, dtype) = {
        let node = match graph.node(fl_id) {
            Some(n) => n,
            None => return,
        };

        let input_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let input_info = match graph.node(input_id) {
            Some(n) => n.meta.val,
            None => return,
        };

        let fill_value = match node.args.get(1) {
            Some(Arg::Float(f)) => *f,
            Some(Arg::Int(i)) => *i as f64,
            _ => 0.0,
        };

        // Check kwargs for dtype, fall back to input dtype
        let dtype = node
            .kwargs
            .iter()
            .find_map(|(k, v)| {
                if k == "dtype" {
                    if let Arg::Dtype(d) = v {
                        return Some(*d);
                    }
                }
                None
            })
            .unwrap_or(input_info.dtype);

        (input_info.shape_slice().to_vec(), fill_value, dtype)
    };

    // Rewrite the node in place: change target and args.
    if let Some(node) = graph.node_mut(fl_id) {
        let shape_sv: smallvec::SmallVec<[i64; 6]> = input_shape.iter().copied().collect();
        node.target = OpTarget::Full;
        node.args = smallvec::smallvec![
            Arg::IntList(shape_sv),
            Arg::Float(fill_value),
        ];
        node.kwargs = vec![("dtype".into(), Arg::Dtype(dtype))];
    }

    crate::shape::propagate_shape(graph, fl_id);
}
