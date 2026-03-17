use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;
use smallvec::smallvec;

/// Decompose `Mean(x, dims, keepdim)` into:
///   sum = Sum(x, dims, keepdim)
///   count_tensor = Full(broadcast_shape, 1.0 / count)
///   result = Mul(sum, count_tensor)
///
/// where count = product of sizes along reduced dimensions.
pub struct DecomposeMeanDimPass;

impl Pass for DecomposeMeanDimPass {
    fn name(&self) -> &str {
        "DecomposeMeanDim"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Mean);
        if targets.is_empty() {
            return false;
        }

        for mean_id in targets {
            decompose_one(graph, mean_id);
        }
        true
    }
}

fn decompose_one(graph: &mut Graph, mean_id: NodeId) {
    let (x_id, dims, keepdim, input_info, output_info) = {
        let node = match graph.node(mean_id) {
            Some(n) => n,
            None => return,
        };

        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let dims: Vec<i64> = match node.args.get(1) {
            Some(Arg::IntList(v)) => v.to_vec(),
            Some(Arg::Int(v)) => vec![*v],
            _ => return,
        };

        let keepdim = match node.args.get(2) {
            Some(Arg::Bool(b)) => *b,
            _ => {
                // Check kwargs
                node.kwargs
                    .iter()
                    .find_map(|(k, v)| {
                        if k == "keepdim" {
                            if let Arg::Bool(b) = v {
                                return Some(*b);
                            }
                        }
                        None
                    })
                    .unwrap_or(false)
            }
        };

        let input_info = match graph.node(x_id) {
            Some(n) => n.meta.val,
            None => return,
        };

        let output_info = node.meta.val;

        (x_id, dims, keepdim, input_info, output_info)
    };

    let in_shape = input_info.shape_slice();
    let rank = in_shape.len() as i64;
    let dtype = input_info.dtype;

    // Compute count = product of reduced dim sizes
    let count: i64 = dims
        .iter()
        .map(|&d| {
            let d = if d < 0 { d + rank } else { d };
            in_shape[d as usize]
        })
        .product();

    if count == 0 {
        return;
    }

    let dims_sv: smallvec::SmallVec<[i64; 6]> = dims.iter().copied().collect();

    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("meandim_decomp_{suffix}_{counter}")
    };

    let placeholder_meta = || NodeMeta::new(TensorInfo::default());

    // 1. sum = Sum(x, dims, keepdim)
    let sum_id = graph.insert_before(
        mean_id,
        OpType::CallFunction,
        OpTarget::Sum,
        smallvec![
            Arg::Node(x_id),
            Arg::IntList(dims_sv),
            Arg::Bool(keepdim),
        ],
        vec![],
        placeholder_meta(),
        next_name("sum"),
    );
    crate::shape::propagate_shape(graph, sum_id);

    // 2. count_tensor = Full(output_shape_as_ones, 1.0/count)
    // Use a scalar-like shape that broadcasts: rank 0 or [1]*rank
    let out_rank = output_info.shape_len as usize;
    let full_shape: Vec<i64> = vec![1; out_rank];
    let full_shape_sv: smallvec::SmallVec<[i64; 6]> = full_shape.iter().copied().collect();
    let inv_count = 1.0 / (count as f64);

    let count_id = graph.insert_before(
        mean_id,
        OpType::CallFunction,
        OpTarget::Full,
        smallvec![Arg::IntList(full_shape_sv), Arg::Float(inv_count)],
        vec![("dtype".into(), Arg::Dtype(dtype))],
        placeholder_meta(),
        next_name("full_inv_count"),
    );
    crate::shape::propagate_shape(graph, count_id);

    // 3. result = Mul(sum, count_tensor)
    let mul_id = graph.insert_before(
        mean_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(sum_id), Arg::Node(count_id)],
        vec![],
        placeholder_meta(),
        next_name("mul"),
    );
    crate::shape::propagate_shape(graph, mul_id);

    // Preserve output info from original mean node
    if let Some(mul_node) = graph.node_mut(mul_id) {
        mul_node.meta.val = output_info;
    }

    graph.replace_all_uses(mean_id, mul_id);
    graph.erase_node(mean_id);
}
