use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Decompose `Var(x, dims, correction=c, keepdim=k)` into:
///   mean = Mean(x, dims, keepdim=true)
///   diff = Sub(x, mean)
///   sq   = Mul(diff, diff)
///   sum_sq = Sum(sq, dims, keepdim)
///   divisor = Full(shape, 1.0 / max(0, N - correction))
///   result = Mul(sum_sq, divisor)
pub struct DecomposeVarPass;

impl Pass for DecomposeVarPass {
    fn name(&self) -> &str {
        "DecomposeVar"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Var);
        if targets.is_empty() {
            return false;
        }

        for var_id in targets {
            decompose_one(graph, store, var_id);
        }
        true
    }
}

fn decompose_one(graph: &mut Graph, store: &mut TensorStore, var_id: NodeId) {
    let (x_id, dims, correction, keepdim, input_info, output_info) = {
        let node = match graph.node(var_id) {
            Some(n) => n,
            None => return,
        };

        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        // dims: IntList at arg[1]
        let dims: Vec<i64> = match node.args.get(1) {
            Some(Arg::IntList(v)) => v.to_vec(),
            Some(Arg::Int(v)) => vec![*v],
            _ => return,
        };

        // correction: check kwargs first (common pattern), then arg[2]
        let correction = node
            .kwargs
            .iter()
            .find_map(|(k, v)| {
                if k == "correction" {
                    if let Arg::Int(i) = v {
                        return Some(*i);
                    }
                }
                None
            })
            .unwrap_or_else(|| match node.args.get(2) {
                Some(Arg::Int(i)) => *i,
                _ => 1, // default correction=1 (Bessel's)
            });

        let keepdim = node
            .kwargs
            .iter()
            .find_map(|(k, v)| {
                if k == "keepdim" {
                    if let Arg::Bool(b) = v {
                        return Some(*b);
                    }
                }
                None
            })
            .unwrap_or_else(|| match node.args.get(3) {
                Some(Arg::Bool(b)) => *b,
                _ => false,
            });

        let input_info = match graph.node(x_id) {
            Some(n) => n.meta.val,
            None => return,
        };

        let output_info = node.meta.val;

        (x_id, dims, correction, keepdim, input_info, output_info)
    };

    let in_shape = input_info.shape_slice();
    let rank = in_shape.len() as i64;
    let dtype = input_info.dtype;

    // N = product of reduced dim sizes
    let n: i64 = dims
        .iter()
        .map(|&d| {
            let d = if d < 0 { d + rank } else { d };
            in_shape[d as usize]
        })
        .product();

    let denominator = (n - correction).max(0);
    if denominator == 0 {
        return;
    }

    let dims_sv: smallvec::SmallVec<[i64; 6]> = dims.iter().copied().collect();

    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("var_decomp_{suffix}_{counter}")
    };

    let placeholder_meta = || NodeMeta::new(TensorInfo::default());

    // 1. mean = Mean(x, dims, keepdim=true)
    //    Always keepdim=true for the mean so broadcasting works with Sub(x, mean)
    let mean_id = graph.insert_before(
        var_id,
        OpType::CallFunction,
        OpTarget::Mean,
        smallvec![
            Arg::Node(x_id),
            Arg::IntList(dims_sv.clone()),
            Arg::Bool(true),
        ],
        vec![],
        placeholder_meta(),
        next_name("mean"),
    );
    crate::shape::propagate_shape(graph, mean_id);

    // 2. diff = Sub(x, mean)
    let diff_id = graph.insert_before(
        var_id,
        OpType::CallFunction,
        OpTarget::Sub,
        smallvec![Arg::Node(x_id), Arg::Node(mean_id)],
        vec![],
        placeholder_meta(),
        next_name("sub"),
    );
    crate::shape::propagate_shape(graph, diff_id);

    // 3. sq = Mul(diff, diff)
    let sq_id = graph.insert_before(
        var_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(diff_id), Arg::Node(diff_id)],
        vec![],
        placeholder_meta(),
        next_name("mul_sq"),
    );
    crate::shape::propagate_shape(graph, sq_id);

    // 4. sum_sq = Sum(sq, dims, keepdim)
    let sum_id = graph.insert_before(
        var_id,
        OpType::CallFunction,
        OpTarget::Sum,
        smallvec![
            Arg::Node(sq_id),
            Arg::IntList(dims_sv),
            Arg::Bool(keepdim),
        ],
        vec![],
        placeholder_meta(),
        next_name("sum"),
    );
    crate::shape::propagate_shape(graph, sum_id);

    // 5. divisor = Full(output_shape_as_ones, 1.0 / (N - correction))
    let out_rank = output_info.shape_len as usize;
    let full_shape: Vec<i64> = vec![1; out_rank];
    let full_shape_sv: smallvec::SmallVec<[i64; 6]> = full_shape.iter().copied().collect();
    let inv_denom = 1.0 / (denominator as f64);

    let div_name = next_name("full_inv_denom");
    let div_id = if let Some(const_id) =
        insert_scalar_broadcast_const(graph, store, var_id, &div_name, &full_shape, inv_denom, dtype)
    {
        const_id
    } else {
        let full_id = graph.insert_before(
            var_id,
            OpType::CallFunction,
            OpTarget::Full,
            smallvec![Arg::IntList(full_shape_sv), Arg::Float(inv_denom)],
            vec![("dtype".into(), Arg::Dtype(dtype))],
            placeholder_meta(),
            div_name,
        );
        crate::shape::propagate_shape(graph, full_id);
        full_id
    };

    // 6. result = Mul(sum_sq, divisor)
    let result_id = graph.insert_before(
        var_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(sum_id), Arg::Node(div_id)],
        vec![],
        placeholder_meta(),
        next_name("mul_result"),
    );
    crate::shape::propagate_shape(graph, result_id);

    // Preserve output info
    if let Some(result_node) = graph.node_mut(result_id) {
        result_node.meta.val = output_info;
    }

    graph.replace_all_uses(var_id, result_id);
    graph.erase_node(var_id);
}

fn insert_scalar_broadcast_const(
    graph: &mut Graph,
    store: &mut TensorStore,
    ref_id: NodeId,
    name: &str,
    shape: &[i64],
    value: f64,
    dtype: DType,
) -> Option<NodeId> {
    let data = match dtype {
        DType::Float32 => (value as f32).to_le_bytes().to_vec(),
        DType::Int32 => (value as i32).to_le_bytes().to_vec(),
        DType::Int64 => (value as i64).to_le_bytes().to_vec(),
        DType::Int8 => vec![(value as i8) as u8],
        _ => return None,
    };

    store.insert(
        name.to_string(),
        TensorData {
            data,
            shape: shape.to_vec(),
            dtype,
        },
    );

    Some(graph.insert_before(
        ref_id,
        OpType::Placeholder,
        OpTarget::TosaConst,
        smallvec![],
        vec![],
        NodeMeta::new(TensorInfo::new(shape, dtype)),
        name.to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn decompose_var_materializes_const_divisor() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        let input_id = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("input".into()),
            smallvec![],
            vec![],
            NodeMeta::new(TensorInfo::new(&[2, 4], DType::Float32)),
            "input".into(),
        );

        let var_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Var,
            smallvec![
                Arg::Node(input_id),
                Arg::IntList(smallvec![1]),
                Arg::Int(0),
                Arg::Bool(true),
            ],
            vec![],
            NodeMeta::new(TensorInfo::new(&[2, 1], DType::Float32)),
            "var".into(),
        );

        let pass = DecomposeVarPass;
        assert!(pass.run(&mut graph, &mut store));

        assert!(graph.node(var_id).is_none());
        assert!(graph.nodes_by_target(&OpTarget::Full).is_empty());

        let const_ids = graph.nodes_by_target(&OpTarget::TosaConst);
        assert_eq!(const_ids.len(), 1);

        let const_node = graph.node(const_ids[0]).unwrap();
        assert_eq!(const_node.op, OpType::Placeholder);
        assert_eq!(const_node.meta.val.shape_slice(), &[1, 1]);

        let tensor = store.get(&const_node.name).unwrap();
        assert_eq!(tensor.shape, vec![1, 1]);
        assert_eq!(tensor.dtype, DType::Float32);
        assert_eq!(tensor.data, 0.25f32.to_le_bytes().to_vec());

        let mul_ids = graph.nodes_by_target(&OpTarget::Mul);
        assert_eq!(mul_ids.len(), 2);
        let result_mul = graph.node(*mul_ids.last().unwrap()).unwrap();
        assert!(result_mul.args.iter().any(|arg| matches!(arg, Arg::Node(id) if *id == const_ids[0])));
    }
}
