use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Find nodes where ALL inputs are constant (Placeholder nodes that have data
/// in TensorStore). Evaluate the op in Rust and replace with a pre-computed
/// constant Placeholder.
pub struct FuseConstantArgsPass;

impl Pass for FuseConstantArgsPass {
    fn name(&self) -> &str {
        "FuseConstantArgs"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let mut modified = false;

        // Iterate until no more fusions are possible (cascading constants)
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > 10 {
                eprintln!("[WARN] FuseConstantArgsPass hit max iterations");
                break;
            }
            let mut fused_any = false;
            let all_ids = graph.all_node_ids();

            for node_id in all_ids {
                let node = match graph.node(node_id) {
                    Some(n) => n,
                    None => continue,
                };

                // Skip placeholders and outputs
                if node.op != OpType::CallFunction {
                    continue;
                }

                // Check if all Node-typed args are constant placeholders in the store
                let input_ids: Vec<NodeId> = node
                    .args
                    .iter()
                    .filter_map(|a| match a {
                        Arg::Node(id) => Some(*id),
                        _ => None,
                    })
                    .collect();

                if input_ids.is_empty() {
                    continue;
                }

                let all_const = input_ids.iter().all(|id| {
                    graph
                        .node(*id)
                        .map(|n| {
                            (n.op == OpType::Placeholder
                                || n.target == OpTarget::TosaConst)
                                && store.get(&n.name).is_some()
                        })
                        .unwrap_or(false)
                });

                if !all_const {
                    continue;
                }

                let target = node.target.clone();
                let node_name = node.name.clone();
                let meta_val = node.meta.val;
                let args_clone = node.args.clone();
                let kwargs_clone = node.kwargs.clone();

                // Try to evaluate
                let result = match &target {
                    OpTarget::Add | OpTarget::TosaAdd => {
                        eval_binary_elementwise(graph, store, &input_ids, meta_val, BinOp::Add)
                    }
                    OpTarget::Sub | OpTarget::TosaSub => {
                        eval_binary_elementwise(graph, store, &input_ids, meta_val, BinOp::Sub)
                    }
                    OpTarget::Mul | OpTarget::TosaMul => {
                        eval_binary_elementwise(graph, store, &input_ids, meta_val, BinOp::Mul)
                    }
                    OpTarget::Reshape | OpTarget::TosaReshape | OpTarget::View
                    | OpTarget::Flatten | OpTarget::Unsqueeze | OpTarget::Squeeze => {
                        eval_reshape(graph, store, &input_ids, &args_clone, meta_val)
                    }
                    OpTarget::Permute | OpTarget::TosaTranspose => {
                        eval_permute(graph, store, &input_ids, &args_clone, meta_val)
                    }
                    OpTarget::Full => eval_full(&args_clone, &kwargs_clone, meta_val),
                    _ => None,
                };

                if let Some(result_data) = result {
                    let const_name = format!("{}_fused_const", node_name);
                    store.insert(const_name.clone(), result_data);

                    // Replace the node with a placeholder
                    if let Some(node) = graph.node_mut(node_id) {
                        node.op = OpType::Placeholder;
                        node.target = OpTarget::TosaConst;
                        node.args = smallvec![];
                        node.kwargs = vec![];
                        node.name = const_name;
                    }

                    fused_any = true;
                    modified = true;
                }
            }

            if !fused_any {
                break;
            }
        }

        modified
    }
}

// ---------------------------------------------------------------------------
// Binary operation type
// ---------------------------------------------------------------------------

enum BinOp {
    Add,
    Sub,
    Mul,
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

fn eval_binary_elementwise(
    graph: &Graph,
    store: &TensorStore,
    input_ids: &[NodeId],
    meta_val: TensorInfo,
    op: BinOp,
) -> Option<TensorData> {
    if input_ids.len() < 2 {
        return None;
    }

    let lhs_name = graph.node(input_ids[0])?.name.clone();
    let rhs_name = graph.node(input_ids[1])?.name.clone();

    let lhs_data = store.get(&lhs_name)?;
    let rhs_data = store.get(&rhs_name)?;

    // Use the output dtype from meta
    let out_dtype = meta_val.dtype;
    let out_shape = meta_val.shape_slice().to_vec();

    let result_bytes = match out_dtype {
        DType::Float32 => {
            let lhs_vals = bytes_to_f32(&lhs_data.data);
            let rhs_vals = bytes_to_f32(&rhs_data.data);
            let out = broadcast_binary_op(&lhs_vals, &lhs_data.shape, &rhs_vals, &rhs_data.shape, &out_shape, |a, b| match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
            });
            f32_to_bytes(&out)
        }
        DType::Int32 => {
            let lhs_vals = bytes_to_i32(&lhs_data.data);
            let rhs_vals = bytes_to_i32(&rhs_data.data);
            let out = broadcast_binary_op(&lhs_vals, &lhs_data.shape, &rhs_vals, &rhs_data.shape, &out_shape, |a, b| match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
            });
            i32_to_bytes(&out)
        }
        DType::Int64 => {
            let lhs_vals = bytes_to_i64(&lhs_data.data);
            let rhs_vals = bytes_to_i64(&rhs_data.data);
            let out = broadcast_binary_op(&lhs_vals, &lhs_data.shape, &rhs_vals, &rhs_data.shape, &out_shape, |a, b| match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
            });
            i64_to_bytes(&out)
        }
        DType::Int8 => {
            let lhs_vals = bytes_to_i8(&lhs_data.data);
            let rhs_vals = bytes_to_i8(&rhs_data.data);
            let out = broadcast_binary_op(&lhs_vals, &lhs_data.shape, &rhs_vals, &rhs_data.shape, &out_shape, |a, b| match op {
                BinOp::Add => a.saturating_add(b),
                BinOp::Sub => a.saturating_sub(b),
                BinOp::Mul => a.saturating_mul(b),
            });
            i8_to_bytes(&out)
        }
        _ => return None,
    };

    Some(TensorData {
        data: result_bytes,
        shape: out_shape,
        dtype: out_dtype,
    })
}

fn eval_reshape(
    graph: &Graph,
    store: &TensorStore,
    input_ids: &[NodeId],
    args: &[Arg],
    meta_val: TensorInfo,
) -> Option<TensorData> {
    if input_ids.is_empty() {
        return None;
    }

    let input_name = graph.node(input_ids[0])?.name.clone();
    let input_data = store.get(&input_name)?;

    // Just change shape metadata, data stays the same
    let new_shape = meta_val.shape_slice().to_vec();

    // If meta shape is empty, try to get from args
    let final_shape = if new_shape.is_empty() {
        match args.get(1) {
            Some(Arg::IntList(v)) => v.to_vec(),
            _ => return None,
        }
    } else {
        new_shape
    };

    Some(TensorData {
        data: input_data.data.clone(),
        shape: final_shape,
        dtype: meta_val.dtype,
    })
}

fn eval_permute(
    graph: &Graph,
    store: &TensorStore,
    input_ids: &[NodeId],
    args: &[Arg],
    meta_val: TensorInfo,
) -> Option<TensorData> {
    if input_ids.is_empty() {
        return None;
    }

    let input_name = graph.node(input_ids[0])?.name.clone();
    let input_data = store.get(&input_name)?;

    let perm = match args.get(1) {
        Some(Arg::IntList(v)) => v.to_vec(),
        _ => return None,
    };

    let in_shape = &input_data.shape;
    let ndim = in_shape.len();
    if perm.len() != ndim {
        return None;
    }

    let out_shape: Vec<i64> = perm.iter().map(|&p| in_shape[p as usize]).collect();
    let total_elems: i64 = in_shape.iter().product();

    // Compute strides for input
    let mut in_strides = vec![1i64; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    let elem_size = dtype_size(input_data.dtype);
    let mut out_data = vec![0u8; total_elems as usize * elem_size];

    // For each output element, compute which input element it maps to
    let mut out_strides = vec![1i64; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    for out_flat in 0..total_elems as usize {
        // Convert flat output index to multi-dim
        let mut remaining = out_flat as i64;
        let mut out_coords = vec![0i64; ndim];
        for d in 0..ndim {
            out_coords[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }

        // Map output coords back to input coords via inverse permutation
        let mut in_flat = 0i64;
        for d in 0..ndim {
            let in_dim = perm[d] as usize;
            in_flat += out_coords[d] * in_strides[in_dim];
        }

        let src_offset = in_flat as usize * elem_size;
        let dst_offset = out_flat * elem_size;
        if src_offset + elem_size <= input_data.data.len()
            && dst_offset + elem_size <= out_data.len()
        {
            out_data[dst_offset..dst_offset + elem_size]
                .copy_from_slice(&input_data.data[src_offset..src_offset + elem_size]);
        }
    }

    Some(TensorData {
        data: out_data,
        shape: out_shape,
        dtype: meta_val.dtype,
    })
}

fn eval_full(
    args: &[Arg],
    kwargs: &[(String, Arg)],
    meta_val: TensorInfo,
) -> Option<TensorData> {
    let shape = match args.first() {
        Some(Arg::IntList(v)) => v.to_vec(),
        _ => meta_val.shape_slice().to_vec(),
    };

    let fill_val = match args.get(1) {
        Some(Arg::Float(f)) => *f,
        Some(Arg::Int(i)) => *i as f64,
        _ => 0.0,
    };

    let dtype = kwargs
        .iter()
        .find_map(|(k, v)| {
            if k == "dtype" {
                if let Arg::Dtype(d) = v {
                    return Some(*d);
                }
            }
            None
        })
        .unwrap_or(meta_val.dtype);

    let total: i64 = shape.iter().product();
    if total < 0 {
        return None;
    }

    let data = match dtype {
        DType::Float32 => {
            let val = fill_val as f32;
            let vals = vec![val; total as usize];
            f32_to_bytes(&vals)
        }
        DType::Int32 => {
            let val = fill_val as i32;
            let vals = vec![val; total as usize];
            i32_to_bytes(&vals)
        }
        DType::Int64 => {
            let val = fill_val as i64;
            let vals = vec![val; total as usize];
            i64_to_bytes(&vals)
        }
        DType::Int8 => {
            let val = fill_val as i8;
            let vals = vec![val; total as usize];
            i8_to_bytes(&vals)
        }
        _ => return None,
    };

    Some(TensorData {
        data,
        shape,
        dtype,
    })
}

// ---------------------------------------------------------------------------
// Broadcasting helpers
// ---------------------------------------------------------------------------

/// Compute a broadcasted binary operation on two tensors.
fn broadcast_binary_op<T: Copy>(
    lhs: &[T],
    lhs_shape: &[i64],
    rhs: &[T],
    rhs_shape: &[i64],
    out_shape: &[i64],
    op: impl Fn(T, T) -> T,
) -> Vec<T> {
    let total: usize = out_shape.iter().product::<i64>() as usize;
    if total == 0 {
        return Vec::new();
    }

    let ndim = out_shape.len();

    // Compute strides for lhs, rhs, out — padded to ndim
    let lhs_strides = compute_broadcast_strides(lhs_shape, ndim);
    let rhs_strides = compute_broadcast_strides(rhs_shape, ndim);

    let mut out_strides = vec![1i64; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let mut result = Vec::with_capacity(total);
    for flat in 0..total {
        let mut remaining = flat as i64;
        let mut lhs_idx = 0i64;
        let mut rhs_idx = 0i64;
        for d in 0..ndim {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            lhs_idx += coord * lhs_strides[d];
            rhs_idx += coord * rhs_strides[d];
        }
        let a = lhs[lhs_idx as usize % lhs.len().max(1)];
        let b = rhs[rhs_idx as usize % rhs.len().max(1)];
        result.push(op(a, b));
    }
    result
}

fn compute_broadcast_strides(shape: &[i64], ndim: usize) -> Vec<i64> {
    let rank = shape.len();
    let mut strides = vec![0i64; ndim];
    if rank == 0 {
        return strides;
    }
    // Right-align shape in ndim
    let offset = ndim - rank;
    // Compute physical strides
    let mut phys = vec![1i64; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        phys[i] = phys[i + 1] * shape[i + 1];
    }
    for i in 0..rank {
        // If dimension size is 1, stride is 0 (broadcast)
        strides[offset + i] = if shape[i] == 1 { 0 } else { phys[i] };
    }
    strides
}

// ---------------------------------------------------------------------------
// Byte conversion helpers
// ---------------------------------------------------------------------------

fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Float32 | DType::Int32 => 4,
        DType::Int64 => 8,
        DType::Int8 | DType::Bool => 1,
        DType::Int16 | DType::Float16 | DType::BFloat16 => 2,
        DType::Int4 => 1,
        DType::Int48 => 6,
    }
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_to_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_i32(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32_to_bytes(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_i64(data: &[u8]) -> Vec<i64> {
    data.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn i64_to_bytes(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_i8(data: &[u8]) -> Vec<i8> {
    data.iter().map(|&b| b as i8).collect()
}

fn i8_to_bytes(vals: &[i8]) -> Vec<u8> {
    vals.iter().map(|v| *v as u8).collect()
}
