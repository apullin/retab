use crate::graph::Graph;
use crate::graph::ir::{Arg, DType, Node, NodeId, OpTarget, TensorInfo};

// ---------------------------------------------------------------------------
// Default for TensorInfo — scalar float32
// ---------------------------------------------------------------------------
impl Default for TensorInfo {
    fn default() -> Self {
        TensorInfo {
            shape: [0; 4],
            shape_len: 0,
            dtype: DType::Float32,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the output TensorInfo for a node based on its target, args, and
/// input shapes.  Updates `node.meta.val` in place.
pub fn propagate_shape(graph: &mut Graph, node_id: NodeId) {
    let output = compute_output_info(graph, node_id);
    if let Some(node) = graph.node_mut(node_id) {
        node.meta.val = output;
    }
}

// ---------------------------------------------------------------------------
// Core dispatch
// ---------------------------------------------------------------------------

/// Compute output shape+dtype without mutating the graph.
fn compute_output_info(graph: &Graph, node_id: NodeId) -> TensorInfo {
    let node = match graph.node(node_id) {
        Some(n) => n,
        None => return TensorInfo::default(),
    };
    match &node.target {
        OpTarget::Conv2d | OpTarget::TosaConv2d | OpTarget::TosaDepthwiseConv2d => {
            compute_conv2d(graph, node)
        }

        OpTarget::Relu | OpTarget::Clamp | OpTarget::Hardtanh | OpTarget::TosaClamp => {
            input_info(graph, node, 0)
        }

        OpTarget::Add
        | OpTarget::Sub
        | OpTarget::Mul
        | OpTarget::Div
        | OpTarget::TosaAdd
        | OpTarget::TosaSub
        | OpTarget::TosaMul => compute_broadcast(graph, node),

        OpTarget::Permute | OpTarget::TosaTranspose => compute_permute(graph, node),

        OpTarget::Reshape | OpTarget::View | OpTarget::TosaReshape
        | OpTarget::Flatten | OpTarget::Unsqueeze | OpTarget::Squeeze => {
            compute_reshape(graph, node)
        }

        OpTarget::Full => compute_full(node),

        OpTarget::AvgPool2d
        | OpTarget::MaxPool2d
        | OpTarget::TosaAvgPool2d
        | OpTarget::TosaMaxPool2d => compute_pool2d(graph, node),

        OpTarget::Sum => compute_sum(graph, node),
        OpTarget::TosaReduceSum => compute_tosa_reduce_sum(graph, node),

        // Element-wise unary ops: same shape as input
        OpTarget::Rsqrt
        | OpTarget::Neg
        | OpTarget::Abs
        | OpTarget::Reciprocal
        | OpTarget::Exp
        | OpTarget::Log
        | OpTarget::Sigmoid
        | OpTarget::Tanh
        | OpTarget::Gelu
        | OpTarget::Ceil
        | OpTarget::Floor
        | OpTarget::Sqrt
        | OpTarget::Log1p
        | OpTarget::Expm1
        | OpTarget::Sign
        | OpTarget::Round
        | OpTarget::Pow
        | OpTarget::LeakyRelu
        | OpTarget::TosaTable
        | OpTarget::TosaSigmoid
        | OpTarget::TosaTanh
        | OpTarget::TosaExp
        | OpTarget::TosaLog
        | OpTarget::TosaNeg
        | OpTarget::TosaAbs
        | OpTarget::TosaCeil
        | OpTarget::TosaFloor
        | OpTarget::TosaReciprocal
        | OpTarget::TosaErf
        | OpTarget::TosaCos
        | OpTarget::TosaSin
        | OpTarget::TosaRsqrt
        | OpTarget::TosaNegate => input_info(graph, node, 0),

        OpTarget::Softmax => input_info(graph, node, 0),

        // TosaCast: same shape, different dtype (preserve existing meta dtype)
        OpTarget::TosaCast => {
            let mut info = input_info(graph, node, 0);
            info.dtype = node.meta.val.dtype;
            info
        }

        // Comparison ops: broadcast shape, Bool dtype
        OpTarget::TosaEqual | OpTarget::TosaGreater | OpTarget::TosaGreaterEqual => {
            let mut info = compute_broadcast(graph, node);
            info.dtype = DType::Bool;
            info
        }

        // TosaSelect: output shape = condition shape (first arg)
        OpTarget::TosaSelect => input_info(graph, node, 0),

        // TosaGather: output shape from indices
        OpTarget::TosaGather => {
            let values = input_info(graph, node, 0);
            let indices = input_info(graph, node, 1);
            let v_shape = values.shape_slice();
            let i_shape = indices.shape_slice();
            // gather(values[N,K,C], indices[N,W]) -> [N,W,C]
            if v_shape.len() >= 3 && i_shape.len() >= 2 {
                let n = v_shape[0];
                let w = i_shape[1];
                let c = v_shape[2];
                TensorInfo::new(&[n, w, c], values.dtype)
            } else {
                values
            }
        }

        // TosaMatmul: (B,M,K) x (B,K,N) -> (B,M,N) or (M,K) x (K,N) -> (M,N)
        OpTarget::TosaMatmul | OpTarget::Matmul | OpTarget::Bmm => {
            compute_matmul(graph, node)
        }

        // TosaConcat: sum the concat dimension
        OpTarget::TosaConcat | OpTarget::Cat => compute_concat(graph, node),

        // TosaPad: add padding to dims
        OpTarget::TosaPad | OpTarget::Pad => compute_pad(graph, node),

        // TosaSlice: output from start/size
        OpTarget::TosaSlice | OpTarget::Slice => compute_slice(graph, node),

        OpTarget::Mean | OpTarget::Var | OpTarget::Amax => compute_mean(graph, node),
        OpTarget::TosaReduceMax => compute_tosa_reduce_sum(graph, node),

        OpTarget::Linear => compute_linear(graph, node),
        OpTarget::Addmm => compute_addmm(graph, node),
        OpTarget::GroupNorm => input_info(graph, node, 0),

        OpTarget::QuantizePerTensor => {
            let mut info = input_info(graph, node, 0);
            info.dtype = DType::Int8;
            info
        }

        OpTarget::DequantizePerTensor | OpTarget::DequantizePerChannel => {
            let mut info = input_info(graph, node, 0);
            info.dtype = DType::Float32;
            info
        }

        OpTarget::TosaRescale => {
            // RESCALE preserves shape from input but keeps its own output dtype
            // (set during insertion, e.g. Int8 for int32->int8 requantization).
            let mut info = input_info(graph, node, 0);
            info.dtype = node.meta.val.dtype;
            info
        }

        OpTarget::TosaConst => node.meta.val,

        _ => {
            if !node.args.is_empty() {
                input_info(graph, node, 0)
            } else {
                node.meta.val
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get TensorInfo of the `arg_idx`'th Node argument.
fn input_info(graph: &Graph, node: &Node, arg_idx: usize) -> TensorInfo {
    match node.args.get(arg_idx) {
        Some(Arg::Node(id)) => graph.node(*id).map(|n| n.meta.val).unwrap_or_default(),
        _ => node.meta.val,
    }
}

/// Extract an IntList from the given arg index, returning an empty slice on
/// type mismatch.
fn int_list_arg(node: &Node, idx: usize) -> Vec<i64> {
    match node.args.get(idx) {
        Some(Arg::IntList(v)) => v.to_vec(),
        Some(Arg::Int(v)) => vec![*v],
        _ => Vec::new(),
    }
}

/// Look up a kwarg by name and return it as a bool. Defaults to `default`.
fn kwarg_bool(node: &Node, name: &str, default: bool) -> bool {
    for (k, v) in &node.kwargs {
        if k == name {
            return match v {
                Arg::Bool(b) => *b,
                _ => default,
            };
        }
    }
    default
}

/// Look up a kwarg by name and return it as a Vec<i64>.
fn kwarg_int_list(node: &Node, name: &str) -> Vec<i64> {
    for (k, v) in &node.kwargs {
        if k == name {
            if let Arg::IntList(list) = v {
                return list.to_vec();
            }
        }
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// Conv2d  (ATen & TOSA)
// ---------------------------------------------------------------------------

fn compute_conv2d(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    let weight = input_info(graph, node, 1);

    let in_shape = input.shape_slice();
    let w_shape = weight.shape_slice();

    // Need at least 4-D input and weight
    if in_shape.len() < 4 || w_shape.len() < 4 {
        return input;
    }

    // After RewriteConvPass, TosaConv2d/TosaDepthwiseConv2d nodes store
    // stride/pad/dilation in kwargs, and args are [input, weight, bias,
    // input_zp, weight_zp]. The ATen Conv2d nodes store them in positional
    // args at indices 3,4,5.
    let is_tosa = matches!(
        node.target,
        OpTarget::TosaConv2d | OpTarget::TosaDepthwiseConv2d
    );

    let n = in_shape[0];
    // ATen: NCHW [N,C,H,W]; weight OIHW [OC,IC,KH,KW]
    let h_in = in_shape[2];
    let w_in = in_shape[3];
    let c_out = w_shape[0];
    let k_h = w_shape[2];
    let k_w = w_shape[3];

    let (stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right, dil_h, dil_w) = if is_tosa
    {
        // Read from kwargs
        let stride = kwarg_int_list(node, "stride");
        let pad = kwarg_int_list(node, "pad");
        let dilation = kwarg_int_list(node, "dilation");

        let sh = stride.first().copied().unwrap_or(1);
        let sw = stride.get(1).copied().unwrap_or(sh);
        // TOSA padding: [top, bottom, left, right]
        let pt = pad.first().copied().unwrap_or(0);
        let pb = pad.get(1).copied().unwrap_or(pt);
        let pl = pad.get(2).copied().unwrap_or(0);
        let pr = pad.get(3).copied().unwrap_or(pl);
        let dh = dilation.first().copied().unwrap_or(1);
        let dw = dilation.get(1).copied().unwrap_or(dh);
        (sh, sw, pt, pb, pl, pr, dh, dw)
    } else {
        // ATen Conv2d: positional args [input, weight, bias, stride, padding, dilation]
        let stride = int_list_arg(node, 3);
        let padding = int_list_arg(node, 4);
        let dilation = int_list_arg(node, 5);

        let sh = stride.first().copied().unwrap_or(1);
        let sw = stride.get(1).copied().unwrap_or(sh);
        // ATen padding: [pad_h, pad_w] (symmetric)
        let ph = padding.first().copied().unwrap_or(0);
        let pw = padding.get(1).copied().unwrap_or(ph);
        let dh = dilation.first().copied().unwrap_or(1);
        let dw = dilation.get(1).copied().unwrap_or(dh);
        (sh, sw, ph, ph, pw, pw, dh, dw)
    };

    let h_out = (h_in + pad_top + pad_bottom - dil_h * (k_h - 1) - 1) / stride_h + 1;
    let w_out = (w_in + pad_left + pad_right - dil_w * (k_w - 1) - 1) / stride_w + 1;

    TensorInfo::new(&[n, c_out, h_out, w_out], input.dtype)
}

// ---------------------------------------------------------------------------
// Broadcast (binary element-wise ops)
// ---------------------------------------------------------------------------

fn compute_broadcast(graph: &Graph, node: &Node) -> TensorInfo {
    let a = input_info(graph, node, 0);
    let b = input_info(graph, node, 1);

    let a_shape = a.shape_slice();
    let b_shape = b.shape_slice();

    let out_len = a_shape.len().max(b_shape.len());
    if out_len == 0 {
        return a;
    }

    let mut out = [0i64; 4];
    let out_len = out_len.min(4);

    for i in 0..out_len {
        let a_offset = out_len.saturating_sub(a_shape.len());
        let ai = if i >= a_offset {
            a_shape[i - a_offset]
        } else {
            1
        };
        let b_offset = out_len.saturating_sub(b_shape.len());
        let bi = if i >= b_offset {
            b_shape[i - b_offset]
        } else {
            1
        };
        out[i] = if ai == 1 { bi } else if bi == 1 { ai } else { ai.max(bi) };
    }

    let dtype = broadcast_dtype(a.dtype, b.dtype);
    TensorInfo::new(&out[..out_len], dtype)
}

fn broadcast_dtype(a: DType, b: DType) -> DType {
    use DType::*;
    match (a, b) {
        (Float32, _) | (_, Float32) => Float32,
        (Float16, _) | (_, Float16) => Float16,
        (BFloat16, _) | (_, BFloat16) => BFloat16,
        (Int32, _) | (_, Int32) => Int32,
        (Int64, _) | (_, Int64) => Int64,
        _ => a,
    }
}

// ---------------------------------------------------------------------------
// Permute / TosaTranspose
// ---------------------------------------------------------------------------

fn compute_permute(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    let perm = int_list_arg(node, 1);

    let in_shape = input.shape_slice();
    if perm.is_empty() || perm.len() != in_shape.len() {
        return input;
    }

    let mut out = [0i64; 4];
    let len = perm.len().min(4);
    for i in 0..len {
        let p = perm[i] as usize;
        if p < in_shape.len() {
            out[i] = in_shape[p];
        }
    }

    TensorInfo::new(&out[..len], input.dtype)
}

// ---------------------------------------------------------------------------
// Reshape / View / TosaReshape
// ---------------------------------------------------------------------------

fn compute_reshape(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    let mut new_shape = int_list_arg(node, 1);

    if new_shape.is_empty() {
        return input;
    }

    // Compute total elements from input shape
    let total: i64 = input.shape_slice().iter().product();

    // Find the -1 dimension (if any) and infer it
    let neg_idx = new_shape.iter().position(|&d| d == -1);
    if let Some(idx) = neg_idx {
        let known: i64 = new_shape.iter().filter(|&&d| d != -1).product();
        if known != 0 {
            new_shape[idx] = total / known;
        } else {
            new_shape[idx] = total;
        }
    }

    let len = new_shape.len().min(4);
    TensorInfo::new(&new_shape[..len], input.dtype)
}

// ---------------------------------------------------------------------------
// Full
// ---------------------------------------------------------------------------

fn compute_full(node: &Node) -> TensorInfo {
    let shape = int_list_arg(node, 0);
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
        .unwrap_or(DType::Float32);

    let len = shape.len().min(4);
    if shape.is_empty() {
        return TensorInfo::new(&[], dtype);
    }
    TensorInfo::new(&shape[..len], dtype)
}

// ---------------------------------------------------------------------------
// Pool2d  (AvgPool2d, MaxPool2d, TOSA variants)
// ---------------------------------------------------------------------------

fn compute_pool2d(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    let in_shape = input.shape_slice();

    if in_shape.len() < 4 {
        return input;
    }

    let n = in_shape[0];
    let c = in_shape[1];
    let h_in = in_shape[2];
    let w_in = in_shape[3];

    let kernel = int_list_arg(node, 1);
    let stride = int_list_arg(node, 2);
    let padding = int_list_arg(node, 3);

    let k_h = kernel.first().copied().unwrap_or(1);
    let k_w = kernel.get(1).copied().unwrap_or(k_h);
    let s_h = stride.first().copied().unwrap_or(k_h);
    let s_w = stride.get(1).copied().unwrap_or(s_h);
    let p_h = padding.first().copied().unwrap_or(0);
    let p_w = padding.get(1).copied().unwrap_or(p_h);

    let h_out = if s_h > 0 {
        (h_in + 2 * p_h - k_h) / s_h + 1
    } else {
        h_in
    };
    let w_out = if s_w > 0 {
        (w_in + 2 * p_w - k_w) / s_w + 1
    } else {
        w_in
    };

    TensorInfo::new(&[n, c, h_out, w_out], input.dtype)
}

// ---------------------------------------------------------------------------
// Sum / TosaReduceSum
// ---------------------------------------------------------------------------

fn compute_sum(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    compute_reduction(node, &input, 1)
}

/// TosaReduceSum: single-axis reduction with axis in kwargs. Always keepdim=true.
fn compute_tosa_reduce_sum(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    let in_shape = input.shape_slice();

    let axis = node
        .kwargs
        .iter()
        .find_map(|(k, v)| {
            if k == "axis" {
                if let Arg::Int(a) = v {
                    return Some(*a);
                }
            }
            None
        })
        .unwrap_or(0);

    let axis = if axis < 0 {
        (axis + in_shape.len() as i64) as usize
    } else {
        axis as usize
    };

    let mut out = [0i64; 4];
    let len = in_shape.len().min(4);
    for i in 0..len {
        out[i] = if i == axis { 1 } else { in_shape[i] };
    }
    TensorInfo::new(&out[..len], input.dtype)
}

// ---------------------------------------------------------------------------
// Mean
// ---------------------------------------------------------------------------

fn compute_mean(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    compute_reduction(node, &input, 1)
}

// ---------------------------------------------------------------------------
// Linear: (*, Ci) x (Co, Ci) -> (*, Co)
// ---------------------------------------------------------------------------

fn compute_linear(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    let weight = input_info(graph, node, 1);
    let in_shape = input.shape_slice();
    let w_shape = weight.shape_slice();
    if in_shape.is_empty() || w_shape.is_empty() {
        return input;
    }
    let mut out_shape = in_shape.to_vec();
    // Last dim becomes Co (first dim of weight)
    *out_shape.last_mut().unwrap() = w_shape[0];
    let len = out_shape.len().min(4);
    TensorInfo::new(&out_shape[..len], input.dtype)
}

// ---------------------------------------------------------------------------
// Addmm: addmm(bias, x, weight) -> matmul(x, weight) + bias
// Output shape from matmul: (N, K) x (K, M) -> (N, M)
// ---------------------------------------------------------------------------

fn compute_addmm(graph: &Graph, node: &Node) -> TensorInfo {
    let x = input_info(graph, node, 1);
    let weight = input_info(graph, node, 2);
    let x_shape = x.shape_slice();
    let w_shape = weight.shape_slice();
    if x_shape.len() < 2 || w_shape.len() < 2 {
        return x;
    }
    TensorInfo::new(&[x_shape[0], w_shape[1]], x.dtype)
}

// ---------------------------------------------------------------------------
// Shared reduction logic for sum / mean
// ---------------------------------------------------------------------------

fn compute_reduction(node: &Node, input: &TensorInfo, dims_arg_idx: usize) -> TensorInfo {
    let dims = int_list_arg(node, dims_arg_idx);
    let in_shape = input.shape_slice();

    if dims.is_empty() || in_shape.is_empty() {
        return *input;
    }

    // Determine keepdim: check arg at dims_arg_idx+1, then kwargs
    let keepdim = match node.args.get(dims_arg_idx + 1) {
        Some(Arg::Bool(b)) => *b,
        _ => kwarg_bool(node, "keepdim", false),
    };

    // Normalize negative dim indices
    let ndim = in_shape.len() as i64;
    let reduce_dims: Vec<usize> = dims
        .iter()
        .map(|&d| {
            let d = if d < 0 { d + ndim } else { d };
            d as usize
        })
        .collect();

    if keepdim {
        let mut out = [0i64; 4];
        let len = in_shape.len().min(4);
        for i in 0..len {
            out[i] = if reduce_dims.contains(&i) {
                1
            } else {
                in_shape[i]
            };
        }
        TensorInfo::new(&out[..len], input.dtype)
    } else {
        // Remove reduced dimensions
        let mut out_shape = Vec::new();
        for (i, &s) in in_shape.iter().enumerate() {
            if !reduce_dims.contains(&i) {
                out_shape.push(s);
            }
        }
        let len = out_shape.len().min(4);
        TensorInfo::new(&out_shape[..len], input.dtype)
    }
}

// ---------------------------------------------------------------------------
// Matmul / Bmm / TosaMatmul
// ---------------------------------------------------------------------------

fn compute_matmul(graph: &Graph, node: &Node) -> TensorInfo {
    let a = input_info(graph, node, 0);
    let b = input_info(graph, node, 1);
    let a_shape = a.shape_slice();
    let b_shape = b.shape_slice();

    match (a_shape.len(), b_shape.len()) {
        // (M,K) x (K,N) -> (M,N)
        (2, 2) => TensorInfo::new(&[a_shape[0], b_shape[1]], a.dtype),
        // (B,M,K) x (B,K,N) -> (B,M,N)
        (3, 3) => TensorInfo::new(&[a_shape[0], a_shape[1], b_shape[2]], a.dtype),
        // (B1,B2,M,K) x (B1,B2,K,N) -> (B1,B2,M,N)
        (4, 4) => TensorInfo::new(
            &[a_shape[0], a_shape[1], a_shape[2], b_shape[3]],
            a.dtype,
        ),
        _ => a,
    }
}

// ---------------------------------------------------------------------------
// Concat / Cat / TosaConcat
// ---------------------------------------------------------------------------

fn compute_concat(graph: &Graph, node: &Node) -> TensorInfo {
    // Cat: args[0] = NodeList of tensors, args[1] = dim
    // TosaConcat: similar, with axis in kwargs
    let first_input = match node.args.first() {
        Some(Arg::NodeList(ids)) if !ids.is_empty() => {
            graph.node(ids[0]).map(|n| n.meta.val).unwrap_or_default()
        }
        Some(Arg::Node(_)) => input_info(graph, node, 0),
        _ => return node.meta.val,
    };
    let in_shape = first_input.shape_slice();
    if in_shape.is_empty() {
        return first_input;
    }

    // Get the concat axis
    let axis = match node.args.get(1) {
        Some(Arg::Int(d)) => *d,
        _ => node
            .kwargs
            .iter()
            .find_map(|(k, v)| {
                if k == "axis" {
                    if let Arg::Int(a) = v {
                        return Some(*a);
                    }
                }
                None
            })
            .unwrap_or(0),
    };
    let ndim = in_shape.len() as i64;
    let axis = if axis < 0 { (axis + ndim) as usize } else { axis as usize };

    // Sum the concat dimension across all inputs
    let total_dim: i64 = match node.args.first() {
        Some(Arg::NodeList(ids)) => ids
            .iter()
            .map(|id| {
                graph
                    .node(*id)
                    .map(|n| {
                        let s = n.meta.val.shape_slice();
                        if axis < s.len() { s[axis] } else { 0 }
                    })
                    .unwrap_or(0)
            })
            .sum(),
        _ => in_shape.get(axis).copied().unwrap_or(0),
    };

    let mut out = [0i64; 4];
    let len = in_shape.len().min(4);
    for i in 0..len {
        out[i] = if i == axis { total_dim } else { in_shape[i] };
    }
    TensorInfo::new(&out[..len], first_input.dtype)
}

// ---------------------------------------------------------------------------
// Pad / TosaPad
// ---------------------------------------------------------------------------

fn compute_pad(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    let in_shape = input.shape_slice();
    if in_shape.is_empty() {
        return input;
    }

    // ATen pad: args[1] = IntList of pad values [last_dim_left, last_dim_right, ...]
    // TosaPad: padding in kwargs or args
    let padding = int_list_arg(node, 1);
    if padding.is_empty() {
        return input;
    }

    let mut out = [0i64; 4];
    let len = in_shape.len().min(4);
    for i in 0..len {
        out[i] = in_shape[i];
    }

    // ATen padding is in reverse dimension order: pairs for last dim, then second-to-last, etc.
    let num_pairs = padding.len() / 2;
    for p in 0..num_pairs {
        let dim = len.saturating_sub(1 + p);
        let left = padding[p * 2];
        let right = padding[p * 2 + 1];
        out[dim] += left + right;
    }

    TensorInfo::new(&out[..len], input.dtype)
}

// ---------------------------------------------------------------------------
// Slice / TosaSlice
// ---------------------------------------------------------------------------

fn compute_slice(graph: &Graph, node: &Node) -> TensorInfo {
    let input = input_info(graph, node, 0);
    let in_shape = input.shape_slice();
    if in_shape.is_empty() {
        return input;
    }

    // ATen slice: args = [input, dim, start, end, step]
    // TosaSlice: start/size in kwargs
    let dim = match node.args.get(1) {
        Some(Arg::Int(d)) => {
            let d = *d;
            if d < 0 { (d + in_shape.len() as i64) as usize } else { d as usize }
        }
        _ => {
            // TosaSlice: check kwargs for start/size
            let _start_kw = node.kwargs.iter().find_map(|(k, v)| {
                if k == "start" {
                    if let Arg::IntList(s) = v { return Some(s.to_vec()); }
                }
                None
            });
            let size_kw = node.kwargs.iter().find_map(|(k, v)| {
                if k == "size" {
                    if let Arg::IntList(s) = v { return Some(s.to_vec()); }
                }
                None
            });
            if let Some(size) = size_kw {
                let len = size.len().min(4).min(in_shape.len());
                return TensorInfo::new(&size[..len], input.dtype);
            }
            return input;
        }
    };

    let start = match node.args.get(2) {
        Some(Arg::Int(s)) => (*s).max(0),
        _ => 0,
    };
    let end = match node.args.get(3) {
        Some(Arg::Int(e)) => {
            let e = *e;
            if e < 0 {
                (e + in_shape[dim]).max(0)
            } else {
                e.min(in_shape[dim])
            }
        }
        _ => in_shape[dim],
    };
    let step = match node.args.get(4) {
        Some(Arg::Int(s)) => (*s).max(1),
        _ => 1,
    };

    let mut out = [0i64; 4];
    let len = in_shape.len().min(4);
    for i in 0..len {
        out[i] = in_shape[i];
    }
    if dim < len {
        let sliced = (end - start + step - 1) / step;
        out[dim] = sliced.max(0);
    }
    TensorInfo::new(&out[..len], input.dtype)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{NodeMeta, OpType};
    use smallvec::smallvec;

    /// Helper: build a minimal graph with a single op node that references
    /// input placeholder nodes.
    fn build_graph_with_op(
        input_shapes: &[(&[i64], DType)],
        target: OpTarget,
        args: Vec<Arg>,
        kwargs: Vec<(String, Arg)>,
    ) -> (Graph, NodeId) {
        let mut g = Graph::new();

        // Create placeholder nodes
        let mut input_ids = Vec::new();
        for &(shape, dtype) in input_shapes {
            let info = TensorInfo::new(shape, dtype);
            let node = Node {
                op: OpType::Placeholder,
                target: OpTarget::Other("placeholder".into()),
                args: smallvec![],
                kwargs: vec![],
                meta: NodeMeta::new(info),
                users: smallvec![],
                name: format!("input_{}", input_ids.len()),
            };
            let id = g.nodes.insert(node);
            g.order.push(id);
            input_ids.push(id);
        }

        // Build op args: replace leading entries with Node refs
        let mut final_args: smallvec::SmallVec<[Arg; 4]> = smallvec![];
        let mut input_idx = 0;
        for a in args {
            match a {
                Arg::None if input_idx < input_ids.len() => {
                    // Sentinel: replace with next input node reference
                    final_args.push(Arg::Node(input_ids[input_idx]));
                    input_idx += 1;
                }
                other => final_args.push(other),
            }
        }

        let op_node = Node {
            op: OpType::CallFunction,
            target,
            args: final_args,
            kwargs,
            meta: NodeMeta::new(TensorInfo::default()),
            users: smallvec![],
            name: "op".into(),
        };
        let op_id = g.nodes.insert(op_node);
        g.order.push(op_id);

        (g, op_id)
    }

    #[test]
    fn test_conv2d_shape() {
        // Input: (1,1,28,28), Weight: (16,1,3,3), stride=1, padding=1, dilation=1
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[1, 1, 28, 28], DType::Float32),
                (&[16, 1, 3, 3], DType::Float32),
            ],
            OpTarget::Conv2d,
            vec![
                Arg::None, // input
                Arg::None, // weight
                Arg::None, // bias (unused placeholder — no input shape for it)
                Arg::IntList(smallvec![1, 1]),    // stride
                Arg::IntList(smallvec![1, 1]),    // padding
                Arg::IntList(smallvec![1, 1]),    // dilation
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 28, 28]);
    }

    #[test]
    fn test_conv2d_no_padding() {
        // Input: (1,1,28,28), Weight: (16,1,3,3), stride=1, padding=0, dilation=1
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[1, 1, 28, 28], DType::Float32),
                (&[16, 1, 3, 3], DType::Float32),
            ],
            OpTarget::Conv2d,
            vec![
                Arg::None,
                Arg::None,
                Arg::None,
                Arg::IntList(smallvec![1, 1]),
                Arg::IntList(smallvec![0, 0]),
                Arg::IntList(smallvec![1, 1]),
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 26, 26]);
    }

    #[test]
    fn test_broadcast() {
        // (1,16,1,1) + (1,16,28,28) → (1,16,28,28)
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[1, 16, 1, 1], DType::Float32),
                (&[1, 16, 28, 28], DType::Float32),
            ],
            OpTarget::Add,
            vec![Arg::None, Arg::None],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 28, 28]);
    }

    #[test]
    fn test_broadcast_different_ranks() {
        // (16,28,28) + (1,16,28,28) → (1,16,28,28)
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[16, 28, 28], DType::Float32),
                (&[1, 16, 28, 28], DType::Float32),
            ],
            OpTarget::Add,
            vec![Arg::None, Arg::None],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 28, 28]);
    }

    #[test]
    fn test_reshape_with_neg1() {
        // (1,16,28,28) → (-1, 16, 784) → (1, 16, 784)
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 16, 28, 28], DType::Float32)],
            OpTarget::Reshape,
            vec![
                Arg::None,
                Arg::IntList(smallvec![-1, 16, 784]),
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 784]);
    }

    #[test]
    fn test_reshape_flatten() {
        // (1,32,7,7) → shape [1, 1568] → (1,1568)
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 32, 7, 7], DType::Float32)],
            OpTarget::Reshape,
            vec![
                Arg::None,
                Arg::IntList(smallvec![1, 1568]),
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 1568]);
    }

    #[test]
    fn test_permute() {
        // (1,2,3,4) with perm [0,3,1,2] → (1,4,2,3)
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 2, 3, 4], DType::Float32)],
            OpTarget::Permute,
            vec![
                Arg::None,
                Arg::IntList(smallvec![0, 3, 1, 2]),
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 4, 2, 3]);
    }

    #[test]
    fn test_pool2d() {
        // (1,16,28,28), kernel=2, stride=2, padding=0 → (1,16,14,14)
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 16, 28, 28], DType::Float32)],
            OpTarget::AvgPool2d,
            vec![
                Arg::None,
                Arg::IntList(smallvec![2, 2]),    // kernel
                Arg::IntList(smallvec![2, 2]),    // stride
                Arg::IntList(smallvec![0, 0]),    // padding
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 14, 14]);
    }

    #[test]
    fn test_sum_keepdim() {
        // (1,16,28,28) reduce dim=[2,3] keepdim=true → (1,16,1,1)
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 16, 28, 28], DType::Float32)],
            OpTarget::Sum,
            vec![
                Arg::None,
                Arg::IntList(smallvec![2, 3]),
                Arg::Bool(true),
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 1, 1]);
    }

    #[test]
    fn test_sum_no_keepdim() {
        // (1,16,28,28) reduce dim=[2,3] keepdim=false → (1,16)
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 16, 28, 28], DType::Float32)],
            OpTarget::Sum,
            vec![
                Arg::None,
                Arg::IntList(smallvec![2, 3]),
                Arg::Bool(false),
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16]);
    }

    #[test]
    fn test_mean_keepdim() {
        // (1,16,28,28) mean dim=[2,3] keepdim=true → (1,16,1,1)
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 16, 28, 28], DType::Float32)],
            OpTarget::Mean,
            vec![
                Arg::None,
                Arg::IntList(smallvec![2, 3]),
                Arg::Bool(true),
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 1, 1]);
    }

    #[test]
    fn test_elementwise_preserves_shape() {
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 16, 28, 28], DType::Float32)],
            OpTarget::Relu,
            vec![Arg::None],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 28, 28]);
    }

    #[test]
    fn test_quantize_changes_dtype() {
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 16, 28, 28], DType::Float32)],
            OpTarget::QuantizePerTensor,
            vec![Arg::None],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 28, 28]);
        assert_eq!(info.dtype, DType::Int8);
    }

    #[test]
    fn test_dequantize_changes_dtype() {
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 16, 28, 28], DType::Int8)],
            OpTarget::DequantizePerTensor,
            vec![Arg::None],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 28, 28]);
        assert_eq!(info.dtype, DType::Float32);
    }

    #[test]
    fn test_full() {
        let (mut g, op_id) = build_graph_with_op(
            &[],
            OpTarget::Full,
            vec![Arg::IntList(smallvec![2, 3, 4, 5]), Arg::Float(0.0)],
            vec![("dtype".into(), Arg::Dtype(DType::Float32))],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[2, 3, 4, 5]);
        assert_eq!(info.dtype, DType::Float32);
    }

    #[test]
    fn test_matmul_2d() {
        // (4,3) x (3,5) -> (4,5)
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[4, 3], DType::Float32),
                (&[3, 5], DType::Float32),
            ],
            OpTarget::Matmul,
            vec![Arg::None, Arg::None],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[4, 5]);
    }

    #[test]
    fn test_matmul_3d() {
        // (2,4,3) x (2,3,5) -> (2,4,5)
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[2, 4, 3], DType::Float32),
                (&[2, 3, 5], DType::Float32),
            ],
            OpTarget::TosaMatmul,
            vec![Arg::None, Arg::None],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[2, 4, 5]);
    }

    #[test]
    fn test_concat() {
        // Cat([t1(1,3,4), t2(1,5,4)], dim=1) -> (1,8,4)
        let mut g = Graph::new();
        let t1 = g.nodes.insert(Node {
            op: OpType::Placeholder,
            target: OpTarget::Other("t1".into()),
            args: smallvec![],
            kwargs: vec![],
            meta: NodeMeta::new(TensorInfo::new(&[1, 3, 4], DType::Float32)),
            users: smallvec![],
            name: "t1".into(),
        });
        g.order.push(t1);
        let t2 = g.nodes.insert(Node {
            op: OpType::Placeholder,
            target: OpTarget::Other("t2".into()),
            args: smallvec![],
            kwargs: vec![],
            meta: NodeMeta::new(TensorInfo::new(&[1, 5, 4], DType::Float32)),
            users: smallvec![],
            name: "t2".into(),
        });
        g.order.push(t2);

        let cat_node = Node {
            op: OpType::CallFunction,
            target: OpTarget::Cat,
            args: smallvec![
                Arg::NodeList(smallvec![t1, t2]),
                Arg::Int(1),
            ],
            kwargs: vec![],
            meta: NodeMeta::new(TensorInfo::default()),
            users: smallvec![],
            name: "cat".into(),
        };
        let cat_id = g.nodes.insert(cat_node);
        g.order.push(cat_id);

        propagate_shape(&mut g, cat_id);
        let info = g.node(cat_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 8, 4]);
    }

    #[test]
    fn test_slice() {
        // slice(input(1,10,4), dim=1, start=2, end=7, step=1) -> (1,5,4)
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 10, 4], DType::Float32)],
            OpTarget::Slice,
            vec![
                Arg::None,
                Arg::Int(1),   // dim
                Arg::Int(2),   // start
                Arg::Int(7),   // end
                Arg::Int(1),   // step
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 5, 4]);
    }

    #[test]
    fn test_pad() {
        // pad(input(1,3,4), [1,1,2,2]) -> (1,7,6)
        // ATen pad: last dim gets [1,1], second-to-last gets [2,2]
        let (mut g, op_id) = build_graph_with_op(
            &[(&[1, 3, 4], DType::Float32)],
            OpTarget::Pad,
            vec![
                Arg::None,
                Arg::IntList(smallvec![1, 1, 2, 2]),
            ],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 7, 6]);
    }

    #[test]
    fn test_tosa_comparison_output_dtype() {
        // TosaEqual should produce Bool dtype
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[1, 16], DType::Float32),
                (&[1, 16], DType::Float32),
            ],
            OpTarget::TosaEqual,
            vec![Arg::None, Arg::None],
            vec![],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16]);
        assert_eq!(info.dtype, DType::Bool);
    }

    #[test]
    fn test_elementwise_unary_ops() {
        // All unary ops should preserve shape
        for target in &[
            OpTarget::Sqrt,
            OpTarget::Log1p,
            OpTarget::Expm1,
            OpTarget::Sign,
            OpTarget::Round,
            OpTarget::TosaErf,
            OpTarget::TosaCos,
            OpTarget::TosaSin,
            OpTarget::TosaRsqrt,
        ] {
            let (mut g, op_id) = build_graph_with_op(
                &[(&[2, 3, 4], DType::Float32)],
                target.clone(),
                vec![Arg::None],
                vec![],
            );

            propagate_shape(&mut g, op_id);
            let info = g.node(op_id).unwrap().meta.val;
            assert_eq!(
                info.shape_slice(),
                &[2, 3, 4],
                "Shape mismatch for {:?}",
                target,
            );
        }
    }

    #[test]
    fn test_tosa_conv2d_with_kwargs() {
        // After RewriteConvPass, TosaConv2d nodes have:
        //   args = [input, weight, bias, input_zp, weight_zp]
        //   kwargs = [("pad", [1,1,1,1]), ("stride", [1,1]), ("dilation", [1,1])]
        // Input: (1,1,28,28), Weight: (16,1,3,3), pad=[1,1,1,1] -> (1,16,28,28)
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[1, 1, 28, 28], DType::Float32), // input
                (&[16, 1, 3, 3], DType::Float32),  // weight
                (&[16], DType::Float32),            // bias
            ],
            OpTarget::TosaConv2d,
            vec![
                Arg::None, // input
                Arg::None, // weight
                Arg::None, // bias
            ],
            vec![
                ("pad".into(), Arg::IntList(smallvec![1, 1, 1, 1])),
                ("stride".into(), Arg::IntList(smallvec![1, 1])),
                ("dilation".into(), Arg::IntList(smallvec![1, 1])),
            ],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 28, 28]);
    }

    #[test]
    fn test_tosa_conv2d_no_padding_kwargs() {
        // TosaConv2d with pad=[0,0,0,0]: output shrinks
        // Input: (1,1,28,28), Weight: (16,1,3,3), pad=[0,0,0,0] -> (1,16,26,26)
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[1, 1, 28, 28], DType::Float32),
                (&[16, 1, 3, 3], DType::Float32),
                (&[16], DType::Float32),
            ],
            OpTarget::TosaConv2d,
            vec![
                Arg::None,
                Arg::None,
                Arg::None,
            ],
            vec![
                ("pad".into(), Arg::IntList(smallvec![0, 0, 0, 0])),
                ("stride".into(), Arg::IntList(smallvec![1, 1])),
                ("dilation".into(), Arg::IntList(smallvec![1, 1])),
            ],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 26, 26]);
    }

    #[test]
    fn test_tosa_depthwise_conv2d_kwargs() {
        // TosaDepthwiseConv2d with kwargs
        // Input: (1,16,28,28), Weight: (16,1,3,3), pad=[1,1,1,1] -> (1,16,28,28)
        let (mut g, op_id) = build_graph_with_op(
            &[
                (&[1, 16, 28, 28], DType::Float32),
                (&[16, 1, 3, 3], DType::Float32),
                (&[16], DType::Float32),
            ],
            OpTarget::TosaDepthwiseConv2d,
            vec![
                Arg::None,
                Arg::None,
                Arg::None,
            ],
            vec![
                ("pad".into(), Arg::IntList(smallvec![1, 1, 1, 1])),
                ("stride".into(), Arg::IntList(smallvec![1, 1])),
                ("dilation".into(), Arg::IntList(smallvec![1, 1])),
            ],
        );

        propagate_shape(&mut g, op_id);
        let info = g.node(op_id).unwrap().meta.val;
        assert_eq!(info.shape_slice(), &[1, 16, 28, 28]);
    }
}
