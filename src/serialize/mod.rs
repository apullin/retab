pub mod tosa_generated;
pub mod visitors;

use crate::fx_graph::FxGraphModule;
use crate::graph::ir::{Arg, Node, OpTarget, OpType};
use crate::graph::Graph;
use crate::store::{TensorData, TensorStore};

use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, Vector, WIPOffset};
use std::collections::HashMap;
use tosa_generated::{Attribute, DType, Op, TosaOperator, TosaShape, TosaTensor, TransposeAttribute};
use visitors::ir_dtype_to_tosa;

// ---------------------------------------------------------------------------
// Weight permutation helpers (OIHW -> OHWI for TOSA)
// ---------------------------------------------------------------------------

/// Describes how a weight tensor should be permuted for TOSA.
#[derive(Debug, Clone, Copy)]
enum WeightPermKind {
    /// Standard conv2d: OIHW [OC, IC, KH, KW] -> OHWI [OC, KH, KW, IC]
    Conv2d,
    /// Standard conv3d: OIDHW [OC, IC, KD, KH, KW] -> ODHWI [OC, KD, KH, KW, IC]
    Conv3d,
    /// Depthwise conv2d: OIHW [OC, 1, KH, KW] -> HWIM [KH, KW, IC, M]
    /// where IC is the true input-channel count and M = depth_multiplier.
    DepthwiseConv2d { in_channels: usize },
}

fn apply_tosa_dim_order(node: &crate::graph::ir::Node, shape: Vec<i32>) -> Vec<i32> {
    let Some(order) = node.meta.tosa_dim_order.as_ref() else {
        return shape;
    };
    if order.len() != shape.len() {
        return shape;
    }
    order
        .iter()
        .filter_map(|&src_idx| shape.get(src_idx as usize).copied())
        .collect()
}

fn raw_i32_shape(node: &Node) -> Vec<i32> {
    node.meta
        .val
        .shape_slice()
        .iter()
        .map(|&dim| dim as i32)
        .collect()
}

fn node_tensor_dtype(node: &Node) -> DType {
    node.meta
        .tosa_special_dtype
        .map(ir_dtype_to_tosa)
        .unwrap_or_else(|| ir_dtype_to_tosa(node.meta.val.dtype))
}

fn inverse_tosa_dim_order(order: &[u8]) -> Option<Vec<i32>> {
    let mut inverse = vec![0i32; order.len()];
    for (idx, &axis) in order.iter().enumerate() {
        let axis = axis as usize;
        if axis >= order.len() {
            return None;
        }
        inverse[axis] = idx as i32;
    }
    Some(inverse)
}

fn infer_shape_permutation(actual_shape: &[i32], raw_shape: &[i32]) -> Option<Vec<i32>> {
    if actual_shape.len() != raw_shape.len() {
        return None;
    }

    let mut used = vec![false; actual_shape.len()];
    let mut perms = Vec::with_capacity(raw_shape.len());
    for &raw_dim in raw_shape {
        let pos = actual_shape
            .iter()
            .enumerate()
            .find_map(|(idx, &actual_dim)| (!used[idx] && actual_dim == raw_dim).then_some(idx))?;
        used[pos] = true;
        perms.push(pos as i32);
    }

    Some(perms)
}

fn permute_shape(shape: &[i32], perms: &[i32]) -> Option<Vec<i32>> {
    if shape.len() != perms.len() {
        return None;
    }

    perms
        .iter()
        .map(|&idx| shape.get(idx as usize).copied())
        .collect()
}

/// Permute convolution weight tensors into their physical TOSA layout.
///
/// For Conv2d (OIHW -> OHWI):
///   perm = [0, 2, 3, 1]
///   output[o][h][w][i] = input[o][i][h][w]
///
/// For DepthwiseConv2d (OIHW -> HWIM):
///   perm = [2, 3, 1, 0]
///   output[h][w][i][m] = input[m][i][h][w]
///   where OC = in_channels * depth_multiplier, I = 1 (per group), M = depth_multiplier
fn permute_weight_data(
    data: &[u8],
    shape: &[i64],
    elem_size: usize,
    kind: WeightPermKind,
) -> (Vec<u8>, Vec<i64>) {
    let expected_rank = match kind {
        WeightPermKind::Conv3d => 5,
        _ => 4,
    };
    if shape.len() != expected_rank {
        return (data.to_vec(), shape.to_vec());
    }

    let (d0, d1, d2, d3) = (
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
    );

    match kind {
        WeightPermKind::Conv2d => {
            // OIHW -> OHWI: perm [0, 2, 3, 1]
            let new_shape = vec![shape[0], shape[2], shape[3], shape[1]];
            let (nh, nw, ni) = (d2, d3, d1);
            let mut out = vec![0u8; data.len()];

            for o in 0..d0 {
                for i in 0..d1 {
                    for h in 0..d2 {
                        for w in 0..d3 {
                            let src_idx = ((o * d1 + i) * d2 + h) * d3 + w;
                            let dst_idx = ((o * nh + h) * nw + w) * ni + i;
                            let src_off = src_idx * elem_size;
                            let dst_off = dst_idx * elem_size;
                            if src_off + elem_size <= data.len() && dst_off + elem_size <= out.len()
                            {
                                out[dst_off..dst_off + elem_size]
                                    .copy_from_slice(&data[src_off..src_off + elem_size]);
                            }
                        }
                    }
                }
            }
            (out, new_shape)
        }
        WeightPermKind::Conv3d => {
            let (d0, d1, d2, d3, d4) = (
                shape[0] as usize,
                shape[1] as usize,
                shape[2] as usize,
                shape[3] as usize,
                shape[4] as usize,
            );
            let new_shape = vec![shape[0], shape[2], shape[3], shape[4], shape[1]];
            let (nd, nh, nw, ni) = (d2, d3, d4, d1);
            let mut out = vec![0u8; data.len()];

            for o in 0..d0 {
                for i in 0..d1 {
                    for d in 0..d2 {
                        for h in 0..d3 {
                            for w in 0..d4 {
                                let src_idx = ((((o * d1 + i) * d2 + d) * d3 + h) * d4) + w;
                                let dst_idx = ((((o * nd + d) * nh + h) * nw + w) * ni) + i;
                                let src_off = src_idx * elem_size;
                                let dst_off = dst_idx * elem_size;
                                if src_off + elem_size <= data.len()
                                    && dst_off + elem_size <= out.len()
                                {
                                    out[dst_off..dst_off + elem_size]
                                        .copy_from_slice(&data[src_off..src_off + elem_size]);
                                }
                            }
                        }
                    }
                }
            }
            (out, new_shape)
        }
        WeightPermKind::DepthwiseConv2d { in_channels } => {
            // Parse, don't validate: depthwise serialization needs the real
            // input-channel count to produce TOSA's HWIM layout. The previous
            // logic only worked for IC=1 and silently serialized shared
            // depthwise weights as [KH, KW, 1, OC].
            if in_channels == 0 || d1 != 1 || d0 % in_channels != 0 {
                return (data.to_vec(), shape.to_vec());
            }

            let depth_multiplier = d0 / in_channels;
            if depth_multiplier == 0 {
                return (data.to_vec(), shape.to_vec());
            }

            let new_shape = vec![
                shape[2],
                shape[3],
                in_channels as i64,
                depth_multiplier as i64,
            ];
            let (_nh, nw, ni, nm) = (d2, d3, in_channels, depth_multiplier);
            let mut out = vec![0u8; data.len()];

            for oc in 0..d0 {
                let ic = oc / depth_multiplier;
                let m = oc % depth_multiplier;
                for h in 0..d2 {
                    for w in 0..d3 {
                        let src_idx = ((oc * d1) * d2 + h) * d3 + w;
                        let dst_idx = ((h * nw + w) * ni + ic) * nm + m;
                        let src_off = src_idx * elem_size;
                        let dst_off = dst_idx * elem_size;
                        if src_off + elem_size <= data.len() && dst_off + elem_size <= out.len() {
                            out[dst_off..dst_off + elem_size]
                                .copy_from_slice(&data[src_off..src_off + elem_size]);
                        }
                    }
                }
            }
            (out, new_shape)
        }
    }
}

/// Permute a tensor's raw bytes according to an arbitrary output-axis order.
///
/// `order[i]` gives the source/input axis for output axis `i`.
/// Example: `[0, 2, 3, 1]` maps NCHW -> NHWC.
fn permute_tensor_by_order(
    data: &[u8],
    shape: &[i64],
    elem_size: usize,
    order: &[u8],
) -> (Vec<u8>, Vec<i64>) {
    if shape.len() != order.len() || shape.is_empty() {
        return (data.to_vec(), shape.to_vec());
    }

    if order
        .iter()
        .enumerate()
        .all(|(idx, &axis)| idx == axis as usize)
    {
        return (data.to_vec(), shape.to_vec());
    }

    let rank = shape.len();
    let numel: usize = match shape.iter().try_fold(1usize, |acc, &dim| {
        usize::try_from(dim).ok().and_then(|d| acc.checked_mul(d))
    }) {
        Some(v) => v,
        None => return (data.to_vec(), shape.to_vec()),
    };

    if numel == 0 || numel.saturating_mul(elem_size) != data.len() {
        return (data.to_vec(), shape.to_vec());
    }

    let new_shape: Vec<i64> = order
        .iter()
        .filter_map(|&axis| shape.get(axis as usize).copied())
        .collect();
    if new_shape.len() != rank {
        return (data.to_vec(), shape.to_vec());
    }

    let mut input_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        input_strides[i] = input_strides[i + 1]
            .checked_mul(shape[i + 1] as usize)
            .unwrap_or(0);
    }

    let mut output_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        output_strides[i] = output_strides[i + 1]
            .checked_mul(new_shape[i + 1] as usize)
            .unwrap_or(0);
    }

    let mut out = vec![0u8; data.len()];
    let mut coords = vec![0usize; rank];
    let mut dst_coords = vec![0usize; rank];

    for src_linear in 0..numel {
        let mut rem = src_linear;
        for axis in 0..rank {
            let stride = input_strides[axis];
            coords[axis] = if stride == 0 { 0 } else { rem / stride };
            rem = if stride == 0 { 0 } else { rem % stride };
        }

        for (out_axis, &in_axis) in order.iter().enumerate() {
            dst_coords[out_axis] = coords[in_axis as usize];
        }

        let mut dst_linear = 0usize;
        for axis in 0..rank {
            dst_linear += dst_coords[axis] * output_strides[axis];
        }

        let src_off = src_linear * elem_size;
        let dst_off = dst_linear * elem_size;
        out[dst_off..dst_off + elem_size].copy_from_slice(&data[src_off..src_off + elem_size]);
    }

    (out, new_shape)
}

/// Element size in bytes for a given DType.
fn dtype_elem_size(dtype: crate::graph::ir::DType) -> usize {
    use crate::graph::ir::DType;
    match dtype {
        DType::Bool | DType::Int8 | DType::Int4 => 1,
        DType::Int16 | DType::Float16 | DType::BFloat16 => 2,
        DType::Int32 | DType::Float32 => 4,
        DType::Int48 => 6,
        DType::Int64 => 8,
    }
}

fn pack_int4_data(data: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((data.len() + 1) / 2);
    let mut idx = 0usize;
    while idx < data.len() {
        let lo = data[idx] & 0x0F;
        let hi = if idx + 1 < data.len() {
            data[idx + 1] & 0x0F
        } else {
            0
        };
        packed.push(lo | (hi << 4));
        idx += 2;
    }
    packed
}

fn widen_int32_data_to_int48(data: &[u8]) -> Vec<u8> {
    let mut widened = Vec::with_capacity((data.len() / 4) * 6);
    for chunk in data.chunks_exact(4) {
        let value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64;
        let bytes = value.to_le_bytes();
        widened.extend_from_slice(&bytes[..6]);
    }
    widened
}

/// Build a map of weight tensor names to their permutation kind by scanning
/// all conv ops in the graph.
fn infer_depthwise_in_channels(graph: &Graph, node: &crate::graph::ir::Node, oc: usize) -> usize {
    let Some(Arg::Node(input_id)) = node.args.first() else {
        return 0;
    };
    let Some(input_node) = graph.node(*input_id) else {
        return 0;
    };

    let shape = input_node.meta.val.shape_slice();
    for candidate in [shape.last().copied(), shape.get(1).copied()]
        .into_iter()
        .flatten()
    {
        let candidate = candidate as usize;
        if candidate > 0 && oc % candidate == 0 {
            return candidate;
        }
    }

    0
}

fn collect_weight_permutations(graph: &Graph) -> HashMap<String, WeightPermKind> {
    let mut perms = HashMap::new();
    for &nid in &graph.order {
        let Some(node) = graph.node(nid) else {
            continue;
        };
        let kind = match &node.target {
            OpTarget::TosaConv2d | OpTarget::Conv2d => Some(WeightPermKind::Conv2d),
            OpTarget::TosaConv3d => Some(WeightPermKind::Conv3d),
            OpTarget::TosaDepthwiseConv2d | OpTarget::DepthwiseConv2d => {
                let Some(Arg::Node(weight_id)) = node.args.get(1) else {
                    continue;
                };
                let Some(weight_node) = graph.node(*weight_id) else {
                    continue;
                };
                let oc = weight_node
                    .meta
                    .val
                    .shape_slice()
                    .first()
                    .copied()
                    .unwrap_or(0) as usize;
                let in_channels = infer_depthwise_in_channels(graph, node, oc);
                if in_channels == 0 {
                    continue;
                }
                Some(WeightPermKind::DepthwiseConv2d { in_channels })
            }
            _ => None,
        };
        if let Some(kind) = kind {
            // Weight is arg[1]
            if let Some(Arg::Node(weight_id)) = node.args.get(1) {
                if let Some(weight_node) = graph.node(*weight_id) {
                    // Only permute placeholder/const nodes with the expected conv weight rank.
                    let expected_rank = match kind {
                        WeightPermKind::Conv3d => 5,
                        _ => 4,
                    };
                    if weight_node.meta.val.shape_len == expected_rank {
                        perms.insert(weight_node.name.clone(), kind);
                    }
                }
            }
        }
    }
    perms
}

fn make_graph_tensor(name: String, shape: Vec<i32>, dtype: DType, data: Vec<u8>) -> TosaTensor {
    TosaTensor {
        name,
        shape,
        dtype,
        data,
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    }
}

fn placeholder_decl(node: &Node) -> (Vec<i32>, DType) {
    let raw_shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();
    (
        apply_tosa_dim_order(node, raw_shape),
        ir_dtype_to_tosa(node.meta.val.dtype),
    )
}

fn serialize_const_placeholder(
    node: &Node,
    tensor_data: &TensorData,
    weight_perms: &HashMap<String, WeightPermKind>,
) -> TosaTensor {
    let store_dtype = ir_dtype_to_tosa(tensor_data.dtype);
    let tensor_dtype = node
        .meta
        .tosa_special_dtype
        .map(ir_dtype_to_tosa)
        .unwrap_or(store_dtype);
    let elem_sz = dtype_elem_size(tensor_data.dtype);

    let (final_data, final_shape_i64) = if let Some(&perm_kind) = weight_perms.get(&node.name) {
        permute_weight_data(&tensor_data.data, &tensor_data.shape, elem_sz, perm_kind)
    } else if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        permute_tensor_by_order(&tensor_data.data, &tensor_data.shape, elem_sz, order)
    } else {
        (tensor_data.data.clone(), tensor_data.shape.clone())
    };
    let final_shape: Vec<i32> = final_shape_i64.iter().map(|&d| d as i32).collect();
    let final_data = match (tensor_dtype, store_dtype) {
        (DType::Int4, _) if store_dtype != DType::Int4 => pack_int4_data(&final_data),
        (DType::Int48, DType::Int32) => widen_int32_data_to_int48(&final_data),
        _ => final_data,
    };

    make_graph_tensor(node.name.clone(), final_shape, tensor_dtype, final_data)
}

fn append_output_names(graph: &Graph, arg: &Arg, output_names: &mut Vec<String>) {
    match arg {
        Arg::Node(id) => {
            if let Some(node) = graph.node(*id) {
                output_names.push(node.name.clone());
            }
        }
        Arg::NodeList(ids) => {
            for id in ids {
                if let Some(node) = graph.node(*id) {
                    output_names.push(node.name.clone());
                }
            }
        }
        _ => {}
    }
}

fn append_normalized_output_names(
    graph: &Graph,
    store: &TensorStore,
    arg: &Arg,
    tensors: &mut Vec<TosaTensor>,
    operators: &mut Vec<TosaOperator>,
    output_names: &mut Vec<String>,
) {
    let mut append_one = |node: &Node| {
        let raw_shape = raw_i32_shape(node);
        let serialized_shape = apply_tosa_dim_order(node, raw_shape.clone());
        let actual_shape = visitors::actual_tensor_shape(graph, store, node);
        let preserves_non_identity_layout = node
            .meta
            .tosa_dim_order
            .as_ref()
            .map(|order| {
                order
                    .iter()
                    .enumerate()
                    .any(|(idx, &axis)| axis as usize != idx)
            })
            .unwrap_or(false);
        let has_explicit_output_layout = matches!(node.target, OpTarget::TosaTranspose);
        if node.meta.arm_control_flow_parent_output_user && raw_shape.len() == 4 {
            if let Some(perms) = node
                .meta
                .tosa_dim_order
                .as_ref()
                .and_then(|order| inverse_tosa_dim_order(order))
            {
                if let Some(output_shape) = permute_shape(&serialized_shape, &perms) {
                    let output_name = format!("{}_output_layout", node.name);
                    tensors.push(TosaTensor::new(
                        output_name.clone(),
                        output_shape,
                        node_tensor_dtype(node),
                    ));
                    operators.push(TosaOperator::new(
                        Op::Transpose,
                        Attribute::Transpose(TransposeAttribute { perms }),
                        vec![node.name.clone()],
                        vec![output_name.clone()],
                    ));
                    output_names.push(output_name);
                    return;
                }
            }
        }
        // External outputs that still advertise a non-identity dim order are
        // intentionally user-visible channels-last tensors. Keep those in the
        // serializer's native layout so the runner can decode them using the
        // FX fake tensor metadata instead of fighting a hidden extra transpose.
        if raw_shape.len() != 4
            || actual_shape == raw_shape
            || preserves_non_identity_layout
            || has_explicit_output_layout
        {
            output_names.push(node.name.clone());
            return;
        }

        let perms = node
            .meta
            .tosa_dim_order
            .as_ref()
            .and_then(|order| {
                (apply_tosa_dim_order(node, raw_shape.clone()) == actual_shape)
                    .then(|| inverse_tosa_dim_order(order))
                    .flatten()
            })
            .or_else(|| infer_shape_permutation(&actual_shape, &raw_shape));
        let Some(perms) = perms else {
            output_names.push(node.name.clone());
            return;
        };

        let output_name = format!("{}_output_layout", node.name);
        tensors.push(TosaTensor::new(
            output_name.clone(),
            raw_shape,
            node_tensor_dtype(node),
        ));
        operators.push(TosaOperator::new(
            Op::Transpose,
            Attribute::Transpose(TransposeAttribute { perms }),
            vec![node.name.clone()],
            vec![output_name.clone()],
        ));
        output_names.push(output_name);
    };

    match arg {
        Arg::Node(id) => {
            if let Some(node) = graph.node(*id) {
                append_one(node);
            }
        }
        Arg::NodeList(ids) => {
            for id in ids {
                if let Some(node) = graph.node(*id) {
                    append_one(node);
                }
            }
        }
        _ => append_output_names(graph, arg, output_names),
    }
}

/// Serialize the IR graph and its constant data into a TOSA flatbuffer byte
/// vector.
struct SerializedGraphItems {
    tensors: Vec<TosaTensor>,
    shapes: Vec<TosaShape>,
    operators: Vec<TosaOperator>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

fn collect_serialized_graph_items(
    graph: &Graph,
    store: &TensorStore,
    normalize_outputs: bool,
) -> SerializedGraphItems {
    // Use insertion order directly instead of topological sort.
    //
    // The FX graph is topologically ordered by construction (Python FX enforces
    // this), and ingestion walks nodes in that order, so `graph.order` is
    // already a valid topological ordering.  IR passes that create new nodes
    // via `insert_before`/`insert_after`/`create_node` also maintain this
    // invariant.
    //
    // The Kahn-based topological_sort() has a subtle inconsistency between
    // in-degree computation (from args) and propagation (from user lists)
    // that causes large numbers of nodes to be flagged as "stale" and
    // appended out of order, producing malformed TOSA flatbuffers.
    let sorted = graph.all_node_ids();
    eprintln!("[Serialize] using insertion order, {} nodes", sorted.len());

    // Debug-mode validation: verify insertion order is topological
    if FxGraphModule::debug() {
        let mut position: std::collections::HashMap<crate::graph::ir::NodeId, usize> =
            std::collections::HashMap::new();
        for (i, &id) in sorted.iter().enumerate() {
            position.insert(id, i);
        }
        let mut violations = 0;
        for (i, &id) in sorted.iter().enumerate() {
            if let Some(node) = graph.node(id) {
                let input_ids = Graph::collect_input_ids(node);
                for input_id in &input_ids {
                    if let Some(&input_pos) = position.get(input_id) {
                        if input_pos >= i {
                            if violations < 10 {
                                eprintln!(
                                    "[Serialize] WARNING: '{}' at position {} depends on input at position {}",
                                    node.name, i, input_pos
                                );
                            }
                            violations += 1;
                        }
                    }
                }
            }
        }
        if violations > 0 {
            eprintln!(
                "[Serialize] WARNING: {} topological order violations detected",
                violations
            );
        }
    }

    // Pre-compute which weight tensors need OIHW -> OHWI permutation
    let weight_perms = collect_weight_permutations(graph);

    let mut tensors: Vec<TosaTensor> = Vec::new();
    let mut shapes: Vec<TosaShape> = Vec::new();
    let mut operators: Vec<TosaOperator> = Vec::new();
    let mut input_names: Vec<String> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();

    for &node_id in &sorted {
        let Some(node) = graph.node(node_id) else {
            continue;
        };

        if FxGraphModule::debug() {
            eprintln!(
                "[Serialize] node '{}' op={:?} target={:?}",
                node.name, node.op, node.target
            );
        }

        match node.op {
            OpType::Placeholder => {
                let name = &node.name;
                let (shape, dtype) = placeholder_decl(node);

                if let Some(tensor_data) = store.get(name) {
                    tensors.push(serialize_const_placeholder(
                        node,
                        tensor_data,
                        &weight_perms,
                    ));
                    operators.push(TosaOperator::new(
                        tosa_generated::Op::Const,
                        Attribute::Const(tosa_generated::ConstAttribute),
                        vec![],
                        vec![name.clone()],
                    ));
                } else {
                    tensors.push(make_graph_tensor(name.clone(), shape, dtype, Vec::new()));
                    input_names.push(name.clone());
                }
            }
            OpType::CallFunction => {
                if let Some(emit_result) = visitors::emit_node(graph, node_id, node, store) {
                    tensors.extend(emit_result.tensors);
                    shapes.extend(emit_result.shapes);
                    operators.extend(emit_result.operators);
                } else {
                    eprintln!(
                        "[serialize] WARNING: skipping unhandled call_function node '{}' target {:?}",
                        node.name, node.target
                    );
                }
            }
            OpType::Output => {
                for arg in &node.args {
                    if normalize_outputs {
                        append_normalized_output_names(
                            graph,
                            store,
                            arg,
                            &mut tensors,
                            &mut operators,
                            &mut output_names,
                        );
                    } else {
                        append_output_names(graph, arg, &mut output_names);
                    }
                }
            }
        }
    }

    eprintln!(
        "[Serialize] topo sort done, collected {} tensors, {} operators, {} shapes",
        tensors.len(),
        operators.len(),
        shapes.len()
    );

    SerializedGraphItems {
        tensors,
        shapes,
        operators,
        input_names,
        output_names,
    }
}

fn build_fb_region<'fbb>(
    builder: &mut FlatBufferBuilder<'fbb>,
    region_name: &str,
    graph: &Graph,
    store: &TensorStore,
) -> WIPOffset<TableFinishedWIPOffset> {
    // Region outputs are only true external program outputs on the root graph.
    // Control-flow subgraphs feed COND_IF/WHILE_LOOP directly and must retain
    // their internal layout instead of being normalized back to the user-facing
    // output shape at region boundaries.
    let normalize_outputs = region_name == "main";
    let serialized = collect_serialized_graph_items(graph, store, normalize_outputs);

    // Build the flatbuffer bottom-up
    // Pre-serialize all sub-tables
    eprintln!("[Serialize] building fb tensors...");
    let fb_tensors = build_fb_tensors(builder, &serialized.tensors);
    eprintln!("[Serialize] building fb shapes...");
    let fb_shapes = build_fb_shapes(builder, &serialized.shapes);
    eprintln!("[Serialize] building fb operators...");
    let fb_operators = build_fb_operators(builder, &serialized.operators);
    eprintln!("[Serialize] assembling graph...");

    // Build input/output name vectors
    let fb_input_strs: Vec<_> = serialized
        .input_names
        .iter()
        .map(|s| builder.create_string(s))
        .collect();
    let fb_output_strs: Vec<_> = serialized
        .output_names
        .iter()
        .map(|s| builder.create_string(s))
        .collect();
    let fb_inputs = builder.create_vector(&fb_input_strs);
    let fb_outputs = builder.create_vector(&fb_output_strs);

    let fb_tensors_vec = builder.create_vector(&fb_tensors);
    let fb_shapes_vec = builder.create_vector(&fb_shapes);
    let fb_operators_vec = builder.create_vector(&fb_operators);

    // Build basic block
    let block_name = builder.create_string(region_name);
    let block = {
        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, block_name);
        builder.push_slot_always::<WIPOffset<_>>(6, fb_operators_vec);
        builder.push_slot_always::<WIPOffset<_>>(8, fb_tensors_vec);
        builder.push_slot_always::<WIPOffset<_>>(10, fb_inputs);
        builder.push_slot_always::<WIPOffset<_>>(12, fb_outputs);
        builder.push_slot_always::<WIPOffset<_>>(14, fb_shapes_vec);
        builder.end_table(start)
    };

    // Build region
    let blocks_vec = builder.create_vector(&[block]);
    let region_name = builder.create_string(region_name);
    let region = {
        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, region_name);
        builder.push_slot_always::<WIPOffset<_>>(6, blocks_vec);
        builder.end_table(start)
    };

    region
}

fn finish_tosa_graph(
    builder: &mut FlatBufferBuilder<'_>,
    regions: &[WIPOffset<TableFinishedWIPOffset>],
) -> Vec<u8> {
    // Build software version
    let modifier_str = builder.create_string("arm-tosa-engine");
    let sw_version = {
        let start = builder.start_table();
        builder.push_slot::<i32>(4, 0, -1); // major
        builder.push_slot::<i32>(6, 1, -1); // minor
        builder.push_slot::<i32>(8, 0, -1); // micro
        builder.push_slot_always::<WIPOffset<_>>(10, modifier_str);
        builder.end_table(start)
    };

    // Build version
    let version = {
        let start = builder.start_table();
        builder.push_slot::<i32>(4, 1, -1); // major
        builder.push_slot::<i32>(6, 0, -1); // minor
        builder.push_slot::<i32>(8, 0, -1); // patch
        builder.push_slot::<bool>(10, false, true); // draft
        builder.end_table(start)
    };

    // Build root TosaGraph
    let regions_vec = builder.create_vector(regions);
    let tosa_graph = {
        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, version);
        builder.push_slot_always::<WIPOffset<_>>(6, regions_vec);
        builder.push_slot_always::<WIPOffset<_>>(8, sw_version);
        builder.end_table(start)
    };

    builder.finish(tosa_graph, Some("TOSA"));
    builder.finished_data().to_vec()
}

pub fn serialize_graphs_to_tosa(graphs: &[(String, Graph)], store: &TensorStore) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::with_capacity(16 * 1024 * 1024); // 16MB initial
    let regions: Vec<_> = graphs
        .iter()
        .map(|(name, graph)| build_fb_region(&mut builder, name, graph, store))
        .collect();
    finish_tosa_graph(&mut builder, &regions)
}

// ============================================================================
// Flatbuffer builder helpers
// ============================================================================

fn create_force_aligned_u8_vector<'fbb>(
    builder: &mut FlatBufferBuilder<'fbb>,
    data: &[u8],
) -> WIPOffset<Vector<'fbb, u8>> {
    // The flatbuffers Rust runtime does not expose schema-level vector
    // `force_align` directly. Treating each `u8` as an 8-byte-aligned element
    // corrupts the payload (e.g. int8 weights become interleaved with zeros).
    //
    // Dense byte data is required for TOSA tensor buffers, so emit a normal
    // `[ubyte]` vector here.
    builder.create_vector(data)
}

fn build_fb_tensors(
    builder: &mut FlatBufferBuilder<'_>,
    tensors: &[TosaTensor],
) -> Vec<WIPOffset<TableFinishedWIPOffset>> {
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        let name = builder.create_string(&t.name);
        let shape = builder.create_vector(&t.shape);
        let data = if !t.data.is_empty() {
            Some(create_force_aligned_u8_vector(builder, &t.data))
        } else {
            None
        };
        let var_name = if !t.variable_name.is_empty() {
            Some(builder.create_string(&t.variable_name))
        } else {
            None
        };

        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, name); // name
        builder.push_slot_always::<WIPOffset<_>>(6, shape); // shape
        builder.push_slot::<u32>(8, t.dtype as u32, 0); // type
        if let Some(d) = data {
            builder.push_slot_always::<WIPOffset<_>>(10, d); // data
        }
        builder.push_slot::<bool>(12, t.variable, false); // variable
        builder.push_slot::<bool>(14, t.is_unranked, false); // is_unranked
        if let Some(vn) = var_name {
            builder.push_slot_always::<WIPOffset<_>>(16, vn); // variable_name
        }
        offsets.push(builder.end_table(start));
    }
    offsets
}

fn build_fb_shapes(
    builder: &mut FlatBufferBuilder<'_>,
    shapes: &[TosaShape],
) -> Vec<WIPOffset<TableFinishedWIPOffset>> {
    let mut offsets = Vec::with_capacity(shapes.len());
    for s in shapes {
        let name = builder.create_string(&s.name);
        let data = if !s.data.is_empty() {
            Some(create_force_aligned_u8_vector(builder, &s.data))
        } else {
            None
        };

        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, name);
        builder.push_slot::<u32>(6, s.rank, 0);
        if let Some(d) = data {
            builder.push_slot_always::<WIPOffset<_>>(8, d);
        }
        offsets.push(builder.end_table(start));
    }
    offsets
}

fn build_fb_operators(
    builder: &mut FlatBufferBuilder<'_>,
    operators: &[TosaOperator],
) -> Vec<WIPOffset<TableFinishedWIPOffset>> {
    let mut offsets = Vec::with_capacity(operators.len());
    for op in operators {
        let input_strs: Vec<_> = op.inputs.iter().map(|s| builder.create_string(s)).collect();
        let output_strs: Vec<_> = op
            .outputs
            .iter()
            .map(|s| builder.create_string(s))
            .collect();
        let inputs = builder.create_vector(&input_strs);
        let outputs = builder.create_vector(&output_strs);

        let (attr_type, attr_offset) = build_fb_attribute(builder, &op.attribute);

        let start = builder.start_table();
        builder.push_slot::<u32>(4, op.op as u32, 0); // op
        builder.push_slot::<u8>(6, attr_type, 0); // attribute_type
        if let Some(attr) = attr_offset {
            builder.push_slot_always::<WIPOffset<_>>(8, attr); // attribute
        }
        builder.push_slot_always::<WIPOffset<_>>(10, inputs); // inputs
        builder.push_slot_always::<WIPOffset<_>>(12, outputs); // outputs
        offsets.push(builder.end_table(start));
    }
    offsets
}

fn finish_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    fill: impl FnOnce(&mut FlatBufferBuilder<'_>),
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let start = builder.start_table();
    fill(builder);
    let off = builder.end_table(start);
    (attr.union_type(), Some(off))
}

fn empty_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    finish_attr(builder, attr, |_| {})
}

fn conv_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
    local_bound: bool,
    acc_type: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let pad = builder.create_vector(pad);
    let stride = builder.create_vector(stride);
    let dilation = builder.create_vector(dilation);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, pad);
        builder.push_slot_always::<WIPOffset<_>>(6, stride);
        builder.push_slot_always::<WIPOffset<_>>(8, dilation);
        builder.push_slot::<bool>(10, local_bound, false);
        builder.push_slot::<u32>(12, acc_type, 0);
    })
}

fn pool_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    kernel: &[i32],
    stride: &[i32],
    pad: &[i32],
    field10: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let kernel = builder.create_vector(kernel);
    let stride = builder.create_vector(stride);
    let pad = builder.create_vector(pad);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, kernel);
        builder.push_slot_always::<WIPOffset<_>>(6, stride);
        builder.push_slot_always::<WIPOffset<_>>(8, pad);
        builder.push_slot::<u32>(10, field10, 0);
    })
}

fn scalar_i32_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    value: i32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    finish_attr(builder, attr, |builder| {
        builder.push_slot::<i32>(4, value, 0);
    })
}

fn scalar_u32_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    value: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    finish_attr(builder, attr, |builder| {
        builder.push_slot::<u32>(4, value, 0);
    })
}

fn vector_i32_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    values: &[i32],
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let values = builder.create_vector(values);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, values);
    })
}

fn axis_nan_mode_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    axis: i32,
    nan_mode: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    finish_attr(builder, attr, |builder| {
        builder.push_slot::<i32>(4, axis, 0);
        builder.push_slot::<u32>(6, nan_mode, 0);
    })
}

fn transpose_conv_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    out_pad: &[i32],
    stride: &[i32],
    local_bound: bool,
    acc_type: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let out_pad = builder.create_vector(out_pad);
    let stride = builder.create_vector(stride);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, out_pad);
        builder.push_slot_always::<WIPOffset<_>>(6, stride);
        builder.push_slot::<bool>(8, local_bound, false);
        builder.push_slot::<u32>(10, acc_type, 0);
    })
}

fn string_pair_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    first: &str,
    second: &str,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let first = builder.create_string(first);
    let second = builder.create_string(second);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, first);
        builder.push_slot_always::<WIPOffset<_>>(6, second);
    })
}

fn clamp_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    min_val: &[u8],
    max_val: &[u8],
    nan_mode: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let min_val = create_force_aligned_u8_vector(builder, min_val);
    let max_val = create_force_aligned_u8_vector(builder, max_val);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, min_val);
        builder.push_slot_always::<WIPOffset<_>>(6, max_val);
        builder.push_slot::<u32>(8, nan_mode, 0);
    })
}

/// Build a flatbuffer attribute table and return (union_type_byte, optional offset).
fn build_fb_attribute(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    match attr {
        Attribute::None => (0, None),

        Attribute::Const(_)
        | Attribute::ConstShape(_)
        | Attribute::Add(_)
        | Attribute::Sub(_)
        | Attribute::Reshape(_)
        | Attribute::Table(_)
        | Attribute::Sigmoid(_)
        | Attribute::Tanh(_)
        | Attribute::Exp(_)
        | Attribute::Log(_)
        | Attribute::Negate(_)
        | Attribute::Abs(_)
        | Attribute::Ceil(_)
        | Attribute::Floor(_)
        | Attribute::Reciprocal(_)
        | Attribute::Erf(_)
        | Attribute::Cos(_)
        | Attribute::Sin(_)
        | Attribute::Rsqrt(_)
        | Attribute::MatMul(_)
        | Attribute::Pad(_)
        | Attribute::Slice(_)
        | Attribute::Gather(_)
        | Attribute::Scatter(_)
        | Attribute::Select(_)
        | Attribute::Cast(_)
        | Attribute::Equal(_)
        | Attribute::Greater(_)
        | Attribute::GreaterEqual(_)
        | Attribute::BitwiseNot(_)
        | Attribute::BitwiseAnd(_)
        | Attribute::BitwiseOr(_)
        | Attribute::BitwiseXor(_)
        | Attribute::LogicalLeftShift(_)
        | Attribute::LogicalRightShift(_)
        | Attribute::ArithmeticRightShift(_)
        | Attribute::LogicalNot(_)
        | Attribute::LogicalAnd(_)
        | Attribute::LogicalOr(_)
        | Attribute::LogicalXor(_)
        | Attribute::Pow(_)
        | Attribute::IntDiv(_) => empty_attr(builder, attr),

        Attribute::Conv2d(a) => conv_attr(
            builder,
            attr,
            &a.pad,
            &a.stride,
            &a.dilation,
            a.local_bound,
            a.acc_type as u32,
        ),
        Attribute::Conv3d(a) => conv_attr(
            builder,
            attr,
            &a.pad,
            &a.stride,
            &a.dilation,
            a.local_bound,
            a.acc_type as u32,
        ),
        Attribute::DepthwiseConv2d(a) => conv_attr(
            builder,
            attr,
            &a.pad,
            &a.stride,
            &a.dilation,
            a.local_bound,
            a.acc_type as u32,
        ),

        Attribute::Rescale(a) => finish_attr(builder, attr, |builder| {
            builder.push_slot::<bool>(4, a.scale32, false);
            builder.push_slot::<u32>(6, a.rounding_mode as u32, 0);
            builder.push_slot::<bool>(8, a.per_channel, false);
            builder.push_slot::<bool>(10, a.input_unsigned, false);
            builder.push_slot::<bool>(12, a.output_unsigned, false);
        }),

        Attribute::Resize(a) => scalar_u32_attr(builder, attr, a.mode as u32),

        Attribute::MaxPool2d(a) => pool_attr(
            builder,
            attr,
            &a.kernel,
            &a.stride,
            &a.pad,
            a.nan_mode as u32,
        ),

        Attribute::Transpose(a) => vector_i32_attr(builder, attr, &a.perms),

        Attribute::Clamp(a) => clamp_attr(builder, attr, &a.min_val, &a.max_val, a.nan_mode as u32),

        Attribute::Mul(a) => scalar_i32_attr(builder, attr, a.shift),

        Attribute::AvgPool2d(a) => pool_attr(
            builder,
            attr,
            &a.kernel,
            &a.stride,
            &a.pad,
            a.acc_type as u32,
        ),

        Attribute::ReduceSum(a) => scalar_i32_attr(builder, attr, a.axis),
        Attribute::Concat(a) => scalar_i32_attr(builder, attr, a.axis),

        Attribute::TransposeConv2d(a) => transpose_conv_attr(
            builder,
            attr,
            &a.out_pad,
            &a.stride,
            a.local_bound,
            a.acc_type as u32,
        ),

        Attribute::CondIf(a) => string_pair_attr(builder, attr, &a.then_graph, &a.else_graph),

        Attribute::WhileLoop(a) => string_pair_attr(builder, attr, &a.cond_branch, &a.body_branch),

        Attribute::Maximum(a) => scalar_u32_attr(builder, attr, a.nan_mode as u32),
        Attribute::Minimum(a) => scalar_u32_attr(builder, attr, a.nan_mode as u32),

        Attribute::ReduceAny(a) => scalar_i32_attr(builder, attr, a.axis),

        Attribute::ReduceMax(a) => axis_nan_mode_attr(builder, attr, a.axis, a.nan_mode as u32),

        Attribute::ReduceMin(a) => axis_nan_mode_attr(builder, attr, a.axis, a.nan_mode as u32),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Conv2d permutation: OIHW [2, 3, 2, 2] -> OHWI [2, 2, 2, 3]
    ///
    /// Uses 1-byte elements for simplicity. Each element is a unique byte
    /// encoding its position for easy verification.
    #[test]
    fn test_permute_conv2d_oihw_to_ohwi() {
        // Shape: [O=2, I=3, H=2, W=2] -> 24 elements
        let shape = [2i64, 3, 2, 2];
        let mut data = vec![0u8; 24];

        // Fill with position encoding: data[o][i][h][w] = o*12 + i*4 + h*2 + w
        for o in 0..2usize {
            for i in 0..3usize {
                for h in 0..2usize {
                    for w in 0..2usize {
                        let idx = ((o * 3 + i) * 2 + h) * 2 + w;
                        data[idx] = idx as u8;
                    }
                }
            }
        }

        let (out, new_shape) = permute_weight_data(&data, &shape, 1, WeightPermKind::Conv2d);

        // Expected shape: [2, 2, 2, 3] (OHWI)
        assert_eq!(new_shape, vec![2, 2, 2, 3]);
        assert_eq!(out.len(), 24);

        // Verify: out[o][h][w][i] = data[o][i][h][w]
        for o in 0..2usize {
            for h in 0..2usize {
                for w in 0..2usize {
                    for i in 0..3usize {
                        let src_idx = ((o * 3 + i) * 2 + h) * 2 + w;
                        let dst_idx = ((o * 2 + h) * 2 + w) * 3 + i;
                        assert_eq!(
                            out[dst_idx], data[src_idx],
                            "Mismatch at o={o},h={h},w={w},i={i}: out[{dst_idx}]={} != data[{src_idx}]={}",
                            out[dst_idx], data[src_idx]
                        );
                    }
                }
            }
        }
    }

    /// Test DepthwiseConv2d permutation: OIHW [16, 1, 3, 3] -> HWIM [3, 3, 1, 16]
    ///
    /// This matches the TinyCNN first conv (IC=1, OC=16, groups=1 -> depthwise).
    #[test]
    fn test_permute_depthwise_oihw_to_hwim() {
        // Shape: [OC=16, IC/groups=1, KH=3, KW=3] -> 144 elements
        let shape = [16i64, 1, 3, 3];
        let mut data = vec![0u8; 144];

        // Fill with sequential values
        for idx in 0..144 {
            data[idx] = idx as u8;
        }

        let (out, new_shape) = permute_weight_data(
            &data,
            &shape,
            1,
            WeightPermKind::DepthwiseConv2d { in_channels: 1 },
        );

        // Expected shape: [3, 3, 1, 16] (HWIM)
        assert_eq!(new_shape, vec![3, 3, 1, 16]);
        assert_eq!(out.len(), 144);

        // Verify: out[h][w][i][m] = data[m][i][h][w]
        for h in 0..3usize {
            for w in 0..3usize {
                for i in 0..1usize {
                    for m in 0..16usize {
                        let src_idx = ((m * 1 + i) * 3 + h) * 3 + w;
                        let dst_idx = ((h * 3 + w) * 1 + i) * 16 + m;
                        assert_eq!(
                            out[dst_idx], data[src_idx],
                            "Mismatch at h={h},w={w},i={i},m={m}: out[{dst_idx}]={} != data[{src_idx}]={}",
                            out[dst_idx], data[src_idx]
                        );
                    }
                }
            }
        }
    }

    /// Test a real depthwise case where IC>1 and depth_multiplier=1.
    #[test]
    fn test_permute_depthwise_oihw_to_hwim_multi_channel() {
        let shape = [6i64, 1, 2, 2];
        let mut data = vec![0u8; 24];
        for (idx, byte) in data.iter_mut().enumerate() {
            *byte = idx as u8;
        }

        let (out, new_shape) = permute_weight_data(
            &data,
            &shape,
            1,
            WeightPermKind::DepthwiseConv2d { in_channels: 6 },
        );

        assert_eq!(new_shape, vec![2, 2, 6, 1]);
        assert_eq!(out.len(), 24);

        for oc in 0..6usize {
            for h in 0..2usize {
                for w in 0..2usize {
                    let src_idx = ((oc * 1) * 2 + h) * 2 + w;
                    let dst_idx = ((h * 2 + w) * 6 + oc) * 1;
                    assert_eq!(
                        out[dst_idx], data[src_idx],
                        "Mismatch at oc={oc},h={h},w={w}: out[{dst_idx}]={} != data[{src_idx}]={}",
                        out[dst_idx], data[src_idx]
                    );
                }
            }
        }
    }

    /// Test that Conv2d permutation with 1x1 kernel is a no-op on data.
    /// Shape [10, 1568, 1, 1] -> [10, 1, 1, 1568] but data bytes are same.
    #[test]
    fn test_permute_conv2d_1x1_kernel_is_noop_on_data() {
        let shape = [10i64, 1568, 1, 1];
        let n = 10 * 1568;
        let data: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();

        let (out, new_shape) = permute_weight_data(&data, &shape, 1, WeightPermKind::Conv2d);

        assert_eq!(new_shape, vec![10, 1, 1, 1568]);
        // When H=1 and W=1, OIHW -> OHWI preserves data order.
        assert_eq!(out, data, "1x1 conv permutation should preserve data bytes");
    }

    /// Test permutation with multi-byte elements (int32 = 4 bytes).
    #[test]
    fn test_permute_conv2d_int32_elements() {
        // Shape: [2, 3, 1, 1] -> 6 elements, each 4 bytes = 24 bytes total
        let shape = [2i64, 3, 1, 1];
        let elem_size = 4;
        let n = 6;
        let mut data = vec![0u8; n * elem_size];

        for idx in 0..n {
            let val = idx as i32;
            let bytes = val.to_le_bytes();
            data[idx * elem_size..idx * elem_size + elem_size].copy_from_slice(&bytes);
        }

        let (out, new_shape) =
            permute_weight_data(&data, &shape, elem_size, WeightPermKind::Conv2d);

        assert_eq!(new_shape, vec![2, 1, 1, 3]);

        // For H=1,W=1: src_idx = o*3+i, dst_idx = o*3+i (same!)
        for idx in 0..n {
            let src_val = i32::from_le_bytes(
                data[idx * elem_size..idx * elem_size + elem_size]
                    .try_into()
                    .unwrap(),
            );
            let dst_val = i32::from_le_bytes(
                out[idx * elem_size..idx * elem_size + elem_size]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(
                src_val, dst_val,
                "Element {idx}: src={src_val} != dst={dst_val}"
            );
        }
    }

    /// Test Conv2d permutation with a regular 3x3 conv and 4-byte elements.
    /// Shape [2, 2, 3, 3] -> [2, 3, 3, 2]
    #[test]
    fn test_permute_conv2d_3x3_int32() {
        let shape = [2i64, 2, 3, 3];
        let elem_size = 4;
        let n = 2 * 2 * 3 * 3; // 36 elements
        let mut data = vec![0u8; n * elem_size];

        for idx in 0..n {
            let bytes = (idx as i32).to_le_bytes();
            data[idx * elem_size..idx * elem_size + elem_size].copy_from_slice(&bytes);
        }

        let (out, new_shape) =
            permute_weight_data(&data, &shape, elem_size, WeightPermKind::Conv2d);

        assert_eq!(new_shape, vec![2, 3, 3, 2]);

        for o in 0..2usize {
            for i in 0..2usize {
                for h in 0..3usize {
                    for w in 0..3usize {
                        let src_idx = ((o * 2 + i) * 3 + h) * 3 + w;
                        let dst_idx = ((o * 3 + h) * 3 + w) * 2 + i;

                        let src_val = i32::from_le_bytes(
                            data[src_idx * elem_size..src_idx * elem_size + elem_size]
                                .try_into()
                                .unwrap(),
                        );
                        let dst_val = i32::from_le_bytes(
                            out[dst_idx * elem_size..dst_idx * elem_size + elem_size]
                                .try_into()
                                .unwrap(),
                        );

                        assert_eq!(
                            src_val, dst_val,
                            "o={o},i={i},h={h},w={w}: src[{src_idx}]={src_val} != dst[{dst_idx}]={dst_val}"
                        );
                    }
                }
            }
        }
    }

    /// Verify that permuting twice with inverse permutations recovers the original data.
    /// OIHW -> OHWI -> OIHW should be identity.
    #[test]
    fn test_permute_roundtrip() {
        let shape = [4i64, 3, 5, 7];
        let n = (4 * 3 * 5 * 7) as usize;
        let data: Vec<u8> = (0..n).map(|i| (i * 17 + 3) as u8).collect();

        let (permuted, new_shape) = permute_weight_data(&data, &shape, 1, WeightPermKind::Conv2d);

        // new_shape is OHWI: [4, 5, 7, 3]
        assert_eq!(new_shape, vec![4, 5, 7, 3]);

        // Now we need to go back: OHWI -> OIHW, which is perm [0, 3, 1, 2].
        // This is NOT directly what our function does (it always does OIHW->OHWI).
        // Instead, manually verify that each element is in the right position.
        for o in 0..4usize {
            for i in 0..3usize {
                for h in 0..5usize {
                    for w in 0..7usize {
                        let src_idx = ((o * 3 + i) * 5 + h) * 7 + w;
                        let dst_idx = ((o * 5 + h) * 7 + w) * 3 + i;
                        assert_eq!(permuted[dst_idx], data[src_idx], "o={o},i={i},h={h},w={w}");
                    }
                }
            }
        }
    }
}
