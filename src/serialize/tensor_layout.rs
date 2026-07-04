use std::borrow::Cow;
use std::collections::HashMap;

use crate::graph::ir::{Arg, Node, OpTarget};
use crate::graph::Graph;
use crate::store::TensorData;

use super::op_semantics;
use super::tensor_decl::SerializedTensor;
use super::tosa_generated::DType;
use super::visitors::ir_dtype_to_tosa;

/// Describes how a weight tensor should be permuted for TOSA.
#[derive(Debug, Clone, Copy)]
enum WeightPermKind {
    /// Standard conv2d: OIHW [OC, IC, KH, KW] -> OHWI [OC, KH, KW, IC].
    Conv2d,
    /// Standard conv3d: OIDHW [OC, IC, KD, KH, KW] -> ODHWI [OC, KD, KH, KW, IC].
    Conv3d,
    /// Depthwise conv2d: OIHW [OC, 1, KH, KW] -> HWIM [KH, KW, IC, M].
    DepthwiseConv2d { in_channels: usize },
}

pub(super) struct TensorDeclarationContext {
    weight_perms: HashMap<String, WeightPermKind>,
    dtype_overrides: HashMap<String, DType>,
}

impl TensorDeclarationContext {
    pub(super) fn new(graph: &Graph) -> Self {
        Self {
            weight_perms: collect_weight_permutations(graph),
            dtype_overrides: collect_tensor_dtype_overrides(graph),
        }
    }

    pub(super) fn runtime_placeholder<'a>(&self, node: &Node) -> SerializedTensor<'a> {
        let (shape, dtype) = placeholder_decl(node);
        SerializedTensor::new(node.name.clone(), shape, dtype)
    }

    pub(super) fn const_placeholder<'a>(
        &self,
        node: &Node,
        tensor_data: &'a TensorData,
    ) -> SerializedTensor<'a> {
        serialize_const_placeholder(node, tensor_data, &self.weight_perms, &self.dtype_overrides)
    }
}

pub(crate) fn apply_tosa_dim_order(node: &Node, shape: Vec<i32>) -> Vec<i32> {
    let Some(order) = node.meta.tosa_dim_order.as_ref() else {
        return shape;
    };
    permute_shape_u8(&shape, order).unwrap_or(shape)
}

pub(super) fn raw_i32_shape(node: &Node) -> Vec<i32> {
    node.meta
        .val
        .shape_slice()
        .iter()
        .map(|&dim| dim as i32)
        .collect()
}

pub(super) fn node_tensor_dtype(node: &Node) -> DType {
    node.meta
        .tosa_special_dtype
        .map(ir_dtype_to_tosa)
        .unwrap_or_else(|| ir_dtype_to_tosa(node.meta.val.dtype))
}

pub(super) fn inverse_tosa_dim_order(order: &[u8]) -> Option<Vec<i32>> {
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

pub(super) fn inverse_permutation(perms: &[i32]) -> Option<Vec<i32>> {
    let mut inverse = vec![0; perms.len()];
    for (idx, &axis) in perms.iter().enumerate() {
        let axis = usize::try_from(axis).ok()?;
        if axis >= perms.len() {
            return None;
        }
        inverse[axis] = idx as i32;
    }
    Some(inverse)
}

pub(super) fn channel_last_order(rank: usize) -> Option<Vec<i32>> {
    if rank < 4 {
        return None;
    }
    let mut order = Vec::with_capacity(rank);
    order.push(0);
    order.extend((2..rank).map(|axis| axis as i32));
    order.push(1);
    Some(order)
}

pub(super) fn infer_shape_permutation(actual_shape: &[i32], raw_shape: &[i32]) -> Option<Vec<i32>> {
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

pub(super) fn permute_shape(shape: &[i32], perms: &[i32]) -> Option<Vec<i32>> {
    permute_shape_by_order(shape, perms)
}

pub(super) fn permute_shape_u8(shape: &[i32], order: &[u8]) -> Option<Vec<i32>> {
    permute_shape_by_order(shape, order)
}

fn permute_i64_shape_u8(shape: &[i64], order: &[u8]) -> Option<Vec<i64>> {
    permute_shape_by_order(shape, order)
}

fn is_identity_order(order: &[u8]) -> bool {
    order
        .iter()
        .enumerate()
        .all(|(idx, &axis)| axis as usize == idx)
}

fn permute_shape_by_order<T, I>(shape: &[T], order: &[I]) -> Option<Vec<T>>
where
    T: Copy,
    I: Copy + TryInto<usize>,
{
    if shape.len() != order.len() {
        return None;
    }

    order
        .iter()
        .map(|&idx| idx.try_into().ok().and_then(|idx| shape.get(idx).copied()))
        .collect()
}

/// Permute convolution weight tensors into their physical TOSA layout.
fn permute_weight_data<'a>(
    data: &'a [u8],
    shape: &[i64],
    elem_size: usize,
    kind: WeightPermKind,
) -> (Cow<'a, [u8]>, Vec<i64>) {
    let Some((logical_shape, order)) = weight_permutation_plan(shape, kind) else {
        return (Cow::Borrowed(data), shape.to_vec());
    };
    permute_tensor_by_order_cow(data, &logical_shape, elem_size, &order)
}

fn weight_permutation_plan(shape: &[i64], kind: WeightPermKind) -> Option<(Vec<i64>, Vec<u8>)> {
    match kind {
        WeightPermKind::Conv2d if shape.len() == 4 => Some((shape.to_vec(), vec![0, 2, 3, 1])),
        WeightPermKind::Conv3d if shape.len() == 5 => Some((shape.to_vec(), vec![0, 2, 3, 4, 1])),
        WeightPermKind::DepthwiseConv2d { in_channels } if shape.len() == 4 => {
            let output_channels = usize::try_from(shape[0]).ok()?;
            if in_channels == 0 || shape[1] != 1 || output_channels % in_channels != 0 {
                return None;
            }
            let depth_multiplier = output_channels / in_channels;
            (depth_multiplier > 0).then(|| {
                (
                    vec![
                        in_channels as i64,
                        depth_multiplier as i64,
                        shape[2],
                        shape[3],
                    ],
                    vec![2, 3, 0, 1],
                )
            })
        }
        _ => None,
    }
}

fn tensor_numel(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        usize::try_from(dim).ok().and_then(|d| acc.checked_mul(d))
    })
}

fn tensor_data_matches_shape(data: &[u8], shape: &[i64], elem_size: usize) -> bool {
    tensor_numel(shape)
        .is_some_and(|numel| numel != 0 && numel.saturating_mul(elem_size) == data.len())
}

fn permutation_preserves_contiguous_data(shape: &[i64], order: &[u8]) -> bool {
    if shape.len() != order.len() {
        return false;
    }
    let mut seen = vec![false; shape.len()];
    let mut output_non_singleton_axes = Vec::new();
    for &axis in order {
        let axis = axis as usize;
        if axis >= shape.len() || seen[axis] {
            return false;
        }
        seen[axis] = true;
        if shape[axis] > 1 {
            output_non_singleton_axes.push(axis);
        }
    }
    if seen.iter().any(|seen| !seen) {
        return false;
    }
    output_non_singleton_axes
        == (0..shape.len())
            .filter(|&axis| shape[axis] > 1)
            .collect::<Vec<_>>()
}

/// Permute a tensor's raw bytes according to an arbitrary output-axis order.
fn permute_tensor_by_order_cow<'a>(
    data: &'a [u8],
    shape: &[i64],
    elem_size: usize,
    order: &[u8],
) -> (Cow<'a, [u8]>, Vec<i64>) {
    if shape.len() != order.len() || shape.is_empty() {
        return (Cow::Borrowed(data), shape.to_vec());
    }

    if order
        .iter()
        .enumerate()
        .all(|(idx, &axis)| idx == axis as usize)
    {
        return (Cow::Borrowed(data), shape.to_vec());
    }

    let Some(new_shape) = permute_i64_shape_u8(shape, order) else {
        return (Cow::Borrowed(data), shape.to_vec());
    };

    if permutation_preserves_contiguous_data(shape, order)
        && tensor_data_matches_shape(data, shape, elem_size)
    {
        return (Cow::Borrowed(data), new_shape);
    }

    if let Some((data, shape)) = permute_tensor_by_known_order(data, shape, elem_size, order) {
        return (Cow::Owned(data), shape);
    }

    let (data, shape) = permute_tensor_by_order(data, shape, elem_size, order);
    (Cow::Owned(data), shape)
}

fn permute_tensor_by_known_order(
    data: &[u8],
    shape: &[i64],
    elem_size: usize,
    order: &[u8],
) -> Option<(Vec<u8>, Vec<i64>)> {
    match (shape.len(), order) {
        (4, [0, 2, 3, 1]) => permute_oihw_to_ohwi_bytes(data, shape, elem_size),
        (5, [0, 2, 3, 4, 1]) => permute_oidhw_to_odhwi_bytes(data, shape, elem_size),
        (4, [2, 3, 0, 1]) => permute_depthwise_hwim_bytes(data, shape, elem_size),
        _ => None,
    }
}

fn read_shape4(shape: &[i64]) -> Option<[usize; 4]> {
    Some([
        usize::try_from(*shape.first()?).ok()?,
        usize::try_from(*shape.get(1)?).ok()?,
        usize::try_from(*shape.get(2)?).ok()?,
        usize::try_from(*shape.get(3)?).ok()?,
    ])
}

fn read_shape5(shape: &[i64]) -> Option<[usize; 5]> {
    Some([
        usize::try_from(*shape.first()?).ok()?,
        usize::try_from(*shape.get(1)?).ok()?,
        usize::try_from(*shape.get(2)?).ok()?,
        usize::try_from(*shape.get(3)?).ok()?,
        usize::try_from(*shape.get(4)?).ok()?,
    ])
}

fn copy_element(dst: &mut [u8], dst_off: usize, src: &[u8], src_off: usize, elem_size: usize) {
    if elem_size == 1 {
        dst[dst_off] = src[src_off];
    } else {
        dst[dst_off..dst_off + elem_size].copy_from_slice(&src[src_off..src_off + elem_size]);
    }
}

fn permute_oihw_to_ohwi_bytes(
    data: &[u8],
    shape: &[i64],
    elem_size: usize,
) -> Option<(Vec<u8>, Vec<i64>)> {
    let [oc, ic, kh, kw] = read_shape4(shape)?;
    if !tensor_data_matches_shape(data, shape, elem_size) {
        return None;
    }
    let mut out = vec![0u8; data.len()];
    for o in 0..oc {
        let src_o = o * ic * kh * kw;
        let dst_o = o * kh * kw * ic;
        for h in 0..kh {
            for w in 0..kw {
                let dst_hw = dst_o + (h * kw + w) * ic;
                for i in 0..ic {
                    let src_idx = src_o + (i * kh + h) * kw + w;
                    let dst_idx = dst_hw + i;
                    copy_element(
                        &mut out,
                        dst_idx * elem_size,
                        data,
                        src_idx * elem_size,
                        elem_size,
                    );
                }
            }
        }
    }
    Some((out, vec![shape[0], shape[2], shape[3], shape[1]]))
}

fn permute_oidhw_to_odhwi_bytes(
    data: &[u8],
    shape: &[i64],
    elem_size: usize,
) -> Option<(Vec<u8>, Vec<i64>)> {
    let [oc, ic, kd, kh, kw] = read_shape5(shape)?;
    if !tensor_data_matches_shape(data, shape, elem_size) {
        return None;
    }
    let mut out = vec![0u8; data.len()];
    for o in 0..oc {
        let src_o = o * ic * kd * kh * kw;
        let dst_o = o * kd * kh * kw * ic;
        for d in 0..kd {
            for h in 0..kh {
                for w in 0..kw {
                    let dst_dhw = dst_o + ((d * kh + h) * kw + w) * ic;
                    for i in 0..ic {
                        let src_idx = src_o + ((i * kd + d) * kh + h) * kw + w;
                        let dst_idx = dst_dhw + i;
                        copy_element(
                            &mut out,
                            dst_idx * elem_size,
                            data,
                            src_idx * elem_size,
                            elem_size,
                        );
                    }
                }
            }
        }
    }
    Some((out, vec![shape[0], shape[2], shape[3], shape[4], shape[1]]))
}

fn permute_depthwise_hwim_bytes(
    data: &[u8],
    logical_shape: &[i64],
    elem_size: usize,
) -> Option<(Vec<u8>, Vec<i64>)> {
    let [ic, multiplier, kh, kw] = read_shape4(logical_shape)?;
    if !tensor_data_matches_shape(data, logical_shape, elem_size) {
        return None;
    }
    let mut out = vec![0u8; data.len()];
    for h in 0..kh {
        for w in 0..kw {
            let dst_hw = (h * kw + w) * ic * multiplier;
            for i in 0..ic {
                for m in 0..multiplier {
                    let src_idx = ((i * multiplier + m) * kh + h) * kw + w;
                    let dst_idx = dst_hw + i * multiplier + m;
                    copy_element(
                        &mut out,
                        dst_idx * elem_size,
                        data,
                        src_idx * elem_size,
                        elem_size,
                    );
                }
            }
        }
    }
    Some((
        out,
        vec![
            logical_shape[2],
            logical_shape[3],
            logical_shape[0],
            logical_shape[1],
        ],
    ))
}

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

    if shape.len() == 2 && order == [1, 0] {
        return transpose_2d_bytes(data, shape, elem_size);
    }

    let rank = shape.len();
    let numel: usize = match tensor_numel(shape) {
        Some(v) => v,
        None => return (data.to_vec(), shape.to_vec()),
    };

    if numel == 0 || numel.saturating_mul(elem_size) != data.len() {
        return (data.to_vec(), shape.to_vec());
    }

    let new_shape = permute_i64_shape_u8(shape, order).expect("shape/order rank checked above");

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

fn transpose_2d_bytes(data: &[u8], shape: &[i64], elem_size: usize) -> (Vec<u8>, Vec<i64>) {
    let (Some(rows), Some(cols)) = (
        usize::try_from(shape[0]).ok(),
        usize::try_from(shape[1]).ok(),
    ) else {
        return (data.to_vec(), shape.to_vec());
    };
    let Some(numel) = rows.checked_mul(cols) else {
        return (data.to_vec(), shape.to_vec());
    };
    if numel == 0 || numel.saturating_mul(elem_size) != data.len() {
        return (data.to_vec(), shape.to_vec());
    }

    let mut out = vec![0u8; data.len()];
    const TILE: usize = 32;
    for col_base in (0..cols).step_by(TILE) {
        let col_end = (col_base + TILE).min(cols);
        for row_base in (0..rows).step_by(TILE) {
            let row_end = (row_base + TILE).min(rows);
            for col in col_base..col_end {
                let dst_col_base = col * rows * elem_size;
                for row in row_base..row_end {
                    let src_off = (row * cols + col) * elem_size;
                    let dst_off = dst_col_base + row * elem_size;
                    if elem_size == 1 {
                        out[dst_off] = data[src_off];
                    } else {
                        out[dst_off..dst_off + elem_size]
                            .copy_from_slice(&data[src_off..src_off + elem_size]);
                    }
                }
            }
        }
    }

    (out, vec![shape[1], shape[0]])
}

fn pack_int4_data(data: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(data.len().div_ceil(2));
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

fn infer_depthwise_in_channels(graph: &Graph, node: &Node, oc: usize) -> usize {
    let Some(Arg::Node(input_id)) = node.args.first() else {
        return 0;
    };
    let Some(input_node) = graph.node(*input_id) else {
        return 0;
    };

    let shape = input_node.meta.val.shape_slice();
    let groups = aten_conv_groups(node);
    if groups > 0 {
        let groups = groups as usize;
        for candidate in [shape.last().copied(), shape.get(1).copied()]
            .into_iter()
            .flatten()
        {
            let candidate = candidate as usize;
            if candidate == groups && oc % candidate == 0 {
                return candidate;
            }
        }
    }

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

fn arg_i64(node: &Node, idx: usize, default: i64) -> i64 {
    match node.args.get(idx) {
        Some(Arg::Int(v)) => *v,
        _ => default,
    }
}

fn aten_conv_groups(node: &Node) -> i64 {
    let groups_arg = if node.args.len() > 8 { 8 } else { 6 };
    arg_i64(node, groups_arg, 1)
}

fn depthwise_weight_perm(graph: &Graph, node: &Node, weight_node: &Node) -> Option<WeightPermKind> {
    let groups = aten_conv_groups(node);
    if groups <= 0 {
        return None;
    }

    let weight_shape = weight_node.meta.val.shape_slice();
    if weight_shape.len() != 4 || weight_shape.get(1).copied() != Some(1) {
        return None;
    }

    let oc = weight_shape.first().copied().unwrap_or(0) as usize;
    let in_channels = infer_depthwise_in_channels(graph, node, oc);
    (in_channels > 0 && in_channels == groups as usize && oc % in_channels == 0)
        .then_some(WeightPermKind::DepthwiseConv2d { in_channels })
}

fn collect_weight_permutations(graph: &Graph) -> HashMap<String, WeightPermKind> {
    let mut perms = HashMap::new();
    for &nid in &graph.order {
        let Some(node) = graph.node(nid) else {
            continue;
        };
        let kind = match &node.target {
            OpTarget::TosaConv2d => Some(WeightPermKind::Conv2d),
            OpTarget::Conv2d => match node.args.get(1) {
                Some(Arg::Node(weight_id)) => graph.node(*weight_id).map(|weight_node| {
                    depthwise_weight_perm(graph, node, weight_node).unwrap_or_else(|| {
                        if weight_node.meta.val.shape_len == 5 {
                            WeightPermKind::Conv3d
                        } else {
                            WeightPermKind::Conv2d
                        }
                    })
                }),
                _ => None,
            },
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
            if let Some(Arg::Node(weight_id)) = node.args.get(1) {
                if let Some(weight_node) = graph.node(*weight_id) {
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

fn has_int48_conv_accumulator(node: &Node) -> bool {
    node.meta.tosa_special_dtype == Some(crate::graph::ir::DType::Int48)
        || node.meta.val.dtype == crate::graph::ir::DType::Int48
        || node
            .meta
            .input_qparams
            .iter()
            .any(|(_, qp)| qp.dtype == crate::graph::ir::DType::Int16)
}

fn collect_tensor_dtype_overrides(graph: &Graph) -> HashMap<String, DType> {
    let mut overrides = HashMap::new();
    for &nid in &graph.order {
        let Some(node) = graph.node(nid) else {
            continue;
        };
        if !op_semantics::is_conv_weight_source(&node.target) || !has_int48_conv_accumulator(node) {
            continue;
        }
        if let Some(Arg::Node(bias_id)) = node.args.get(2) {
            if let Some(bias) = graph.node(*bias_id) {
                overrides.insert(bias.name.clone(), DType::Int48);
            }
        }
    }
    overrides
}

fn normalize_placeholder_shape(node: &Node, shape: Vec<i32>) -> Vec<i32> {
    if node.meta.preserve_scalar_shape || !shape.is_empty() {
        shape
    } else {
        vec![1]
    }
}

fn placeholder_decl(node: &Node) -> (Vec<i32>, DType) {
    let raw_shape = raw_i32_shape(node);
    (
        normalize_placeholder_shape(node, apply_tosa_dim_order(node, raw_shape)),
        ir_dtype_to_tosa(node.meta.val.dtype),
    )
}

fn serialize_const_placeholder<'a>(
    node: &Node,
    tensor_data: &'a TensorData,
    weight_perms: &HashMap<String, WeightPermKind>,
    dtype_overrides: &HashMap<String, DType>,
) -> SerializedTensor<'a> {
    let store_dtype = ir_dtype_to_tosa(tensor_data.dtype);
    let tensor_dtype = dtype_overrides
        .get(&node.name)
        .copied()
        .or_else(|| node.meta.tosa_special_dtype.map(ir_dtype_to_tosa))
        .unwrap_or(store_dtype);
    let elem_sz = tensor_data.dtype.storage_size_bytes();

    let (final_data, final_shape_i64): (Cow<'_, [u8]>, Vec<i64>) =
        if let Some(&perm_kind) = weight_perms.get(&node.name) {
            permute_weight_data(&tensor_data.data, &tensor_data.shape, elem_sz, perm_kind)
        } else if let Some(order) = node
            .meta
            .tosa_dim_order
            .as_ref()
            .filter(|order| !is_identity_order(order))
        {
            permute_tensor_by_order_cow(&tensor_data.data, &tensor_data.shape, elem_sz, order)
        } else {
            (
                Cow::Borrowed(tensor_data.data.as_slice()),
                tensor_data.shape.clone(),
            )
        };
    let final_shape: Vec<i32> =
        normalize_placeholder_shape(node, final_shape_i64.iter().map(|&d| d as i32).collect());

    match (tensor_dtype, store_dtype) {
        (DType::Int4, _) if store_dtype != DType::Int4 => SerializedTensor::with_owned_data(
            node.name.clone(),
            final_shape,
            tensor_dtype,
            pack_int4_data(final_data.as_ref()),
        ),
        (DType::Int48, DType::Int32) => SerializedTensor::with_owned_data(
            node.name.clone(),
            final_shape,
            tensor_dtype,
            widen_int32_data_to_int48(final_data.as_ref()),
        ),
        _ => SerializedTensor::with_data(node.name.clone(), final_shape, tensor_dtype, final_data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{NodeMeta, OpType};
    use crate::store::TensorData;

    #[test]
    fn test_permute_conv2d_oihw_to_ohwi() {
        let shape = [2i64, 3, 2, 2];
        let mut data = vec![0u8; 24];

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

        assert_eq!(new_shape, vec![2, 2, 2, 3]);
        assert_eq!(out.len(), 24);

        for o in 0..2usize {
            for h in 0..2usize {
                for w in 0..2usize {
                    for i in 0..3usize {
                        let src_idx = ((o * 3 + i) * 2 + h) * 2 + w;
                        let dst_idx = ((o * 2 + h) * 2 + w) * 3 + i;
                        assert_eq!(
                            out[dst_idx], data[src_idx],
                            "Mismatch at o={o},h={h},w={w},i={i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_permute_depthwise_oihw_to_hwim() {
        let shape = [16i64, 1, 3, 3];
        let mut data = vec![0u8; 144];

        for (idx, byte) in data.iter_mut().enumerate() {
            *byte = idx as u8;
        }

        let (out, new_shape) = permute_weight_data(
            &data,
            &shape,
            1,
            WeightPermKind::DepthwiseConv2d { in_channels: 1 },
        );

        assert_eq!(new_shape, vec![3, 3, 1, 16]);
        assert_eq!(out.len(), 144);

        for h in 0..3usize {
            for w in 0..3usize {
                for i in 0..1usize {
                    for m in 0..16usize {
                        let src_idx = ((m * 1 + i) * 3 + h) * 3 + w;
                        let dst_idx = ((h * 3 + w) * 1 + i) * 16 + m;
                        assert_eq!(
                            out[dst_idx], data[src_idx],
                            "Mismatch at h={h},w={w},i={i},m={m}"
                        );
                    }
                }
            }
        }
    }

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
                        "Mismatch at oc={oc},h={h},w={w}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_permute_conv2d_1x1_kernel_is_noop_on_data() {
        let shape = [10i64, 1568, 1, 1];
        let n = 10 * 1568;
        let data: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();

        let (out, new_shape) = permute_weight_data(&data, &shape, 1, WeightPermKind::Conv2d);

        assert_eq!(new_shape, vec![10, 1, 1, 1568]);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out, data);
    }

    #[test]
    fn test_permute_conv2d_int32_elements() {
        let shape = [2i64, 3, 1, 1];
        let elem_size = 4;
        let n = 6;
        let mut data = vec![0u8; n * elem_size];

        for idx in 0..n {
            let bytes = (idx as i32).to_le_bytes();
            data[idx * elem_size..idx * elem_size + elem_size].copy_from_slice(&bytes);
        }

        let (out, new_shape) =
            permute_weight_data(&data, &shape, elem_size, WeightPermKind::Conv2d);

        assert_eq!(new_shape, vec![2, 1, 1, 3]);

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
            assert_eq!(src_val, dst_val);
        }
    }

    #[test]
    fn test_permute_conv2d_3x3_int32() {
        let shape = [2i64, 2, 3, 3];
        let elem_size = 4;
        let n = 2 * 2 * 3 * 3;
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

    #[test]
    fn test_permute_roundtrip() {
        let shape = [4i64, 3, 5, 7];
        let n = (4 * 3 * 5 * 7) as usize;
        let data: Vec<u8> = (0..n).map(|i| (i * 17 + 3) as u8).collect();

        let (permuted, new_shape) = permute_weight_data(&data, &shape, 1, WeightPermKind::Conv2d);

        assert_eq!(new_shape, vec![4, 5, 7, 3]);

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

    #[test]
    fn identity_dim_order_const_placeholder_borrows_store_data() {
        let mut node = Node {
            op: OpType::Placeholder,
            target: OpTarget::Other("weight_view".to_string()),
            args: smallvec::SmallVec::new(),
            kwargs: Vec::new(),
            meta: NodeMeta::new(crate::graph::ir::TensorInfo::new(
                &[2, 3],
                crate::graph::ir::DType::Int8,
            )),
            users: smallvec::SmallVec::new(),
            name: "weight_view".to_string(),
        };
        node.meta.tosa_dim_order = Some(smallvec::smallvec![0, 1]);

        let tensor = TensorData {
            data: vec![1, 2, 3, 4, 5, 6],
            shape: vec![2, 3],
            dtype: crate::graph::ir::DType::Int8,
        };
        let serialized =
            TensorDeclarationContext::new(&Graph::new()).const_placeholder(&node, &tensor);
        let data = serialized.data.as_ref().expect("const tensor data");
        assert!(matches!(data, std::borrow::Cow::Borrowed(_)));
        assert_eq!(data.as_ref(), tensor.data.as_slice());
    }

    #[test]
    fn scalar_placeholders_normalize_to_rank1_unless_preserved() {
        let mut node = Node {
            op: OpType::Placeholder,
            target: OpTarget::Other("scalar".to_string()),
            args: smallvec::SmallVec::new(),
            kwargs: Vec::new(),
            meta: NodeMeta::new(crate::graph::ir::TensorInfo::new(
                &[],
                crate::graph::ir::DType::Int8,
            )),
            users: smallvec::SmallVec::new(),
            name: "scalar".to_string(),
        };
        assert_eq!(placeholder_decl(&node).0, vec![1]);

        let tensor = TensorData {
            data: vec![7u8],
            shape: vec![],
            dtype: crate::graph::ir::DType::Int8,
        };
        let context = TensorDeclarationContext::new(&Graph::new());
        let serialized = context.const_placeholder(&node, &tensor);
        assert_eq!(serialized.shape, vec![1]);

        node.meta.preserve_scalar_shape = true;
        assert_eq!(placeholder_decl(&node).0, Vec::<i32>::new());
        let preserved = context.const_placeholder(&node, &tensor);
        assert_eq!(preserved.shape, Vec::<i32>::new());
    }
}
