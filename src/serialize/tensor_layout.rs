use std::collections::HashMap;

use crate::graph::ir::{Arg, Node, OpTarget};
use crate::graph::Graph;
use crate::store::TensorData;

use super::op_semantics;
use super::tosa_types::{DType, TosaTensor};
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

    pub(super) fn runtime_placeholder(&self, node: &Node) -> TosaTensor {
        let (shape, dtype) = placeholder_decl(node);
        make_graph_tensor(node.name.clone(), shape, dtype, Vec::new())
    }

    pub(super) fn const_placeholder(&self, node: &Node, tensor_data: &TensorData) -> TosaTensor {
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

pub(super) fn permute_shape(shape: &[i32], perms: &[i32]) -> Option<Vec<i32>> {
    permute_shape_by_order(shape, perms)
}

pub(super) fn permute_shape_u8(shape: &[i32], order: &[u8]) -> Option<Vec<i32>> {
    permute_shape_by_order(shape, order)
}

fn permute_i64_shape_u8(shape: &[i64], order: &[u8]) -> Option<Vec<i64>> {
    permute_shape_by_order(shape, order)
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
fn permute_weight_data(
    data: &[u8],
    shape: &[i64],
    elem_size: usize,
    kind: WeightPermKind,
) -> (Vec<u8>, Vec<i64>) {
    let Some((logical_shape, order)) = weight_permutation_plan(shape, kind) else {
        return (data.to_vec(), shape.to_vec());
    };
    permute_tensor_by_order(data, &logical_shape, elem_size, &order)
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

/// Permute a tensor's raw bytes according to an arbitrary output-axis order.
pub(crate) fn permute_tensor_by_order(
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

    if numel == 0 || elem_size == 0 || numel.saturating_mul(elem_size) != data.len() {
        return (data.to_vec(), shape.to_vec());
    }

    let Some(new_shape) = permute_i64_shape_u8(shape, order) else {
        return (data.to_vec(), shape.to_vec());
    };

    // Size-1 axes don't affect the byte order: when the other axes keep their
    // relative order (e.g. a 1x1 conv's OIHW -> OHWI), the bytes are already right.
    let mut moved = order
        .iter()
        .map(|&axis| axis as usize)
        .filter(|&axis| shape[axis] != 1);
    if moved
        .try_fold(None, |last: Option<usize>, axis| match last {
            Some(prev) if axis < prev => None,
            _ => Some(Some(axis)),
        })
        .is_some()
    {
        return (data.to_vec(), new_shape);
    }

    let mut input_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        input_strides[i] = input_strides[i + 1] * shape[i + 1] as usize;
    }
    let src_strides: Vec<usize> = order
        .iter()
        .map(|&axis| input_strides[axis as usize])
        .collect();
    let out_shape: Vec<usize> = new_shape.iter().map(|&dim| dim as usize).collect();

    let mut out = vec![0u8; data.len()];
    match elem_size {
        1 => gather_permuted::<1>(data, &out_shape, &src_strides, &mut out),
        2 => gather_permuted::<2>(data, &out_shape, &src_strides, &mut out),
        4 => gather_permuted::<4>(data, &out_shape, &src_strides, &mut out),
        8 => gather_permuted::<8>(data, &out_shape, &src_strides, &mut out),
        _ => gather_permuted_bytes(data, elem_size, &out_shape, &src_strides, &mut out),
    }
    (out, new_shape)
}

/// Fill `out` (row-major in `out_shape`) from `data`, where output axis `k` steps
/// `src_strides[k]` elements in the source. Walks the output with an odometer, so
/// there is no per-element division.
fn gather_permuted<const N: usize>(
    data: &[u8],
    out_shape: &[usize],
    src_strides: &[usize],
    out: &mut [u8],
) {
    let (src, _) = data.as_chunks::<N>();
    let (dst, _) = out.as_chunks_mut::<N>();
    let inner = out_shape.len() - 1;
    let (inner_len, inner_stride) = (out_shape[inner], src_strides[inner]);
    let mut index = vec![0usize; inner];
    let mut base = 0usize;
    for row in dst.chunks_exact_mut(inner_len) {
        if inner_stride == 1 {
            row.copy_from_slice(&src[base..base + inner_len]);
        } else {
            for (i, element) in row.iter_mut().enumerate() {
                *element = src[base + i * inner_stride];
            }
        }
        for axis in (0..inner).rev() {
            index[axis] += 1;
            base += src_strides[axis];
            if index[axis] < out_shape[axis] {
                break;
            }
            base -= src_strides[axis] * out_shape[axis];
            index[axis] = 0;
        }
    }
}

fn gather_permuted_bytes(
    data: &[u8],
    elem_size: usize,
    out_shape: &[usize],
    src_strides: &[usize],
    out: &mut [u8],
) {
    let rank = out_shape.len();
    let mut index = vec![0usize; rank];
    let mut src = 0usize;
    for dst in out.chunks_exact_mut(elem_size) {
        dst.copy_from_slice(&data[src * elem_size..(src + 1) * elem_size]);
        for axis in (0..rank).rev() {
            index[axis] += 1;
            src += src_strides[axis];
            if index[axis] < out_shape[axis] {
                break;
            }
            src -= src_strides[axis] * out_shape[axis];
            index[axis] = 0;
        }
    }
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
    for chunk in data.as_chunks::<4>().0.iter() {
        let value = i32::from_le_bytes(*chunk) as i64;
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
            if candidate == groups && oc.is_multiple_of(candidate) {
                return candidate;
            }
        }
    }

    for candidate in [shape.last().copied(), shape.get(1).copied()]
        .into_iter()
        .flatten()
    {
        let candidate = candidate as usize;
        if candidate > 0 && oc.is_multiple_of(candidate) {
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
    (in_channels > 0 && in_channels == groups as usize && oc.is_multiple_of(in_channels))
        .then_some(WeightPermKind::DepthwiseConv2d { in_channels })
}

fn collect_weight_permutations(graph: &Graph) -> HashMap<String, WeightPermKind> {
    let mut perms = HashMap::new();
    // Explicit-mode conv weights arrive already in OHWI/HWCM/ODHWI; permuting them again
    // by the consumer's target would corrupt them (P2 design B1).
    if graph.is_explicit_layout() {
        return perms;
    }
    for &nid in &graph.order {
        let Some(node) = graph.node(nid) else {
            continue;
        };
        let kind = match &node.target {
            OpTarget::TosaConv2d => Some(WeightPermKind::Conv2d),
            OpTarget::Conv2d => match node.args.get(1) {
                Some(Arg::Node(weight_id)) => graph.node(*weight_id).map(|weight_node| {
                    depthwise_weight_perm(graph, node, weight_node).unwrap_or({
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

fn make_graph_tensor(name: String, shape: Vec<i32>, dtype: DType, data: Vec<u8>) -> TosaTensor {
    if data.is_empty() {
        TosaTensor::new(&name, shape, dtype)
    } else {
        TosaTensor::with_data(&name, shape, dtype, data)
    }
}

fn placeholder_decl(node: &Node) -> (Vec<i32>, DType) {
    let raw_shape = raw_i32_shape(node);
    (
        apply_tosa_dim_order(node, raw_shape),
        ir_dtype_to_tosa(node.meta.val.dtype),
    )
}

fn serialize_const_placeholder(
    node: &Node,
    tensor_data: &TensorData,
    weight_perms: &HashMap<String, WeightPermKind>,
    dtype_overrides: &HashMap<String, DType>,
) -> TosaTensor {
    let store_dtype = ir_dtype_to_tosa(tensor_data.dtype);
    let tensor_dtype = dtype_overrides
        .get(&node.name)
        .copied()
        .or_else(|| node.meta.tosa_special_dtype.map(ir_dtype_to_tosa))
        .unwrap_or(store_dtype);
    let elem_sz = tensor_data.dtype.storage_size_bytes();

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

#[cfg(test)]
mod tests {
    fn reference_permute(data: &[u8], shape: &[i64], elem_size: usize, order: &[u8]) -> Vec<u8> {
        let rank = shape.len();
        let new_shape: Vec<usize> = order.iter().map(|&a| shape[a as usize] as usize).collect();
        let mut in_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * shape[i + 1] as usize;
        }
        let numel: usize = new_shape.iter().product();
        let mut out = vec![0u8; data.len()];
        for dst in 0..numel {
            let (mut rem, mut src) = (dst, 0usize);
            for axis in (0..rank).rev() {
                let coord = rem % new_shape[axis];
                rem /= new_shape[axis];
                src += coord * in_strides[order[axis] as usize];
            }
            out[dst * elem_size..(dst + 1) * elem_size]
                .copy_from_slice(&data[src * elem_size..(src + 1) * elem_size]);
        }
        out
    }

    #[test]
    fn permute_tensor_by_order_matches_a_reference_gather() {
        let cases: &[(&[i64], &[u8])] = &[
            (&[2, 3, 4, 5], &[0, 2, 3, 1]),
            (&[3, 1, 4, 2], &[2, 3, 0, 1]),
            (&[4, 6, 1, 1], &[0, 2, 3, 1]),
            (&[2, 3, 4, 5, 2], &[0, 2, 3, 4, 1]),
            (&[5, 7], &[1, 0]),
            (&[1, 3, 1, 4], &[2, 0, 3, 1]),
            (&[2, 1, 3], &[1, 2, 0]),
        ];
        for &(shape, order) in cases {
            for elem_size in [1usize, 2, 3, 4, 8] {
                let numel: i64 = shape.iter().product();
                let data: Vec<u8> = (0..numel as usize * elem_size).map(|i| (i * 7 + 3) as u8).collect();
                let (out, new_shape) = super::permute_tensor_by_order(&data, shape, elem_size, order);
                let expected_shape: Vec<i64> = order.iter().map(|&a| shape[a as usize]).collect();
                assert_eq!(new_shape, expected_shape, "{shape:?} {order:?}");
                assert_eq!(out, reference_permute(&data, shape, elem_size, order), "{shape:?} {order:?} {elem_size}");
            }
        }
    }

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
                        let src_idx = ((m + i) * 3 + h) * 3 + w;
                        let dst_idx = ((h * 3 + w) + i) * 16 + m;
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
                    let src_idx = (oc * 2 + h) * 2 + w;
                    let dst_idx = (h * 2 + w) * 6 + oc;
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
    fn scalar_placeholders_keep_rank0() {
        let node = Node {
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
        assert_eq!(placeholder_decl(&node).0, Vec::<i32>::new());

        let tensor = TensorData {
            data: vec![7u8],
            shape: vec![],
            dtype: crate::graph::ir::DType::Int8,
        };
        let context = TensorDeclarationContext::new(&Graph::new());
        let serialized = context.const_placeholder(&node, &tensor);
        assert_eq!(serialized.shape, Vec::<i32>::new());
    }
}
