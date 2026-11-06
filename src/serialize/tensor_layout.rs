use std::collections::HashMap;

use crate::graph::ir::{Arg, Node};
use crate::graph::Graph;
use crate::store::TensorData;

use super::op_semantics;
use super::tosa_types::{DType, TosaTensor};
use super::visitors::ir_dtype_to_tosa;

pub(super) struct TensorDeclarationContext {
    dtype_overrides: HashMap<String, DType>,
}

impl TensorDeclarationContext {
    pub(super) fn new(graph: &Graph) -> Self {
        Self {
            dtype_overrides: collect_tensor_dtype_overrides(graph),
        }
    }

    pub(super) fn runtime_placeholder(&self, node: &Node) -> TosaTensor {
        let (shape, dtype) = placeholder_decl(node);
        make_graph_tensor(node.name.clone(), shape, dtype, Vec::new())
    }

    pub(super) fn const_placeholder(&self, node: &Node, tensor_data: &TensorData) -> TosaTensor {
        serialize_const_placeholder(node, tensor_data, &self.dtype_overrides)
    }
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
    (raw_i32_shape(node), ir_dtype_to_tosa(node.meta.val.dtype))
}

/// rc4's conv rewrite already put conv weights in TOSA's OHWI/HWCM/ODHWI layouts, so a
/// constant is not permuted for its consumer (P2 design B1).
fn serialize_const_placeholder(
    node: &Node,
    tensor_data: &TensorData,
    dtype_overrides: &HashMap<String, DType>,
) -> TosaTensor {
    let store_dtype = ir_dtype_to_tosa(tensor_data.dtype);
    let tensor_dtype = dtype_overrides
        .get(&node.name)
        .copied()
        .or_else(|| node.meta.tosa_special_dtype.map(ir_dtype_to_tosa))
        .unwrap_or(store_dtype);
    let final_shape: Vec<i32> = tensor_data.shape.iter().map(|&d| d as i32).collect();
    let final_data = match (tensor_dtype, store_dtype) {
        (DType::Int4, _) if store_dtype != DType::Int4 => pack_int4_data(&tensor_data.data),
        (DType::Int48, DType::Int32) => widen_int32_data_to_int48(&tensor_data.data),
        _ => tensor_data.data.clone(),
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
    use crate::graph::ir::{NodeMeta, OpTarget, OpType};
    use crate::store::TensorData;

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
