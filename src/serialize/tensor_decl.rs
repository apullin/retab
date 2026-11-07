//! The serializer's tensor declarations for placeholders: a runtime input at its val's
//! shape and dtype, a constant at its stored shape with INT4 data packed and an INT48
//! bias widened.

use crate::graph::ir::Node;
use crate::store::TensorData;

use super::tosa_types::{DType, TosaTensor};
use super::visitors::ir_dtype_to_tosa;

/// A runtime input's tensor, at its val's shape and dtype.
pub(super) fn runtime_placeholder(node: &Node) -> TosaTensor {
    let (shape, dtype) = placeholder_decl(node);
    make_graph_tensor(node.name.clone(), shape, dtype, Vec::new())
}

fn raw_i32_shape(node: &Node) -> Vec<i32> {
    node.meta
        .val
        .shape_slice()
        .iter()
        .map(|&dim| dim as i32)
        .collect()
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

/// A constant's tensor, at its stored shape, typed by its `tosa_special_dtype` when it has
/// one (an INT4 weight; an INT48 conv bias, which IrRewriteConv and captured_conv mark),
/// else by its stored dtype. rc4's conv rewrite already put conv weights in TOSA's
/// OHWI/HWCM/ODHWI layouts, so a constant is not permuted for its consumer (P2 design B1).
pub(super) fn const_placeholder(node: &Node, tensor_data: &TensorData) -> TosaTensor {
    let store_dtype = ir_dtype_to_tosa(tensor_data.dtype);
    let tensor_dtype = node
        .meta
        .tosa_special_dtype
        .map(ir_dtype_to_tosa)
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
        let serialized = const_placeholder(&node, &tensor);
        assert_eq!(serialized.shape, Vec::<i32>::new());
    }
}
