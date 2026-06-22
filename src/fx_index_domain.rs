use crate::fx_ir::{IrArg, IrGraph, NodeId};
use crate::fx_meta::{NodeMeta, TensorMetaStamp};
use crate::graph::ir::DType;

pub(crate) const DEFAULT_INDEX_DTYPE: &str = "torch.int64";
pub(crate) const TOSA_INDEX_DTYPE: &str = "torch.int32";
pub(crate) const BOOL_DTYPE: &str = "torch.bool";
const BOOL_GATHER_WORK_DTYPE: &str = "torch.int8";

pub(crate) fn first_index_dtype(graph: &IrGraph, indices: &[IrArg]) -> Option<String> {
    indices.iter().find_map(|item| match item {
        IrArg::Node(id) => graph.node_dtype(*id).map(str::to_string),
        _ => None,
    })
}

pub(crate) fn node_index_dtype_or_default(graph: &IrGraph, id: NodeId) -> String {
    graph
        .node_dtype(id)
        .unwrap_or(DEFAULT_INDEX_DTYPE)
        .to_string()
}

pub(crate) fn index_stamp<'a>(
    base: &'a NodeMeta,
    index_dtype: impl Into<std::borrow::Cow<'a, str>>,
) -> TensorMetaStamp<'a> {
    base.tensor_stamp(index_dtype)
}

pub(crate) fn int32_index_stamp(base: &NodeMeta) -> TensorMetaStamp<'_> {
    index_stamp(base, TOSA_INDEX_DTYPE)
}

pub(crate) fn bool_stamp(base: &NodeMeta) -> TensorMetaStamp<'_> {
    base.tensor_stamp(BOOL_DTYPE)
}

pub(crate) fn gather_work_stamp_dtype_of_or_float32<'a>(
    base: &'a NodeMeta,
    dtype_source: &'a NodeMeta,
) -> (TensorMetaStamp<'a>, bool) {
    let dtype = dtype_source.val_dtype_or_float32();
    if DType::from_torch_str(dtype) == DType::Bool {
        (base.tensor_stamp(BOOL_GATHER_WORK_DTYPE), true)
    } else {
        (base.tensor_stamp(dtype), false)
    }
}

pub(crate) fn gather_work_stamp_or_float32(base: &NodeMeta) -> (TensorMetaStamp<'_>, bool) {
    gather_work_stamp_dtype_of_or_float32(base, base)
}

pub(crate) fn mask_broadcast_shape(
    mask_pos: usize,
    mask_shape: &[i64],
    dest_rank: usize,
) -> Vec<i64> {
    let mut target = vec![1_i64; dest_rank];
    for (axis, &extent) in mask_shape.iter().enumerate() {
        if mask_pos + axis < dest_rank {
            target[mask_pos + axis] = extent;
        }
    }
    target
}

pub(crate) fn rank_unit_shape(rank: usize) -> Vec<i64> {
    vec![1_i64; rank]
}

pub(crate) fn is_scalar_shape(shape: &[i64]) -> bool {
    shape.iter().product::<i64>() == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::placeholder;

    #[test]
    fn index_dtype_uses_first_tensor_index() {
        let mut graph = IrGraph::empty();
        let int_index = placeholder(&mut graph, "idx", [3], "torch.int32");

        assert_eq!(
            first_index_dtype(
                &graph,
                &[IrArg::None, IrArg::Node(int_index), IrArg::Int(0)]
            ),
            Some("torch.int32".to_string())
        );
    }

    #[test]
    fn mask_shape_is_positioned_for_destination_broadcast() {
        assert_eq!(mask_broadcast_shape(1, &[4, 5], 4), vec![1, 4, 5, 1]);
        assert_eq!(rank_unit_shape(3), vec![1, 1, 1]);
    }

    #[test]
    fn bool_gather_uses_int8_work_dtype() {
        let base = NodeMeta::tensor([2, 3], "torch.bool");
        let (stamp, needs_cast) = gather_work_stamp_or_float32(&base);

        assert!(needs_cast);
        assert_eq!(stamp.dtype(), "torch.int8");
    }
}
