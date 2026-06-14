use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, NodeId, NodeMeta};

pub(crate) fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1i64;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

pub(crate) fn packed_linear_scales(shape: &[i64]) -> Vec<i64> {
    contiguous_strides(shape)
}

pub(crate) struct LinearIndexSpec<'a> {
    pub anchor: NodeId,
    pub name: &'a str,
    pub meta: &'a NodeMeta,
    pub index_ids: &'a [NodeId],
    pub index_shapes: &'a [Vec<i64>],
    pub broadcast_shape: &'a [i64],
    pub linear_scales: &'a [i64],
    pub index_dtype: &'a str,
    pub flat_len: i64,
}

pub(crate) fn build_linear_index(graph: &mut IrGraph, spec: LinearIndexSpec<'_>) -> Option<NodeId> {
    if spec.index_ids.len() != spec.index_shapes.len()
        || spec.index_ids.len() != spec.linear_scales.len()
    {
        return None;
    }

    let mut linear_index = None;
    for (i, (&index_id, (index_shape, &linear_scale))) in spec
        .index_ids
        .iter()
        .zip(spec.index_shapes.iter().zip(spec.linear_scales.iter()))
        .enumerate()
    {
        let mut current = index_id;

        if index_shape.len() > spec.broadcast_shape.len() {
            return None;
        }
        if index_shape.len() != spec.broadcast_shape.len() {
            let mut aligned = vec![1; spec.broadcast_shape.len() - index_shape.len()];
            aligned.extend(index_shape);
            let aligned_id = graph.add_anchored_call_function(
                spec.anchor,
                format!("{}_idx{}_align", spec.name, i),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(current), i64_list(&aligned)],
                spec.meta.cloned_without_qparams(aligned, spec.index_dtype),
            );
            current = aligned_id;
        }

        let broadcast_id = graph.add_anchored_call_function(
            spec.anchor,
            format!("{}_idx{}_bcast", spec.name, i),
            "aten.expand_copy.default".to_string(),
            vec![IrArg::Node(current), i64_list(spec.broadcast_shape)],
            spec.meta
                .cloned_without_qparams(spec.broadcast_shape.to_vec(), spec.index_dtype),
        );

        let flat_id = graph.add_anchored_call_function(
            spec.anchor,
            format!("{}_idx{}_flat", spec.name, i),
            "aten.view_copy.default".to_string(),
            vec![IrArg::Node(broadcast_id), i64_list(&[spec.flat_len])],
            spec.meta
                .cloned_without_qparams(vec![spec.flat_len], spec.index_dtype),
        );

        let scale_id = graph.add_anchored_full_int(
            spec.anchor,
            format!("{}_idx{}_scale", spec.name, i),
            &[1],
            linear_scale,
            spec.index_dtype,
            spec.meta.cloned_without_qparams(vec![1], spec.index_dtype),
        );

        let scaled_id = graph.add_anchored_call_function(
            spec.anchor,
            format!("{}_idx{}_scaled", spec.name, i),
            "aten.mul.Tensor".to_string(),
            vec![IrArg::Node(flat_id), IrArg::Node(scale_id)],
            spec.meta
                .cloned_without_qparams(vec![spec.flat_len], spec.index_dtype),
        );

        linear_index = Some(match linear_index {
            None => scaled_id,
            Some(prev) => graph.add_anchored_call_function(
                spec.anchor,
                format!("{}_idx{}_add", spec.name, i),
                "aten.add.Tensor".to_string(),
                vec![IrArg::Node(prev), IrArg::Node(scaled_id)],
                spec.meta
                    .cloned_without_qparams(vec![spec.flat_len], spec.index_dtype),
            ),
        });
    }

    linear_index
}
