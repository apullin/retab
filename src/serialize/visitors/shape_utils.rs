use super::*;
use crate::serialize::op_semantics;
use std::cell::RefCell;
use std::collections::HashMap as StdHashMap;

thread_local! {
    static SHAPE_CACHE: RefCell<StdHashMap<String, Vec<i32>>> = RefCell::new(StdHashMap::new());
}

/// Clear the shape cache (call before each graph serialization).
pub(crate) fn clear_shape_cache() {
    SHAPE_CACHE.with(|c| c.borrow_mut().clear());
}

pub(super) fn literal_output_shape(node: &Node) -> Option<Vec<i32>> {
    match node.args.get(1) {
        Some(Arg::IntList(shape)) => Some(shape.iter().map(|&d| d as i32).collect()),
        _ => None,
    }
}

pub(super) fn raw_i32_shape(node: &Node) -> Vec<i32> {
    node.meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect()
}

/// The rank of `node`'s val.
pub(super) fn val_rank(node: &Node) -> usize {
    node.meta.val.shape_len as usize
}

fn resolve_inferred_shape(shape: &[i32], input_shape: &[i32]) -> Option<Vec<i32>> {
    let negative_positions: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (dim < 0).then_some(idx))
        .collect();
    if negative_positions.is_empty() {
        return Some(shape.to_vec());
    }
    if negative_positions.len() > 1 {
        return None;
    }

    let input_numel = input_shape
        .iter()
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim as i64))?;
    let known_product = shape
        .iter()
        .filter(|&&dim| dim >= 0)
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim as i64))?;
    if known_product == 0 || input_numel % known_product != 0 {
        return None;
    }

    let mut resolved = shape.to_vec();
    resolved[negative_positions[0]] = (input_numel / known_product) as i32;
    Some(resolved)
}

/// `axis` counted from the front of `node`'s val; an axis outside it comes back as is.
pub(super) fn normalize_axis(node: &Node, axis: i32) -> i32 {
    let rank = node.meta.val.shape_len as i32;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if (0..rank).contains(&normalized) {
        normalized
    } else {
        axis
    }
}

pub(super) fn first_input_node<'a>(graph: &'a Graph, node: &Node) -> Option<&'a Node> {
    let Arg::Node(input_id) = node.args.first()? else {
        return None;
    };
    graph.node(*input_id)
}

fn first_input_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    Some(actual_tensor_shape(
        graph,
        store,
        first_input_node(graph, node)?,
    ))
}

pub(super) fn transpose_perms(node: &Node) -> Vec<i32> {
    let perms = arg_or_kwarg_i32_list(node, 1, "dims");
    let rank = perms.len() as i32;
    perms.into_iter().map(|dim| dim.rem_euclid(rank)).collect()
}

fn passthrough_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !op_semantics::is_qdq_shape_passthrough(&node.target) {
        return None;
    }
    let input_node = first_input_node(graph, node)?;
    Some(actual_tensor_shape(graph, store, input_node))
}

fn shape_preserving_unary_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    if !op_semantics::is_shape_preserving_unary(&node.target) {
        return None;
    }
    first_input_shape(graph, store, node)
}

pub(super) fn transposed_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    if !op_semantics::is_transpose_shape_op(&node.target) {
        return None;
    }

    let input_shape = first_input_shape(graph, store, node)?;
    let perms = transpose_perms(node);
    if input_shape.len() != perms.len() {
        return None;
    }

    permute_shape(&input_shape, &perms)
}

fn sliced_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if !op_semantics::is_slice_shape_op(&node.target) {
        return None;
    }

    let input_node = first_input_node(graph, node)?;
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if input_shape.is_empty() {
        return None;
    }

    if node.target == OpTarget::TosaSlice {
        let size = shape_arg_i32_list(graph, node, 2);
        return (size.len() == input_shape.len()).then_some(size);
    }

    strided_slice_shape(node, &input_shape)
        .or_else(|| explicit_slice_size_shape(node, &input_shape))
}

/// An aten sum serializes at its val.
fn reduced_tensor_shape(node: &Node) -> Option<Vec<i32>> {
    op_semantics::is_aten_sum(&node.target).then(|| raw_i32_shape(node))
}

fn checked_numel(shape: &[i32]) -> Option<i64> {
    shape.iter().try_fold(1i64, |acc, &dim| {
        (dim >= 0)
            .then_some(dim as i64)
            .and_then(|dim| acc.checked_mul(dim))
    })
}

pub(super) fn broadcast_two_shapes(lhs: &[i32], rhs: &[i32]) -> Option<Vec<i32>> {
    let rank = lhs.len().max(rhs.len());
    let lhs_pad = rank.saturating_sub(lhs.len());
    let rhs_pad = rank.saturating_sub(rhs.len());
    let mut out = Vec::with_capacity(rank);

    for idx in 0..rank {
        let lhs_dim = if idx < lhs_pad { 1 } else { lhs[idx - lhs_pad] };
        let rhs_dim = if idx < rhs_pad { 1 } else { rhs[idx - rhs_pad] };
        let out_dim = match (lhs_dim, rhs_dim) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => return None,
        };
        out.push(out_dim);
    }

    Some(out)
}

pub(super) fn binary_input_broadcast_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    let mut input_shapes = node.args.iter().filter_map(|arg| {
        graph
            .node(arg.as_node()?)
            .map(|input_node| actual_tensor_shape(graph, store, input_node))
    });

    let mut candidate = input_shapes.next()?;
    for input_shape in input_shapes {
        candidate = broadcast_two_shapes(&candidate, &input_shape)?;
    }

    Some(candidate)
}

fn broadcasted_input_tensor_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    let candidate = binary_input_broadcast_shape(graph, store, node)?;

    let raw_shape = raw_i32_shape(node);
    if raw_shape.len() != candidate.len() {
        return None;
    }

    (checked_numel(&raw_shape)? == checked_numel(&candidate)?).then_some(candidate)
}

fn table_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Vec<i32>> {
    if node.target != OpTarget::TosaTable {
        return None;
    }
    first_input_shape(graph, store, node)
}

fn strided_slice_shape(node: &Node, input_shape: &[i32]) -> Option<Vec<i32>> {
    if node.args.len() < 4 {
        return None;
    }

    let rank = input_shape.len() as i64;
    let raw_dim = arg_i64(node, 1, 0);
    let normalized_dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
    let dim = normalized_dim.clamp(0, rank.saturating_sub(1)) as usize;
    let input_dim = input_shape.get(dim).copied()? as i64;
    let start = arg_i64(node, 2, 0);
    let end = arg_i64(node, 3, input_dim);
    let step = arg_i64(node, 4, 1);
    let start_idx = if start < 0 {
        (input_dim + start).max(0)
    } else {
        start.min(input_dim)
    };
    let mut end_idx = if end < 0 { input_dim + end } else { end };
    end_idx = end_idx.clamp(start_idx, input_dim);

    let slice_len = if step == 1 {
        (end_idx - start_idx).max(0)
    } else {
        ((end_idx - start_idx).max(0) + step - 1) / step
    };

    let mut output_shape = input_shape.to_vec();
    *output_shape.get_mut(dim)? = slice_len as i32;
    Some(output_shape)
}

fn explicit_slice_size_shape(node: &Node, input_shape: &[i32]) -> Option<Vec<i32>> {
    let size = kwarg_i32_list(node, "size");
    (size.len() == input_shape.len()).then_some(size)
}

/// Reconstruct the concrete TOSA tensor shape once at the serializer boundary
/// instead of ad hoc at each emitter.
pub(crate) fn actual_tensor_shape(graph: &Graph, store: &TensorStore, node: &Node) -> Vec<i32> {
    // Memoize: check cache first
    if let Some(cached) = SHAPE_CACHE.with(|c| c.borrow().get(&node.name).cloned()) {
        return cached;
    }
    let result = actual_tensor_shape_impl(graph, store, node);
    SHAPE_CACHE.with(|c| c.borrow_mut().insert(node.name.clone(), result.clone()));
    result
}

fn actual_tensor_shape_impl(graph: &Graph, store: &TensorStore, node: &Node) -> Vec<i32> {
    // A TOSA conv/pool/resize val is its TOSA shape (P2 design G1). The generic rules
    // below don't know these ops' operands: the broadcast rule would combine a conv's
    // input, weight and bias, or AVG_POOL2D's input and zero points (critique B1).
    if op_semantics::has_tosa_shape_val(&node.target) {
        return raw_i32_shape(node);
    }

    if op_semantics::is_shape_literal_reshape(&node.target) {
        if let Some(shape) = literal_output_shape(node) {
            let input_shape = first_input_node(graph, node)
                .map(|input| actual_tensor_shape(graph, store, input))
                .unwrap_or_default();
            if let Some(resolved_shape) = resolve_inferred_shape(&shape, &input_shape) {
                return resolved_shape;
            }
        }
    }

    if let Some(shape) = passthrough_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = shape_preserving_unary_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = transposed_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = sliced_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = reduced_tensor_shape(node) {
        return shape;
    }

    if let Some(shape) = table_tensor_shape(graph, store, node) {
        return shape;
    }

    if let Some(shape) = broadcasted_input_tensor_shape(graph, store, node) {
        return shape;
    }

    let mut raw_shape = raw_i32_shape(node);
    if raw_shape.iter().any(|&dim| dim < 0) {
        if let Some(input_node) = first_input_node(graph, node) {
            let input_shape = actual_tensor_shape(graph, store, input_node);
            if let Some(resolved_shape) = resolve_inferred_shape(&raw_shape, &input_shape) {
                raw_shape = resolved_shape;
            }
        }
    }
    raw_shape
}

pub(super) fn promoted_2d_matmul_shape(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> Option<Vec<i32>> {
    if !op_semantics::is_matmul_shape_op(&node.target) {
        return None;
    }

    match (node.args.first(), node.args.get(1)) {
        (Some(Arg::Node(a_id)), Some(Arg::Node(b_id))) => {
            let a_shape = graph
                .node(*a_id)
                .map(|n| actual_tensor_shape(graph, store, n))
                .unwrap_or_default();
            let b_shape = graph
                .node(*b_id)
                .map(|n| actual_tensor_shape(graph, store, n))
                .unwrap_or_default();
            if a_shape.len() == 2 && b_shape.len() == 2 {
                Some(vec![1, a_shape[0], b_shape[1]])
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn make_output_tensor(node: &Node) -> TosaTensor {
    make_empty_tensor(
        node.name.clone(),
        raw_i32_shape(node),
        node_output_dtype(node),
    )
}

pub(super) fn make_actual_output_tensor(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
) -> TosaTensor {
    make_empty_tensor(
        node.name.clone(),
        actual_tensor_shape(graph, store, node),
        node_output_dtype(node),
    )
}

pub(super) fn push_reshape_with_shape(
    result: &mut EmitResult,
    input_name: String,
    output_name: String,
    shape_name: String,
    out_shape: Vec<i32>,
    dtype: DType,
) {
    let output_tensor = make_empty_tensor(output_name, out_shape.clone(), dtype);
    push_const_shape(result, shape_name.clone(), out_shape);
    result.push_output_op(
        output_tensor,
        Op::Reshape,
        Attribute::Reshape(ReshapeAttribute {}),
        vec![input_name, shape_name],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{Arg, DType, NodeMeta, OpTarget, OpType, TensorInfo};
    use crate::graph::Graph;
    use crate::store::TensorStore;
    use smallvec::smallvec;

    fn meta(shape: &[i64], dtype: DType) -> NodeMeta {
        NodeMeta::new(TensorInfo::new(shape, dtype))
    }

    #[test]
    fn scalar_while_predicate_reshape_keeps_shape_and_output_ranks_equal() {
        clear_shape_cache();
        let mut graph = Graph::new();
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("predicate".into()),
            smallvec![],
            Vec::new(),
            meta(&[1], DType::Bool),
            "predicate".into(),
        );
        let reshape = graph.create_node(
            OpType::CallFunction,
            OpTarget::View,
            smallvec![Arg::Node(input), Arg::IntList(smallvec![])],
            Vec::new(),
            meta(&[], DType::Bool),
            "predicate_scalar".into(),
        );
        let result = emit_node(
            &graph,
            reshape,
            graph.node(reshape).unwrap(),
            &TensorStore::new(),
        )
        .unwrap();
        assert_eq!(result.operators.last().unwrap().op, Op::Reshape);
        assert!(result.tensors.last().unwrap().shape.is_empty());
        assert_eq!(result.shapes[0].rank, 0);
        assert!(result.shapes[0].data.is_empty());
    }

    #[test]
    fn regression_negative_permutation_dims_emit_nonnegative_transpose_perms() {
        clear_shape_cache();
        let mut graph = Graph::new();
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("x".into()),
            smallvec![],
            Vec::new(),
            meta(&[1, 2, 3, 5], DType::Float32),
            "x".into(),
        );
        let transpose = graph.create_node(
            OpType::CallFunction,
            OpTarget::Permute,
            smallvec![Arg::Node(input), Arg::IntList(smallvec![-4, -2, -1, -3])],
            Vec::new(),
            meta(&[1, 3, 5, 2], DType::Float32),
            "transpose".into(),
        );
        let store = TensorStore::default();
        let node = graph.node(transpose).unwrap();
        let result = emit_node(&graph, transpose, node, &store).unwrap();

        assert_eq!(result.operators.len(), 1);
        assert_eq!(result.operators[0].op, Op::Transpose);
        let Attribute::Transpose(attr) = &result.operators[0].attribute else {
            panic!("expected transpose attribute");
        };
        assert_eq!(attr.perms, vec![0, 2, 3, 1]);
        assert_eq!(result.tensors[0].shape, vec![1, 3, 5, 2]);
        assert_eq!(
            transposed_tensor_shape(&graph, &store, node),
            Some(vec![1, 3, 5, 2])
        );
    }

    #[test]
    fn reciprocal_uses_input_shape_when_meta_is_stale_scalar() {
        clear_shape_cache();

        let mut graph = Graph::new();
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::Relu,
            smallvec![],
            Vec::new(),
            meta(&[10, 10, 10], DType::Float32),
            "x".to_string(),
        );
        let reciprocal = graph.create_node(
            OpType::CallFunction,
            OpTarget::Reciprocal,
            smallvec![Arg::Node(input)],
            Vec::new(),
            meta(&[], DType::Float32),
            "x_recip".to_string(),
        );

        let store = TensorStore::default();
        let reciprocal_node = graph.node(reciprocal).expect("reciprocal node");
        assert_eq!(
            actual_tensor_shape(&graph, &store, reciprocal_node),
            vec![10, 10, 10]
        );
    }
}
