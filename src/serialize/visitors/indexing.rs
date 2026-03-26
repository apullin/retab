use super::*;
use crate::graph::ir::Arg;

#[derive(Debug)]
struct ParsedSelectCopySpec {
    input_name: String,
    input_shape: Vec<i32>,
    mapped_dim: usize,
    index: i32,
    dtype: DType,
}

#[derive(Debug)]
struct ParsedSliceSpec {
    input_name: String,
    input_shape: Vec<i32>,
    raw_dim: i64,
    mapped_dim: usize,
    start: i64,
    end: i64,
    step: i64,
    start_shape: Vec<i32>,
    size_shape: Vec<i32>,
    input_target: String,
    input_order: String,
}

impl ParsedSelectCopySpec {
    // Parse and normalize axis/index once so emission works with a precise,
    // already-bounded view of the operation.
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Self {
        let input_names = collect_input_names(graph, node);
        let input_name = input_names.first().cloned().unwrap_or_default();
        let input_node = match node.args.first() {
            Some(Arg::Node(id)) => graph.node(*id),
            _ => None,
        };
        let input_shape = input_node
            .map(|n| actual_tensor_shape(graph, store, n))
            .unwrap_or_else(|| raw_i32_shape(node));

        let raw_dim = arg_i64(node, 1, 0);
        let rank = input_shape.len() as i64;
        let normalized_dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
        let mapped_dim = input_node
            .map(|n| map_axis_to_tosa_dim_order(n, normalized_dim as i32) as usize)
            .unwrap_or(normalized_dim as usize);

        let raw_index = arg_i64(node, 2, 0);
        let dim_size = input_shape.get(mapped_dim).copied().unwrap_or(1) as i64;
        let index = if raw_index < 0 {
            (dim_size + raw_index).max(0) as i32
        } else {
            raw_index.min(dim_size - 1) as i32
        };

        Self {
            input_name,
            input_shape,
            mapped_dim,
            index,
            dtype: node_output_dtype(node),
        }
    }
}

impl ParsedSliceSpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Self {
        let input_names = collect_input_names(graph, node);
        let input_name = input_names.first().cloned().unwrap_or_default();
        let input_node = match node.args.first() {
            Some(Arg::Node(id)) => graph.node(*id),
            _ => None,
        };
        let input_shape = input_node
            .map(|n| actual_tensor_shape(graph, store, n))
            .unwrap_or_else(|| raw_i32_shape(node));

        let raw_dim = arg_i64(node, 1, 0);
        let rank = input_shape.len() as i64;
        let raw_dim = if raw_dim < 0 { raw_dim + rank } else { raw_dim };
        let raw_dim = raw_dim.clamp(0, rank.saturating_sub(1));
        let mapped_dim = input_node
            .map(|n| map_axis_to_tosa_dim_order(n, raw_dim as i32) as i64)
            .unwrap_or(raw_dim) as usize;
        let start = arg_i64(node, 2, 0);
        let end = arg_i64(node, 3, input_shape.get(mapped_dim).copied().unwrap_or(0) as i64);
        let step = arg_i64(node, 4, 1);
        let input_dim = input_shape.get(mapped_dim).copied().unwrap_or(0) as i64;
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

        let mut start_shape = vec![0; input_shape.len()];
        start_shape[mapped_dim] = start_idx as i32;
        let mut size_shape = input_shape.clone();
        size_shape[mapped_dim] = slice_len as i32;

        let input_target = input_node
            .map(|n| format!("{:?}", n.target))
            .unwrap_or_else(|| "<none>".to_string());
        let input_order = input_node
            .and_then(|n| n.meta.tosa_dim_order.as_ref().map(|order| format!("{:?}", order)))
            .unwrap_or_else(|| "<none>".to_string());

        Self {
            input_name,
            input_shape,
            raw_dim,
            mapped_dim,
            start,
            end,
            step,
            start_shape,
            size_shape,
            input_target,
            input_order,
        }
    }
}

pub(super) fn emit_select(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Select,
        attribute: Attribute::Select(SelectAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

pub(super) fn emit_select_copy(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let spec = ParsedSelectCopySpec::parse(graph, node, store);
    let mut result = EmitResult::default();

    let slice_name = format!("{}_slice", node.name);
    let mut start_vec = vec![0i32; spec.input_shape.len()];
    start_vec[spec.mapped_dim] = spec.index;
    let mut size_vec = spec.input_shape.clone();
    size_vec[spec.mapped_dim] = 1;

    push_const_shape(&mut result, format!("{}_start", slice_name), start_vec);
    push_const_shape(&mut result, format!("{}_size", slice_name), size_vec.clone());
    result.tensors.push(TosaTensor {
        name: slice_name.clone(),
        shape: size_vec,
        dtype: spec.dtype,
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Slice,
        attribute: Attribute::Slice(SliceAttribute),
        inputs: vec![
            spec.input_name,
            format!("{}_start", slice_name),
            format!("{}_size", slice_name),
        ],
        outputs: vec![slice_name.clone()],
        location: None,
    });

    let output_shape = actual_tensor_shape(graph, store, node);
    push_reshape_with_shape(
        &mut result,
        slice_name,
        node.name.clone(),
        format!("{}_shape", node.name),
        output_shape,
        spec.dtype,
    );

    result
}

pub(super) fn emit_index_select(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);

    if input_names.len() < 2 {
        eprintln!(
            "[emit_index_select] WARNING: expected 2 node inputs, got {}",
            input_names.len()
        );
        return result;
    }

    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Gather,
        attribute: Attribute::Gather(GatherAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

pub(super) fn emit_gather(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    result.push_output_op(
        make_output_tensor(node),
        Op::Gather,
        Attribute::Gather(GatherAttribute),
        collect_input_names(graph, node),
    );
    result
}

pub(super) fn emit_scatter(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    result.push_output_op(
        make_output_tensor(node),
        Op::Scatter,
        Attribute::Scatter(ScatterAttribute),
        collect_input_names(graph, node),
    );
    result
}

pub(super) fn emit_slice(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let spec = ParsedSliceSpec::parse(graph, node, store);
    let mut result = EmitResult::default();

    if std::env::var("ARM_TOSA_ENGINE_DEBUG_SLICE").ok().as_deref() == Some("1") {
        eprintln!(
            "[SliceDebug] node={} input_target={} input_order={} input_shape={:?} raw_dim={} mapped_dim={} start={} end={} step={} start_shape={:?} size_shape={:?}",
            node.name,
            spec.input_target,
            spec.input_order,
            spec.input_shape,
            spec.raw_dim,
            spec.mapped_dim,
            spec.start,
            spec.end,
            spec.step,
            spec.start_shape,
            spec.size_shape
        );
    }

    push_const_shape(
        &mut result,
        format!("{}_slice_start", node.name),
        spec.start_shape,
    );
    push_const_shape(
        &mut result,
        format!("{}_slice_size", node.name),
        spec.size_shape.clone(),
    );
    result.tensors.push(TosaTensor {
        name: node.name.clone(),
        shape: spec.size_shape,
        dtype: ir_dtype_to_tosa(node.meta.val.dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });

    result.operators.push(TosaOperator {
        op: Op::Slice,
        attribute: Attribute::Slice(SliceAttribute),
        inputs: vec![
            spec.input_name,
            format!("{}_slice_start", node.name),
            format!("{}_slice_size", node.name),
        ],
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}
