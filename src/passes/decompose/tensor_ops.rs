use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;
use smallvec::smallvec;

// ---------------------------------------------------------------------------
// DecomposeSelectPass: Select -> Slice + Squeeze
// ---------------------------------------------------------------------------

/// Decompose `Select(input, dim, index)` into:
///   slice = Slice(input, dim, index, index+1)
///   result = Squeeze(slice, [dim])
pub struct DecomposeSelectPass;

impl Pass for DecomposeSelectPass {
    fn name(&self) -> &str {
        "DecomposeSelect"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Select);
        if targets.is_empty() {
            return false;
        }

        for id in targets {
            decompose_select(graph, id);
        }
        true
    }
}

fn decompose_select(graph: &mut Graph, node_id: NodeId) {
    let (input_id, dim, index, input_shape) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return,
        };

        let input_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let dim = match node.args.get(1) {
            Some(Arg::Int(d)) => *d,
            _ => return,
        };

        let index = match node.args.get(2) {
            Some(Arg::Int(i)) => *i,
            _ => return,
        };

        let input_shape = graph
            .node(input_id)
            .map(|n| n.meta.val.shape_slice().to_vec())
            .unwrap_or_default();

        (input_id, dim, index, input_shape)
    };

    if input_shape.is_empty() {
        return;
    }

    let rank = input_shape.len() as i64;
    let dim = if dim < 0 { dim + rank } else { dim };
    let dim_size = input_shape[dim as usize];
    let index = if index < 0 { index + dim_size } else { index };

    let input_dtype = graph
        .node(input_id)
        .map(|n| n.meta.val.dtype)
        .unwrap_or(DType::Int8);

    let node_name = graph
        .node(node_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();

    // 1. Slice(input, dim, index, index+1)
    let slice_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Slice,
        smallvec![
            Arg::Node(input_id),
            Arg::Int(dim),
            Arg::Int(index),
            Arg::Int(index + 1),
        ],
        vec![],
        NodeMeta::new(TensorInfo::default()),
        format!("{}_slice", node_name),
    );
    crate::shape::propagate_shape(graph, slice_id);

    // 2. Squeeze(slice, [dim]) — this will be rewritten to TosaReshape in cleanup
    // Compute output shape: remove the dim
    let mut out_shape: Vec<i64> = input_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim as usize)
        .map(|(_, &s)| s)
        .collect();
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let out_shape_sv: smallvec::SmallVec<[i64; 6]> = out_shape.iter().copied().collect();

    let squeeze_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Squeeze,
        smallvec![Arg::Node(slice_id), Arg::IntList(out_shape_sv)],
        vec![],
        NodeMeta::new(TensorInfo::new(&out_shape, input_dtype)),
        format!("{}_squeeze", node_name),
    );
    crate::shape::propagate_shape(graph, squeeze_id);

    graph.replace_all_uses(node_id, squeeze_id);
    graph.erase_node(node_id);
}

// ---------------------------------------------------------------------------
// DecomposeEmbeddingPass: Embedding -> Reshape + IndexSelect(→Gather)
// ---------------------------------------------------------------------------

/// Decompose `Embedding(weight, indices)` into:
///   flat_indices = Reshape(indices, [prod(indices_shape)])
///   selected = Select(weight, 0, flat_indices) — rewritten later
///   result = Reshape(selected, indices_shape + [embed_dim])
pub struct DecomposeEmbeddingPass;

impl Pass for DecomposeEmbeddingPass {
    fn name(&self) -> &str {
        "DecomposeEmbedding"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Embedding);
        if targets.is_empty() {
            return false;
        }

        for id in targets {
            decompose_embedding(graph, id);
        }
        true
    }
}

fn decompose_embedding(graph: &mut Graph, node_id: NodeId) {
    let (weight_id, indices_id, weight_shape, indices_shape, output_info) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return,
        };

        let weight_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let indices_id = match node.args.get(1) {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let weight_shape = graph
            .node(weight_id)
            .map(|n| n.meta.val.shape_slice().to_vec())
            .unwrap_or_default();

        let indices_shape = graph
            .node(indices_id)
            .map(|n| n.meta.val.shape_slice().to_vec())
            .unwrap_or_default();

        let output_info = node.meta.val;

        (weight_id, indices_id, weight_shape, indices_shape, output_info)
    };

    if weight_shape.len() < 2 || indices_shape.is_empty() {
        return;
    }

    let embed_dim = weight_shape[1];
    let flat_size: i64 = indices_shape.iter().product();
    let dtype = graph
        .node(weight_id)
        .map(|n| n.meta.val.dtype)
        .unwrap_or(DType::Int8);

    let node_name = graph
        .node(node_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();

    // 1. Flatten indices: Reshape(indices, [flat_size])
    let flat_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Reshape,
        smallvec![Arg::Node(indices_id), Arg::IntList(smallvec![flat_size])],
        vec![],
        NodeMeta::new(TensorInfo::new(&[flat_size], DType::Int32)),
        format!("{}_flat_idx", node_name),
    );
    crate::shape::propagate_shape(graph, flat_id);

    // 2. IndexSelect: Select(weight, 0, flat_indices)
    let select_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Select,
        smallvec![
            Arg::Node(weight_id),
            Arg::Int(0),
            Arg::Node(flat_id),
        ],
        vec![],
        NodeMeta::new(TensorInfo::new(&[flat_size, embed_dim], dtype)),
        format!("{}_index_select", node_name),
    );

    // 3. Reshape back: Reshape(selected, indices_shape + [embed_dim])
    let mut out_shape = indices_shape;
    out_shape.push(embed_dim);
    let out_shape_sv: smallvec::SmallVec<[i64; 6]> = out_shape.iter().copied().collect();

    let reshape_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Reshape,
        smallvec![Arg::Node(select_id), Arg::IntList(out_shape_sv)],
        vec![],
        NodeMeta::new(output_info),
        format!("{}_reshape", node_name),
    );
    crate::shape::propagate_shape(graph, reshape_id);

    graph.replace_all_uses(node_id, reshape_id);
    graph.erase_node(node_id);
}

// ---------------------------------------------------------------------------
// DecomposeCatPass: Cat -> TosaConcat
// ---------------------------------------------------------------------------

/// Rewrite `Cat(tensors, dim)` to `TosaConcat(tensors)` with axis kwarg.
pub struct DecomposeCatPass;

impl Pass for DecomposeCatPass {
    fn name(&self) -> &str {
        "DecomposeCat"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Cat);
        if targets.is_empty() {
            return false;
        }

        let mut modified = false;
        for id in targets {
            if rewrite_cat(graph, id) {
                modified = true;
            }
        }
        modified
    }
}

fn rewrite_cat(graph: &mut Graph, node_id: NodeId) -> bool {
    let (tensor_ids, dim) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        // Cat args: [NodeList(tensors), Int(dim)]
        let tensor_ids = match node.args.first() {
            Some(Arg::NodeList(ids)) => ids.clone(),
            _ => return false,
        };

        let dim = match node.args.get(1) {
            Some(Arg::Int(d)) => *d,
            _ => 0,
        };

        (tensor_ids, dim)
    };

    // Build args as individual node refs for TOSA concat
    let mut new_args: smallvec::SmallVec<[Arg; 4]> = smallvec![];
    for id in &tensor_ids {
        new_args.push(Arg::Node(*id));
    }

    if let Some(node) = graph.node_mut(node_id) {
        node.target = OpTarget::TosaConcat;
        node.args = new_args;
        node.kwargs = vec![("axis".into(), Arg::Int(dim))];
    }

    crate::shape::propagate_shape(graph, node_id);

    true
}

// ---------------------------------------------------------------------------
// DecomposeDivPass: Div -> Reciprocal + Mul
// ---------------------------------------------------------------------------

/// Decompose `Div(a, b)` into:
///   recip = Reciprocal(b)
///   result = Mul(a, recip)
pub struct DecomposeDivPass;

impl Pass for DecomposeDivPass {
    fn name(&self) -> &str {
        "DecomposeDiv"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Div);
        if targets.is_empty() {
            return false;
        }

        for id in targets {
            decompose_div(graph, id);
        }
        true
    }
}

fn decompose_div(graph: &mut Graph, node_id: NodeId) {
    let (a_id, b_id, output_info) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return,
        };

        let a_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let b_id = match node.args.get(1) {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        (a_id, b_id, node.meta.val)
    };

    let b_info = graph
        .node(b_id)
        .map(|n| n.meta.val)
        .unwrap_or_default();

    let node_name = graph
        .node(node_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();

    // 1. Reciprocal(b)
    let recip_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Reciprocal,
        smallvec![Arg::Node(b_id)],
        vec![],
        NodeMeta::new(b_info),
        format!("{}_recip", node_name),
    );

    // 2. Mul(a, recip)
    let mul_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(a_id), Arg::Node(recip_id)],
        vec![],
        NodeMeta::new(output_info),
        format!("{}_mul", node_name),
    );
    crate::shape::propagate_shape(graph, mul_id);

    graph.replace_all_uses(node_id, mul_id);
    graph.erase_node(node_id);
}

// ---------------------------------------------------------------------------
// DecomposeFloorDividePass: FloorDivide -> Div + Floor
// ---------------------------------------------------------------------------

/// Decompose `FloorDivide(a, b)` into:
///   div = Div(a, b)
///   result = Floor(div)
pub struct DecomposeFloorDividePass;

impl Pass for DecomposeFloorDividePass {
    fn name(&self) -> &str {
        "DecomposeFloorDivide"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::FloorDivide);
        if targets.is_empty() {
            return false;
        }

        for id in targets {
            decompose_floor_divide(graph, id);
        }
        true
    }
}

fn decompose_floor_divide(graph: &mut Graph, node_id: NodeId) {
    let (a_id, b_id, output_info) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return,
        };

        let a_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let b_id = match node.args.get(1) {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        (a_id, b_id, node.meta.val)
    };

    let node_name = graph
        .node(node_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();

    // 1. Div(a, b)
    let div_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Div,
        smallvec![Arg::Node(a_id), Arg::Node(b_id)],
        vec![],
        NodeMeta::new(output_info),
        format!("{}_div", node_name),
    );
    crate::shape::propagate_shape(graph, div_id);

    // 2. Floor(div)
    let floor_id = graph.insert_before(
        node_id,
        OpType::CallFunction,
        OpTarget::Floor,
        smallvec![Arg::Node(div_id)],
        vec![],
        NodeMeta::new(output_info),
        format!("{}_floor", node_name),
    );

    graph.replace_all_uses(node_id, floor_id);
    graph.erase_node(node_id);
}

// ---------------------------------------------------------------------------
// DecomposeWherePass: Where(cond, x, y) -> TosaSelect(cond, x, y)
// ---------------------------------------------------------------------------

/// Rewrite `Where(cond, x, y)` to `TosaSelect(cond, x, y)`.
pub struct DecomposeWherePass;

impl Pass for DecomposeWherePass {
    fn name(&self) -> &str {
        "DecomposeWhere"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Where);
        if targets.is_empty() {
            return false;
        }

        let mut modified = false;
        for id in targets {
            if let Some(node) = graph.node_mut(id) {
                node.target = OpTarget::TosaSelect;
                modified = true;
            }
        }
        modified
    }
}
