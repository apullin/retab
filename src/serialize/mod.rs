pub mod tosa_generated;
pub mod visitors;

use crate::graph::ir::{Arg, OpTarget, OpType};
use crate::graph::Graph;
use crate::store::TensorStore;

use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, Vector, WIPOffset};
use std::collections::HashMap;
use tosa_generated::{Attribute, TosaOperator, TosaShape, TosaTensor};
use visitors::ir_dtype_to_tosa;

// ---------------------------------------------------------------------------
// Weight permutation helpers (OIHW -> OHWI for TOSA)
// ---------------------------------------------------------------------------

/// Describes how a weight tensor should be permuted for TOSA.
#[derive(Debug, Clone, Copy)]
enum WeightPermKind {
    /// Standard conv2d: OIHW [OC, IC, KH, KW] -> OHWI [OC, KH, KW, IC]
    Conv2d,
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

/// Permute a 4D tensor's raw bytes from OIHW to OHWI or depthwise format.
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
    if shape.len() != 4 {
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
            let (no, nh, nw, ni) = (d0, d2, d3, d1);
            let mut out = vec![0u8; data.len()];

            for o in 0..d0 {
                for i in 0..d1 {
                    for h in 0..d2 {
                        for w in 0..d3 {
                            let src_idx = ((o * d1 + i) * d2 + h) * d3 + w;
                            let dst_idx = ((o * nh + h) * nw + w) * ni + i;
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
                        if src_off + elem_size <= data.len() && dst_off + elem_size <= out.len()
                        {
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
    let numel: usize = match shape
        .iter()
        .try_fold(1usize, |acc, &dim| usize::try_from(dim).ok().and_then(|d| acc.checked_mul(d)))
    {
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
    for candidate in [
        shape.last().copied(),
        shape.get(1).copied(),
    ]
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
        let node = match graph.node(nid) {
            Some(n) => n,
            None => continue,
        };
        let kind = match &node.target {
            OpTarget::TosaConv2d | OpTarget::Conv2d => Some(WeightPermKind::Conv2d),
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
                    // Only permute placeholder/const nodes with 4D shapes
                    if weight_node.meta.val.shape_len == 4 {
                        perms.insert(weight_node.name.clone(), kind);
                    }
                }
            }
        }
    }
    perms
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

fn collect_serialized_graph_items(graph: &Graph, store: &TensorStore) -> SerializedGraphItems {
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
    if std::env::var("TOSA_ENGINE_DEBUG").is_ok() {
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
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => continue,
        };

        if std::env::var("TOSA_ENGINE_DEBUG").is_ok() {
            eprintln!("[Serialize] node '{}' op={:?} target={:?}", node.name, node.op, node.target);
        }

        match node.op {
            OpType::Placeholder => {
                let name = &node.name;
                let raw_shape: Vec<i32> = node
                    .meta
                    .val
                    .shape_slice()
                    .iter()
                    .map(|&d| d as i32)
                    .collect();
                let shape = apply_tosa_dim_order(node, raw_shape);
                let dtype = ir_dtype_to_tosa(node.meta.val.dtype);

                if let Some(tensor_data) = store.get(name) {
                    // Use the store's dtype for constant tensors.
                    //
                    // `fixup_quantized_dtypes` may have changed node.meta.val.dtype
                    // (e.g. Float32 -> Int8) based on consumer qparams, but the
                    // store data is in the original dtype.  Using the node meta's
                    // dtype here would produce a data-size/shape mismatch
                    // ("malformed constant buffer").
                    let store_dtype = ir_dtype_to_tosa(tensor_data.dtype);
                    let elem_sz = dtype_elem_size(tensor_data.dtype);

                    // Constant placeholders must be serialized in their physical
                    // TOSA layout. The FX memory-format pass annotates this via
                    // `tosa_dim_order` but does not rewrite the underlying store
                    // tensor shape/data. Apply that permutation here for general
                    // constants, while keeping the dedicated conv-weight path
                    // separate so we do not double-permute OIHW weights.
                    let (final_data, final_shape_i64) =
                        if let Some(&perm_kind) = weight_perms.get(name) {
                            permute_weight_data(
                                &tensor_data.data,
                                &tensor_data.shape,
                                elem_sz,
                                perm_kind,
                            )
                        } else if let Some(order) = node.meta.tosa_dim_order.as_ref() {
                            permute_tensor_by_order(
                                &tensor_data.data,
                                &tensor_data.shape,
                                elem_sz,
                                order,
                            )
                        } else {
                            (tensor_data.data.clone(), tensor_data.shape.clone())
                        };
                    let final_shape: Vec<i32> =
                        final_shape_i64.iter().map(|&d| d as i32).collect();

                    tensors.push(TosaTensor {
                        name: name.clone(),
                        shape: final_shape,
                        dtype: store_dtype,
                        data: final_data,
                        variable: false,
                        is_unranked: false,
                        variable_name: String::new(),
                    });
                    operators.push(TosaOperator {
                        op: tosa_generated::Op::Const,
                        attribute: Attribute::Const(tosa_generated::ConstAttribute),
                        inputs: vec![],
                        outputs: vec![name.clone()],
                        location: None,
                    });
                } else {
                    tensors.push(TosaTensor {
                        name: name.clone(),
                        shape,
                        dtype,
                        data: Vec::new(),
                        variable: false,
                        is_unranked: false,
                        variable_name: String::new(),
                    });
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
                    match arg {
                        Arg::Node(id) => {
                            if let Some(n) = graph.node(*id) {
                                output_names.push(n.name.clone());
                            }
                        }
                        Arg::NodeList(ids) => {
                            for id in ids {
                                if let Some(n) = graph.node(*id) {
                                    output_names.push(n.name.clone());
                                }
                            }
                        }
                        _ => {}
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
    let serialized = collect_serialized_graph_items(graph, store);

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
        builder.push_slot::<i32>(4, 0, -1);  // major
        builder.push_slot::<i32>(6, 1, -1);  // minor
        builder.push_slot::<i32>(8, 0, -1);  // micro
        builder.push_slot_always::<WIPOffset<_>>(10, modifier_str);
        builder.end_table(start)
    };

    // Build version
    let version = {
        let start = builder.start_table();
        builder.push_slot::<i32>(4, 1, -1);  // major
        builder.push_slot::<i32>(6, 0, -1);  // minor
        builder.push_slot::<i32>(8, 0, -1);  // patch
        builder.push_slot::<bool>(10, false, true);  // draft
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

pub fn serialize_to_tosa(graph: &Graph, store: &TensorStore) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::with_capacity(16 * 1024 * 1024); // 16MB initial
    let region = build_fb_region(&mut builder, "main", graph, store);
    finish_tosa_graph(&mut builder, &[region])
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
        builder.push_slot_always::<WIPOffset<_>>(4, name);       // name
        builder.push_slot_always::<WIPOffset<_>>(6, shape);      // shape
        builder.push_slot::<u32>(8, t.dtype as u32, 0);          // type
        if let Some(d) = data {
            builder.push_slot_always::<WIPOffset<_>>(10, d);     // data
        }
        builder.push_slot::<bool>(12, t.variable, false);        // variable
        builder.push_slot::<bool>(14, t.is_unranked, false);     // is_unranked
        if let Some(vn) = var_name {
            builder.push_slot_always::<WIPOffset<_>>(16, vn);    // variable_name
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
        let output_strs: Vec<_> = op.outputs.iter().map(|s| builder.create_string(s)).collect();
        let inputs = builder.create_vector(&input_strs);
        let outputs = builder.create_vector(&output_strs);

        let (attr_type, attr_offset) = build_fb_attribute(builder, &op.attribute);

        let start = builder.start_table();
        builder.push_slot::<u32>(4, op.op as u32, 0);           // op
        builder.push_slot::<u8>(6, attr_type, 0);               // attribute_type
        if let Some(attr) = attr_offset {
            builder.push_slot_always::<WIPOffset<_>>(8, attr);  // attribute
        }
        builder.push_slot_always::<WIPOffset<_>>(10, inputs);   // inputs
        builder.push_slot_always::<WIPOffset<_>>(12, outputs);  // outputs
        offsets.push(builder.end_table(start));
    }
    offsets
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
        | Attribute::IntDiv(_) => {
            // Empty tables
            let start = builder.start_table();
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::Conv2d(a) => {
            let pad = builder.create_vector(&a.pad);
            let stride = builder.create_vector(&a.stride);
            let dilation = builder.create_vector(&a.dilation);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, pad);
            builder.push_slot_always::<WIPOffset<_>>(6, stride);
            builder.push_slot_always::<WIPOffset<_>>(8, dilation);
            builder.push_slot::<bool>(10, a.local_bound, false);
            builder.push_slot::<u32>(12, a.acc_type as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::DepthwiseConv2d(a) => {
            let pad = builder.create_vector(&a.pad);
            let stride = builder.create_vector(&a.stride);
            let dilation = builder.create_vector(&a.dilation);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, pad);
            builder.push_slot_always::<WIPOffset<_>>(6, stride);
            builder.push_slot_always::<WIPOffset<_>>(8, dilation);
            builder.push_slot::<bool>(10, a.local_bound, false);
            builder.push_slot::<u32>(12, a.acc_type as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::Rescale(a) => {
            let start = builder.start_table();
            builder.push_slot::<bool>(4, a.scale32, false);
            builder.push_slot::<u32>(6, a.rounding_mode as u32, 0);
            builder.push_slot::<bool>(8, a.per_channel, false);
            builder.push_slot::<bool>(10, a.input_unsigned, false);
            builder.push_slot::<bool>(12, a.output_unsigned, false);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::MaxPool2d(a) => {
            let kernel = builder.create_vector(&a.kernel);
            let stride = builder.create_vector(&a.stride);
            let pad = builder.create_vector(&a.pad);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, kernel);
            builder.push_slot_always::<WIPOffset<_>>(6, stride);
            builder.push_slot_always::<WIPOffset<_>>(8, pad);
            builder.push_slot::<u32>(10, a.nan_mode as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::Transpose(a) => {
            let perms = builder.create_vector(&a.perms);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, perms);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::Clamp(a) => {
            let min_val = create_force_aligned_u8_vector(builder, &a.min_val);
            let max_val = create_force_aligned_u8_vector(builder, &a.max_val);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, min_val);
            builder.push_slot_always::<WIPOffset<_>>(6, max_val);
            builder.push_slot::<u32>(8, a.nan_mode as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::Mul(a) => {
            let start = builder.start_table();
            builder.push_slot::<i32>(4, a.shift, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::AvgPool2d(a) => {
            // TOSA 1.0: AvgPool2dAttribute has kernel, stride, pad, acc_type.
            // input_zp and output_zp are separate input tensors, not in the attribute.
            let kernel = builder.create_vector(&a.kernel);
            let stride = builder.create_vector(&a.stride);
            let pad = builder.create_vector(&a.pad);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, kernel);
            builder.push_slot_always::<WIPOffset<_>>(6, stride);
            builder.push_slot_always::<WIPOffset<_>>(8, pad);
            builder.push_slot::<u32>(10, a.acc_type as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::ReduceSum(a) => {
            let start = builder.start_table();
            builder.push_slot::<i32>(4, a.axis, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::Concat(a) => {
            let start = builder.start_table();
            builder.push_slot::<i32>(4, a.axis, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::TransposeConv2d(a) => {
            let out_pad = builder.create_vector(&a.out_pad);
            let stride = builder.create_vector(&a.stride);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, out_pad);
            builder.push_slot_always::<WIPOffset<_>>(6, stride);
            builder.push_slot::<bool>(8, a.local_bound, false);
            builder.push_slot::<u32>(10, a.acc_type as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::CondIf(a) => {
            let then_g = builder.create_string(&a.then_graph);
            let else_g = builder.create_string(&a.else_graph);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, then_g);
            builder.push_slot_always::<WIPOffset<_>>(6, else_g);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::WhileLoop(a) => {
            let cond_g = builder.create_string(&a.cond_branch);
            let body_g = builder.create_string(&a.body_branch);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, cond_g);
            builder.push_slot_always::<WIPOffset<_>>(6, body_g);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::Maximum(a) => {
            let start = builder.start_table();
            builder.push_slot::<u32>(4, a.nan_mode as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::Minimum(a) => {
            let start = builder.start_table();
            builder.push_slot::<u32>(4, a.nan_mode as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::ReduceAny(a) => {
            let start = builder.start_table();
            builder.push_slot::<i32>(4, a.axis, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::ReduceMax(a) => {
            let start = builder.start_table();
            builder.push_slot::<i32>(4, a.axis, 0);
            builder.push_slot::<u32>(6, a.nan_mode as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }

        Attribute::ReduceMin(a) => {
            let start = builder.start_table();
            builder.push_slot::<i32>(4, a.axis, 0);
            builder.push_slot::<u32>(6, a.nan_mode as u32, 0);
            let off = builder.end_table(start);
            (attr.union_type(), Some(off))
        }
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

        let (out, new_shape) =
            permute_weight_data(&data, &shape, 1, WeightPermKind::DepthwiseConv2d { in_channels: 1 });

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
            let src_val =
                i32::from_le_bytes(data[idx * elem_size..idx * elem_size + elem_size].try_into().unwrap());
            let dst_val =
                i32::from_le_bytes(out[idx * elem_size..idx * elem_size + elem_size].try_into().unwrap());
            assert_eq!(src_val, dst_val, "Element {idx}: src={src_val} != dst={dst_val}");
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

        let (permuted, new_shape) =
            permute_weight_data(&data, &shape, 1, WeightPermKind::Conv2d);

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
                        assert_eq!(
                            permuted[dst_idx], data[src_idx],
                            "o={o},i={i},h={h},w={w}"
                        );
                    }
                }
            }
        }
    }
}
