pub mod tosa_generated;
pub mod visitors;

use crate::graph::ir::{Arg, OpTarget, OpType};
use crate::graph::Graph;
use crate::store::TensorStore;

use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, WIPOffset};
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
    /// where IC = in_channels/groups = 1 (per group), M = depth_multiplier
    DepthwiseConv2d,
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
        WeightPermKind::DepthwiseConv2d => {
            // OIHW -> HWIM: perm [2, 3, 1, 0]
            // TOSA depthwise weight: [KH, KW, IC_per_group, depth_multiplier]
            // PyTorch weight: [OC, IC/groups, KH, KW] where IC/groups=1 for depthwise
            let new_shape = vec![shape[2], shape[3], shape[1], shape[0]];
            let (_nh, nw, ni, nm) = (d2, d3, d1, d0);
            let mut out = vec![0u8; data.len()];

            for o in 0..d0 {
                for i in 0..d1 {
                    for h in 0..d2 {
                        for w in 0..d3 {
                            let src_idx = ((o * d1 + i) * d2 + h) * d3 + w;
                            let dst_idx = ((h * nw + w) * ni + i) * nm + o;
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
    }
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
                Some(WeightPermKind::DepthwiseConv2d)
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
pub fn serialize_to_tosa(graph: &Graph, store: &TensorStore) -> Vec<u8> {
    eprintln!("[Serialize] starting topo sort on {} nodes...", graph.order.len());
    let sorted = graph.topological_sort();
    eprintln!("[Serialize] topo sort done, {} sorted nodes", sorted.len());

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
                let shape: Vec<i32> = node
                    .meta
                    .val
                    .shape_slice()
                    .iter()
                    .map(|&d| d as i32)
                    .collect();
                let dtype = ir_dtype_to_tosa(node.meta.val.dtype);

                if let Some(tensor_data) = store.get(name) {
                    // Check if this weight tensor needs OIHW -> OHWI permutation
                    let (final_data, final_shape) =
                        if let Some(&perm_kind) = weight_perms.get(name) {
                            let elem_sz = dtype_elem_size(node.meta.val.dtype);
                            let (pdata, pshape) = permute_weight_data(
                                &tensor_data.data,
                                node.meta.val.shape_slice(),
                                elem_sz,
                                perm_kind,
                            );
                            let pshape_i32: Vec<i32> =
                                pshape.iter().map(|&d| d as i32).collect();
                            (pdata, pshape_i32)
                        } else {
                            (tensor_data.data.clone(), shape)
                        };

                    tensors.push(TosaTensor {
                        name: name.clone(),
                        shape: final_shape,
                        dtype,
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
                    let shape: Vec<i32> = node
                        .meta
                        .val
                        .shape_slice()
                        .iter()
                        .map(|&d| d as i32)
                        .collect();
                    let dtype = ir_dtype_to_tosa(node.meta.val.dtype);
                    tensors.push(TosaTensor {
                        name: node.name.clone(),
                        shape,
                        dtype,
                        data: Vec::new(),
                        variable: false,
                        is_unranked: false,
                        variable_name: String::new(),
                    });
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

    eprintln!("[Serialize] topo sort done, collected {} tensors, {} operators, {} shapes", tensors.len(), operators.len(), shapes.len());

    // Build the flatbuffer bottom-up
    let mut builder = FlatBufferBuilder::with_capacity(2 * 1024 * 1024); // 2MB initial

    // Pre-serialize all sub-tables
    eprintln!("[Serialize] building fb tensors...");
    let fb_tensors = build_fb_tensors(&mut builder, &tensors);
    eprintln!("[Serialize] building fb shapes...");
    let fb_shapes = build_fb_shapes(&mut builder, &shapes);
    eprintln!("[Serialize] building fb operators...");
    let fb_operators = build_fb_operators(&mut builder, &operators);
    eprintln!("[Serialize] assembling graph...");

    // Build input/output name vectors
    let fb_input_strs: Vec<_> = input_names
        .iter()
        .map(|s| builder.create_string(s))
        .collect();
    let fb_output_strs: Vec<_> = output_names
        .iter()
        .map(|s| builder.create_string(s))
        .collect();
    let fb_inputs = builder.create_vector(&fb_input_strs);
    let fb_outputs = builder.create_vector(&fb_output_strs);

    let fb_tensors_vec = builder.create_vector(&fb_tensors);
    let fb_shapes_vec = builder.create_vector(&fb_shapes);
    let fb_operators_vec = builder.create_vector(&fb_operators);

    // Build basic block
    let block_name = builder.create_string("main");
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
    let region_name = builder.create_string("main");
    let region = {
        let start = builder.start_table();
        builder.push_slot_always::<WIPOffset<_>>(4, region_name);
        builder.push_slot_always::<WIPOffset<_>>(6, blocks_vec);
        builder.end_table(start)
    };

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
    let regions_vec = builder.create_vector(&[region]);
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

// ============================================================================
// Flatbuffer builder helpers
// ============================================================================

fn build_fb_tensors(
    builder: &mut FlatBufferBuilder<'_>,
    tensors: &[TosaTensor],
) -> Vec<WIPOffset<TableFinishedWIPOffset>> {
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        let name = builder.create_string(&t.name);
        let shape = builder.create_vector(&t.shape);
        let data = if !t.data.is_empty() {
            Some(builder.create_vector(&t.data))
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
            Some(builder.create_vector(&s.data))
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
        | Attribute::Select(_)
        | Attribute::Cast(_)
        | Attribute::Equal(_)
        | Attribute::Greater(_)
        | Attribute::GreaterEqual(_) => {
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
            let min_val = builder.create_vector(&a.min_val);
            let max_val = builder.create_vector(&a.max_val);
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
            let kernel = builder.create_vector(&a.kernel);
            let stride = builder.create_vector(&a.stride);
            let pad = builder.create_vector(&a.pad);
            let start = builder.start_table();
            builder.push_slot_always::<WIPOffset<_>>(4, kernel);
            builder.push_slot_always::<WIPOffset<_>>(6, stride);
            builder.push_slot_always::<WIPOffset<_>>(8, pad);
            builder.push_slot::<u32>(10, a.acc_type as u32, 0);
            builder.push_slot::<i32>(12, a.input_zp, 0);
            builder.push_slot::<i32>(14, a.output_zp, 0);
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
    }
}
