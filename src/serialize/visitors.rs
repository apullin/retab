// Per-op TOSA emission visitors.
//
// Each function takes the graph node, its metadata, and the tensor store, and
// returns the TOSA operator(s) + tensor(s) that should be added to the basic
// block being built.

use crate::graph::ir::{Arg, Node, NodeId};
use crate::graph::Graph;
use crate::store::TensorStore;

use super::tosa_generated::{
    AbsAttribute, AddAttribute, Attribute, AvgPool2dAttribute, CastAttribute, CeilAttribute,
    ClampAttribute, ConcatAttribute, ConstAttribute, ConstShapeAttribute, Conv2dAttribute,
    CosAttribute, DType, DepthwiseConv2dAttribute, EqualAttribute, ErfAttribute, ExpAttribute,
    FloorAttribute, GatherAttribute, GreaterAttribute, GreaterEqualAttribute, LogAttribute,
    MatMulAttribute, MaxPool2dAttribute, MulAttribute, NanPropagationMode, NegateAttribute, Op,
    PadAttribute, ReciprocalAttribute, ReduceSumAttribute, RescaleAttribute, ReshapeAttribute,
    RoundingMode, RsqrtAttribute, SelectAttribute, SigmoidAttribute, SinAttribute, SliceAttribute,
    SubAttribute, TableAttribute, TanhAttribute, TosaOperator, TosaShape, TosaTensor,
    TransposeAttribute, TransposeConv2dAttribute,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert the IR DType to the TOSA flatbuffer DType.
pub fn ir_dtype_to_tosa(dt: crate::graph::ir::DType) -> DType {
    use crate::graph::ir::DType as IrDtype;
    match dt {
        IrDtype::Float32 => DType::Fp32,
        IrDtype::Float16 => DType::Fp16,
        IrDtype::BFloat16 => DType::Bf16,
        IrDtype::Int8 => DType::Int8,
        IrDtype::Int16 => DType::Int16,
        IrDtype::Int32 => DType::Int32,
        IrDtype::Int64 => DType::Int64,
        IrDtype::Bool => DType::Bool,
        IrDtype::Int4 => DType::Int4,
        IrDtype::Int48 => DType::Int48,
    }
}

/// Result of visiting a single graph node: zero or more TOSA operators and
/// tensors (including const data tensors and shape entries) to append to the
/// basic block.
#[derive(Debug, Default)]
pub struct EmitResult {
    pub operators: Vec<TosaOperator>,
    pub tensors: Vec<TosaTensor>,
    pub shapes: Vec<TosaShape>,
}

/// Get the name of a node referenced by an Arg::Node.
fn resolve_input_name(graph: &Graph, arg: &Arg) -> Option<String> {
    match arg {
        Arg::Node(id) => graph.node(*id).map(|n| n.name.clone()),
        _ => None,
    }
}

/// Collect input tensor names from node args (only Arg::Node entries).
fn collect_input_names(graph: &Graph, node: &Node) -> Vec<String> {
    node.args
        .iter()
        .filter_map(|a| resolve_input_name(graph, a))
        .collect()
}

/// Build the output tensor entry for a node.
fn make_output_tensor(node: &Node) -> TosaTensor {
    let shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();
    let dtype = ir_dtype_to_tosa(node.meta.val.dtype);
    TosaTensor {
        name: node.name.clone(),
        shape,
        dtype,
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    }
}

/// Extract an i32 list from a kwarg by name.
fn kwarg_i32_list(node: &Node, name: &str) -> Vec<i32> {
    for (k, v) in &node.kwargs {
        if k == name {
            if let Arg::IntList(list) = v {
                return list.iter().map(|&x| x as i32).collect();
            }
        }
    }
    Vec::new()
}

/// Extract an i32 list from an arg at the given index.
fn arg_i32_list(node: &Node, idx: usize) -> Vec<i32> {
    match node.args.get(idx) {
        Some(Arg::IntList(list)) => list.iter().map(|&x| x as i32).collect(),
        _ => Vec::new(),
    }
}

/// Extract a single i32 kwarg by name.
fn kwarg_i32(node: &Node, name: &str, default: i32) -> i32 {
    for (k, v) in &node.kwargs {
        if k == name {
            if let Arg::Int(i) = v {
                return *i as i32;
            }
        }
    }
    default
}

/// Extract a bool kwarg.
fn kwarg_bool(node: &Node, name: &str, default: bool) -> bool {
    for (k, v) in &node.kwargs {
        if k == name {
            if let Arg::Bool(b) = v {
                return *b;
            }
        }
    }
    default
}

// ---------------------------------------------------------------------------
// Per-op visitors
// ---------------------------------------------------------------------------

/// Emit a CONST operator.
pub fn emit_const(
    _graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let name = &node.name;
    let shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();
    let dtype = ir_dtype_to_tosa(node.meta.val.dtype);

    let data = store
        .get(name)
        .map(|td| td.data.clone())
        .unwrap_or_default();

    result.tensors.push(TosaTensor {
        name: name.clone(),
        shape,
        dtype,
        data,
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });

    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![name.clone()],
        location: None,
    });

    result
}

/// Emit a CONV2D operator.
pub fn emit_conv2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    // Extract attribute from kwargs or use defaults
    let pad = kwarg_i32_list(node, "pad");
    let stride = kwarg_i32_list(node, "stride");
    let dilation = kwarg_i32_list(node, "dilation");
    let local_bound = kwarg_bool(node, "local_bound", false);

    // Determine accumulator type — default to Int32 for quantized
    let acc_type = DType::Int32;

    result.operators.push(TosaOperator {
        op: Op::Conv2d,
        attribute: Attribute::Conv2d(Conv2dAttribute {
            pad: if pad.is_empty() {
                vec![0, 0, 0, 0]
            } else {
                pad
            },
            stride: if stride.is_empty() {
                vec![1, 1]
            } else {
                stride
            },
            dilation: if dilation.is_empty() {
                vec![1, 1]
            } else {
                dilation
            },
            local_bound,
            acc_type,
        }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a DEPTHWISE_CONV2D operator.
pub fn emit_depthwise_conv2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let pad = kwarg_i32_list(node, "pad");
    let stride = kwarg_i32_list(node, "stride");
    let dilation = kwarg_i32_list(node, "dilation");
    let local_bound = kwarg_bool(node, "local_bound", false);
    let acc_type = DType::Int32;

    result.operators.push(TosaOperator {
        op: Op::DepthwiseConv2d,
        attribute: Attribute::DepthwiseConv2d(DepthwiseConv2dAttribute {
            pad: if pad.is_empty() {
                vec![0, 0, 0, 0]
            } else {
                pad
            },
            stride: if stride.is_empty() {
                vec![1, 1]
            } else {
                stride
            },
            dilation: if dilation.is_empty() {
                vec![1, 1]
            } else {
                dilation
            },
            local_bound,
            acc_type,
        }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a RESCALE operator.
///
/// TOSA RESCALE takes 5 inputs: [input, multiplier, shift, input_zp, output_zp].
/// The multiplier and shift come as Node args; input_zp and output_zp come as kwargs.
pub fn emit_rescale(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let scale32 = kwarg_bool(node, "scale32", true);
    let per_channel = kwarg_bool(node, "per_channel", false);
    let input_unsigned = kwarg_bool(node, "input_unsigned", false);
    let output_unsigned = kwarg_bool(node, "output_unsigned", false);
    let input_zp_val = kwarg_i32(node, "input_zp", 0);
    let output_zp_val = kwarg_i32(node, "output_zp", 0);

    // Create input_zp const tensor (scalar int32)
    let input_zp_name = format!("{}_input_zp", node.name);
    result.tensors.push(TosaTensor {
        name: input_zp_name.clone(),
        shape: vec![1],
        dtype: DType::Int32,
        data: input_zp_val.to_le_bytes().to_vec(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![input_zp_name.clone()],
        location: None,
    });

    // Create output_zp const tensor (scalar int32)
    let output_zp_name = format!("{}_output_zp", node.name);
    result.tensors.push(TosaTensor {
        name: output_zp_name.clone(),
        shape: vec![1],
        dtype: DType::Int32,
        data: output_zp_val.to_le_bytes().to_vec(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![output_zp_name.clone()],
        location: None,
    });

    // RESCALE inputs: [input, multiplier, shift, input_zp, output_zp]
    let mut rescale_inputs = input_names;
    rescale_inputs.push(input_zp_name);
    rescale_inputs.push(output_zp_name);

    result.operators.push(TosaOperator {
        op: Op::Rescale,
        attribute: Attribute::Rescale(RescaleAttribute {
            scale32,
            rounding_mode: RoundingMode::SingleRound,
            per_channel,
            input_unsigned,
            output_unsigned,
        }),
        inputs: rescale_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a MAX_POOL2D operator.
pub fn emit_max_pool2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let kernel = kwarg_i32_list(node, "kernel");
    let stride = kwarg_i32_list(node, "stride");
    let pad = kwarg_i32_list(node, "pad");

    result.operators.push(TosaOperator {
        op: Op::MaxPool2d,
        attribute: Attribute::MaxPool2d(MaxPool2dAttribute {
            kernel: if kernel.is_empty() {
                vec![2, 2]
            } else {
                kernel
            },
            stride: if stride.is_empty() {
                vec![2, 2]
            } else {
                stride
            },
            pad: if pad.is_empty() {
                vec![0, 0, 0, 0]
            } else {
                pad
            },
            nan_mode: NanPropagationMode::Propagate,
        }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a TRANSPOSE operator.
pub fn emit_transpose(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let perms = arg_i32_list(node, 1);

    result.operators.push(TosaOperator {
        op: Op::Transpose,
        attribute: Attribute::Transpose(TransposeAttribute {
            perms: if perms.is_empty() {
                kwarg_i32_list(node, "perms")
            } else {
                perms
            },
        }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a RESHAPE operator.
///
/// TOSA RESHAPE takes 2 inputs: [input, shape]. We create a CONST_SHAPE
/// tensor from the output shape.
pub fn emit_reshape(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    // Create a CONST_SHAPE tensor for the target shape
    let out_shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();
    let rank = out_shape.len() as u32;
    let shape_name = format!("{}_shape", node.name);

    // Encode shape as little-endian i64 bytes (TOSA Shape dtype)
    let shape_data: Vec<u8> = out_shape
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();

    result.shapes.push(TosaShape {
        name: shape_name.clone(),
        rank,
        data: shape_data,
    });

    result.operators.push(TosaOperator {
        op: Op::ConstShape,
        attribute: Attribute::ConstShape(ConstShapeAttribute),
        inputs: vec![],
        outputs: vec![shape_name.clone()],
        location: None,
    });

    // RESHAPE inputs: [input, shape]
    let mut reshape_inputs = input_names;
    reshape_inputs.push(shape_name);

    result.operators.push(TosaOperator {
        op: Op::Reshape,
        attribute: Attribute::Reshape(ReshapeAttribute),
        inputs: reshape_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a CONST_SHAPE operator.
pub fn emit_const_shape(
    _graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let name = &node.name;
    let shape_data = store
        .get(name)
        .map(|td| td.data.clone())
        .unwrap_or_default();

    // Shape rank from metadata
    let rank = node.meta.val.shape_len as u32;

    result.shapes.push(TosaShape {
        name: name.clone(),
        rank,
        data: shape_data,
    });

    result.operators.push(TosaOperator {
        op: Op::ConstShape,
        attribute: Attribute::ConstShape(ConstShapeAttribute),
        inputs: vec![],
        outputs: vec![name.clone()],
        location: None,
    });

    result
}

/// Emit a CLAMP operator.
pub fn emit_clamp(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    // Extract min/max bytes from kwargs
    let mut min_val = Vec::new();
    let mut max_val = Vec::new();
    for (k, v) in &node.kwargs {
        if k == "min_val" {
            if let Arg::IntList(list) = v {
                min_val = list.iter().flat_map(|&x| (x as i8).to_le_bytes()).collect();
            } else if let Arg::Int(i) = v {
                min_val = (*i as i8).to_le_bytes().to_vec();
            }
        }
        if k == "max_val" {
            if let Arg::IntList(list) = v {
                max_val = list.iter().flat_map(|&x| (x as i8).to_le_bytes()).collect();
            } else if let Arg::Int(i) = v {
                max_val = (*i as i8).to_le_bytes().to_vec();
            }
        }
    }

    result.operators.push(TosaOperator {
        op: Op::Clamp,
        attribute: Attribute::Clamp(ClampAttribute {
            min_val,
            max_val,
            nan_mode: NanPropagationMode::Propagate,
        }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit an ADD operator.
pub fn emit_add(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    result.operators.push(TosaOperator {
        op: Op::Add,
        attribute: Attribute::Add(AddAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a SUB operator.
pub fn emit_sub(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    result.operators.push(TosaOperator {
        op: Op::Sub,
        attribute: Attribute::Sub(SubAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a MUL operator.
///
/// TOSA MUL takes 3 inputs: lhs, rhs, and a shift tensor (scalar int8, typically 0).
/// We create an inline const tensor for the shift value.
pub fn emit_mul(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    // Create the shift const tensor (scalar int8 = 0)
    let shift_name = format!("{}_shift", node.name);
    result.tensors.push(TosaTensor {
        name: shift_name.clone(),
        shape: vec![1],
        dtype: DType::Int8,
        data: vec![0u8],
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![shift_name.clone()],
        location: None,
    });

    // Build the MUL inputs: [lhs, rhs, shift]
    let mut mul_inputs = input_names;
    mul_inputs.push(shift_name);

    result.operators.push(TosaOperator {
        op: Op::Mul,
        attribute: Attribute::Mul(MulAttribute { shift: 0 }),
        inputs: mul_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit an AVG_POOL2D operator.
///
/// TOSA AVG_POOL2D takes 3 inputs: input, input_zp, output_zp.
/// We create inline const tensors for the zero-point scalars.
pub fn emit_avg_pool2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let kernel = kwarg_i32_list(node, "kernel");
    let stride = kwarg_i32_list(node, "stride");
    let pad = kwarg_i32_list(node, "pad");
    let acc_type = DType::Int32;
    let input_zp = kwarg_i32(node, "input_zp", 0);
    let output_zp = kwarg_i32(node, "output_zp", 0);

    // Create input_zp const tensor (scalar int32)
    let input_zp_name = format!("{}_input_zp", node.name);
    result.tensors.push(TosaTensor {
        name: input_zp_name.clone(),
        shape: vec![1],
        dtype: DType::Int32,
        data: (input_zp).to_le_bytes().to_vec(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![input_zp_name.clone()],
        location: None,
    });

    // Create output_zp const tensor (scalar int32)
    let output_zp_name = format!("{}_output_zp", node.name);
    result.tensors.push(TosaTensor {
        name: output_zp_name.clone(),
        shape: vec![1],
        dtype: DType::Int32,
        data: (output_zp).to_le_bytes().to_vec(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![output_zp_name.clone()],
        location: None,
    });

    // Build AVG_POOL2D inputs: [input, input_zp, output_zp]
    let mut pool_inputs = input_names;
    pool_inputs.push(input_zp_name);
    pool_inputs.push(output_zp_name);

    result.operators.push(TosaOperator {
        op: Op::AvgPool2d,
        attribute: Attribute::AvgPool2d(AvgPool2dAttribute {
            kernel: if kernel.is_empty() {
                vec![2, 2]
            } else {
                kernel
            },
            stride: if stride.is_empty() {
                vec![2, 2]
            } else {
                stride
            },
            pad: if pad.is_empty() {
                vec![0, 0, 0, 0]
            } else {
                pad
            },
            acc_type,
            input_zp,
            output_zp,
        }),
        inputs: pool_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a REDUCE_SUM operator.
///
/// TOSA REDUCE_SUM reduces one axis at a time. The axis is stored
/// in the attribute; the decomposition pass ensures single-axis reductions.
pub fn emit_reduce_sum(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let axis = kwarg_i32(node, "axis", 0);

    result.operators.push(TosaOperator {
        op: Op::ReduceSum,
        attribute: Attribute::ReduceSum(ReduceSumAttribute { axis }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a TABLE operator.
///
/// TOSA TABLE takes 2 inputs: input tensor and table tensor.
/// The table is a const tensor (typically 513 int16 values for int8 input)
/// pre-computed by InsertTableOpsPass.
pub fn emit_table(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    result.operators.push(TosaOperator {
        op: Op::Table,
        attribute: Attribute::Table(TableAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

// ---------------------------------------------------------------------------
// Simple unary/binary visitors (empty attribute)
// ---------------------------------------------------------------------------

/// Emit a SIGMOID operator.
pub fn emit_sigmoid(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Sigmoid,
        attribute: Attribute::Sigmoid(SigmoidAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a TANH operator.
pub fn emit_tanh(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Tanh,
        attribute: Attribute::Tanh(TanhAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit an EXP operator.
pub fn emit_exp(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Exp,
        attribute: Attribute::Exp(ExpAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a LOG operator.
pub fn emit_log(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Log,
        attribute: Attribute::Log(LogAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a NEGATE operator.
pub fn emit_negate(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Negate,
        attribute: Attribute::Negate(NegateAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit an ABS operator.
pub fn emit_abs(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Abs,
        attribute: Attribute::Abs(AbsAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a CEIL operator.
pub fn emit_ceil(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Ceil,
        attribute: Attribute::Ceil(CeilAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a FLOOR operator.
pub fn emit_floor(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Floor,
        attribute: Attribute::Floor(FloorAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a RECIPROCAL operator.
pub fn emit_reciprocal(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Reciprocal,
        attribute: Attribute::Reciprocal(ReciprocalAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit an ERF operator.
pub fn emit_erf(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Erf,
        attribute: Attribute::Erf(ErfAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a COS operator.
pub fn emit_cos(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Cos,
        attribute: Attribute::Cos(CosAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a SIN operator.
pub fn emit_sin(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Sin,
        attribute: Attribute::Sin(SinAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit an RSQRT operator.
pub fn emit_rsqrt(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Rsqrt,
        attribute: Attribute::Rsqrt(RsqrtAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a SELECT operator (ternary: condition, on_true, on_false).
pub fn emit_select(
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

/// Emit an EQUAL operator.
pub fn emit_equal(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Equal,
        attribute: Attribute::Equal(EqualAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a GREATER operator.
pub fn emit_greater(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Greater,
        attribute: Attribute::Greater(GreaterAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a GREATER_EQUAL operator.
pub fn emit_greater_equal(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::GreaterEqual,
        attribute: Attribute::GreaterEqual(GreaterEqualAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a CAST operator.
pub fn emit_cast(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator {
        op: Op::Cast,
        attribute: Attribute::Cast(CastAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

/// Emit a GATHER operator.
pub fn emit_gather(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
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

// ---------------------------------------------------------------------------
// Parameterised visitors
// ---------------------------------------------------------------------------

/// Emit a MATMUL operator.
///
/// TOSA MATMUL takes 4 inputs: [A, B, A_zp, B_zp].
/// We create inline const tensors for the zero-point scalars.
pub fn emit_matmul(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let a_zp = kwarg_i32(node, "A_zp", 0);
    let b_zp = kwarg_i32(node, "B_zp", 0);

    // Create A_zp const tensor
    let a_zp_name = format!("{}_A_zp", node.name);
    result.tensors.push(TosaTensor {
        name: a_zp_name.clone(),
        shape: vec![1],
        dtype: DType::Int32,
        data: a_zp.to_le_bytes().to_vec(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![a_zp_name.clone()],
        location: None,
    });

    // Create B_zp const tensor
    let b_zp_name = format!("{}_B_zp", node.name);
    result.tensors.push(TosaTensor {
        name: b_zp_name.clone(),
        shape: vec![1],
        dtype: DType::Int32,
        data: b_zp.to_le_bytes().to_vec(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![b_zp_name.clone()],
        location: None,
    });

    // MATMUL inputs: [A, B, A_zp, B_zp]
    let mut matmul_inputs = input_names;
    matmul_inputs.push(a_zp_name);
    matmul_inputs.push(b_zp_name);

    result.operators.push(TosaOperator {
        op: Op::Matmul,
        attribute: Attribute::MatMul(MatMulAttribute),
        inputs: matmul_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a CONCAT operator.
pub fn emit_concat(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let axis = kwarg_i32(node, "axis", 0);

    result.operators.push(TosaOperator {
        op: Op::Concat,
        attribute: Attribute::Concat(ConcatAttribute { axis }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a PAD operator.
///
/// TOSA PAD takes 2 inputs: [input, padding_shape].
/// The padding shape is a CONST_SHAPE tensor.
pub fn emit_pad(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    result.operators.push(TosaOperator {
        op: Op::Pad,
        attribute: Attribute::Pad(PadAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a SLICE operator.
///
/// TOSA SLICE takes 3 inputs: [input, start_shape, size_shape].
pub fn emit_slice(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    result.operators.push(TosaOperator {
        op: Op::Slice,
        attribute: Attribute::Slice(SliceAttribute),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a TRANSPOSE_CONV2D operator.
pub fn emit_transpose_conv2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));

    let out_pad = kwarg_i32_list(node, "out_pad");
    let stride = kwarg_i32_list(node, "stride");
    let local_bound = kwarg_bool(node, "local_bound", false);
    let acc_type = DType::Int32;

    result.operators.push(TosaOperator {
        op: Op::TransposeConv2d,
        attribute: Attribute::TransposeConv2d(TransposeConv2dAttribute {
            out_pad: if out_pad.is_empty() {
                vec![0, 0, 0, 0]
            } else {
                out_pad
            },
            stride: if stride.is_empty() {
                vec![1, 1]
            } else {
                stride
            },
            local_bound,
            acc_type,
        }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

use crate::graph::ir::OpTarget;

/// Dispatch to the correct per-op visitor based on the node's target.
pub fn emit_node(
    graph: &Graph,
    node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> Option<EmitResult> {
    match &node.target {
        OpTarget::TosaConst => Some(emit_const(graph, node_id, node, store)),
        OpTarget::TosaConv2d => Some(emit_conv2d(graph, node_id, node, store)),
        OpTarget::TosaDepthwiseConv2d => {
            Some(emit_depthwise_conv2d(graph, node_id, node, store))
        }
        OpTarget::TosaRescale => Some(emit_rescale(graph, node_id, node, store)),
        OpTarget::TosaMaxPool2d => Some(emit_max_pool2d(graph, node_id, node, store)),
        OpTarget::TosaTranspose => Some(emit_transpose(graph, node_id, node, store)),
        OpTarget::TosaReshape => Some(emit_reshape(graph, node_id, node, store)),
        OpTarget::TosaClamp => Some(emit_clamp(graph, node_id, node, store)),
        OpTarget::TosaAdd => Some(emit_add(graph, node_id, node, store)),
        OpTarget::TosaSub => Some(emit_sub(graph, node_id, node, store)),
        OpTarget::TosaMul => Some(emit_mul(graph, node_id, node, store)),
        OpTarget::TosaAvgPool2d => Some(emit_avg_pool2d(graph, node_id, node, store)),
        OpTarget::TosaReduceSum => Some(emit_reduce_sum(graph, node_id, node, store)),
        OpTarget::TosaTable => Some(emit_table(graph, node_id, node, store)),
        OpTarget::TosaSigmoid => Some(emit_sigmoid(graph, node_id, node, store)),
        OpTarget::TosaTanh => Some(emit_tanh(graph, node_id, node, store)),
        OpTarget::TosaExp => Some(emit_exp(graph, node_id, node, store)),
        OpTarget::TosaLog => Some(emit_log(graph, node_id, node, store)),
        OpTarget::TosaNeg | OpTarget::TosaNegate => {
            Some(emit_negate(graph, node_id, node, store))
        }
        OpTarget::TosaAbs => Some(emit_abs(graph, node_id, node, store)),
        OpTarget::TosaCeil => Some(emit_ceil(graph, node_id, node, store)),
        OpTarget::TosaFloor => Some(emit_floor(graph, node_id, node, store)),
        OpTarget::TosaReciprocal => Some(emit_reciprocal(graph, node_id, node, store)),
        OpTarget::TosaErf => Some(emit_erf(graph, node_id, node, store)),
        OpTarget::TosaCos => Some(emit_cos(graph, node_id, node, store)),
        OpTarget::TosaSin => Some(emit_sin(graph, node_id, node, store)),
        OpTarget::TosaRsqrt => Some(emit_rsqrt(graph, node_id, node, store)),
        OpTarget::TosaMatmul => Some(emit_matmul(graph, node_id, node, store)),
        OpTarget::TosaConcat => Some(emit_concat(graph, node_id, node, store)),
        OpTarget::TosaPad => Some(emit_pad(graph, node_id, node, store)),
        OpTarget::TosaSlice => Some(emit_slice(graph, node_id, node, store)),
        OpTarget::TosaGather => Some(emit_gather(graph, node_id, node, store)),
        OpTarget::TosaSelect => Some(emit_select(graph, node_id, node, store)),
        OpTarget::TosaCast => Some(emit_cast(graph, node_id, node, store)),
        OpTarget::TosaEqual => Some(emit_equal(graph, node_id, node, store)),
        OpTarget::TosaGreater => Some(emit_greater(graph, node_id, node, store)),
        OpTarget::TosaGreaterEqual => Some(emit_greater_equal(graph, node_id, node, store)),
        OpTarget::TosaTransposeConv2d => {
            Some(emit_transpose_conv2d(graph, node_id, node, store))
        }
        // Ops not yet handled — skip silently for now.
        _ => None,
    }
}
