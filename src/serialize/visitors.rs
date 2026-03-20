// Per-op TOSA emission visitors.
//
// Each function takes the graph node, its metadata, and the tensor store, and
// returns the TOSA operator(s) + tensor(s) that should be added to the basic
// block being built.

use crate::graph::ir::{Arg, Node, NodeId, OpTarget};
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

fn node_output_dtype(node: &Node) -> DType {
    if let Some((_, qp)) = node.meta.output_qparams.first() {
        return ir_dtype_to_tosa(qp.dtype);
    }
    ir_dtype_to_tosa(node.meta.val.dtype)
}

fn matmul_input_dtype(graph: &Graph, node: &Node, index: usize) -> DType {
    match node.args.get(index) {
        Some(Arg::Node(id)) => graph
            .node(*id)
            .map(|input_node| ir_dtype_to_tosa(input_node.meta.val.dtype))
            .unwrap_or(DType::Int32),
        _ => DType::Int32,
    }
}

fn normalize_rescale_zp(dtype: DType, zp: i32, is_unsigned: bool) -> i32 {
    match dtype {
        DType::Int8 => zp,
        DType::Int16 if is_unsigned => zp,
        _ => 0,
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
        Arg::Node(id) => {
            let node = graph.node(*id)?;
            if is_getitem_node(node) {
                let source_id = match node.args.first() {
                    Some(Arg::Node(source_id)) => *source_id,
                    _ => return Some(node.name.clone()),
                };
                let source = graph.node(source_id)?;
                let output_idx = match node.args.get(1) {
                    Some(Arg::Int(idx)) => *idx,
                    _ => 0,
                };
                if output_idx == 0 {
                    return Some(primary_output_name(graph, source));
                }
            }
            Some(node.name.clone())
        }
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

fn is_getitem_node(node: &Node) -> bool {
    matches!(&node.target, OpTarget::Other(target) if target.contains("getitem"))
}

/// Match the Python serializer's multi-output convention: when an op's users
/// are all `getitem` nodes, emit the first real tensor result under the
/// `getitem(..., 0)` node name instead of the tuple-producing node name.
fn primary_output_name(graph: &Graph, node: &Node) -> String {
    if node.users.is_empty() {
        return node.name.clone();
    }

    let mut all_getitems = true;
    let mut first_output_alias: Option<String> = None;
    let mut fallback_alias: Option<String> = None;

    for user_id in &node.users {
        let user = match graph.node(*user_id) {
            Some(user) => user,
            None => {
                all_getitems = false;
                break;
            }
        };
        if !is_getitem_node(user) {
            all_getitems = false;
            break;
        }
        if fallback_alias.is_none() {
            fallback_alias = Some(user.name.clone());
        }
        if matches!(user.args.get(1), Some(Arg::Int(0))) && first_output_alias.is_none() {
            first_output_alias = Some(user.name.clone());
        }
    }

    if all_getitems {
        first_output_alias
            .or(fallback_alias)
            .unwrap_or_else(|| node.name.clone())
    } else {
        node.name.clone()
    }
}

fn literal_output_shape(node: &Node) -> Option<Vec<i32>> {
    match node.args.get(1) {
        Some(Arg::IntList(shape)) => Some(shape.iter().map(|&d| d as i32).collect()),
        _ => None,
    }
}

fn apply_tosa_dim_order(node: &Node, shape: Vec<i32>) -> Vec<i32> {
    if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        if order.len() == shape.len() {
            let reordered: Vec<i32> = order
                .iter()
                .filter_map(|&idx| shape.get(idx as usize).copied())
                .collect();
            if reordered.len() == shape.len() {
                return reordered;
            }
        }
    }
    shape
}

fn map_axis_to_tosa_dim_order(node: &Node, axis: i32) -> i32 {
    let rank = node.meta.val.shape_len as i32;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if normalized < 0 || normalized >= rank {
        return axis;
    }

    if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        if let Some(pos) = order.iter().position(|&dim| dim as i32 == normalized) {
            return pos as i32;
        }
    }

    normalized
}

/// Best-effort reconstruction of the actual TOSA tensor shape for a node.
///
/// This differs from `meta.val.shape` in two important cases:
/// 1. NHWC-native op outputs still carry PyTorch's logical NCHW shape.
/// 2. FX-inserted TRANSPOSE/RESHAPE nodes may already have a physical TOSA
///    layout that should be derived from their input tensor, not reused from
///    stale metadata.
fn actual_tensor_shape(graph: &Graph, node: &Node) -> Vec<i32> {
    if node.target == OpTarget::TosaReshape {
        if let Some(shape) = literal_output_shape(node) {
            return shape;
        }
    }

    if node.target == OpTarget::TosaRescale {
        if let Some(Arg::Node(input_id)) = node.args.first() {
            if let Some(input_node) = graph.node(*input_id) {
                return actual_tensor_shape(graph, input_node);
            }
        }
    }

    if matches!(node.target, OpTarget::TosaTranspose | OpTarget::Permute) {
        let perms = {
            let perms = arg_i32_list(node, 1);
            if perms.is_empty() {
                kwarg_i32_list(node, "perms")
            } else {
                perms
            }
        };
        if let Some(Arg::Node(input_id)) = node.args.first() {
            if let Some(input_node) = graph.node(*input_id) {
                let input_shape = actual_tensor_shape(graph, input_node);
                if input_shape.len() == perms.len() {
                    let output_shape: Vec<i32> = perms
                        .iter()
                        .filter_map(|&idx| input_shape.get(idx as usize).copied())
                        .collect();
                    if output_shape.len() == input_shape.len() {
                        return output_shape;
                    }
                }
            }
        }
    }

    let raw_shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();

    let reordered_shape = apply_tosa_dim_order(node, raw_shape.clone());

    let logical_nhwc = raw_shape.len() == 4
        && (is_getitem_node(node)
            || matches!(
                node.target,
                OpTarget::Conv2d
                    | OpTarget::DepthwiseConv2d
                    | OpTarget::TosaConv2d
                    | OpTarget::TosaDepthwiseConv2d
                    | OpTarget::MaxPool2d
                    | OpTarget::AvgPool2d
                    | OpTarget::TosaMaxPool2d
                    | OpTarget::TosaAvgPool2d
            ));

    if logical_nhwc && reordered_shape == raw_shape {
        vec![raw_shape[0], raw_shape[2], raw_shape[3], raw_shape[1]]
    } else {
        reordered_shape
    }
}

fn maybe_materialize_binary_broadcast(
    graph: &Graph,
    node: &Node,
    result: &mut EmitResult,
    input_names: &mut [String],
) {
    if input_names.len() < 2 {
        return;
    }

    let input_nodes: Vec<&Node> = node
        .args
        .iter()
        .filter_map(|arg| match arg {
            Arg::Node(id) => graph.node(*id),
            _ => None,
        })
        .collect();
    if input_nodes.len() < 2 {
        return;
    }

    let input_shapes = [
        actual_tensor_shape(graph, input_nodes[0]),
        actual_tensor_shape(graph, input_nodes[1]),
    ];

    let (smaller_idx, target_shape) = if input_shapes[0].len() < input_shapes[1].len() {
        let target_shape = broadcast_shape_for_binary_inputs(&input_shapes[0], &input_shapes[1]);
        (0usize, target_shape)
    } else if input_shapes[1].len() < input_shapes[0].len() {
        let target_shape = broadcast_shape_for_binary_inputs(&input_shapes[1], &input_shapes[0]);
        (1usize, target_shape)
    } else {
        return;
    };

    let broadcast_name = format!("{}_input{}_broadcast", node.name, smaller_idx);
    let shape_name = format!("{}_shape", broadcast_name);
    let rank = target_shape.len() as u32;
    let shape_data: Vec<u8> = target_shape
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();

    result.tensors.push(TosaTensor {
        name: broadcast_name.clone(),
        shape: target_shape,
        dtype: ir_dtype_to_tosa(input_nodes[smaller_idx].meta.val.dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
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
    result.operators.push(TosaOperator {
        op: Op::Reshape,
        attribute: Attribute::Reshape(ReshapeAttribute),
        inputs: vec![input_names[smaller_idx].clone(), shape_name],
        outputs: vec![broadcast_name.clone()],
        location: None,
    });
    input_names[smaller_idx] = broadcast_name;
}

/// Best-effort rank-4 NHWC broadcast shape for lower-rank binary operands.
///
/// The direct fast lane commonly produces per-channel constants with shapes
/// like `[C, 1, 1]` feeding rank-4 NHWC binary ops. TOSA requires equal ranks
/// for these binary ops, so we materialize a broadcast reshape that preserves
/// the channel value in the last dimension.
fn broadcast_shape_for_binary_inputs(smaller: &[i32], larger: &[i32]) -> Vec<i32> {
    if larger.len() == 4 && smaller.len() < 4 {
        let non_ones: Vec<i32> = smaller.iter().copied().filter(|&d| d != 1).collect();
        if non_ones.len() <= 1 {
            return vec![1, 1, 1, non_ones.first().copied().unwrap_or(1)];
        }
    }

    let mut shape = vec![1; larger.len().saturating_sub(smaller.len())];
    shape.extend_from_slice(smaller);
    shape
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
    let final_shape = apply_tosa_dim_order(node, shape);
    let dtype = ir_dtype_to_tosa(node.meta.val.dtype);
    TosaTensor {
        name: node.name.clone(),
        shape: final_shape,
        dtype,
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    }
}

fn shape_numel(shape: &[i32]) -> Option<usize> {
    if shape.is_empty() {
        return Some(1);
    }

    shape.iter().try_fold(1usize, |acc, &dim| {
        let dim = usize::try_from(dim).ok()?;
        acc.checked_mul(dim)
    })
}

fn scalar_fill_bytes(dtype: crate::graph::ir::DType, value: f64) -> Option<Vec<u8>> {
    use crate::graph::ir::DType as IrDType;

    Some(match dtype {
        IrDType::Float32 => (value as f32).to_le_bytes().to_vec(),
        IrDType::Int8 => (value as i8).to_le_bytes().to_vec(),
        IrDType::Int16 => (value as i16).to_le_bytes().to_vec(),
        IrDType::Int32 => (value as i32).to_le_bytes().to_vec(),
        IrDType::Int64 => (value as i64).to_le_bytes().to_vec(),
        IrDType::Bool => vec![(value != 0.0) as u8],
        // The raw fast-lane currently only materializes FP32/INT32/INT8
        // scalar fills here. Add half-precision packing when a real model
        // needs it instead of silently emitting malformed payloads.
        IrDType::Float16 | IrDType::BFloat16 | IrDType::Int4 | IrDType::Int48 => {
            return None
        }
    })
}

fn make_filled_const_tensor(
    name: String,
    shape: Vec<i32>,
    dtype: crate::graph::ir::DType,
    value: f64,
) -> Option<TosaTensor> {
    let scalar = scalar_fill_bytes(dtype, value)?;
    let numel = shape_numel(&shape)?;
    let mut data = Vec::with_capacity(scalar.len().saturating_mul(numel));
    for _ in 0..numel {
        data.extend_from_slice(&scalar);
    }

    Some(TosaTensor {
        name,
        shape,
        dtype: ir_dtype_to_tosa(dtype),
        data,
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    })
}

fn push_inline_const(result: &mut EmitResult, tensor: TosaTensor) {
    let name = tensor.name.clone();
    result.tensors.push(tensor);
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![name],
        location: None,
    });
}

/// Build an output tensor with NHWC shape for TOSA conv/pool ops.
///
/// For 4D tensors, always permutes NCHW [N,C,H,W] to NHWC [N,H,W,C].
/// The `meta.val` shape is always in NCHW (PyTorch convention) -- the FX
/// memory-format pass annotates `tosa_dim_order` but never rewrites shapes.
fn make_nhwc_output_tensor(node: &Node) -> TosaTensor {
    let shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();
    let dtype = ir_dtype_to_tosa(node.meta.val.dtype);

    // Always permute 4D shapes from NCHW to NHWC.
    //
    // The `meta.val` shape is always in NCHW order (PyTorch convention) — the
    // FX memory-format pass sets `tosa_dim_order` to flag logical layout but
    // never rewrites the actual shape values.  Checking `tosa_dim_order` here
    // would incorrectly skip the permutation, leaving shapes in NCHW and
    // causing Vela to fuse fewer ops.
    let final_shape = if shape.len() == 4 {
        vec![shape[0], shape[2], shape[3], shape[1]]
    } else {
        shape
    };

    TosaTensor {
        name: node.name.clone(),
        shape: final_shape,
        dtype,
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    }
}

/// Convert a node's NCHW meta shape to NHWC, returning `[N, H, W, C]`.
///
/// Always permutes: `meta.val` shapes are always NCHW (PyTorch convention).
/// `tosa_dim_order` is a logical flag that does not change actual shape values.
fn node_nhwc_shape(node: &Node) -> Vec<i32> {
    let shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();
    if shape.len() == 4 {
        vec![shape[0], shape[2], shape[3], shape[1]]
    } else {
        shape
    }
}

/// Compute the NHWC output shape for a TOSA CONV2D from the input and weight
/// shapes and the conv attributes.
///
/// TOSA CONV2D:
///   input  shape = [N, IH, IW, IC]  (NHWC)
///   weight shape = [OC, KH, KW, IC] (OHWI)
///   output shape = [N, OH, OW, OC]
/// where:
///   OH = (IH + pad_top + pad_bottom - (KH-1)*dilation_h - 1) / stride_h + 1
///   OW = (IW + pad_left + pad_right - (KW-1)*dilation_w - 1) / stride_w + 1
fn compute_conv2d_nhwc_output(
    graph: &Graph,
    node: &Node,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> TosaTensor {
    // Quantized convs accumulate into Int32/Int48. Ingestion already fixes
    // `meta.val.dtype` for accumulator ops, so do not override it from
    // output_qparams here or we will incorrectly serialize the conv output as
    // the post-rescale activation dtype.
    let dtype = ir_dtype_to_tosa(node.meta.val.dtype);

    // Try to compute from input + weight shapes
    let computed = (|| -> Option<Vec<i32>> {
        let input_id = match node.args.first()? {
            Arg::Node(id) => *id,
            _ => return None,
        };
        let weight_id = match node.args.get(1)? {
            Arg::Node(id) => *id,
            _ => return None,
        };
        let input_node = graph.node(input_id)?;
        let weight_node = graph.node(weight_id)?;

        // Input in NHWC: [N, IH, IW, IC]
        let inp = actual_tensor_shape(graph, input_node);
        if inp.len() != 4 {
            return None;
        }
        let (n, ih, iw) = (inp[0], inp[1], inp[2]);

        // Weight may be either:
        //   - an original PyTorch constant in OIHW [OC, IC, KH, KW], or
        //   - a synthesized reshape already in TOSA OHWI [OC, KH, KW, IC].
        let ws = actual_tensor_shape(graph, weight_node);
        if ws.len() != 4 {
            return None;
        }
        let ic = *inp.get(3)?;
        let (oc, kh, kw) = if ws[3] == ic {
            // Already OHWI
            (ws[0], ws[1], ws[2])
        } else {
            // Original OIHW -> interpret as OHWI after permutation
            (ws[0], ws[2], ws[3])
        };

        let pad_top = *pad.first().unwrap_or(&0);
        let pad_bottom = *pad.get(1).unwrap_or(&0);
        let pad_left = *pad.get(2).unwrap_or(&0);
        let pad_right = *pad.get(3).unwrap_or(&0);
        let stride_h = *stride.first().unwrap_or(&1);
        let stride_w = *stride.get(1).unwrap_or(&1);
        let dil_h = *dilation.first().unwrap_or(&1);
        let dil_w = *dilation.get(1).unwrap_or(&1);

        let oh = (ih + pad_top + pad_bottom - (kh - 1) * dil_h - 1) / stride_h + 1;
        let ow = (iw + pad_left + pad_right - (kw - 1) * dil_w - 1) / stride_w + 1;

        Some(vec![n, oh, ow, oc])
    })();

    let final_shape = computed.unwrap_or_else(|| {
        // Fallback: use the existing NCHW -> NHWC permutation
        let shape: Vec<i32> = node
            .meta
            .val
            .shape_slice()
            .iter()
            .map(|&d| d as i32)
            .collect();
        if shape.len() == 4 {
            vec![shape[0], shape[2], shape[3], shape[1]]
        } else {
            shape
        }
    });

    TosaTensor {
        name: node.name.clone(),
        shape: final_shape,
        dtype,
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    }
}

/// Compute the NHWC output shape for a TOSA DEPTHWISE_CONV2D from the input
/// and weight shapes and the conv attributes.
///
/// TOSA DEPTHWISE_CONV2D:
///   input  shape = [N, IH, IW, IC]         (NHWC)
///   weight shape = [KH, KW, IC, M]         (HWIM)
///   output shape = [N, OH, OW, IC * M]
/// where:
///   OH = (IH + pad_top + pad_bottom - (KH-1)*dilation_h - 1) / stride_h + 1
///   OW = (IW + pad_left + pad_right - (KW-1)*dilation_w - 1) / stride_w + 1
fn compute_depthwise_conv2d_nhwc_output(
    graph: &Graph,
    node: &Node,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> TosaTensor {
    let dtype = ir_dtype_to_tosa(node.meta.val.dtype);

    // Try to compute from input + weight shapes
    let computed = (|| -> Option<Vec<i32>> {
        let input_id = match node.args.first()? {
            Arg::Node(id) => *id,
            _ => return None,
        };
        let weight_id = match node.args.get(1)? {
            Arg::Node(id) => *id,
            _ => return None,
        };
        let input_node = graph.node(input_id)?;
        let weight_node = graph.node(weight_id)?;

        // Input in NHWC: [N, IH, IW, IC]
        let inp = actual_tensor_shape(graph, input_node);
        if inp.len() != 4 {
            return None;
        }
        let (n, ih, iw) = (inp[0], inp[1], inp[2]);

        // Weight meta.val is OIHW: [OC, IC/groups, KH, KW]
        // After permute to HWIM: [KH, KW, IC_per_group, M]
        // OC = IC_per_group * M, so total output channels = OC
        let ws = weight_node.meta.val.shape_slice();
        if ws.len() != 4 {
            return None;
        }
        // OIHW: oc=ws[0], kh=ws[2], kw=ws[3]
        let (oc, kh, kw) = (ws[0] as i32, ws[2] as i32, ws[3] as i32);

        let pad_top = *pad.first().unwrap_or(&0);
        let pad_bottom = *pad.get(1).unwrap_or(&0);
        let pad_left = *pad.get(2).unwrap_or(&0);
        let pad_right = *pad.get(3).unwrap_or(&0);
        let stride_h = *stride.first().unwrap_or(&1);
        let stride_w = *stride.get(1).unwrap_or(&1);
        let dil_h = *dilation.first().unwrap_or(&1);
        let dil_w = *dilation.get(1).unwrap_or(&1);

        let oh = (ih + pad_top + pad_bottom - (kh - 1) * dil_h - 1) / stride_h + 1;
        let ow = (iw + pad_left + pad_right - (kw - 1) * dil_w - 1) / stride_w + 1;

        Some(vec![n, oh, ow, oc])
    })();

    let final_shape = computed.unwrap_or_else(|| {
        // Fallback: use the existing NCHW -> NHWC permutation
        let shape: Vec<i32> = node
            .meta
            .val
            .shape_slice()
            .iter()
            .map(|&d| d as i32)
            .collect();
        if shape.len() == 4 {
            vec![shape[0], shape[2], shape[3], shape[1]]
        } else {
            shape
        }
    });

    TosaTensor {
        name: node.name.clone(),
        shape: final_shape,
        dtype,
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    }
}

/// Compute the NHWC output shape for a TOSA MAX_POOL2D / AVG_POOL2D from the
/// input shape and the pool attributes.
///
/// Pool ops preserve channels:
///   input  shape = [N, IH, IW, C]  (NHWC)
///   output shape = [N, OH, OW, C]
/// where:
///   OH = (IH + pad_top + pad_bottom - kernel_h) / stride_h + 1
///   OW = (IW + pad_left + pad_right - kernel_w) / stride_w + 1
fn compute_pool_nhwc_output(
    graph: &Graph,
    node: &Node,
    kernel: &[i32],
    stride: &[i32],
    pad: &[i32],
) -> TosaTensor {
    let dtype = node_output_dtype(node);

    let computed = (|| -> Option<Vec<i32>> {
        let input_id = match node.args.first()? {
            Arg::Node(id) => *id,
            _ => return None,
        };
        let input_node = graph.node(input_id)?;

        // Input in NHWC: [N, IH, IW, C]
        let inp = actual_tensor_shape(graph, input_node);
        if inp.len() != 4 {
            return None;
        }
        let (n, ih, iw, c) = (inp[0], inp[1], inp[2], inp[3]);

        let kh = *kernel.first().unwrap_or(&2);
        let kw = *kernel.get(1).unwrap_or(&2);
        let pad_top = *pad.first().unwrap_or(&0);
        let pad_bottom = *pad.get(1).unwrap_or(&0);
        let pad_left = *pad.get(2).unwrap_or(&0);
        let pad_right = *pad.get(3).unwrap_or(&0);
        let stride_h = *stride.first().unwrap_or(&2);
        let stride_w = *stride.get(1).unwrap_or(&2);

        let oh = (ih + pad_top + pad_bottom - kh) / stride_h + 1;
        let ow = (iw + pad_left + pad_right - kw) / stride_w + 1;

        Some(vec![n, oh, ow, c])
    })();

    let final_shape = computed.unwrap_or_else(|| {
        let shape: Vec<i32> = node
            .meta
            .val
            .shape_slice()
            .iter()
            .map(|&d| d as i32)
            .collect();
        if shape.len() == 4 {
            vec![shape[0], shape[2], shape[3], shape[1]]
        } else {
            shape
        }
    });

    TosaTensor {
        name: node.name.clone(),
        shape: final_shape,
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

/// Extract a single i64 from an arg at the given index.
fn arg_i64(node: &Node, idx: usize, default: i64) -> i64 {
    match node.args.get(idx) {
        Some(Arg::Int(i)) => *i,
        _ => default,
    }
}

/// Extract a bool from an arg at the given index.
fn arg_bool(node: &Node, idx: usize, default: bool) -> bool {
    match node.args.get(idx) {
        Some(Arg::Bool(b)) => *b,
        _ => default,
    }
}

/// Get the number of output channels from the weight tensor at arg position 1.
///
/// For conv weights in OIHW format, OC = shape[0].
fn get_weight_out_channels(graph: &Graph, node: &Node) -> Option<usize> {
    let weight_id = match node.args.get(1) {
        Some(Arg::Node(id)) => *id,
        _ => return None,
    };
    let weight_node = graph.node(weight_id)?;
    let ws = weight_node.meta.val.shape_slice();
    if ws.is_empty() {
        return None;
    }
    Some(ws[0] as usize)
}

/// Encode a zero-point value using the correct number of bytes for the dtype.
///
/// INT8 -> 1 byte, INT16 -> 2 bytes, INT32 -> 4 bytes, etc.
/// This matches the Python serializer's `convertDataToUint8Vec` behavior.
fn encode_zp_for_dtype(zp_val: i32, dtype: DType) -> Vec<u8> {
    match dtype {
        DType::Int8 | DType::Bool => vec![zp_val as u8],
        DType::Int16 | DType::Fp16 | DType::Bf16 => (zp_val as i16).to_le_bytes().to_vec(),
        DType::Int32 | DType::Fp32 => zp_val.to_le_bytes().to_vec(),
        DType::Int48 => {
            let bytes = (zp_val as i64).to_le_bytes();
            bytes[..6].to_vec()
        }
        DType::Int64 => (zp_val as i64).to_le_bytes().to_vec(),
        _ => vec![zp_val as u8],
    }
}

/// Match the Python reference's odd `weight_zp` encoding:
/// tensor shape is scalar `[1]`, but payload length equals output channels.
fn make_weight_zp_data(zp_val: i32, num_channels: usize) -> Vec<u8> {
    vec![zp_val as u8; num_channels]
}

fn default_clamp_bound_bytes(dtype: crate::graph::ir::DType, is_min: bool) -> Vec<u8> {
    use crate::graph::ir::DType as IrDtype;
    match dtype {
        IrDtype::Bool => vec![if is_min { 0 } else { 1 }],
        IrDtype::Int4 | IrDtype::Int8 => vec![(if is_min { i8::MIN } else { i8::MAX }) as u8],
        IrDtype::Int16 => (if is_min { i16::MIN } else { i16::MAX })
            .to_le_bytes()
            .to_vec(),
        IrDtype::Int32 => (if is_min { i32::MIN } else { i32::MAX })
            .to_le_bytes()
            .to_vec(),
        IrDtype::Int64 => (if is_min { i64::MIN } else { i64::MAX })
            .to_le_bytes()
            .to_vec(),
        IrDtype::Int48 => {
            let bytes = (if is_min { i64::MIN } else { i64::MAX }).to_le_bytes();
            bytes[..6].to_vec()
        }
        IrDtype::Float32 => (if is_min { f32::MIN } else { f32::MAX })
            .to_le_bytes()
            .to_vec(),
        // Match the finite dtype bounds used by torch.finfo for default clamp args.
        IrDtype::Float16 => {
            if is_min {
                vec![0xff, 0xfb]
            } else {
                vec![0xff, 0x7b]
            }
        }
        IrDtype::BFloat16 => {
            if is_min {
                vec![0x7f, 0xff]
            } else {
                vec![0x7f, 0x7f]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ATen-to-TOSA fallback emitters
//
// When lower_to_tosa_v2's FX passes don't fully rewrite an op to a TOSA
// target, these emitters allow the serializer to handle common ATen ops
// by emitting the equivalent TOSA operator.
// ---------------------------------------------------------------------------

/// Emit TOSA CONV2D (or DEPTHWISE_CONV2D) from an ATen convolution node.
///
/// ATen convolution args: [input, weight, bias, stride, padding, dilation,
///                         transposed, output_padding, groups]
///
/// TOSA CONV2D requires 5 inputs: [input, weight, bias, input_zp, weight_zp].
/// We create inline const tensors for the zero-point scalars from qparams
/// metadata, defaulting to 0 for symmetric quantization.
///
/// Depthwise detection: if groups == in_channels and weight.shape[1] == 1,
/// emit DEPTHWISE_CONV2D instead.
pub fn emit_aten_conv2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);

    // Extract from positional args (ATen convolution layout)
    let stride = arg_i32_list(node, 3);
    let padding = arg_i32_list(node, 4);
    let dilation = arg_i32_list(node, 5);
    let groups = arg_i64(node, 8, 1);

    // TOSA pad format is [top, bottom, left, right] from ATen [H, W]
    let pad = if padding.len() >= 2 {
        vec![padding[0], padding[0], padding[1], padding[1]]
    } else if padding.len() == 1 {
        vec![padding[0], padding[0], padding[0], padding[0]]
    } else {
        vec![0, 0, 0, 0]
    };

    let acc_type = DType::Int32;

    // Get zero-points from qparams (default to 0 for symmetric quantization)
    let input_zp_val: i32 = node
        .meta
        .input_qparams
        .iter()
        .find(|(pos, _)| *pos == 0)
        .map(|(_, qp)| qp.zero_point as i32)
        .unwrap_or(0);
    let weight_zp_val: i32 = node
        .meta
        .input_qparams
        .iter()
        .find(|(pos, _)| *pos == 1)
        .map(|(_, qp)| qp.zero_point as i32)
        .unwrap_or(0);

    // Create input_zp const tensor (scalar int8, matching input dtype)
    let input_zp_name = format!("{}_input_zp", node.name);
    result.tensors.push(TosaTensor {
        name: input_zp_name.clone(),
        shape: vec![1],
        dtype: DType::Int8,
        data: vec![input_zp_val as u8],
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

    // Match the reference serializer exactly: scalar shape but one byte per
    // output channel in the payload.
    let weight_zp_name = format!("{}_weight_zp", node.name);
    let num_oc = get_weight_out_channels(graph, node).unwrap_or(1);
    let weight_zp_data = make_weight_zp_data(weight_zp_val, num_oc);
    result.tensors.push(TosaTensor {
        name: weight_zp_name.clone(),
        shape: vec![1],
        dtype: DType::Int8,
        data: weight_zp_data,
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![weight_zp_name.clone()],
        location: None,
    });

    // Build 5-input list: [input, weight, bias, input_zp, weight_zp]
    // ATen conv has more args (stride, padding, etc.) as Arg::IntList, not
    // Arg::Node, so collect_input_names only picks up the first 3 node refs.
    let mut conv_inputs = input_names;
    conv_inputs.truncate(3);
    conv_inputs.push(input_zp_name);
    conv_inputs.push(weight_zp_name);

    // Detect depthwise: prefer the conv_info from FxRewriteConvPass if available.
    // Fallback: use groups == in_channels heuristic, but require groups > 1 to
    // avoid false positives on 1x1 convs from decomposed linear (whose shapes
    // may be in NHWC format making s[1] != in_channels).
    let is_depthwise = if let Some(conv_info_entry) = node
        .kwargs
        .iter()
        .find(|(k, _)| k == "conv_info_is_depthwise")
    {
        match &conv_info_entry.1 {
            Arg::Int(v) => *v != 0,
            _ => false,
        }
    } else if groups > 1 {
        // groups > 1 is a reliable indicator — check if groups == in_channels
        let in_channels = if let Some(Arg::Node(input_id)) = node.args.first() {
            graph
                .node(*input_id)
                .map(|n| {
                    let s = n.meta.val.shape_slice();
                    if s.len() >= 2 {
                        s[1]
                    } else {
                        0
                    }
                })
                .unwrap_or(0)
        } else {
            0
        };
        groups == in_channels
    } else {
        false
    };

    let stride = if stride.is_empty() {
        vec![1, 1]
    } else {
        stride
    };
    let dilation = if dilation.is_empty() {
        vec![1, 1]
    } else {
        dilation
    };

    // Compute NHWC output shape from TOSA formula
    if is_depthwise {
        result.tensors.push(compute_depthwise_conv2d_nhwc_output(
            graph, node, &pad, &stride, &dilation,
        ));
        result.operators.push(TosaOperator {
            op: Op::DepthwiseConv2d,
            attribute: Attribute::DepthwiseConv2d(DepthwiseConv2dAttribute {
                pad,
                stride,
                dilation,
                local_bound: false,
                acc_type,
            }),
            inputs: conv_inputs,
            outputs: vec![node.name.clone()],
            location: None,
        });
    } else {
        result.tensors.push(compute_conv2d_nhwc_output(
            graph, node, &pad, &stride, &dilation,
        ));
        result.operators.push(TosaOperator {
            op: Op::Conv2d,
            attribute: Attribute::Conv2d(Conv2dAttribute {
                pad,
                stride,
                dilation,
                local_bound: false,
                acc_type,
            }),
            inputs: conv_inputs,
            outputs: vec![node.name.clone()],
            location: None,
        });
    }

    result
}

/// Emit TOSA MAX_POOL2D from an ATen max_pool2d node.
///
/// ATen max_pool2d args: [input, kernel_size, stride, padding, dilation, ceil_mode]
pub fn emit_aten_max_pool2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let output_name = primary_output_name(graph, node);

    let input_names = collect_input_names(graph, node);

    let kernel = arg_i32_list(node, 1);
    let stride = arg_i32_list(node, 2);
    let padding = arg_i32_list(node, 3);

    let pad = if padding.len() >= 2 {
        vec![padding[0], padding[0], padding[1], padding[1]]
    } else if padding.len() == 1 {
        vec![padding[0], padding[0], padding[0], padding[0]]
    } else {
        vec![0, 0, 0, 0]
    };

    // Compute NHWC output shape from pool formula
    let kernel_for_shape = if kernel.is_empty() {
        vec![2, 2]
    } else {
        kernel.clone()
    };
    let stride_for_shape = if stride.is_empty() {
        vec![2, 2]
    } else {
        stride.clone()
    };
    let mut output_tensor =
        compute_pool_nhwc_output(graph, node, &kernel_for_shape, &stride_for_shape, &pad);
    output_tensor.name = output_name.clone();
    result.tensors.push(output_tensor);

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
            pad,
            nan_mode: NanPropagationMode::Propagate,
        }),
        inputs: input_names,
        outputs: vec![output_name],
        location: None,
    });

    result
}

/// Emit TOSA CLAMP from an ATen clamp node.
///
/// ATen clamp args: [input, min_val, max_val]
pub fn emit_aten_clamp(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    let clamp_dtype = node.meta.val.dtype;

    // Extract min/max from positional args
    let min_val = match node.args.get(1) {
        Some(Arg::Int(i)) => (*i as i8).to_le_bytes().to_vec(),
        Some(Arg::Float(f)) => (*f as f32).to_le_bytes().to_vec(),
        _ => default_clamp_bound_bytes(clamp_dtype, true),
    };
    let max_val = match node.args.get(2) {
        Some(Arg::Int(i)) => (*i as i8).to_le_bytes().to_vec(),
        Some(Arg::Float(f)) => (*f as f32).to_le_bytes().to_vec(),
        _ => default_clamp_bound_bytes(clamp_dtype, false),
    };

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

/// Emit TOSA RESHAPE from an ATen reshape/view/flatten node.
///
/// Uses the node's output shape from metadata.
pub fn emit_aten_reshape(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    // Delegate to the existing reshape emitter (it uses output shape from meta)
    emit_reshape(graph, _node_id, node, _store)
}

/// Emit TOSA TRANSPOSE from an ATen permute node.
///
/// ATen permute args: [input, dims]
pub fn emit_aten_permute(
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
        attribute: Attribute::Transpose(TransposeAttribute { perms }),
        inputs: input_names,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit TOSA ADD from an ATen add node.
pub fn emit_aten_add(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_add(graph, _node_id, node, _store)
}

/// Emit TOSA SUB from an ATen sub node.
pub fn emit_aten_sub(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_sub(graph, _node_id, node, _store)
}

/// Emit TOSA MUL from an ATen mul node.
pub fn emit_aten_mul(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    emit_mul(graph, _node_id, node, _store)
}

/// Emit one or more TOSA REDUCE_SUM ops from an ATen `sum.dim_IntList` node.
///
/// ATen sum args: [input, dims_intlist, keepdim].
/// After `FxConvertSumPass`, each Sum node typically has a single dim,
/// but we handle multi-dim by chaining REDUCE_SUM ops (descending axis
/// order so earlier reductions don't shift later indices).
pub fn emit_aten_sum(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    let input_name = input_names.first().cloned().unwrap_or_default();
    let input_node = match node.args.first() {
        Some(Arg::Node(id)) => graph.node(*id),
        _ => None,
    };

    // Extract dims from args[1] (IntList)
    let mut dims = arg_i32_list(node, 1);
    if dims.is_empty() {
        // Fallback: check kwargs "axis" (IR-level nodes use this)
        let axis = kwarg_i32(node, "axis", -1);
        if axis >= 0 {
            dims = vec![axis];
        }
    }

    if dims.is_empty() || input_name.is_empty() {
        // Can't emit without dims or input — fall back to a single REDUCE_SUM axis=0
        result.tensors.push(make_output_tensor(node));
        result.operators.push(TosaOperator {
            op: Op::ReduceSum,
            attribute: Attribute::ReduceSum(ReduceSumAttribute { axis: 0 }),
            inputs: vec![input_name],
            outputs: vec![node.name.clone()],
            location: None,
        });
        return result;
    }

    // Sort dims descending so higher axes are reduced first
    dims.sort_unstable();
    dims.reverse();

    if dims.len() == 1 {
        // Single axis — straightforward
        let axis = input_node
            .map(|n| map_axis_to_tosa_dim_order(n, dims[0]))
            .unwrap_or(dims[0]);
        result.tensors.push(make_output_tensor(node));
        result.operators.push(TosaOperator {
            op: Op::ReduceSum,
            attribute: Attribute::ReduceSum(ReduceSumAttribute { axis }),
            inputs: vec![input_name],
            outputs: vec![node.name.clone()],
            location: None,
        });
    } else {
        // Multi-axis — chain REDUCE_SUM ops with intermediate tensors
        let mut current_input = input_name;

        // Compute intermediate shapes by reducing dims one at a time
        let base_shape: Vec<i32> = node
            .meta
            .val
            .shape_slice()
            .iter()
            .map(|&d| d as i32)
            .collect();
        let base_dtype = ir_dtype_to_tosa(node.meta.val.dtype);

        // We need the input shape to compute intermediates
        let input_shape: Vec<i32> = graph
            .node(match node.args.first() {
                Some(Arg::Node(id)) => *id,
                _ => return result,
            })
            .map(|n| n.meta.val.shape_slice().iter().map(|&d| d as i32).collect())
            .unwrap_or_else(|| base_shape.clone());

        let mut current_shape = input_shape;

        for (i, &dim) in dims.iter().enumerate() {
            let is_last = i == dims.len() - 1;
            let dim_idx = dim as usize;
            let axis = input_node
                .map(|n| map_axis_to_tosa_dim_order(n, dim))
                .unwrap_or(dim);

            // Output shape: current shape with this dim set to 1
            let mut out_shape = current_shape.clone();
            if dim_idx < out_shape.len() {
                out_shape[dim_idx] = 1;
            }

            let out_name = if is_last {
                node.name.clone()
            } else {
                format!("{}_reduce_axis{}", node.name, dim)
            };

            result.tensors.push(TosaTensor {
                name: out_name.clone(),
                shape: out_shape.clone(),
                dtype: base_dtype,
                data: Vec::new(),
                variable: false,
                is_unranked: false,
                variable_name: String::new(),
            });

            result.operators.push(TosaOperator {
                op: Op::ReduceSum,
                attribute: Attribute::ReduceSum(ReduceSumAttribute { axis }),
                inputs: vec![current_input],
                outputs: vec![out_name.clone()],
                location: None,
            });

            current_input = out_name;
            current_shape = out_shape;
        }
    }

    result
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
    let (shape, dtype, data) = if let Some(tensor_data) = store.get(name) {
        // Generated TOSA CONST nodes can retain stale FakeTensor metadata after
        // dtype/shape fixups. When backing bytes exist in the tensor store,
        // serialize the store's authoritative shape and dtype to keep the
        // flatbuffer internally consistent.
        (
            tensor_data.shape.iter().map(|&d| d as i32).collect(),
            ir_dtype_to_tosa(tensor_data.dtype),
            tensor_data.data.clone(),
        )
    } else {
        (
            node.meta
                .val
                .shape_slice()
                .iter()
                .map(|&d| d as i32)
                .collect(),
            ir_dtype_to_tosa(node.meta.val.dtype),
            Vec::new(),
        )
    };

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
///
/// TOSA CONV2D requires 5 inputs: [input, weight, bias, input_zp, weight_zp].
/// If the incoming node only provides 3 inputs, we create inline const tensors
/// for the zero-point scalars from kwargs or qparams metadata.
pub fn emit_conv2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);

    // Extract attribute from kwargs, falling back to positional args.
    // Nodes created by rewrite_conv_to_tosa store pad/stride/dilation as
    // positional args: (input, weight, bias, stride, pad, dilation).
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if kw.is_empty() {
            arg_i32_list(node, 4)
        } else {
            kw
        }
    };
    let stride = {
        let kw = kwarg_i32_list(node, "stride");
        if kw.is_empty() {
            arg_i32_list(node, 3)
        } else {
            kw
        }
    };
    let dilation = {
        let kw = kwarg_i32_list(node, "dilation");
        if kw.is_empty() {
            arg_i32_list(node, 5)
        } else {
            kw
        }
    };
    let local_bound = kwarg_bool(node, "local_bound", false);

    // Compute NHWC output shape from TOSA formula using input/weight shapes
    // and conv attributes, rather than relying on node.meta.val which may be
    // incorrect NCHW values from the original FX graph.
    let pad_for_shape = if pad.is_empty() {
        vec![0, 0, 0, 0]
    } else {
        pad.clone()
    };
    let stride_for_shape = if stride.is_empty() {
        vec![1, 1]
    } else {
        stride.clone()
    };
    let dilation_for_shape = if dilation.is_empty() {
        vec![1, 1]
    } else {
        dilation.clone()
    };
    result.tensors.push(compute_conv2d_nhwc_output(
        graph,
        node,
        &pad_for_shape,
        &stride_for_shape,
        &dilation_for_shape,
    ));

    // Determine accumulator type — default to Int32 for quantized
    let acc_type = DType::Int32;

    // If we already have 5 inputs, pass them through; otherwise create zp tensors.
    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        // Get zero-points from qparams metadata or conv_info kwargs
        let input_zp_val: i32 = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "input_zp", 0));
        let weight_zp_val: i32 = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 1)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "weight_zp", 0));

        // Create input_zp const tensor (scalar int8, matching input dtype)
        let input_zp_name = format!("{}_input_zp", node.name);
        result.tensors.push(TosaTensor {
            name: input_zp_name.clone(),
            shape: vec![1],
            dtype: DType::Int8,
            data: vec![input_zp_val as u8],
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

        // Match the reference serializer exactly: scalar shape but one byte
        // per output channel in the payload.
        let weight_zp_name = format!("{}_weight_zp", node.name);
        let num_oc = get_weight_out_channels(graph, node).unwrap_or(1);
        let weight_zp_data = make_weight_zp_data(weight_zp_val, num_oc);
        result.tensors.push(TosaTensor {
            name: weight_zp_name.clone(),
            shape: vec![1],
            dtype: DType::Int8,
            data: weight_zp_data,
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });
        result.operators.push(TosaOperator {
            op: Op::Const,
            attribute: Attribute::Const(ConstAttribute),
            inputs: vec![],
            outputs: vec![weight_zp_name.clone()],
            location: None,
        });

        let mut inputs = input_names;
        inputs.truncate(3);
        inputs.push(input_zp_name);
        inputs.push(weight_zp_name);
        inputs
    };

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
        inputs: conv_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a DEPTHWISE_CONV2D operator.
///
/// TOSA DEPTHWISE_CONV2D requires 5 inputs: [input, weight, bias, input_zp, weight_zp].
/// If the incoming node only provides 3 inputs, we create inline const tensors
/// for the zero-point scalars from kwargs or qparams metadata.
pub fn emit_depthwise_conv2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);

    // Extract attribute from kwargs, falling back to positional args.
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if kw.is_empty() {
            arg_i32_list(node, 4)
        } else {
            kw
        }
    };
    let stride = {
        let kw = kwarg_i32_list(node, "stride");
        if kw.is_empty() {
            arg_i32_list(node, 3)
        } else {
            kw
        }
    };
    let dilation = {
        let kw = kwarg_i32_list(node, "dilation");
        if kw.is_empty() {
            arg_i32_list(node, 5)
        } else {
            kw
        }
    };
    let local_bound = kwarg_bool(node, "local_bound", false);
    let acc_type = DType::Int32;

    // Compute NHWC output shape from TOSA formula using input/weight shapes
    // and conv attributes.
    let pad_for_shape = if pad.is_empty() {
        vec![0, 0, 0, 0]
    } else {
        pad.clone()
    };
    let stride_for_shape = if stride.is_empty() {
        vec![1, 1]
    } else {
        stride.clone()
    };
    let dilation_for_shape = if dilation.is_empty() {
        vec![1, 1]
    } else {
        dilation.clone()
    };
    result.tensors.push(compute_depthwise_conv2d_nhwc_output(
        graph,
        node,
        &pad_for_shape,
        &stride_for_shape,
        &dilation_for_shape,
    ));

    // If we already have 5 inputs, pass them through; otherwise create zp tensors.
    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        // Get zero-points from qparams metadata or conv_info kwargs
        let input_zp_val: i32 = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "input_zp", 0));
        let weight_zp_val: i32 = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 1)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "weight_zp", 0));

        // Create input_zp const tensor (scalar int8, matching input dtype)
        let input_zp_name = format!("{}_input_zp", node.name);
        result.tensors.push(TosaTensor {
            name: input_zp_name.clone(),
            shape: vec![1],
            dtype: DType::Int8,
            data: vec![input_zp_val as u8],
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

        // Match the reference serializer exactly: scalar shape but one byte
        // per output channel in the payload.
        let weight_zp_name = format!("{}_weight_zp", node.name);
        let num_oc = get_weight_out_channels(graph, node).unwrap_or(1);
        let weight_zp_data = make_weight_zp_data(weight_zp_val, num_oc);
        result.tensors.push(TosaTensor {
            name: weight_zp_name.clone(),
            shape: vec![1],
            dtype: DType::Int8,
            data: weight_zp_data,
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });
        result.operators.push(TosaOperator {
            op: Op::Const,
            attribute: Attribute::Const(ConstAttribute),
            inputs: vec![],
            outputs: vec![weight_zp_name.clone()],
            location: None,
        });

        let mut inputs = input_names;
        inputs.truncate(3);
        inputs.push(input_zp_name);
        inputs.push(weight_zp_name);
        inputs
    };

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
        inputs: conv_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a RESCALE operator.
///
/// Handles two ingestion formats:
/// 1. **Graph-IR format** (from InsertRescalePass): args = [input, multiplier_const, shift_const],
///    kwargs = {input_zp, output_zp}. Multiplier/shift are separate placeholder nodes.
/// 2. **FX-direct format** (from fx_helpers.insert_rescale_node): args = [input, dtype, scales, input_zp, output_zp].
///    Scales are FloatList and we compute multiplier/shift inline.
///
/// TOSA RESCALE takes 5 inputs: [input, multiplier, shift, input_zp, output_zp].
pub fn emit_rescale(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);

    // RESCALE is format-transparent: serialize exactly the same physical shape
    // that downstream shape propagation will observe.
    result.tensors.push(TosaTensor {
        name: node.name.clone(),
        shape: actual_tensor_shape(graph, node),
        dtype: ir_dtype_to_tosa(node.meta.val.dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });

    // Detect format: Graph-IR has >=3 Node args (input, mult, shift);
    // FX-direct has 1 Node arg (input) + literal scales/zps.
    let is_fx_direct = input_names.len() < 3;

    let (input_zp_val, output_zp_val, multiplier_name, shift_name, scale32, per_channel);

    if is_fx_direct {
        // FX-direct format: args = [Node(input), Dtype, FloatList(scales), Int(input_zp), Int(output_zp)]
        let scales: Vec<f64> = match node.args.get(2) {
            Some(Arg::FloatList(list)) => list.iter().copied().collect(),
            Some(Arg::Float(f)) => vec![*f],
            _ => vec![1.0],
        };
        input_zp_val = arg_i64(node, 3, 0) as i32;
        output_zp_val = arg_i64(node, 4, 0) as i32;
        per_channel = scales.len() > 1;
        scale32 = true;

        // Compute multiplier/shift from scales
        let (multipliers, shifts): (Vec<i32>, Vec<i8>) = scales
            .iter()
            .map(|&s| compute_multiplier_and_shift(s))
            .unzip();

        // Create multiplier const tensor
        let mult_name = format!("{}_multipliers", node.name);
        let mult_data: Vec<u8> = multipliers.iter().flat_map(|m| m.to_le_bytes()).collect();
        result.tensors.push(TosaTensor {
            name: mult_name.clone(),
            shape: vec![multipliers.len() as i32],
            dtype: DType::Int32,
            data: mult_data,
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });
        result.operators.push(TosaOperator {
            op: Op::Const,
            attribute: Attribute::Const(ConstAttribute),
            inputs: vec![],
            outputs: vec![mult_name.clone()],
            location: None,
        });

        // Create shift const tensor
        let shift_name_str = format!("{}_shifts", node.name);
        let shift_data: Vec<u8> = shifts.iter().map(|&s| s as u8).collect();
        result.tensors.push(TosaTensor {
            name: shift_name_str.clone(),
            shape: vec![shifts.len() as i32],
            dtype: DType::Int8,
            data: shift_data,
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });
        result.operators.push(TosaOperator {
            op: Op::Const,
            attribute: Attribute::Const(ConstAttribute),
            inputs: vec![],
            outputs: vec![shift_name_str.clone()],
            location: None,
        });

        multiplier_name = mult_name;
        shift_name = shift_name_str;
    } else {
        // Graph-IR format: input_names = [input, multiplier, shift]
        multiplier_name = input_names.get(1).cloned().unwrap_or_default();
        shift_name = input_names.get(2).cloned().unwrap_or_default();
        input_zp_val = kwarg_i32(node, "input_zp", 0);
        output_zp_val = kwarg_i32(node, "output_zp", 0);
        scale32 = kwarg_bool(node, "scale32", true);
        per_channel = kwarg_bool(node, "per_channel", false);
    }

    let input_unsigned = kwarg_bool(node, "input_unsigned", false);
    let output_unsigned = kwarg_bool(node, "output_unsigned", false);

    // Create input_zp const tensor (scalar, uses input node's dtype)
    let input_zp_name = format!("{}_input_zp", node.name);
    let input_zp_tosa_dtype = if let Some(Arg::Node(input_id)) = node.args.first() {
        graph
            .node(*input_id)
            .map(|n| ir_dtype_to_tosa(n.meta.val.dtype))
            .unwrap_or(DType::Int8)
    } else {
        DType::Int8
    };
    let input_zp_val = normalize_rescale_zp(input_zp_tosa_dtype, input_zp_val, input_unsigned);
    let input_zp_data = encode_zp_for_dtype(input_zp_val, input_zp_tosa_dtype);
    result.tensors.push(TosaTensor {
        name: input_zp_name.clone(),
        shape: vec![1],
        dtype: input_zp_tosa_dtype,
        data: input_zp_data,
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

    // Create output_zp const tensor (scalar, uses rescale output dtype)
    let output_zp_name = format!("{}_output_zp", node.name);
    let output_zp_tosa_dtype = ir_dtype_to_tosa(node.meta.val.dtype);
    let output_zp_val =
        normalize_rescale_zp(output_zp_tosa_dtype, output_zp_val, output_unsigned);
    let output_zp_data = encode_zp_for_dtype(output_zp_val, output_zp_tosa_dtype);
    result.tensors.push(TosaTensor {
        name: output_zp_name.clone(),
        shape: vec![1],
        dtype: output_zp_tosa_dtype,
        data: output_zp_data,
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
    let input_name = input_names.first().cloned().unwrap_or_default();
    let rescale_inputs = vec![
        input_name,
        multiplier_name,
        shift_name,
        input_zp_name,
        output_zp_name,
    ];

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

/// Convert a floating-point scale to a fixed-point (multiplier, shift) pair
/// matching the Python `_compute_multiplier_and_shift` behavior exactly.
///
/// Uses `frexp`-equivalent decomposition: scale = mantissa * 2^exponent
/// where 0.5 <= mantissa < 1.0. Then:
///   shifted_mantissa = round(mantissa * 2^31)
///   right_shift = 31 - exponent
fn compute_multiplier_and_shift(scale: f64) -> (i32, i8) {
    if scale == 0.0 {
        return (0, 0);
    }

    // frexp: scale = mantissa * 2^exponent, 0.5 <= mantissa < 1.0
    let bits = scale.to_bits();
    let biased_exp = ((bits >> 52) & 0x7FF) as i32;
    let exponent = biased_exp - 1022; // = frexp exponent
    let mantissa = f64::from_bits((bits & 0x800F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000);

    let const_2_31: i64 = 1 << 31;
    let mut shifted_mantissa = (mantissa * const_2_31 as f64).round() as i64;
    let mut shift = exponent;

    // Handle rounding up to 2^31
    if shifted_mantissa == const_2_31 {
        shifted_mantissa /= 2;
        shift += 1;
    }

    // TOSA expects right shift = offset - shift where offset = 31
    let right_shift = 31 - shift;

    // Clamp shift to valid range and adjust multiplier if needed
    let mut multiplier = shifted_mantissa as i32;
    let mut final_shift = right_shift;
    if final_shift > 62 {
        multiplier >>= (final_shift - 62).min(31);
        final_shift = 62;
    }

    (multiplier, final_shift as i8)
}

/// Emit a MAX_POOL2D operator.
pub fn emit_max_pool2d(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let output_name = primary_output_name(graph, node);

    let input_names = collect_input_names(graph, node);

    let kernel = kwarg_i32_list(node, "kernel");
    let stride = kwarg_i32_list(node, "stride");
    let pad = kwarg_i32_list(node, "pad");

    // Compute NHWC output shape from pool formula
    let kernel_for_shape = if kernel.is_empty() {
        vec![2, 2]
    } else {
        kernel.clone()
    };
    let stride_for_shape = if stride.is_empty() {
        vec![2, 2]
    } else {
        stride.clone()
    };
    let pad_for_shape = if pad.is_empty() {
        vec![0, 0, 0, 0]
    } else {
        pad.clone()
    };
    let mut output_tensor = compute_pool_nhwc_output(
        graph,
        node,
        &kernel_for_shape,
        &stride_for_shape,
        &pad_for_shape,
    );
    output_tensor.name = output_name.clone();
    result.tensors.push(output_tensor);

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
        outputs: vec![output_name],
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
    let perms = {
        let perms = arg_i32_list(node, 1);
        if perms.is_empty() {
            kwarg_i32_list(node, "perms")
        } else {
            perms
        }
    };
    let mut output_shape: Vec<i32> = node
        .meta
        .val
        .shape_slice()
        .iter()
        .map(|&d| d as i32)
        .collect();
    if let Some(Arg::Node(input_id)) = node.args.first() {
        if let Some(input_node) = graph.node(*input_id) {
            let input_shape = actual_tensor_shape(graph, input_node);
            if input_shape.len() == perms.len() {
                let candidate: Vec<i32> = perms
                    .iter()
                    .filter_map(|&idx| input_shape.get(idx as usize).copied())
                    .collect();
                if candidate.len() == input_shape.len() {
                    output_shape = candidate;
                }
            }
        }
    }
    result.tensors.push(TosaTensor {
        name: node.name.clone(),
        shape: output_shape,
        dtype: ir_dtype_to_tosa(node.meta.val.dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });

    result.operators.push(TosaOperator {
        op: Op::Transpose,
        attribute: Attribute::Transpose(TransposeAttribute { perms }),
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

    // Use one physical shape consistently for both the RESHAPE output tensor
    // and its CONST_SHAPE input. Mixing raw meta.val with reordered output
    // tensor shapes produces an internally inconsistent flatbuffer.
    let out_shape = actual_tensor_shape(graph, node);
    result.tensors.push(TosaTensor {
        name: node.name.clone(),
        shape: out_shape.clone(),
        dtype: ir_dtype_to_tosa(node.meta.val.dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });

    // Create a CONST_SHAPE tensor for the target shape
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
    let clamp_dtype = node.meta.val.dtype;

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

    if min_val.is_empty() {
        min_val = default_clamp_bound_bytes(clamp_dtype, true);
    }
    if max_val.is_empty() {
        max_val = default_clamp_bound_bytes(clamp_dtype, false);
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
pub fn emit_add(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    let mut add_inputs = input_names;
    maybe_materialize_binary_broadcast(graph, node, &mut result, &mut add_inputs);
    result.tensors.push(make_output_tensor(node));

    result.operators.push(TosaOperator {
        op: Op::Add,
        attribute: Attribute::Add(AddAttribute),
        inputs: add_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a SUB operator.
pub fn emit_sub(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    let mut sub_inputs = input_names;
    maybe_materialize_binary_broadcast(graph, node, &mut result, &mut sub_inputs);
    result.tensors.push(make_output_tensor(node));

    result.operators.push(TosaOperator {
        op: Op::Sub,
        attribute: Attribute::Sub(SubAttribute),
        inputs: sub_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

/// Emit a MUL operator.
///
/// TOSA MUL takes 3 inputs: lhs, rhs, and a shift tensor (scalar int8, typically 0).
/// We create an inline const tensor for the shift value.
pub fn emit_mul(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
    let mut result = EmitResult::default();

    let input_names = collect_input_names(graph, node);
    let mut mul_inputs = input_names;
    maybe_materialize_binary_broadcast(graph, node, &mut result, &mut mul_inputs);
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

    let kernel = {
        let kw = kwarg_i32_list(node, "kernel");
        if kw.is_empty() {
            arg_i32_list(node, 1)
        } else {
            kw
        }
    };
    let mut stride = {
        let kw = kwarg_i32_list(node, "stride");
        if kw.is_empty() {
            arg_i32_list(node, 2)
        } else {
            kw
        }
    };
    let pad = {
        let kw = kwarg_i32_list(node, "pad");
        if !kw.is_empty() {
            kw
        } else {
            let padding = arg_i32_list(node, 3);
            if padding.len() >= 2 {
                vec![padding[0], padding[0], padding[1], padding[1]]
            } else if padding.len() == 1 {
                vec![padding[0], padding[0], padding[0], padding[0]]
            } else {
                vec![0, 0, 0, 0]
            }
        }
    };
    if stride.is_empty() {
        stride = kernel.clone();
    }

    // Compute NHWC output shape from pool formula
    result.tensors.push(compute_pool_nhwc_output(
        graph, node, &kernel, &stride, &pad,
    ));
    let acc_type = DType::Int32;
    let input_zp = kwarg_i32(node, "input_zp", 0);
    let output_zp = kwarg_i32(node, "output_zp", 0);

    // Determine input dtype (typically Int8 for quantized models)
    let input_dtype = match node.args.first() {
        Some(Arg::Node(input_id)) => graph
            .node(*input_id)
            .map(|n| ir_dtype_to_tosa(n.meta.val.dtype))
            .unwrap_or(DType::Int8),
        _ => DType::Int8,
    };
    let output_dtype = ir_dtype_to_tosa(node.meta.val.dtype);

    // Create input_zp const tensor (matches input dtype)
    let input_zp_name = format!("{}_input_zp", node.name);
    let input_zp_data = encode_zp_for_dtype(input_zp, input_dtype);
    result.tensors.push(TosaTensor {
        name: input_zp_name.clone(),
        shape: vec![1],
        dtype: input_dtype,
        data: input_zp_data,
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

    // Create output_zp const tensor (matches output dtype)
    let output_zp_name = format!("{}_output_zp", node.name);
    let output_zp_data = encode_zp_for_dtype(output_zp, output_dtype);
    result.tensors.push(TosaTensor {
        name: output_zp_name.clone(),
        shape: vec![1],
        dtype: output_dtype,
        data: output_zp_data,
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
            kernel,
            stride,
            pad,
            acc_type,
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

    let axis = match node.args.first() {
        Some(Arg::Node(input_id)) => graph
            .node(*input_id)
            .map(|input_node| map_axis_to_tosa_dim_order(input_node, kwarg_i32(node, "axis", 0)))
            .unwrap_or_else(|| kwarg_i32(node, "axis", 0)),
        _ => kwarg_i32(node, "axis", 0),
    };

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
pub fn emit_tanh(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
pub fn emit_exp(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
pub fn emit_log(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
pub fn emit_abs(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
pub fn emit_ceil(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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

/// Emit a ROUND operator by expanding it into primitive TOSA ops.
///
/// TOSA has no standalone ROUND op. For backend-owned raw graphs, expand:
///   round(x) = select(x >= 0, floor(x + 0.5), ceil(x - 0.5))
/// using filled const tensors that match the output shape.
pub fn emit_round(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let input_name = match input_names.first() {
        Some(name) => name.clone(),
        None => return result,
    };

    let shape = actual_tensor_shape(graph, node);
    let dtype = node.meta.val.dtype;

    let zero_name = format!("{}_round_zero", node.name);
    let half_name = format!("{}_round_half", node.name);
    let ge_name = format!("{}_round_ge", node.name);
    let plus_half_name = format!("{}_round_plus_half", node.name);
    let minus_half_name = format!("{}_round_minus_half", node.name);
    let floor_name = format!("{}_round_floor", node.name);
    let ceil_name = format!("{}_round_ceil", node.name);

    let zero_tensor = make_filled_const_tensor(zero_name.clone(), shape.clone(), dtype, 0.0)
        .unwrap_or_else(|| {
            panic!(
                "emit_round: unsupported zero-fill dtype {:?} for node '{}'",
                dtype, node.name
            )
        });
    let half_tensor = make_filled_const_tensor(half_name.clone(), shape.clone(), dtype, 0.5)
        .unwrap_or_else(|| {
            panic!(
                "emit_round: unsupported half-fill dtype {:?} for node '{}'",
                dtype, node.name
            )
        });

    push_inline_const(&mut result, zero_tensor);
    push_inline_const(&mut result, half_tensor);

    result.tensors.push(TosaTensor {
        name: ge_name.clone(),
        shape: shape.clone(),
        dtype: DType::Bool,
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::GreaterEqual,
        attribute: Attribute::GreaterEqual(GreaterEqualAttribute),
        inputs: vec![input_name.clone(), zero_name],
        outputs: vec![ge_name.clone()],
        location: None,
    });

    result.tensors.push(TosaTensor {
        name: plus_half_name.clone(),
        shape: shape.clone(),
        dtype: ir_dtype_to_tosa(dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Add,
        attribute: Attribute::Add(AddAttribute),
        inputs: vec![input_name.clone(), half_name.clone()],
        outputs: vec![plus_half_name.clone()],
        location: None,
    });

    result.tensors.push(TosaTensor {
        name: minus_half_name.clone(),
        shape: shape.clone(),
        dtype: ir_dtype_to_tosa(dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Sub,
        attribute: Attribute::Sub(SubAttribute),
        inputs: vec![input_name.clone(), half_name],
        outputs: vec![minus_half_name.clone()],
        location: None,
    });

    result.tensors.push(TosaTensor {
        name: floor_name.clone(),
        shape: shape.clone(),
        dtype: ir_dtype_to_tosa(dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Floor,
        attribute: Attribute::Floor(FloorAttribute),
        inputs: vec![plus_half_name],
        outputs: vec![floor_name.clone()],
        location: None,
    });

    result.tensors.push(TosaTensor {
        name: ceil_name.clone(),
        shape: shape.clone(),
        dtype: ir_dtype_to_tosa(dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Ceil,
        attribute: Attribute::Ceil(CeilAttribute),
        inputs: vec![minus_half_name],
        outputs: vec![ceil_name.clone()],
        location: None,
    });

    result.tensors.push(TosaTensor {
        name: node.name.clone(),
        shape,
        dtype: ir_dtype_to_tosa(dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Select,
        attribute: Attribute::Select(SelectAttribute),
        inputs: vec![ge_name, floor_name, ceil_name],
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
pub fn emit_erf(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
pub fn emit_cos(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
pub fn emit_sin(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
pub fn emit_cast(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
        dtype: matmul_input_dtype(graph, node, 0),
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
        dtype: matmul_input_dtype(graph, node, 1),
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
pub fn emit_pad(graph: &Graph, _node_id: NodeId, node: &Node, _store: &TensorStore) -> EmitResult {
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
///
/// TOSA TRANSPOSE_CONV2D requires 5 inputs: [input, weight, bias, input_zp, weight_zp].
/// If the incoming node only provides 3 inputs, we create inline const tensors
/// for the zero-point scalars.
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

    // If we already have 5 inputs, pass them through; otherwise create zp tensors.
    let conv_inputs = if input_names.len() >= 5 {
        input_names
    } else {
        let input_zp_val = kwarg_i32(node, "input_zp", 0);
        let weight_zp_val = kwarg_i32(node, "weight_zp", 0);

        // Create input_zp const tensor (scalar int8)
        let input_zp_name = format!("{}_input_zp", node.name);
        result.tensors.push(TosaTensor {
            name: input_zp_name.clone(),
            shape: vec![1],
            dtype: DType::Int8,
            data: vec![input_zp_val as i8 as u8],
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

        // Match the reference serializer exactly: scalar shape but one byte
        // per output channel in the payload.
        let weight_zp_name = format!("{}_weight_zp", node.name);
        let num_oc = get_weight_out_channels(graph, node).unwrap_or(1);
        let weight_zp_data = make_weight_zp_data(weight_zp_val, num_oc);
        result.tensors.push(TosaTensor {
            name: weight_zp_name.clone(),
            shape: vec![1],
            dtype: DType::Int8,
            data: weight_zp_data,
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });
        result.operators.push(TosaOperator {
            op: Op::Const,
            attribute: Attribute::Const(ConstAttribute),
            inputs: vec![],
            outputs: vec![weight_zp_name.clone()],
            location: None,
        });

        let mut inputs = input_names;
        inputs.truncate(3);
        inputs.push(input_zp_name);
        inputs.push(weight_zp_name);
        inputs
    };

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
        inputs: conv_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });

    result
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

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
        OpTarget::TosaDepthwiseConv2d => Some(emit_depthwise_conv2d(graph, node_id, node, store)),
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
        OpTarget::TosaNeg | OpTarget::TosaNegate => Some(emit_negate(graph, node_id, node, store)),
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
        OpTarget::TosaTransposeConv2d => Some(emit_transpose_conv2d(graph, node_id, node, store)),

        // -----------------------------------------------------------------
        // ATen fallbacks: handle ops that FX-direct passes didn't rewrite
        // to TOSA form. This allows lower_to_tosa_v2 (hybrid pipeline) to
        // produce valid TOSA output even when not all passes run.
        // -----------------------------------------------------------------
        OpTarget::Conv2d | OpTarget::DepthwiseConv2d => {
            Some(emit_aten_conv2d(graph, node_id, node, store))
        }
        OpTarget::MaxPool2d => Some(emit_aten_max_pool2d(graph, node_id, node, store)),
        OpTarget::Clamp | OpTarget::Hardtanh | OpTarget::Relu => {
            Some(emit_aten_clamp(graph, node_id, node, store))
        }
        OpTarget::Reshape | OpTarget::Flatten | OpTarget::View => {
            Some(emit_aten_reshape(graph, node_id, node, store))
        }
        OpTarget::Permute => Some(emit_aten_permute(graph, node_id, node, store)),
        OpTarget::To => Some(emit_cast(graph, node_id, node, store)),
        OpTarget::Add => Some(emit_aten_add(graph, node_id, node, store)),
        OpTarget::Sub => Some(emit_aten_sub(graph, node_id, node, store)),
        OpTarget::Mul => Some(emit_aten_mul(graph, node_id, node, store)),
        OpTarget::AvgPool2d => Some(emit_avg_pool2d(graph, node_id, node, store)),
        OpTarget::Sigmoid => Some(emit_sigmoid(graph, node_id, node, store)),
        OpTarget::Tanh => Some(emit_tanh(graph, node_id, node, store)),
        OpTarget::Exp => Some(emit_exp(graph, node_id, node, store)),
        OpTarget::Log => Some(emit_log(graph, node_id, node, store)),
        OpTarget::Neg => Some(emit_negate(graph, node_id, node, store)),
        OpTarget::Abs => Some(emit_abs(graph, node_id, node, store)),
        OpTarget::Ceil => Some(emit_ceil(graph, node_id, node, store)),
        OpTarget::Floor => Some(emit_floor(graph, node_id, node, store)),
        OpTarget::Round => Some(emit_round(graph, node_id, node, store)),
        OpTarget::Reciprocal => Some(emit_reciprocal(graph, node_id, node, store)),
        OpTarget::Rsqrt => Some(emit_rsqrt(graph, node_id, node, store)),
        OpTarget::Matmul | OpTarget::Bmm => Some(emit_matmul(graph, node_id, node, store)),
        OpTarget::Cat => Some(emit_concat(graph, node_id, node, store)),
        OpTarget::Pad => Some(emit_pad(graph, node_id, node, store)),
        OpTarget::Slice => Some(emit_slice(graph, node_id, node, store)),
        OpTarget::Select => Some(emit_select(graph, node_id, node, store)),
        OpTarget::Sum => Some(emit_aten_sum(graph, node_id, node, store)),
        OpTarget::ConvTranspose2d => Some(emit_transpose_conv2d(graph, node_id, node, store)),
        OpTarget::Other(target) if target.contains("getitem") => Some(EmitResult::default()),

        // Ops not yet handled — skip silently for now.
        _ => {
            if std::env::var("TOSA_ENGINE_DEBUG").is_ok() {
                eprintln!(
                    "[emit_node] SKIPPED unhandled target: {:?} (node '{}')",
                    node.target, node.name
                );
            }
            None
        }
    }
}
