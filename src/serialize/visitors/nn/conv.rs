use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaConv2d, emit_conv2d as EmitFn),
        (OpTarget::TosaConv3d, emit_conv3d),
        (OpTarget::TosaDepthwiseConv2d, emit_depthwise_conv2d),
        (OpTarget::Conv2d, emit_aten_conv2d),
        (OpTarget::DepthwiseConv2d, emit_aten_conv2d),
    ]);
}

// ---------------------------------------------------------------------------
// Shared infrastructure for all conv emitters
// ---------------------------------------------------------------------------

struct ParsedConvSpec {
    input_names: Vec<String>,
    pad: Vec<i32>,
    stride: Vec<i32>,
    dilation: Vec<i32>,
    input_zp: i32,
    weight_zp: i32,
    acc_type: DType,
    local_bound: bool,
}

impl ParsedConvSpec {
    /// Parse from aten-style positional args (used by `emit_aten_conv2d`).
    fn parse_aten(ctx: &EmitContext) -> Self {
        let node = ctx.node;
        let input_names = ctx.input_names();
        let stride = arg_i32_list(node, 3);
        let dilation = arg_i32_list(node, 5);
        let acc_type = conv_accumulator_dtype(node);
        let (input_zp, weight_zp) = Self::extract_zp_from_qparams(node);
        // For aten, pad is computed externally via adjusted_tosa_pad_for_raw_aten_conv,
        // so we store the raw padding arg here for the caller to use.
        let pad = arg_i32_list(node, 4);
        Self {
            input_names,
            pad,
            stride,
            dilation,
            input_zp,
            weight_zp,
            acc_type,
            local_bound: false,
        }
    }

    /// Parse from TOSA-style kwargs-first, positional-fallback args
    /// (used by `emit_conv2d`, `emit_conv3d`, `emit_depthwise_conv2d`).
    fn parse_tosa(ctx: &EmitContext) -> Self {
        let node = ctx.node;
        let input_names = ctx.input_names();
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
        let acc_type = conv_accumulator_dtype(node);
        let (input_zp, weight_zp) = Self::extract_zp_tosa(node);
        Self {
            input_names,
            pad,
            stride,
            dilation,
            input_zp,
            weight_zp,
            acc_type,
            local_bound,
        }
    }

    /// Extract zero-point values from input_qparams (aten path -- no kwarg fallback).
    fn extract_zp_from_qparams(node: &Node) -> (i32, i32) {
        let input_zp = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or(0);
        let weight_zp = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 1)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or(0);
        (input_zp, weight_zp)
    }

    /// Extract zero-point values with kwarg fallback (tosa path).
    fn extract_zp_tosa(node: &Node) -> (i32, i32) {
        let input_zp = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "input_zp", 0));
        let weight_zp = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 1)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or_else(|| kwarg_i32(node, "weight_zp", 0));
        (input_zp, weight_zp)
    }

    /// Create zero-point tensors and build the full conv input name list.
    /// If inputs already have 5+ entries (zp already present), returns them as-is.
    fn prepare_inputs(&self, result: &mut EmitResult, graph: &Graph, node: &Node) -> Vec<String> {
        let mut input_names = self.input_names.clone();
        if matches!(input_names.len(), 2 | 4) {
            input_names.insert(
                2,
                push_zero_bias_tensor(result, node, synthetic_bias_dtype(node, self.acc_type)),
            );
        }
        if input_names.len() >= 5 {
            return input_names;
        }
        let (izp, wzp) =
            push_input_weight_zp_tensors(result, graph, node, self.input_zp, self.weight_zp);
        extend_conv_inputs(input_names, izp, wzp)
    }

    /// Return stride with a default fill value when empty.
    fn stride_or(&self, default: Vec<i32>) -> Vec<i32> {
        if self.stride.is_empty() {
            default
        } else {
            self.stride.clone()
        }
    }

    /// Return dilation with a default fill value when empty.
    fn dilation_or(&self, default: Vec<i32>) -> Vec<i32> {
        if self.dilation.is_empty() {
            default
        } else {
            self.dilation.clone()
        }
    }

    /// Return pad with a default fill value when empty.
    fn pad_or(&self, default: Vec<i32>) -> Vec<i32> {
        if self.pad.is_empty() {
            default
        } else {
            self.pad.clone()
        }
    }
}

fn synthetic_bias_dtype(node: &Node, acc_type: DType) -> DType {
    let semantic_dtype = node_output_dtype(node);
    if node.meta.input_qparams.is_empty()
        && matches!(semantic_dtype, DType::Fp32 | DType::Fp16 | DType::Bf16)
    {
        semantic_dtype
    } else {
        acc_type
    }
}

fn output_channels(node: &Node) -> i32 {
    node.meta
        .val
        .shape_slice()
        .get(1)
        .copied()
        .or_else(|| node.meta.val.shape_slice().first().copied())
        .unwrap_or(0) as i32
}

fn zero_bytes(dtype: DType, len: usize) -> Vec<u8> {
    let bytes_per_elem = match dtype {
        DType::Bool | DType::Int4 | DType::Int8 => 1,
        DType::Int16 | DType::Fp16 | DType::Bf16 => 2,
        DType::Int32 | DType::Fp32 => 4,
        DType::Int48 => 6,
        DType::Int64 => 8,
        DType::Shape => 8,
        _ => 4,
    };
    vec![0; bytes_per_elem * len]
}

fn push_zero_bias_tensor(result: &mut EmitResult, node: &Node, dtype: DType) -> String {
    let channels = output_channels(node).max(0) as usize;
    result.push_const(TosaTensor::with_data(
        &format!("{}_bias", node.name),
        vec![channels as i32],
        dtype,
        zero_bytes(dtype, channels),
    ))
}

fn input_rank(graph: &Graph, node: &Node) -> usize {
    match node.args.first() {
        Some(Arg::Node(id)) => graph
            .node(*id)
            .map(|n| n.meta.val.shape_len as usize)
            .unwrap_or(0),
        _ => 0,
    }
}

fn logical_weight_shape<'a>(graph: &'a Graph, mut weight_node: &'a Node) -> &'a [i64] {
    for _ in 0..4 {
        let shape = weight_node.meta.val.shape_slice();
        if shape.len() == 4 && shape.get(1).copied() == Some(1) {
            return shape;
        }
        let next = match weight_node.target {
            OpTarget::TosaTranspose
            | OpTarget::Permute
            | OpTarget::Reshape
            | OpTarget::TosaReshape
            | OpTarget::View => weight_node.args.first().and_then(|arg| match arg {
                Arg::Node(id) => graph.node(*id),
                _ => None,
            }),
            _ => None,
        };
        let Some(next) = next else {
            return shape;
        };
        weight_node = next;
    }
    weight_node.meta.val.shape_slice()
}

fn is_depthwise_aten_conv2d(graph: &Graph, store: &TensorStore, node: &Node, groups: i64) -> bool {
    if let Some(conv_info_entry) = node
        .kwargs
        .iter()
        .find(|(k, _)| k == "conv_info_is_depthwise")
    {
        return matches!(&conv_info_entry.1, Arg::Int(v) if *v != 0);
    }

    let Some(input_node) = node.args.first().and_then(|arg| match arg {
        Arg::Node(id) => graph.node(*id),
        _ => None,
    }) else {
        return false;
    };

    let input_shape = actual_tensor_shape(graph, store, input_node);
    if input_shape.len() != 4 || groups <= 0 {
        return false;
    }

    let in_channels = input_shape.last().copied().map(i64::from).unwrap_or(0);
    if in_channels <= 0 || in_channels != groups {
        return false;
    }

    let Some(weight_node) = node.args.get(1).and_then(|arg| match arg {
        Arg::Node(id) => graph.node(*id),
        _ => None,
    }) else {
        return false;
    };

    let logical_weight = logical_weight_shape(graph, weight_node);
    logical_weight.len() >= 2
        && logical_weight[1] == 1
        && logical_weight[0] > 0
        && logical_weight[0] % in_channels == 0
}

fn aten_conv_groups(node: &Node) -> i64 {
    let groups_arg = if node.args.len() > 8 { 8 } else { 6 };
    arg_i64(node, groups_arg, 1)
}

fn symmetric_tosa_pad(padding: &[i32], spatial_rank: usize) -> Vec<i32> {
    if padding.len() == spatial_rank * 2 {
        return padding.to_vec();
    }
    let mut pad = Vec::with_capacity(spatial_rank * 2);
    for axis in 0..spatial_rank {
        let value = padding
            .get(axis)
            .copied()
            .or_else(|| padding.first().copied())
            .unwrap_or(0);
        pad.push(value);
        pad.push(value);
    }
    pad
}

// ---------------------------------------------------------------------------
// Emitters
// ---------------------------------------------------------------------------

pub(in crate::serialize::visitors) fn emit_aten_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();

    let spec = ParsedConvSpec::parse_aten(ctx);
    if input_rank(graph, node) == 5 {
        let raw_pad = symmetric_tosa_pad(&spec.pad, 3);
        let stride = spec.stride_or(vec![1, 1, 1]);
        let dilation = spec.dilation_or(vec![1, 1, 1]);
        let pad =
            adjusted_tosa_pad_for_emitted_conv3d(graph, store, node, &raw_pad, &stride, &dilation);
        let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

        result.push_output_op(
            compute_conv3d_ndhwc_output(graph, store, node, &pad, &stride, &dilation),
            Op::Conv3d,
            Attribute::Conv3d(Conv3dAttribute {
                pad,
                stride,
                dilation,
                local_bound: false,
                acc_type: spec.acc_type,
            }),
            conv_inputs,
        );
        return result;
    }

    let groups = aten_conv_groups(node);
    let pad = adjusted_tosa_pad_for_raw_aten_conv(
        graph,
        store,
        node,
        &spec.pad,
        &spec.stride,
        &spec.dilation,
    );
    let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

    let is_depthwise = is_depthwise_aten_conv2d(graph, store, node, groups);

    let stride = spec.stride_or(vec![1, 1]);
    let dilation = spec.dilation_or(vec![1, 1]);

    if is_depthwise {
        result.push_output_op(
            compute_depthwise_conv2d_nhwc_output(graph, store, node, &pad, &stride, &dilation),
            Op::DepthwiseConv2d,
            Attribute::DepthwiseConv2d(DepthwiseConv2dAttribute {
                pad,
                stride,
                dilation,
                local_bound: false,
                acc_type: spec.acc_type,
            }),
            conv_inputs,
        );
    } else {
        result.push_output_op(
            compute_conv2d_nhwc_output(graph, store, node, &pad, &stride, &dilation),
            Op::Conv2d,
            Attribute::Conv2d(Conv2dAttribute {
                pad,
                stride,
                dilation,
                local_bound: false,
                acc_type: spec.acc_type,
            }),
            conv_inputs,
        );
    }

    result
}

pub(in crate::serialize::visitors) fn emit_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();

    let spec = ParsedConvSpec::parse_tosa(ctx);
    let raw_pad = spec.pad_or(vec![0, 0, 0, 0]);
    let stride = spec.stride_or(vec![1, 1]);
    let dilation = spec.dilation_or(vec![1, 1]);
    let pad =
        adjusted_tosa_pad_for_emitted_conv2d(graph, store, node, &raw_pad, &stride, &dilation);
    let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

    result.push_output_op(
        compute_conv2d_nhwc_output(graph, store, node, &pad, &stride, &dilation),
        Op::Conv2d,
        Attribute::Conv2d(Conv2dAttribute {
            pad,
            stride,
            dilation,
            local_bound: spec.local_bound,
            acc_type: spec.acc_type,
        }),
        conv_inputs,
    );
    result
}

pub(in crate::serialize::visitors) fn emit_conv3d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();

    let spec = ParsedConvSpec::parse_tosa(ctx);
    let raw_pad = spec.pad_or(vec![0, 0, 0, 0, 0, 0]);
    let stride = spec.stride_or(vec![1, 1, 1]);
    let dilation = spec.dilation_or(vec![1, 1, 1]);
    let pad =
        adjusted_tosa_pad_for_emitted_conv3d(graph, store, node, &raw_pad, &stride, &dilation);
    let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

    result.push_output_op(
        compute_conv3d_ndhwc_output(graph, store, node, &pad, &stride, &dilation),
        Op::Conv3d,
        Attribute::Conv3d(Conv3dAttribute {
            pad,
            stride,
            dilation,
            local_bound: spec.local_bound,
            acc_type: spec.acc_type,
        }),
        conv_inputs,
    );
    result
}

pub(in crate::serialize::visitors) fn emit_depthwise_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();

    let spec = ParsedConvSpec::parse_tosa(ctx);
    let raw_pad = spec.pad_or(vec![0, 0, 0, 0]);
    let stride = spec.stride_or(vec![1, 1]);
    let dilation = spec.dilation_or(vec![1, 1]);
    let pad =
        adjusted_tosa_pad_for_emitted_conv2d(graph, store, node, &raw_pad, &stride, &dilation);
    let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

    result.push_output_op(
        compute_depthwise_conv2d_nhwc_output(graph, store, node, &pad, &stride, &dilation),
        Op::DepthwiseConv2d,
        Attribute::DepthwiseConv2d(DepthwiseConv2dAttribute {
            pad,
            stride,
            dilation,
            local_bound: spec.local_bound,
            acc_type: spec.acc_type,
        }),
        conv_inputs,
    );
    result
}
