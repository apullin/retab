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
        let (input_zp, weight_zp) = input_weight_zero_points(node, false);
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
        let pad = kwarg_or_arg_i32_list(node, "pad", 4);
        let stride = kwarg_or_arg_i32_list(node, "stride", 3);
        let dilation = kwarg_or_arg_i32_list(node, "dilation", 5);
        let local_bound = kwarg_bool(node, "local_bound", false);
        let acc_type = conv_accumulator_dtype(node);
        let (input_zp, weight_zp) = input_weight_zero_points(node, true);
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

    /// Create zero-point tensors and build the full conv input name list.
    /// If inputs already have 5+ entries (zp already present), returns them as-is.
    fn prepare_inputs(&self, result: &mut EmitResult, graph: &Graph, node: &Node) -> Vec<String> {
        prepare_conv_inputs(
            result,
            graph,
            node,
            self.input_names.clone(),
            self.acc_type,
            self.input_zp,
            self.weight_zp,
        )
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
            | OpTarget::View => weight_node
                .args
                .first()
                .and_then(Arg::as_node)
                .and_then(|id| graph.node(id)),
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

    let Some(input_node) = node
        .args
        .first()
        .and_then(Arg::as_node)
        .and_then(|id| graph.node(id))
    else {
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

    let Some(weight_node) = node
        .args
        .get(1)
        .and_then(Arg::as_node)
        .and_then(|id| graph.node(id))
    else {
        return false;
    };

    let logical_weight = logical_weight_shape(graph, weight_node);
    logical_weight.len() >= 2
        && logical_weight[1] == 1
        && logical_weight[0] > 0
        && logical_weight[0] % in_channels == 0
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
        let raw_pad = symmetric_spatial_pad(&spec.pad, 3);
        let stride = i32_list_or_repeat(spec.stride.clone(), 3, 1);
        let dilation = i32_list_or_repeat(spec.dilation.clone(), 3, 1);
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

    let stride = i32_list_or_repeat(spec.stride.clone(), 2, 1);
    let dilation = i32_list_or_repeat(spec.dilation.clone(), 2, 1);

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
    let raw_pad = i32_list_or_repeat(spec.pad.clone(), 4, 0);
    let stride = i32_list_or_repeat(spec.stride.clone(), 2, 1);
    let dilation = i32_list_or_repeat(spec.dilation.clone(), 2, 1);
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
    let raw_pad = i32_list_or_repeat(spec.pad.clone(), 6, 0);
    let stride = i32_list_or_repeat(spec.stride.clone(), 3, 1);
    let dilation = i32_list_or_repeat(spec.dilation.clone(), 3, 1);
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
    let raw_pad = i32_list_or_repeat(spec.pad.clone(), 4, 0);
    let stride = i32_list_or_repeat(spec.stride.clone(), 2, 1);
    let dilation = i32_list_or_repeat(spec.dilation.clone(), 2, 1);
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
