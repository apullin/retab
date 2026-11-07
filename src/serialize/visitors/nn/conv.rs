use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaConv2d, emit_conv2d as EmitFn),
        (OpTarget::TosaConv3d, emit_conv3d),
        (OpTarget::TosaDepthwiseConv2d, emit_depthwise_conv2d),
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
    /// Parse rc4's TOSA conv arguments, kwargs first with a positional fallback.
    fn parse(ctx: &EmitContext) -> Self {
        let node = ctx.node;
        let input_names = ctx.input_names();
        let pad = kwarg_or_arg_i32_list(node, "pad", 4);
        let stride = kwarg_or_arg_i32_list(node, "stride", 3);
        let dilation = kwarg_or_arg_i32_list(node, "dilation", 5);
        let local_bound = kwarg_bool(node, "local_bound", false);
        let acc_type = conv_accumulator_dtype(node);
        let (input_zp, weight_zp) = input_weight_zero_points(node);
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

// ---------------------------------------------------------------------------
// Emitters
// ---------------------------------------------------------------------------

pub(in crate::serialize::visitors) fn emit_conv2d(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();

    let spec = ParsedConvSpec::parse(ctx);
    // rc4's pad, already adjusted for TOSA's exact division (P2 design G2).
    let pad = i32_list_or_repeat(spec.pad.clone(), 4, 0);
    let stride = i32_list_or_repeat(spec.stride.clone(), 2, 1);
    let dilation = i32_list_or_repeat(spec.dilation.clone(), 2, 1);
    let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

    result.push_output_op(
        conv_output_tensor(node),
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
    let mut result = EmitResult::default();

    let spec = ParsedConvSpec::parse(ctx);
    // rc4's pad, already adjusted for TOSA's exact division (P2 design G2).
    let pad = i32_list_or_repeat(spec.pad.clone(), 6, 0);
    let stride = i32_list_or_repeat(spec.stride.clone(), 3, 1);
    let dilation = i32_list_or_repeat(spec.dilation.clone(), 3, 1);
    let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

    result.push_output_op(
        conv_output_tensor(node),
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
    let mut result = EmitResult::default();

    let spec = ParsedConvSpec::parse(ctx);
    // rc4's pad, already adjusted for TOSA's exact division (P2 design G2).
    let pad = i32_list_or_repeat(spec.pad.clone(), 4, 0);
    let stride = i32_list_or_repeat(spec.stride.clone(), 2, 1);
    let dilation = i32_list_or_repeat(spec.dilation.clone(), 2, 1);
    let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

    result.push_output_op(
        conv_output_tensor(node),
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
