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
            if kw.is_empty() { arg_i32_list(node, 4) } else { kw }
        };
        let stride = {
            let kw = kwarg_i32_list(node, "stride");
            if kw.is_empty() { arg_i32_list(node, 3) } else { kw }
        };
        let dilation = {
            let kw = kwarg_i32_list(node, "dilation");
            if kw.is_empty() { arg_i32_list(node, 5) } else { kw }
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
    fn prepare_inputs(
        &self,
        result: &mut EmitResult,
        graph: &Graph,
        node: &Node,
    ) -> Vec<String> {
        if self.input_names.len() >= 5 {
            return self.input_names.clone();
        }
        let (izp, wzp) =
            push_input_weight_zp_tensors(result, graph, node, self.input_zp, self.weight_zp);
        extend_conv_inputs(self.input_names.clone(), izp, wzp)
    }

    /// Return stride with a default fill value when empty.
    fn stride_or(&self, default: Vec<i32>) -> Vec<i32> {
        if self.stride.is_empty() { default } else { self.stride.clone() }
    }

    /// Return dilation with a default fill value when empty.
    fn dilation_or(&self, default: Vec<i32>) -> Vec<i32> {
        if self.dilation.is_empty() { default } else { self.dilation.clone() }
    }

    /// Return pad with a default fill value when empty.
    fn pad_or(&self, default: Vec<i32>) -> Vec<i32> {
        if self.pad.is_empty() { default } else { self.pad.clone() }
    }
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
    let groups = arg_i64(node, 8, 1);
    let pad = adjusted_tosa_pad_for_raw_aten_conv(
        graph, store, node, &spec.pad, &spec.stride, &spec.dilation,
    );
    let conv_inputs = spec.prepare_inputs(&mut result, graph, node);

    let is_depthwise = if let Some(conv_info_entry) = node
        .kwargs
        .iter()
        .find(|(k, _)| k == "conv_info_is_depthwise")
    {
        matches!(&conv_info_entry.1, Arg::Int(v) if *v != 0)
    } else if groups > 1 {
        let in_channels = if let Some(Arg::Node(input_id)) = node.args.first() {
            graph
                .node(*input_id)
                .map(|n| {
                    let s = n.meta.val.shape_slice();
                    if s.len() >= 2 { s[1] } else { 0 }
                })
                .unwrap_or(0)
        } else {
            0
        };
        groups == in_channels
    } else {
        false
    };

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
    let pad = adjusted_tosa_pad_for_emitted_conv2d(graph, store, node, &raw_pad, &stride, &dilation);
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
    let pad = adjusted_tosa_pad_for_emitted_conv3d(graph, store, node, &raw_pad, &stride, &dilation);
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
    let pad = adjusted_tosa_pad_for_emitted_conv2d(graph, store, node, &raw_pad, &stride, &dilation);
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
