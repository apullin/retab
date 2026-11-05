use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, WIPOffset};

use super::tosa_types::Attribute;

pub(super) fn build_attribute(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    is_u55_subset: bool,
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
        | Attribute::LogicalNot(_)
        | Attribute::LogicalAnd(_)
        | Attribute::LogicalOr(_)
        | Attribute::LogicalXor(_)
        | Attribute::Pow(_)
        | Attribute::IntDiv(_) => empty_attr(builder, attr),

        // tosa-rs currently drops the schema's round field.
        Attribute::ArithmeticRightShift(_) => finish_attr(builder, attr, |builder| {
            builder.push_slot::<bool>(4, is_u55_subset, false);
        }),

        Attribute::Conv2d(a) => conv_attr(
            builder,
            attr,
            &a.pad,
            &a.stride,
            &a.dilation,
            a.local_bound,
            a.acc_type as u32,
        ),
        Attribute::Conv3d(a) => conv_attr(
            builder,
            attr,
            &a.pad,
            &a.stride,
            &a.dilation,
            a.local_bound,
            a.acc_type as u32,
        ),
        Attribute::DepthwiseConv2d(a) => conv_attr(
            builder,
            attr,
            &a.pad,
            &a.stride,
            &a.dilation,
            a.local_bound,
            a.acc_type as u32,
        ),

        Attribute::Rescale(a) => finish_attr(builder, attr, |builder| {
            builder.push_slot::<bool>(4, a.scale32, false);
            builder.push_slot::<u32>(6, a.rounding_mode as u32, 0);
            builder.push_slot::<bool>(8, a.per_channel, false);
            builder.push_slot::<bool>(10, a.input_unsigned, false);
            builder.push_slot::<bool>(12, a.output_unsigned, false);
        }),

        Attribute::Resize(a) => scalar_u32_attr(builder, attr, a.mode as u32),

        Attribute::MaxPool2d(a) => pool_attr(
            builder,
            attr,
            &a.kernel,
            &a.stride,
            &a.pad,
            a.nan_mode as u32,
        ),

        Attribute::Transpose(a) => vector_i32_attr(builder, attr, &a.perms),

        Attribute::Clamp(a) => clamp_attr(builder, attr, &a.min_val, &a.max_val, a.nan_mode as u32),

        Attribute::Mul(a) => scalar_i32_attr(builder, attr, a.shift),

        Attribute::AvgPool2d(a) => pool_attr(
            builder,
            attr,
            &a.kernel,
            &a.stride,
            &a.pad,
            a.acc_type as u32,
        ),

        Attribute::ReduceSum(a) => scalar_i32_attr(builder, attr, a.axis),
        Attribute::Concat(a) => scalar_i32_attr(builder, attr, a.axis),
        Attribute::Reverse(a) => scalar_i32_attr(builder, attr, a.axis),

        Attribute::TransposeConv2d(a) => transpose_conv_attr(
            builder,
            attr,
            &a.out_pad,
            &a.stride,
            a.local_bound,
            a.acc_type as u32,
        ),

        Attribute::CondIf(a) => string_pair_attr(builder, attr, &a.then_graph, &a.else_graph),

        Attribute::WhileLoop(a) => string_pair_attr(builder, attr, &a.cond_branch, &a.body_branch),

        Attribute::Maximum(a) => scalar_u32_attr(builder, attr, a.nan_mode as u32),
        Attribute::Minimum(a) => scalar_u32_attr(builder, attr, a.nan_mode as u32),

        Attribute::ReduceAny(a) => scalar_i32_attr(builder, attr, a.axis),

        Attribute::ReduceMax(a) => axis_nan_mode_attr(builder, attr, a.axis, a.nan_mode as u32),
        Attribute::ArgMax(a) => axis_nan_mode_attr(builder, attr, a.axis, a.nan_mode as u32),

        Attribute::ReduceMin(a) => axis_nan_mode_attr(builder, attr, a.axis, a.nan_mode as u32),

        other => unimplemented!("build_attribute: unsupported attribute variant {:?}", other),
    }
}

fn finish_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    fill: impl FnOnce(&mut FlatBufferBuilder<'_>),
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let start = builder.start_table();
    fill(builder);
    let off = builder.end_table(start);
    (attr.union_type(), Some(off))
}

fn empty_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    finish_attr(builder, attr, |_| {})
}

fn conv_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
    local_bound: bool,
    acc_type: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let pad = builder.create_vector(pad);
    let stride = builder.create_vector(stride);
    let dilation = builder.create_vector(dilation);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, pad);
        builder.push_slot_always::<WIPOffset<_>>(6, stride);
        builder.push_slot_always::<WIPOffset<_>>(8, dilation);
        builder.push_slot::<bool>(10, local_bound, false);
        builder.push_slot::<u32>(12, acc_type, 0);
    })
}

fn pool_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    kernel: &[i32],
    stride: &[i32],
    pad: &[i32],
    field10: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let kernel = builder.create_vector(kernel);
    let stride = builder.create_vector(stride);
    let pad = builder.create_vector(pad);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, kernel);
        builder.push_slot_always::<WIPOffset<_>>(6, stride);
        builder.push_slot_always::<WIPOffset<_>>(8, pad);
        builder.push_slot::<u32>(10, field10, 0);
    })
}

fn scalar_i32_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    value: i32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    finish_attr(builder, attr, |builder| {
        builder.push_slot::<i32>(4, value, 0);
    })
}

fn scalar_u32_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    value: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    finish_attr(builder, attr, |builder| {
        builder.push_slot::<u32>(4, value, 0);
    })
}

fn vector_i32_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    values: &[i32],
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let values = builder.create_vector(values);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, values);
    })
}

fn axis_nan_mode_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    axis: i32,
    nan_mode: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    finish_attr(builder, attr, |builder| {
        builder.push_slot::<i32>(4, axis, 0);
        builder.push_slot::<u32>(6, nan_mode, 0);
    })
}

fn transpose_conv_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    out_pad: &[i32],
    stride: &[i32],
    local_bound: bool,
    acc_type: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let out_pad = builder.create_vector(out_pad);
    let stride = builder.create_vector(stride);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, out_pad);
        builder.push_slot_always::<WIPOffset<_>>(6, stride);
        builder.push_slot::<bool>(8, local_bound, false);
        builder.push_slot::<u32>(10, acc_type, 0);
    })
}

fn string_pair_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    first: &str,
    second: &str,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let first = builder.create_string(first);
    let second = builder.create_string(second);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, first);
        builder.push_slot_always::<WIPOffset<_>>(6, second);
    })
}

fn clamp_attr(
    builder: &mut FlatBufferBuilder<'_>,
    attr: &Attribute,
    min_val: &[u8],
    max_val: &[u8],
    nan_mode: u32,
) -> (u8, Option<WIPOffset<TableFinishedWIPOffset>>) {
    let min_val = create_u8_vector(builder, min_val);
    let max_val = create_u8_vector(builder, max_val);
    finish_attr(builder, attr, |builder| {
        builder.push_slot_always::<WIPOffset<_>>(4, min_val);
        builder.push_slot_always::<WIPOffset<_>>(6, max_val);
        builder.push_slot::<u32>(8, nan_mode, 0);
    })
}

fn create_u8_vector<'fbb>(
    builder: &mut FlatBufferBuilder<'fbb>,
    data: &[u8],
) -> WIPOffset<flatbuffers::Vector<'fbb, u8>> {
    builder.create_vector(data)
}
