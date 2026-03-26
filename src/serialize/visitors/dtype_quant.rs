use super::*;

pub(crate) fn ir_dtype_to_tosa(dt: crate::graph::ir::DType) -> DType {
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

pub(super) fn node_output_dtype(node: &Node) -> DType {
    if let Some(dt) = node.meta.tosa_special_dtype {
        return ir_dtype_to_tosa(dt);
    }
    if matches!(
        node.target,
        OpTarget::Eq
            | OpTarget::Gt
            | OpTarget::Ge
            | OpTarget::Lt
            | OpTarget::Le
            | OpTarget::TosaEqual
            | OpTarget::TosaGreater
            | OpTarget::TosaGreaterEqual
            | OpTarget::LogicalNot
            | OpTarget::LogicalAnd
            | OpTarget::LogicalOr
            | OpTarget::LogicalXor
    ) {
        return DType::Bool;
    }
    if node.op == OpType::Placeholder {
        return ir_dtype_to_tosa(node.meta.val.dtype);
    }
    if let Some((_, qp)) = node.meta.output_qparams.first() {
        return ir_dtype_to_tosa(qp.dtype);
    }
    ir_dtype_to_tosa(node.meta.val.dtype)
}

pub(super) fn conv_accumulator_dtype(node: &Node) -> DType {
    if let Some(dt) = node.meta.tosa_special_dtype {
        return ir_dtype_to_tosa(dt);
    }
    if node
        .meta
        .input_qparams
        .iter()
        .any(|(_, qp)| qp.dtype == crate::graph::ir::DType::Int16)
    {
        return DType::Int48;
    }
    match node.meta.val.dtype {
        crate::graph::ir::DType::Int32 => DType::Int32,
        crate::graph::ir::DType::Int48 => DType::Int48,
        _ => {
            if !node.meta.input_qparams.is_empty() {
                DType::Int32
            } else {
                ir_dtype_to_tosa(node.meta.val.dtype)
            }
        }
    }
}

pub(super) fn node_tensor_dtype(graph: &Graph, node: &Node, index: usize) -> DType {
    match node.args.get(index) {
        Some(Arg::Node(id)) => graph.node(*id).map(node_output_dtype).unwrap_or(DType::Int32),
        _ => DType::Int32,
    }
}

pub(super) fn passthrough_input_dtype(graph: &Graph, node: &Node) -> DType {
    match node.args.first() {
        Some(Arg::Node(id)) => graph
            .node(*id)
            .map(node_output_dtype)
            .unwrap_or_else(|| node_output_dtype(node)),
        _ => node_output_dtype(node),
    }
}

pub(super) fn accumulator_output_dtype(node: &Node) -> DType {
    if let Some(dt) = node.meta.tosa_special_dtype {
        return ir_dtype_to_tosa(dt);
    }
    if node
        .meta
        .input_qparams
        .iter()
        .any(|(_, qp)| qp.dtype == crate::graph::ir::DType::Int16)
    {
        return DType::Int48;
    }
    match node.meta.val.dtype {
        crate::graph::ir::DType::Int32 => DType::Int32,
        crate::graph::ir::DType::Int48 => DType::Int48,
        _ => {
            if !node.meta.input_qparams.is_empty() {
                DType::Int32
            } else {
                node_output_dtype(node)
            }
        }
    }
}

pub(super) fn actual_tensor_dtype(graph: &Graph, node: &Node) -> DType {
    match node.target {
        OpTarget::Reshape
        | OpTarget::View
        | OpTarget::Permute
        | OpTarget::Flatten
        | OpTarget::Squeeze
        | OpTarget::Unsqueeze
        | OpTarget::Expand
        | OpTarget::Contiguous
        | OpTarget::TosaReshape
        | OpTarget::TosaTranspose => match node.args.first() {
            Some(Arg::Node(id)) => graph
                .node(*id)
                .map(|input| actual_tensor_dtype(graph, input))
                .unwrap_or_else(|| node_output_dtype(node)),
            _ => node_output_dtype(node),
        },
        OpTarget::To => node_output_dtype(node),
        OpTarget::Conv2d
        | OpTarget::DepthwiseConv2d
        | OpTarget::Linear
        | OpTarget::Addmm
        | OpTarget::Matmul
        | OpTarget::Bmm
        | OpTarget::TosaConv2d
        | OpTarget::TosaDepthwiseConv2d
        | OpTarget::TosaTransposeConv2d
        | OpTarget::TosaFullyConnected
        | OpTarget::TosaMatmul => accumulator_output_dtype(node),
        OpTarget::Sum => match passthrough_input_dtype(graph, node) {
            DType::Int8 | DType::Int16 | DType::Int32 => DType::Int32,
            other => other,
        },
        _ => node_output_dtype(node),
    }
}

fn producer_output_zero_point(graph: &Graph, node: &Node, pos: usize) -> Option<i32> {
    let pos = pos as u32;
    let Arg::Node(input_id) = node.args.get(pos as usize)? else {
        return None;
    };
    let input_node = graph.node(*input_id)?;
    input_node
        .meta
        .output_qparams
        .iter()
        .find(|(qp_pos, _)| *qp_pos == 0)
        .or_else(|| input_node.meta.input_qparams.iter().find(|(qp_pos, _)| *qp_pos == 0))
        .map(|(_, qp)| qp.zero_point as i32)
}

pub(super) fn node_input_zero_point(
    graph: &Graph,
    node: &Node,
    pos: usize,
    kwarg_name: &str,
    default: i32,
) -> i32 {
    let pos_u32 = pos as u32;
    producer_output_zero_point(graph, node, pos)
        .or_else(|| {
            node.meta
                .input_qparams
                .iter()
                .find(|(qp_pos, _)| *qp_pos == pos_u32)
                .map(|(_, qp)| qp.zero_point as i32)
        })
        .unwrap_or_else(|| kwarg_i32(node, kwarg_name, default))
}

pub(super) fn node_output_zero_point(node: &Node, kwarg_name: &str, default: i32) -> i32 {
    node.meta
        .output_qparams
        .iter()
        .find(|(pos, _)| *pos == 0)
        .map(|(_, qp)| qp.zero_point as i32)
        .unwrap_or_else(|| kwarg_i32(node, kwarg_name, default))
}

pub(super) fn matmul_input_dtype(graph: &Graph, node: &Node, index: usize) -> DType {
    match node.args.get(index) {
        Some(Arg::Node(id)) => graph
            .node(*id)
            .map(|input_node| ir_dtype_to_tosa(input_node.meta.val.dtype))
            .unwrap_or(DType::Int32),
        _ => DType::Int32,
    }
}

pub(super) fn has_only_tosa_rescale_users(graph: &Graph, node: &Node) -> bool {
    !node.users.is_empty()
        && node.users.iter().all(|user_id| {
            graph
                .node(*user_id)
                .map(|user| user.target == OpTarget::TosaRescale)
                .unwrap_or(false)
        })
}

pub(super) fn normalize_rescale_zp(dtype: DType, zp: i32, is_unsigned: bool) -> i32 {
    match dtype {
        DType::Int8 => zp,
        DType::Int16 if is_unsigned => zp,
        _ => 0,
    }
}

pub(super) fn get_weight_out_channels(graph: &Graph, node: &Node) -> Option<usize> {
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

pub(super) fn encode_zp_for_dtype(zp_val: i32, dtype: DType) -> Vec<u8> {
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

pub(super) fn make_weight_zp_data(zp_val: i32, num_channels: usize, dtype: DType) -> Vec<u8> {
    let encoded = encode_zp_for_dtype(zp_val, dtype);
    let mut out = Vec::with_capacity(encoded.len() * num_channels);
    for _ in 0..num_channels {
        out.extend_from_slice(&encoded);
    }
    out
}

pub(super) fn default_clamp_bound_bytes(dtype: crate::graph::ir::DType, is_min: bool) -> Vec<u8> {
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

fn clamp_bound_bytes_from_scalar(dtype: crate::graph::ir::DType, value: f64) -> Option<Vec<u8>> {
    use crate::graph::ir::DType as IrDtype;

    Some(match dtype {
        IrDtype::Bool => vec![(value != 0.0) as u8],
        IrDtype::Int4 | IrDtype::Int8 => (value as i8).to_le_bytes().to_vec(),
        IrDtype::Int16 => (value as i16).to_le_bytes().to_vec(),
        IrDtype::Int32 => (value as i32).to_le_bytes().to_vec(),
        IrDtype::Int64 => (value as i64).to_le_bytes().to_vec(),
        IrDtype::Int48 => {
            let bytes = (value as i64).to_le_bytes();
            bytes[..6].to_vec()
        }
        IrDtype::Float32 => (value as f32).to_le_bytes().to_vec(),
        IrDtype::Float16 | IrDtype::BFloat16 => return None,
    })
}

pub(super) fn clamp_bound_bytes_from_arg(
    dtype: crate::graph::ir::DType,
    arg: Option<&Arg>,
    is_min: bool,
) -> Vec<u8> {
    match arg {
        Some(Arg::Int(i)) => {
            clamp_bound_bytes_from_scalar(dtype, *i as f64)
                .unwrap_or_else(|| default_clamp_bound_bytes(dtype, is_min))
        }
        Some(Arg::Float(f)) => {
            clamp_bound_bytes_from_scalar(dtype, *f)
                .unwrap_or_else(|| default_clamp_bound_bytes(dtype, is_min))
        }
        Some(Arg::Bool(b)) => clamp_bound_bytes_from_scalar(dtype, if *b { 1.0 } else { 0.0 })
            .unwrap_or_else(|| default_clamp_bound_bytes(dtype, is_min)),
        Some(Arg::IntList(list)) => {
            let bytes: Option<Vec<u8>> = list
                .iter()
                .map(|&value| clamp_bound_bytes_from_scalar(dtype, value as f64))
                .collect::<Option<Vec<Vec<u8>>>>()
                .map(|chunks| chunks.into_iter().flatten().collect());
            bytes.unwrap_or_else(|| default_clamp_bound_bytes(dtype, is_min))
        }
        _ => default_clamp_bound_bytes(dtype, is_min),
    }
}

pub(super) fn compute_multiplier_and_shift_with_width(scale: f64, scale_width: i32) -> (i32, i8) {
    if scale == 0.0 {
        return (0, 0);
    }

    let bits = scale.to_bits();
    let biased_exp = ((bits >> 52) & 0x7FF) as i32;
    let exponent = biased_exp - 1022;
    let mantissa = f64::from_bits((bits & 0x800F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000);

    let offset = if scale_width == 16 { 15 } else { 31 };
    let const_2_pow_offset: i64 = 1_i64 << offset;
    let mut shifted_mantissa = (mantissa * const_2_pow_offset as f64).round() as i64;
    let mut shift = exponent;

    if shifted_mantissa == const_2_pow_offset {
        shifted_mantissa /= 2;
        shift += 1;
    }

    let right_shift = offset - shift;

    let mut multiplier = shifted_mantissa as i32;
    let mut final_shift = right_shift;
    if final_shift > 62 {
        multiplier >>= (final_shift - 62).min(31);
        final_shift = 62;
    }

    (multiplier, final_shift as i8)
}
