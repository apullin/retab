use super::*;
use crate::serialize::op_semantics;
use std::cell::RefCell;
use std::collections::HashMap as StdHashMap;

thread_local! {
    static DTYPE_CACHE: RefCell<StdHashMap<String, DType>> = RefCell::new(StdHashMap::new());
}

pub(crate) fn clear_dtype_cache() {
    DTYPE_CACHE.with(|c| c.borrow_mut().clear());
}

pub(crate) fn ir_dtype_to_tosa(dt: crate::graph::ir::DType) -> DType {
    use crate::graph::ir::DType as IrDtype;
    match dt {
        IrDtype::Float32 => DType::Fp32,
        IrDtype::Float16 => DType::Fp16,
        IrDtype::BFloat16 => DType::Bf16,
        IrDtype::Float8E4M3 => DType::Fp8E4M3,
        IrDtype::Float8E5M2 => DType::Fp8E5M2,
        IrDtype::Int8 => DType::Int8,
        IrDtype::Int16 => DType::Int16,
        IrDtype::Int32 => DType::Int32,
        IrDtype::Int64 => DType::Int64,
        IrDtype::Bool => DType::Bool,
        IrDtype::Int4 => DType::Int4,
        IrDtype::Int48 => DType::Int48,
    }
}

fn tosa_float_dtype_to_ir(dtype: DType) -> Option<crate::graph::ir::DType> {
    use crate::graph::ir::DType as IrDtype;
    Some(match dtype {
        DType::Fp32 => IrDtype::Float32,
        DType::Fp16 => IrDtype::Float16,
        DType::Bf16 => IrDtype::BFloat16,
        DType::Fp8E4M3 => IrDtype::Float8E4M3,
        DType::Fp8E5M2 => IrDtype::Float8E5M2,
        _ => return None,
    })
}

pub(super) fn tosa_float_dtype_to_ir_or_float32(dtype: DType) -> crate::graph::ir::DType {
    tosa_float_dtype_to_ir(dtype).unwrap_or(crate::graph::ir::DType::Float32)
}

pub(super) fn ir_float_dtype_or_float32(dtype: crate::graph::ir::DType) -> crate::graph::ir::DType {
    if dtype.is_float() {
        dtype
    } else {
        crate::graph::ir::DType::Float32
    }
}

pub(super) fn tosa_dtype_storage_size(dtype: DType) -> usize {
    match dtype {
        DType::Bool | DType::Int4 | DType::Int8 | DType::Fp8E4M3 | DType::Fp8E5M2 => 1,
        DType::Int16 | DType::Fp16 | DType::Bf16 => 2,
        DType::Int32 | DType::Fp32 => 4,
        DType::Int48 => 6,
        DType::Int64 | DType::Shape => 8,
        _ => 4,
    }
}

pub(super) fn node_output_dtype(node: &Node) -> DType {
    if let Some(dt) = node.meta.tosa_special_dtype {
        return ir_dtype_to_tosa(dt);
    }
    if node.target == OpTarget::TosaTable {
        return ir_dtype_to_tosa(node.meta.val.dtype);
    }
    if op_semantics::has_bool_output(&node.target) {
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

pub(super) fn input_qparam_zero_point(node: &Node, pos: usize) -> Option<i32> {
    let pos_u32 = pos as u32;
    node.meta
        .input_qparams
        .iter()
        .find(|(qp_pos, _)| *qp_pos == pos_u32)
        .map(|(_, qp)| qp.zero_point as i32)
}

pub(super) fn input_qparam_zero_point_or_kwarg(
    node: &Node,
    pos: usize,
    kwarg_name: &str,
    default: i32,
) -> i32 {
    input_qparam_zero_point(node, pos).unwrap_or_else(|| kwarg_i32(node, kwarg_name, default))
}

pub(super) fn output_qparam_zero_point(node: &Node) -> Option<i32> {
    node.meta
        .output_qparams
        .iter()
        .find(|(pos, _)| *pos == 0)
        .map(|(_, qp)| qp.zero_point as i32)
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
        dtype if dtype.is_narrow_float() => {
            if node.meta.input_qparams.is_empty() {
                DType::Fp32
            } else {
                ir_dtype_to_tosa(node.meta.val.dtype)
            }
        }
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
        Some(Arg::Node(id)) => graph
            .node(*id)
            .map(node_output_dtype)
            .unwrap_or(DType::Int32),
        _ => DType::Int32,
    }
}

pub(super) fn passthrough_input_dtype(graph: &Graph, node: &Node) -> DType {
    match node.args.first() {
        Some(Arg::Node(id)) => graph
            .node(*id)
            .map(|input| actual_tensor_dtype(graph, input))
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
    if let Some(cached) = DTYPE_CACHE.with(|c| c.borrow().get(&node.name).cloned()) {
        return cached;
    }
    let result = actual_tensor_dtype_impl(graph, node);
    DTYPE_CACHE.with(|c| c.borrow_mut().insert(node.name.clone(), result));
    result
}

fn actual_tensor_dtype_impl(graph: &Graph, node: &Node) -> DType {
    if matches!(node.target, OpTarget::Sum)
        && node.meta.val.dtype.is_float() {
            return ir_dtype_to_tosa(node.meta.val.dtype);
        }
    let explicit_layout = graph.is_explicit_layout();
    // An explicit-mode TOSA RESIZE inherits the upsample's int8 output qparam, but an int8
    // bilinear RESIZE produces int32 (INT48 when marked), which its RESCALE reads (P2
    // critique A8/B4).
    if explicit_layout && node.target == OpTarget::TosaResize {
        return node
            .meta
            .tosa_special_dtype
            .map(ir_dtype_to_tosa)
            .unwrap_or_else(|| ir_dtype_to_tosa(node.meta.val.dtype));
    }
    match node.target {
        ref target if op_semantics::is_layout_dtype_passthrough(target) => {
            match node.args.first() {
                Some(Arg::Node(id)) => graph
                    .node(*id)
                    .map(|input| actual_tensor_dtype(graph, input))
                    .unwrap_or_else(|| node_output_dtype(node)),
                _ => node_output_dtype(node),
            }
        }
        OpTarget::To => node_output_dtype(node),
        // TOSA CONV3D only reaches the serializer in explicit mode (P2 design X1).
        ref target
            if op_semantics::is_accumulator_output(target)
                || (explicit_layout && *target == OpTarget::TosaConv3d) =>
        {
            accumulator_output_dtype(node)
        }
        OpTarget::Sum => {
            let semantic_dtype = node_output_dtype(node);
            if is_float_dtype(semantic_dtype) {
                semantic_dtype
            } else {
                match passthrough_input_dtype(graph, node) {
                    DType::Int8 | DType::Int16 | DType::Int32 => DType::Int32,
                    other => other,
                }
            }
        }
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
        .or_else(|| {
            input_node
                .meta
                .input_qparams
                .iter()
                .find(|(qp_pos, _)| *qp_pos == 0)
        })
        .map(|(_, qp)| qp.zero_point as i32)
}

pub(super) fn node_input_zero_point(
    graph: &Graph,
    node: &Node,
    pos: usize,
    kwarg_name: &str,
    default: i32,
) -> i32 {
    producer_output_zero_point(graph, node, pos)
        .or_else(|| input_qparam_zero_point(node, pos))
        .unwrap_or_else(|| kwarg_i32(node, kwarg_name, default))
}

pub(super) fn node_output_zero_point(node: &Node, kwarg_name: &str, default: i32) -> i32 {
    output_qparam_zero_point(node).unwrap_or_else(|| kwarg_i32(node, kwarg_name, default))
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
        IrDtype::Float8E4M3 => vec![if is_min { 0xfe } else { 0x7e }],
        IrDtype::Float8E5M2 => vec![if is_min { 0xfb } else { 0x7b }],
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
        IrDtype::Float16 | IrDtype::BFloat16 => return scalar_fill_bytes(dtype, value),
        IrDtype::Float8E4M3 | IrDtype::Float8E5M2 => return None,
    })
}

pub(super) fn clamp_bound_bytes_from_arg(
    dtype: crate::graph::ir::DType,
    arg: Option<&Arg>,
    is_min: bool,
) -> Vec<u8> {
    match arg {
        Some(arg @ (Arg::Int(_) | Arg::Float(_) | Arg::Bool(_))) => {
            clamp_bound_bytes_from_scalar(dtype, arg.as_f64())
                .unwrap_or_else(|| default_clamp_bound_bytes(dtype, is_min))
        }
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

pub(super) fn single_qparam(node: &Node) -> Option<crate::graph::ir::QuantArgs> {
    node.meta
        .output_qparams
        .first()
        .or_else(|| node.meta.input_qparams.first())
        .map(|(_, qp)| *qp)
}

pub(super) fn is_float_dtype(dtype: DType) -> bool {
    tosa_float_dtype_to_ir(dtype).is_some()
}

pub(super) fn is_narrow_float_dtype(dtype: DType) -> bool {
    tosa_float_dtype_to_ir(dtype)
        .map(crate::graph::ir::DType::is_narrow_float)
        .unwrap_or(false)
}

/// Upstream `_compute_multiplier_and_shift` (op_tosa_rescale.py) for one scale.
pub(super) fn compute_multiplier_and_shift_with_width(scale: f64, scale_width: i32) -> (i32, i8) {
    let offset = if scale_width == 16 { 15 } else { 31 };
    // frexp(0) is (0, 0), so upstream emits multiplier 0 with shift `offset`; TOSA
    // requires 2 <= shift <= 62 even when the multiplier is 0.
    if scale == 0.0 {
        return (0, offset as i8);
    }

    let bits = scale.to_bits();
    let biased_exp = ((bits >> 52) & 0x7FF) as i32;
    let exponent = biased_exp - 1022;
    let mantissa = f64::from_bits((bits & 0x800F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::DType as IrDType;

    /// Upstream's algorithm, written with frexp semantics.
    fn upstream_multiplier_and_shift(scale: f64, scale_width: i32) -> (i32, i8) {
        let offset = if scale_width == 16 { 15 } else { 31 };
        let (mantissa, exponent) = if scale == 0.0 {
            (0.0, 0)
        } else {
            let exponent = scale.abs().log2().floor() as i32 + 1;
            (scale / 2f64.powi(exponent), exponent)
        };
        let limit = 1i64 << offset;
        let mut shifted = (mantissa * limit as f64).round() as i64;
        let mut shift = exponent;
        if shifted == limit {
            shifted /= 2;
            shift += 1;
        }
        let mut shift = offset - shift;
        let mut multiplier = shifted;
        if shift > 62 {
            multiplier >>= (shift - 62).min(31);
            shift = 62;
        }
        (multiplier as i32, shift as i8)
    }

    #[test]
    // 0.7071 is an arbitrary scale near the rounding boundary, not 1/sqrt(2).
    #[allow(clippy::approx_constant)]
    fn multiplier_and_shift_match_upstream() {
        for scale in [
            0.0, 1e-12, 3.1e-7, 0.000123, 0.00390625, 0.0213, 0.25, 0.4999999, 0.5, 0.7071,
            0.99999999, 1.0, 1.5, 3.999, 255.0, 4096.5,
        ] {
            for width in [16, 32] {
                assert_eq!(
                    compute_multiplier_and_shift_with_width(scale, width),
                    upstream_multiplier_and_shift(scale, width),
                    "scale {scale} width {width}"
                );
            }
        }
        assert_eq!(compute_multiplier_and_shift_with_width(0.0, 32), (0, 31));
        assert_eq!(compute_multiplier_and_shift_with_width(0.0, 16), (0, 15));
    }

    #[test]
    fn fp8_ir_dtypes_map_to_tosa_and_float_domains() {
        assert_eq!(ir_dtype_to_tosa(IrDType::Float8E4M3), DType::Fp8E4M3);
        assert_eq!(ir_dtype_to_tosa(IrDType::Float8E5M2), DType::Fp8E5M2);
        assert!(is_float_dtype(DType::Fp8E4M3));
        assert!(is_float_dtype(DType::Fp8E5M2));
        assert!(is_narrow_float_dtype(DType::Fp8E4M3));
        assert!(is_narrow_float_dtype(DType::Fp8E5M2));
    }

    #[test]
    fn fp8_default_clamp_bounds_match_torch_storage_bytes() {
        assert_eq!(
            default_clamp_bound_bytes(IrDType::Float8E4M3, true),
            vec![0xfe]
        );
        assert_eq!(
            default_clamp_bound_bytes(IrDType::Float8E4M3, false),
            vec![0x7e]
        );
        assert_eq!(
            default_clamp_bound_bytes(IrDType::Float8E5M2, true),
            vec![0xfb]
        );
        assert_eq!(
            default_clamp_bound_bytes(IrDType::Float8E5M2, false),
            vec![0x7b]
        );
    }
}
