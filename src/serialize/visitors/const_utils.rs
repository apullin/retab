use super::*;
use crate::constant_materialization::{
    repeated_scalar_bytes_i32, scalar_bytes as constant_scalar_bytes,
};

pub(super) fn normalize_tensor_shape(shape: Vec<i32>) -> Vec<i32> {
    if shape.is_empty() {
        vec![1]
    } else {
        shape
    }
}

pub(super) fn make_tensor(
    name: String,
    shape: Vec<i32>,
    dtype: DType,
    data: Vec<u8>,
) -> TosaTensor {
    TosaTensor::with_data(&name, normalize_tensor_shape(shape), dtype, data)
}

pub(super) fn make_empty_tensor(name: String, shape: Vec<i32>, dtype: DType) -> TosaTensor {
    TosaTensor::new(&name, normalize_tensor_shape(shape), dtype)
}

pub(super) fn make_empty_tensor_for_node(
    node: &Node,
    name: String,
    shape: Vec<i32>,
    dtype: DType,
) -> TosaTensor {
    if node.meta.preserve_scalar_shape && shape.is_empty() {
        TosaTensor::new(&name, shape, dtype)
    } else {
        make_empty_tensor(name, shape, dtype)
    }
}

pub(super) fn make_rank1_const_tensor(
    name: String,
    dtype: DType,
    data: Vec<u8>,
    len: usize,
) -> TosaTensor {
    make_tensor(name, vec![len as i32], dtype, data)
}

pub(super) fn push_const_shape(result: &mut EmitResult, shape_name: String, values: Vec<i32>) {
    let rank = values.len() as u32;
    let data: Vec<u8> = values
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();
    result.poolable.push(shape_name.clone());
    result.shapes.push(TosaShape {
        name: shape_name.clone(),
        rank,
        data,
    });
    result.push_op(
        Op::ConstShape,
        Attribute::ConstShape(ConstShapeAttribute {}),
        vec![],
        vec![shape_name],
    );
}

pub(super) fn scalar_fill_bytes(dtype: crate::graph::ir::DType, value: f64) -> Option<Vec<u8>> {
    constant_scalar_bytes(dtype, value)
}

fn append_scalar_fill_bytes(
    data: &mut Vec<u8>,
    dtype: crate::graph::ir::DType,
    value: f64,
) -> Option<()> {
    data.extend_from_slice(&scalar_fill_bytes(dtype, value)?);
    Some(())
}

pub(super) fn make_filled_const_tensor(
    name: String,
    shape: Vec<i32>,
    dtype: crate::graph::ir::DType,
    value: f64,
) -> Option<TosaTensor> {
    let scalar = scalar_fill_bytes(dtype, value)?;
    let data = repeated_scalar_bytes_i32(&shape, &scalar)?;
    Some(make_tensor(name, shape, ir_dtype_to_tosa(dtype), data))
}

pub(super) fn make_eye_const_tensor(
    name: String,
    shape: Vec<i32>,
    dtype: crate::graph::ir::DType,
    zero_value: f64,
    one_value: f64,
) -> Option<TosaTensor> {
    if shape.len() != 2 {
        return None;
    }
    let rows = usize::try_from(shape[0]).ok()?;
    let cols = usize::try_from(shape[1]).ok()?;
    let elem_size = scalar_fill_bytes(dtype, zero_value)?.len();
    let mut data = Vec::with_capacity(rows.checked_mul(cols)?.checked_mul(elem_size)?);
    for row in 0..rows {
        for col in 0..cols {
            let value = if row == col { one_value } else { zero_value };
            append_scalar_fill_bytes(&mut data, dtype, value)?;
        }
    }
    Some(make_tensor(name, shape, ir_dtype_to_tosa(dtype), data))
}

pub(super) fn quantize_fill_value(node: &Node, value: f64) -> f64 {
    match node.meta.output_qparams.first() {
        Some((_, qp))
            if matches!(
                qp.dtype,
                crate::graph::ir::DType::Int4
                    | crate::graph::ir::DType::Int8
                    | crate::graph::ir::DType::Int16
                    | crate::graph::ir::DType::Int32
                    | crate::graph::ir::DType::Int64
            ) =>
        {
            quantize_scalar_with_qparam(value, *qp)
        }
        _ => value,
    }
}

/// Mirrors `quantized_decomposed.quantize_per_tensor` on one value, as rc4's
/// `QuantArgs.quantize_value` calls it: float32 `value * (1 / scale)`, round half to even,
/// add the zero point, clamp to [qmin, qmax].
pub(super) fn quantize_scalar_with_qparam(value: f64, qp: crate::graph::ir::QuantArgs) -> f64 {
    let inv_scale = (1.0 / qp.scale) as f32;
    let rounded = ((value as f32) * inv_scale).round_ties_even() as f64;
    (rounded + qp.zero_point as f64).clamp(qp.qmin as f64, qp.qmax as f64)
}

pub(super) fn push_inline_const(result: &mut EmitResult, tensor: TosaTensor) {
    result.push_const(tensor);
}

pub(super) fn push_filled_const_if_supported(
    result: &mut EmitResult,
    name: String,
    shape: Vec<i32>,
    dtype: crate::graph::ir::DType,
    value: f64,
) -> bool {
    let Some(tensor) = make_filled_const_tensor(name, shape, dtype, value) else {
        return false;
    };
    push_inline_const(result, tensor);
    true
}

pub(super) fn make_shape_const_tensor(name: String, values: Vec<i32>) -> TosaTensor {
    let data: Vec<u8> = values
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();
    TosaTensor::with_data(&name, vec![values.len() as i32], DType::Shape, data)
}

#[cfg(test)]
mod tests {
    use super::{quantize_scalar_with_qparam, scalar_fill_bytes};
    use crate::graph::ir::{DType as IrDType, QuantArgs};

    fn int8_qargs(scale: f64, zero_point: i64) -> QuantArgs {
        QuantArgs {
            scale,
            zero_point,
            dtype: IrDType::Int8,
            qmin: -128,
            qmax: 127,
        }
    }

    #[test]
    fn quantize_scalar_matches_quantize_per_tensor() {
        // Ties round to even, as torch.round does.
        assert_eq!(quantize_scalar_with_qparam(2.5, int8_qargs(1.0, 0)), 2.0);
        assert_eq!(quantize_scalar_with_qparam(-2.5, int8_qargs(1.0, 3)), 1.0);
        // An avg-pool zero point quantized through its own output qparams saturates.
        assert_eq!(
            quantize_scalar_with_qparam(-3.0, int8_qargs(0.012688863091170788, -3)),
            -128.0
        );
        assert_eq!(quantize_scalar_with_qparam(0.0, int8_qargs(0.02, -7)), -7.0);
    }

    #[test]
    fn scalar_fill_bytes_encodes_fp32_constants_as_float_bytes() {
        assert_eq!(
            scalar_fill_bytes(IrDType::Float32, 1.0).expect("fp32 bytes"),
            1.0f32.to_le_bytes().to_vec()
        );
    }

    #[test]
    fn scalar_fill_bytes_encodes_fp16_constants() {
        assert_eq!(
            scalar_fill_bytes(IrDType::Float16, 1.0).expect("fp16 bytes"),
            vec![0x00, 0x3c]
        );
        assert_eq!(
            scalar_fill_bytes(IrDType::Float16, -0.5).expect("fp16 bytes"),
            vec![0x00, 0xb8]
        );
    }

    #[test]
    fn scalar_fill_bytes_encodes_bf16_constants() {
        assert_eq!(
            scalar_fill_bytes(IrDType::BFloat16, 1.0).expect("bf16 bytes"),
            vec![0x80, 0x3f]
        );
        assert_eq!(
            scalar_fill_bytes(IrDType::BFloat16, -0.5).expect("bf16 bytes"),
            vec![0x00, 0xbf]
        );
    }
}
