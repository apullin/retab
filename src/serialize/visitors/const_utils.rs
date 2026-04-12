use super::*;

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
    result.shapes.push(TosaShape {
        name: shape_name.clone(),
        rank,
        data,
    });
    result.operators.push(TosaOperator::new(
        Op::ConstShape,
        Attribute::ConstShape(ConstShapeAttribute {}),
        vec![],
        vec![shape_name],
    ));
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

fn encode_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        return if mant == 0 { sign | 0x7c00 } else { sign | 0x7e00 };
    }

    let exp16 = exp - 127 + 15;
    if exp16 >= 0x1f {
        return sign | 0x7c00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = (14 - exp16) as u32;
        let mut half = (mantissa >> shift) as u16;
        let round_mask = (1u32 << shift) - 1;
        let halfway = 1u32 << (shift - 1);
        let remainder = mantissa & round_mask;
        if remainder > halfway || (remainder == halfway && (half & 1) != 0) {
            half = half.wrapping_add(1);
        }
        return sign | half;
    }

    let mut half = ((exp16 as u16) << 10) | ((mant >> 13) as u16);
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) != 0) {
        half = half.wrapping_add(1);
    }
    sign | half
}

fn encode_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

pub(super) fn scalar_fill_bytes(dtype: crate::graph::ir::DType, value: f64) -> Option<Vec<u8>> {
    use crate::graph::ir::DType as IrDType;

    Some(match dtype {
        IrDType::Float32 => (value as f32).to_le_bytes().to_vec(),
        IrDType::Float16 => encode_f16_bits(value as f32).to_le_bytes().to_vec(),
        IrDType::BFloat16 => encode_bf16_bits(value as f32).to_le_bytes().to_vec(),
        IrDType::Int8 => (value as i8).to_le_bytes().to_vec(),
        IrDType::Int16 => (value as i16).to_le_bytes().to_vec(),
        IrDType::Int32 => (value as i32).to_le_bytes().to_vec(),
        IrDType::Int64 => (value as i64).to_le_bytes().to_vec(),
        IrDType::Bool => vec![(value != 0.0) as u8],
        IrDType::Int4 | IrDType::Int48 => return None,
    })
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
    let numel = shape_numel(&shape)?;
    let mut data = Vec::with_capacity(scalar.len().saturating_mul(numel));
    for _ in 0..numel {
        data.extend_from_slice(&scalar);
    }

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
            let quantized = ((value / qp.scale) + qp.zero_point as f64).round() as i64;
            quantized.clamp(qp.qmin, qp.qmax) as f64
        }
        _ => value,
    }
}

pub(super) fn push_inline_const(result: &mut EmitResult, tensor: TosaTensor) {
    result.push_const(tensor);
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
    use super::scalar_fill_bytes;
    use crate::graph::ir::DType as IrDType;

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
