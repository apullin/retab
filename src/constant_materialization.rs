use crate::graph::ir::{DType, QuantArgs};

/// Mirrors `quantized_decomposed.quantize_per_tensor` on one value, as rc4's
/// `QuantArgs.quantize_value` calls it: float32 `value * (1 / scale)`, round half to even,
/// add the zero point, clamp to [qmin, qmax].
pub fn quantize_value_stock(value: f64, scale: f64, zp: i64, qmin: i64, qmax: i64) -> i64 {
    let inv_scale = (1.0 / scale) as f32;
    let rounded = ((value as f32) * inv_scale).round_ties_even() as f64;
    (rounded + zp as f64).clamp(qmin as f64, qmax as f64) as i64
}

pub fn quantize_scalar_with_qparam(value: f64, qp: QuantArgs) -> f64 {
    quantize_value_stock(value, qp.scale, qp.zero_point, qp.qmin, qp.qmax) as f64
}

fn scalar_bytes_impl(dtype: DType, value: f64, int4_as_i8: bool) -> Option<Vec<u8>> {
    Some(match dtype {
        DType::Float32 => (value as f32).to_le_bytes().to_vec(),
        DType::Float16 => crate::tensor_data::encode_f16(value as f32)
            .to_le_bytes()
            .to_vec(),
        DType::BFloat16 => crate::tensor_data::encode_bf16(value as f32)
            .to_le_bytes()
            .to_vec(),
        DType::Float8E4M3 | DType::Float8E5M2 => return None,
        DType::Int8 => (value as i8).to_le_bytes().to_vec(),
        DType::Int4 if int4_as_i8 => (value as i8).to_le_bytes().to_vec(),
        DType::Int16 => (value as i16).to_le_bytes().to_vec(),
        DType::Int32 => (value as i32).to_le_bytes().to_vec(),
        DType::Int64 => (value as i64).to_le_bytes().to_vec(),
        DType::Bool => vec![u8::from(value != 0.0)],
        DType::Int4 | DType::Int48 => return None,
    })
}

pub fn scalar_bytes(dtype: DType, value: f64) -> Option<Vec<u8>> {
    scalar_bytes_impl(dtype, value, false)
}

pub fn scalar_storage_bytes(dtype: DType, value: f64) -> Option<Vec<u8>> {
    scalar_bytes_impl(dtype, value, true)
}

fn shape_numel_i64(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        let dim = usize::try_from(dim).ok()?;
        acc.checked_mul(dim)
    })
}

fn shape_numel_i32(shape: &[i32]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        let dim = usize::try_from(dim).ok()?;
        acc.checked_mul(dim)
    })
}

fn repeat_scalar_bytes(numel: usize, scalar: &[u8]) -> Option<Vec<u8>> {
    let capacity = scalar.len().checked_mul(numel)?;
    let mut data = Vec::with_capacity(capacity);
    for _ in 0..numel {
        data.extend_from_slice(scalar);
    }
    Some(data)
}

pub fn repeated_scalar_bytes(shape: &[i64], scalar: &[u8]) -> Option<Vec<u8>> {
    repeat_scalar_bytes(shape_numel_i64(shape)?, scalar)
}

pub fn repeated_scalar_bytes_i32(shape: &[i32], scalar: &[u8]) -> Option<Vec<u8>> {
    repeat_scalar_bytes(shape_numel_i32(shape)?, scalar)
}

#[cfg(test)]
mod tests {
    use super::quantize_value_stock;

    #[test]
    fn stock_quantize_rounds_half_to_even_in_float32() {
        assert_eq!(quantize_value_stock(2.5, 1.0, 0, -128, 127), 2);
        assert_eq!(quantize_value_stock(-2.5, 1.0, 3, -128, 127), 1);
        assert_eq!(quantize_value_stock(100.0, 0.1, 0, -128, 127), 127);
        assert_eq!(quantize_value_stock(0.25, 0.5, 0, -128, 127), 0);
        assert_eq!(quantize_value_stock(0.75, 0.5, 0, -128, 127), 2);
    }
}
