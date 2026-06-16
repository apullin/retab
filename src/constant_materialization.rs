use crate::graph::ir::DType;

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
