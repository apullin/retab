use pyo3::prelude::*;

use crate::graph::ir::DType;
use crate::store::TensorData;

fn decode_f16(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = match exp {
        0 if frac == 0 => sign << 31,
        0 => {
            let mut frac = frac;
            let mut exp_shift = 0u32;
            while (frac & 0x0400) == 0 {
                frac <<= 1;
                exp_shift += 1;
            }
            frac &= 0x03ff;
            (sign << 31) | ((127 - 15 - exp_shift) << 23) | (frac << 13)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (frac << 13),
        _ => (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13),
    };
    f32::from_bits(f32_bits)
}

fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn extract_tensor_shape(value: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    value
        .getattr("shape")?
        .iter()?
        .map(|dim| dim.and_then(|v| v.extract::<i64>()))
        .collect()
}

pub fn extract_tensor_dtype(value: &Bound<'_, PyAny>) -> DType {
    value
        .getattr("dtype")
        .and_then(|dtype| dtype.str()?.extract::<String>())
        .map(|dtype| parse_torch_dtype(&dtype))
        .unwrap_or(DType::Float32)
}

pub fn extract_tensor_bytes(value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let tensor = value.call_method0("contiguous")?.call_method0("cpu")?;
    if extract_tensor_dtype(&tensor) == DType::BFloat16 {
        let torch = PyModule::import_bound(tensor.py(), "torch")?;
        return tensor
            .call_method1("view", (torch.getattr("uint16")?,))?
            .call_method0("numpy")?
            .call_method0("tobytes")?
            .extract();
    }
    tensor
        .call_method0("numpy")?
        .call_method0("tobytes")?
        .extract()
}

pub fn extract_tensor_data(value: &Bound<'_, PyAny>) -> PyResult<TensorData> {
    let shape = extract_tensor_shape(value)?;
    let mut dtype = extract_tensor_dtype(value);
    let mut data = extract_tensor_bytes(value)?;
    if dtype == DType::Int64 {
        data = cast_i64_bytes_to_i32(&data);
        dtype = DType::Int32;
    }
    Ok(TensorData { data, shape, dtype })
}

pub fn parse_torch_dtype(dtype_str: &str) -> DType {
    match dtype_str {
        s if s.contains("float32") => DType::Float32,
        s if s.contains("bfloat16") => DType::BFloat16,
        s if s.contains("float16") => DType::Float16,
        s if s.contains("int8") => DType::Int8,
        s if s.contains("int16") => DType::Int16,
        s if s.contains("int32") => DType::Int32,
        s if s.contains("int64") => DType::Int64,
        s if s.contains("bool") => DType::Bool,
        _ => DType::Float32,
    }
}

impl TensorData {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product::<i64>().max(0) as usize
    }

    pub fn scalar_i64(&self) -> Option<i64> {
        if self.element_count() != 1 {
            return None;
        }
        self.vec_i64().and_then(|vals| vals.into_iter().next())
    }

    pub fn scalar_f64(&self) -> Option<f64> {
        if self.element_count() != 1 {
            return None;
        }
        self.vec_f64().and_then(|vals| vals.into_iter().next())
    }

    pub fn vec_i64(&self) -> Option<Vec<i64>> {
        match self.dtype {
            DType::Bool => Some(self.data.iter().map(|&b| i64::from(b != 0)).collect()),
            DType::Int8 => Some(self.data.iter().map(|&b| i64::from(b as i8)).collect()),
            DType::Int16 => Some(
                self.data
                    .chunks_exact(2)
                    .map(|chunk| i64::from(i16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect(),
            ),
            DType::Int32 => Some(
                self.data
                    .chunks_exact(4)
                    .map(|chunk| {
                        i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    })
                    .collect(),
            ),
            DType::Int64 => Some(
                self.data
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn vec_f64(&self) -> Option<Vec<f64>> {
        match self.dtype {
            DType::Bool | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => self
                .vec_i64()
                .map(|vals| vals.into_iter().map(|v| v as f64).collect()),
            DType::Float32 => Some(
                self.data
                    .chunks_exact(4)
                    .map(|chunk| {
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64
                    })
                    .collect(),
            ),
            DType::Float16 => Some(
                self.data
                    .chunks_exact(2)
                    .map(|chunk| decode_f16(u16::from_le_bytes([chunk[0], chunk[1]])) as f64)
                    .collect(),
            ),
            DType::BFloat16 => Some(
                self.data
                    .chunks_exact(2)
                    .map(|chunk| decode_bf16(u16::from_le_bytes([chunk[0], chunk[1]])) as f64)
                    .collect(),
            ),
            _ => None,
        }
    }
}

fn cast_i64_bytes_to_i32(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(8)
        .flat_map(|chunk| {
            let val = i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            (val as i32).to_le_bytes()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_data_decodes_int_and_float_scalars() {
        let int_tensor = TensorData {
            data: (7i32).to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: DType::Int32,
        };
        assert_eq!(int_tensor.scalar_i64(), Some(7));
        assert_eq!(int_tensor.scalar_f64(), Some(7.0));

        let float_tensor = TensorData {
            data: (0.25f32).to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: DType::Float32,
        };
        assert_eq!(float_tensor.scalar_f64(), Some(0.25));
        assert_eq!(float_tensor.scalar_i64(), None);
    }

    #[test]
    fn tensor_data_decodes_vectors_and_casts_i64_bytes() {
        let int64_tensor = TensorData {
            data: [1i64.to_le_bytes(), 3i64.to_le_bytes()].concat(),
            shape: vec![2],
            dtype: DType::Int64,
        };
        assert_eq!(int64_tensor.vec_i64(), Some(vec![1, 3]));

        let cast = cast_i64_bytes_to_i32(&int64_tensor.data);
        assert_eq!(cast, [1i32.to_le_bytes(), 3i32.to_le_bytes()].concat(),);
    }
}
