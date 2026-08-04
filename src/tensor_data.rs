use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;

use crate::fx_py_attr::attrs;
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

/// f32 -> bfloat16 bits, round-to-nearest-even (matches torch's `.to(bfloat16)`).
pub fn encode_bf16(value: f32) -> u16 {
    let x = value.to_bits();
    if (x & 0x7fff_ffff) > 0x7f80_0000 {
        // NaN: keep sign, force quiet.
        return ((x >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x0000_7fff + ((x >> 16) & 1);
    ((x + rounding_bias) >> 16) as u16
}

/// f32 -> IEEE-754 binary16 bits, round-to-nearest-even (matches torch's `.to(float16)`).
pub fn encode_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf / NaN
        return if mantissa != 0 {
            sign | 0x7e00
        } else {
            sign | 0x7c00
        };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00; // overflow -> Inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign; // underflow -> signed zero
        }
        let mantissa = mantissa | 0x0080_0000; // restore implicit 1
        let shift = (14 - new_exp) as u32;
        let mut half = mantissa >> shift;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half as u16;
    }
    let mut half = sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) == 1) {
        half += 1; // carry into exponent is handled by the bit layout
    }
    half
}

pub fn extract_tensor_shape(value: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    attrs(value)
        .object("shape")?
        .try_iter()?
        .map(|dim| dim.and_then(|v| v.extract::<i64>()))
        .collect()
}

pub fn extract_tensor_dtype(value: &Bound<'_, PyAny>) -> DType {
    attrs(value)
        .display_optional("dtype")
        .map(|dtype| DType::from_torch_str(&dtype))
        .unwrap_or(DType::Float32)
}

pub fn extract_tensor_bytes(value: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Vec<u8>> {
    let py = value.py();
    let tensor = value.call_method0("contiguous")?.call_method0("cpu")?;
    let array = match dtype {
        DType::BFloat16 => {
            let torch = PyModule::import(py, "torch")?;
            tensor
                .call_method1("view", (torch.getattr("uint16")?,))?
                .call_method0("numpy")?
        }
        DType::Float8E4M3 | DType::Float8E5M2 => {
            let torch = PyModule::import(py, "torch")?;
            tensor
                .call_method1("view", (torch.getattr("uint8")?,))?
                .call_method0("numpy")?
        }
        _ => tensor.call_method0("numpy")?,
    };
    let view = PyMemoryView::from(&array)?;
    let byte_view = view.call_method1("cast", ("B",))?;
    PyBuffer::<u8>::get(&byte_view)?.to_vec(py)
}

pub fn extract_tensor_data(value: &Bound<'_, PyAny>) -> PyResult<TensorData> {
    let shape = extract_tensor_shape(value)?;
    let mut dtype = extract_tensor_dtype(value);
    let mut data = extract_tensor_bytes(value, dtype)?;
    if dtype == DType::Int64 {
        data = cast_i64_bytes_to_i32(&data);
        dtype = DType::Int32;
    }
    Ok(TensorData { data, shape, dtype })
}

impl TensorData {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product::<i64>().max(0) as usize
    }

    #[cfg(test)]
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
            DType::Float8E4M3 | DType::Float8E5M2 => None,
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
    fn bf16_and_f16_encoders_match_torch_for_epsilon() {
        // torch.full((1,), 1e-5, dtype=...).view(uint16): bf16=0x3728, f16=0x00a8.
        assert_eq!(encode_bf16(1e-5_f32), 0x3728);
        assert_eq!(encode_f16(1e-5_f32), 0x00a8);
        // Exact powers of two round-trip through both formats.
        for v in [0.0_f32, 1.0, -1.0, 0.5, 2.0, -0.25] {
            assert_eq!(decode_bf16(encode_bf16(v)), v, "bf16 {v}");
            assert_eq!(f32::from(half_decode_f16(encode_f16(v))), v, "f16 {v}");
        }
        // Signed zero and infinities.
        assert_eq!(encode_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(encode_f16(f32::INFINITY), 0x7c00);
        assert_eq!(encode_f16(f32::NEG_INFINITY), 0xfc00);
    }

    // Minimal f16 -> f32 decode for the round-trip assertions above.
    fn half_decode_f16(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let mant = (bits & 0x3ff) as u32;
        let f = if exp == 0 {
            (mant as f32) * 2.0_f32.powi(-24)
        } else if exp == 0x1f {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        } else {
            (1.0 + (mant as f32) / 1024.0) * 2.0_f32.powi(exp as i32 - 15)
        };
        if sign == 1 {
            -f
        } else {
            f
        }
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
