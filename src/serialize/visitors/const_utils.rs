use super::*;

pub(super) fn make_tensor(
    name: String,
    shape: Vec<i32>,
    dtype: DType,
    data: Vec<u8>,
) -> TosaTensor {
    TosaTensor::with_data(name, shape, dtype, data)
}

pub(super) fn make_empty_tensor(name: String, shape: Vec<i32>, dtype: DType) -> TosaTensor {
    make_tensor(name, shape, dtype, Vec::new())
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
        Attribute::ConstShape(ConstShapeAttribute),
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

fn scalar_fill_bytes(dtype: crate::graph::ir::DType, value: f64) -> Option<Vec<u8>> {
    use crate::graph::ir::DType as IrDType;

    Some(match dtype {
        IrDType::Float32 => (value as f32).to_le_bytes().to_vec(),
        IrDType::Int8 => (value as i8).to_le_bytes().to_vec(),
        IrDType::Int16 => (value as i16).to_le_bytes().to_vec(),
        IrDType::Int32 => (value as i32).to_le_bytes().to_vec(),
        IrDType::Int64 => (value as i64).to_le_bytes().to_vec(),
        IrDType::Bool => vec![(value != 0.0) as u8],
        IrDType::Float16 | IrDType::BFloat16 | IrDType::Int4 | IrDType::Int48 => return None,
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
    match (node.meta.val.dtype, node.meta.output_qparams.first()) {
        (crate::graph::ir::DType::Int8 | crate::graph::ir::DType::Int16, Some((_, qp))) => {
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
    make_tensor(name, vec![values.len() as i32], DType::Shape, data)
}
