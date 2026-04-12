use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrNode, QParam};
use crate::graph::ir::DType;
use crate::store::TensorData;
use crate::tensor_data::parse_torch_dtype;

pub fn tensor_numel(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        let dim = usize::try_from(dim).ok()?;
        acc.checked_mul(dim)
    })
}

pub fn arg_int_list(arg: &IrArg) -> Option<Vec<i64>> {
    match arg {
        IrArg::List(items) => items
            .iter()
            .map(|item| match item {
                IrArg::Int(value) => Some(*value),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

pub fn normalize_dim(dim: i64, rank: usize, allow_end: bool) -> Option<usize> {
    let rank_i64 = i64::try_from(rank).ok()?;
    let max = if allow_end { rank_i64 } else { rank_i64 - 1 };
    let normalized = if dim < 0 {
        dim + rank_i64 + i64::from(allow_end)
    } else {
        dim
    };
    if normalized < 0 || normalized > max {
        return None;
    }
    usize::try_from(normalized).ok()
}

pub fn reshape_target_shape(input_shape: &[i64], node: &IrNode) -> Option<Vec<i64>> {
    let spec = arg_int_list(node.args.get(1)?)?;
    let input_numel = tensor_numel(input_shape)?;
    let mut target = spec.clone();
    let mut inferred = None;
    let mut known_product = 1usize;

    for (idx, &dim) in spec.iter().enumerate() {
        match dim {
            -1 => {
                if inferred.replace(idx).is_some() {
                    return None;
                }
            }
            dim if dim >= 0 => {
                known_product = known_product.checked_mul(usize::try_from(dim).ok()?)?;
            }
            _ => return None,
        }
    }

    if let Some(idx) = inferred {
        if known_product == 0 || input_numel % known_product != 0 {
            return None;
        }
        target[idx] = (input_numel / known_product) as i64;
    } else if known_product != input_numel {
        return None;
    }

    Some(target)
}

pub fn unsqueeze_target_shape(input_shape: &[i64], node: &IrNode) -> Option<Vec<i64>> {
    let dim = match node.args.get(1)? {
        IrArg::Int(value) => *value,
        _ => return None,
    };
    let axis = normalize_dim(dim, input_shape.len(), true)?;
    let mut shape = input_shape.to_vec();
    shape.insert(axis, 1);
    Some(shape)
}

pub fn permute_order(node: &IrNode, rank: usize) -> Option<Vec<usize>> {
    arg_int_list(node.args.get(1)?)?
        .into_iter()
        .map(|dim| normalize_dim(dim, rank, false))
        .collect()
}

fn dtype_elem_size(dtype: DType) -> usize {
    match dtype {
        DType::Bool | DType::Int8 | DType::Int4 => 1,
        DType::Int16 | DType::Float16 | DType::BFloat16 => 2,
        DType::Int32 | DType::Float32 => 4,
        DType::Int48 => 6,
        DType::Int64 => 8,
    }
}

fn contiguous_strides(shape: &[i64]) -> Option<Vec<usize>> {
    let rank = shape.len();
    let shape_usize: Vec<usize> = shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()?;
    let mut strides = vec![1usize; rank];
    for idx in (0..rank.saturating_sub(1)).rev() {
        strides[idx] = strides[idx + 1].checked_mul(shape_usize[idx + 1])?;
    }
    Some(strides)
}

fn arg_scalar(arg: Option<&IrArg>) -> Option<f64> {
    match arg? {
        IrArg::Float(value) => Some(*value),
        IrArg::Int(value) => Some(*value as f64),
        IrArg::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn encode_tensor_values(values: &[f64], shape: Vec<i64>, dtype: DType) -> Option<TensorData> {
    let data = match dtype {
        DType::Float32 => values
            .iter()
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect(),
        DType::Int8 | DType::Int4 => values
            .iter()
            .flat_map(|value| (*value as i8).to_le_bytes())
            .collect(),
        DType::Int16 => values
            .iter()
            .flat_map(|value| (*value as i16).to_le_bytes())
            .collect(),
        DType::Int32 => values
            .iter()
            .flat_map(|value| (*value as i32).to_le_bytes())
            .collect(),
        DType::Int64 => values
            .iter()
            .flat_map(|value| (*value as i64).to_le_bytes())
            .collect(),
        DType::Bool => values.iter().map(|value| u8::from(*value != 0.0)).collect(),
        DType::Float16 | DType::BFloat16 | DType::Int48 => return None,
    };
    Some(TensorData { data, shape, dtype })
}

fn decode_tensor_values(tensor: &TensorData, qparam: Option<&QParam>) -> Option<Vec<f64>> {
    let values = tensor.vec_f64()?;
    let Some(qparam) = qparam else {
        return Some(values);
    };
    if qparam.is_per_channel() {
        return None;
    }
    Some(
        values
            .into_iter()
            .map(|value| (value - qparam.zp as f64) * qparam.scale)
            .collect(),
    )
}

fn quantize_values(values: &[f64], qparam: &QParam) -> Option<Vec<f64>> {
    if qparam.is_per_channel() || qparam.scale == 0.0 {
        return None;
    }
    Some(
        values
            .iter()
            .map(|value| {
                ((value / qparam.scale).round_ties_even() as i64 + qparam.zp)
                    .clamp(qparam.qmin, qparam.qmax) as f64
            })
            .collect(),
    )
}

fn broadcast_offset(out_linear: usize, out_shape: &[i64], input_shape: &[i64]) -> Option<usize> {
    let out_rank = out_shape.len();
    let in_rank = input_shape.len();
    if in_rank > out_rank {
        return None;
    }
    let out_strides = contiguous_strides(out_shape)?;
    let in_strides = contiguous_strides(input_shape)?;
    let input_shape_usize: Vec<usize> = input_shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()?;
    let output_shape_usize: Vec<usize> = out_shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()?;

    let mut rem = out_linear;
    let mut out_coords = vec![0usize; out_rank];
    for axis in 0..out_rank {
        let stride = out_strides[axis];
        out_coords[axis] = if stride == 0 { 0 } else { rem / stride };
        rem = if stride == 0 { 0 } else { rem % stride };
    }

    let rank_delta = out_rank - in_rank;
    let mut offset = 0usize;
    for in_axis in 0..in_rank {
        let out_axis = rank_delta + in_axis;
        let in_dim = input_shape_usize[in_axis];
        let out_dim = output_shape_usize[out_axis];
        if in_dim != 1 && in_dim != out_dim {
            return None;
        }
        let coord = if in_dim == 1 { 0 } else { out_coords[out_axis] };
        offset = offset.checked_add(coord.checked_mul(in_strides[in_axis])?)?;
    }
    Some(offset)
}

fn add_broadcasted_bias(
    output: &mut [f64],
    output_shape: &[i64],
    bias: &[f64],
    bias_shape: &[i64],
    beta: f64,
) -> Option<()> {
    if bias.len() == 1 {
        let bias_value = bias[0] * beta;
        for value in output.iter_mut() {
            *value += bias_value;
        }
        return Some(());
    }
    for (idx, out) in output.iter_mut().enumerate() {
        let bias_idx = broadcast_offset(idx, output_shape, bias_shape)?;
        *out += bias.get(bias_idx)? * beta;
    }
    Some(())
}

fn matmul_2d(lhs: &[f64], lhs_shape: &[i64], rhs: &[f64], rhs_shape: &[i64]) -> Option<TensorData> {
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return None;
    }
    let m = usize::try_from(lhs_shape[0]).ok()?;
    let k = usize::try_from(lhs_shape[1]).ok()?;
    let rhs_k = usize::try_from(rhs_shape[0]).ok()?;
    let n = usize::try_from(rhs_shape[1]).ok()?;
    if k != rhs_k || lhs.len() != m.checked_mul(k)? || rhs.len() != rhs_k.checked_mul(n)? {
        return None;
    }

    let mut values = vec![0.0; m.checked_mul(n)?];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += lhs[row * k + kk] * rhs[kk * n + col];
            }
            values[row * n + col] = acc;
        }
    }
    Some(
        TensorData {
            data: Vec::new(),
            shape: vec![m as i64, n as i64],
            dtype: DType::Float32,
        }
        .with_values(values),
    )
}

fn batched_matmul_3d(
    lhs: &[f64],
    lhs_shape: &[i64],
    rhs: &[f64],
    rhs_shape: &[i64],
) -> Option<TensorData> {
    if lhs_shape.len() != 3 || rhs_shape.len() != 3 {
        return None;
    }
    let batch = usize::try_from(lhs_shape[0]).ok()?;
    let m = usize::try_from(lhs_shape[1]).ok()?;
    let k = usize::try_from(lhs_shape[2]).ok()?;
    let rhs_batch = usize::try_from(rhs_shape[0]).ok()?;
    let rhs_k = usize::try_from(rhs_shape[1]).ok()?;
    let n = usize::try_from(rhs_shape[2]).ok()?;
    if batch != rhs_batch
        || k != rhs_k
        || lhs.len() != batch.checked_mul(m)?.checked_mul(k)?
        || rhs.len() != batch.checked_mul(rhs_k)?.checked_mul(n)?
    {
        return None;
    }

    let mut values = vec![0.0; batch.checked_mul(m)?.checked_mul(n)?];
    for b in 0..batch {
        let lhs_base = b * m * k;
        let rhs_base = b * k * n;
        let out_base = b * m * n;
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += lhs[lhs_base + row * k + kk] * rhs[rhs_base + kk * n + col];
                }
                values[out_base + row * n + col] = acc;
            }
        }
    }
    Some(
        TensorData {
            data: Vec::new(),
            shape: vec![batch as i64, m as i64, n as i64],
            dtype: DType::Float32,
        }
        .with_values(values),
    )
}

trait TensorValuesExt {
    fn with_values(self, values: Vec<f64>) -> Self;
}

impl TensorValuesExt for TensorData {
    fn with_values(mut self, values: Vec<f64>) -> Self {
        self.data = values
            .into_iter()
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect();
        self
    }
}

fn output_numel_allowed(node: &IrNode, shape: &[i64]) -> Option<bool> {
    let out_numel = tensor_numel(shape)?;
    let Some(fake_shape) = node.meta.val_shape() else {
        return Some(true);
    };
    Some(out_numel <= tensor_numel(fake_shape)?)
}

fn constant_input_tensor<'a>(
    node: &IrNode,
    arg_index: usize,
    constant_tensors: &'a HashMap<usize, TensorData>,
) -> Option<&'a TensorData> {
    match node.args.get(arg_index)? {
        IrArg::Node(node_id) => constant_tensors.get(node_id),
        _ => None,
    }
}

fn resolve_alpha_beta(node: &IrNode) -> (f64, f64) {
    let beta = arg_scalar(node.args.get(3))
        .or_else(|| arg_scalar(node.kwargs.get("beta")))
        .unwrap_or(1.0);
    let alpha = arg_scalar(node.args.get(4))
        .or_else(|| arg_scalar(node.kwargs.get("alpha")))
        .unwrap_or(1.0);
    (alpha, beta)
}

pub fn plan_constant_matmul_tensor(
    node: &IrNode,
    constant_tensors: &HashMap<usize, TensorData>,
) -> Option<TensorData> {
    let lhs = constant_input_tensor(
        node,
        usize::from(node.target.contains("addmm")),
        constant_tensors,
    )?;
    let rhs = constant_input_tensor(
        node,
        usize::from(node.target.contains("addmm")) + 1,
        constant_tensors,
    )?;
    let lhs_qparam = node
        .meta
        .input_qparams
        .as_ref()
        .and_then(|qps| qps.get(&usize::from(node.target.contains("addmm"))));
    let rhs_qparam = node
        .meta
        .input_qparams
        .as_ref()
        .and_then(|qps| qps.get(&(usize::from(node.target.contains("addmm")) + 1)));
    let lhs_values = decode_tensor_values(lhs, lhs_qparam)?;
    let rhs_values = decode_tensor_values(rhs, rhs_qparam)?;

    let mut output =
        if node.target.contains("aten.mm.default") || node.target.contains("aten.addmm.default") {
            matmul_2d(&lhs_values, &lhs.shape, &rhs_values, &rhs.shape)?
        } else if node.target.contains("aten.bmm.default") {
            batched_matmul_3d(&lhs_values, &lhs.shape, &rhs_values, &rhs.shape)?
        } else {
            return None;
        };

    if node.target.contains("aten.addmm.default") {
        let bias = constant_input_tensor(node, 0, constant_tensors)?;
        let bias_qparam = node.meta.input_qparams.as_ref().and_then(|qps| qps.get(&0));
        let bias_values = decode_tensor_values(bias, bias_qparam)?;
        let (alpha, beta) = resolve_alpha_beta(node);
        let mut values = output.vec_f64()?;
        if (alpha - 1.0).abs() > 1e-10 {
            for value in &mut values {
                *value *= alpha;
            }
        }
        add_broadcasted_bias(&mut values, &output.shape, &bias_values, &bias.shape, beta)?;
        output.data = values
            .into_iter()
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect();
    }

    if !output_numel_allowed(node, &output.shape)? {
        return None;
    }

    if let Some(output_qparam) = node.meta.output_qparam() {
        let output_values = quantize_values(&output.vec_f64()?, output_qparam)?;
        let output_dtype = parse_torch_dtype(&output_qparam.dtype_str);
        return encode_tensor_values(&output_values, output.shape, output_dtype);
    }

    let output_dtype = node
        .meta
        .val_dtype()
        .map(parse_torch_dtype)
        .unwrap_or(lhs.dtype);
    if output_dtype != DType::Float32 {
        return None;
    }
    encode_tensor_values(&output.vec_f64()?, output.shape, output_dtype)
}

pub fn permute_tensor_data(input: &TensorData, order: &[usize]) -> Option<TensorData> {
    let rank = input.shape.len();
    if rank != order.len() {
        return None;
    }
    if order.iter().enumerate().all(|(idx, &axis)| idx == axis) {
        return Some(input.clone());
    }

    let numel = tensor_numel(&input.shape)?;
    let elem_size = dtype_elem_size(input.dtype);
    if numel.saturating_mul(elem_size) != input.data.len() {
        return None;
    }

    let shape_usize: Vec<usize> = input
        .shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()?;
    let new_shape: Vec<i64> = order
        .iter()
        .map(|&axis| input.shape.get(axis).copied())
        .collect::<Option<_>>()?;

    let mut input_strides = vec![1usize; rank];
    for idx in (0..rank.saturating_sub(1)).rev() {
        input_strides[idx] = input_strides[idx + 1].checked_mul(shape_usize[idx + 1])?;
    }

    let new_shape_usize: Vec<usize> = new_shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()?;
    let mut output_strides = vec![1usize; rank];
    for idx in (0..rank.saturating_sub(1)).rev() {
        output_strides[idx] = output_strides[idx + 1].checked_mul(new_shape_usize[idx + 1])?;
    }

    let mut out = vec![0u8; input.data.len()];
    let mut coords = vec![0usize; rank];
    let mut dst_coords = vec![0usize; rank];

    for src_linear in 0..numel {
        let mut rem = src_linear;
        for axis in 0..rank {
            let stride = input_strides[axis];
            coords[axis] = if stride == 0 { 0 } else { rem / stride };
            rem = if stride == 0 { 0 } else { rem % stride };
        }
        for (out_axis, &in_axis) in order.iter().enumerate() {
            dst_coords[out_axis] = coords[in_axis];
        }
        let dst_linear = dst_coords
            .iter()
            .zip(output_strides.iter())
            .try_fold(0usize, |acc, (&coord, &stride)| {
                acc.checked_add(coord.checked_mul(stride)?)
            })?;
        let src_off = src_linear.checked_mul(elem_size)?;
        let dst_off = dst_linear.checked_mul(elem_size)?;
        out[dst_off..dst_off + elem_size]
            .copy_from_slice(&input.data[src_off..src_off + elem_size]);
    }

    Some(TensorData {
        data: out,
        shape: new_shape,
        dtype: input.dtype,
    })
}

pub fn plan_shape_only_tensor(node: &IrNode, input: &TensorData) -> Option<TensorData> {
    if node.target.contains("aten.reshape.default")
        || node.target.contains("aten.view_copy.default")
    {
        return Some(TensorData {
            data: input.data.clone(),
            shape: reshape_target_shape(&input.shape, node)?,
            dtype: input.dtype,
        });
    }
    if node.target.contains("aten.unsqueeze_copy.default") {
        return Some(TensorData {
            data: input.data.clone(),
            shape: unsqueeze_target_shape(&input.shape, node)?,
            dtype: input.dtype,
        });
    }
    if node.target.contains("aten.permute_copy.default") {
        return permute_tensor_data(input, &permute_order(node, input.shape.len())?);
    }
    None
}

fn normalize_slice_bound(index: Option<i64>, dim_size: i64, default: i64) -> i64 {
    let mut value = index.unwrap_or(default);
    if value < 0 {
        value += dim_size;
    }
    value.clamp(0, dim_size)
}

pub fn slice_tensor_data(
    input: &TensorData,
    dim: usize,
    start: i64,
    end: i64,
    step: i64,
) -> Option<TensorData> {
    if step <= 0 || dim >= input.shape.len() {
        return None;
    }
    let dim_size = *input.shape.get(dim)?;
    let start = normalize_slice_bound(Some(start), dim_size, 0);
    let end = normalize_slice_bound(Some(end), dim_size, dim_size);
    let slice_len = if end <= start {
        0
    } else {
        ((end - start) + step - 1) / step
    };

    let mut shape = input.shape.clone();
    shape[dim] = slice_len;
    let elem_size = dtype_elem_size(input.dtype);
    let numel = tensor_numel(&shape)?;
    if numel == 0 {
        return Some(TensorData {
            data: Vec::new(),
            shape,
            dtype: input.dtype,
        });
    }

    let rank = input.shape.len();
    let input_shape_usize: Vec<usize> = input
        .shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()?;
    let output_shape_usize: Vec<usize> = shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<_, _>>()
        .ok()?;

    let mut input_strides = vec![1usize; rank];
    for idx in (0..rank.saturating_sub(1)).rev() {
        input_strides[idx] = input_strides[idx + 1].checked_mul(input_shape_usize[idx + 1])?;
    }
    let mut output_strides = vec![1usize; rank];
    for idx in (0..rank.saturating_sub(1)).rev() {
        output_strides[idx] = output_strides[idx + 1].checked_mul(output_shape_usize[idx + 1])?;
    }

    let mut out = vec![0u8; numel.checked_mul(elem_size)?];
    let mut out_coords = vec![0usize; rank];
    let mut in_coords = vec![0usize; rank];

    for out_linear in 0..numel {
        let mut rem = out_linear;
        for axis in 0..rank {
            let stride = output_strides[axis];
            out_coords[axis] = if stride == 0 { 0 } else { rem / stride };
            rem = if stride == 0 { 0 } else { rem % stride };
            in_coords[axis] = out_coords[axis];
        }
        in_coords[dim] = usize::try_from(start + (out_coords[dim] as i64 * step)).ok()?;
        let in_linear = in_coords
            .iter()
            .zip(input_strides.iter())
            .try_fold(0usize, |acc, (&coord, &stride)| {
                acc.checked_add(coord.checked_mul(stride)?)
            })?;
        let src_off = in_linear.checked_mul(elem_size)?;
        let dst_off = out_linear.checked_mul(elem_size)?;
        out[dst_off..dst_off + elem_size]
            .copy_from_slice(&input.data[src_off..src_off + elem_size]);
    }

    Some(TensorData {
        data: out,
        shape,
        dtype: input.dtype,
    })
}

pub fn plan_constant_slice_tensor(node: &IrNode, input: &TensorData) -> Option<TensorData> {
    let dim = match node.args.get(1) {
        Some(IrArg::Int(value)) => normalize_dim(*value, input.shape.len(), false)?,
        _ => 0,
    };
    let dim_size = *input.shape.get(dim)?;
    let start = match node.args.get(2) {
        Some(IrArg::Int(value)) => *value,
        Some(IrArg::None) | None => 0,
        _ => return None,
    };
    let end = match node.args.get(3) {
        Some(IrArg::Int(value)) => *value,
        Some(IrArg::None) | None => dim_size,
        _ => return None,
    };
    let step = match node.args.get(4) {
        Some(IrArg::Int(value)) => *value,
        Some(IrArg::None) | None => 1,
        _ => return None,
    };
    slice_tensor_data(input, dim, start, end, step)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int32_tensor(shape: Vec<i64>, values: &[i32]) -> TensorData {
        TensorData {
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape,
            dtype: DType::Int32,
        }
    }

    #[test]
    fn permute_tensor_data_reorders_dense_tensor_bytes() {
        let input = int32_tensor(vec![2, 2], &[1, 2, 3, 4]);
        let output = permute_tensor_data(&input, &[1, 0]).expect("permute");
        let values = output
            .data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(values, vec![1, 3, 2, 4]);
    }

    #[test]
    fn slice_tensor_data_extracts_dense_axis_window() {
        let input = int32_tensor(vec![2, 3], &[1, 2, 3, 4, 5, 6]);
        let output = slice_tensor_data(&input, 1, 1, 3, 1).expect("slice");
        let values = output
            .data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(values, vec![2, 3, 5, 6]);
    }
}
