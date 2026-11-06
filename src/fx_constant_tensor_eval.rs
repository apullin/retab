use std::collections::{HashMap, HashSet};

use crate::constant_materialization::{quantize_scalar_with_qparam, scalar_storage_bytes};
use crate::fx_ir::{IrArg, IrGraph, IrNode, NodeOp, QParam, TensorSource};
use crate::fx_target::{shape_only_constant_target, ShapeOnlyConstantTarget};
use crate::graph::ir::{DType, QuantArgs};
use crate::lowering_effects::LoweringInputKind;
use crate::store::TensorData;

pub fn tensor_numel(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        let dim = usize::try_from(dim).ok()?;
        acc.checked_mul(dim)
    })
}

pub fn arg_has_node(arg: &IrArg) -> bool {
    match arg {
        IrArg::Node(_) => true,
        IrArg::List(values) => values.iter().any(arg_has_node),
        _ => false,
    }
}

pub fn node_has_tensor_inputs(node: &IrNode) -> bool {
    node.args.iter().any(arg_has_node) || node.kwargs.values().any(arg_has_node)
}

pub fn arg_nodes_are_constant(arg: &IrArg, constant_nodes: &HashSet<usize>) -> bool {
    match arg {
        IrArg::Node(node_id) => constant_nodes.contains(node_id),
        IrArg::List(values) => values
            .iter()
            .all(|value| arg_nodes_are_constant(value, constant_nodes)),
        _ => true,
    }
}

pub fn node_inputs_are_constant(node: &IrNode, constant_nodes: &HashSet<usize>) -> bool {
    node.args
        .iter()
        .all(|arg| arg_nodes_are_constant(arg, constant_nodes))
        && node
            .kwargs
            .values()
            .all(|arg| arg_nodes_are_constant(arg, constant_nodes))
}

pub fn placeholder_is_constant_input(node: &IrNode) -> bool {
    if !node.alive || node.op != NodeOp::Placeholder {
        return false;
    }
    matches!(
        node.tensor.as_ref().map(|tensor| &tensor.source),
        Some(TensorSource::ExportedProgramInput(binding))
            if matches!(
                binding.input_kind,
                LoweringInputKind::Parameter
                    | LoweringInputKind::Buffer
                    | LoweringInputKind::LiftedTensorConstant
            )
    )
}

/// Constant tensors known to a folding sweep: the graph's constant placeholders,
/// borrowed in place, plus the tensors the sweep planned itself.
#[derive(Default)]
pub struct ConstantTensors<'g> {
    seeds: HashMap<usize, &'g TensorData>,
    planned: HashMap<usize, TensorData>,
}

impl ConstantTensors<'_> {
    pub fn get(&self, id: &usize) -> Option<&TensorData> {
        self.planned.get(id).or_else(|| self.seeds.get(id).copied())
    }

    pub fn insert(&mut self, id: usize, tensor: TensorData) {
        self.planned.insert(id, tensor);
    }
}

pub fn constant_input_seeds(graph: &IrGraph) -> (HashSet<usize>, ConstantTensors<'_>) {
    let mut constant_nodes = HashSet::new();
    let mut constant_tensors = ConstantTensors::default();
    for node in graph
        .nodes
        .iter()
        .filter(|node| placeholder_is_constant_input(node))
    {
        constant_nodes.insert(node.id);
        if let Some(tensor) = node.tensor.as_ref() {
            constant_tensors.seeds.insert(node.id, &tensor.data);
        }
    }
    (constant_nodes, constant_tensors)
}

pub fn first_constant_input_tensor<'a>(
    node: &IrNode,
    constant_tensors: &'a ConstantTensors<'_>,
) -> Option<&'a TensorData> {
    node.args.iter().find_map(|arg| match arg {
        IrArg::Node(node_id) => constant_tensors.get(node_id),
        _ => None,
    })
}

pub fn plan_shape_only_tensor_from_constants(
    node: &IrNode,
    constant_tensors: &ConstantTensors<'_>,
) -> Option<TensorData> {
    first_constant_input_tensor(node, constant_tensors)
        .and_then(|input| plan_shape_only_tensor(node, input))
}

pub fn plan_constant_slice_tensor_from_constants(
    node: &IrNode,
    constant_tensors: &ConstantTensors<'_>,
) -> Option<TensorData> {
    first_constant_input_tensor(node, constant_tensors)
        .and_then(|input| plan_constant_slice_tensor(node, input))
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
    let spec = node.args.get(1)?.as_int_list()?;
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
    node.args
        .get(1)?
        .as_int_list()?
        .into_iter()
        .map(|dim| normalize_dim(dim, rank, false))
        .collect()
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

fn encode_tensor_values(values: &[f64], shape: Vec<i64>, dtype: DType) -> Option<TensorData> {
    let mut data = Vec::new();
    for &value in values {
        data.extend_from_slice(&scalar_storage_bytes(dtype, value)?);
    }
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
                crate::constant_materialization::quantize_value_stock(
                    *value,
                    qparam.scale,
                    qparam.zp,
                    qparam.qmin,
                    qparam.qmax,
                ) as f64
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
        out_coords[axis] = rem.checked_div(stride).unwrap_or(0);
        rem = rem.checked_rem(stride).unwrap_or(0);
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
    encode_tensor_values(&values, vec![m as i64, n as i64], DType::Float32)
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
    encode_tensor_values(
        &values,
        vec![batch as i64, m as i64, n as i64],
        DType::Float32,
    )
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
    constant_tensors: &'a ConstantTensors<'_>,
) -> Option<&'a TensorData> {
    match node.args.get(arg_index)? {
        IrArg::Node(node_id) => constant_tensors.get(node_id),
        _ => None,
    }
}

fn resolve_alpha_beta(node: &IrNode) -> (f64, f64) {
    let beta = node
        .arg_or_kwarg_scalar_literal_f64(3, "beta")
        .unwrap_or(1.0);
    let alpha = node
        .arg_or_kwarg_scalar_literal_f64(4, "alpha")
        .unwrap_or(1.0);
    (alpha, beta)
}

pub fn plan_constant_matmul_tensor(
    node: &IrNode,
    constant_tensors: &ConstantTensors<'_>,
) -> Option<TensorData> {
    let lhs_arg = usize::from(node.is_addmm());
    let rhs_arg = lhs_arg + 1;
    let lhs = constant_input_tensor(node, lhs_arg, constant_tensors)?;
    let rhs = constant_input_tensor(node, rhs_arg, constant_tensors)?;
    let lhs_qparam = node.meta.input_qparam_at(lhs_arg);
    let rhs_qparam = node.meta.input_qparam_at(rhs_arg);
    let lhs_values = decode_tensor_values(lhs, lhs_qparam)?;
    let rhs_values = decode_tensor_values(rhs, rhs_qparam)?;

    let mut output = if node.is_mm() || node.is_addmm() {
        matmul_2d(&lhs_values, &lhs.shape, &rhs_values, &rhs.shape)?
    } else if node.is_bmm() {
        batched_matmul_3d(&lhs_values, &lhs.shape, &rhs_values, &rhs.shape)?
    } else {
        return None;
    };

    if node.is_addmm() {
        let bias = constant_input_tensor(node, 0, constant_tensors)?;
        let bias_qparam = node.meta.input_qparam();
        let bias_values = decode_tensor_values(bias, bias_qparam)?;
        let (alpha, beta) = resolve_alpha_beta(node);
        let mut values = output.vec_f64()?;
        if (alpha - 1.0).abs() > 1e-10 {
            for value in &mut values {
                *value *= alpha;
            }
        }
        add_broadcasted_bias(&mut values, &output.shape, &bias_values, &bias.shape, beta)?;
        output = encode_tensor_values(&values, output.shape, DType::Float32)?;
    }

    if !output_numel_allowed(node, &output.shape)? {
        return None;
    }

    if let Some(output_qparam) = node.meta.output_qparam() {
        let output_values = quantize_values(&output.vec_f64()?, output_qparam)?;
        let output_dtype = output_qparam.dtype();
        return encode_tensor_values(&output_values, output.shape, output_dtype);
    }

    let output_dtype = node
        .meta
        .val_dtype()
        .map(DType::from_torch_str)
        .unwrap_or(lhs.dtype);
    if output_dtype != DType::Float32 {
        return None;
    }
    encode_tensor_values(&output.vec_f64()?, output.shape, output_dtype)
}

/// Dequantize with stock FuseConstantArgsPass semantics: `(q - zp) * scale`
/// per tensor, or per channel slice along the qparam axis for per-channel
/// weights. Empty per-channel zero points mean all zeros.
fn dequantize_stock_values(tensor: &TensorData, qparam: Option<&QParam>) -> Option<Vec<f64>> {
    let values = tensor.vec_f64()?;
    let Some(qparam) = qparam else {
        return Some(values);
    };
    if !qparam.is_per_channel() {
        return Some(
            values
                .into_iter()
                .map(|value| (value - qparam.zp as f64) * qparam.scale)
                .collect(),
        );
    }
    let axis = normalize_dim(qparam.axis, tensor.shape.len(), false)?;
    let channels = usize::try_from(*tensor.shape.get(axis)?).ok()?;
    if qparam.per_channel_scales.len() != channels {
        return None;
    }
    let zero_points = if qparam.per_channel_zps.is_empty() {
        vec![0i64; channels]
    } else if qparam.per_channel_zps.len() == channels {
        qparam.per_channel_zps.clone()
    } else {
        return None;
    };
    let mut axis_stride = 1usize;
    for dim in tensor.shape.iter().skip(axis + 1) {
        axis_stride = axis_stride.checked_mul(usize::try_from(*dim).ok()?)?;
    }
    Some(
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let channel = (index / axis_stride) % channels.max(1);
                (value - zero_points[channel] as f64) * qparam.per_channel_scales[channel]
            })
            .collect(),
    )
}

/// Shared tail for the linear/conv folds: stock's numel rule, then quantize
/// with the output qparam (or keep float32 when unquantized, like stock).
fn finish_stock_fold(
    node: &IrNode,
    values: Vec<f64>,
    shape: Vec<i64>,
    fallback_dtype: DType,
) -> Option<TensorData> {
    if values.iter().any(|value| value.is_nan()) {
        return None;
    }
    if !output_numel_allowed(node, &shape)? {
        return None;
    }
    if let Some(output_qparam) = node.meta.output_qparam() {
        let quantized = quantize_values(&values, output_qparam)?;
        return encode_tensor_values(&quantized, shape, output_qparam.dtype());
    }
    let output_dtype = node
        .meta
        .val_dtype()
        .map(DType::from_torch_str)
        .unwrap_or(fallback_dtype);
    if output_dtype != DType::Float32 {
        return None;
    }
    encode_tensor_values(&values, shape, output_dtype)
}

/// Fold `aten.linear.default(x, weight, bias?)` with all-constant inputs:
/// dequantize each input with its input qparam, compute `x @ W^T + b` in the
/// same order torch does, and quantize with the output qparam.
pub fn plan_constant_linear_tensor(
    node: &IrNode,
    constant_tensors: &ConstantTensors<'_>,
) -> Option<TensorData> {
    if !node.is_linear() || !node.kwargs.is_empty() {
        return None;
    }
    let input = constant_input_tensor(node, 0, constant_tensors)?;
    let weight = constant_input_tensor(node, 1, constant_tensors)?;
    if weight.shape.len() != 2 || input.shape.last() != weight.shape.last() {
        return None;
    }
    let input_values = dequantize_stock_values(input, node.meta.input_qparam_at(0))?;
    let weight_values = dequantize_stock_values(weight, node.meta.input_qparam_at(1))?;
    if input_values
        .iter()
        .chain(weight_values.iter())
        .any(|v| v.is_nan())
    {
        return None;
    }
    let out_features = usize::try_from(weight.shape[0]).ok()?;
    let in_features = usize::try_from(weight.shape[1]).ok()?;
    let mut out_shape = input.shape.clone();
    *out_shape.last_mut()? = weight.shape[0];
    let out_numel = tensor_numel(&out_shape)?;
    if input_values.len() != out_numel / out_features.max(1) * in_features
        || weight_values.len() != out_features.checked_mul(in_features)?
    {
        return None;
    }
    let mut values = vec![0.0; out_numel];
    for (index, out) in values.iter_mut().enumerate() {
        let row = index / out_features;
        let col = index % out_features;
        let mut acc = 0.0;
        for k in 0..in_features {
            acc += input_values[row * in_features + k] * weight_values[col * in_features + k];
        }
        *out = acc;
    }
    match node.args.get(2) {
        None | Some(IrArg::None) => {}
        Some(IrArg::Node(bias_id)) => {
            let bias = constant_tensors.get(bias_id)?;
            let bias_values = dequantize_stock_values(bias, node.meta.input_qparam_at(2))?;
            add_broadcasted_bias(&mut values, &out_shape, &bias_values, &bias.shape, 1.0)?;
        }
        Some(_) => return None,
    }
    finish_stock_fold(node, values, out_shape, input.dtype)
}

/// Fold `aten.convolution`/convNd with all-constant inputs the way stock runs
/// the whole op in float32: direct convolution over dequantized inputs plus
/// the output-channel bias, then quantize with the output qparam. Transposed
/// convs and output padding stay runtime ops.
pub fn plan_constant_conv_tensor(
    node: &IrNode,
    constant_tensors: &ConstantTensors<'_>,
) -> Option<TensorData> {
    if !node.kwargs.is_empty() {
        return None;
    }
    let conv = crate::fx_conv_args::ConvArgs::parse(node)?;
    if conv.transposed || conv.output_padding.iter().any(|&pad| pad != 0) || !conv.bias_well_formed
    {
        return None;
    }
    if conv.stride.iter().any(|&s| s < 1)
        || conv.padding.iter().any(|&p| p < 0)
        || conv.dilation.iter().any(|&d| d < 1)
    {
        return None;
    }
    let input = constant_tensors.get(&conv.input)?;
    let weight = constant_tensors.get(&conv.weight)?;
    if input.shape.len() < 3 || input.shape.len() != weight.shape.len() {
        return None;
    }
    let spatial_rank = input.shape.len() - 2;
    if conv.stride.len() != spatial_rank
        || conv.padding.len() != spatial_rank
        || conv.dilation.len() != spatial_rank
    {
        return None;
    }
    let input_values = dequantize_stock_values(input, node.meta.input_qparam_at(0))?;
    let weight_values = dequantize_stock_values(weight, node.meta.input_qparam_at(1))?;
    if input_values
        .iter()
        .chain(weight_values.iter())
        .any(|v| v.is_nan())
    {
        return None;
    }
    let batch = usize::try_from(*input.shape.first()?).ok()?;
    let in_channels = usize::try_from(input.shape[1]).ok()?;
    let out_channels = usize::try_from(*weight.shape.first()?).ok()?;
    let groups = usize::try_from(conv.groups).ok()?;
    if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
        return None;
    }
    if usize::try_from(weight.shape[1]).ok()? != in_channels / groups {
        return None;
    }
    let in_spatial: Vec<usize> = input.shape[2..]
        .iter()
        .map(|dim| usize::try_from(*dim))
        .collect::<Result<_, _>>()
        .ok()?;
    let kernel: Vec<usize> = weight.shape[2..]
        .iter()
        .map(|dim| usize::try_from(*dim))
        .collect::<Result<_, _>>()
        .ok()?;
    if in_spatial.contains(&0) || kernel.contains(&0) {
        return None;
    }
    let mut out_spatial = Vec::with_capacity(spatial_rank);
    for dim in 0..spatial_rank {
        let numerator = in_spatial[dim] as i64 + 2 * conv.padding[dim]
            - conv.dilation[dim] * (kernel[dim] as i64 - 1)
            - 1;
        if numerator < 0 || conv.stride[dim] <= 0 {
            return None;
        }
        out_spatial.push((numerator / conv.stride[dim] + 1) as usize);
    }
    let bias_values = match conv.bias {
        None => None,
        Some(bias_id) => {
            let bias = constant_tensors.get(&bias_id)?;
            if bias.shape != [out_channels as i64] {
                return None;
            }
            Some(dequantize_stock_values(bias, node.meta.input_qparam_at(2))?)
        }
    };
    let in_spatial_numel: usize = in_spatial.iter().product();
    let out_spatial_numel: usize = out_spatial.iter().product();
    let kernel_numel: usize = kernel.iter().product();
    if input_values.len()
        != batch
            .checked_mul(in_channels)?
            .checked_mul(in_spatial_numel)?
        || weight_values.len()
            != out_channels
                .checked_mul(in_channels / groups)?
                .checked_mul(kernel_numel)?
    {
        return None;
    }
    let group_in = in_channels / groups;
    let group_out = out_channels / groups;
    let mut spatial_strides = vec![1usize; spatial_rank];
    for dim in (0..spatial_rank.saturating_sub(1)).rev() {
        spatial_strides[dim] = spatial_strides[dim + 1].checked_mul(out_spatial[dim + 1])?;
    }
    let mut in_strides = vec![1usize; spatial_rank];
    for dim in (0..spatial_rank.saturating_sub(1)).rev() {
        in_strides[dim] = in_strides[dim + 1].checked_mul(in_spatial[dim + 1])?;
    }
    let mut kernel_strides = vec![1usize; spatial_rank];
    for dim in (0..spatial_rank.saturating_sub(1)).rev() {
        kernel_strides[dim] = kernel_strides[dim + 1].checked_mul(kernel[dim + 1])?;
    }
    let mut values = Vec::with_capacity(
        batch
            .checked_mul(out_channels)?
            .checked_mul(out_spatial_numel)?,
    );
    for n in 0..batch {
        for co in 0..out_channels {
            let group = co / group_out;
            for out_flat in 0..out_spatial_numel {
                // Start from the bias, then accumulate (ci, kernel) products
                // in dense order like a direct convolution.
                let mut acc = bias_values.as_ref().map(|bias| bias[co]).unwrap_or(0.0);
                for ci in 0..group_in {
                    for k_flat in 0..kernel_numel {
                        let mut in_offset =
                            (n * in_channels + group * group_in + ci) * in_spatial_numel;
                        let mut inside = true;
                        for dim in 0..spatial_rank {
                            let out_coord = out_flat / spatial_strides[dim] % out_spatial[dim];
                            let k_coord = k_flat / kernel_strides[dim] % kernel[dim];
                            let pos = out_coord as i64 * conv.stride[dim]
                                + k_coord as i64 * conv.dilation[dim]
                                - conv.padding[dim];
                            if pos < 0 || pos >= in_spatial[dim] as i64 {
                                inside = false;
                                break;
                            }
                            in_offset += (pos as usize).checked_mul(in_strides[dim])?;
                        }
                        if inside {
                            let w_flat = (co * group_in + ci) * kernel_numel + k_flat;
                            acc += input_values[in_offset] * weight_values[w_flat];
                        }
                    }
                }
                values.push(acc);
            }
        }
    }
    let mut out_shape = vec![batch as i64, out_channels as i64];
    out_shape.extend(out_spatial.iter().map(|dim| *dim as i64));
    finish_stock_fold(node, values, out_shape, input.dtype)
}

pub fn permute_tensor_data(input: &TensorData, order: &[usize]) -> Option<TensorData> {
    let rank = input.shape.len();
    if rank != order.len() {
        return None;
    }
    if order.iter().enumerate().all(|(idx, &axis)| idx == axis) {
        return Some(input.clone());
    }
    let mut seen = vec![false; rank];
    for &axis in order {
        if axis >= rank || std::mem::replace(&mut seen[axis], true) {
            return None;
        }
    }

    let numel = tensor_numel(&input.shape)?;
    let elem_size = input.dtype.storage_size_bytes();
    if numel.saturating_mul(elem_size) != input.data.len() {
        return None;
    }
    let new_shape: Vec<i64> = order.iter().map(|&axis| input.shape[axis]).collect();
    if numel == 0 || elem_size == 0 {
        return Some(TensorData {
            data: input.data.clone(),
            shape: new_shape,
            dtype: input.dtype,
        });
    }

    // The serializer's odometer gather: no per-element division, and no byte
    // movement when only size-1 axes move (a 1x1 conv's OIHW -> OHWI).
    let order_u8: Vec<u8> = order
        .iter()
        .map(|&axis| u8::try_from(axis).ok())
        .collect::<Option<_>>()?;
    let (data, shape) = crate::serialize::tensor_layout::permute_tensor_by_order(
        &input.data,
        &input.shape,
        elem_size,
        &order_u8,
    );
    debug_assert_eq!(shape, new_shape);
    Some(TensorData {
        data,
        shape: new_shape,
        dtype: input.dtype,
    })
}

pub fn plan_shape_only_tensor(node: &IrNode, input: &TensorData) -> Option<TensorData> {
    match shape_only_constant_target(&node.target)? {
        ShapeOnlyConstantTarget::Reshape => Some(TensorData {
            data: input.data.clone(),
            shape: reshape_target_shape(&input.shape, node)?,
            dtype: input.dtype,
        }),
        ShapeOnlyConstantTarget::Unsqueeze => Some(TensorData {
            data: input.data.clone(),
            shape: unsqueeze_target_shape(&input.shape, node)?,
            dtype: input.dtype,
        }),
        ShapeOnlyConstantTarget::Permute => {
            permute_tensor_data(input, &permute_order(node, input.shape.len())?)
        }
    }
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
    let elem_size = input.dtype.storage_size_bytes();
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
            out_coords[axis] = rem.checked_div(stride).unwrap_or(0);
            rem = rem.checked_rem(stride).unwrap_or(0);
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

fn cast_output_dtype(node: &IrNode, input_dtype: DType) -> Option<DType> {
    match node.kwargs.get("dtype") {
        None | Some(IrArg::None) => Some(input_dtype),
        Some(IrArg::Dtype(name)) | Some(IrArg::Str(name)) => Some(DType::from_torch_str(name)),
        Some(_) => None,
    }
}

fn decode_scalar(tensor: &TensorData, linear: usize) -> Option<f64> {
    let elem_size = tensor.dtype.storage_size_bytes();
    let bytes = tensor
        .data
        .get(linear.checked_mul(elem_size)?..)?
        .get(..elem_size)?;
    let probe = TensorData {
        data: bytes.to_vec(),
        shape: vec![1],
        dtype: tensor.dtype,
    };
    probe.vec_f64()?.into_iter().next()
}

/// Encode one value for a `_to_copy` fold, succeeding only when the value
/// survives the cast exactly. Bool accepts anything (nonzero maps to true);
/// every other dtype must round-trip, so saturating or truncating casts stay
/// runtime ops instead of silently changing values.
fn encode_cast_scalar(dtype: DType, value: f64) -> Option<Vec<u8>> {
    let bytes = scalar_storage_bytes(dtype, value)?;
    if dtype == DType::Bool {
        return Some(bytes);
    }
    let roundtrip = decode_scalar(
        &TensorData {
            data: bytes.clone(),
            shape: vec![1],
            dtype,
        },
        0,
    )?;
    (roundtrip == value).then_some(bytes)
}

/// Fold a dtype cast (`aten._to_copy`, dim-order copies) whose single tensor
/// input is constant. Like stock's FuseConstantArgsPass this covers the plain
/// cast; quantized re-scales (qparams on either side) stay runtime ops.
pub fn plan_constant_cast_tensor(node: &IrNode, input: &TensorData) -> Option<TensorData> {
    if node.meta.input_qparams().is_some() || node.meta.output_qparams().is_some() {
        return None;
    }
    let dtype = cast_output_dtype(node, input.dtype)?;
    let values = input.vec_f64()?;
    let mut data = Vec::new();
    for value in values {
        data.extend_from_slice(&encode_cast_scalar(dtype, value)?);
    }
    Some(TensorData {
        data,
        shape: input.shape.clone(),
        dtype,
    })
}

fn comparison_int_values(tensor: &TensorData) -> Option<Vec<i64>> {
    match tensor.dtype {
        DType::Bool | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => tensor.vec_i64(),
        _ => None,
    }
}

fn comparison_float_values(tensor: &TensorData) -> Option<Vec<f64>> {
    match tensor.dtype {
        DType::Float32 | DType::Float16 | DType::BFloat16 => tensor.vec_f64(),
        _ => None,
    }
}

fn compare_floats(target: &str, lhs: f64, rhs: f64) -> Option<bool> {
    match target {
        "aten.gt.Tensor" => Some(lhs > rhs),
        "aten.ge.Tensor" => Some(lhs >= rhs),
        "aten.lt.Tensor" => Some(lhs < rhs),
        "aten.le.Tensor" => Some(lhs <= rhs),
        "aten.eq.Tensor" => Some(lhs == rhs),
        _ => None,
    }
}

fn compare_ints(target: &str, lhs: i64, rhs: i64) -> Option<bool> {
    match target {
        "aten.gt.Tensor" => Some(lhs > rhs),
        "aten.ge.Tensor" => Some(lhs >= rhs),
        "aten.lt.Tensor" => Some(lhs < rhs),
        "aten.le.Tensor" => Some(lhs <= rhs),
        "aten.eq.Tensor" => Some(lhs == rhs),
        _ => None,
    }
}

/// Fold an elementwise comparison (`aten.gt/ge/lt/le/eq.Tensor`) whose tensor
/// inputs are all constant. Integer pairs compare exactly as i64; float pairs
/// compare as f64, which embeds f32/f16/bf16 exactly. Quantized (qparam
/// annotated) and mixed-dtype comparisons stay runtime ops.
pub fn plan_constant_comparison_tensors(
    node: &IrNode,
    lhs: &TensorData,
    rhs: &TensorData,
) -> Option<TensorData> {
    if node.meta.input_qparams().is_some() || node.meta.output_qparams().is_some() {
        return None;
    }
    let shape = crate::fx_ir::broadcast_shapes(&[lhs.shape.clone(), rhs.shape.clone()])?;
    let numel = tensor_numel(&shape)?;
    if let Some(fake) = node.meta.val_shape() {
        if numel > tensor_numel(fake)? {
            return None;
        }
    }

    enum Operands {
        Ints(Vec<i64>, Vec<i64>),
        Floats(Vec<f64>, Vec<f64>),
    }
    let operands = match (comparison_int_values(lhs), comparison_int_values(rhs)) {
        (Some(lhs_values), Some(rhs_values)) => Operands::Ints(lhs_values, rhs_values),
        _ => Operands::Floats(comparison_float_values(lhs)?, comparison_float_values(rhs)?),
    };
    let mut data = Vec::with_capacity(numel);
    for idx in 0..numel {
        let lhs_idx = broadcast_offset(idx, &shape, &lhs.shape)?;
        let rhs_idx = broadcast_offset(idx, &shape, &rhs.shape)?;
        let result = match &operands {
            Operands::Ints(lhs_values, rhs_values) => compare_ints(
                &node.target,
                lhs_values.get(lhs_idx).copied()?,
                rhs_values.get(rhs_idx).copied()?,
            )?,
            Operands::Floats(lhs_values, rhs_values) => compare_floats(
                &node.target,
                lhs_values.get(lhs_idx).copied()?,
                rhs_values.get(rhs_idx).copied()?,
            )?,
        };
        data.push(u8::from(result));
    }
    Some(TensorData {
        data,
        shape,
        dtype: DType::Bool,
    })
}

fn per_tensor_quant_args(qp: &QParam) -> Option<QuantArgs> {
    if qp.is_per_channel() {
        return None;
    }
    Some(QuantArgs {
        scale: qp.scale,
        zero_point: qp.zp,
        dtype: qp.dtype(),
        qmin: qp.qmin,
        qmax: qp.qmax,
    })
}

/// Fold `aten.reciprocal` on a constant. The input is dequantized with the
/// node's input qparam when present, the reciprocal runs in float32 like
/// torch, and a quantized output goes through the faithful per-tensor
/// quantize. NaN inputs stay runtime ops; infinities clamp like stock.
pub fn plan_constant_reciprocal_tensor(node: &IrNode, input: &TensorData) -> Option<TensorData> {
    if let Some(qparams) = node.meta.input_qparams() {
        if !qparams.is_empty() && node.meta.input_qparam_at(0).is_none() {
            return None;
        }
    }
    let input_qparam = node.meta.input_qparam_at(0);
    if input_qparam.is_some_and(|qp| qp.is_per_channel()) {
        return None;
    }
    let raw = input.vec_f64()?;
    let base: Vec<f64> = match input_qparam {
        Some(qp) => raw
            .into_iter()
            .map(|value| (value - qp.zp as f64) * qp.scale)
            .collect(),
        None => raw,
    };
    if base.iter().any(|value| value.is_nan()) {
        return None;
    }
    let values: Vec<f64> = base
        .into_iter()
        .map(|value| (value as f32).recip() as f64)
        .collect();

    if let Some(output_qparam) = node.meta.output_qparam() {
        let quant_args = per_tensor_quant_args(output_qparam)?;
        if !output_qparam.is_integer_dtype() {
            return None;
        }
        let dtype = output_qparam.dtype();
        let mut data = Vec::new();
        for value in values {
            let quantized = quantize_scalar_with_qparam(value, quant_args);
            data.extend_from_slice(&scalar_storage_bytes(dtype, quantized)?);
        }
        let shape = input.shape.clone();
        if let Some(fake) = node.meta.val_shape() {
            if tensor_numel(&shape)? > tensor_numel(fake)? {
                return None;
            }
        }
        return Some(TensorData { data, shape, dtype });
    }

    match input.dtype {
        DType::Float32 | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => {}
        _ => return None,
    }
    let mut data = Vec::new();
    for value in values {
        data.extend_from_slice(&(value as f32).to_le_bytes());
    }
    let shape = input.shape.clone();
    if let Some(fake) = node.meta.val_shape() {
        if tensor_numel(&shape)? > tensor_numel(fake)? {
            return None;
        }
    }
    Some(TensorData {
        data,
        shape,
        dtype: DType::Float32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_meta::NodeMeta;

    fn int32_tensor(shape: Vec<i64>, values: &[i32]) -> TensorData {
        TensorData {
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape,
            dtype: DType::Int32,
        }
    }

    fn node_tensor(graph: &IrGraph, id: usize) -> TensorData {
        graph.node(id).tensor.as_ref().unwrap().data.clone()
    }

    /// The fast gather equals the per-element reference it replaced, over seeded
    /// random shapes (with size-1 axes), permutations and element sizes.
    #[test]
    fn permute_tensor_data_matches_a_per_element_reference() {
        fn reference(input: &TensorData, order: &[usize]) -> Vec<u8> {
            let rank = input.shape.len();
            let shape: Vec<usize> = input.shape.iter().map(|&d| d as usize).collect();
            let new_shape: Vec<usize> = order.iter().map(|&a| shape[a]).collect();
            let elem = input.dtype.storage_size_bytes();
            let strides = |dims: &[usize]| {
                let mut s = vec![1usize; rank];
                for i in (0..rank.saturating_sub(1)).rev() {
                    s[i] = s[i + 1] * dims[i + 1];
                }
                s
            };
            let (in_strides, out_strides) = (strides(&shape), strides(&new_shape));
            let mut out = vec![0u8; input.data.len()];
            for src in 0..input.data.len() / elem {
                let coords: Vec<usize> = (0..rank).map(|a| (src / in_strides[a]) % shape[a]).collect();
                let dst: usize = order.iter().enumerate().map(|(o, &a)| coords[a] * out_strides[o]).sum();
                out[dst * elem..(dst + 1) * elem].copy_from_slice(&input.data[src * elem..(src + 1) * elem]);
            }
            out
        }
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        for case in 0..400 {
            let rank = 1 + next(5) as usize;
            let shape: Vec<i64> = (0..rank).map(|_| 1 + next(4) as i64).collect();
            let mut order: Vec<usize> = (0..rank).collect();
            for i in (1..rank).rev() {
                order.swap(i, next(i as u64 + 1) as usize);
            }
            let dtype = [DType::Int8, DType::Int16, DType::Int32, DType::Int64][case % 4];
            let numel: i64 = shape.iter().product();
            let bytes = numel as usize * dtype.storage_size_bytes();
            let input = TensorData {
                data: (0..bytes).map(|_| next(256) as u8).collect(),
                shape: shape.clone(),
                dtype,
            };
            let output = permute_tensor_data(&input, &order).expect("permute");
            assert_eq!(output.data, reference(&input, &order), "case {case}: {shape:?} {order:?}");
            assert_eq!(output.shape, order.iter().map(|&a| shape[a]).collect::<Vec<_>>());
        }
        assert!(permute_tensor_data(&int32_tensor(vec![2, 2], &[1, 2, 3, 4]), &[0, 0]).is_none());
    }

    #[test]
    fn permute_tensor_data_reorders_dense_tensor_bytes() {
        let input = int32_tensor(vec![2, 2], &[1, 2, 3, 4]);
        let output = permute_tensor_data(&input, &[1, 0]).expect("permute");
        let values = output
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| i32::from_le_bytes(*chunk))
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| i32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(values, vec![2, 3, 5, 6]);
    }

    fn cast_node(target: &str, dtype: Option<&str>) -> (IrGraph, usize) {
        let mut graph = IrGraph::empty();
        let weight =
            crate::test_ir::parameter(&mut graph, "weight", int32_tensor(vec![2], &[9, -3]));
        let id = graph.add_node(
            NodeOp::CallFunction,
            "cast".to_string(),
            target.to_string(),
            vec![IrArg::Node(weight)],
            NodeMeta::tensor(vec![2], "torch.int32"),
        );
        if let Some(dtype) = dtype {
            graph.set_kwarg(id, "dtype", IrArg::Dtype(dtype.to_string()));
        }
        (graph, id)
    }

    #[test]
    fn cast_folds_exact_int_narrowing() {
        let (graph, id) = cast_node("aten._to_copy.default", Some("torch.int8"));
        let input = node_tensor(&graph, 0);
        let output = plan_constant_cast_tensor(graph.node(id), &input).expect("cast");
        assert_eq!(output.dtype, DType::Int8);
        assert_eq!(output.shape, vec![2]);
        assert_eq!(output.data, vec![9u8, (-3i8) as u8]);
    }

    #[test]
    fn cast_blocks_saturating_narrowing() {
        let mut graph = IrGraph::empty();
        let big = crate::test_ir::parameter(
            &mut graph,
            "big",
            TensorData {
                data: 300i32.to_le_bytes().to_vec(),
                shape: vec![1],
                dtype: DType::Int32,
            },
        );
        let id = graph.add_node(
            NodeOp::CallFunction,
            "cast".to_string(),
            "aten._to_copy.default".to_string(),
            vec![IrArg::Node(big)],
            NodeMeta::tensor(vec![1], "torch.int8"),
        );
        graph.set_kwarg(id, "dtype", IrArg::Dtype("torch.int8".to_string()));
        let input = node_tensor(&graph, big);
        assert!(plan_constant_cast_tensor(graph.node(id), &input).is_none());
    }

    #[test]
    fn cast_blocks_quantized_rescales() {
        let (mut graph, id) = cast_node("aten._to_copy.default", Some("torch.int8"));
        graph
            .node_mut(id)
            .meta
            .set_single_input_qparam(QParam::per_tensor(
                0.5,
                0,
                -128,
                127,
                "torch.int8".to_string(),
            ));
        let input = node_tensor(&graph, 0);
        assert!(plan_constant_cast_tensor(graph.node(id), &input).is_none());
    }

    fn comparison_graph(
        target: &str,
        lhs: TensorData,
        rhs: TensorData,
        shape: Vec<i64>,
    ) -> (IrGraph, usize) {
        let mut graph = IrGraph::empty();
        let lhs_id = crate::test_ir::parameter(&mut graph, "lhs", lhs);
        let rhs_id = crate::test_ir::parameter(&mut graph, "rhs", rhs);
        let id = graph.add_node(
            NodeOp::CallFunction,
            "cmp".to_string(),
            target.to_string(),
            vec![IrArg::Node(lhs_id), IrArg::Node(rhs_id)],
            NodeMeta::tensor(shape, "torch.bool"),
        );
        (graph, id)
    }

    #[test]
    fn comparison_folds_broadcast_int_gt() {
        let (graph, id) = comparison_graph(
            "aten.gt.Tensor",
            int32_tensor(vec![1, 1], &[0]),
            int32_tensor(vec![1, 1], &[9]),
            vec![1, 1],
        );
        let lhs_tensor = node_tensor(&graph, 0);
        let rhs_tensor = node_tensor(&graph, 1);
        let output =
            plan_constant_comparison_tensors(graph.node(id), &lhs_tensor, &rhs_tensor).expect("gt");
        assert_eq!(output.dtype, DType::Bool);
        assert_eq!(output.shape, vec![1, 1]);
        assert_eq!(output.data, vec![0]);
    }

    #[test]
    fn comparison_rejects_mixed_dtypes() {
        let (graph, id) = comparison_graph(
            "aten.gt.Tensor",
            int32_tensor(vec![1], &[1]),
            TensorData {
                data: 1.0f32.to_le_bytes().to_vec(),
                shape: vec![1],
                dtype: DType::Float32,
            },
            vec![1],
        );
        let lhs = node_tensor(&graph, 0);
        let rhs = node_tensor(&graph, 1);
        assert!(plan_constant_comparison_tensors(graph.node(id), &lhs, &rhs).is_none());
    }

    #[test]
    fn reciprocal_folds_quantized_output_faithfully() {
        let mut graph = IrGraph::empty();
        let input_id = crate::test_ir::parameter(
            &mut graph,
            "divisor",
            TensorData {
                data: 2.0f32.to_le_bytes().to_vec(),
                shape: vec![1, 1],
                dtype: DType::Float32,
            },
        );
        let mut meta = NodeMeta::tensor(vec![1, 1], "torch.float32");
        meta.set_single_output_qparam(QParam::per_tensor(
            0.5,
            0,
            -128,
            127,
            "torch.int8".to_string(),
        ));
        let id = graph.add_node(
            NodeOp::CallFunction,
            "recip".to_string(),
            "aten.reciprocal.default".to_string(),
            vec![IrArg::Node(input_id)],
            meta,
        );
        let input = node_tensor(&graph, input_id);
        let output = plan_constant_reciprocal_tensor(graph.node(id), &input).expect("recip");
        assert_eq!(output.dtype, DType::Int8);
        // 1/2 = 0.5 in float32, quantized at scale 0.5 -> 1.
        assert_eq!(output.data, vec![1u8]);
    }

    #[test]
    fn reciprocal_blocks_nan() {
        let mut graph = IrGraph::empty();
        let input_id = crate::test_ir::parameter(
            &mut graph,
            "nan",
            TensorData {
                data: f32::NAN.to_le_bytes().to_vec(),
                shape: vec![1],
                dtype: DType::Float32,
            },
        );
        let id = graph.add_node(
            NodeOp::CallFunction,
            "recip".to_string(),
            "aten.reciprocal.default".to_string(),
            vec![IrArg::Node(input_id)],
            NodeMeta::tensor(vec![1], "torch.float32"),
        );
        let input = node_tensor(&graph, input_id);
        assert!(plan_constant_reciprocal_tensor(graph.node(id), &input).is_none());
    }

    fn i8_tensor(shape: Vec<i64>, values: &[i8]) -> TensorData {
        TensorData {
            data: values.iter().map(|value| *value as u8).collect(),
            shape,
            dtype: DType::Int8,
        }
    }

    fn per_tensor_qparam(scale: f64, zp: i64) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, "torch.int8".to_string())
    }

    fn per_channel_qparam(scales: &[f64], zps: &[i64]) -> QParam {
        QParam {
            scale: 1.0,
            zp: 0,
            qmin: -128,
            qmax: 127,
            dtype_str: "torch.int8".to_string(),
            per_channel_scales: scales.to_vec(),
            per_channel_zps: zps.to_vec(),
            axis: 0,
        }
    }

    fn constant_map(graph: &IrGraph) -> ConstantTensors<'static> {
        let mut map = ConstantTensors::default();
        for node in graph.nodes.iter().filter(|node| node.op == NodeOp::Placeholder) {
            map.insert(node.id, node.tensor.as_ref().unwrap().data.clone());
        }
        map
    }

    #[allow(clippy::too_many_arguments)]
    fn quantized_linear_graph(
        input: TensorData,
        input_qp: QParam,
        weight: TensorData,
        weight_qp: QParam,
        bias: Option<TensorData>,
        bias_qp: Option<QParam>,
        output_qp: QParam,
        out_shape: Vec<i64>,
    ) -> (IrGraph, usize) {
        let mut graph = IrGraph::empty();
        let input_id = crate::test_ir::parameter(&mut graph, "input", input);
        let weight_id = crate::test_ir::parameter(&mut graph, "weight", weight);
        let mut args = vec![IrArg::Node(input_id), IrArg::Node(weight_id)];
        if let Some(bias_data) = bias {
            args.push(IrArg::Node(crate::test_ir::parameter(
                &mut graph, "bias", bias_data,
            )));
        } else {
            args.push(IrArg::None);
        }
        let mut meta = NodeMeta::tensor(out_shape, "torch.float32");
        // NOTE: set_input_qparam_at replaces the whole map, so all entries
        // must be set in one call.
        let mut input_qps = vec![(0, input_qp), (1, weight_qp)];
        if let Some(bias_qp) = bias_qp {
            input_qps.push((2, bias_qp));
        }
        meta.set_input_qparams_from(input_qps);
        meta.set_single_output_qparam(output_qp);
        let id = graph.add_node(
            NodeOp::CallFunction,
            "linear".to_string(),
            "aten.linear.default".to_string(),
            args,
            meta,
        );
        (graph, id)
    }

    #[test]
    fn linear_folds_zero_input_to_biased_output() {
        // h0-style zero input: dequantized input is all zeros, so the output
        // is the dequantized bias, requantized: 2.0 / 0.5 = 4.
        let (graph, id) = quantized_linear_graph(
            i8_tensor(vec![1, 2], &[-128, -128]),
            per_tensor_qparam(0.5, -128),
            i8_tensor(vec![1, 2], &[10, 20]),
            per_tensor_qparam(0.25, 0),
            Some(i8_tensor(vec![1], &[8])),
            Some(per_tensor_qparam(0.25, 0)),
            per_tensor_qparam(0.5, 0),
            vec![1, 1],
        );
        let output =
            plan_constant_linear_tensor(graph.node(id), &constant_map(&graph)).expect("linear");
        assert_eq!(output.dtype, DType::Int8);
        assert_eq!(output.shape, vec![1, 1]);
        assert_eq!(output.data, vec![4u8]);
    }

    #[test]
    fn linear_folds_nonzero_input_without_bias() {
        // out = 1.0 * 0.5 + 2.0 * 0.25 = 1.0, quantized at scale 0.5 -> 2.
        let (graph, id) = quantized_linear_graph(
            i8_tensor(vec![1, 2], &[2, 4]),
            per_tensor_qparam(0.5, 0),
            i8_tensor(vec![1, 2], &[2, 1]),
            per_tensor_qparam(0.25, 0),
            None,
            None,
            per_tensor_qparam(0.5, 0),
            vec![1, 1],
        );
        let output =
            plan_constant_linear_tensor(graph.node(id), &constant_map(&graph)).expect("linear");
        assert_eq!(output.dtype, DType::Int8);
        assert_eq!(output.shape, vec![1, 1]);
        assert_eq!(output.data, vec![2u8]);
    }

    #[test]
    fn linear_folds_per_channel_weights() {
        // Rows dequantize with their own scale: [0.5, 0.0] and [0.0, 1.0];
        // out = [1.0 * 0.5, 3.0 * 1.0] = [0.5, 3.0] -> [1, 6] at scale 0.5.
        let (graph, id) = quantized_linear_graph(
            i8_tensor(vec![1, 2], &[2, 6]),
            per_tensor_qparam(0.5, 0),
            i8_tensor(vec![2, 2], &[2, 0, 0, 2]),
            per_channel_qparam(&[0.25, 0.5], &[0, 0]),
            None,
            None,
            per_tensor_qparam(0.5, 0),
            vec![1, 2],
        );
        let output =
            plan_constant_linear_tensor(graph.node(id), &constant_map(&graph)).expect("linear");
        assert_eq!(output.dtype, DType::Int8);
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(output.data, vec![1u8, 6u8]);
    }

    #[test]
    fn linear_rejects_oversized_output() {
        // Computed [1, 2] is larger than the node's own [1, 1] fake output.
        let (graph, id) = quantized_linear_graph(
            i8_tensor(vec![1, 2], &[2, 4]),
            per_tensor_qparam(0.5, 0),
            i8_tensor(vec![2, 2], &[2, 1, 4, 0]),
            per_tensor_qparam(0.25, 0),
            None,
            None,
            per_tensor_qparam(0.5, 0),
            vec![1, 1],
        );
        assert!(plan_constant_linear_tensor(graph.node(id), &constant_map(&graph)).is_none());
    }

    #[allow(clippy::too_many_arguments)]
    fn quantized_conv_graph(
        input: TensorData,
        input_qp: QParam,
        weight: TensorData,
        weight_qp: QParam,
        bias: Option<TensorData>,
        bias_qp: Option<QParam>,
        output_qp: QParam,
        out_shape: Vec<i64>,
    ) -> (IrGraph, usize) {
        let mut graph = IrGraph::empty();
        let input_id = crate::test_ir::parameter(&mut graph, "input", input);
        let weight_id = crate::test_ir::parameter(&mut graph, "weight", weight);
        let bias_arg = match bias {
            Some(bias_data) => {
                IrArg::Node(crate::test_ir::parameter(&mut graph, "bias", bias_data))
            }
            None => IrArg::None,
        };
        let mut meta = NodeMeta::tensor(out_shape, "torch.float32");
        let mut input_qps = vec![(0, input_qp), (1, weight_qp)];
        if let Some(bias_qp) = bias_qp {
            input_qps.push((2, bias_qp));
        }
        meta.set_input_qparams_from(input_qps);
        meta.set_single_output_qparam(output_qp);
        let id = graph.add_node(
            NodeOp::CallFunction,
            "conv".to_string(),
            "aten.convolution.default".to_string(),
            vec![
                IrArg::Node(input_id),
                IrArg::Node(weight_id),
                bias_arg,
                crate::fx_ir::i64_list_arg(&[1, 1]),
                crate::fx_ir::i64_list_arg(&[0, 0]),
                crate::fx_ir::i64_list_arg(&[1, 1]),
                IrArg::Bool(false),
                crate::fx_ir::i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            meta,
        );
        (graph, id)
    }

    #[test]
    fn conv_folds_zero_input_to_biased_output() {
        // 1x1 conv over a zero (zp) input with a per-channel weight and an
        // int32 bias, the decomposed-linear shape: out = 40 * 0.125 = 5.0,
        // quantized at scale 0.5 -> 10.
        let (graph, id) = quantized_conv_graph(
            i8_tensor(vec![1, 2, 1, 1], &[-128, -128]),
            per_tensor_qparam(0.5, -128),
            i8_tensor(vec![1, 2, 1, 1], &[4, 8]),
            per_channel_qparam(&[0.25], &[0]),
            Some(int32_tensor(vec![1], &[40])),
            Some(QParam::per_tensor(
                0.125,
                0,
                -2147483648,
                2147483647,
                "torch.int32".to_string(),
            )),
            per_tensor_qparam(0.5, 0),
            vec![1, 1, 1, 1],
        );
        let output =
            plan_constant_conv_tensor(graph.node(id), &constant_map(&graph)).expect("conv");
        assert_eq!(output.dtype, DType::Int8);
        assert_eq!(output.shape, vec![1, 1, 1, 1]);
        assert_eq!(output.data, vec![10u8]);
    }

    #[test]
    fn conv_folds_nonzero_input_without_bias() {
        // 1x1 conv sliding a 2-tap kernel: 1.0 * 0.25 + 2.0 * 0.5 = 1.25,
        // quantized at scale 0.25 -> 5.
        let (graph, id) = quantized_conv_graph(
            i8_tensor(vec![1, 1, 1, 2], &[2, 4]),
            per_tensor_qparam(0.5, 0),
            i8_tensor(vec![1, 1, 1, 2], &[1, 2]),
            per_tensor_qparam(0.25, 0),
            None,
            None,
            per_tensor_qparam(0.25, 0),
            vec![1, 1, 1, 1],
        );
        let output =
            plan_constant_conv_tensor(graph.node(id), &constant_map(&graph)).expect("conv");
        assert_eq!(output.dtype, DType::Int8);
        assert_eq!(output.shape, vec![1, 1, 1, 1]);
        assert_eq!(output.data, vec![5u8]);
    }
}
