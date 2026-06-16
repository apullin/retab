use crate::fx_ir::{
    IrArg, IrGraph, IrNode, IrPass, NodeId, NodeMeta, NodeOp, NodeTensor, TensorSource,
};
use crate::graph::ir::DType;
use crate::store::TensorData;

pub struct IrFuseBatchNorm2dPass;

impl IrPass for IrFuseBatchNorm2dPass {
    fn name(&self) -> &str {
        "IrFuseBatchNorm2d"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            modified |= fuse_batch_norm_node(graph, id);
        }
        modified
    }
}

fn is_batch_norm_target(target: &str) -> bool {
    target.contains("aten.batch_norm")
        || target.contains("aten.native_batch_norm")
        || target.contains("aten._native_batch_norm_legit_no_training")
        || target.contains("aten._native_batch_norm_legit.no_stats")
}

fn batch_norm_training(node: &IrNode) -> bool {
    node.arg_or_kwarg_bool(5, "training").unwrap_or(false)
}

fn batch_norm_eps(node: &IrNode) -> f64 {
    node.kwarg_numeric_f64("eps")
        .or_else(|| node.args.iter().rev().find_map(IrArg::as_numeric_f64))
        .unwrap_or(1.0e-5)
}

fn tensor_f32_values(tensor: &TensorData) -> Option<Vec<f32>> {
    (tensor.dtype == DType::Float32).then_some(())?;
    tensor
        .vec_f64()
        .map(|values| values.into_iter().map(|value| value as f32).collect())
}

fn optional_f32_values(tensor: Option<&TensorData>) -> Option<Vec<f32>> {
    match tensor {
        Some(tensor) => tensor_f32_values(tensor),
        None => Some(Vec::new()),
    }
}

fn encode_f32_tensor(shape: Vec<i64>, values: Vec<f32>) -> TensorData {
    TensorData {
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        shape,
        dtype: DType::Float32,
    }
}

fn optional_node_tensor(graph: &IrGraph, arg: Option<&IrArg>) -> Option<Option<TensorData>> {
    match arg {
        Some(IrArg::Node(id)) => graph
            .node_opt(*id)?
            .tensor
            .as_ref()
            .map(|t| Some(t.data.clone())),
        Some(IrArg::None) | None => Some(None),
        _ => None,
    }
}

fn tensor_arg(graph: &IrGraph, node_id: NodeId, index: usize) -> Option<TensorData> {
    let id = graph.node(node_id).arg_node(index)?;
    Some(graph.node(id).tensor.as_ref()?.data.clone())
}

fn fuse_conv_bn_tensors(
    conv_weight: &TensorData,
    conv_bias: Option<&TensorData>,
    bn_weight: Option<&TensorData>,
    bn_bias: Option<&TensorData>,
    bn_mean: &TensorData,
    bn_var: &TensorData,
    eps: f64,
) -> Option<(TensorData, TensorData)> {
    let weight = tensor_f32_values(conv_weight)?;
    let bias = optional_f32_values(conv_bias)?;
    let gamma = optional_f32_values(bn_weight)?;
    let beta = optional_f32_values(bn_bias)?;
    let mean = tensor_f32_values(bn_mean)?;
    let var = tensor_f32_values(bn_var)?;

    let out_channels = *conv_weight.shape.first()? as usize;
    if out_channels == 0
        || mean.len() != out_channels
        || var.len() != out_channels
        || (!bias.is_empty() && bias.len() != out_channels)
        || (!gamma.is_empty() && gamma.len() != out_channels)
        || (!beta.is_empty() && beta.len() != out_channels)
        || weight.len() % out_channels != 0
    {
        return None;
    }

    let per_output_weight_elems = weight.len() / out_channels;
    let mut scales = Vec::with_capacity(out_channels);
    for channel in 0..out_channels {
        let gamma = gamma.get(channel).copied().unwrap_or(1.0);
        scales.push(gamma / (var[channel] + eps as f32).sqrt());
    }

    let mut fused_weight = weight;
    for channel in 0..out_channels {
        let scale = scales[channel];
        let start = channel * per_output_weight_elems;
        let end = start + per_output_weight_elems;
        for value in &mut fused_weight[start..end] {
            *value *= scale;
        }
    }

    let mut fused_bias = Vec::with_capacity(out_channels);
    for channel in 0..out_channels {
        let conv_bias = bias.get(channel).copied().unwrap_or(0.0);
        let beta = beta.get(channel).copied().unwrap_or(0.0);
        fused_bias.push((conv_bias - mean[channel]) * scales[channel] + beta);
    }

    Some((
        encode_f32_tensor(conv_weight.shape.clone(), fused_weight),
        encode_f32_tensor(vec![out_channels as i64], fused_bias),
    ))
}

fn add_tensor_placeholder(
    graph: &mut IrGraph,
    name: String,
    tensor: TensorData,
    meta: NodeMeta,
) -> NodeId {
    let id = graph.add_node(NodeOp::Placeholder, name.clone(), name, Vec::new(), meta);
    graph.node_mut(id).tensor = Some(NodeTensor {
        source: TensorSource::GraphModuleAttr,
        data: tensor,
    });
    id
}

fn has_direct_non_getitem_user(graph: &IrGraph, id: NodeId) -> bool {
    graph
        .live_user_ids(id)
        .into_iter()
        .any(|user_id| user_id != id && !graph.node(user_id).is_getitem())
}

fn replace_primary_result_users(graph: &mut IrGraph, id: NodeId, result: NodeId) -> bool {
    let getitem_users = graph.live_getitem_users(id);
    if getitem_users.is_empty() {
        graph.replace_and_erase(id, result);
        return true;
    }
    if has_direct_non_getitem_user(graph, id) {
        return false;
    }
    for (getitem_id, idx) in getitem_users {
        if idx != 0 {
            return false;
        }
        graph.replace_and_erase(getitem_id, result);
    }
    graph.erase(id);
    true
}

fn fuse_batch_norm_node(graph: &mut IrGraph, bn_id: NodeId) -> bool {
    if !graph.node(bn_id).alive
        || !is_batch_norm_target(&graph.node(bn_id).target)
        || batch_norm_training(graph.node(bn_id))
    {
        return false;
    }

    let Some(conv_id) = graph.node(bn_id).arg_node(0) else {
        return false;
    };
    if !graph.node(conv_id).is_convolution() || graph.single_live_user(conv_id) != Some(bn_id) {
        return false;
    }

    let Some(weight_id) = graph.node(conv_id).arg_node(1) else {
        return false;
    };
    let conv_bias_id = match graph.node(conv_id).args.get(2) {
        Some(IrArg::Node(id)) => Some(*id),
        Some(IrArg::None) => None,
        _ => return false,
    };

    let Some(conv_weight) = tensor_arg(graph, conv_id, 1) else {
        return false;
    };
    let conv_bias =
        conv_bias_id.and_then(|id| graph.node(id).tensor.as_ref().map(|t| t.data.clone()));
    let Some(bn_weight) = optional_node_tensor(graph, graph.node(bn_id).args.get(1)) else {
        return false;
    };
    let Some(bn_bias) = optional_node_tensor(graph, graph.node(bn_id).args.get(2)) else {
        return false;
    };
    let Some(bn_mean) = tensor_arg(graph, bn_id, 3) else {
        return false;
    };
    let Some(bn_var) = tensor_arg(graph, bn_id, 4) else {
        return false;
    };
    let eps = batch_norm_eps(graph.node(bn_id));

    let Some((fused_weight, fused_bias)) = fuse_conv_bn_tensors(
        &conv_weight,
        conv_bias.as_ref(),
        bn_weight.as_ref(),
        bn_bias.as_ref(),
        &bn_mean,
        &bn_var,
        eps,
    ) else {
        return false;
    };

    let weight_name = graph.node(weight_id).name.clone();
    let bn_name = graph.node(bn_id).name.clone();
    let fused_weight_name = format!("{weight_name}_{bn_name}_fused_bn");
    let fused_bias_name = format!("{weight_name}_{bn_name}_bias_fused_bn");
    let fused_weight_meta = graph.node(weight_id).meta.clone();
    let fused_bias_meta = conv_bias_id
        .map(|id| graph.node(id).meta.clone())
        .unwrap_or_else(|| NodeMeta::tensor(fused_bias.shape.clone(), "torch.float32"));

    let fused_weight_id =
        add_tensor_placeholder(graph, fused_weight_name, fused_weight, fused_weight_meta);
    let fused_bias_id = add_tensor_placeholder(graph, fused_bias_name, fused_bias, fused_bias_meta);

    graph.set_arg(conv_id, 1, IrArg::Node(fused_weight_id));
    graph.set_arg(conv_id, 2, IrArg::Node(fused_bias_id));
    replace_primary_result_users(graph, bn_id, conv_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeOp;
    use crate::test_ir::{f32_call, f32_meta, f32_placeholder, f32_tensor, output, parameter};

    #[test]
    fn fuses_conv_batch_norm_into_weight_and_bias_placeholders() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 2, 4, 4]);
        let weight = parameter(
            &mut graph,
            "conv_weight",
            f32_tensor(vec![2, 1, 1, 1], &[2.0, 4.0]),
        );
        let mean = parameter(&mut graph, "bn_mean", f32_tensor(vec![2], &[1.0, -1.0]));
        let var = parameter(&mut graph, "bn_var", f32_tensor(vec![2], &[3.0, 15.0]));
        let gamma = parameter(&mut graph, "bn_gamma", f32_tensor(vec![2], &[3.0, 2.0]));
        let beta = parameter(&mut graph, "bn_beta", f32_tensor(vec![2], &[5.0, -7.0]));
        let conv = f32_call(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(x),
                IrArg::Node(weight),
                IrArg::None,
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(1)]),
                IrArg::List(vec![IrArg::Int(0), IrArg::Int(0)]),
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(1)]),
                IrArg::Bool(false),
                IrArg::List(vec![IrArg::Int(0), IrArg::Int(0)]),
                IrArg::Int(1),
            ],
            [1, 2, 4, 4],
        );
        let bn = graph.add_node(
            NodeOp::CallFunction,
            "bn".to_string(),
            "aten._native_batch_norm_legit_no_training.default".to_string(),
            vec![
                IrArg::Node(conv),
                IrArg::Node(gamma),
                IrArg::Node(beta),
                IrArg::Node(mean),
                IrArg::Node(var),
                IrArg::Bool(false),
                IrArg::Float(1.0),
                IrArg::Float(0.0),
                IrArg::Float(1.0),
            ],
            f32_meta([1, 2, 4, 4]),
        );
        let getitem = graph.add_node(
            NodeOp::CallFunction,
            "bn_value".to_string(),
            "operator.getitem".to_string(),
            vec![IrArg::Node(bn), IrArg::Int(0)],
            f32_meta([1, 2, 4, 4]),
        );
        output(&mut graph, getitem);

        assert!(IrFuseBatchNorm2dPass.run(&mut graph));
        assert!(!graph.node(bn).alive);
        assert!(!graph.node(getitem).alive);
        assert!(matches!(graph.node(conv).args[1], IrArg::Node(_)));
        assert!(matches!(graph.node(conv).args[2], IrArg::Node(_)));

        let IrArg::Node(fused_weight_id) = graph.node(conv).args[1] else {
            panic!("missing fused weight")
        };
        let IrArg::Node(fused_bias_id) = graph.node(conv).args[2] else {
            panic!("missing fused bias")
        };
        let fused_weight = graph.node(fused_weight_id).tensor.as_ref().unwrap();
        let fused_bias = graph.node(fused_bias_id).tensor.as_ref().unwrap();
        let weight_values = tensor_f32_values(&fused_weight.data).unwrap();
        let bias_values = tensor_f32_values(&fused_bias.data).unwrap();

        assert!((weight_values[0] - 3.0).abs() < 1e-6);
        assert!((weight_values[1] - 2.0).abs() < 1e-6);
        assert!((bias_values[0] - 3.5).abs() < 1e-6);
        assert!((bias_values[1] - -6.5).abs() < 1e-6);
    }
}
