use crate::fx_batch_norm_args::BatchNormArgs;
use crate::fx_conv_args::ConvArgs;
use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp, NodeTensor, TensorSource};
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

fn fuse_conv_bn_tensors(
    conv_weight: &TensorData,
    conv_bias: Option<&TensorData>,
    bn_weight: Option<&TensorData>,
    bn_bias: Option<&TensorData>,
    bn_mean: &TensorData,
    bn_var: &TensorData,
    eps: f64,
    transposed: bool,
    groups: i64,
) -> Option<(TensorData, TensorData)> {
    let weight = tensor_f32_values(conv_weight)?;
    let bias = optional_f32_values(conv_bias)?;
    let gamma = optional_f32_values(bn_weight)?;
    let beta = optional_f32_values(bn_bias)?;
    let mean = tensor_f32_values(bn_mean)?;
    let var = tensor_f32_values(bn_var)?;

    let input_channels = usize::try_from(*conv_weight.shape.first()?).ok()?;
    let groups = usize::try_from(groups).ok()?;
    if groups == 0 || input_channels == 0 || input_channels % groups != 0 {
        return None;
    }
    let out_channels = if transposed {
        usize::try_from(*conv_weight.shape.get(1)?)
            .ok()?
            .checked_mul(groups)?
    } else {
        input_channels
    };
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
    if transposed {
        let output_channels_per_group = out_channels / groups;
        let input_channels_per_group = input_channels / groups;
        let kernel_elems = fused_weight.len() / (input_channels * output_channels_per_group);
        if kernel_elems == 0
            || fused_weight.len() % (input_channels * output_channels_per_group) != 0
        {
            return None;
        }
        // ConvTranspose weights are [Cin, Cout/groups, ...].
        for (input_channel, channel_weights) in fused_weight
            .chunks_exact_mut(output_channels_per_group * kernel_elems)
            .enumerate()
        {
            let group = input_channel / input_channels_per_group;
            for (output_channel, kernel) in
                channel_weights.chunks_exact_mut(kernel_elems).enumerate()
            {
                let scale = scales[group * output_channels_per_group + output_channel];
                for value in kernel {
                    *value *= scale;
                }
            }
        }
    } else {
        for channel in 0..out_channels {
            let scale = scales[channel];
            let start = channel * per_output_weight_elems;
            let end = start + per_output_weight_elems;
            for value in &mut fused_weight[start..end] {
                *value *= scale;
            }
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
    if !graph.node(bn_id).alive {
        return false;
    }
    let Some(bn_args) = BatchNormArgs::parse(graph.node(bn_id)) else {
        return false;
    };
    if bn_args.training {
        return false;
    }

    let conv_id = bn_args.input;
    if !graph.node(conv_id).is_convolution() || graph.single_live_user(conv_id) != Some(bn_id) {
        return false;
    }

    let Some(conv_args) = ConvArgs::parse(graph.node(conv_id)) else {
        return false;
    };
    let Some(conv_bias_id) = conv_args.well_formed_bias() else {
        return false;
    };

    let weight_id = conv_args.weight;
    let Some(conv_weight) = graph
        .node(weight_id)
        .tensor
        .as_ref()
        .map(|tensor| tensor.data.clone())
    else {
        return false;
    };
    let conv_bias =
        conv_bias_id.and_then(|id| graph.node(id).tensor.as_ref().map(|t| t.data.clone()));
    let Some(bn_weight) = optional_node_tensor(graph, Some(&bn_args.weight_arg)) else {
        return false;
    };
    let Some(bn_bias) = optional_node_tensor(graph, Some(&bn_args.bias_arg)) else {
        return false;
    };
    let Some(bn_mean) = graph
        .node(bn_args.running_mean)
        .tensor
        .as_ref()
        .map(|tensor| tensor.data.clone())
    else {
        return false;
    };
    let Some(bn_var) = graph
        .node(bn_args.running_var)
        .tensor
        .as_ref()
        .map(|tensor| tensor.data.clone())
    else {
        return false;
    };

    let Some((fused_weight, fused_bias)) = fuse_conv_bn_tensors(
        &conv_weight,
        conv_bias.as_ref(),
        bn_weight.as_ref(),
        bn_bias.as_ref(),
        &bn_mean,
        &bn_var,
        bn_args.eps,
        conv_args.transposed,
        conv_args.groups,
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

    graph.set_arg(
        conv_id,
        ConvArgs::WEIGHT_INDEX,
        IrArg::Node(fused_weight_id),
    );
    graph.set_arg(conv_id, ConvArgs::BIAS_INDEX, IrArg::Node(fused_bias_id));
    replace_primary_result_users(graph, bn_id, conv_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeOp;
    use crate::test_ir::{f32_call, f32_meta, f32_placeholder, f32_tensor, output, parameter};

    #[test]
    fn regression_transposed_conv_bn_scales_output_channels_within_each_group() {
        for (groups, input_channels, scales, expected_weights) in [
            (1, 2, vec![2.0, 3.0], vec![2.0, 6.0, 6.0, 12.0]),
            (
                2,
                4,
                vec![2.0, 3.0, 5.0, 7.0],
                vec![2.0, 6.0, 6.0, 12.0, 25.0, 42.0, 35.0, 56.0],
            ),
            (1, 3, vec![2.0, 3.0], vec![2.0, 6.0, 6.0, 12.0, 10.0, 18.0]),
        ] {
            let mut graph = IrGraph::empty();
            let output_channels = scales.len() as i64;
            let x = f32_placeholder(&mut graph, "x", [1, input_channels, 1, 1]);
            let weights: Vec<f32> = (1..=input_channels * 2).map(|v| v as f32).collect();
            let weight = parameter(
                &mut graph,
                "weight",
                f32_tensor(vec![input_channels, 2, 1, 1], &weights),
            );
            let bias = parameter(
                &mut graph,
                "bias",
                f32_tensor(vec![output_channels], &vec![2.0; scales.len()]),
            );
            let mean = parameter(
                &mut graph,
                "mean",
                f32_tensor(vec![output_channels], &vec![1.0; scales.len()]),
            );
            let var = parameter(
                &mut graph,
                "var",
                f32_tensor(vec![output_channels], &vec![1.0; scales.len()]),
            );
            let gamma = parameter(
                &mut graph,
                "gamma",
                f32_tensor(vec![output_channels], &scales),
            );
            let conv = f32_call(
                &mut graph,
                "conv_transpose",
                "aten.convolution.default",
                vec![
                    IrArg::Node(x),
                    IrArg::Node(weight),
                    IrArg::Node(bias),
                    crate::fx_ir::i64_list_arg(&[1, 1]),
                    crate::fx_ir::i64_list_arg(&[0, 0]),
                    crate::fx_ir::i64_list_arg(&[1, 1]),
                    IrArg::Bool(true),
                    crate::fx_ir::i64_list_arg(&[0, 0]),
                    IrArg::Int(groups),
                ],
                [1, output_channels, 1, 1],
            );
            let bn = f32_call(
                &mut graph,
                "bn",
                "aten.batch_norm.default",
                vec![
                    IrArg::Node(conv),
                    IrArg::Node(gamma),
                    IrArg::None,
                    IrArg::Node(mean),
                    IrArg::Node(var),
                    IrArg::Bool(false),
                    IrArg::Float(0.1),
                    IrArg::Float(0.0),
                    IrArg::Bool(true),
                ],
                [1, output_channels, 1, 1],
            );
            let out = output(&mut graph, bn);

            assert!(IrFuseBatchNorm2dPass.run(&mut graph));
            assert_eq!(graph.node(out).arg_node(0), Some(conv));
            let fused_weight = graph.node(conv).arg_node(1).unwrap();
            let fused_bias = graph.node(conv).arg_node(2).unwrap();
            let weight_data = &graph.node(fused_weight).tensor.as_ref().unwrap().data;
            let bias_data = &graph.node(fused_bias).tensor.as_ref().unwrap().data;
            assert_eq!(tensor_f32_values(weight_data), Some(expected_weights));
            assert_eq!(tensor_f32_values(bias_data), Some(scales));
            assert_eq!(bias_data.shape, vec![output_channels]);
        }
    }

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

    #[test]
    fn fuses_existing_conv_bias_without_bn_affine_params() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 2, 4, 4]);
        let weight = parameter(
            &mut graph,
            "conv_weight",
            f32_tensor(vec![2, 1, 1, 1], &[8.0, 12.0]),
        );
        let bias = parameter(&mut graph, "conv_bias", f32_tensor(vec![2], &[2.0, -3.0]));
        let mean = parameter(&mut graph, "bn_mean", f32_tensor(vec![2], &[4.0, 1.0]));
        let var = parameter(&mut graph, "bn_var", f32_tensor(vec![2], &[4.0, 9.0]));
        let conv = f32_call(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(x),
                IrArg::Node(weight),
                IrArg::Node(bias),
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
                IrArg::None,
                IrArg::None,
                IrArg::Node(mean),
                IrArg::Node(var),
                IrArg::Bool(false),
                IrArg::Float(1.0),
                IrArg::Float(0.0),
                IrArg::Float(0.0),
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
        let IrArg::Node(fused_weight_id) = graph.node(conv).args[1] else {
            panic!("missing fused weight")
        };
        let IrArg::Node(fused_bias_id) = graph.node(conv).args[2] else {
            panic!("missing fused bias")
        };
        let weight_values =
            tensor_f32_values(&graph.node(fused_weight_id).tensor.as_ref().unwrap().data).unwrap();
        let bias_values =
            tensor_f32_values(&graph.node(fused_bias_id).tensor.as_ref().unwrap().data).unwrap();

        assert!((weight_values[0] - 4.0).abs() < 1e-5);
        assert!((weight_values[1] - 4.0).abs() < 1e-5);
        assert!((bias_values[0] - -1.0).abs() < 1e-5);
        assert!((bias_values[1] - -1.3333334).abs() < 1e-5);
    }

    #[test]
    fn skips_non_float32_conv_weights() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 1, 4, 4]);
        let weight = parameter(
            &mut graph,
            "conv_weight",
            TensorData {
                data: vec![0; 2],
                shape: vec![1, 1, 1, 1],
                dtype: DType::Float16,
            },
        );
        let mean = parameter(&mut graph, "bn_mean", f32_tensor(vec![1], &[0.0]));
        let var = parameter(&mut graph, "bn_var", f32_tensor(vec![1], &[1.0]));
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
            [1, 1, 4, 4],
        );
        let bn = graph.add_node(
            NodeOp::CallFunction,
            "bn".to_string(),
            "aten._native_batch_norm_legit_no_training.default".to_string(),
            vec![
                IrArg::Node(conv),
                IrArg::None,
                IrArg::None,
                IrArg::Node(mean),
                IrArg::Node(var),
                IrArg::Bool(false),
            ],
            f32_meta([1, 1, 4, 4]),
        );
        output(&mut graph, bn);

        assert!(!IrFuseBatchNorm2dPass.run(&mut graph));
        assert!(graph.node(bn).alive);
    }
}
