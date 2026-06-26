//! Pure Rust IrPass version of FxDecomposeGroupedConvPass.
//!
//! Splits grouped, non-depthwise convolutions into:
//!   slice_copy(input) + slice_copy(weight) + slice_copy(bias)
//!   convolution/conv2d(groups=1) per group
//!   cat(outputs, dim=1)

use crate::fx_conv_args::{ConvArgKind, ConvArgs};
use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_target::is_boundary_rescale_target;
use crate::runner_logging::log_line;

fn slice_meta(graph: &IrGraph, id: NodeId, dim: usize, start: i64, stop: i64) -> Option<NodeMeta> {
    let mut shape = graph.node_shape_vec(id)?;
    if dim >= shape.len() {
        return None;
    }
    shape[dim] = (stop - start).max(0);
    graph.node_dtype(id)?;
    Some(
        graph.nodes[id]
            .meta
            .tensor_stamp_same_dtype_or_float32()
            .preserve_qparams(shape),
    )
}

fn conv_slice_output_meta(graph: &IrGraph, id: NodeId, output_slice_size: i64) -> NodeMeta {
    if let Some(shape) = graph.node_shape(id) {
        let mut shape = shape.to_vec();
        if shape.len() >= 2 {
            shape[1] = output_slice_size;
        }
        return graph.nodes[id]
            .meta
            .cloned_with_same_dtype_or_float32(shape);
    }
    graph.nodes[id].meta.clone()
}

fn configure_conv_qparams(
    graph: &IrGraph,
    original_id: NodeId,
    group_index: i64,
    output_slice_size: i64,
    transposed: bool,
    conv_meta: &mut NodeMeta,
) -> bool {
    conv_meta.copy_output_qparams_from(&graph.nodes[original_id].meta);

    let Some(mut input_qparams) = graph.nodes[original_id].meta.cloned_input_qparams() else {
        return true;
    };

    let start = (group_index * output_slice_size).max(0) as usize;
    let end = ((group_index + 1) * output_slice_size).max(0) as usize;

    if let Some(weight_qparam) = input_qparams.get_mut(&1) {
        if weight_qparam.is_per_channel() {
            if transposed {
                return false;
            }
            let Some(sliced) = weight_qparam.slice_axis0_channels(start, end) else {
                return false;
            };
            *weight_qparam = sliced;
        }
    }
    if let Some(bias_qparam) = input_qparams.get_mut(&2) {
        if bias_qparam.is_per_channel() {
            let Some(sliced) = bias_qparam.slice_axis0_channels(start, end) else {
                return false;
            };
            *bias_qparam = sliced;
        }
    }

    conv_meta.set_input_qparams(input_qparams);
    true
}

fn qparams_support_group_slicing(
    graph: &IrGraph,
    original_id: NodeId,
    groups: i64,
    output_slice_size: i64,
    transposed: bool,
) -> bool {
    let Some(input_qparams) = graph.nodes[original_id].meta.input_qparams() else {
        return true;
    };
    let end = (groups * output_slice_size).max(0) as usize;
    for (index, qparam) in input_qparams {
        if !qparam.is_per_channel() {
            continue;
        }
        if *index == 1 && transposed {
            return false;
        }
        if (*index == 1 || *index == 2) && !qparam.supports_axis0_channel_slice(end) {
            return false;
        }
    }
    true
}

fn cat_meta_from_outputs(
    graph: &IrGraph,
    original_id: NodeId,
    output_slices: &[NodeId],
) -> NodeMeta {
    let mut meta = graph.nodes[original_id].meta.clone();
    meta.set_empty_qparam_maps();
    for &slice_id in output_slices {
        if let Some(qparam) = graph.nodes[slice_id].meta.output_qparam().cloned() {
            meta.set_single_output_qparam(qparam);
            break;
        }
    }
    meta
}

fn bypass_redundant_output_boundaries(graph: &mut IrGraph, cat_id: NodeId) {
    let Some(cat_qparam) = graph.nodes[cat_id].meta.output_qparam().cloned() else {
        return;
    };
    let users = graph.live_user_ids(cat_id);
    for user_id in users {
        if !graph.nodes[user_id].alive
            || graph.nodes[user_id].op != NodeOp::CallFunction
            || !is_boundary_rescale_target(&graph.nodes[user_id].target)
        {
            continue;
        }
        if !matches!(graph.nodes[user_id].args.first(), Some(IrArg::Node(id)) if *id == cat_id) {
            continue;
        }
        let Some(user_qparam) = graph.nodes[user_id].meta.output_qparam().cloned() else {
            continue;
        };
        if !cat_qparam.same_per_tensor_exact_affine_dtype(&user_qparam) {
            continue;
        }
        graph.replace_and_erase(user_id, cat_id);
    }
}

fn parse_grouped_conv(graph: &IrGraph, id: NodeId) -> Option<ConvArgs> {
    let node = &graph.nodes[id];
    if !node.alive || node.op != NodeOp::CallFunction {
        return None;
    }
    let spec = ConvArgs::parse(node)?;
    if !matches!(
        spec.kind,
        ConvArgKind::Conv2dDefault | ConvArgKind::ConvolutionDefault
    ) {
        return None;
    }
    if spec.groups <= 1 {
        return None;
    }
    let in_channels = graph.node_shape(spec.input)?.get(1).copied().unwrap_or(0);
    if !spec.transposed && in_channels == spec.groups {
        return None;
    }

    Some(spec)
}

fn decompose_grouped_conv(graph: &mut IrGraph, id: NodeId, spec: ConvArgs) -> bool {
    let node_name = graph.nodes[id].name.clone();
    let args = graph.nodes[id].args.clone();
    let weight_shape = match graph.node_shape(spec.weight) {
        Some(shape) if shape.len() >= 2 => shape.to_vec(),
        _ => return false,
    };

    let (input_slice_size, output_slice_size) = if spec.transposed {
        (weight_shape[0] / spec.groups, weight_shape[1])
    } else {
        (weight_shape[1], weight_shape[0] / spec.groups)
    };
    if !qparams_support_group_slicing(graph, id, spec.groups, output_slice_size, spec.transposed) {
        return false;
    }

    let conv_target = if spec.is_raw_conv2d() {
        "aten.conv2d.default"
    } else {
        "aten.convolution.default"
    };
    let mut output_slices = Vec::new();
    let mut cursor = graph.rewrite_cursor(id);

    for group in 0..spec.groups {
        let input_start = group * input_slice_size;
        let input_stop = (group + 1) * input_slice_size;
        let Some(input_meta) = slice_meta(cursor.graph(), spec.input, 1, input_start, input_stop)
        else {
            return false;
        };
        let input_slice = cursor.slice_copy(
            format!("{}_input_slice_{}", node_name, group),
            spec.input,
            1,
            input_start,
            input_stop,
            None,
            input_meta,
        );

        let (weight_start, weight_stop) = if spec.transposed {
            (group * input_slice_size, (group + 1) * input_slice_size)
        } else {
            (group * output_slice_size, (group + 1) * output_slice_size)
        };
        let Some(weight_meta) =
            slice_meta(cursor.graph(), spec.weight, 0, weight_start, weight_stop)
        else {
            return false;
        };
        let weight_slice = cursor.slice_copy(
            format!("{}_weight_slice_{}", node_name, group),
            spec.weight,
            0,
            weight_start,
            weight_stop,
            None,
            weight_meta,
        );

        let bias_arg = match &spec.bias_arg {
            IrArg::Node(bias_id) => {
                let bias_start = group * output_slice_size;
                let bias_stop = (group + 1) * output_slice_size;
                let Some(bias_meta) =
                    slice_meta(cursor.graph(), *bias_id, 0, bias_start, bias_stop)
                else {
                    return false;
                };
                let bias_slice = cursor.slice_copy(
                    format!("{}_bias_slice_{}", node_name, group),
                    *bias_id,
                    0,
                    bias_start,
                    bias_stop,
                    None,
                    bias_meta,
                );
                IrArg::Node(bias_slice)
            }
            _ => IrArg::None,
        };

        let Some(conv_args) =
            spec.args_with_replaced_io_bias_groups(&args, input_slice, weight_slice, bias_arg, 1)
        else {
            return false;
        };

        let mut conv_meta = conv_slice_output_meta(cursor.graph(), id, output_slice_size);
        if !configure_conv_qparams(
            cursor.graph(),
            id,
            group,
            output_slice_size,
            spec.transposed,
            &mut conv_meta,
        ) {
            return false;
        }
        let conv = cursor.call_function(
            format!("{}_group_conv_{}", node_name, group),
            conv_target,
            conv_args,
            conv_meta,
        );
        output_slices.push(conv);
    }

    let cat_meta = cat_meta_from_outputs(cursor.graph(), id, &output_slices);
    let cat = cursor.call_function(
        format!("{}_cat", node_name),
        "aten.cat.default",
        vec![
            IrArg::List(
                output_slices
                    .iter()
                    .map(|&node_id| IrArg::Node(node_id))
                    .collect(),
            ),
            IrArg::Int(1),
        ],
        cat_meta,
    );
    cursor.replace_anchor_with(cat);
    bypass_redundant_output_boundaries(graph, cat);
    true
}

pub struct IrDecomposeGroupedConvPass;

impl IrPass for IrDecomposeGroupedConvPass {
    fn name(&self) -> &str {
        "IrDecomposeGroupedConv"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let Some(spec) = parse_grouped_conv(graph, id) else {
                return false;
            };
            decompose_grouped_conv(graph, id, spec)
        });
        if modified {
            log_line(format_args!(
                "[IrDecomposeGroupedConv] decomposed grouped conv ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{i64_list_arg as i64_list, QParam};
    use crate::test_ir::{call_with_meta, output, placeholder};

    fn per_channel_qparam(scales: &[f64]) -> QParam {
        QParam {
            scale: 1.0,
            zp: 0,
            qmin: -128,
            qmax: 127,
            dtype_str: "torch.int8".to_string(),
            per_channel_scales: scales.to_vec(),
            per_channel_zps: vec![0; scales.len()],
            axis: 0,
        }
    }

    fn per_tensor_qparam(scale: f64) -> QParam {
        QParam::per_tensor(scale, 0, -128, 127, "torch.int8".to_string())
    }

    fn conv_call(
        graph: &mut IrGraph,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        groups: i64,
        meta: NodeMeta,
    ) -> NodeId {
        call_with_meta(
            graph,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                bias.map(IrArg::Node).unwrap_or(IrArg::None),
                i64_list(&[1, 1]),
                i64_list(&[0, 0]),
                i64_list(&[1, 1]),
                IrArg::Bool(false),
                i64_list(&[0, 0]),
                IrArg::Int(groups),
            ],
            meta,
        )
    }

    #[test]
    fn decomposes_grouped_conv_and_slices_per_channel_qparams() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [1, 6, 8, 8], "torch.int8");
        let weight = placeholder(&mut graph, "weight", [12, 2, 3, 3], "torch.int8");
        let bias = placeholder(&mut graph, "bias", [12], "torch.int8");
        graph.nodes[input].meta.set_single_output_qparam(per_tensor_qparam(0.25));
        let mut conv_meta = NodeMeta::tensor([1, 12, 6, 6], "torch.int8");
        conv_meta.set_input_qparams_from([
            (0, per_tensor_qparam(0.25)),
            (
                1,
                per_channel_qparam(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
            ),
            (
                2,
                per_channel_qparam(&[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]),
            ),
        ]);
        conv_meta.set_single_output_qparam(per_tensor_qparam(0.5));

        let conv = conv_call(&mut graph, input, weight, Some(bias), 3, conv_meta);
        output(&mut graph, conv);

        assert!(IrDecomposeGroupedConvPass.run(&mut graph));
        assert!(!graph.nodes[conv].alive);
        let cat = graph.by_name("conv_cat").unwrap();
        assert_eq!(cat.target, "aten.cat.default");
        assert_eq!(cat.meta.val_shape(), Some(&[1, 12, 6, 6][..]));

        let first_group = graph.by_name("conv_group_conv_0").unwrap();
        assert!(matches!(first_group.args.last(), Some(IrArg::Int(1))));
        let qparams = first_group.meta.input_qparams().unwrap();
        assert_eq!(qparams[&1].per_channel_scales, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(qparams[&2].per_channel_scales, vec![1.1, 1.2, 1.3, 1.4]);

        let input_slice = graph.by_name("conv_input_slice_0").unwrap();
        assert_eq!(input_slice.meta.output_qparam().unwrap().dtype_str, "torch.int8");
        assert_eq!(input_slice.meta.val_shape(), Some(&[1, 2, 8, 8][..]));
    }

    #[test]
    fn skips_depthwise_conv() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [1, 4, 8, 8], "torch.int8");
        let weight = placeholder(&mut graph, "weight", [4, 1, 3, 3], "torch.int8");
        conv_call(
            &mut graph,
            input,
            weight,
            None,
            4,
            NodeMeta::tensor([1, 4, 6, 6], "torch.int8"),
        );

        assert!(!IrDecomposeGroupedConvPass.run(&mut graph));
    }
}
