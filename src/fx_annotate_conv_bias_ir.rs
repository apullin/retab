use crate::fx_ir::{IrArg, IrGraph, IrPass, MetaValue, NodeMeta, NodeOp};
use crate::fx_target::{is_aten_convolution_default_target, is_conv_bias_annotation_target};
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

pub struct IrAnnotateConvBiasDtypePass;

fn needs_int16_activation_bias_decomposition(node: &crate::fx_ir::IrNode) -> bool {
    if !is_aten_convolution_default_target(&node.target) {
        return false;
    }
    if matches!(node.args.get(6), Some(IrArg::Bool(true))) {
        return false;
    }
    if !matches!(node.args.get(2), Some(IrArg::Node(_))) {
        return false;
    }
    let Some(input_qp) = node.meta.input_qparam_at(0) else {
        return false;
    };
    let Some(output_qp) = node.meta.output_qparam() else {
        return false;
    };
    matches!(node.meta.rank(), Some(4 | 5))
        && input_qp.dtype() == DType::Int16
        && output_qp.dtype() == DType::Int16
}

fn needs_int48_bias(meta: &NodeMeta) -> bool {
    meta.extra_str("tosa_special_dtype")
        .map(|value| value.contains("INT48"))
        .unwrap_or(false)
        || meta
            .input_qparams()
            .map(|qparams| qparams.values().any(|qp| qp.dtype() == DType::Int16))
            .unwrap_or(false)
}

impl IrPass for IrAnnotateConvBiasDtypePass {
    fn name(&self) -> &str {
        "IrAnnotateConvBiasDtype"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let conv_ids: Vec<_> = graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| {
                (node.alive
                    && node.op == NodeOp::CallFunction
                    && is_conv_bias_annotation_target(&node.target)
                    && !needs_int16_activation_bias_decomposition(node)
                    && needs_int48_bias(&node.meta))
                .then_some(id)
            })
            .collect();

        let mut modified = false;
        for conv_id in conv_ids {
            let Some(IrArg::Node(bias_id)) = graph.nodes[conv_id].args.get(2) else {
                continue;
            };
            let bias = graph.node_mut(*bias_id);
            if bias.meta.extra_str("tosa_special_dtype") == Some("INT48") {
                continue;
            }
            bias.meta
                .set_extra("tosa_special_dtype", MetaValue::Str("INT48".to_string()));
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrAnnotateConvBiasDtype] marked INT16 conv biases as INT48"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{i64_list_arg, NodeMeta, QParam};
    use crate::test_ir::{call_with_meta, placeholder};

    fn qparam(scale: f64, zp: i64, dtype: &str) -> QParam {
        QParam::per_tensor(scale, zp, -32767, 32767, dtype.to_string())
    }

    fn int16_conv_meta(shape: impl Into<Vec<i64>>, dtype: &str) -> NodeMeta {
        let mut meta = NodeMeta::tensor(shape, dtype);
        meta.set_input_qparams_from([
            (0, qparam(0.5, 0, "torch.int16")),
            (1, qparam(0.25, 0, "torch.int8")),
            (2, qparam(0.125, 0, "torch.int32")),
        ]);
        meta.set_single_output_qparam(qparam(0.0625, 0, "torch.int16"));
        meta
    }

    #[test]
    fn skips_bias_annotation_for_int16_bias_decomposition_case() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [1, 3, 8, 8], "torch.int16");
        let weight = placeholder(&mut graph, "weight", [4, 3, 3, 3], "torch.int8");
        let bias = placeholder(&mut graph, "bias", [4], "torch.int32");
        call_with_meta(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![IrArg::Node(input), IrArg::Node(weight), IrArg::Node(bias)],
            int16_conv_meta([1, 4, 8, 8], "torch.int16"),
        );

        assert!(!IrAnnotateConvBiasDtypePass.run(&mut graph));
        assert_eq!(graph.node(bias).meta.extra_str("tosa_special_dtype"), None);
    }

    #[test]
    fn marks_transpose_conv_bias_as_int48_for_int16_path() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [1, 3, 8, 8], "torch.int16");
        let weight = placeholder(&mut graph, "weight", [3, 4, 3, 3], "torch.int8");
        let bias = placeholder(&mut graph, "bias", [4], "torch.int32");
        call_with_meta(
            &mut graph,
            "deconv",
            "aten.convolution.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::Node(bias),
                i64_list_arg(&[2, 2]),
                i64_list_arg(&[1, 1]),
                i64_list_arg(&[1, 1]),
                IrArg::Bool(true),
                i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            int16_conv_meta([1, 4, 16, 16], "torch.int32"),
        );

        assert!(IrAnnotateConvBiasDtypePass.run(&mut graph));
        assert_eq!(
            graph.node(bias).meta.extra_str("tosa_special_dtype"),
            Some("INT48")
        );
    }
}
