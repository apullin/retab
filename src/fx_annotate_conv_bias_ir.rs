use crate::fx_ir::{IrArg, IrGraph, IrPass, MetaValue, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

pub struct IrAnnotateConvBiasDtypePass;

fn is_conv_target(target: &str) -> bool {
    target.contains("convolution.default")
        || target.contains("conv2d.default")
        || target.contains("conv3d.default")
        || target.contains("backend.tosa.CONV")
        || target.contains("backend.tosa.DEPTHWISE_CONV")
}

fn needs_int48_bias(meta: &NodeMeta) -> bool {
    meta.extra_str("tosa_special_dtype")
        .map(|value| value.contains("INT48"))
        .unwrap_or(false)
        || meta
            .input_qparams
            .as_ref()
            .map(|qparams| qparams.values().any(|qp| qp.dtype_str.contains("int16")))
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
                    && is_conv_target(&node.target)
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
