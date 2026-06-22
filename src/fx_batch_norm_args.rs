use crate::fx_ir::{IrArg, IrNode, NodeId, NodeOp};
use crate::fx_target::is_batch_norm_target;

#[derive(Clone, Debug)]
pub(crate) struct BatchNormArgs {
    pub(crate) input: NodeId,
    pub(crate) weight_arg: IrArg,
    pub(crate) bias_arg: IrArg,
    pub(crate) running_mean: NodeId,
    pub(crate) running_var: NodeId,
    pub(crate) training: bool,
    pub(crate) eps: f64,
}

impl BatchNormArgs {
    pub(crate) fn parse(node: &IrNode) -> Option<Self> {
        if node.op != NodeOp::CallFunction || !is_batch_norm_target(&node.target) {
            return None;
        }
        Some(Self {
            input: node.arg_node(0)?,
            weight_arg: node.arg_clone_or_none(1),
            bias_arg: node.arg_clone_or_none(2),
            running_mean: node.arg_node(3)?,
            running_var: node.arg_node(4)?,
            training: node.arg_or_kwarg_bool(5, "training").unwrap_or(false),
            eps: node
                .kwarg_numeric_f64("eps")
                .or_else(|| node.args.iter().rev().find_map(IrArg::as_numeric_f64))
                .unwrap_or(1.0e-5),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{IrGraph, NodeMeta};
    use crate::test_ir::{call_with_meta, f32_placeholder, f32_tensor, parameter};

    #[test]
    fn parses_batch_norm_args_and_eps_fallback() {
        let mut graph = IrGraph::empty();
        let input = f32_placeholder(&mut graph, "input", [1, 3, 4, 4]);
        let weight = parameter(&mut graph, "weight", f32_tensor(vec![3], &[1.0; 3]));
        let bias = parameter(&mut graph, "bias", f32_tensor(vec![3], &[0.0; 3]));
        let mean = parameter(&mut graph, "mean", f32_tensor(vec![3], &[0.0; 3]));
        let var = parameter(&mut graph, "var", f32_tensor(vec![3], &[1.0; 3]));
        let bn = call_with_meta(
            &mut graph,
            "bn",
            "aten.batch_norm.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::Node(bias),
                IrArg::Node(mean),
                IrArg::Node(var),
                IrArg::Bool(false),
                IrArg::Float(0.1),
                IrArg::Float(0.001),
            ],
            NodeMeta::tensor([1, 3, 4, 4], "torch.float32"),
        );

        let args = BatchNormArgs::parse(graph.node(bn)).unwrap();
        assert_eq!(args.input, input);
        assert_eq!(args.running_mean, mean);
        assert_eq!(args.running_var, var);
        assert!(!args.training);
        assert_eq!(args.eps, 0.001);
    }
}
