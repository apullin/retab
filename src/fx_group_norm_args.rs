use crate::fx_ir::{IrArg, IrNode, NodeId, NodeOp};
use crate::fx_target::group_norm_reshape_target;

#[derive(Clone, Debug)]
pub(crate) struct GroupNormArgs {
    pub(crate) input: NodeId,
    pub(crate) weight_arg: IrArg,
    pub(crate) bias_arg: IrArg,
    pub(crate) group: i64,
    pub(crate) eps: f64,
    pub(crate) reshape_target: &'static str,
}

impl GroupNormArgs {
    pub(crate) fn parse(node: &IrNode) -> Option<Self> {
        if node.op != NodeOp::CallFunction {
            return None;
        }
        let reshape_target = group_norm_reshape_target(&node.target)?;
        let args = &node.args;
        let (weight_arg, bias_arg, group, eps) = match args.len() {
            8 => (
                args.get(1).cloned().unwrap_or(IrArg::None),
                args.get(2).cloned().unwrap_or(IrArg::None),
                args.get(6).and_then(IrArg::as_int),
                args.get(7).and_then(IrArg::as_numeric_f64),
            ),
            6 | 5 => (
                args.get(2).cloned().unwrap_or(IrArg::None),
                args.get(3).cloned().unwrap_or(IrArg::None),
                args.get(1).and_then(IrArg::as_int),
                args.get(4).and_then(IrArg::as_numeric_f64),
            ),
            4 => (
                args.get(2).cloned().unwrap_or(IrArg::None),
                args.get(3).cloned().unwrap_or(IrArg::None),
                args.get(1).and_then(IrArg::as_int),
                None,
            ),
            2 => (
                IrArg::None,
                IrArg::None,
                args.get(1).and_then(IrArg::as_int),
                None,
            ),
            _ => (
                node.kwarg_clone_or_none("weight"),
                node.kwarg_clone_or_none("bias"),
                node.kwarg_int("num_groups")
                    .or_else(|| node.kwarg_int("group")),
                node.kwarg_numeric_f64("eps"),
            ),
        };
        let group = group?;
        if group <= 0 {
            return None;
        }
        Some(Self {
            input: node.first_node_arg()?,
            weight_arg,
            bias_arg,
            group,
            eps: eps.unwrap_or(1.0e-5),
            reshape_target,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{IrGraph, NodeMeta};
    use crate::test_ir::{call_with_meta, f32_placeholder, f32_tensor, parameter};

    #[test]
    fn parses_native_group_norm_signature() {
        let mut graph = IrGraph::empty();
        let input = f32_placeholder(&mut graph, "input", [2, 4, 3, 3]);
        let weight = parameter(&mut graph, "weight", f32_tensor(vec![4], &[1.0; 4]));
        let bias = parameter(&mut graph, "bias", f32_tensor(vec![4], &[0.0; 4]));
        let group_norm = call_with_meta(
            &mut graph,
            "group_norm",
            "aten.native_group_norm.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::Node(bias),
                IrArg::Int(2),
                IrArg::Int(4),
                IrArg::Int(9),
                IrArg::Int(2),
                IrArg::Float(0.001),
            ],
            NodeMeta::tensor([2, 4, 3, 3], "torch.float32"),
        );

        let args = GroupNormArgs::parse(graph.node(group_norm)).unwrap();
        assert_eq!(args.input, input);
        assert_eq!(args.group, 2);
        assert_eq!(args.eps, 0.001);
        assert_eq!(args.reshape_target, "aten.view_copy.default");
        assert!(matches!(args.weight_arg, IrArg::Node(id) if id == weight));
        assert!(matches!(args.bias_arg, IrArg::Node(id) if id == bias));
    }
}
