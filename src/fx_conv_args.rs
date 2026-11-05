use crate::fx_ir::{IrArg, IrNode, NodeId, NodeOp};
use crate::fx_target::{
    is_aten_convolution_default_target, is_conv1d_default_target, is_conv2d_default_target,
    is_conv3d_default_target,
};

#[allow(clippy::enum_variant_names)] // named after the aten overloads
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConvArgKind {
    Conv1dDefault,
    Conv2dDefault,
    Conv3dDefault,
    ConvolutionDefault,
}

#[derive(Clone, Debug)]
pub(crate) struct ConvArgs {
    pub(crate) kind: ConvArgKind,
    pub(crate) input: NodeId,
    pub(crate) weight: NodeId,
    pub(crate) bias: Option<NodeId>,
    pub(crate) bias_arg: IrArg,
    pub(crate) bias_well_formed: bool,
    pub(crate) stride: Vec<i64>,
    pub(crate) padding: Vec<i64>,
    pub(crate) dilation: Vec<i64>,
    pub(crate) transposed: bool,
    pub(crate) output_padding: Vec<i64>,
    pub(crate) groups: i64,
}

impl ConvArgs {
    pub(crate) const INPUT_INDEX: usize = 0;
    pub(crate) const WEIGHT_INDEX: usize = 1;
    pub(crate) const BIAS_INDEX: usize = 2;

    pub(crate) fn parse(node: &IrNode) -> Option<Self> {
        if node.op != NodeOp::CallFunction {
            return None;
        }
        if is_conv1d_default_target(&node.target) {
            return Self::parse_raw_default(node, ConvArgKind::Conv1dDefault, 1);
        }
        if is_conv2d_default_target(&node.target) {
            return Self::parse_raw_default(node, ConvArgKind::Conv2dDefault, 2);
        }
        if is_conv3d_default_target(&node.target) {
            return Self::parse_raw_default(node, ConvArgKind::Conv3dDefault, 3);
        }
        if is_aten_convolution_default_target(&node.target) {
            return Self::parse_convolution_default(node);
        }
        None
    }

    pub(crate) fn parse_convolution_default(node: &IrNode) -> Option<Self> {
        if node.op != NodeOp::CallFunction || !is_aten_convolution_default_target(&node.target) {
            return None;
        }
        let args = &node.args;
        if args.len() < 9 {
            return None;
        }
        let (bias, bias_arg, bias_well_formed) =
            parse_optional_node_arg(args.get(Self::BIAS_INDEX));
        Some(Self {
            kind: ConvArgKind::ConvolutionDefault,
            input: args.get(Self::INPUT_INDEX)?.as_node()?,
            weight: args.get(Self::WEIGHT_INDEX)?.as_node()?,
            bias,
            bias_arg,
            bias_well_formed,
            stride: args.get(3)?.as_int_list()?,
            padding: args.get(4)?.as_int_list()?,
            dilation: args.get(5)?.as_int_list()?,
            transposed: args.get(6)?.as_bool().unwrap_or(false),
            output_padding: args.get(7)?.as_int_list()?,
            groups: args.get(8)?.as_int()?,
        })
    }

    pub(crate) fn parse_convolution_prefix(node: &IrNode) -> Option<Self> {
        if node.op != NodeOp::CallFunction || !is_aten_convolution_default_target(&node.target) {
            return None;
        }
        let args = &node.args;
        if args.len() < 2 {
            return None;
        }
        let (bias, bias_arg, bias_well_formed) =
            parse_optional_node_arg(args.get(Self::BIAS_INDEX));
        Some(Self {
            kind: ConvArgKind::ConvolutionDefault,
            input: args.get(Self::INPUT_INDEX)?.as_node()?,
            weight: args.get(Self::WEIGHT_INDEX)?.as_node()?,
            bias,
            bias_arg,
            bias_well_formed,
            stride: args
                .get(3)
                .and_then(IrArg::as_int_list)
                .unwrap_or_else(|| vec![1, 1]),
            padding: args
                .get(4)
                .and_then(IrArg::as_int_list)
                .unwrap_or_else(|| vec![0, 0]),
            dilation: args
                .get(5)
                .and_then(IrArg::as_int_list)
                .unwrap_or_else(|| vec![1, 1]),
            transposed: args.get(6).and_then(IrArg::as_bool).unwrap_or(false),
            output_padding: args
                .get(7)
                .and_then(IrArg::as_int_list)
                .unwrap_or_else(|| vec![0, 0]),
            groups: args.get(8).and_then(IrArg::as_int).unwrap_or(1),
        })
    }

    fn parse_raw_default(node: &IrNode, kind: ConvArgKind, spatial_rank: usize) -> Option<Self> {
        let args = &node.args;
        if args.len() < 2 {
            return None;
        }
        let (bias, bias_arg, bias_well_formed) =
            parse_optional_node_arg(args.get(Self::BIAS_INDEX));
        Some(Self {
            kind,
            input: args.get(Self::INPUT_INDEX)?.as_node()?,
            weight: args.get(Self::WEIGHT_INDEX)?.as_node()?,
            bias,
            bias_arg,
            bias_well_formed,
            stride: node
                .arg_int_list(3)
                .unwrap_or_else(|| vec![1; spatial_rank]),
            padding: node
                .arg_int_list(4)
                .unwrap_or_else(|| vec![0; spatial_rank]),
            dilation: node
                .arg_int_list(5)
                .unwrap_or_else(|| vec![1; spatial_rank]),
            transposed: false,
            output_padding: vec![0; spatial_rank],
            groups: node.arg_int_or(6, 1),
        })
    }

    pub(crate) fn is_conv1d_form(&self) -> bool {
        self.kind == ConvArgKind::Conv1dDefault
            || (self.kind == ConvArgKind::ConvolutionDefault && self.stride.len() == 1)
    }

    pub(crate) fn is_raw_conv2d(&self) -> bool {
        self.kind == ConvArgKind::Conv2dDefault
    }

    pub(crate) fn group_arg_index(&self) -> usize {
        match self.kind {
            ConvArgKind::Conv1dDefault
            | ConvArgKind::Conv2dDefault
            | ConvArgKind::Conv3dDefault => 6,
            ConvArgKind::ConvolutionDefault => 8,
        }
    }

    pub(crate) fn args_with_replaced_io_bias_groups(
        &self,
        original_args: &[IrArg],
        input: NodeId,
        weight: NodeId,
        bias_arg: IrArg,
        groups: i64,
    ) -> Option<Vec<IrArg>> {
        let mut args = vec![IrArg::Node(input), IrArg::Node(weight), bias_arg];
        match self.kind {
            ConvArgKind::Conv1dDefault
            | ConvArgKind::Conv2dDefault
            | ConvArgKind::Conv3dDefault => {
                for index in 3..self.group_arg_index() {
                    if let Some(arg) = original_args.get(index) {
                        args.push(arg.clone());
                    }
                }
            }
            ConvArgKind::ConvolutionDefault => {
                args.extend(
                    original_args
                        .get(3..self.group_arg_index())?
                        .iter()
                        .cloned(),
                );
            }
        }
        args.push(IrArg::Int(groups));
        Some(args)
    }

    pub(crate) fn well_formed_bias(&self) -> Option<Option<NodeId>> {
        self.bias_well_formed.then_some(self.bias)
    }
}

fn parse_optional_node_arg(arg: Option<&IrArg>) -> (Option<NodeId>, IrArg, bool) {
    match arg {
        Some(IrArg::Node(id)) => (Some(*id), IrArg::Node(*id), true),
        Some(IrArg::None) | None => (None, IrArg::None, true),
        _ => (None, IrArg::None, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{i64_list_arg, IrGraph, NodeMeta};
    use crate::test_ir::{call_with_meta, placeholder};

    #[test]
    fn parses_conv1d_defaults_as_1d_form() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [1, 3, 16], "torch.float32");
        let weight = placeholder(&mut graph, "weight", [4, 3, 3], "torch.float32");
        let conv = call_with_meta(
            &mut graph,
            "conv",
            "aten.conv1d.default",
            vec![IrArg::Node(input), IrArg::Node(weight)],
            NodeMeta::tensor([1, 4, 14], "torch.float32"),
        );

        let args = ConvArgs::parse(graph.node(conv)).unwrap();
        assert_eq!(args.kind, ConvArgKind::Conv1dDefault);
        assert!(args.is_conv1d_form());
        assert_eq!(args.stride, vec![1]);
        assert_eq!(args.padding, vec![0]);
        assert_eq!(args.dilation, vec![1]);
        assert_eq!(args.output_padding, vec![0]);
        assert_eq!(args.groups, 1);
        assert_eq!(args.group_arg_index(), 6);
    }

    #[test]
    fn parses_convolution_default_strict_signature() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [1, 3, 16], "torch.float32");
        let weight = placeholder(&mut graph, "weight", [4, 3, 3], "torch.float32");
        let bias = placeholder(&mut graph, "bias", [4], "torch.float32");
        let conv = call_with_meta(
            &mut graph,
            "conv",
            "aten.convolution.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::Node(bias),
                i64_list_arg(&[2]),
                i64_list_arg(&[1]),
                i64_list_arg(&[3]),
                IrArg::Bool(false),
                i64_list_arg(&[0]),
                IrArg::Int(2),
            ],
            NodeMeta::tensor([1, 4, 8], "torch.float32"),
        );

        let args = ConvArgs::parse_convolution_default(graph.node(conv)).unwrap();
        assert_eq!(args.input, input);
        assert_eq!(args.weight, weight);
        assert_eq!(args.bias, Some(bias));
        assert_eq!(args.well_formed_bias(), Some(Some(bias)));
        assert!(args.is_conv1d_form());
        assert_eq!(args.groups, 2);
        assert_eq!(args.group_arg_index(), 8);
    }

    #[test]
    fn parses_conv2d_group_signature() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [1, 4, 8, 8], "torch.float32");
        let weight = placeholder(&mut graph, "weight", [8, 2, 3, 3], "torch.float32");
        let conv = call_with_meta(
            &mut graph,
            "conv",
            "aten.conv2d.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::None,
                i64_list_arg(&[1, 1]),
                i64_list_arg(&[0, 0]),
                i64_list_arg(&[1, 1]),
                IrArg::Int(2),
            ],
            NodeMeta::tensor([1, 8, 6, 6], "torch.float32"),
        );

        let args = ConvArgs::parse(graph.node(conv)).unwrap();
        assert_eq!(args.kind, ConvArgKind::Conv2dDefault);
        assert!(!args.is_conv1d_form());
        assert!(args.is_raw_conv2d());
        assert_eq!(args.groups, 2);
        assert_eq!(args.group_arg_index(), 6);
    }

    #[test]
    fn parses_conv3d_spatial_signature() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "input", [1, 4, 8, 8, 8], "torch.float32");
        let weight = placeholder(&mut graph, "weight", [8, 2, 3, 3, 3], "torch.float32");
        let conv = call_with_meta(
            &mut graph,
            "conv",
            "aten.conv3d.default",
            vec![
                IrArg::Node(input),
                IrArg::Node(weight),
                IrArg::None,
                i64_list_arg(&[1, 2, 3]),
                i64_list_arg(&[0, 1, 2]),
                i64_list_arg(&[1, 1, 2]),
                IrArg::Int(2),
            ],
            NodeMeta::tensor([1, 8, 6, 3, 2], "torch.float32"),
        );

        let args = ConvArgs::parse(graph.node(conv)).unwrap();
        assert_eq!(args.kind, ConvArgKind::Conv3dDefault);
        assert_eq!(args.weight, weight);
        assert_eq!(args.stride, vec![1, 2, 3]);
        assert_eq!(args.padding, vec![0, 1, 2]);
        assert_eq!(args.dilation, vec![1, 1, 2]);
        assert_eq!(args.groups, 2);
        assert_eq!(args.group_arg_index(), 6);
    }
}
