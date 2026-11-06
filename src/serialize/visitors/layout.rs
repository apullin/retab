use super::*;

/// A TOSA conv's output tensor. Its val is the TOSA output shape; only the shape comes
/// from it, the dtype stays the accumulator's, because the TOSA conv inherits the aten
/// conv's int8 output qparams (P2 design G1, critique A9).
pub(super) fn conv_output_tensor(node: &Node) -> TosaTensor {
    let semantic_dtype = node_output_dtype(node);
    let dtype = if node.meta.input_qparams.is_empty() && is_float_dtype(semantic_dtype) {
        semantic_dtype
    } else {
        conv_accumulator_dtype(node)
    };
    TosaTensor::new(&node.name, raw_i32_shape(node), dtype)
}

/// A TOSA pool's output tensor: its val is the TOSA output shape (P2 design G1,
/// critique B1).
pub(super) fn pool_output_tensor(node: &Node) -> TosaTensor {
    TosaTensor::new(&node.name, raw_i32_shape(node), node_output_dtype(node))
}

fn conv_pad_remainder(input_len: i32, pad: i32, dilation: i32, kernel: i32, stride: i32) -> i32 {
    (input_len + 2 * pad - dilation * (kernel - 1) - 1) % stride
}

fn conv_spatial_inputs_and_kernels(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    spatial_rank: usize,
    is_depthwise: bool,
) -> Option<(Vec<i32>, Vec<i32>)> {
    let tensor_rank = spatial_rank + 2;
    let input_node = graph.node(node.args.first()?.as_node()?)?;
    let weight_node = graph.node(node.args.get(1)?.as_node()?)?;

    let input_shape = actual_tensor_shape(graph, store, input_node);
    let weight_shape = actual_tensor_shape(graph, store, weight_node);
    if input_shape.len() != tensor_rank || weight_shape.len() != tensor_rank {
        return None;
    }

    let input_channels = *input_shape.get(spatial_rank + 1)?;
    let input_dims = input_shape.get(1..1 + spatial_rank)?.to_vec();

    // `actual_tensor_shape` may have reordered a depthwise placeholder's dims and
    // swapped KH/KW, so a conv1d kernel of 14 taps shows up as [OC, 1, 14, 1]
    // instead of [OC, 1, 1, 14]. Reading spatial dims off that yields KW=1, the
    // padding remainder comes out 0, the trailing-pad trim is skipped, and the
    // op fails TOSA's exact-division rule. The logical OIHW metadata is
    // authoritative; fall back to the HWCM form only when it is not.
    let kernels = if is_depthwise {
        let logical = weight_node.meta.val.shape_slice();
        if logical.len() == tensor_rank && logical.get(1).copied() == Some(1) {
            logical
                .get(2..2 + spatial_rank)?
                .iter()
                .map(|&d| d as i32)
                .collect()
        } else if spatial_rank == 2 && weight_shape.get(2).copied() == Some(input_channels) {
            weight_shape.get(0..2)?.to_vec()
        } else {
            return None;
        }
    } else {
        let kernel_start = if weight_shape.get(spatial_rank + 1).copied() == Some(input_channels) {
            1
        } else {
            2
        };
        weight_shape
            .get(kernel_start..kernel_start + spatial_rank)?
            .to_vec()
    };

    if crate::config::debug_layout_enabled() {
        eprintln!(
            "[ConvPadDebug] node={} depthwise={} input={:?} weight={:?} logical={:?} kernels={:?}",
            node.name,
            is_depthwise,
            input_shape,
            weight_shape,
            weight_node.meta.val.shape_slice(),
            kernels
        );
    }
    Some((input_dims, kernels))
}

fn same_padding_pair(input_len: i32, kernel: i32, stride: i32, dilation: i32) -> (i32, i32) {
    let out_len = (input_len + stride - 1) / stride;
    let needed = ((out_len - 1) * stride + dilation * (kernel - 1) + 1 - input_len).max(0);
    let before = needed / 2;
    let after = needed - before;
    (before, after)
}

pub(super) fn adjusted_tosa_pad_for_raw_aten_conv(
    graph: &Graph,
    store: &TensorStore,
    node: &Node,
    padding: &[i32],
    stride: &[i32],
    dilation: &[i32],
    is_depthwise: bool,
) -> Vec<i32> {
    let symmetric_pad = if padding.len() >= 2 {
        vec![padding[0], padding[0], padding[1], padding[1]]
    } else if padding.len() == 1 {
        vec![padding[0], padding[0], padding[0], padding[0]]
    } else {
        vec![0, 0, 0, 0]
    };

    let Some((input_dims, kernel_dims)) =
        conv_spatial_inputs_and_kernels(graph, store, node, 2, is_depthwise)
    else {
        return symmetric_pad;
    };

    let input_h = input_dims[0];
    let input_w = input_dims[1];
    let kernel_h = kernel_dims[0];
    let kernel_w = kernel_dims[1];
    let (stride_h, stride_w) = i32_pair_or_first(stride, 1);
    let (dilation_h, dilation_w) = i32_pair_or_first(dilation, 1);

    let padding_mode = arg_string(node, 4).or_else(|| kwarg_string(node, "padding"));
    if matches!(padding_mode.as_deref(), Some("same")) {
        let (pad_top, pad_bottom) = same_padding_pair(input_h, kernel_h, stride_h, dilation_h);
        let (pad_left, pad_right) = same_padding_pair(input_w, kernel_w, stride_w, dilation_w);
        return vec![pad_top, pad_bottom, pad_left, pad_right];
    }

    let (pad_h, pad_w) = i32_pair_or_first(padding, 0);

    let pad_bottom = {
        let remainder = conv_pad_remainder(input_h, pad_h, dilation_h, kernel_h, stride_h);
        if remainder > pad_h {
            pad_h
        } else {
            pad_h - remainder
        }
    };
    let pad_right = {
        let remainder = conv_pad_remainder(input_w, pad_w, dilation_w, kernel_w, stride_w);
        if remainder > pad_w {
            pad_w
        } else {
            pad_w - remainder
        }
    };

    vec![pad_h, pad_bottom, pad_w, pad_right]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{DType as IrDType, NodeMeta, TensorInfo};
    use smallvec::{smallvec, SmallVec};

    fn add(
        graph: &mut Graph,
        name: &str,
        op: OpType,
        target: OpTarget,
        args: SmallVec<[Arg; 4]>,
        shape: &[i64],
    ) -> NodeId {
        graph.create_node(
            op,
            target,
            args,
            Vec::new(),
            NodeMeta::new(TensorInfo::new(shape, IrDType::Int8)),
            name.into(),
        )
    }

    fn input(graph: &mut Graph, name: &str, shape: &[i64]) -> NodeId {
        add(
            graph,
            name,
            OpType::Placeholder,
            OpTarget::Other(name.into()),
            smallvec![],
            shape,
        )
    }

    fn list(values: &[i64]) -> Arg {
        Arg::IntList(values.iter().copied().collect())
    }

    /// `<target>(x', w, b, stride, pad, [1, 1])` on a channels-last input.
    fn tosa_conv(
        target: OpTarget,
        input_shape: &[i64],
        weight: &[i64],
        stride: &[i64],
        pad: &[i64],
        val: &[i64],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        let x = input(&mut graph, "x", input_shape);
        let w = input(&mut graph, "w", weight);
        let b = input(&mut graph, "b", &val[3..]);
        let conv = add(
            &mut graph,
            "conv",
            OpType::CallFunction,
            target,
            smallvec![
                Arg::Node(x),
                Arg::Node(w),
                Arg::Node(b),
                list(stride),
                list(pad),
                list(&[1, 1])
            ],
            val,
        );
        (graph, conv)
    }

    fn emit(graph: &Graph, id: NodeId) -> EmitResult {
        clear_shape_cache();
        emit_node(graph, id, graph.node(id).unwrap(), &TensorStore::new()).unwrap()
    }

    #[test]
    fn conv_pads_are_used_as_given() {
        // P2 design G2: rc4 already adjusted the pad, so [1, 1, 2, 2] stays as it is, and
        // the output shape is the val (G1).
        let (graph, conv) = tosa_conv(
            OpTarget::TosaConv2d,
            &[1, 6, 10, 3],
            &[8, 3, 5, 3],
            &[2, 2],
            &[1, 1, 2, 2],
            &[1, 3, 5, 8],
        );
        let result = emit(&graph, conv);
        let op = result.operators.last().unwrap();
        let Attribute::Conv2d(attr) = &op.attribute else {
            panic!("expected CONV2D, got {:?}", op.op);
        };
        assert_eq!(attr.pad, vec![1, 1, 2, 2]);
        assert_eq!(result.tensors.last().unwrap().shape, vec![1, 3, 5, 8]);
    }

    #[test]
    fn pool_outputs_are_the_val() {
        // P2 design G1: the output shape is the val, not recomputed from the arguments,
        // which would give [1, 4, 4, 4].
        let mut graph = Graph::new();
        let x = input(&mut graph, "x", &[1, 9, 9, 4]);
        let pool = add(
            &mut graph,
            "pool",
            OpType::CallFunction,
            OpTarget::TosaMaxPool2d,
            smallvec![
                Arg::Node(x),
                list(&[2, 2]),
                list(&[2, 2]),
                list(&[0, 0, 0, 0])
            ],
            &[1, 5, 5, 4],
        );
        let result = emit(&graph, pool);
        assert_eq!(result.operators.last().unwrap().op, Op::MaxPool2d);
        assert_eq!(result.tensors.last().unwrap().shape, vec![1, 5, 5, 4]);
    }
}
