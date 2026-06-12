use crate::graph::ir::*;
use crate::graph::Graph;

// ---------------------------------------------------------------------------
// Post-ingestion shape fixup
// ---------------------------------------------------------------------------

/// Fix output shapes for Reshape/View/Flatten nodes and their downstream conv.
///
/// The linear decomposition pass (`FxDecomposeLinearPass`) creates nodes whose
/// meta (including `val` FakeTensor) is copied from the original linear node.
/// This means the intermediate view_copy and convolution nodes have the linear
/// output's shape (e.g. `[1, 10]`) instead of their actual output shapes:
///   - view_copy(input, [1, 1568, 1, 1]) should have shape [1, 1568, 1, 1]
///   - view_copy(weights, [10, 1568, 1, 1]) should have shape [10, 1568, 1, 1]
///   - conv2d(input_4d, weight_4d, ...) should have 4D output shape
///   - view_copy(conv_out, [1, 10]) should have shape [1, 10]
///
/// For reshape-like ops: read the target shape from args[1] (IntList).
/// For conv: compute output shape from input shape, weight shape, and params.
fn resolve_inferred_reshape_shape(shape: &[i64], input_shape: &[i64]) -> Option<Vec<i64>> {
    let negative_positions: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (dim < 0).then_some(idx))
        .collect();
    if negative_positions.is_empty() {
        return Some(shape.to_vec());
    }
    if negative_positions.len() > 1 {
        return None;
    }

    let input_numel = input_shape
        .iter()
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim))?;
    let known_product = shape
        .iter()
        .filter(|&&dim| dim >= 0)
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim))?;
    if known_product == 0 || input_numel % known_product != 0 {
        return None;
    }

    let mut resolved = shape.to_vec();
    resolved[negative_positions[0]] = input_numel / known_product;
    Some(resolved)
}

fn fixup_reshape_shapes(graph: &mut Graph) {
    let all_ids = graph.all_node_ids();
    for node_id in &all_ids {
        let (arg1_shape, input_id) = {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            if node.op != OpType::CallFunction {
                continue;
            }
            let is_reshape = matches!(
                node.target,
                OpTarget::Reshape | OpTarget::View | OpTarget::Flatten
            );
            if !is_reshape {
                continue;
            }
            let shape = match node.args.get(1) {
                Some(Arg::IntList(list)) => {
                    let v: Vec<i64> = list.iter().copied().collect();
                    Some(v)
                }
                _ => None,
            };
            let input_id = match node.args.first() {
                Some(Arg::Node(id)) => Some(*id),
                _ => None,
            };
            (shape, input_id)
        };

        if let Some(new_shape) = arg1_shape {
            let resolved_shape = if new_shape.iter().any(|&dim| dim < 0) {
                input_id
                    .and_then(|id| graph.node(id))
                    .and_then(|input| {
                        resolve_inferred_reshape_shape(&new_shape, input.meta.val.shape_slice())
                    })
                    .unwrap_or(new_shape)
            } else {
                new_shape
            };
            let node = graph.node(*node_id).unwrap();
            let current = node.meta.val.shape_slice();
            if current != resolved_shape.as_slice() {
                let dt = node.meta.val.dtype;
                if let Some(node_mut) = graph.node_mut(*node_id) {
                    node_mut.meta.val = TensorInfo::new(&resolved_shape, dt);
                }
            }
        }
    }

    // Also fix Conv2d nodes whose output shape doesn't match their inputs.
    let all_ids2 = graph.all_node_ids();
    for node_id in &all_ids2 {
        let (is_conv, input_id, weight_id, current_shape_len) = {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            if node.op != OpType::CallFunction {
                continue;
            }
            let is_conv = matches!(node.target, OpTarget::Conv2d);
            if !is_conv {
                continue;
            }
            let input_id = match node.args.first() {
                Some(Arg::Node(id)) => Some(*id),
                _ => None,
            };
            let weight_id = match node.args.get(1) {
                Some(Arg::Node(id)) => Some(*id),
                _ => None,
            };
            (true, input_id, weight_id, node.meta.val.shape_len)
        };

        if !is_conv || current_shape_len == 4 {
            continue;
        }

        if let (Some(input_id), Some(weight_id)) = (input_id, weight_id) {
            let (input_shape, weight_shape) = {
                let input_node = graph.node(input_id);
                let weight_node = graph.node(weight_id);
                (
                    input_node.map(|n| n.meta.val.shape_slice().to_vec()),
                    weight_node.map(|n| n.meta.val.shape_slice().to_vec()),
                )
            };

            if let (Some(ishape), Some(wshape)) = (input_shape, weight_shape) {
                if ishape.len() == 4 && wshape.len() == 4 {
                    let node = graph.node(*node_id).unwrap();
                    let stride = match node.args.get(3) {
                        Some(Arg::IntList(l)) => l.iter().copied().collect::<Vec<_>>(),
                        _ => vec![1, 1],
                    };
                    let padding = match node.args.get(4) {
                        Some(Arg::IntList(l)) => l.iter().copied().collect::<Vec<_>>(),
                        _ => vec![0, 0],
                    };
                    let dilation = match node.args.get(5) {
                        Some(Arg::IntList(l)) => l.iter().copied().collect::<Vec<_>>(),
                        _ => vec![1, 1],
                    };

                    let batch = ishape[0];
                    let ih = ishape[1];
                    let iw = ishape[2];
                    let out_channels = wshape[0];
                    let kh = wshape[1];
                    let kw = wshape[2];
                    let sh = if stride.len() >= 2 { stride[0] } else { 1 };
                    let sw = if stride.len() >= 2 { stride[1] } else { 1 };
                    let ph = if padding.len() >= 2 { padding[0] } else { 0 };
                    let pw = if padding.len() >= 2 { padding[1] } else { 0 };
                    let dh = if dilation.len() >= 2 { dilation[0] } else { 1 };
                    let dw = if dilation.len() >= 2 { dilation[1] } else { 1 };

                    let oh = (ih + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
                    let ow = (iw + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

                    let new_shape = [batch, oh, ow, out_channels];
                    let dt = node.meta.val.dtype;
                    if let Some(node_mut) = graph.node_mut(*node_id) {
                        node_mut.meta.val = TensorInfo::new(&new_shape, dt);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Post-ingestion dtype fixup
// ---------------------------------------------------------------------------

/// Returns true for ops that produce int32 accumulator outputs.
fn is_accumulator_op(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Conv2d
            | OpTarget::DepthwiseConv2d
            | OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::TosaTransposeConv2d
            | OpTarget::TosaFullyConnected
            | OpTarget::Linear
            | OpTarget::Addmm
            | OpTarget::Matmul
            | OpTarget::Bmm
            | OpTarget::TosaMatmul
    )
}

/// Fix up `meta.val.dtype` on all nodes based on quantization parameters.
fn fixup_quantized_dtypes(graph: &mut Graph) {
    for _outer in 0..8 {
        let before: Vec<DType> = graph
            .all_node_ids()
            .iter()
            .filter_map(|id| graph.node(*id))
            .map(|n| n.meta.val.dtype)
            .collect();
        fixup_quantized_dtypes_once(graph);
        let after: Vec<DType> = graph
            .all_node_ids()
            .iter()
            .filter_map(|id| graph.node(*id))
            .map(|n| n.meta.val.dtype)
            .collect();
        if before == after {
            break;
        }
    }
}

/// Run the post-ingest graph fixups on a Graph built by the IrGraph converter.
pub(crate) fn finalize_converted_graph(graph: &mut Graph) {
    fixup_reshape_shapes(graph);
    fixup_quantized_dtypes(graph);
}

fn fixup_quantized_dtypes_once(graph: &mut Graph) {
    let all_ids = graph.all_node_ids();
    for node_id in all_ids {
        let (
            op,
            target,
            has_input_qp,
            has_output_qp,
            input_qp_dtype,
            output_qp_dtype,
            current_dtype,
        ) = {
            let Some(node) = graph.node(node_id) else {
                continue;
            };
            let iqp_dtype = node.meta.input_qparams.first().map(|(_, qa)| qa.dtype);
            let oqp_dtype = node.meta.output_qparams.first().map(|(_, qa)| qa.dtype);
            (
                node.op,
                node.target.clone(),
                !node.meta.input_qparams.is_empty(),
                !node.meta.output_qparams.is_empty(),
                iqp_dtype,
                oqp_dtype,
                node.meta.val.dtype,
            )
        };

        if op == OpType::CallFunction {
            if is_accumulator_op(&target) && (has_input_qp || has_output_qp) {
                if let Some(node) = graph.node_mut(node_id) {
                    node.meta.val.dtype = DType::Int32;
                }
                continue;
            }

            if target == OpTarget::TosaTable {
                let table_dtype = graph.node(node_id).and_then(|n| match n.args.first() {
                    Some(Arg::Node(input_id)) => {
                        graph.node(*input_id).map(|input| input.meta.val.dtype)
                    }
                    _ => input_qp_dtype,
                });
                if let Some(dt) = table_dtype {
                    if let Some(node) = graph.node_mut(node_id) {
                        node.meta.val.dtype = if dt == DType::Int16 { DType::Int32 } else { dt };
                    }
                }
                continue;
            }

            if target == OpTarget::TosaRescale {
                let rescale_dtype = graph.node(node_id).and_then(|n| match n.args.get(1) {
                    Some(Arg::Dtype(dt)) => Some(*dt),
                    _ => output_qp_dtype,
                });
                if let Some(dt) = rescale_dtype {
                    if let Some(node) = graph.node_mut(node_id) {
                        node.meta.val.dtype = dt;
                    }
                }
                continue;
            }

            if matches!(
                target,
                OpTarget::DequantizePerTensor | OpTarget::DequantizePerChannel
            ) {
                continue;
            }

            let new_dtype = output_qp_dtype.or(input_qp_dtype);
            if let Some(dt) = new_dtype {
                if let Some(node) = graph.node_mut(node_id) {
                    node.meta.val.dtype = dt;
                }
            }
        }

        if op == OpType::Placeholder && current_dtype == DType::Int64 {
            if let Some(node) = graph.node_mut(node_id) {
                node.meta.val.dtype = DType::Int32;
            }
        }
    }

    let all_ids2 = graph.all_node_ids();
    for node_id in &all_ids2 {
        let (is_placeholder, current_dt, users) = {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            if node.op != OpType::Placeholder {
                continue;
            }
            if node.meta.val.dtype != DType::Float32 {
                continue;
            }
            (true, node.meta.val.dtype, node.users.to_vec())
        };

        if !is_placeholder {
            continue;
        }

        for user_id in &users {
            let inferred_dtype = {
                let Some(user) = graph.node(*user_id) else {
                    continue;
                };
                let pos = user.args.iter().enumerate().find_map(|(i, arg)| match arg {
                    Arg::Node(id) if *id == *node_id => Some(i as u32),
                    _ => None,
                });
                let Some(pos) = pos else {
                    continue;
                };
                user.meta
                    .input_qparams
                    .iter()
                    .find(|(p, _)| *p == pos)
                    .map(|(_, qa)| qa.dtype)
            };

            if let Some(dt) = inferred_dtype {
                if dt != current_dt {
                    if let Some(node) = graph.node_mut(*node_id) {
                        node.meta.val.dtype = dt;
                    }
                }
                break;
            }
        }
    }

    for _iter in 0..10 {
        let all_ids4 = graph.all_node_ids();
        let mut changed = false;
        for node_id in &all_ids4 {
            let (is_reshape, current_dt, input_id) = {
                let Some(node) = graph.node(*node_id) else {
                    continue;
                };
                if node.op != OpType::CallFunction {
                    continue;
                }
                let is_reshape = matches!(
                    node.target,
                    OpTarget::Reshape
                        | OpTarget::View
                        | OpTarget::Flatten
                        | OpTarget::Squeeze
                        | OpTarget::Unsqueeze
                        | OpTarget::TosaReshape
                        | OpTarget::Permute
                        | OpTarget::TosaTranspose
                        | OpTarget::AvgPool2d
                        | OpTarget::MaxPool2d
                        | OpTarget::AdaptiveAvgPool2d
                        | OpTarget::TosaAvgPool2d
                        | OpTarget::TosaMaxPool2d
                );
                if !is_reshape {
                    continue;
                }
                if node.meta.val.dtype != DType::Float32 {
                    continue;
                }
                let input_id = match node.args.first() {
                    Some(Arg::Node(id)) => Some(*id),
                    _ => None,
                };
                (true, node.meta.val.dtype, input_id)
            };

            if !is_reshape {
                continue;
            }

            if let Some(input_id) = input_id {
                let input_dt = graph.node(input_id).map(|n| n.meta.val.dtype);
                if let Some(dt) = input_dt {
                    if dt != current_dt && dt != DType::Float32 {
                        if let Some(node) = graph.node_mut(*node_id) {
                            node.meta.val.dtype = dt;
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}
