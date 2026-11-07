//! Post-conversion graph finalization for serialize graphs produced from Rust IR.

use crate::graph::ir::*;
use crate::graph::Graph;

// ---------------------------------------------------------------------------
// Post-ingestion shape fixup
// ---------------------------------------------------------------------------

/// `shape` with its one -1 (if any) resolved against `input_shape`'s element count.
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

/// Give each Reshape/View/Flatten node its target shape (`args[1]`) as its val.
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
            let shape = node.args.get(1).and_then(Arg::as_int_list).map(Vec::from);
            let input_id = node.args.first().and_then(Arg::as_node);
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
}

// ---------------------------------------------------------------------------
// Post-ingestion dtype fixup
// ---------------------------------------------------------------------------

/// Returns true for ops that produce int32 accumulator outputs, IrRewriteConv's TOSA
/// CONV3D included (P2 design X1).
fn is_accumulator_op(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::TosaConv3d
            | OpTarget::TosaTransposeConv2d
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
            // TOSA ARGMAX always produces INT32 indices, whatever the input quantization.
            if target == OpTarget::ArgMax {
                if let Some(node) = graph.node_mut(node_id) {
                    node.meta.val.dtype = DType::Int32;
                }
                continue;
            }
            if is_accumulator_op(&target) && (has_input_qp || has_output_qp) {
                if let Some(node) = graph.node_mut(node_id) {
                    node.meta.val.dtype = DType::Int32;
                }
                continue;
            }

            // rc4's TOSA RESIZE inherits the upsample's qparams, but its dtype is its own:
            // int8 bilinear produces int32 (`tosa/dialect/ops/resize.py:28-36`). An
            // integer IR val already carries it, so don't retype it from the inherited
            // int8 output qparam (P2 design F10/X4). A float val (RETAB's usual typing of
            // quantized compute nodes) still takes the qparam dtype.
            if target == OpTarget::TosaResize && current_dtype.is_integer() {
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

            // A bool result (comparison, logical op, or a view of one) is never
            // quantized: its input qparams describe the operands, not the output.
            if current_dtype == DType::Bool && output_qp_dtype.is_none() {
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
        let (current_dt, users) = {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            if node.op != OpType::Placeholder {
                continue;
            }
            if node.meta.val.dtype != DType::Float32 {
                continue;
            }
            (node.meta.val.dtype, node.users.to_vec())
        };

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
            let (current_dt, input_id) = {
                let Some(node) = graph.node(*node_id) else {
                    continue;
                };
                if node.op != OpType::CallFunction {
                    continue;
                }
                if !matches!(
                    node.target,
                    OpTarget::Reshape
                        | OpTarget::View
                        | OpTarget::Flatten
                        | OpTarget::Squeeze
                        | OpTarget::Unsqueeze
                        | OpTarget::TosaReshape
                        | OpTarget::TosaIdentity
                        | OpTarget::Permute
                        | OpTarget::TosaAvgPool2d
                        | OpTarget::TosaMaxPool2d
                ) {
                    continue;
                }
                if node.meta.val.dtype != DType::Float32 {
                    continue;
                }
                (
                    node.meta.val.dtype,
                    node.args.first().and_then(Arg::as_node),
                )
            };

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
