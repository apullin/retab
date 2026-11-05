//! Rust-native fold_qdq implementation and parity helpers.
//!
//! This module is intentionally narrow:
//! - target-name matching helpers
//! - foldable-node classification from typed `NodeMeta`
//! - qparam resolution from literal or tensor-backed IR args

use crate::fx_ir::{IrArg, IrGraph, IrNode, IrPass, MetaValue, NodeId, NodeMeta, NodeOp, QParam};
use crate::fx_target::{
    classify_ir_target, is_add_sub_target, is_comparison_target, is_fold_qdq_constant_target,
    is_minmax_target, is_mul_target, is_sum_dim_int_list_target, IrTargetKind,
};
use crate::store::TensorData;

fn is_default_partial_binary_tensor_target(target: &str) -> bool {
    is_add_sub_target(target)
}

fn is_mixed_profile_partial_binary_tensor_target(target: &str) -> bool {
    is_add_sub_target(target)
        || is_mul_target(target)
        || target == "aten.div.Tensor"
        || is_minmax_target(target)
        || target == "aten.mm.default"
        || target == "aten.bmm.default"
        || is_comparison_target(target)
        || target == "aten.grid_sampler_2d.default"
}

fn has_partial_binary_tensor_qdq_inputs(
    node: &IrNode,
    quantized_arg_indices: &[usize],
    preserve_mixed_profile: bool,
) -> bool {
    if preserve_mixed_profile {
        if !is_mixed_profile_partial_binary_tensor_target(&node.target) {
            return false;
        }
    } else if !is_default_partial_binary_tensor_target(&node.target) {
        return false;
    }
    let [lhs, rhs, ..] = node.args.as_slice() else {
        return false;
    };
    if !matches!(lhs, IrArg::Node(_)) || !matches!(rhs, IrArg::Node(_)) {
        return false;
    }
    quantized_arg_indices.contains(&0) != quantized_arg_indices.contains(&1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FoldQdqQParamSource {
    InputNodeMeta,
    OutputNodeMeta,
    LiteralArgs,
    TensorBackedArgs,
}

#[derive(Debug, Clone)]
struct ResolvedFoldQdqQParam {
    qparam: QParam,
    source: FoldQdqQParamSource,
}

#[derive(Debug, Clone)]
struct ResolvedFoldQdqInput {
    arg_index: usize,
    qparam: ResolvedFoldQdqQParam,
    dq_node_ids: Vec<NodeId>,
}

fn has_arm_quantized_annotation(meta: &NodeMeta) -> bool {
    meta.extra_bool("arm_quantized_annotation") == Some(true)
}

fn is_foldable_node(node: &IrNode) -> bool {
    node.op == NodeOp::CallFunction
        && !node.is_q_or_dq()
        && (is_fold_qdq_constant_target(&node.target) || has_arm_quantized_annotation(&node.meta))
}

fn resolve_input_qparam_from_meta(meta: &NodeMeta) -> Option<ResolvedFoldQdqQParam> {
    Some(ResolvedFoldQdqQParam {
        qparam: meta.input_qparam()?.clone(),
        source: FoldQdqQParamSource::InputNodeMeta,
    })
}

fn resolve_output_qparam_from_meta(meta: &NodeMeta) -> Option<ResolvedFoldQdqQParam> {
    Some(ResolvedFoldQdqQParam {
        qparam: meta.output_qparam()?.clone(),
        source: FoldQdqQParamSource::OutputNodeMeta,
    })
}

fn resolve_qparam_from_meta(meta: &NodeMeta) -> Option<ResolvedFoldQdqQParam> {
    resolve_output_qparam_from_meta(meta).or_else(|| resolve_input_qparam_from_meta(meta))
}

fn resolve_qparam_from_node_meta(node: &IrNode) -> Option<ResolvedFoldQdqQParam> {
    resolve_qparam_from_meta(&node.meta)
}

fn resolve_qparam_from_q_or_dq(graph: &IrGraph, node: &IrNode) -> Option<ResolvedFoldQdqQParam> {
    let mut tensor_lookup = |id| {
        graph
            .node(id)
            .tensor
            .as_ref()
            .map(|node_tensor| &node_tensor.data)
    };
    resolve_qparam_from_q_or_dq_args(&node.target, &node.args, &mut tensor_lookup)
}

fn resolve_qparam_from_arg(graph: &IrGraph, arg: &IrArg) -> Option<ResolvedFoldQdqQParam> {
    let IrArg::Node(id) = arg else {
        return None;
    };
    let node = graph.node(*id);
    resolve_qparam_from_q_or_dq(graph, node).or_else(|| resolve_qparam_from_node_meta(node))
}

fn resolve_input_qparams_for_foldable_node(
    graph: &IrGraph,
    node: &IrNode,
) -> Vec<ResolvedFoldQdqInput> {
    node.args
        .iter()
        .enumerate()
        .filter_map(|(idx, arg)| resolve_input_qparam_for_arg(graph, idx, arg))
        .collect()
}

fn resolve_input_qparam_for_arg(
    graph: &IrGraph,
    arg_index: usize,
    arg: &IrArg,
) -> Option<ResolvedFoldQdqInput> {
    match arg {
        IrArg::Node(id) => {
            let qparam = resolve_qparam_from_arg(graph, arg)?;
            let dq_node_ids = if graph.node(*id).is_dequantize_any() {
                vec![*id]
            } else {
                Vec::new()
            };
            Some(ResolvedFoldQdqInput {
                arg_index,
                qparam,
                dq_node_ids,
            })
        }
        IrArg::List(items) => {
            let mut qparam = None;
            let mut dq_node_ids = Vec::new();

            for item in items {
                let IrArg::Node(id) = item else {
                    return None;
                };
                let item_node = graph.node(*id);
                if !item_node.is_dequantize_any() {
                    continue;
                }
                if let Some(resolved) = resolve_qparam_from_q_or_dq(graph, item_node) {
                    qparam = Some(resolved);
                    dq_node_ids.push(*id);
                }
            }

            qparam.map(|qparam| ResolvedFoldQdqInput {
                arg_index,
                qparam,
                dq_node_ids,
            })
        }
        _ => None,
    }
}

fn resolve_qparam_from_q_or_dq_args<'a, F>(
    target: &str,
    args: &[IrArg],
    tensor_lookup: &mut F,
) -> Option<ResolvedFoldQdqQParam>
where
    F: FnMut(usize) -> Option<&'a TensorData>,
{
    let target_kind = classify_ir_target(target);
    if !matches!(
        target_kind,
        IrTargetKind::QuantizePerTensor
            | IrTargetKind::QuantizePerChannel
            | IrTargetKind::DequantizePerTensor
            | IrTargetKind::DequantizePerChannel
    ) {
        return None;
    }

    if matches!(
        target_kind,
        IrTargetKind::QuantizePerChannel | IrTargetKind::DequantizePerChannel
    ) {
        resolve_per_channel_qparam(args, tensor_lookup)
    } else {
        resolve_per_tensor_qparam(args, tensor_lookup)
    }
}

fn resolve_per_tensor_qparam<'a, F>(
    args: &[IrArg],
    tensor_lookup: &mut F,
) -> Option<ResolvedFoldQdqQParam>
where
    F: FnMut(usize) -> Option<&'a TensorData>,
{
    if args.len() < 6 {
        return None;
    }

    let (scale_values, scale_source) = f64_values_from_arg(&args[1], tensor_lookup)?;
    let (zp_values, zp_source) = i64_values_from_arg(&args[2], tensor_lookup)?;
    let qmin = scalar_i64_from_arg(&args[3], tensor_lookup)?;
    let qmax = scalar_i64_from_arg(&args[4], tensor_lookup)?;
    let dtype_str = dtype_string_from_arg(&args[5], tensor_lookup)?;

    let scale = *scale_values.first()?;
    let zp = *zp_values.first()?;

    Some(ResolvedFoldQdqQParam {
        qparam: QParam::per_tensor(scale, zp, qmin, qmax, dtype_str),
        source: merge_qparam_sources(scale_source, zp_source),
    })
}

fn resolve_per_channel_qparam<'a, F>(
    args: &[IrArg],
    tensor_lookup: &mut F,
) -> Option<ResolvedFoldQdqQParam>
where
    F: FnMut(usize) -> Option<&'a TensorData>,
{
    if args.len() < 7 {
        return None;
    }

    let (scale_values, scale_source) = f64_values_from_arg(&args[1], tensor_lookup)?;
    let (zp_values, zp_source) = i64_values_from_arg(&args[2], tensor_lookup)?;
    let axis = scalar_i64_from_arg(&args[3], tensor_lookup)?;
    let qmin = scalar_i64_from_arg(&args[4], tensor_lookup)?;
    let qmax = scalar_i64_from_arg(&args[5], tensor_lookup)?;
    let dtype_str = dtype_string_from_arg(&args[6], tensor_lookup)?;

    if scale_values.is_empty() {
        return None;
    }

    let per_channel_zps = match zp_values.len() {
        0 => vec![0; scale_values.len()],
        1 if scale_values.len() > 1 => vec![zp_values[0]; scale_values.len()],
        len if len == scale_values.len() => zp_values,
        _ => return None,
    };

    Some(ResolvedFoldQdqQParam {
        qparam: QParam {
            scale: scale_values[0],
            zp: per_channel_zps[0],
            qmin,
            qmax,
            dtype_str,
            per_channel_scales: scale_values,
            per_channel_zps,
            axis,
        },
        source: merge_qparam_sources(scale_source, zp_source),
    })
}

fn merge_qparam_sources(lhs: FoldQdqQParamSource, rhs: FoldQdqQParamSource) -> FoldQdqQParamSource {
    if lhs == FoldQdqQParamSource::TensorBackedArgs || rhs == FoldQdqQParamSource::TensorBackedArgs
    {
        FoldQdqQParamSource::TensorBackedArgs
    } else {
        FoldQdqQParamSource::LiteralArgs
    }
}

fn scalar_i64_from_arg<'a, F>(arg: &IrArg, tensor_lookup: &mut F) -> Option<i64>
where
    F: FnMut(usize) -> Option<&'a TensorData>,
{
    let (values, _) = i64_values_from_arg(arg, tensor_lookup)?;
    values.first().copied()
}

fn i64_values_from_arg<'a, F>(
    arg: &IrArg,
    tensor_lookup: &mut F,
) -> Option<(Vec<i64>, FoldQdqQParamSource)>
where
    F: FnMut(usize) -> Option<&'a TensorData>,
{
    match arg {
        IrArg::Int(value) => Some((vec![*value], FoldQdqQParamSource::LiteralArgs)),
        IrArg::Float(value) => Some((vec![*value as i64], FoldQdqQParamSource::LiteralArgs)),
        IrArg::Bool(value) => Some((vec![i64::from(*value)], FoldQdqQParamSource::LiteralArgs)),
        IrArg::List(items) => {
            let values: Option<Vec<i64>> = items
                .iter()
                .map(|item| match item {
                    IrArg::Int(value) => Some(*value),
                    IrArg::Float(value) => Some(*value as i64),
                    IrArg::Bool(value) => Some(i64::from(*value)),
                    _ => None,
                })
                .collect();
            values.map(|values| (values, FoldQdqQParamSource::LiteralArgs))
        }
        IrArg::Node(id) => tensor_lookup(*id)
            .and_then(|tensor| tensor.vec_i64())
            .map(|values| (values, FoldQdqQParamSource::TensorBackedArgs)),
        _ => None,
    }
}

fn f64_values_from_arg<'a, F>(
    arg: &IrArg,
    tensor_lookup: &mut F,
) -> Option<(Vec<f64>, FoldQdqQParamSource)>
where
    F: FnMut(usize) -> Option<&'a TensorData>,
{
    match arg {
        IrArg::Float(value) => Some((vec![*value], FoldQdqQParamSource::LiteralArgs)),
        IrArg::Int(value) => Some((vec![*value as f64], FoldQdqQParamSource::LiteralArgs)),
        IrArg::Bool(value) => Some((
            vec![f64::from(*value as u8)],
            FoldQdqQParamSource::LiteralArgs,
        )),
        IrArg::List(items) => {
            let values: Option<Vec<f64>> = items
                .iter()
                .map(|item| match item {
                    IrArg::Float(value) => Some(*value),
                    IrArg::Int(value) => Some(*value as f64),
                    IrArg::Bool(value) => Some(f64::from(*value as u8)),
                    _ => None,
                })
                .collect();
            values.map(|values| (values, FoldQdqQParamSource::LiteralArgs))
        }
        IrArg::Node(id) => tensor_lookup(*id)
            .and_then(|tensor| tensor.vec_f64())
            .map(|values| (values, FoldQdqQParamSource::TensorBackedArgs)),
        _ => None,
    }
}

fn dtype_string_from_arg<'a, F>(arg: &IrArg, tensor_lookup: &mut F) -> Option<String>
where
    F: FnMut(usize) -> Option<&'a TensorData>,
{
    match arg {
        IrArg::Dtype(value) | IrArg::Str(value) => Some(value.clone()),
        IrArg::Node(id) => tensor_lookup(*id).map(|tensor| tensor.dtype.torch_str().to_string()),
        _ => None,
    }
}

pub struct IrFoldQdqPass {
    pub(crate) preserve_partial_binary_tensor_qdq: bool,
}

impl Default for IrFoldQdqPass {
    fn default() -> Self {
        Self::new()
    }
}

impl IrFoldQdqPass {
    pub fn new() -> Self {
        Self {
            preserve_partial_binary_tensor_qdq: false,
        }
    }

    #[cfg(test)]
    pub fn with_mixed_profile_partial_binary_qdq() -> Self {
        Self {
            preserve_partial_binary_tensor_qdq: true,
        }
    }

    fn candidate_node_ids(&self, graph: &IrGraph) -> Vec<usize> {
        graph
            .live_node_ids()
            .into_iter()
            .filter(|&id| is_foldable_node(graph.node(id)))
            .collect()
    }
}

impl IrPass for IrFoldQdqPass {
    fn name(&self) -> &str {
        "IrFoldQdq"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let candidate_ids = self.candidate_node_ids(graph);
        let mut modified = false;

        for node_id in candidate_ids {
            if !graph.node(node_id).alive {
                continue;
            }

            let node = graph.node(node_id).clone();
            let resolved_inputs = resolve_input_qparams_for_foldable_node(graph, &node);
            let quantized_arg_indices: Vec<usize> = resolved_inputs
                .iter()
                .map(|resolved| resolved.arg_index)
                .collect();
            let preserve_qdq = has_partial_binary_tensor_qdq_inputs(
                &node,
                &quantized_arg_indices,
                self.preserve_partial_binary_tensor_qdq,
            );

            let mut input_qparams = Vec::new();

            for resolved_input in resolved_inputs {
                let idx = resolved_input.arg_index;
                let resolved = resolved_input.qparam;
                input_qparams.push((idx, resolved.qparam.clone()));

                if preserve_qdq {
                    continue;
                }
                for arg_id in resolved_input.dq_node_ids {
                    let arg_node = graph.node(arg_id).clone();
                    if !arg_node.alive || !arg_node.is_dequantize_any() {
                        continue;
                    }
                    let Some(IrArg::Node(source_id)) = arg_node.args.first() else {
                        continue;
                    };

                    if matches!(
                        resolved.source,
                        FoldQdqQParamSource::TensorBackedArgs | FoldQdqQParamSource::LiteralArgs
                    ) && resolved.qparam.qmin == -7
                        && resolved.qparam.qmax == 7
                    {
                        graph
                            .node_mut(*source_id)
                            .meta
                            .set_extra("tosa_special_dtype", MetaValue::Str("INT4".to_string()));
                    }

                    modified |= graph.replace_input_use(node_id, arg_id, *source_id);
                    if !graph.has_live_users(arg_id) {
                        graph.erase(arg_id);
                        modified = true;
                    }
                }
            }

            if !input_qparams.is_empty() {
                graph
                    .node_mut(node_id)
                    .meta
                    .set_input_qparams_from(input_qparams);
                modified = true;
            }

            let snapshot_users = graph.live_user_ids(node_id);
            let mut output_qparams = Vec::new();
            for (idx, user_id) in snapshot_users.into_iter().enumerate() {
                let user = graph.node(user_id).clone();
                if !user.alive || !user.is_quantize_any() {
                    continue;
                }
                // A Q node may consume this node as scale or zero point; it is
                // an output quantizer only when this node is its data input.
                if user.arg_node(0) != Some(node_id) {
                    continue;
                }
                let Some(resolved) = resolve_qparam_from_q_or_dq(graph, &user) else {
                    continue;
                };
                output_qparams.push((idx, resolved.qparam.clone()));
                if preserve_qdq {
                    continue;
                }
                graph.replace_and_erase(user_id, node_id);
                modified = true;
            }

            if !output_qparams.is_empty() {
                graph
                    .node_mut(node_id)
                    .meta
                    .set_output_qparams_from(output_qparams.iter().cloned());
                modified = true;

                let is_dim_order_copy = matches!(
                    node.target.as_str(),
                    "dim_order_ops._to_dim_order_copy.default" | "_to_dim_order_copy.default"
                );
                if is_sum_dim_int_list_target(&node.target) || is_dim_order_copy {
                    if let Some((_, output_qparam)) =
                        output_qparams.iter().find(|(index, _)| *index == 0)
                    {
                        // Stock raises a ValueError here; let it.
                        if is_dim_order_copy
                            && (output_qparam.scale != 1.0 || output_qparam.zp != 0)
                        {
                            crate::stock_fallback::request(format!(
                                "quantized _to_dim_order_copy without unit qparams: {}",
                                node.name
                            ));
                            continue;
                        }
                        graph.set_kwarg(
                            node_id,
                            "dtype",
                            IrArg::Dtype(output_qparam.dtype_str.clone()),
                        );
                        modified = true;
                    }
                }
            }
        }

        modified
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::fx_ir::{IrNode, MetaValue};
    use crate::graph::ir::DType;
    use crate::test_ir::{call_default as qdq_node, output, placeholder_default as placeholder};

    use super::*;

    fn make_tensor(dtype: DType, shape: Vec<i64>, data: Vec<u8>) -> TensorData {
        TensorData { data, shape, dtype }
    }

    fn tensor_lookup<'a>(
        tensors: &'a HashMap<usize, TensorData>,
    ) -> impl FnMut(usize) -> Option<&'a TensorData> {
        move |id| tensors.get(&id)
    }

    fn add_node(
        graph: &mut IrGraph,
        op: NodeOp,
        name: &str,
        target: &str,
        args: Vec<IrArg>,
        meta: NodeMeta,
    ) -> usize {
        graph.add_node(op, name.to_string(), target.to_string(), args, meta)
    }

    fn annotated_meta() -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_extra("arm_quantized_annotation", MetaValue::Bool(true));
        meta
    }

    fn per_tensor_args(
        input: NodeId,
        scale: f64,
        zp: i64,
        qmin: i64,
        qmax: i64,
        dtype: &str,
    ) -> Vec<IrArg> {
        vec![
            IrArg::Node(input),
            IrArg::Float(scale),
            IrArg::Int(zp),
            IrArg::Int(qmin),
            IrArg::Int(qmax),
            IrArg::Dtype(dtype.to_string()),
        ]
    }

    fn int8_qdq_args(input: NodeId, scale: f64, zp: i64) -> Vec<IrArg> {
        per_tensor_args(input, scale, zp, -128, 127, "torch.int8")
    }

    fn int32_qdq_args(input: NodeId, scale: f64) -> Vec<IrArg> {
        per_tensor_args(
            input,
            scale,
            0,
            i32::MIN as i64,
            i32::MAX as i64,
            "torch.int32",
        )
    }

    #[test]
    fn foldable_node_requires_annotation_or_constant_target() {
        let annotated = IrNode {
            id: 0,
            op: NodeOp::CallFunction,
            name: "annotated".to_string(),
            target: "aten.add.Tensor".to_string(),
            args: Vec::new(),
            kwargs: HashMap::new(),
            meta: annotated_meta(),
            tensor: None,
            alive: true,
        };
        assert!(is_foldable_node(&annotated));

        let constant = IrNode {
            target: "aten.full.default".to_string(),
            ..annotated.clone()
        };
        assert!(is_foldable_node(&constant));

        let q_node = IrNode {
            target: "quantized_decomposed.quantize_per_tensor.default".to_string(),
            ..annotated.clone()
        };
        assert!(!is_foldable_node(&q_node));
    }

    #[test]
    fn resolves_per_tensor_qparam_from_literal_args() {
        let tensors = HashMap::new();
        let mut lookup = tensor_lookup(&tensors);
        let resolved = resolve_qparam_from_q_or_dq_args(
            "quantized_decomposed.dequantize_per_tensor.default",
            &[
                IrArg::Node(0),
                IrArg::Float(0.25),
                IrArg::Int(3),
                IrArg::Int(-128),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
            &mut lookup,
        )
        .expect("expected per-tensor qparam");

        assert_eq!(resolved.source, FoldQdqQParamSource::LiteralArgs);
        assert_eq!(resolved.qparam.scale, 0.25);
        assert_eq!(resolved.qparam.zp, 3);
        assert_eq!(resolved.qparam.qmin, -128);
        assert_eq!(resolved.qparam.qmax, 127);
        assert_eq!(resolved.qparam.dtype_str, "torch.int8");
        assert!(resolved.qparam.per_channel_scales.is_empty());
    }

    #[test]
    fn resolves_per_channel_qparam_from_tensor_backed_args() {
        let mut tensors = HashMap::new();
        tensors.insert(
            11,
            make_tensor(
                DType::Float32,
                vec![2],
                [0.5f32.to_le_bytes(), 0.25f32.to_le_bytes()].concat(),
            ),
        );
        tensors.insert(
            12,
            make_tensor(
                DType::Int32,
                vec![2],
                [2i32.to_le_bytes(), 4i32.to_le_bytes()].concat(),
            ),
        );

        let mut lookup = tensor_lookup(&tensors);
        let resolved = resolve_qparam_from_q_or_dq_args(
            "quantized_decomposed.dequantize_per_channel.default",
            &[
                IrArg::Node(0),
                IrArg::Node(11),
                IrArg::Node(12),
                IrArg::Int(1),
                IrArg::Int(-127),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
            &mut lookup,
        )
        .expect("expected per-channel qparam");

        assert_eq!(resolved.source, FoldQdqQParamSource::TensorBackedArgs);
        assert_eq!(resolved.qparam.axis, 1);
        assert_eq!(resolved.qparam.per_channel_scales, vec![0.5, 0.25]);
        assert_eq!(resolved.qparam.per_channel_zps, vec![2, 4]);
        assert_eq!(resolved.qparam.dtype_str, "torch.int8");
    }

    #[test]
    fn resolves_qparam_from_node_meta_with_output_preferred() {
        let mut meta = NodeMeta::default();
        meta.set_unary_input_output_qparams(
            QParam::per_tensor(0.125, 2, -128, 127, "torch.int8".to_string()),
            QParam::per_tensor(0.25, 4, -127, 127, "torch.int8".to_string()),
        );

        let resolved = resolve_qparam_from_meta(&meta).expect("expected node-meta qparam");
        assert_eq!(resolved.source, FoldQdqQParamSource::OutputNodeMeta);
        assert_eq!(resolved.qparam.scale, 0.25);
        assert_eq!(resolved.qparam.zp, 4);
    }

    #[test]
    fn fold_qdq_pass_bypasses_simple_dq_q_chain_and_annotates_qparams() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x");
        let dq = qdq_node(
            &mut graph,
            "dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            int8_qdq_args(input, 0.25, 3),
        );

        let add = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(dq), IrArg::Int(1)],
            annotated_meta(),
        );
        let q = qdq_node(
            &mut graph,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(add, 0.25, 3),
        );
        let _output = output(&mut graph, q);

        let pass = IrFoldQdqPass::new();
        assert!(pass.run(&mut graph));

        assert!(graph.node(dq).alive == false);
        assert!(graph.node(q).alive == false);
        assert!(matches!(graph.node(add).args[0], IrArg::Node(id) if id == input));
        assert_eq!(graph.node(add).meta.input_qparam().unwrap().scale, 0.25);
        assert_eq!(graph.node(add).meta.output_qparam().unwrap().zp, 3);
    }

    #[test]
    fn fold_qdq_pass_bypasses_dq_nodes_inside_list_arg() {
        let mut graph = IrGraph::empty();
        let input0 = placeholder(&mut graph, "x0");
        let input1 = placeholder(&mut graph, "x1");
        let dq0 = qdq_node(
            &mut graph,
            "dq0",
            "quantized_decomposed.dequantize_per_tensor.default",
            int8_qdq_args(input0, 0.25, 3),
        );
        let dq1 = qdq_node(
            &mut graph,
            "dq1",
            "quantized_decomposed.dequantize_per_tensor.default",
            int8_qdq_args(input1, 0.5, 7),
        );

        let cat = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "cat",
            "aten.cat.default",
            vec![
                IrArg::List(vec![IrArg::Node(dq0), IrArg::Node(dq1)]),
                IrArg::Int(0),
            ],
            annotated_meta(),
        );
        let q = qdq_node(
            &mut graph,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(cat, 0.5, 7),
        );
        let _output = output(&mut graph, q);

        let pass = IrFoldQdqPass::new();
        assert!(pass.run(&mut graph));

        assert!(!graph.node(dq0).alive);
        assert!(!graph.node(dq1).alive);
        assert!(matches!(
            graph.node(cat).args[0],
            IrArg::List(ref items)
                if matches!(items.as_slice(), [IrArg::Node(a), IrArg::Node(b)] if *a == input0 && *b == input1)
        ));
        assert_eq!(graph.node(cat).meta.input_qparam().unwrap().scale, 0.5);
        assert_eq!(graph.node(cat).meta.output_qparam().unwrap().zp, 7);
    }

    #[test]
    fn fold_qdq_pass_updates_sum_dtype_kwarg() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x");
        let sum = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "sum",
            "aten.sum.dim_IntList",
            vec![IrArg::Node(input)],
            annotated_meta(),
        );
        graph
            .node_mut(sum)
            .kwargs
            .insert("dtype".to_string(), IrArg::Dtype("torch.int32".to_string()));
        let q = qdq_node(
            &mut graph,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(sum, 0.5, 5),
        );
        let _output = output(&mut graph, q);

        let pass = IrFoldQdqPass::new();
        assert!(pass.run(&mut graph));
        assert!(matches!(
            graph.node(sum).kwargs.get("dtype"),
            Some(IrArg::Dtype(dtype)) if dtype == "torch.int8"
        ));
    }

    #[test]
    fn int32_domain_add_chain_lowers_to_rescale_add_rescale_without_float() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x");
        let dq0 = qdq_node(
            &mut graph,
            "dq0",
            "quantized_decomposed.dequantize_per_tensor.default",
            int8_qdq_args(input, 0.5, 3),
        );
        let q1 = qdq_node(
            &mut graph,
            "q1",
            "quantized_decomposed.quantize_per_tensor.default",
            int32_qdq_args(dq0, 0.25),
        );
        let dq1 = qdq_node(
            &mut graph,
            "dq1",
            "quantized_decomposed.dequantize_per_tensor.default",
            int32_qdq_args(q1, 0.25),
        );
        let add = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(dq1), IrArg::Node(dq1)],
            annotated_meta(),
        );
        let q2 = qdq_node(
            &mut graph,
            "q2",
            "quantized_decomposed.quantize_per_tensor.default",
            int32_qdq_args(add, 0.25),
        );
        let dq2 = qdq_node(
            &mut graph,
            "dq2",
            "quantized_decomposed.dequantize_per_tensor.default",
            int32_qdq_args(q2, 0.25),
        );
        let q3 = qdq_node(
            &mut graph,
            "q3",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(dq2, 0.125, 7),
        );
        let _output = output(&mut graph, q3);

        assert!(IrFoldQdqPass::new().run(&mut graph));
        assert!(crate::fx_insert_qdq_rescale_ir::IrInsertQdqRescalePass.run(&mut graph));
        assert!(!crate::fx_insert_int32_rescale_ir::IrInsertInt32RescalePass.run(&mut graph));
        assert!(!crate::fx_decompose_quant_ir::IrDecomposeQuantPass.run(&mut graph));

        assert!(graph.live_nodes().all(|node| !node.is_q_or_dq()));
        assert!(
            graph
                .live_nodes()
                .all(|node| node.meta.val_dtype() != Some("torch.float32"))
        );
        let rescales: Vec<_> = graph.live_nodes().filter(|node| node.is_rescale()).collect();
        assert_eq!(rescales.len(), 2);
        assert!(graph.node(add).alive);
        assert!(matches!(
            graph.node(add).args.as_slice(),
            [IrArg::Node(a), IrArg::Node(b)] if a == b
                && graph.node(*a).is_rescale()
        ));
        assert_eq!(
            graph.node(add).meta.input_qparam().unwrap().dtype_str,
            "torch.int32"
        );
        assert_eq!(
            graph.node(add).meta.output_qparam().unwrap().dtype_str,
            "torch.int32"
        );
    }

    fn partial_binary_qdq_graph(
        target: &str,
        extra_args: Vec<IrArg>,
    ) -> (IrGraph, NodeId, NodeId, NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x");
        let y = placeholder(&mut graph, "y");
        let x_q = qdq_node(
            &mut graph,
            "x_q",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(x, 0.5, 0),
        );
        let x_dq = qdq_node(
            &mut graph,
            "x_dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            int8_qdq_args(x_q, 0.5, 0),
        );
        let mut args = vec![IrArg::Node(x_dq), IrArg::Node(y)];
        args.extend(extra_args);
        let binary = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "binary",
            target,
            args,
            annotated_meta(),
        );
        let out_q = qdq_node(
            &mut graph,
            "out_q",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(binary, 0.5, 0),
        );
        let out_dq = qdq_node(
            &mut graph,
            "out_dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            int8_qdq_args(out_q, 0.5, 0),
        );
        let _output = output(&mut graph, out_dq);
        (graph, binary, x_dq, y, out_q)
    }

    #[test]
    fn preserves_qdq_for_partially_quantized_add_and_sub() {
        for target in ["aten.add.Tensor", "aten.sub.Tensor"] {
            let (mut graph, binary, x_dq, y, out_q) =
                partial_binary_qdq_graph(target, Vec::new());

            assert!(IrFoldQdqPass::new().run(&mut graph));

            let input_keys: Vec<usize> = graph
                .node(binary)
                .meta
                .input_qparams()
                .unwrap()
                .iter()
                .map(|(&index, _)| index)
                .collect();
            assert_eq!(input_keys, vec![0]);
            assert!(graph.node(binary).meta.output_qparam().is_some());
            assert!(graph.node(x_dq).alive);
            assert!(graph.node(out_q).alive);
            assert!(matches!(graph.node(binary).args[0], IrArg::Node(id) if id == x_dq));
            assert!(matches!(graph.node(binary).args[1], IrArg::Node(id) if id == y));
            assert!(graph.live_user_ids(binary).contains(&out_q));
        }
    }

    #[test]
    fn folds_mul_with_partial_inputs_by_default() {
        let (mut graph, binary, x_dq, _, out_q) =
            partial_binary_qdq_graph("aten.mul.Tensor", Vec::new());

        assert!(IrFoldQdqPass::new().run(&mut graph));

        assert!(!graph.node(x_dq).alive);
        assert!(!graph.node(out_q).alive);
        assert!(matches!(graph.node(binary).args[1], IrArg::Node(_)));
    }

    #[test]
    fn preserves_mixed_profile_partial_qdq_when_enabled() {
        let grid_extra = vec![IrArg::Int(0), IrArg::Int(0), IrArg::Bool(false)];
        let cases: &[(&str, Vec<IrArg>)] = &[
            ("aten.add.Tensor", Vec::new()),
            ("aten.sub.Tensor", Vec::new()),
            ("aten.mul.Tensor", Vec::new()),
            ("aten.div.Tensor", Vec::new()),
            ("aten.minimum.default", Vec::new()),
            ("aten.maximum.default", Vec::new()),
            ("aten.mm.default", Vec::new()),
            ("aten.bmm.default", Vec::new()),
            ("aten.eq.Tensor", Vec::new()),
            ("aten.ge.Tensor", Vec::new()),
            ("aten.gt.Tensor", Vec::new()),
            ("aten.le.Tensor", Vec::new()),
            ("aten.lt.Tensor", Vec::new()),
            ("aten.grid_sampler_2d.default", grid_extra),
        ];
        for (target, extra_args) in cases {
            let (mut graph, binary, x_dq, y, out_q) =
                partial_binary_qdq_graph(target, extra_args.clone());

            assert!(
                IrFoldQdqPass::with_mixed_profile_partial_binary_qdq().run(&mut graph),
                "target {target}"
            );

            let input_keys: Vec<usize> = graph
                .node(binary)
                .meta
                .input_qparams()
                .unwrap()
                .iter()
                .map(|(&index, _)| index)
                .collect();
            assert_eq!(input_keys, vec![0], "target {target}");
            assert!(graph.node(x_dq).alive, "target {target}");
            assert!(graph.node(out_q).alive, "target {target}");
            assert!(
                matches!(graph.node(binary).args[0], IrArg::Node(id) if id == x_dq),
                "target {target}"
            );
            assert!(
                matches!(graph.node(binary).args[1], IrArg::Node(id) if id == y),
                "target {target}"
            );
            assert!(graph.live_user_ids(binary).contains(&out_q), "target {target}");
        }
    }

    #[test]
    fn does_not_treat_index_as_binary_operand() {
        for target in ["aten.index_select.default", "aten.gather.default"] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x");
            let index = placeholder(&mut graph, "index");
            let node = add_node(
                &mut graph,
                NodeOp::CallFunction,
                "gatherish",
                target,
                vec![IrArg::Node(x), IrArg::Int(0), IrArg::Node(index)],
                annotated_meta(),
            );
            let snapshot = graph.node(node).clone();
            assert!(!has_partial_binary_tensor_qdq_inputs(&snapshot, &[0], false));
            assert!(!has_partial_binary_tensor_qdq_inputs(&snapshot, &[0], true));
        }
    }

    #[test]
    fn mixed_profile_allowlist_matches_upstream_targets() {
        use std::collections::HashSet;

        let probe = [
            "aten.add.Tensor",
            "aten.sub.Tensor",
            "aten.mul.Tensor",
            "aten.div.Tensor",
            "aten.div.Tensor_mode",
            "aten.minimum.default",
            "aten.maximum.default",
            "aten.mm.default",
            "aten.bmm.default",
            "aten.eq.Tensor",
            "aten.ge.Tensor",
            "aten.gt.Tensor",
            "aten.le.Tensor",
            "aten.lt.Tensor",
            "aten.grid_sampler_2d.default",
            "aten.index_select.default",
            "aten.gather.default",
            "aten.cat.default",
        ];
        let matched: HashSet<&str> = probe
            .iter()
            .copied()
            .filter(|target| is_mixed_profile_partial_binary_tensor_target(target))
            .collect();
        let expected: HashSet<&str> = [
            "aten.add.Tensor",
            "aten.sub.Tensor",
            "aten.mul.Tensor",
            "aten.div.Tensor",
            "aten.minimum.default",
            "aten.maximum.default",
            "aten.mm.default",
            "aten.bmm.default",
            "aten.eq.Tensor",
            "aten.ge.Tensor",
            "aten.gt.Tensor",
            "aten.le.Tensor",
            "aten.lt.Tensor",
            "aten.grid_sampler_2d.default",
        ]
        .into_iter()
        .collect();
        assert_eq!(matched, expected);
        assert!(is_default_partial_binary_tensor_target("aten.add.Tensor"));
        assert!(is_default_partial_binary_tensor_target("aten.sub.Tensor"));
        assert!(!is_default_partial_binary_tensor_target("aten.mul.Tensor"));
    }

    #[test]
    fn erases_shared_input_dq_once() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x");
        let dq = qdq_node(
            &mut graph,
            "dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            int8_qdq_args(input, 0.25, 3),
        );

        let add = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(dq), IrArg::Node(dq)],
            annotated_meta(),
        );
        let q = qdq_node(
            &mut graph,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(add, 0.25, 3),
        );
        let _output = output(&mut graph, q);

        assert!(IrFoldQdqPass::new().run(&mut graph));
        assert!(!graph.node(dq).alive);
        assert!(!graph.node(q).alive);
        assert!(matches!(
            graph.node(add).args.as_slice(),
            [IrArg::Node(a), IrArg::Node(b)] if *a == input && *b == input
        ));
    }

    #[test]
    fn does_not_fold_qparam_node_as_quantized_data() {
        use crate::fx_ir::{NodeTensor, TensorSource};
        use crate::store::TensorData;

        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x");
        let zp = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "zp",
            "aten.full.default",
            vec![
                IrArg::List(vec![IrArg::Int(1)]),
                IrArg::Int(0),
            ],
            NodeMeta::default(),
        );
        graph.node_mut(zp).tensor = Some(NodeTensor {
            source: TensorSource::GraphModuleAttr,
            data: TensorData {
                data: 0i32.to_le_bytes().to_vec(),
                shape: vec![1],
                dtype: DType::Int32,
            },
        });
        let q = qdq_node(
            &mut graph,
            "q",
            "quantized_decomposed.quantize_per_tensor.tensor",
            vec![
                IrArg::Node(x),
                IrArg::Float(0.5),
                IrArg::Node(zp),
                IrArg::Int(-127),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
        );
        let viewer = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "viewer",
            "aten.view_copy.default",
            vec![IrArg::Node(q)],
            NodeMeta::default(),
        );
        let _output = output(&mut graph, viewer);

        IrFoldQdqPass::new().run(&mut graph);

        assert!(graph.node(q).alive);
        assert_eq!(graph.node(viewer).first_node_arg(), Some(q));
    }

    #[test]
    fn non_unit_to_dim_order_copy_requests_stock_fallback() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x");
        let copy = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "copy",
            "dim_order_ops._to_dim_order_copy.default",
            vec![IrArg::Node(input)],
            annotated_meta(),
        );
        graph.set_kwarg(copy, "dtype", IrArg::Dtype("torch.int32".to_string()));
        let q = qdq_node(
            &mut graph,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(copy, 0.5, 0),
        );
        let _output = output(&mut graph, q);

        crate::stock_fallback::take();
        IrFoldQdqPass::new().run(&mut graph);
        assert_eq!(
            crate::stock_fallback::take().as_deref(),
            Some("quantized _to_dim_order_copy without unit qparams: copy")
        );
        assert!(matches!(
            graph.node(copy).kwargs.get("dtype"),
            Some(IrArg::Dtype(dtype)) if dtype == "torch.int32"
        ));
    }

    #[test]
    fn to_dim_order_copy_gets_output_dtype() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x");
        let copy = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "copy",
            "dim_order_ops._to_dim_order_copy.default",
            vec![IrArg::Node(input)],
            annotated_meta(),
        );
        graph.set_kwarg(
            copy,
            "dtype",
            IrArg::Dtype("torch.int32".to_string()),
        );
        let q = qdq_node(
            &mut graph,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            int8_qdq_args(copy, 1.0, 0),
        );
        let _output = output(&mut graph, q);

        assert!(IrFoldQdqPass::new().run(&mut graph));
        assert!(matches!(
            graph.node(copy).kwargs.get("dtype"),
            Some(IrArg::Dtype(dtype)) if dtype == "torch.int8"
        ));
    }

    #[test]
    fn fold_qdq_pass_folds_int32_quantized_add() {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x");
        let dq = qdq_node(
            &mut graph,
            "dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            int32_qdq_args(input, 0.25),
        );

        let add = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(dq), IrArg::Node(dq)],
            annotated_meta(),
        );
        let q = qdq_node(
            &mut graph,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            int32_qdq_args(add, 0.25),
        );
        let _output = output(&mut graph, q);

        let pass = IrFoldQdqPass::new();
        assert!(pass.run(&mut graph));
        assert!(!graph.node(dq).alive);
        assert!(!graph.node(q).alive);
        assert!(matches!(graph.node(add).args[0], IrArg::Node(id) if id == input));
        assert_eq!(
            graph.node(add).meta.input_qparam().unwrap().dtype_str,
            "torch.int32"
        );
        assert_eq!(
            graph.node(add).meta.output_qparam().unwrap().dtype_str,
            "torch.int32"
        );
    }
}
