//! First-pass scaffolding for a Rust-native fold_qdq port.
//!
//! This module is intentionally narrow:
//! - target-name matching helpers
//! - foldable-node classification from typed `NodeMeta`
//! - qparam resolution from literal or tensor-backed IR args
//!
//! It does not perform any graph mutation yet and is not wired into the
//! pipeline in this first pass.

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrNode, IrPass, MetaValue, NodeMeta, NodeOp, QParam};
use crate::graph::ir::DType;
use crate::store::TensorData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FoldQdqQParamSource {
    InputNodeMeta,
    OutputNodeMeta,
    LiteralArgs,
    TensorBackedArgs,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedFoldQdqQParam {
    pub qparam: QParam,
    pub source: FoldQdqQParamSource,
}

pub(crate) fn target_matches_any(target: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| target.contains(marker))
}

pub(crate) fn is_quantize_target(target: &str) -> bool {
    target_matches_any(target, &["quantize_per_tensor", "quantize_per_channel"])
}

pub(crate) fn is_dequantize_target(target: &str) -> bool {
    target_matches_any(target, &["dequantize_per_tensor", "dequantize_per_channel"])
}

pub(crate) fn is_q_or_dq_target(target: &str) -> bool {
    is_quantize_target(target) || is_dequantize_target(target)
}

pub(crate) fn is_constant_op_target(target: &str) -> bool {
    target_matches_any(
        target,
        &[
            "full_like.default",
            "full.default",
            "arange.start_step",
            "eye.default",
            "linspace.default",
            "scalar_tensor.default",
        ],
    )
}

pub(crate) fn has_arm_quantized_annotation(meta: &NodeMeta) -> bool {
    meta.extra_bool("arm_quantized_annotation") == Some(true)
}

pub(crate) fn is_foldable_node(node: &IrNode) -> bool {
    node.op == NodeOp::CallFunction
        && !is_q_or_dq_target(&node.target)
        && (is_constant_op_target(&node.target) || has_arm_quantized_annotation(&node.meta))
}

pub(crate) fn resolve_input_qparam_from_meta(meta: &NodeMeta) -> Option<ResolvedFoldQdqQParam> {
    Some(ResolvedFoldQdqQParam {
        qparam: meta.input_qparam()?.clone(),
        source: FoldQdqQParamSource::InputNodeMeta,
    })
}

pub(crate) fn resolve_output_qparam_from_meta(meta: &NodeMeta) -> Option<ResolvedFoldQdqQParam> {
    Some(ResolvedFoldQdqQParam {
        qparam: meta.output_qparam()?.clone(),
        source: FoldQdqQParamSource::OutputNodeMeta,
    })
}

pub(crate) fn resolve_qparam_from_meta(meta: &NodeMeta) -> Option<ResolvedFoldQdqQParam> {
    resolve_output_qparam_from_meta(meta).or_else(|| resolve_input_qparam_from_meta(meta))
}

pub(crate) fn resolve_qparam_from_node_meta(node: &IrNode) -> Option<ResolvedFoldQdqQParam> {
    resolve_qparam_from_meta(&node.meta)
}

pub(crate) fn resolve_qparam_from_q_or_dq(
    graph: &IrGraph,
    node: &IrNode,
) -> Option<ResolvedFoldQdqQParam> {
    let mut tensor_lookup =
        |id| graph.node(id).tensor.as_ref().map(|node_tensor| &node_tensor.data);
    resolve_qparam_from_q_or_dq_args(&node.target, &node.args, &mut tensor_lookup)
}

pub(crate) fn resolve_qparam_from_arg(
    graph: &IrGraph,
    arg: &IrArg,
) -> Option<ResolvedFoldQdqQParam> {
    let IrArg::Node(id) = arg else {
        return None;
    };
    let node = graph.node(*id);
    resolve_qparam_from_q_or_dq(graph, node).or_else(|| resolve_qparam_from_node_meta(node))
}

pub(crate) fn resolve_input_qparams_for_foldable_node(
    graph: &IrGraph,
    node: &IrNode,
) -> Vec<(usize, ResolvedFoldQdqQParam)> {
    node.args
        .iter()
        .enumerate()
        .filter_map(|(idx, arg)| resolve_qparam_from_arg(graph, arg).map(|qp| (idx, qp)))
        .collect()
}

pub(crate) fn resolve_qparam_from_q_or_dq_args<'a, F>(
    target: &str,
    args: &[IrArg],
    tensor_lookup: &mut F,
) -> Option<ResolvedFoldQdqQParam>
where
    F: FnMut(usize) -> Option<&'a TensorData>,
{
    if !is_q_or_dq_target(target) {
        return None;
    }

    if target.contains("per_channel") {
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

fn merge_qparam_sources(
    lhs: FoldQdqQParamSource,
    rhs: FoldQdqQParamSource,
) -> FoldQdqQParamSource {
    if lhs == FoldQdqQParamSource::TensorBackedArgs
        || rhs == FoldQdqQParamSource::TensorBackedArgs
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
        IrArg::Bool(value) => Some((vec![f64::from(*value as u8)], FoldQdqQParamSource::LiteralArgs)),
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
        IrArg::Node(id) => tensor_lookup(*id).map(|tensor| dtype_to_torch_str(tensor.dtype)),
        _ => None,
    }
}

fn dtype_to_torch_str(dtype: DType) -> String {
    match dtype {
        DType::Float32 => "torch.float32",
        DType::Float16 => "torch.float16",
        DType::BFloat16 => "torch.bfloat16",
        DType::Int8 => "torch.int8",
        DType::Int16 => "torch.int16",
        DType::Int32 => "torch.int32",
        DType::Int64 => "torch.int64",
        DType::Bool => "torch.bool",
        DType::Int4 => "torch.int8",
        DType::Int48 => "torch.int64",
    }
    .to_string()
}

pub struct IrFoldQdqPass;

impl IrFoldQdqPass {
    pub(crate) fn candidate_node_ids(&self, graph: &IrGraph) -> Vec<usize> {
        graph
            .nodes
            .iter()
            .filter(|node| node.alive && is_foldable_node(node))
            .map(|node| node.id)
            .collect()
    }

    pub(crate) fn resolve_input_qparams(
        &self,
        graph: &IrGraph,
        node: &IrNode,
    ) -> Vec<(usize, ResolvedFoldQdqQParam)> {
        resolve_input_qparams_for_foldable_node(graph, node)
    }

    pub(crate) fn resolve_output_qparam(
        &self,
        node: &IrNode,
    ) -> Option<ResolvedFoldQdqQParam> {
        resolve_qparam_from_node_meta(node)
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
            let mut input_qparams = HashMap::new();

            for (idx, resolved) in self.resolve_input_qparams(graph, &node) {
                input_qparams.insert(idx, resolved.qparam.clone());

                let IrArg::Node(arg_id) = node.args[idx] else {
                    continue;
                };
                let arg_node = graph.node(arg_id).clone();
                if !arg_node.alive || !is_dequantize_target(&arg_node.target) {
                    continue;
                }
                let Some(IrArg::Node(source_id)) = arg_node.args.first() else {
                    continue;
                };

                if matches!(resolved.source, FoldQdqQParamSource::TensorBackedArgs | FoldQdqQParamSource::LiteralArgs)
                    && resolved.qparam.qmin == -7
                    && resolved.qparam.qmax == 7
                {
                    graph.node_mut(*source_id).meta.set_extra(
                        "tosa_special_dtype",
                        MetaValue::Str("INT4".to_string()),
                    );
                }

                modified |= graph.replace_input_use(node_id, arg_id, *source_id);
                if !graph.has_live_users(arg_id) {
                    graph.erase(arg_id);
                    modified = true;
                }
            }

            if !input_qparams.is_empty() {
                graph.node_mut(node_id).meta.set_input_qparams(input_qparams);
                modified = true;
            }

            let snapshot_users = graph.live_user_ids(node_id);
            let mut output_qparams = HashMap::new();
            for (idx, user_id) in snapshot_users.into_iter().enumerate() {
                let user = graph.node(user_id).clone();
                if !user.alive || !is_quantize_target(&user.target) {
                    continue;
                }
                let Some(resolved) = resolve_qparam_from_q_or_dq(graph, &user) else {
                    continue;
                };
                output_qparams.insert(idx, resolved.qparam.clone());
                graph.replace_and_erase(user_id, node_id);
                modified = true;
            }

            if !output_qparams.is_empty() {
                graph.node_mut(node_id).meta.set_output_qparams(output_qparams.clone());
                modified = true;

                if node.target.contains("sum.dim_IntList") {
                    if let Some(output_qparam) = output_qparams.get(&0) {
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
    use crate::fx_ir::{IrNode, MetaValue};

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

    #[test]
    fn foldable_node_requires_annotation_or_constant_target() {
        let mut meta = NodeMeta::default();
        meta.set_extra("arm_quantized_annotation", MetaValue::Bool(true));

        let annotated = IrNode {
            id: 0,
            op: NodeOp::CallFunction,
            name: "annotated".to_string(),
            target: "aten.add.Tensor".to_string(),
            args: Vec::new(),
            kwargs: HashMap::new(),
            meta,
            tensor: None,
            alive: true,
            cf_kind: None,
            args_dirty: false,
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
        meta.set_single_input_qparam(QParam::per_tensor(
            0.125,
            2,
            -128,
            127,
            "torch.int8".to_string(),
        ));
        meta.set_single_output_qparam(QParam::per_tensor(
            0.25,
            4,
            -127,
            127,
            "torch.int8".to_string(),
        ));

        let resolved = resolve_qparam_from_meta(&meta).expect("expected node-meta qparam");
        assert_eq!(resolved.source, FoldQdqQParamSource::OutputNodeMeta);
        assert_eq!(resolved.qparam.scale, 0.25);
        assert_eq!(resolved.qparam.zp, 4);
    }

    #[test]
    fn fold_qdq_pass_bypasses_simple_dq_q_chain_and_annotates_qparams() {
        let mut graph = IrGraph::empty();
        let input = add_node(
            &mut graph,
            NodeOp::Placeholder,
            "x",
            "x",
            vec![],
            NodeMeta::default(),
        );
        let dq = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "dq",
            "quantized_decomposed.dequantize_per_tensor.default",
            vec![
                IrArg::Node(input),
                IrArg::Float(0.25),
                IrArg::Int(3),
                IrArg::Int(-128),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
            NodeMeta::default(),
        );

        let mut add_meta = NodeMeta::default();
        add_meta.set_extra("arm_quantized_annotation", MetaValue::Bool(true));
        let add = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(dq), IrArg::Int(1)],
            add_meta,
        );
        let q = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            vec![
                IrArg::Node(add),
                IrArg::Float(0.25),
                IrArg::Int(3),
                IrArg::Int(-128),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
            NodeMeta::default(),
        );
        let _output = add_node(
            &mut graph,
            NodeOp::Output,
            "output",
            "output",
            vec![IrArg::Node(q)],
            NodeMeta::default(),
        );

        let pass = IrFoldQdqPass;
        assert!(pass.run(&mut graph));

        assert!(graph.node(dq).alive == false);
        assert!(graph.node(q).alive == false);
        assert!(matches!(graph.node(add).args[0], IrArg::Node(id) if id == input));
        assert_eq!(graph.node(add).meta.input_qparam().unwrap().scale, 0.25);
        assert_eq!(graph.node(add).meta.output_qparam().unwrap().zp, 3);
    }

    #[test]
    fn fold_qdq_pass_updates_sum_dtype_kwarg() {
        let mut graph = IrGraph::empty();
        let input = add_node(
            &mut graph,
            NodeOp::Placeholder,
            "x",
            "x",
            vec![],
            NodeMeta::default(),
        );
        let mut sum_meta = NodeMeta::default();
        sum_meta.set_extra("arm_quantized_annotation", MetaValue::Bool(true));
        let sum = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "sum",
            "aten.sum.dim_IntList",
            vec![IrArg::Node(input)],
            sum_meta,
        );
        graph.node_mut(sum)
            .kwargs
            .insert("dtype".to_string(), IrArg::Dtype("torch.int32".to_string()));
        let q = add_node(
            &mut graph,
            NodeOp::CallFunction,
            "q",
            "quantized_decomposed.quantize_per_tensor.default",
            vec![
                IrArg::Node(sum),
                IrArg::Float(0.5),
                IrArg::Int(5),
                IrArg::Int(-128),
                IrArg::Int(127),
                IrArg::Dtype("torch.int8".to_string()),
            ],
            NodeMeta::default(),
        );
        let _output = add_node(
            &mut graph,
            NodeOp::Output,
            "output",
            "output",
            vec![IrArg::Node(q)],
            NodeMeta::default(),
        );

        let pass = IrFoldQdqPass;
        assert!(pass.run(&mut graph));
        assert!(matches!(
            graph.node(sum).kwargs.get("dtype"),
            Some(IrArg::Dtype(dtype)) if dtype == "torch.int8"
        ));
    }
}
