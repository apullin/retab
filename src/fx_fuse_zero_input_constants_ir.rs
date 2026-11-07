use std::collections::HashSet;

use crate::constant_materialization::{
    quantize_scalar_with_qparam, quantize_value_stock, repeated_scalar_bytes,
    scalar_storage_bytes,
};
use crate::fx_constant_tensor_eval::{node_has_tensor_inputs, tensor_numel};
use crate::fx_ir::{
    EffectfulIrPass, EffectfulIrPassResult, IrArg, IrGraph, IrNode, NodeOp, QParam,
};
use crate::fx_target::{
    arange_target_kind, is_add_sub_target, zero_input_constant_target, ArangeTargetKind,
    ZeroInputConstantTarget,
};
use crate::graph::ir::DType;
use crate::lowering_effects::{LoweringEffect, LoweringEffectBatch, MaterializeTensorNodeEffect};
use crate::store::TensorData;

pub struct IrFuseZeroInputConstantsEffectPass;

/// Which qparams quantize the fill value of a `full` the planners evaluate.
/// The other factories already follow rc4: only their own output qparams
/// quantize them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FullQParams {
    /// The effect passes' rule: the node's output qparam, else its input
    /// qparam, else one its users imply ([`infer_binary_full_qparam`]).
    Planner,
    /// rc4's ComputeConstantOpsAOTPass (`fuse_constant_ops_pass.py:327-335`),
    /// which IrMaterializeConstantsPass applies. rc4
    /// quantizes a `full` only with its own output qparams; without them it
    /// evaluates `full(*args, **kwargs)`, so the fill value stays as is, in
    /// the dtype kwarg's dtype, input qparams or not. rc4's passes copy their
    /// node's meta onto the `full` nodes they create (`ArmPass.call_scalar`,
    /// `arm_pass.py:177-205`), so a `full` that carries qparam maps (even
    /// empty ones, as rc4's `create_node` leaves them) gets that rule. A bare
    /// `full`, with no maps, keeps the planners' inference: RETAB's own
    /// decompositions create their `full` nodes bare and rely on it
    /// (IrDecomposeAvgPoolPad's padding, for one).
    Rc4,
}

fn parse_dtype_arg(arg: &IrArg) -> Option<DType> {
    match arg {
        IrArg::Dtype(name) | IrArg::Str(name) => Some(DType::from_torch_str(name)),
        _ => None,
    }
}

fn full_fill_value(node: &IrNode) -> Option<f64> {
    node.args.get(1)?.as_scalar_literal_f64()
}

fn full_shape(node: &IrNode) -> Option<Vec<i64>> {
    node.args
        .first()
        .and_then(IrArg::as_int_list)
        .or_else(|| node.meta.val_shape().map(|shape| shape.to_vec()))
}

fn int_qparam(node: &IrNode) -> Option<QParam> {
    node.meta
        .output_qparam()
        .or_else(|| node.meta.input_qparam())
        .cloned()
        .filter(QParam::is_integer_dtype)
}

fn effective_int_qparam(graph: &IrGraph, start: usize) -> Option<QParam> {
    let mut current = start;
    let mut seen = HashSet::new();
    while seen.insert(current) {
        let node = graph.node(current);
        if !node.alive {
            return None;
        }
        if let Some(qp) = int_qparam(node) {
            return Some(qp);
        }
        current = match node.args.first() {
            Some(IrArg::Node(next)) => *next,
            _ => return None,
        };
    }
    None
}

fn infer_binary_full_qparam(graph: &IrGraph, full_id: usize) -> Option<QParam> {
    for user_id in graph.live_user_ids(full_id) {
        let user = graph.node(user_id);
        if let Some(qp) = effective_int_qparam(graph, user_id) {
            return Some(qp);
        }
        if user.op != NodeOp::CallFunction || !is_add_sub_target(&user.target) {
            continue;
        }
        for arg in user.args.iter().take(2) {
            let IrArg::Node(other_id) = arg else {
                continue;
            };
            if *other_id == full_id {
                continue;
            }
            if let Some(qp) = effective_int_qparam(graph, *other_id) {
                return Some(qp);
            }
        }
    }
    None
}

fn is_quant_decompose_helper_full(node: &IrNode) -> bool {
    zero_input_constant_target(&node.target) == Some(ZeroInputConstantTarget::Full)
        && (node.name.ends_with("_sc") || node.name.ends_with("_zp"))
        && (node.name.contains("quantize_per_tensor")
            || node.name.contains("dequantize_per_tensor"))
}

/// The qparam a `full`'s fill value is quantized with under `rule`, if any.
fn full_qparam(graph: &IrGraph, node: &IrNode, rule: FullQParams) -> Option<QParam> {
    if is_quant_decompose_helper_full(node) {
        return None;
    }
    if rule == FullQParams::Rc4 && node.meta.has_qparam_metadata() {
        return node
            .meta
            .output_qparam()
            .cloned()
            .filter(QParam::is_integer_dtype);
    }
    int_qparam(node).or_else(|| infer_binary_full_qparam(graph, node.id))
}

fn explicit_zero_input_dtype(node: &IrNode) -> Option<DType> {
    node.kwargs
        .get("dtype")
        .and_then(parse_dtype_arg)
        .or_else(|| node.meta.val_dtype_kind())
}

fn quantized_scalar_bytes(qp: &QParam, value: f64) -> Option<(DType, Vec<u8>)> {
    if qp.is_per_channel() {
        return None;
    }
    let dtype = qp.dtype();
    let quantized =
        quantize_value_stock(value, qp.scale, qp.zp, qp.qmin, qp.qmax) as f64;
    Some((dtype, scalar_storage_bytes(dtype, quantized)?))
}

fn output_int_qparam_and_dtype(node: &IrNode) -> Option<(Option<QParam>, DType)> {
    let qparam = node
        .meta
        .output_qparam()
        .cloned()
        .filter(QParam::is_integer_dtype);
    let dtype = qparam
        .as_ref()
        .map(QParam::dtype)
        .or_else(|| explicit_zero_input_dtype(node))?;
    Some((qparam, dtype))
}

fn scalar_bytes_in_domain(dtype: DType, qparam: Option<&QParam>, value: f64) -> Option<Vec<u8>> {
    if let Some(qp) = qparam {
        let (qp_dtype, scalar) = quantized_scalar_bytes(qp, value)?;
        return (qp_dtype == dtype).then_some(scalar);
    }
    scalar_storage_bytes(dtype, value)
}

fn scalar_sequence_bytes<I>(dtype: DType, qparam: Option<&QParam>, values: I) -> Option<Vec<u8>>
where
    I: IntoIterator<Item = f64>,
{
    values.into_iter().try_fold(Vec::new(), |mut data, value| {
        data.extend_from_slice(&scalar_bytes_in_domain(dtype, qparam, value)?);
        Some(data)
    })
}

fn arange_spec(node: &IrNode) -> Option<(f64, f64, f64)> {
    match arange_target_kind(&node.target)? {
        ArangeTargetKind::StartStep => {
            let start = node.args.first()?.as_scalar_literal_f64()?;
            let end = node.args.get(1)?.as_scalar_literal_f64()?;
            let step = node
                .args
                .get(2)
                .and_then(IrArg::as_scalar_literal_f64)
                .unwrap_or(1.0);
            Some((start, end, step))
        }
        ArangeTargetKind::Start => {
            if node.args.len() >= 2 {
                return Some((
                    node.args.first()?.as_scalar_literal_f64()?,
                    node.args.get(1)?.as_scalar_literal_f64()?,
                    1.0,
                ));
            }
            Some((0.0, node.args.first()?.as_scalar_literal_f64()?, 1.0))
        }
    }
}

fn arange_output_shape(node: &IrNode, start: f64, end: f64, step: f64) -> Option<Vec<i64>> {
    if let Some(shape) = node.meta.val_shape() {
        return Some(shape.to_vec());
    }
    if step == 0.0 {
        return None;
    }
    let raw_len = ((end - start) / step).ceil();
    let len = raw_len.max(0.0) as i64;
    Some(vec![len])
}

fn plan_zero_input_arange(node: &IrNode) -> Option<TensorData> {
    let (start, end, step) = arange_spec(node)?;
    let shape = arange_output_shape(node, start, end, step)?;
    let numel = tensor_numel(&shape)?;

    let (qparam, dtype) = output_int_qparam_and_dtype(node)?;
    let data = scalar_sequence_bytes(
        dtype,
        qparam.as_ref(),
        (0..numel).map(|idx| start + (idx as f64 * step)),
    )?;

    Some(TensorData { data, shape, dtype })
}

fn plan_zero_input_linspace(node: &IrNode) -> Option<TensorData> {
    let start = node.args.first()?.as_scalar_literal_f64()?;
    let end = node.args.get(1)?.as_scalar_literal_f64()?;
    let steps = match node.args.get(2)? {
        IrArg::Int(value) => usize::try_from(*value).ok()?,
        _ => return None,
    };

    let shape = node
        .meta
        .val_shape()
        .map(|shape| shape.to_vec())
        .unwrap_or_else(|| vec![steps as i64]);

    let (qparam, dtype) = output_int_qparam_and_dtype(node)?;
    let data = scalar_sequence_bytes(
        dtype,
        qparam.as_ref(),
        (0..steps).map(|idx| {
            if steps <= 1 {
                start
            } else {
                start + (idx as f64 * (end - start) / (steps - 1) as f64)
            }
        }),
    )?;

    Some(TensorData { data, shape, dtype })
}

fn full_bail_reason(graph: &IrGraph, node: &IrNode, full_qparams: FullQParams) -> &'static str {
    if full_shape(node).is_none() {
        return "no_shape";
    }
    let Some(fill_value) = full_fill_value(node) else {
        return "no_fill_value";
    };
    if let Some(qp) = full_qparam(graph, node, full_qparams) {
        if qp.is_per_channel() {
            return "qparam_per_channel";
        }
        if quantized_scalar_bytes(&qp, fill_value).is_none() {
            return "quantized_scalar_none";
        }
    } else {
        let Some(dtype) = explicit_zero_input_dtype(node) else {
            return "no_explicit_dtype";
        };
        if scalar_storage_bytes(dtype, fill_value).is_none() {
            return "scalar_fill_none";
        }
    }
    "ok_but_planned_none"
}

fn plan_zero_input_eye(node: &IrNode) -> Option<TensorData> {
    let n = match node.args.first()? {
        IrArg::Int(value) => usize::try_from(*value).ok()?,
        _ => return None,
    };
    // aten.eye.m takes a second dim; aten.eye.default is square.
    let m = match node.args.get(1) {
        Some(IrArg::Int(value)) => usize::try_from(*value).ok()?,
        _ => n,
    };
    let shape = node
        .meta
        .val_shape()
        .map(|shape| shape.to_vec())
        .unwrap_or_else(|| vec![n as i64, m as i64]);

    let (qparam, dtype) = output_int_qparam_and_dtype(node)?;
    let data = scalar_sequence_bytes(
        dtype,
        qparam.as_ref(),
        (0..n).flat_map(|row| (0..m).map(move |col| if row == col { 1.0 } else { 0.0 })),
    )?;

    Some(TensorData { data, shape, dtype })
}

fn plan_zero_input_full(
    graph: &IrGraph,
    node: &IrNode,
    full_qparams: FullQParams,
) -> Option<TensorData> {
    let shape = full_shape(node)?;
    let fill_value = full_fill_value(node)?;

    let (dtype, scalar) = if let Some(qp) = full_qparam(graph, node, full_qparams) {
        quantized_scalar_bytes(&qp, fill_value)?
    } else {
        let dtype = explicit_zero_input_dtype(node)?;
        (dtype, scalar_storage_bytes(dtype, fill_value)?)
    };

    Some(TensorData {
        data: repeated_scalar_bytes(&shape, &scalar)?,
        shape,
        dtype,
    })
}

fn plan_zero_input_scalar_tensor(node: &IrNode) -> Option<TensorData> {
    let value = node.args.first()?.as_scalar_literal_f64()?;
    let shape = node
        .meta
        .val_shape()
        .map(|shape| shape.to_vec())
        .unwrap_or_default();

    // Like stock's ComputeConstantOpsAOTPass: compute the scalar in float and
    // quantize it with the node's output qparams when they are present.
    if let Some(qp) = node
        .meta
        .output_qparam()
        .filter(|qp| !qp.is_per_channel() && qp.is_integer_dtype())
    {
        let dtype = qp.dtype();
        let quantized = quantize_scalar_with_qparam(
            value,
            crate::graph::ir::QuantArgs {
                scale: qp.scale,
                zero_point: qp.zp,
                dtype,
                qmin: qp.qmin,
                qmax: qp.qmax,
            },
        );
        return Some(TensorData {
            data: scalar_storage_bytes(dtype, quantized)?,
            shape,
            dtype,
        });
    }

    let dtype = explicit_zero_input_dtype(node)?;
    Some(TensorData {
        data: scalar_storage_bytes(dtype, value)?,
        shape,
        dtype,
    })
}

/// Planned bytes for a zero-input factory node (full/arange/linspace/eye/
/// scalar_tensor), shared by the materializing sweep and by the constant
/// propagation in the other constant-folding passes.
pub(crate) fn zero_input_tensor(graph: &IrGraph, node: &IrNode) -> Option<TensorData> {
    zero_input_tensor_with(graph, node, FullQParams::Planner)
}

/// [`zero_input_tensor`] with `full_qparams` deciding how a `full` is
/// quantized.
pub(crate) fn zero_input_tensor_with(
    graph: &IrGraph,
    node: &IrNode,
    full_qparams: FullQParams,
) -> Option<TensorData> {
    match planned_zero_input_constant_target(node)? {
        ZeroInputConstantTarget::Full => plan_zero_input_full(graph, node, full_qparams),
        ZeroInputConstantTarget::Arange => plan_zero_input_arange(node),
        ZeroInputConstantTarget::Linspace => plan_zero_input_linspace(node),
        ZeroInputConstantTarget::Eye => plan_zero_input_eye(node),
        ZeroInputConstantTarget::ScalarTensor => plan_zero_input_scalar_tensor(node),
    }
}

fn planned_zero_input_constant_target(node: &IrNode) -> Option<ZeroInputConstantTarget> {
    if !node.alive || node.op != NodeOp::CallFunction || node_has_tensor_inputs(node) {
        return None;
    }
    zero_input_constant_target(&node.target)
}

impl EffectfulIrPass for IrFuseZeroInputConstantsEffectPass {
    fn name(&self) -> &str {
        "IrFuseZeroInputConstantsEp"
    }

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult {
        zero_input_constant_effects(graph, FullQParams::Planner)
    }
}

/// IrFuseZeroInputConstantsEffectPass's plan, with `full_qparams` deciding
/// how a `full` is quantized: a MaterializeTensorNode effect for every
/// zero-input factory the planners can evaluate.
pub(crate) fn zero_input_constant_effects(
    graph: &IrGraph,
    full_qparams: FullQParams,
) -> EffectfulIrPassResult {
    let mut effects = Vec::new();
    for node in &graph.nodes {
        let Some(target) = planned_zero_input_constant_target(node) else {
            continue;
        };
        if let Some(tensor) = zero_input_tensor_with(graph, node, full_qparams) {
            effects.push(LoweringEffect::MaterializeTensorNode(
                MaterializeTensorNodeEffect {
                    node_name: node.name.clone(),
                    data: tensor.data,
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                    allow_missing: true,
                },
            ));
            continue;
        }
        let why = if target == ZeroInputConstantTarget::Full {
            full_bail_reason(graph, node, full_qparams)
        } else {
            "n/a"
        };
        crate::fold_fallback_log(&format!(
            "ZeroInputConstantPrecomputeSkipped target={} kind={:?} why={}",
            node.target, target, why
        ));
    }
    let modified = !effects.is_empty();
    EffectfulIrPassResult {
        modified,
        effects: LoweringEffectBatch { items: effects },
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::IrGraph;
    use crate::test_ir::{call_default as add_call, placeholder_default as placeholder};

    use super::*;

    fn add_f32_call(graph: &mut IrGraph, name: &str, target: &str, args: Vec<IrArg>) -> usize {
        let id = add_call(graph, name, target, args);
        graph.set_kwarg(id, "dtype", IrArg::Dtype("torch.float32".to_string()));
        id
    }

    fn placeholder_with_qparam(graph: &mut IrGraph, name: &str, qparam: QParam) -> usize {
        let id = placeholder(graph, name);
        graph.node_mut(id).meta.set_single_output_qparam(qparam);
        id
    }

    fn int8_qparam(scale: f64) -> QParam {
        QParam::per_tensor(scale, 0, -128, 127, "torch.int8".to_string())
    }

    fn run_pass(graph: &mut IrGraph) -> EffectfulIrPassResult {
        IrFuseZeroInputConstantsEffectPass.run(graph)
    }

    #[test]
    fn plans_zero_input_factory_constants() {
        let mut graph = IrGraph::empty();
        add_f32_call(
            &mut graph,
            "full",
            "aten.full.default",
            vec![IrArg::List(vec![IrArg::Int(2)]), IrArg::Float(3.0)],
        );
        add_call(
            &mut graph,
            "arange",
            "aten.arange.start",
            vec![IrArg::Int(7)],
        );
        add_call(
            &mut graph,
            "linspace",
            "aten.linspace.default",
            vec![IrArg::Float(0.0), IrArg::Float(1.0), IrArg::Int(5)],
        );
        add_call(&mut graph, "eye", "aten.eye.default", vec![IrArg::Int(4)]);

        let result = run_pass(&mut graph);

        assert!(result.modified);
        assert_eq!(result.effects.items.len(), 1);
        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "full"
                    && effect.dtype == DType::Float32
                    && effect.shape == vec![2]
                    && effect.data == [3.0f32.to_le_bytes(), 3.0f32.to_le_bytes()].concat()
        ));
    }

    #[test]
    fn materializes_quantized_full_from_binary_user_domain() {
        let mut graph = IrGraph::empty();
        let full = add_call(
            &mut graph,
            "full",
            "aten.full.default",
            vec![IrArg::List(vec![IrArg::Int(2)]), IrArg::Float(3.0)],
        );
        let other = placeholder_with_qparam(&mut graph, "other", int8_qparam(0.25));
        add_call(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(full), IrArg::Node(other)],
        );

        let result = run_pass(&mut graph);

        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "full"
                    && effect.dtype == DType::Int8
                    && effect.shape == vec![2]
                    && effect.data == vec![12u8, 12u8]
        ));
    }

    #[test]
    fn keeps_quant_decompose_helper_full_in_explicit_float_dtype() {
        let mut graph = IrGraph::empty();
        let scale = add_f32_call(
            &mut graph,
            "quantize_per_tensor_3_sc",
            "aten.full.default",
            vec![
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(1), IrArg::Int(1)]),
                IrArg::Float(4.0),
            ],
        );
        let other = placeholder_with_qparam(
            &mut graph,
            "other",
            QParam::per_tensor(
                0.25,
                0,
                i32::MIN as i64,
                i32::MAX as i64,
                "torch.int32".to_string(),
            ),
        );
        add_call(
            &mut graph,
            "quantize_per_tensor_3_mul",
            "aten.mul.Tensor",
            vec![IrArg::Node(other), IrArg::Node(scale)],
        );

        let result = run_pass(&mut graph);

        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "quantize_per_tensor_3_sc"
                    && effect.dtype == DType::Float32
                    && effect.shape == vec![1, 1, 1]
                    && effect.data == 4.0f32.to_le_bytes().to_vec()
        ));
    }

    #[test]
    fn materializes_arange_start_step_from_rust() {
        let mut graph = IrGraph::empty();
        add_f32_call(
            &mut graph,
            "arange",
            "aten.arange.start_step",
            vec![IrArg::Float(0.0), IrArg::Float(1.0), IrArg::Float(0.25)],
        );

        let result = run_pass(&mut graph);

        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "arange"
                    && effect.dtype == DType::Float32
                    && effect.shape == vec![4]
                    && effect.data
                        == [
                            0.0f32.to_le_bytes(),
                            0.25f32.to_le_bytes(),
                            0.5f32.to_le_bytes(),
                            0.75f32.to_le_bytes(),
                        ]
                        .concat()
        ));
    }

    #[test]
    fn materializes_quantized_arange_from_output_qparams() {
        let mut graph = IrGraph::empty();
        let arange = add_call(
            &mut graph,
            "arange",
            "aten.arange.start_step",
            vec![IrArg::Int(0), IrArg::Int(4), IrArg::Int(1)],
        );
        graph
            .node_mut(arange)
            .meta
            .set_single_output_qparam(int8_qparam(0.5));

        let result = run_pass(&mut graph);

        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "arange"
                    && effect.dtype == DType::Int8
                    && effect.shape == vec![4]
                    && effect.data == vec![0u8, 2u8, 4u8, 6u8]
        ));
    }

    #[test]
    fn materializes_linspace_from_rust() {
        let mut graph = IrGraph::empty();
        add_f32_call(
            &mut graph,
            "linspace",
            "aten.linspace.default",
            vec![IrArg::Float(0.0), IrArg::Float(1.0), IrArg::Int(5)],
        );

        let result = run_pass(&mut graph);

        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "linspace"
                    && effect.dtype == DType::Float32
                    && effect.shape == vec![5]
                    && effect.data
                        == [
                            0.0f32.to_le_bytes(),
                            0.25f32.to_le_bytes(),
                            0.5f32.to_le_bytes(),
                            0.75f32.to_le_bytes(),
                            1.0f32.to_le_bytes(),
                        ]
                        .concat()
        ));
    }

    #[test]
    fn materializes_identity_eye_from_rust() {
        let mut graph = IrGraph::empty();
        add_f32_call(&mut graph, "eye", "aten.eye.default", vec![IrArg::Int(3)]);

        let result = run_pass(&mut graph);

        let effect = match &result.effects.items[0] {
            LoweringEffect::MaterializeTensorNode(effect) => effect,
            other => panic!("unexpected effect kind: {other:?}"),
        };
        assert_eq!(effect.node_name, "eye");
        assert_eq!(effect.shape, vec![3, 3]);
        let values = effect
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn materializes_scalar_tensor_from_rust() {
        let mut graph = IrGraph::empty();
        let id = add_call(
            &mut graph,
            "scalar",
            "aten.scalar_tensor.default",
            vec![IrArg::Int(9)],
        );
        graph
            .node_mut(id)
            .meta
            .set_tensor_facts(crate::fx_meta::TensorFacts::new(vec![], "torch.int32"));

        let result = run_pass(&mut graph);

        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "scalar"
                    && effect.dtype == DType::Int32
                    && effect.shape.is_empty()
                    && effect.data == 9i32.to_le_bytes().to_vec()
        ));
    }

    #[test]
    fn materializes_quantized_scalar_tensor_from_output_qparams() {
        let mut graph = IrGraph::empty();
        let id = add_call(
            &mut graph,
            "scalar",
            "aten.scalar_tensor.default",
            vec![IrArg::Float(2.0)],
        );
        graph
            .node_mut(id)
            .meta
            .set_tensor_facts(crate::fx_meta::TensorFacts::new(vec![], "torch.float32"));
        graph
            .node_mut(id)
            .meta
            .set_single_output_qparam(int8_qparam(0.5));

        let result = run_pass(&mut graph);

        // Computed in float (2.0) then quantized at scale 0.5, like stock's
        // ComputeConstantOpsAOTPass.
        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "scalar"
                    && effect.dtype == DType::Int8
                    && effect.shape.is_empty()
                    && effect.data == vec![4u8]
        ));
    }

    #[test]
    fn materializes_quantized_linspace_from_output_qparams() {
        let mut graph = IrGraph::empty();
        let linspace = add_call(
            &mut graph,
            "linspace",
            "aten.linspace.default",
            vec![IrArg::Float(0.0), IrArg::Float(1.0), IrArg::Int(5)],
        );
        graph
            .node_mut(linspace)
            .meta
            .set_single_output_qparam(int8_qparam(0.25));

        let result = run_pass(&mut graph);

        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "linspace"
                    && effect.dtype == DType::Int8
                    && effect.shape == vec![5]
                    && effect.data == vec![0u8, 1u8, 2u8, 3u8, 4u8]
        ));
    }

    #[test]
    fn skips_nodes_with_tensor_inputs() {
        let mut graph = IrGraph::empty();
        let placeholder = placeholder(&mut graph, "x");
        add_call(
            &mut graph,
            "not_zero_input",
            "aten.full.default",
            vec![IrArg::Node(placeholder), IrArg::Int(3)],
        );

        let result = run_pass(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
    }
}
