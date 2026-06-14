use std::collections::HashSet;

use crate::constant_materialization::{repeated_scalar_bytes, scalar_storage_bytes};
use crate::fx_constant_tensor_eval::{node_has_tensor_inputs, tensor_numel};
use crate::fx_fuse_constant_args::ZeroInputConstantKind;
use crate::fx_ir::{
    EffectfulIrPass, EffectfulIrPassResult, IrArg, IrGraph, IrNode, NodeOp, QParam,
};
use crate::graph::ir::DType;
use crate::lowering_effects::{
    LoweringEffect, LoweringEffectBatch, MaterializeTensorNodeEffect,
    MaterializeZeroInputConstantEffect,
};
use crate::tensor_data::parse_torch_dtype;

pub struct IrFuseZeroInputConstantsEffectPass;

fn parse_dtype_arg(arg: &IrArg) -> Option<DType> {
    match arg {
        IrArg::Dtype(name) | IrArg::Str(name) => Some(parse_torch_dtype(name)),
        _ => None,
    }
}

fn full_fill_value(node: &IrNode) -> Option<f64> {
    node.args.get(1)?.as_scalar_literal_f64()
}

fn int_qparam(node: &IrNode) -> Option<QParam> {
    node.meta
        .output_qparam()
        .or_else(|| node.meta.input_qparam())
        .cloned()
        .filter(|qp| qp.dtype_str.contains("int"))
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
    for &user_id in graph.users(full_id) {
        let user = graph.node(user_id);
        if !user.alive {
            continue;
        }
        if let Some(qp) = effective_int_qparam(graph, user_id) {
            return Some(qp);
        }
        if user.op != NodeOp::CallFunction
            || (!user.target.contains("aten.add.Tensor")
                && !user.target.contains("aten.sub.Tensor"))
        {
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
    node.target.contains("aten.full.default")
        && (node.name.ends_with("_sc") || node.name.ends_with("_zp"))
        && (node.name.contains("quantize_per_tensor")
            || node.name.contains("dequantize_per_tensor"))
}

fn explicit_zero_input_dtype(node: &IrNode) -> Option<DType> {
    node.kwargs
        .get("dtype")
        .and_then(parse_dtype_arg)
        .or_else(|| node.meta.val_dtype().map(parse_torch_dtype))
}

fn quantized_scalar_bytes(qp: &QParam, value: f64) -> Option<(DType, Vec<u8>)> {
    if qp.is_per_channel() {
        return None;
    }
    let dtype = parse_torch_dtype(&qp.dtype_str);
    let quantized = (value / qp.scale + qp.zp as f64)
        .round()
        .clamp(qp.qmin as f64, qp.qmax as f64);
    Some((dtype, scalar_storage_bytes(dtype, quantized)?))
}

fn output_int_qparam_and_dtype(node: &IrNode) -> Option<(Option<QParam>, DType)> {
    let qparam = node
        .meta
        .output_qparam()
        .cloned()
        .filter(|qp| qp.dtype_str.contains("int"));
    let dtype = qparam
        .as_ref()
        .map(|qp| parse_torch_dtype(&qp.dtype_str))
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

fn arange_spec(node: &IrNode) -> Option<(f64, f64, f64)> {
    if node.target.contains("aten.arange.start_step") {
        let start = node.args.first()?.as_scalar_literal_f64()?;
        let end = node.args.get(1)?.as_scalar_literal_f64()?;
        let step = node
            .args
            .get(2)
            .and_then(IrArg::as_scalar_literal_f64)
            .unwrap_or(1.0);
        return Some((start, end, step));
    }
    if node.target.contains("aten.arange.start") {
        if node.args.len() >= 2 {
            return Some((
                node.args.first()?.as_scalar_literal_f64()?,
                node.args.get(1)?.as_scalar_literal_f64()?,
                1.0,
            ));
        }
        return Some((0.0, node.args.first()?.as_scalar_literal_f64()?, 1.0));
    }
    None
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

fn plan_zero_input_arange(node: &IrNode) -> Option<MaterializeTensorNodeEffect> {
    let (start, end, step) = arange_spec(node)?;
    let shape = arange_output_shape(node, start, end, step)?;
    let numel = tensor_numel(&shape)?;

    let (qparam, dtype) = output_int_qparam_and_dtype(node)?;

    let mut data = Vec::new();
    for idx in 0..numel {
        let value = start + (idx as f64 * step);
        let scalar = scalar_bytes_in_domain(dtype, qparam.as_ref(), value)?;
        data.extend_from_slice(&scalar);
    }

    Some(MaterializeTensorNodeEffect {
        node_name: node.name.clone(),
        data,
        shape,
        dtype,
        allow_missing: true,
    })
}

fn plan_zero_input_linspace(node: &IrNode) -> Option<MaterializeTensorNodeEffect> {
    if !node.target.contains("aten.linspace.default") {
        return None;
    }

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

    let mut data = Vec::new();
    for idx in 0..steps {
        let value = if steps <= 1 {
            start
        } else {
            start + (idx as f64 * (end - start) / (steps - 1) as f64)
        };
        let scalar = scalar_bytes_in_domain(dtype, qparam.as_ref(), value)?;
        data.extend_from_slice(&scalar);
    }

    Some(MaterializeTensorNodeEffect {
        node_name: node.name.clone(),
        data,
        shape,
        dtype,
        allow_missing: true,
    })
}

fn full_bail_reason(graph: &IrGraph, node: &IrNode) -> &'static str {
    let shape = match node.args.first() {
        Some(arg) => arg
            .as_int_list()
            .or_else(|| node.meta.val_shape().map(|s| s.to_vec())),
        None => node.meta.val_shape().map(|s| s.to_vec()),
    };
    if shape.is_none() {
        return "no_shape";
    }
    if full_fill_value(node).is_none() {
        return "no_fill_value";
    }
    let qparam = if is_quant_decompose_helper_full(node) {
        None
    } else {
        int_qparam(node).or_else(|| infer_binary_full_qparam(graph, node.id))
    };
    if let Some(qp) = qparam {
        if qp.is_per_channel() {
            return "qparam_per_channel";
        }
        if quantized_scalar_bytes(&qp, full_fill_value(node).unwrap()).is_none() {
            return "quantized_scalar_none";
        }
    } else if explicit_zero_input_dtype(node).is_none() {
        return "no_explicit_dtype";
    } else if scalar_storage_bytes(
        explicit_zero_input_dtype(node).unwrap(),
        full_fill_value(node).unwrap(),
    )
    .is_none()
    {
        return "scalar_fill_none";
    }
    "ok_but_planned_none"
}

fn plan_zero_input_eye(node: &IrNode) -> Option<MaterializeTensorNodeEffect> {
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

    let mut data = Vec::new();
    for row in 0..n {
        for col in 0..m {
            let value = if row == col { 1.0 } else { 0.0 };
            let scalar = scalar_bytes_in_domain(dtype, qparam.as_ref(), value)?;
            data.extend_from_slice(&scalar);
        }
    }

    Some(MaterializeTensorNodeEffect {
        node_name: node.name.clone(),
        data,
        shape,
        dtype,
        allow_missing: true,
    })
}

fn plan_zero_input_full(graph: &IrGraph, node: &IrNode) -> Option<MaterializeTensorNodeEffect> {
    let shape = match node.args.first() {
        Some(arg) => arg
            .as_int_list()
            .or_else(|| node.meta.val_shape().map(|shape| shape.to_vec())),
        None => node.meta.val_shape().map(|shape| shape.to_vec()),
    }?;
    let fill_value = full_fill_value(node)?;

    let qparam = if is_quant_decompose_helper_full(node) {
        None
    } else {
        int_qparam(node).or_else(|| infer_binary_full_qparam(graph, node.id))
    };

    let (dtype, scalar) = if let Some(qp) = qparam {
        quantized_scalar_bytes(&qp, fill_value)?
    } else {
        let dtype = explicit_zero_input_dtype(node)?;
        (dtype, scalar_storage_bytes(dtype, fill_value)?)
    };

    Some(MaterializeTensorNodeEffect {
        node_name: node.name.clone(),
        data: repeated_scalar_bytes(&shape, &scalar)?,
        shape,
        dtype,
        allow_missing: true,
    })
}

fn planned_zero_input_constant_kind(node: &IrNode) -> Option<ZeroInputConstantKind> {
    if !node.alive || node.op != NodeOp::CallFunction || node_has_tensor_inputs(node) {
        return None;
    }
    if node.target.contains("aten.full.default") {
        return Some(ZeroInputConstantKind::Full);
    }
    if node.target.contains("arange")
        || node.target.contains("linspace")
        || node.target.contains("eye")
    {
        return Some(ZeroInputConstantKind::FactoryCall);
    }
    None
}

impl EffectfulIrPass for IrFuseZeroInputConstantsEffectPass {
    fn name(&self) -> &str {
        "IrFuseZeroInputConstantsEp"
    }

    fn run(&self, graph: &mut IrGraph) -> EffectfulIrPassResult {
        let mut effects = Vec::new();
        for node in &graph.nodes {
            let Some(kind) = planned_zero_input_constant_kind(node) else {
                continue;
            };
            if kind == ZeroInputConstantKind::Full {
                if let Some(effect) = plan_zero_input_full(graph, node) {
                    effects.push(LoweringEffect::MaterializeTensorNode(effect));
                    continue;
                }
            } else if node.target.contains("aten.arange") {
                if let Some(effect) = plan_zero_input_arange(node) {
                    effects.push(LoweringEffect::MaterializeTensorNode(effect));
                    continue;
                }
            } else if node.target.contains("aten.linspace.default") {
                if let Some(effect) = plan_zero_input_linspace(node) {
                    effects.push(LoweringEffect::MaterializeTensorNode(effect));
                    continue;
                }
            } else if node.target.contains("aten.eye") {
                if let Some(effect) = plan_zero_input_eye(node) {
                    effects.push(LoweringEffect::MaterializeTensorNode(effect));
                    continue;
                }
            }
            let why = if kind == ZeroInputConstantKind::Full {
                full_bail_reason(graph, node)
            } else {
                "n/a"
            };
            crate::fold_fallback_log(&format!(
                "MaterializeZeroInputConstant target={} kind={:?} why={}",
                node.target, kind, why
            ));
            effects.push(LoweringEffect::MaterializeZeroInputConstant(
                MaterializeZeroInputConstantEffect {
                    node_name: node.name.clone(),
                    kind,
                    allow_missing: true,
                },
            ));
        }
        let modified = !effects.is_empty();
        EffectfulIrPassResult {
            modified,
            effects: LoweringEffectBatch { items: effects },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fx_ir::{IrGraph, NodeMeta};

    use super::*;

    fn add_call(graph: &mut IrGraph, name: &str, target: &str, args: Vec<IrArg>) -> usize {
        graph.add_node(
            NodeOp::CallFunction,
            name.to_string(),
            target.to_string(),
            args,
            NodeMeta::default(),
        )
    }

    #[test]
    fn plans_zero_input_factory_constants() {
        let mut graph = IrGraph::empty();
        let full = add_call(
            &mut graph,
            "full",
            "aten.full.default",
            vec![IrArg::List(vec![IrArg::Int(2)]), IrArg::Float(3.0)],
        );
        graph.node_mut(full).kwargs.insert(
            "dtype".to_string(),
            IrArg::Dtype("torch.float32".to_string()),
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

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

        assert!(result.modified);
        assert_eq!(result.effects.items.len(), 4);
        assert!(matches!(
            &result.effects.items[0],
            LoweringEffect::MaterializeTensorNode(effect)
                if effect.node_name == "full"
                    && effect.dtype == DType::Float32
                    && effect.shape == vec![2]
                    && effect.data == [3.0f32.to_le_bytes(), 3.0f32.to_le_bytes()].concat()
        ));
        assert!(matches!(
            &result.effects.items[1],
            LoweringEffect::MaterializeZeroInputConstant(effect)
                if effect.kind == ZeroInputConstantKind::FactoryCall
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
        let other = graph.add_node(
            NodeOp::Placeholder,
            "other".to_string(),
            "other".to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        graph
            .node_mut(other)
            .meta
            .set_single_output_qparam(QParam::per_tensor(
                0.25,
                0,
                -128,
                127,
                "torch.int8".to_string(),
            ));
        add_call(
            &mut graph,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(full), IrArg::Node(other)],
        );

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

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
        let scale = add_call(
            &mut graph,
            "quantize_per_tensor_3_sc",
            "aten.full.default",
            vec![
                IrArg::List(vec![IrArg::Int(1), IrArg::Int(1), IrArg::Int(1)]),
                IrArg::Float(4.0),
            ],
        );
        graph.node_mut(scale).kwargs.insert(
            "dtype".to_string(),
            IrArg::Dtype("torch.float32".to_string()),
        );
        let other = graph.add_node(
            NodeOp::Placeholder,
            "other".to_string(),
            "other".to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        graph
            .node_mut(other)
            .meta
            .set_single_output_qparam(QParam::per_tensor(
                0.25,
                0,
                i32::MIN as i64,
                i32::MAX as i64,
                "torch.int32".to_string(),
            ));
        add_call(
            &mut graph,
            "quantize_per_tensor_3_mul",
            "aten.mul.Tensor",
            vec![IrArg::Node(other), IrArg::Node(scale)],
        );

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

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
        let arange = add_call(
            &mut graph,
            "arange",
            "aten.arange.start_step",
            vec![IrArg::Float(0.0), IrArg::Float(1.0), IrArg::Float(0.25)],
        );
        graph.node_mut(arange).kwargs.insert(
            "dtype".to_string(),
            IrArg::Dtype("torch.float32".to_string()),
        );

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

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
            .set_single_output_qparam(QParam::per_tensor(
                0.5,
                0,
                -128,
                127,
                "torch.int8".to_string(),
            ));

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

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
        let linspace = add_call(
            &mut graph,
            "linspace",
            "aten.linspace.default",
            vec![IrArg::Float(0.0), IrArg::Float(1.0), IrArg::Int(5)],
        );
        graph.node_mut(linspace).kwargs.insert(
            "dtype".to_string(),
            IrArg::Dtype("torch.float32".to_string()),
        );

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

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
        let eye = add_call(&mut graph, "eye", "aten.eye.default", vec![IrArg::Int(3)]);
        graph.node_mut(eye).kwargs.insert(
            "dtype".to_string(),
            IrArg::Dtype("torch.float32".to_string()),
        );

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

        let effect = match &result.effects.items[0] {
            LoweringEffect::MaterializeTensorNode(effect) => effect,
            other => panic!("unexpected effect kind: {other:?}"),
        };
        assert_eq!(effect.node_name, "eye");
        assert_eq!(effect.shape, vec![3, 3]);
        let values = effect
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
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
            .set_single_output_qparam(QParam::per_tensor(
                0.25,
                0,
                -128,
                127,
                "torch.int8".to_string(),
            ));

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

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
        let placeholder = graph.add_node(
            NodeOp::Placeholder,
            "x".to_string(),
            "x".to_string(),
            Vec::new(),
            NodeMeta::default(),
        );
        add_call(
            &mut graph,
            "not_zero_input",
            "aten.full.default",
            vec![IrArg::Node(placeholder), IrArg::Int(3)],
        );

        let pass = IrFuseZeroInputConstantsEffectPass;
        let result = pass.run(&mut graph);

        assert!(!result.modified);
        assert!(result.effects.items.is_empty());
    }
}
