use std::collections::{HashMap, HashSet};

use crate::graph::ir::{Arg, OpTarget, OpType};
use crate::graph::Graph;
use crate::store::TensorStore;

use super::constant_pool::ConstantPool;
use super::tensor_decl;
use super::tosa_types::{
    Attribute, ConstShapeAttribute, Op, ReshapeAttribute, TosaOperator, TosaShape, TosaTensor,
};
use super::{op_semantics, tosa_types, visitors};

pub(super) struct SerializedGraphItems {
    pub(super) tensors: Vec<TosaTensor>,
    pub(super) shapes: Vec<TosaShape>,
    pub(super) operators: Vec<TosaOperator>,
    pub(super) input_names: Vec<String>,
    pub(super) output_names: Vec<String>,
}

pub(super) struct SerializedRegion {
    pub(super) name: String,
    pub(super) items: SerializedGraphItems,
}

pub(super) fn collect_regions(
    graphs: &[(String, Graph)],
    store: &TensorStore,
) -> Result<Vec<SerializedRegion>, String> {
    graphs
        .iter()
        .map(|(name, graph)| collect_region(name, graph, store))
        .collect()
}

fn collect_region(
    name: &str,
    graph: &Graph,
    store: &TensorStore,
) -> Result<SerializedRegion, String> {
    Ok(SerializedRegion {
        name: name.to_string(),
        items: collect_graph_items(name, graph, store)?,
    })
}

/// The names of the tensors a region returns. IrNormalizeDelegateIOLayout owns the
/// delegate boundary's layout, so each output is the node's own tensor (P2 design G4).
fn append_output_names(graph: &Graph, arg: &Arg, output_names: &mut Vec<String>) {
    match arg {
        Arg::Node(id) => {
            if let Some(node) = graph.node(*id) {
                output_names.push(node.name.clone());
            }
        }
        Arg::NodeList(ids) => {
            for id in ids {
                if let Some(node) = graph.node(*id) {
                    output_names.push(node.name.clone());
                }
            }
        }
        _ => {}
    }
}

fn collect_graph_items(
    region: &str,
    graph: &Graph,
    store: &TensorStore,
) -> Result<SerializedGraphItems, String> {
    let debug = crate::config::retab_debug_enabled();
    let sorted = graph.all_node_ids();
    if debug {
        eprintln!("[Serialize] using insertion order, {} nodes", sorted.len());
        validate_insertion_order(graph, &sorted);
    }
    reject_aten_spatial_ops(region, graph)?;
    if debug {
        check_layout_invariants(region, graph, store)?;
    }

    let mut tensors: Vec<TosaTensor> = Vec::new();
    let mut shapes: Vec<TosaShape> = Vec::new();
    let mut operators: Vec<TosaOperator> = Vec::new();
    let mut constant_pool = ConstantPool::default();
    let mut input_names: Vec<String> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();
    let mut unhandled: Vec<String> = Vec::new();

    for (loop_idx, &node_id) in sorted.iter().enumerate() {
        let Some(node) = graph.node(node_id) else {
            continue;
        };
        let node_start = std::time::Instant::now();

        if debug {
            eprintln!(
                "[Serialize] node '{}' op={:?} target={:?}",
                node.name, node.op, node.target
            );
        }

        match node.op {
            OpType::Placeholder => {
                let name = &node.name;

                if let Some(tensor_data) = store.get(name) {
                    tensors.push(tensor_decl::const_placeholder(node, tensor_data));
                    operators.push(TosaOperator::new(
                        tosa_types::Op::Const,
                        Attribute::Const(tosa_types::ConstAttribute {}),
                        vec![],
                        vec![name.clone()],
                    ));
                } else {
                    tensors.push(tensor_decl::runtime_placeholder(node));
                    input_names.push(name.clone());
                }
            }
            OpType::CallFunction => {
                if node.target == OpTarget::ArgMax
                    && node.args.get(1).and_then(Arg::as_i64).is_none()
                    && !node
                        .kwargs
                        .iter()
                        .any(|(key, value)| key == "dim" && value.as_i64().is_some())
                {
                    return Err(format!(
                        "argmax node '{}' in region '{}' requires an explicit dim",
                        node.name, region
                    ));
                }
                if let Some(mut emit_result) = visitors::emit_node(graph, node_id, node, store) {
                    constant_pool.absorb(&mut emit_result);
                    tensors.extend(emit_result.tensors);
                    shapes.extend(emit_result.shapes);
                    operators.extend(emit_result.operators);
                } else {
                    unhandled.push(format!("'{}' ({:?})", node.name, node.target));
                }
            }
            OpType::Output => {
                for arg in &node.args {
                    append_output_names(graph, arg, &mut output_names);
                }
            }
        }

        let elapsed = node_start.elapsed();
        if debug && elapsed.as_millis() > 100 {
            eprintln!(
                "[Serialize] SLOW node {}/{} name={} target={:?} took {:?}",
                loop_idx,
                sorted.len(),
                node.name,
                node.target,
                elapsed
            );
        }
    }

    match_scalar_operand_ranks(
        &mut tensors,
        &mut shapes,
        &mut operators,
        &mut constant_pool,
    );

    if debug {
        eprintln!(
            "[Serialize] topo sort done, collected {} tensors, {} operators, {} shapes",
            tensors.len(),
            operators.len(),
            shapes.len()
        );
    }

    if !unhandled.is_empty() {
        return Err(format!(
            "RETAB has no TOSA emitter for {} node(s) in region '{}': {}",
            unhandled.len(),
            region,
            unhandled.join(", ")
        ));
    }
    if let Some(tensor) = tensors
        .iter()
        .find(|tensor| tensor.dtype == tosa_types::DType::Int64)
    {
        return Err(format!(
            "TOSA 1.0 does not support INT64 tensor '{}'",
            tensor.name
        ));
    }
    // Shape values come from CONST_SHAPE; Vela rejects a CONST whose output has shape type.
    if let Some(tensor) = tensors
        .iter()
        .find(|tensor| tensor.dtype == tosa_types::DType::Shape)
    {
        return Err(format!(
            "shape-typed tensor '{}' must be produced by CONST_SHAPE, not stored as a tensor",
            tensor.name
        ));
    }
    // Floating-profile compile specs use stock preprocess, so a quantized region (any
    // runtime INT4/INT8/INT16 tensor; constants such as MUL's INT8 shift don't count)
    // must not carry float tensors. Pure-float regions only come from tests calling the
    // native API directly; the spec-based check arrives with the pipeline config (APU-124).
    let quantized_region = tensors.iter().any(|tensor| {
        use tosa_types::DType;
        tensor.data.is_none() && matches!(tensor.dtype, DType::Int4 | DType::Int8 | DType::Int16)
    });
    if let Some(tensor) = tensors.iter().filter(|_| quantized_region).find(|tensor| {
        use tosa_types::DType;
        matches!(
            tensor.dtype,
            DType::Fp32
                | DType::Fp16
                | DType::Bf16
                | DType::Fp8E4M3
                | DType::Fp8E5M2
                | DType::Fp6E2M3
                | DType::Fp6E3M2
                | DType::Fp4E2M1
                | DType::Fp8Ue8M0
        )
    }) {
        return Err(format!(
            "TOSA INT profile does not support {} tensor '{}' in region '{}'",
            format!("{:?}", tensor.dtype).to_uppercase(),
            tensor.name,
            region
        ));
    }
    validate_elementwise_ranks(region, &tensors, &operators)?;

    Ok(SerializedGraphItems {
        tensors,
        shapes,
        operators,
        input_names,
        output_names,
    })
}

/// Every aten conv, pool and upsample must have been rewritten to its TOSA op (P2 design
/// X5, critique B11). They have no visitor, so a survivor (a rewrite declined it) sends
/// the graph to stock, through the same seam as `control_flow::assembly`.
fn reject_aten_spatial_ops(region: &str, graph: &Graph) -> Result<(), String> {
    let Some(node) = graph
        .order
        .iter()
        .filter_map(|&id| graph.node(id))
        .find(|node| {
            node.op == OpType::CallFunction && op_semantics::is_aten_spatial_layout_op(&node.target)
        })
    else {
        return Ok(());
    };
    Err(format!(
        "RETAB_STOCK_FALLBACK: aten spatial op not rewritten to a TOSA op: '{}' ({:?}) in \
         region '{}'",
        node.name, node.target, region
    ))
}

/// Under RETAB_DEBUG (P2 design D1): a node's val is its TOSA shape, so the serializer's
/// re-derived shape must equal the val. This catches stale vals.
pub(super) fn check_layout_invariants(
    region: &str,
    graph: &Graph,
    store: &TensorStore,
) -> Result<(), String> {
    let mut violations = Vec::new();
    for node in graph.order.iter().filter_map(|&id| graph.node(id)) {
        if node.op == OpType::Output {
            continue;
        }
        let val: Vec<i32> = node
            .meta
            .val
            .shape_slice()
            .iter()
            .map(|&dim| dim as i32)
            .collect();
        let actual = visitors::actual_tensor_shape(graph, store, node);
        if actual != val {
            violations.push(format!(
                "'{}' ({:?}) has val {val:?} but serializes as {actual:?}",
                node.name, node.target
            ));
        }
    }
    if violations.is_empty() {
        return Ok(());
    }
    Err(format!(
        "layout invariant broken in region '{region}': {}",
        violations.join("; ")
    ))
}

fn elementwise_data_input_count(op: Op) -> usize {
    // MUL's shift and RESCALE/TABLE parameters keep their own ranks.
    match op {
        Op::Select => 3,
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::IntDiv
        | Op::Pow
        | Op::ArithmeticRightShift
        | Op::LogicalLeftShift
        | Op::LogicalRightShift
        | Op::BitwiseAnd
        | Op::BitwiseOr
        | Op::BitwiseXor
        | Op::Equal
        | Op::Greater
        | Op::GreaterEqual
        | Op::LogicalAnd
        | Op::LogicalOr
        | Op::LogicalXor
        | Op::Maximum
        | Op::Minimum => 2,
        Op::Abs
        | Op::BitwiseNot
        | Op::Cast
        | Op::Ceil
        | Op::Clamp
        | Op::Clz
        | Op::Cos
        | Op::Erf
        | Op::Exp
        | Op::Floor
        | Op::Identity
        | Op::Log
        | Op::LogicalNot
        | Op::Negate
        | Op::Reciprocal
        | Op::Rescale
        | Op::Rsqrt
        | Op::Sigmoid
        | Op::Sin
        | Op::Table
        | Op::Tanh => 1,
        _ => 0,
    }
}

fn validate_elementwise_ranks(
    region: &str,
    tensors: &[TosaTensor],
    operators: &[TosaOperator],
) -> Result<(), String> {
    let tensors: HashMap<_, _> = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    for op in operators {
        let data_inputs = elementwise_data_input_count(op.op);
        if data_inputs == 0 {
            continue;
        }
        for output in &op.outputs {
            let Some(output_tensor) = tensors.get(output.as_str()) else {
                continue;
            };
            for input in op.inputs.iter().take(data_inputs) {
                let Some(input_tensor) = tensors.get(input.as_str()) else {
                    continue;
                };
                if input_tensor.shape.len() != output_tensor.shape.len() {
                    return Err(format!(
                        "TOSA {} in region '{}' has mismatched ranks: input '{}' {:?} (rank {}), output '{}' {:?} (rank {})",
                        format!("{:?}", op.op).to_uppercase(),
                        region,
                        input,
                        input_tensor.shape,
                        input_tensor.shape.len(),
                        output,
                        output_tensor.shape,
                        output_tensor.shape.len(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn match_scalar_operand_ranks(
    tensors: &mut Vec<TosaTensor>,
    shapes: &mut Vec<TosaShape>,
    operators: &mut Vec<TosaOperator>,
    constant_pool: &mut ConstantPool,
) {
    if !tensors.iter().any(|tensor| tensor.shape.is_empty()) {
        return;
    }
    let mut tensor_ids: HashMap<String, usize> = tensors
        .iter()
        .enumerate()
        .map(|(id, tensor)| (tensor.name.clone(), id))
        .collect();
    let mut matched = Vec::with_capacity(operators.len());
    let mut declared_shapes = HashSet::new();
    for mut op in std::mem::take(operators) {
        if op.op == Op::ConstShape && !declared_shapes.insert(op.outputs[0].clone()) {
            continue;
        }
        let data_inputs = elementwise_data_input_count(op.op);
        if data_inputs > 1 {
            let rank = op
                .inputs
                .iter()
                .take(data_inputs)
                .filter_map(|name| tensor_ids.get(name))
                .map(|&id| tensors[id].shape.len())
                .max()
                .unwrap_or(0);
            for (index, input) in op.inputs.iter_mut().take(data_inputs).enumerate() {
                let Some(&id) = tensor_ids.get(input) else {
                    continue;
                };
                if rank == 0 || !tensors[id].shape.is_empty() {
                    continue;
                }
                let name = format!("{}_input{index}_rank", op.outputs[0]);
                let shape_name = format!("{name}_shape");
                let target_shape = vec![1; rank];
                let mut result = visitors::EmitResult::default();
                result.poolable.push(shape_name.clone());
                result.shapes.push(TosaShape {
                    name: shape_name.clone(),
                    rank: rank as u32,
                    data: target_shape
                        .iter()
                        .flat_map(|&dim| (dim as i64).to_le_bytes())
                        .collect(),
                });
                result.push_op(
                    Op::ConstShape,
                    Attribute::ConstShape(ConstShapeAttribute {}),
                    vec![],
                    vec![shape_name.clone()],
                );
                result.push_output_op(
                    TosaTensor::new(&name, target_shape, tensors[id].dtype),
                    Op::Reshape,
                    Attribute::Reshape(ReshapeAttribute {}),
                    vec![input.clone(), shape_name],
                );
                constant_pool.absorb(&mut result);
                tensor_ids.insert(name.clone(), tensors.len());
                tensors.extend(result.tensors);
                shapes.extend(result.shapes);
                for generated in result.operators {
                    if generated.op == Op::ConstShape {
                        declared_shapes.insert(generated.outputs[0].clone());
                    }
                    if generated.op == Op::Reshape
                        && declared_shapes.insert(generated.inputs[1].clone())
                    {
                        // A pooled shape may originally have been declared after this consumer.
                        matched.push(TosaOperator::new(
                            Op::ConstShape,
                            Attribute::ConstShape(ConstShapeAttribute {}),
                            vec![],
                            vec![generated.inputs[1].clone()],
                        ));
                    }
                    matched.push(generated);
                }
                *input = name;
            }
        }
        matched.push(op);
    }
    *operators = matched;
}

fn validate_insertion_order(graph: &Graph, sorted: &[crate::graph::ir::NodeId]) {
    let mut position = std::collections::HashMap::new();
    for (i, &id) in sorted.iter().enumerate() {
        position.insert(id, i);
    }

    let mut violations = 0;
    for (i, &id) in sorted.iter().enumerate() {
        let Some(node) = graph.node(id) else {
            continue;
        };
        for input_id in Graph::collect_input_ids(node) {
            let Some(&input_pos) = position.get(&input_id) else {
                continue;
            };
            if input_pos < i {
                continue;
            }
            if violations < 10 {
                eprintln!(
                    "[Serialize] WARNING: '{}' at position {} depends on input at position {}",
                    node.name, i, input_pos
                );
            }
            violations += 1;
        }
    }

    if violations > 0 {
        eprintln!(
            "[Serialize] WARNING: {} topological order violations detected",
            violations
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, PipelineStep};
    use crate::fx_pipeline::{build_ir_cond_submodule_pipeline, build_ir_while_submodule_pipeline};
    use crate::graph::ir::{Arg, DType, NodeMeta, OpTarget, TensorInfo};
    use crate::ingest::target_map::parse_op_target;
    use crate::lowering_effects::LoweringEffectBatch;
    use crate::pipeline_config::PipelineConfig;
    use crate::store::TensorData;
    use crate::test_ir::{call, output, placeholder, qparam};
    use smallvec::smallvec;

    fn lower_region(graph: &mut IrGraph, pipeline: Vec<PipelineStep>) -> SerializedGraphItems {
        let mut effects = LoweringEffectBatch::default();
        for step in pipeline {
            match step {
                PipelineStep::Ir(pass) => {
                    pass.run(graph);
                }
                PipelineStep::Effectful(pass) => {
                    effects.items.extend(pass.run(graph).effects.items);
                }
            }
        }
        let mut store = TensorStore::new();
        let mut converted =
            crate::ingest::from_ir::graph_from_ir_preserve_placeholders(graph, &mut store);
        crate::ingest::from_ir::apply_effects_preserve_placeholders(
            &mut converted,
            &mut store,
            &effects,
        );
        crate::ingest::finalize::finalize_converted_graph(&mut converted);
        visitors::shape_utils::clear_shape_cache();
        visitors::dtype_quant::clear_dtype_cache();
        collect_graph_items("region", &converted, &store).unwrap()
    }

    #[test]
    fn unique_output_identities_serialize_without_requantizing_the_input() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 3], "torch.int8");
        let out = output(&mut graph, x);
        graph.set_args(out, vec![IrArg::List(vec![IrArg::Node(x), IrArg::Node(x)])]);
        let items = lower_region(
            &mut graph,
            vec![PipelineStep::Ir(Box::new(
                crate::fx_ensure_unique_output_nodes_ir::IrEnsureUniqueOutputNodesPass,
            ))],
        );
        assert_eq!(
            items.operators.iter().map(|op| op.op).collect::<Vec<_>>(),
            vec![Op::Identity, Op::Identity]
        );
        assert_ne!(items.output_names[0], items.output_names[1]);
        assert!(items.operators.iter().all(|op| op.inputs == ["x"]));
        assert!(items
            .tensors
            .iter()
            .all(|tensor| tensor.dtype == tosa_types::DType::Int8));
    }

    #[test]
    fn rejects_elementwise_input_output_rank_mismatches() {
        for (op, data_inputs) in [
            (Op::Add, 2),
            (Op::Sub, 2),
            (Op::Mul, 2),
            (Op::Rescale, 1),
            (Op::Cast, 1),
            (Op::Select, 3),
            (Op::Equal, 2),
            (Op::Greater, 2),
            (Op::GreaterEqual, 2),
            (Op::Table, 1),
            (Op::Identity, 1),
            (Op::Clamp, 1),
            (Op::LogicalAnd, 2),
            (Op::BitwiseAnd, 2),
            (Op::Negate, 1),
            (Op::Maximum, 2),
        ] {
            for (input_shape, output_shape) in
                [(vec![1], vec![]), (vec![], vec![1]), (vec![1], vec![1, 1])]
            {
                for bad_input in 0..data_inputs {
                    let inputs: Vec<_> = (0..data_inputs).map(|i| format!("input{i}")).collect();
                    let mut tensors: Vec<_> = inputs
                        .iter()
                        .map(|name| {
                            TosaTensor::new(
                                name,
                                output_shape.clone(),
                                tosa_types::DType::Int32,
                            )
                        })
                        .collect();
                    tensors[bad_input].shape = input_shape.clone();
                    tensors.push(TosaTensor::new(
                        "output",
                        output_shape.clone(),
                        tosa_types::DType::Int32,
                    ));
                    let operators = vec![TosaOperator::new(
                        op,
                        Attribute::None,
                        inputs,
                        vec!["output".into()],
                    )];
                    let error =
                        validate_elementwise_ranks("cond", &tensors, &operators).unwrap_err();
                    assert!(error.contains(&format!("TOSA {}", format!("{op:?}").to_uppercase())));
                    assert!(error.contains("region 'cond'"));
                    assert!(error.contains(&format!("input 'input{bad_input}' {input_shape:?}")));
                    assert!(error.contains(&format!("output 'output' {output_shape:?}")));
                }
            }
        }
    }

    #[test]
    fn elementwise_rank_check_allows_broadcast_and_parameter_ranks() {
        for shape in [vec![], vec![2, 3]] {
            let tensors = vec![
                TosaTensor::new("x", shape.clone(), tosa_types::DType::Int32),
                TosaTensor::new("y", vec![1; shape.len()], tosa_types::DType::Int32),
                TosaTensor::new("predicate", shape.clone(), tosa_types::DType::Bool),
                TosaTensor::new("output", shape, tosa_types::DType::Int32),
                TosaTensor::new("shift", vec![1], tosa_types::DType::Int8),
                TosaTensor::new("multiplier", vec![1], tosa_types::DType::Int32),
                TosaTensor::new("zp", vec![1], tosa_types::DType::Int32),
                TosaTensor::new("table", vec![256], tosa_types::DType::Int8),
            ];
            for (op, inputs) in [
                (Op::Add, vec!["x", "y"]),
                (Op::Mul, vec!["x", "y", "shift"]),
                (Op::Rescale, vec!["x", "multiplier", "shift", "zp", "zp"]),
                (Op::Table, vec!["x", "table"]),
                (Op::Select, vec!["predicate", "x", "y"]),
                (Op::Cast, vec!["x"]),
            ] {
                let operators = vec![TosaOperator::new(
                    op,
                    Attribute::None,
                    inputs.into_iter().map(str::to_string).collect(),
                    vec!["output".into()],
                )];
                assert_eq!(
                    validate_elementwise_ranks("main", &tensors, &operators),
                    Ok(())
                );
            }
        }
    }

    #[test]
    fn rank_matching_does_not_hide_unary_rank_mismatches() {
        for op in [Op::Rescale, Op::Cast, Op::Table, Op::Identity] {
            let mut tensors = vec![
                TosaTensor::new("x", vec![], tosa_types::DType::Int32),
                TosaTensor::new("output", vec![1], tosa_types::DType::Int32),
            ];
            let mut shapes = Vec::new();
            let mut operators = vec![TosaOperator::new(
                op,
                Attribute::None,
                vec!["x".into()],
                vec!["output".into()],
            )];
            match_scalar_operand_ranks(
                &mut tensors,
                &mut shapes,
                &mut operators,
                &mut ConstantPool::default(),
            );
            assert_eq!(operators.len(), 1);
            assert!(shapes.is_empty());
            assert!(validate_elementwise_ranks("main", &tensors, &operators).is_err());
        }
    }

    #[test]
    fn collection_rejects_elementwise_output_rank_mismatches() {
        visitors::shape_utils::clear_shape_cache();
        visitors::dtype_quant::clear_dtype_cache();
        let mut graph = Graph::new();
        let predicate = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("predicate".into()),
            smallvec![],
            Vec::new(),
            NodeMeta::new(TensorInfo::new(&[1], DType::Bool)),
            "predicate".into(),
        );
        let x = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("x".into()),
            smallvec![],
            Vec::new(),
            NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
            "x".into(),
        );
        graph.create_node(
            OpType::CallFunction,
            OpTarget::Where,
            smallvec![Arg::Node(predicate), Arg::Node(x), Arg::Node(x)],
            Vec::new(),
            NodeMeta::new(TensorInfo::new(&[], DType::Int32)),
            "select".into(),
        );
        let error = collect_graph_items("cond", &graph, &TensorStore::new())
            .err()
            .unwrap();
        assert_eq!(
            error,
            "TOSA SELECT in region 'cond' has mismatched ranks: input 'predicate' [1] (rank 1), output 'select' [] (rank 0)",
        );
    }

    #[test]
    fn scalar_mean_and_constants_keep_rank0() {
        visitors::shape_utils::clear_shape_cache();
        visitors::dtype_quant::clear_dtype_cache();
        let mut ir = IrGraph::empty();
        let x = placeholder(&mut ir, "x", [2, 3], "torch.float32");
        let mean = call(
            &mut ir,
            "mean",
            "aten.mean.dim",
            vec![IrArg::Node(x), i64_list_arg(&[0, 1]), IrArg::Bool(false)],
            [],
            "torch.float32",
        );
        let scalar = call(
            &mut ir,
            "scalar",
            "aten.scalar_tensor.default",
            vec![IrArg::Float(0.5)],
            [],
            "torch.float32",
        );
        let add = call(
            &mut ir,
            "add",
            "aten.add.Tensor",
            vec![IrArg::Node(mean), IrArg::Node(scalar)],
            [],
            "torch.float32",
        );
        output(&mut ir, add);
        let items = lower_region(&mut ir, Vec::new());
        for name in ["mean", "scalar", "add"] {
            assert!(items
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap()
                .shape
                .is_empty());
        }
        assert_eq!(
            items
                .operators
                .iter()
                .filter(|op| op.op == Op::ReduceSum)
                .count(),
            2
        );
        assert_eq!(
            items.operators.iter().filter(|op| op.op == Op::Mul).count(),
            1
        );
        assert_eq!(
            items
                .operators
                .iter()
                .filter(|op| op.op == Op::Reshape)
                .count(),
            1
        );
        assert_eq!(items.output_names, vec!["add"]);
        assert_eq!(items.shapes[0].rank, 0);
        assert!(items.shapes[0].data.is_empty());
    }

    #[test]
    fn scalar_sum_and_rescale_keep_rank0() {
        for decompose in [false, true] {
            let mut ir = IrGraph::empty();
            let x = placeholder(&mut ir, "x", [2, 3], "torch.int32");
            let sum = call(
                &mut ir,
                "sum",
                "aten.sum.dim_IntList",
                vec![IrArg::Node(x), i64_list_arg(&[0, 1]), IrArg::Bool(false)],
                [],
                "torch.int32",
            );
            let rescale = call(
                &mut ir,
                "rescale",
                "backend.tosa.RESCALE.default",
                vec![
                    IrArg::Node(sum),
                    IrArg::Dtype("torch.int8".into()),
                    IrArg::Float(0.25),
                    IrArg::Int(0),
                    IrArg::Int(-5),
                ],
                [],
                "torch.int8",
            );
            output(&mut ir, rescale);
            let pipeline = if decompose {
                vec![PipelineStep::Ir(Box::new(
                    crate::fx_convert_sum_ir::IrConvertSumPass,
                ))]
            } else {
                Vec::new()
            };
            let items = lower_region(&mut ir, pipeline);
            let rescale = items
                .operators
                .iter()
                .find(|op| op.op == Op::Rescale)
                .unwrap();
            for (name, dtype) in [
                (&rescale.inputs[0], tosa_types::DType::Int32),
                (&rescale.outputs[0], tosa_types::DType::Int8),
            ] {
                let tensor = items
                    .tensors
                    .iter()
                    .find(|tensor| &tensor.name == name)
                    .unwrap();
                assert!(tensor.shape.is_empty());
                assert_eq!(tensor.dtype, dtype);
            }
            assert_eq!(items.output_names, vec!["rescale"]);
            let graph = crate::ingest::from_ir::graph_from_ir(&ir, &mut TensorStore::new());
            visitors::shape_utils::clear_shape_cache();
            for id in graph.all_node_ids() {
                let node = graph.node(id).unwrap();
                if let Some(tensor) = items.tensors.iter().find(|tensor| tensor.name == node.name) {
                    assert_eq!(
                        visitors::actual_tensor_shape(&graph, &TensorStore::new(), node),
                        tensor.shape,
                        "{}",
                        node.name,
                    );
                }
            }
            let reshape = items
                .operators
                .iter()
                .rev()
                .find(|op| op.op == Op::Reshape)
                .unwrap();
            let shape = items
                .shapes
                .iter()
                .find(|shape| shape.name == reshape.inputs[1])
                .unwrap();
            assert_eq!(shape.rank, 0);
            assert!(shape.data.is_empty());
            assert_eq!(
                items
                    .operators
                    .iter()
                    .filter(|op| op.op == Op::ReduceSum)
                    .count(),
                2
            );
            assert_eq!(
                items
                    .operators
                    .iter()
                    .filter(|op| op.op == Op::Reshape)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn while_regions_keep_scalar_predicates_and_carried_values() {
        let mut main = IrGraph::empty();
        let x = placeholder(&mut main, "x", [2, 3], "torch.int32");
        let counter = placeholder(&mut main, "counter", [], "torch.int32");
        let threshold = placeholder(&mut main, "threshold", [1], "torch.int32");
        let while_loop = call(
            &mut main,
            "loop",
            "while_loop",
            vec![
                IrArg::Str("cond".into()),
                IrArg::Str("body".into()),
                IrArg::List(vec![IrArg::Node(x), IrArg::Node(counter)]),
                IrArg::List(vec![IrArg::Node(threshold)]),
            ],
            [],
            "torch.int32",
        );
        let result = call(
            &mut main,
            "result",
            "getitem",
            vec![IrArg::Node(while_loop), IrArg::Int(1)],
            [],
            "torch.int32",
        );
        output(&mut main, result);

        let mut cond = IrGraph::empty();
        let x = placeholder(&mut cond, "x", [2, 3], "torch.int32");
        placeholder(&mut cond, "counter", [], "torch.int32");
        let threshold = placeholder(&mut cond, "threshold", [1], "torch.int32");
        let sum = call(
            &mut cond,
            "sum",
            "aten.sum.dim_IntList",
            vec![IrArg::Node(x), i64_list_arg(&[0, 1]), IrArg::Bool(false)],
            [],
            "torch.int32",
        );
        let quantized = call(
            &mut cond,
            "quantized",
            "backend.tosa.RESCALE.default",
            vec![
                IrArg::Node(sum),
                IrArg::Dtype("torch.int8".into()),
                IrArg::Float(0.25),
            ],
            [],
            "torch.int8",
        );
        let comparison_input = call(
            &mut cond,
            "comparison_input",
            "backend.tosa.RESCALE.default",
            vec![
                IrArg::Node(quantized),
                IrArg::Dtype("torch.int32".into()),
                IrArg::Float(1.0),
            ],
            [],
            "torch.int32",
        );
        let gt = call(
            &mut cond,
            "gt",
            "aten.gt.Tensor",
            vec![IrArg::Node(comparison_input), IrArg::Node(threshold)],
            [1],
            "torch.bool",
        );
        let predicate = call(
            &mut cond,
            "predicate",
            "aten.view_copy.default",
            vec![IrArg::Node(gt), i64_list_arg(&[])],
            [],
            "torch.bool",
        );
        output(&mut cond, predicate);

        let mut body = IrGraph::empty();
        let x = placeholder(&mut body, "x", [2, 3], "torch.int32");
        let counter = placeholder(&mut body, "counter", [], "torch.int32");
        let threshold = placeholder(&mut body, "threshold", [1], "torch.int32");
        let out = output(&mut body, counter);
        body.set_args(
            out,
            vec![IrArg::List(vec![
                IrArg::Node(x),
                IrArg::Node(counter),
                IrArg::Node(threshold),
            ])],
        );
        let mut store = TensorStore::new();
        let graph = crate::ingest::from_ir::graph_from_ir(&main, &mut store);
        let mut graphs = vec![("main".into(), graph)];
        for (region, ir) in [("cond", cond), ("body", body)] {
            let mut graph =
                crate::ingest::from_ir::graph_from_ir_preserve_placeholders(&ir, &mut store);
            crate::ingest::finalize::finalize_converted_graph(&mut graph);
            crate::ingest::from_ir::suffix_graph(&mut graph, &mut store, region);
            graphs.push((region.into(), graph));
        }
        visitors::shape_utils::clear_shape_cache();
        visitors::dtype_quant::clear_dtype_cache();
        let regions = collect_regions(&graphs, &store).unwrap();
        let items = |name: &str| {
            &regions
                .iter()
                .find(|region| region.name == name)
                .unwrap()
                .items
        };
        let shape = |region: &str, name: &str| {
            &items(region)
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .unwrap()
                .shape
        };
        let while_loop = items("main")
            .operators
            .iter()
            .find(|op| op.op == Op::WhileLoop)
            .unwrap();
        assert_eq!(while_loop.inputs.len(), 3);
        assert_eq!(while_loop.outputs.len(), 3);
        assert!(shape("main", &while_loop.inputs[1]).is_empty());
        assert!(shape("main", &while_loop.outputs[1]).is_empty());
        for region in ["cond", "body"] {
            assert!(shape(region, &items(region).input_names[1]).is_empty());
        }
        assert!(shape("body", &items("body").output_names[1]).is_empty());
        assert_eq!(items("body").output_names.len(), 3);
        assert_eq!(items("main").output_names, vec!["result"]);
        assert_eq!(items("cond").output_names, vec!["predicate_cond"]);
        for name in [
            "sum_cond",
            "quantized_cond",
            "comparison_input_cond",
            "predicate_cond",
        ] {
            assert!(shape("cond", name).is_empty());
        }
        assert_eq!(shape("cond", "gt_cond"), &[1]);
        let comparison = items("cond")
            .operators
            .iter()
            .find(|op| op.op == Op::Greater)
            .unwrap();
        assert_eq!(shape("cond", &comparison.inputs[0]), &[1]);
        assert_eq!(shape("cond", &comparison.inputs[1]), &[1]);
        for (region, graph) in &graphs {
            for id in graph.all_node_ids() {
                let node = graph.node(id).unwrap();
                if let Some(tensor) = items(region)
                    .tensors
                    .iter()
                    .find(|tensor| tensor.name == node.name)
                {
                    assert_eq!(
                        visitors::actual_tensor_shape(graph, &store, node),
                        tensor.shape,
                        "{}",
                        node.name
                    );
                }
            }
        }
    }

    #[test]
    fn cond_and_while_regions_serialize_multi_axis_flip_and_roll() {
        use tosa_types::Op;
        let config = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+cf");
        for pipeline in [
            build_ir_cond_submodule_pipeline(&config),
            build_ir_while_submodule_pipeline(&config),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [2, 3], "torch.int8");
            let qp = qparam(0.25, 3, "torch.int8");
            graph.node_mut(x).meta.set_both_qparams(qp.clone());
            let flip = call(
                &mut graph,
                "flip",
                "aten.flip.default",
                vec![IrArg::Node(x), i64_list_arg(&[0, 1])],
                [2, 3],
                "torch.int8",
            );
            graph.node_mut(flip).meta.set_both_qparams(qp.clone());
            let roll = call(
                &mut graph,
                "roll",
                "aten.roll.default",
                vec![
                    IrArg::Node(flip),
                    i64_list_arg(&[1, -1]),
                    i64_list_arg(&[0, 1]),
                ],
                [2, 3],
                "torch.int8",
            );
            graph.node_mut(roll).meta.set_both_qparams(qp.clone());
            output(&mut graph, roll);

            let items = lower_region(&mut graph, pipeline);
            let ops: Vec<_> = items.operators.iter().map(|op| op.op).collect();
            assert_eq!(ops.iter().filter(|&&op| op == Op::Reverse).count(), 2);
            assert_eq!(ops.iter().filter(|&&op| op == Op::Slice).count(), 4);
            assert_eq!(ops.iter().filter(|&&op| op == Op::Concat).count(), 2);
            assert!(!ops.contains(&Op::Rescale));
            let result = graph.output_node().unwrap().first_node_arg().unwrap();
            assert!(graph
                .node(result)
                .meta
                .output_qparam()
                .unwrap()
                .same_exact_affine_dtype(&qp));
            let output = items
                .tensors
                .iter()
                .find(|tensor| items.output_names.contains(&tensor.name))
                .unwrap();
            assert_eq!(output.shape, vec![2, 3]);
            assert_eq!(output.dtype, tosa_types::DType::Int8);
        }
    }

    #[test]
    fn cond_and_while_regions_lower_integer_div_and_rank3_max_pool() {
        use tosa_types::Op;
        let config = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+cf");
        for pipeline in [
            build_ir_cond_submodule_pipeline(&config),
            build_ir_while_submodule_pipeline(&config),
        ] {
            let mut graph = IrGraph::empty();
            let lhs = placeholder(&mut graph, "lhs", [2, 3], "torch.int32");
            let rhs = placeholder(&mut graph, "rhs", [2, 3], "torch.int32");
            let div = call(
                &mut graph,
                "div",
                "aten.div.Tensor_mode",
                vec![
                    IrArg::Node(lhs),
                    IrArg::Node(rhs),
                    IrArg::Str("floor".into()),
                ],
                [2, 3],
                "torch.int32",
            );
            let x = placeholder(&mut graph, "x", [3, 8, 8], "torch.int8");
            let pool = call(
                &mut graph,
                "pool",
                "aten.max_pool2d.default",
                vec![IrArg::Node(x), i64_list_arg(&[2, 2]), i64_list_arg(&[2, 2])],
                [3, 4, 4],
                "torch.int8",
            );
            let out = output(&mut graph, div);
            graph.set_args(
                out,
                vec![IrArg::List(vec![IrArg::Node(div), IrArg::Node(pool)])],
            );

            let items = lower_region(&mut graph, pipeline);
            assert_eq!(
                items
                    .operators
                    .iter()
                    .filter(|op| op.op == Op::IntDiv)
                    .count(),
                1
            );
            let pool = items
                .operators
                .iter()
                .find(|op| op.op == Op::MaxPool2d)
                .unwrap();
            for name in [&pool.inputs[0], &pool.outputs[0]] {
                let tensor = items
                    .tensors
                    .iter()
                    .find(|tensor| &tensor.name == name)
                    .unwrap();
                assert_eq!(tensor.shape.len(), 4);
                assert_eq!(tensor.dtype, tosa_types::DType::Int8);
            }
            assert_eq!(items.output_names.len(), 2);
        }
    }

    #[test]
    fn rejects_floating_constants_and_runtime_inputs_in_int_regions() {
        for (dtype, name) in [
            (DType::Float32, "FP32"),
            (DType::Float16, "FP16"),
            (DType::BFloat16, "BF16"),
            (DType::Float8E4M3, "FP8E4M3"),
            (DType::Float8E5M2, "FP8E5M2"),
        ] {
            for constant in [false, true] {
                let mut graph = Graph::new();
                graph.create_node(
                    OpType::Placeholder,
                    OpTarget::Other("x".into()),
                    smallvec![],
                    Vec::new(),
                    NodeMeta::new(TensorInfo::new(&[1], dtype)),
                    "x".into(),
                );
                graph.create_node(
                    OpType::Placeholder,
                    OpTarget::Other("q".into()),
                    smallvec![],
                    Vec::new(),
                    NodeMeta::new(TensorInfo::new(&[1], DType::Int8)),
                    "q".into(),
                );
                let mut store = TensorStore::new();
                if constant {
                    store.insert(
                        "x".into(),
                        TensorData {
                            data: vec![0; dtype.storage_size_bytes()],
                            shape: vec![1],
                            dtype,
                        },
                    );
                }
                assert_eq!(
                    collect_graph_items("branch", &graph, &store).err(),
                    Some(format!(
                        "TOSA INT profile does not support {name} tensor 'x' in region 'branch'"
                    )),
                );
            }
        }
    }

    #[test]
    fn leaves_pure_float_regions_alone() {
        let mut graph = Graph::new();
        graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("x".into()),
            smallvec![],
            Vec::new(),
            NodeMeta::new(TensorInfo::new(&[1], DType::Float32)),
            "x".into(),
        );
        assert!(collect_graph_items("main", &graph, &TensorStore::new()).is_ok());
    }

    #[test]
    fn rejects_floating_intermediates_generated_by_emitters() {
        let mut graph = Graph::new();
        let input = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("x".into()),
            smallvec![],
            Vec::new(),
            NodeMeta::new(TensorInfo::new(&[2], DType::Int32)),
            "x".into(),
        );
        // An INT8 runtime input marks the region as quantized.
        graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("q".into()),
            smallvec![],
            Vec::new(),
            NodeMeta::new(TensorInfo::new(&[2], DType::Int8)),
            "q".into(),
        );
        graph.create_node(
            OpType::CallFunction,
            OpTarget::DequantizePerTensor,
            smallvec![Arg::Node(input), Arg::Float(0.5), Arg::Int(0)],
            Vec::new(),
            NodeMeta::new(TensorInfo::new(&[2], DType::Float32)),
            "dq".into(),
        );
        let error = collect_graph_items("main", &graph, &TensorStore::new())
            .err()
            .unwrap();
        assert!(error.contains("TOSA INT profile does not support FP32 tensor 'dq_cast'"));
    }

    #[test]
    fn argmax_consumers_match_emitted_scalar_ranks() {
        use tosa_types::Op;

        for (cast, scalar_shape) in [
            (false, vec![]),
            (true, vec![]),
            (false, vec![1]),
            (true, vec![1]),
            (false, vec![3]),
            (true, vec![3]),
        ] {
            visitors::shape_utils::clear_shape_cache();
            visitors::dtype_quant::clear_dtype_cache();
            let mut graph = Graph::new();
            let x = graph.create_node(
                OpType::Placeholder,
                OpTarget::Other("x".into()),
                smallvec![],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&[5], DType::Int8)),
                "x".into(),
            );
            let (args, kwargs) = if cast {
                (smallvec![Arg::Node(x), Arg::Int(0)], Vec::new())
            } else {
                (smallvec![Arg::Node(x)], vec![("dim".into(), Arg::Int(-1))])
            };
            let argmax = graph.create_node(
                OpType::CallFunction,
                OpTarget::ArgMax,
                args,
                kwargs,
                NodeMeta::new(TensorInfo::new(&[], DType::Int32)),
                "argmax".into(),
            );
            let input = if cast {
                graph.create_node(
                    OpType::CallFunction,
                    OpTarget::To,
                    smallvec![Arg::Node(argmax)],
                    Vec::new(),
                    NodeMeta::new(TensorInfo::new(&[], DType::Int32)),
                    "cast".into(),
                )
            } else {
                argmax
            };
            let scalar = graph.create_node(
                OpType::Placeholder,
                OpTarget::Other("scalar".into()),
                smallvec![],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&scalar_shape, DType::Int32)),
                "scalar".into(),
            );
            let add = graph.create_node(
                OpType::CallFunction,
                OpTarget::Add,
                smallvec![Arg::Node(input), Arg::Node(scalar)],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&scalar_shape, DType::Int32)),
                "add".into(),
            );
            let reshape = graph.create_node(
                OpType::CallFunction,
                OpTarget::Reshape,
                smallvec![Arg::Node(input), Arg::IntList(smallvec![1])],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
                "reshape".into(),
            );
            graph.create_node(
                OpType::Output,
                OpTarget::Other("output".into()),
                smallvec![Arg::NodeList(smallvec![add, reshape])],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&[], DType::Int32)),
                "output".into(),
            );
            let store = TensorStore::new();
            assert!(
                visitors::actual_tensor_shape(&graph, &store, graph.node(argmax).unwrap())
                    .is_empty()
            );
            assert!(
                visitors::actual_tensor_shape(&graph, &store, graph.node(input).unwrap())
                    .is_empty()
            );
            let items = collect_graph_items("main", &graph, &TensorStore::new()).unwrap();
            let shape = |name: &str| {
                &items
                    .tensors
                    .iter()
                    .find(|tensor| tensor.name == name)
                    .unwrap()
                    .shape
            };
            assert!(shape("argmax").is_empty());
            let mut declared_shapes = std::collections::HashSet::new();
            for op in &items.operators {
                if op.op == Op::ConstShape {
                    declared_shapes.extend(op.outputs.iter().cloned());
                }
                for input in &op.inputs {
                    if items.shapes.iter().any(|shape| shape.name == *input) {
                        assert!(
                            declared_shapes.contains(input),
                            "shape {input} used before CONST_SHAPE"
                        );
                    }
                }
                if matches!(op.op, Op::Add | Op::Identity) {
                    let rank = shape(&op.outputs[0]).len();
                    for input in &op.inputs {
                        assert_eq!(shape(input).len(), rank, "{:?}: {input}", op.op);
                    }
                }
            }
            assert_eq!(
                items
                    .operators
                    .iter()
                    .filter(|op| op.op == Op::Reshape)
                    .count(),
                1 + usize::from(!scalar_shape.is_empty())
            );
            assert_eq!(
                shape("add"),
                &scalar_shape.iter().map(|&d| d as i32).collect::<Vec<_>>()
            );
            assert_eq!(shape("reshape"), &[1]);
            assert_eq!(items.shapes.len(), 1);
            assert_eq!(items.output_names, vec!["add", "reshape"]);
        }
    }

    #[test]
    fn rejects_argmax_without_an_explicit_dim() {
        for dim in [None, Some(Arg::None)] {
            let mut graph = Graph::new();
            let x = graph.create_node(
                OpType::Placeholder,
                OpTarget::Other("x".into()),
                smallvec![],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&[2, 3], DType::Int8)),
                "x".into(),
            );
            let mut args = smallvec![Arg::Node(x)];
            args.extend(dim);
            graph.create_node(
                OpType::CallFunction,
                OpTarget::ArgMax,
                args,
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&[], DType::Int32)),
                "argmax".into(),
            );
            assert_eq!(
                collect_graph_items("branch", &graph, &TensorStore::new()).err(),
                Some("argmax node 'argmax' in region 'branch' requires an explicit dim".into())
            );
        }
    }

    #[test]
    fn rejects_unknown_and_unsupported_targets_with_their_names() {
        for target in [
            "backend.tosa.UNKNOWN.default",
            "backend.tosa.CONV2D_ADAPTIVE.default",
            "backend.tosa.LOGICAL_LEFT_SHIFT.default",
            "backend.tosa.CONST_SHAPE.out",
            "aten.getitem_renamed.default",
            "renamed.getitem",
            "aten.gather.default",
        ] {
            let mut graph = Graph::new();
            graph.create_node(
                OpType::CallFunction,
                parse_op_target(target),
                smallvec::smallvec![],
                Vec::new(),
                NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
                "unknown".to_string(),
            );
            let result = collect_graph_items("main", &graph, &TensorStore::new());
            assert_eq!(
                result.err(),
                Some(format!(
                    "RETAB has no TOSA emitter for 1 node(s) in region 'main': 'unknown' ({:?})",
                    parse_op_target(target)
                )),
                "{target}"
            );
        }
    }
}
