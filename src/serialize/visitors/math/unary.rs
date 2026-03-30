use super::*;

pub(in crate::serialize::visitors) fn emit_negate(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let input_dtype = passthrough_input_dtype(graph, node);
    result.tensors.push(make_output_tensor(node));

    let (input_zp, output_zp) = if input_dtype == DType::Int8 {
        let input_zp = node
            .meta
            .input_qparams
            .iter()
            .find(|(pos, _)| *pos == 0)
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or(0);
        let output_zp = node
            .meta
            .output_qparams
            .first()
            .map(|(_, qp)| qp.zero_point as i32)
            .unwrap_or(0);
        (input_zp, output_zp)
    } else {
        (0, 0)
    };

    let input_zp_name = format!("{}_input_zp", node.name);
    let output_zp_name = format!("{}_output_zp", node.name);
    for (name, value) in [
        (input_zp_name.clone(), input_zp),
        (output_zp_name.clone(), output_zp),
    ] {
        result.tensors.push(TosaTensor {
            name: name.clone(),
            shape: vec![1],
            dtype: input_dtype,
            data: encode_zp_for_dtype(value, input_dtype),
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });
        result.operators.push(TosaOperator {
            op: Op::Const,
            attribute: Attribute::Const(ConstAttribute),
            inputs: vec![],
            outputs: vec![name],
            location: None,
        });
    }

    let mut negate_inputs = input_names;
    negate_inputs.push(input_zp_name);
    negate_inputs.push(output_zp_name);
    result.operators.push(TosaOperator {
        op: Op::Negate,
        attribute: Attribute::Negate(NegateAttribute),
        inputs: negate_inputs,
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}

pub(in crate::serialize::visitors) fn emit_ceil(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let input_dtype = passthrough_input_dtype(graph, node);
    if is_integral_dtype(input_dtype) {
        emit_identity(graph, node)
    } else {
        let mut result = EmitResult::default();
        let input_names = collect_input_names(graph, node);
        result.tensors.push(make_output_tensor(node));
        result.operators.push(TosaOperator {
            op: Op::Ceil,
            attribute: Attribute::Ceil(CeilAttribute),
            inputs: input_names,
            outputs: vec![node.name.clone()],
            location: None,
        });
        result
    }
}

pub(in crate::serialize::visitors) fn emit_floor(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let input_dtype = passthrough_input_dtype(graph, node);
    if is_integral_dtype(input_dtype) {
        emit_identity(graph, node)
    } else {
        let mut result = EmitResult::default();
        let input_names = collect_input_names(graph, node);
        result.tensors.push(make_output_tensor(node));
        result.operators.push(TosaOperator {
            op: Op::Floor,
            attribute: Attribute::Floor(FloorAttribute),
            inputs: input_names,
            outputs: vec![node.name.clone()],
            location: None,
        });
        result
    }
}

pub(in crate::serialize::visitors) fn emit_round(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let input_dtype = passthrough_input_dtype(graph, node);
    if is_integral_dtype(input_dtype) {
        return emit_identity(graph, node);
    }

    let mut result = EmitResult::default();
    let spec = ParsedUnaryTensorSpec::parse(graph, store, node)
        .unwrap_or_else(|| panic!("emit_round: missing input for node '{}'", node.name));

    let zero_name = format!("{}_round_zero", node.name);
    let half_name = format!("{}_round_half", node.name);
    let ge_name = format!("{}_round_ge", node.name);
    let plus_half_name = format!("{}_round_plus_half", node.name);
    let minus_half_name = format!("{}_round_minus_half", node.name);
    let floor_name = format!("{}_round_floor", node.name);
    let ceil_name = format!("{}_round_ceil", node.name);

    let zero_tensor = make_filled_const_tensor(
        zero_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        0.0,
    )
    .unwrap_or_else(|| {
        panic!(
            "emit_round: unsupported zero-fill dtype {:?} for node '{}'",
            spec.ir_dtype, node.name
        )
    });
    let half_tensor = make_filled_const_tensor(
        half_name.clone(),
        spec.output_shape.clone(),
        spec.ir_dtype,
        0.5,
    )
    .unwrap_or_else(|| {
        panic!(
            "emit_round: unsupported half-fill dtype {:?} for node '{}'",
            spec.ir_dtype, node.name
        )
    });
    push_inline_const(&mut result, zero_tensor);
    push_inline_const(&mut result, half_tensor);

    for tensor_name in [
        ge_name.clone(),
        plus_half_name.clone(),
        minus_half_name.clone(),
    ] {
        let is_bool = tensor_name == ge_name;
        result.tensors.push(TosaTensor {
            name: tensor_name,
            shape: spec.output_shape.clone(),
            dtype: if is_bool {
                DType::Bool
            } else {
                ir_dtype_to_tosa(spec.ir_dtype)
            },
            data: Vec::new(),
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });
    }

    result.operators.push(TosaOperator {
        op: Op::GreaterEqual,
        attribute: Attribute::GreaterEqual(GreaterEqualAttribute),
        inputs: vec![spec.input_name.clone(), zero_name],
        outputs: vec![ge_name.clone()],
        location: None,
    });
    result.operators.push(TosaOperator {
        op: Op::Add,
        attribute: Attribute::Add(AddAttribute),
        inputs: vec![spec.input_name.clone(), half_name.clone()],
        outputs: vec![plus_half_name.clone()],
        location: None,
    });
    result.operators.push(TosaOperator {
        op: Op::Sub,
        attribute: Attribute::Sub(SubAttribute),
        inputs: vec![spec.input_name.clone(), half_name],
        outputs: vec![minus_half_name.clone()],
        location: None,
    });

    for (tensor_name, op, input) in [
        (floor_name.clone(), Op::Floor, plus_half_name),
        (ceil_name.clone(), Op::Ceil, minus_half_name),
    ] {
        result.tensors.push(TosaTensor {
            name: tensor_name.clone(),
            shape: spec.output_shape.clone(),
            dtype: ir_dtype_to_tosa(spec.ir_dtype),
            data: Vec::new(),
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        });
        result.operators.push(TosaOperator {
            op,
            attribute: if op == Op::Floor {
                Attribute::Floor(FloorAttribute)
            } else {
                Attribute::Ceil(CeilAttribute)
            },
            inputs: vec![input],
            outputs: vec![tensor_name],
            location: None,
        });
    }

    result.tensors.push(TosaTensor {
        name: node.name.clone(),
        shape: spec.output_shape,
        dtype: ir_dtype_to_tosa(spec.ir_dtype),
        data: Vec::new(),
        variable: false,
        is_unranked: false,
        variable_name: String::new(),
    });
    result.operators.push(TosaOperator {
        op: Op::Select,
        attribute: Attribute::Select(SelectAttribute),
        inputs: vec![ge_name, floor_name, ceil_name],
        outputs: vec![node.name.clone()],
        location: None,
    });
    result
}
