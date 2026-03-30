use super::*;

pub(in crate::serialize::visitors) fn emit_aten_mean(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let spec = ParsedReduceSpec::parse(graph, node, store, false);
    let mut result = EmitResult::default();

    let mut reduce_count: f64 = 1.0;
    for axis in &spec.axes {
        if let Some(&dim_size) = spec.input_shape.get(axis.tosa_axis as usize) {
            reduce_count *= dim_size as f64;
        }
    }

    let base_dtype = node_output_dtype(node);
    let sum_name = format!("{}_sum", node.name);
    let mut current_input = spec.input_name;
    let mut current_shape = spec.input_shape.clone();

    for (idx, axis) in spec.axes.iter().enumerate() {
        let mut out_shape = current_shape.clone();
        if let Some(slot) = out_shape.get_mut(axis.tosa_axis as usize) {
            *slot = 1;
        }

        let is_last = idx == spec.axes.len() - 1;
        let out_name = if is_last {
            sum_name.clone()
        } else {
            format!("{}_reduce_sum_axis{}", node.name, axis.raw_dim)
        };

        result.tensors.push(TosaTensor::new(
            out_name.clone(),
            out_shape.clone(),
            base_dtype,
        ));
        result.operators.push(TosaOperator {
            op: Op::ReduceSum,
            attribute: Attribute::ReduceSum(ReduceSumAttribute {
                axis: axis.tosa_axis,
            }),
            inputs: vec![current_input.clone()],
            outputs: vec![out_name.clone()],
            location: None,
        });

        current_input = out_name;
        current_shape = out_shape;
    }

    let recip_val = 1.0 / reduce_count;
    let recip_name = format!("{}_mean_recip", node.name);
    let ir_dtype = node.meta.val.dtype;
    if let Some(recip_tensor) =
        make_filled_const_tensor(recip_name.clone(), vec![1], ir_dtype, recip_val)
    {
        push_inline_const(&mut result, recip_tensor);
    }

    let final_name = if spec.keepdim {
        node.name.clone()
    } else {
        format!("{}_mean_pre_reshape", node.name)
    };
    result.tensors.push(TosaTensor::new(
        final_name.clone(),
        current_shape.clone(),
        base_dtype,
    ));

    let shift_name = format!("{}_mul_shift", node.name);
    result.tensors.push(TosaTensor::with_data(
        shift_name.clone(),
        vec![1],
        DType::Int8,
        vec![0u8],
    ));
    result.operators.push(TosaOperator {
        op: Op::Const,
        attribute: Attribute::Const(ConstAttribute),
        inputs: vec![],
        outputs: vec![shift_name.clone()],
        location: None,
    });
    result.operators.push(TosaOperator {
        op: Op::Mul,
        attribute: Attribute::Mul(MulAttribute { shift: 0 }),
        inputs: vec![sum_name, recip_name, shift_name],
        outputs: vec![final_name.clone()],
        location: None,
    });

    if !spec.keepdim {
        let final_shape = actual_tensor_shape(graph, store, node);
        push_reshape_with_shape(
            &mut result,
            final_name,
            node.name.clone(),
            format!("{}_shape", node.name),
            final_shape,
            base_dtype,
        );
    }

    result
}
