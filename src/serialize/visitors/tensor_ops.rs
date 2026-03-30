use super::*;

struct ParsedResizeSpec {
    input_name: String,
    resize_output_name: String,
    final_output_name: String,
    input_shape: Vec<i32>,
    output_shape: Vec<i32>,
    align_corners: bool,
    resize_mode: ResizeMode,
    resize_output_dtype: DType,
    final_output_dtype: Option<DType>,
    bilinear_rescale_scale: Option<f64>,
}

fn gcd_i32(mut lhs: i32, mut rhs: i32) -> i32 {
    lhs = lhs.abs();
    rhs = rhs.abs();
    while rhs != 0 {
        let rem = lhs % rhs;
        lhs = rhs;
        rhs = rem;
    }
    lhs.max(1)
}

fn resize_axis_parameters(
    input_size: i32,
    output_size: i32,
    align_corners: bool,
) -> (i32, i32, i32, i32) {
    let scale_n = if align_corners && input_size > 1 && output_size > 1 {
        output_size - 1
    } else {
        output_size
    };
    let scale_d = if align_corners && input_size > 1 && output_size > 1 {
        input_size - 1
    } else {
        input_size
    };
    let gcd = gcd_i32(scale_n, scale_d);
    let scale_n = 2 * scale_n / gcd;
    let scale_d = 2 * scale_d / gcd;
    let offset = if align_corners {
        0
    } else {
        scale_d / 2 - scale_n / 2
    };
    let border = scale_d * (output_size - 1) - scale_n * (input_size - 1) + offset;
    (scale_n, scale_d, offset, border)
}

fn parse_resize_mode(node: &Node) -> ResizeMode {
    if matches!(node.target, OpTarget::UpsampleBilinear2d) {
        return ResizeMode::Bilinear;
    }
    if matches!(node.target, OpTarget::UpsampleNearest2d) {
        return ResizeMode::Nearest;
    }
    match kwarg_string(node, "resize_mode").as_deref() {
        Some("bilinear") => ResizeMode::Bilinear,
        _ => ResizeMode::Nearest,
    }
}

fn first_input_node<'a>(graph: &'a Graph, node: &Node) -> Option<&'a Node> {
    let Arg::Node(input_id) = node.args.first()? else {
        return None;
    };
    graph.node(*input_id)
}

fn push_rank1_const(
    result: &mut EmitResult,
    name: String,
    dtype: DType,
    data: Vec<u8>,
    len: usize,
) -> String {
    result.push_const(make_rank1_const_tensor(name.clone(), dtype, data, len));
    name
}

fn emit_direct_rescale_params(
    result: &mut EmitResult,
    base_name: &str,
    input_dtype: DType,
    output_dtype: DType,
    scale: f64,
) -> (String, String, String, String, bool) {
    let scale32 = input_dtype != DType::Int48;
    let scale_width = if scale32 { 32 } else { 16 };
    let (multiplier, shift) = compute_multiplier_and_shift_with_width(scale, scale_width);
    let multiplier_name = format!("{base_name}_multipliers");
    let shift_name = format!("{base_name}_shifts");
    let input_zp_name = format!("{base_name}_input_zp");
    let output_zp_name = format!("{base_name}_output_zp");

    if scale32 {
        push_rank1_const(
            result,
            multiplier_name.clone(),
            DType::Int32,
            multiplier.to_le_bytes().to_vec(),
            1,
        );
    } else {
        push_rank1_const(
            result,
            multiplier_name.clone(),
            DType::Int16,
            (multiplier as i16).to_le_bytes().to_vec(),
            1,
        );
    }
    push_rank1_const(
        result,
        shift_name.clone(),
        DType::Int8,
        vec![shift as u8],
        1,
    );
    push_rank1_const(
        result,
        input_zp_name.clone(),
        input_dtype,
        encode_zp_for_dtype(0, input_dtype),
        1,
    );
    push_rank1_const(
        result,
        output_zp_name.clone(),
        output_dtype,
        encode_zp_for_dtype(0, output_dtype),
        1,
    );

    (
        multiplier_name,
        shift_name,
        input_zp_name,
        output_zp_name,
        scale32,
    )
}

impl ParsedResizeSpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Option<Self> {
        let input_name = collect_input_names(graph, node).first()?.clone();
        let input_node = first_input_node(graph, node)?;
        let input_shape = actual_tensor_shape(graph, store, input_node);
        let output_shape = actual_tensor_shape(graph, store, node);
        if input_shape.len() != 4 || output_shape.len() != 4 {
            return None;
        }

        let resize_mode = parse_resize_mode(node);
        let align_corners = matches!(resize_mode, ResizeMode::Bilinear)
            && arg_bool(node, 2, kwarg_bool(node, "align_corners", false));
        let input_dtype = actual_tensor_dtype(graph, input_node);
        let final_output_name = primary_output_name(graph, node);
        let final_output_dtype = node_output_dtype(node);
        let needs_bilinear_rescale = matches!(node.target, OpTarget::UpsampleBilinear2d)
            && matches!(input_dtype, DType::Int8 | DType::Int16);

        if needs_bilinear_rescale {
            let (scale_n_y, _, _, _) =
                resize_axis_parameters(input_shape[1], output_shape[1], align_corners);
            let (scale_n_x, _, _, _) =
                resize_axis_parameters(input_shape[2], output_shape[2], align_corners);

            return Some(Self {
                input_name,
                resize_output_name: format!("{final_output_name}_resize"),
                final_output_name,
                input_shape,
                output_shape,
                align_corners,
                resize_mode,
                resize_output_dtype: if input_dtype == DType::Int16 {
                    DType::Int48
                } else {
                    DType::Int32
                },
                final_output_dtype: Some(final_output_dtype),
                bilinear_rescale_scale: Some(1.0 / f64::from(scale_n_y * scale_n_x)),
            });
        }

        Some(Self {
            input_name,
            resize_output_name: final_output_name.clone(),
            final_output_name,
            input_shape,
            output_shape,
            align_corners,
            resize_mode,
            resize_output_dtype: final_output_dtype,
            final_output_dtype: None,
            bilinear_rescale_scale: None,
        })
    }
}

struct ParsedShapeTransformSpec {
    input_names: Vec<String>,
    output_shape: Vec<i32>,
    output_name: String,
    output_dtype: DType,
}

impl ParsedShapeTransformSpec {
    // Parse once at the shape-op boundary so emitters can work from a typed
    // representation of the target tensor instead of re-deriving it ad hoc.
    fn parse(graph: &Graph, store: &TensorStore, node: &Node) -> Self {
        Self {
            input_names: collect_input_names(graph, node),
            output_shape: actual_tensor_shape(graph, store, node),
            output_name: primary_output_name(graph, node),
            output_dtype: passthrough_input_dtype(graph, node),
        }
    }
}

pub(super) fn emit_full(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let output_shape = actual_tensor_shape(graph, store, node);
    let fill_value = node.args.get(1).map_or(0.0, Arg::as_f64);
    let dtype = node.meta.val.dtype;
    let tensor = make_filled_const_tensor(node.name.clone(), output_shape, dtype, fill_value)
        .unwrap_or_else(|| {
            panic!(
                "emit_full: unsupported dtype {:?} for node '{}'",
                dtype, node.name
            )
        });
    let mut result = EmitResult::default();
    push_inline_const(&mut result, tensor);
    result
}

pub(super) fn emit_scalar_tensor(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut output_shape = actual_tensor_shape(graph, store, node);
    if output_shape.is_empty() {
        // TOSA scalar constants are represented as rank-1 [1] tensors and then
        // broadcast by consumers such as ADD/WHERE.
        output_shape.push(1);
    }
    let fill_value = node.args.first().map_or(0.0, Arg::as_f64);
    let fill_value = match (node.meta.val.dtype, node.meta.output_qparams.first()) {
        (crate::graph::ir::DType::Int8 | crate::graph::ir::DType::Int16, Some((_, qp))) => {
            let quantized = ((fill_value / qp.scale) + qp.zero_point as f64).round() as i64;
            quantized.clamp(qp.qmin, qp.qmax) as f64
        }
        _ => fill_value,
    };
    let dtype = node.meta.val.dtype;
    let tensor = make_filled_const_tensor(node.name.clone(), output_shape, dtype, fill_value)
        .unwrap_or_else(|| {
            panic!(
                "emit_scalar_tensor: unsupported dtype {:?} for node '{}'",
                dtype, node.name
            )
        });
    let mut result = EmitResult::default();
    push_inline_const(&mut result, tensor);
    result
}

pub(super) fn emit_resize(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let Some(spec) = ParsedResizeSpec::parse(graph, node, store) else {
        return EmitResult::default();
    };

    let (scale_n_y, scale_d_y, offset_y, border_y) = resize_axis_parameters(
        spec.input_shape[1],
        spec.output_shape[1],
        spec.align_corners,
    );
    let (scale_n_x, scale_d_x, offset_x, border_x) = resize_axis_parameters(
        spec.input_shape[2],
        spec.output_shape[2],
        spec.align_corners,
    );

    let mut result = EmitResult::default();
    let scales_name = result.push_const(make_shape_const_tensor(
        format!("{}_scales", spec.final_output_name),
        vec![scale_n_y, scale_d_y, scale_n_x, scale_d_x],
    ));
    let offset_name = result.push_const(make_shape_const_tensor(
        format!("{}_offset", spec.final_output_name),
        vec![offset_y, offset_x],
    ));
    let border_name = result.push_const(make_shape_const_tensor(
        format!("{}_border", spec.final_output_name),
        vec![border_y, border_x],
    ));

    result.push_output_op(
        make_empty_tensor(
            spec.resize_output_name.clone(),
            spec.output_shape.clone(),
            spec.resize_output_dtype,
        ),
        Op::Resize,
        Attribute::Resize(ResizeAttribute {
            mode: spec.resize_mode,
        }),
        vec![spec.input_name, scales_name, offset_name, border_name],
    );

    let Some(final_output_dtype) = spec.final_output_dtype else {
        return result;
    };
    let Some(scale) = spec.bilinear_rescale_scale else {
        return result;
    };
    let (multiplier_name, shift_name, input_zp_name, output_zp_name, scale32) =
        emit_direct_rescale_params(
            &mut result,
            &spec.final_output_name,
            spec.resize_output_dtype,
            final_output_dtype,
            scale,
        );
    result.push_output_op(
        make_empty_tensor(
            spec.final_output_name,
            spec.output_shape,
            final_output_dtype,
        ),
        Op::Rescale,
        Attribute::Rescale(RescaleAttribute {
            scale32,
            rounding_mode: RoundingMode::SingleRound,
            per_channel: false,
            input_unsigned: false,
            output_unsigned: false,
        }),
        vec![
            spec.resize_output_name,
            multiplier_name,
            shift_name,
            input_zp_name,
            output_zp_name,
        ],
    );
    result
}

pub(super) fn emit_aten_reshape(
    graph: &Graph,
    node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    emit_reshape(graph, node_id, node, store)
}

pub(super) fn emit_aten_permute(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let spec = ParsedShapeTransformSpec::parse(graph, store, node);
    let perms = arg_i32_list(node, 1);
    result.tensors.push(make_empty_tensor(
        spec.output_name.clone(),
        spec.output_shape,
        spec.output_dtype,
    ));
    result.operators.push(TosaOperator::new(
        Op::Transpose,
        Attribute::Transpose(TransposeAttribute { perms }),
        spec.input_names,
        vec![spec.output_name],
    ));
    result
}

fn emit_tile_like(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    multiples: Vec<i32>,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let Some(input_name) = input_names.first().cloned() else {
        return result;
    };
    let Some(Arg::Node(input_id)) = node.args.first() else {
        return result;
    };
    let Some(input_node) = graph.node(*input_id) else {
        return result;
    };

    let input_shape = actual_tensor_shape(graph, store, input_node);
    let output_shape = actual_tensor_shape(graph, store, node);
    if output_shape.is_empty() || multiples.is_empty() {
        return result;
    }

    let tile_rank = multiples.len();
    let mut tile_input_name = input_name;
    if input_shape.len() < tile_rank {
        let mut padded_input_shape = vec![1; tile_rank - input_shape.len()];
        padded_input_shape.extend(input_shape.iter().copied());

        let reshaped_name = format!("{}_input_reshape", node.name);
        let reshape_shape_name = format!("{}_input_shape", node.name);
        push_const_shape(
            &mut result,
            reshape_shape_name.clone(),
            padded_input_shape.clone(),
        );
        result.tensors.push(make_empty_tensor(
            reshaped_name.clone(),
            padded_input_shape.clone(),
            actual_tensor_dtype(graph, input_node),
        ));
        result.operators.push(TosaOperator::new(
            Op::Reshape,
            Attribute::Reshape(ReshapeAttribute),
            vec![tile_input_name, reshape_shape_name],
            vec![reshaped_name.clone()],
        ));
        tile_input_name = reshaped_name;

        if let Some(perms) = node
            .meta
            .tosa_dim_order
            .as_ref()
            .map(|order| order.iter().map(|&dim| dim as i32).collect::<Vec<_>>())
        {
            let is_non_identity = perms
                .iter()
                .enumerate()
                .any(|(idx, &dim)| dim as usize != idx);
            let transposed_shape: Vec<i32> = perms
                .iter()
                .filter_map(|&idx| padded_input_shape.get(idx as usize).copied())
                .collect();
            if is_non_identity && transposed_shape.len() == padded_input_shape.len() {
                let transposed_name = format!("{}_input_layout", node.name);
                result.tensors.push(make_empty_tensor(
                    transposed_name.clone(),
                    transposed_shape,
                    actual_tensor_dtype(graph, input_node),
                ));
                result.operators.push(TosaOperator::new(
                    Op::Transpose,
                    Attribute::Transpose(TransposeAttribute { perms }),
                    vec![tile_input_name],
                    vec![transposed_name.clone()],
                ));
                tile_input_name = transposed_name;
            }
        }
    }

    let multiples_name = format!("{}_multiples", node.name);
    push_inline_const(
        &mut result,
        make_shape_const_tensor(multiples_name.clone(), multiples),
    );
    result.tensors.push(make_empty_tensor(
        primary_output_name(graph, node),
        output_shape,
        actual_tensor_dtype(graph, input_node),
    ));
    result.operators.push(TosaOperator::new(
        Op::Tile,
        Attribute::None,
        vec![tile_input_name, multiples_name],
        vec![primary_output_name(graph, node)],
    ));
    result
}

pub(super) fn emit_expand(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let Some(Arg::Node(input_id)) = node.args.first() else {
        return EmitResult::default();
    };
    let Some(input_node) = graph.node(*input_id) else {
        return EmitResult::default();
    };
    let input_shape = actual_tensor_shape(graph, store, input_node);
    let output_shape = actual_tensor_shape(graph, store, node);
    if output_shape.is_empty() {
        return EmitResult::default();
    }

    let padding = output_shape.len().saturating_sub(input_shape.len());
    let mut padded_input_shape = vec![1; padding];
    padded_input_shape.extend(input_shape.iter().copied());
    let multiples: Vec<i32> = padded_input_shape
        .iter()
        .zip(output_shape.iter())
        .map(|(&input_dim, &output_dim)| {
            if input_dim == output_dim {
                1
            } else if input_dim == 1 && output_dim >= 1 {
                output_dim
            } else {
                1
            }
        })
        .collect();
    emit_tile_like(graph, node, store, multiples)
}

pub(super) fn emit_repeat(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let multiples = apply_tosa_dim_order(node, arg_i32_list(node, 1));
    emit_tile_like(graph, node, store, multiples)
}

pub(super) fn emit_const(
    _graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let name = &node.name;
    let (shape, dtype, data) = if let Some(tensor_data) = store.get(name) {
        (
            tensor_data.shape.iter().map(|&d| d as i32).collect(),
            ir_dtype_to_tosa(tensor_data.dtype),
            tensor_data.data.clone(),
        )
    } else {
        (
            raw_i32_shape(node),
            ir_dtype_to_tosa(node.meta.val.dtype),
            Vec::new(),
        )
    };

    result
        .tensors
        .push(make_tensor(name.clone(), shape, dtype, data));
    result.operators.push(TosaOperator::new(
        Op::Const,
        Attribute::Const(ConstAttribute),
        vec![],
        vec![name.clone()],
    ));
    result
}

pub(super) fn emit_transpose(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let perms = {
        let perms = arg_i32_list(node, 1);
        if perms.is_empty() {
            kwarg_i32_list(node, "perms")
        } else {
            perms
        }
    };
    let mut output_shape = raw_i32_shape(node);
    if let Some(Arg::Node(input_id)) = node.args.first() {
        if let Some(input_node) = graph.node(*input_id) {
            let input_shape = actual_tensor_shape(graph, store, input_node);
            if input_shape.len() == perms.len() {
                let candidate: Vec<i32> = perms
                    .iter()
                    .filter_map(|&idx| input_shape.get(idx as usize).copied())
                    .collect();
                if candidate.len() == input_shape.len() {
                    output_shape = candidate;
                }
            }
        }
    }
    result.tensors.push(make_empty_tensor(
        primary_output_name(graph, node),
        output_shape,
        passthrough_input_dtype(graph, node),
    ));
    result.operators.push(TosaOperator::new(
        Op::Transpose,
        Attribute::Transpose(TransposeAttribute { perms }),
        input_names,
        vec![primary_output_name(graph, node)],
    ));
    result
}

pub(super) fn emit_reshape(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let out_shape = actual_tensor_shape(graph, store, node);
    result.tensors.push(make_empty_tensor(
        primary_output_name(graph, node),
        out_shape.clone(),
        passthrough_input_dtype(graph, node),
    ));

    let shape_name = format!("{}_shape", node.name);
    let shape_data: Vec<u8> = out_shape
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();
    result.shapes.push(TosaShape {
        name: shape_name.clone(),
        rank: out_shape.len() as u32,
        data: shape_data,
    });
    result.operators.push(TosaOperator::new(
        Op::ConstShape,
        Attribute::ConstShape(ConstShapeAttribute),
        vec![],
        vec![shape_name.clone()],
    ));

    let mut reshape_inputs = input_names;
    reshape_inputs.push(shape_name);
    result.operators.push(TosaOperator::new(
        Op::Reshape,
        Attribute::Reshape(ReshapeAttribute),
        reshape_inputs,
        vec![primary_output_name(graph, node)],
    ));
    result
}

pub(super) fn emit_concat(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    _store: &TensorStore,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let raw_axis = match node.args.get(1) {
        Some(Arg::Int(axis)) => *axis as i32,
        _ => kwarg_i32(node, "axis", 0),
    };
    let axis = match node.args.first() {
        Some(Arg::Node(input_id)) => graph
            .node(*input_id)
            .map(|input_node| map_axis_to_tosa_dim_order(input_node, raw_axis))
            .unwrap_or(raw_axis),
        Some(Arg::NodeList(ids)) => ids
            .first()
            .and_then(|input_id| graph.node(*input_id))
            .map(|input_node| map_axis_to_tosa_dim_order(input_node, raw_axis))
            .unwrap_or(raw_axis),
        _ => raw_axis,
    };
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator::new(
        Op::Concat,
        Attribute::Concat(ConcatAttribute { axis }),
        input_names,
        vec![node.name.clone()],
    ));
    result
}
