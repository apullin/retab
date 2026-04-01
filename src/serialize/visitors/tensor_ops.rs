use super::emitter::EmitContext;
use super::*;

pub fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaConst, emit_const as EmitFn),
        (OpTarget::Full, emit_full),
        (OpTarget::Copy, emit_copy),
        (OpTarget::Eye, emit_eye),
        (OpTarget::ScalarTensor, emit_scalar_tensor),
        (OpTarget::TosaResize, emit_resize),
        (OpTarget::TosaTranspose, emit_transpose),
        (OpTarget::TosaReshape, emit_reshape),
        (OpTarget::TosaConcat, emit_concat),
        (OpTarget::Expand, emit_expand),
        (OpTarget::Repeat, emit_repeat),
        (OpTarget::DequantizePerTensor, emit_dequantize_per_tensor),
        (OpTarget::Reshape, emit_aten_reshape),
        (OpTarget::Flatten, emit_aten_reshape),
        (OpTarget::View, emit_aten_reshape),
        (OpTarget::Permute, emit_aten_permute),
        (OpTarget::UpsampleNearest2d, emit_resize),
        (OpTarget::UpsampleBilinear2d, emit_resize),
        (OpTarget::Cat, emit_concat),
        (OpTarget::Zeros, emit_full),
    ]);
}

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

pub(super) fn emit_dequantize_per_tensor(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let Some(input_name) = collect_input_names(graph, node).first().cloned() else {
        return result;
    };

    let output_shape = actual_tensor_shape(graph, ctx.store, node);
    let ir_output_dtype = node.meta.val.dtype;
    let output_dtype = match ir_output_dtype {
        crate::graph::ir::DType::Float16 => DType::Fp16,
        crate::graph::ir::DType::BFloat16 => DType::Bf16,
        crate::graph::ir::DType::Float32 => DType::Fp32,
        _ => DType::Fp32,
    };
    let scale = node.args.get(1).map_or(1.0, Arg::as_f64);
    let zero_point = node.args.get(2).and_then(Arg::as_i64).unwrap_or(0) as f64;

    let cast_output_name = if zero_point == 0.0 && scale == 1.0 {
        node.name.clone()
    } else {
        format!("{}_cast", node.name)
    };
    result.push_output_op(
        make_empty_tensor(cast_output_name.clone(), output_shape.clone(), output_dtype),
        Op::Cast,
        Attribute::Cast(CastAttribute),
        vec![input_name],
    );

    let mut current_name = cast_output_name;
    if zero_point != 0.0 {
        let zp_name = format!("{}_zp", node.name);
        if let Some(zp_tensor) = make_filled_const_tensor(
            zp_name.clone(),
            output_shape.clone(),
            match output_dtype {
                DType::Fp16 => crate::graph::ir::DType::Float16,
                DType::Bf16 => crate::graph::ir::DType::BFloat16,
                _ => crate::graph::ir::DType::Float32,
            },
            zero_point,
        ) {
            push_inline_const(&mut result, zp_tensor);
            let sub_output_name = if scale == 1.0 {
                node.name.clone()
            } else {
                format!("{}_sub", node.name)
            };
            result.push_output_op(
                make_empty_tensor(sub_output_name.clone(), output_shape.clone(), output_dtype),
                Op::Sub,
                Attribute::Sub(SubAttribute),
                vec![current_name, zp_name],
            );
            current_name = sub_output_name;
        }
    }

    if scale != 1.0 {
        let scale_name = format!("{}_scale", node.name);
        if let Some(scale_tensor) = make_filled_const_tensor(
            scale_name.clone(),
            output_shape.clone(),
            match output_dtype {
                DType::Fp16 => crate::graph::ir::DType::Float16,
                DType::Bf16 => crate::graph::ir::DType::BFloat16,
                _ => crate::graph::ir::DType::Float32,
            },
            scale,
        ) {
            push_inline_const(&mut result, scale_tensor);
            let shift_name = push_shift_const(&mut result, &node.name);
            result.push_output_op(
                make_output_tensor(node),
                Op::Mul,
                Attribute::Mul(MulAttribute { shift: 0 }),
                vec![current_name, scale_name, shift_name],
            );
        }
    }

    result
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
    fn parse(graph: &Graph, store: &TensorStore, node: &Node) -> Self {
        Self {
            input_names: collect_input_names(graph, node),
            output_shape: actual_tensor_shape(graph, store, node),
            output_name: primary_output_name(graph, node),
            output_dtype: passthrough_input_dtype(graph, node),
        }
    }
}

fn emit_filled_const(
    node: &Node,
    output_shape: Vec<i32>,
    fill_value: f64,
    context: &str,
) -> EmitResult {
    let dtype = node.meta.val.dtype;
    let fill_value = quantize_fill_value(node, fill_value);
    let tensor = make_filled_const_tensor(node.name.clone(), output_shape, dtype, fill_value)
        .unwrap_or_else(|| {
            panic!(
                "{context}: unsupported dtype {:?} for node '{}'",
                dtype, node.name
            )
        });
    let mut result = EmitResult::default();
    push_inline_const(&mut result, tensor);
    result
}

fn emit_shape_transform(
    spec: ParsedShapeTransformSpec,
    op: Op,
    attribute: Attribute,
) -> EmitResult {
    let mut result = EmitResult::default();
    result.push_output_op(
        make_empty_tensor(spec.output_name, spec.output_shape, spec.output_dtype),
        op,
        attribute,
        spec.input_names,
    );
    result
}

pub(super) fn emit_full(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let output_shape = actual_tensor_shape(graph, store, node);
    let fill_value = node.args.get(1).map_or(0.0, Arg::as_f64);
    emit_filled_const(node, output_shape, fill_value, "emit_full")
}

pub(super) fn emit_copy(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut inputs = collect_input_names(graph, node);
    let source = inputs
        .pop()
        .unwrap_or_else(|| panic!("emit_copy: missing source input for node '{}'", node.name));
    let mut result = EmitResult::default();
    result.push_output_op(
        make_output_tensor(node),
        Op::Identity,
        Attribute::None,
        vec![source],
    );
    result
}

pub(super) fn emit_eye(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let output_shape = actual_tensor_shape(graph, store, node);
    let zero_value = quantize_fill_value(node, 0.0);
    let one_value = quantize_fill_value(node, 1.0);
    let tensor = make_eye_const_tensor(
        node.name.clone(),
        output_shape,
        node.meta.val.dtype,
        zero_value,
        one_value,
    )
    .unwrap_or_else(|| {
        panic!(
            "emit_eye: unsupported eye shape/dtype for node '{}'",
            node.name
        )
    });
    let mut result = EmitResult::default();
    push_inline_const(&mut result, tensor);
    result
}

pub(super) fn emit_scalar_tensor(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut output_shape = actual_tensor_shape(graph, store, node);
    if output_shape.is_empty() {
        output_shape.push(1);
    }
    let fill_value = node.args.first().map_or(0.0, Arg::as_f64);
    emit_filled_const(node, output_shape, fill_value, "emit_scalar_tensor")
}

pub(super) fn emit_resize(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
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

pub(super) fn emit_aten_reshape(ctx: &EmitContext) -> EmitResult {
    emit_reshape(ctx)
}

pub(super) fn emit_aten_permute(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let spec = ParsedShapeTransformSpec::parse(graph, store, node);
    let perms = arg_i32_list(node, 1);
    emit_shape_transform(
        spec,
        Op::Transpose,
        Attribute::Transpose(TransposeAttribute { perms }),
    )
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
    if multiples.is_empty() {
        return result;
    }

    let tile_rank = multiples.len();
    let mut tile_multiples = multiples;
    let mut tile_input_name = input_name;
    let mut tile_input_shape = input_shape.clone();
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
        result.push_output_op(
            make_empty_tensor(
                reshaped_name.clone(),
                padded_input_shape.clone(),
                actual_tensor_dtype(graph, input_node),
            ),
            Op::Reshape,
            Attribute::Reshape(ReshapeAttribute),
            vec![tile_input_name, reshape_shape_name],
        );
        tile_input_name = reshaped_name;
        tile_input_shape = padded_input_shape.clone();

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
                result.push_output_op(
                    make_empty_tensor(
                        transposed_name.clone(),
                        transposed_shape.clone(),
                        actual_tensor_dtype(graph, input_node),
                    ),
                    Op::Transpose,
                    Attribute::Transpose(TransposeAttribute {
                        perms: perms.clone(),
                    }),
                    vec![tile_input_name],
                );
                tile_multiples = perms
                    .iter()
                    .filter_map(|&idx| tile_multiples.get(idx as usize).copied())
                    .collect();
                tile_input_name = transposed_name;
                tile_input_shape = transposed_shape;
            }
        }
    }

    let output_shape: Vec<i32> = tile_input_shape
        .iter()
        .zip(tile_multiples.iter())
        .map(|(&input_dim, &multiple)| input_dim.saturating_mul(multiple))
        .collect();
    if output_shape.len() != tile_rank {
        return result;
    }

    let multiples_name = format!("{}_multiples", node.name);
    push_inline_const(
        &mut result,
        make_shape_const_tensor(multiples_name.clone(), tile_multiples),
    );
    result.push_output_op(
        make_empty_tensor(
            primary_output_name(graph, node),
            output_shape,
            actual_tensor_dtype(graph, input_node),
        ),
        Op::Tile,
        Attribute::None,
        vec![tile_input_name, multiples_name],
    );
    result
}

pub(super) fn emit_expand(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
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
    let raw_input_shape = raw_i32_shape(input_node);
    let raw_output_shape = raw_i32_shape(node);
    let input_shape = actual_tensor_shape(graph, store, input_node);
    let output_shape = if raw_input_shape.len() == raw_output_shape.len() {
        actual_tensor_shape(graph, store, node)
    } else {
        apply_tosa_dim_order(node, raw_output_shape)
    };
    if output_shape.is_empty() {
        return result;
    }

    let tile_rank = output_shape.len();
    let mut tile_input_name = input_name;
    let mut tile_input_shape = input_shape.clone();
    if tile_input_shape.len() < tile_rank {
        let mut padded_input_shape = vec![1; tile_rank - tile_input_shape.len()];
        padded_input_shape.extend(tile_input_shape.iter().copied());

        let reshaped_name = format!("{}_input_reshape", node.name);
        let reshape_shape_name = format!("{}_input_shape", node.name);
        push_const_shape(
            &mut result,
            reshape_shape_name.clone(),
            padded_input_shape.clone(),
        );
        result.push_output_op(
            make_empty_tensor(
                reshaped_name.clone(),
                padded_input_shape.clone(),
                actual_tensor_dtype(graph, input_node),
            ),
            Op::Reshape,
            Attribute::Reshape(ReshapeAttribute),
            vec![tile_input_name, reshape_shape_name],
        );
        tile_input_name = reshaped_name;
        tile_input_shape = padded_input_shape.clone();

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
                result.push_output_op(
                    make_empty_tensor(
                        transposed_name.clone(),
                        transposed_shape.clone(),
                        actual_tensor_dtype(graph, input_node),
                    ),
                    Op::Transpose,
                    Attribute::Transpose(TransposeAttribute {
                        perms: perms.clone(),
                    }),
                    vec![tile_input_name],
                );
                tile_input_name = transposed_name;
                tile_input_shape = transposed_shape;
            }
        }
    }

    let multiples: Vec<i32> = tile_input_shape
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

    let multiples_name = format!("{}_multiples", node.name);
    push_inline_const(
        &mut result,
        make_shape_const_tensor(multiples_name.clone(), multiples),
    );
    result.push_output_op(
        make_empty_tensor(
            primary_output_name(graph, node),
            output_shape,
            actual_tensor_dtype(graph, input_node),
        ),
        Op::Tile,
        Attribute::None,
        vec![tile_input_name, multiples_name],
    );
    result
}

pub(super) fn emit_repeat(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut multiples = arg_i32_list(node, 1);
    if let Some(Arg::Node(input_id)) = node.args.first() {
        if let Some(input_node) = graph.node(*input_id) {
            if raw_i32_shape(input_node).len() == multiples.len() {
                multiples = apply_tosa_dim_order(node, multiples);
            }
        }
    }
    emit_tile_like(graph, node, store, multiples)
}

pub(super) fn emit_const(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let store = ctx.store;
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

    result.push_output_op(
        make_tensor(name.clone(), shape, dtype, data),
        Op::Const,
        Attribute::Const(ConstAttribute),
        vec![],
    );
    result
}

pub(super) fn emit_transpose(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut spec = ParsedShapeTransformSpec::parse(graph, store, node);
    spec.output_shape = transposed_tensor_shape(graph, store, node).unwrap_or(spec.output_shape);
    emit_shape_transform(
        spec,
        Op::Transpose,
        Attribute::Transpose(TransposeAttribute {
            perms: transpose_perms(node),
        }),
    )
}

pub(super) fn emit_reshape(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let out_shape = actual_tensor_shape(graph, store, node);
    let output_name = primary_output_name(graph, node);
    let output_dtype = passthrough_input_dtype(graph, node);

    if let Some(input_node) = first_input_node(graph, node) {
        let input_shape = actual_tensor_shape(graph, store, input_node);
        let needs_avgpool_rank_drop_fix = out_shape.len() == 3
            && input_shape.len() == 4
            && (input_shape.get(1) == Some(&1) || input_shape.get(2) == Some(&1))
            && matches!(
                input_node.target,
                crate::graph::ir::OpTarget::AvgPool2d | crate::graph::ir::OpTarget::TosaAvgPool2d
            );

        if needs_avgpool_rank_drop_fix {
            let transpose_name = format!("{}_pre_reshape_nchw", node.name);
            let transpose_shape = vec![input_shape[0], input_shape[3], input_shape[1], input_shape[2]];
            result.push_output_op(
                make_empty_tensor(
                    transpose_name.clone(),
                    transpose_shape,
                    output_dtype,
                ),
                Op::Transpose,
                Attribute::Transpose(TransposeAttribute {
                    perms: vec![0, 3, 1, 2],
                }),
                input_names.clone(),
            );

            push_reshape_with_shape(
                &mut result,
                transpose_name,
                output_name,
                format!("{}_shape", node.name),
                out_shape,
                output_dtype,
            );
            return result;
        }
    }

    result.tensors.push(make_empty_tensor(
        output_name.clone(),
        out_shape.clone(),
        output_dtype,
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
        vec![output_name],
    ));
    result
}

pub(super) fn emit_concat(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    let raw_axis = match node.args.get(1) {
        Some(Arg::Int(axis)) => *axis as i32,
        _ => kwarg_i32(node, "axis", 0),
    };
    let axis = match node.args.first() {
        Some(Arg::Node(input_id)) => graph
            .node(*input_id)
            .map(|input_node| map_axis_to_tosa_dim_order(graph, input_node, raw_axis))
            .unwrap_or(raw_axis),
        Some(Arg::NodeList(ids)) => ids
            .first()
            .and_then(|input_id| graph.node(*input_id))
            .map(|input_node| map_axis_to_tosa_dim_order(graph, input_node, raw_axis))
            .unwrap_or(raw_axis),
        _ => raw_axis,
    };
    result.push_output_op(
        make_output_tensor(node),
        Op::Concat,
        Attribute::Concat(ConcatAttribute { axis }),
        input_names,
    );
    result
}
