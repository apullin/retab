use super::emitter::EmitContext;
use super::*;
use crate::serialize::op_semantics;

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
        (OpTarget::QuantizePerTensor, emit_quantize_per_tensor),
        (OpTarget::DequantizePerTensor, emit_dequantize_per_tensor),
        (OpTarget::Reshape, emit_reshape),
        (OpTarget::Flatten, emit_reshape),
        (OpTarget::View, emit_reshape),
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

fn quantize_intermediate_ir_dtype(graph: &Graph, node: &Node) -> crate::graph::ir::DType {
    first_input_node(graph, node)
        .map(|input| ir_float_dtype_or_float32(input.meta.val.dtype))
        .unwrap_or(crate::graph::ir::DType::Float32)
}

fn full_quant_bounds(dtype: DType) -> Option<(i64, i64)> {
    match dtype {
        DType::Int8 => Some((-128, 127)),
        DType::Int16 => Some((-32_768, 32_767)),
        DType::Int32 => Some((-2_147_483_648, 2_147_483_647)),
        _ => None,
    }
}

pub(super) fn emit_quantize_per_tensor(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let graph = ctx.graph;
    let mut result = EmitResult::default();
    let Some(mut current_name) = ctx.input_name(0) else {
        return result;
    };

    let output_shape = ctx.shape();
    let output_dtype = ctx.dtype();
    let intermediate_ir_dtype = quantize_intermediate_ir_dtype(graph, node);
    let intermediate_dtype = ir_dtype_to_tosa(intermediate_ir_dtype);
    let scale = arg_f64(node, 1, 1.0);
    if scale == 0.0 {
        return result;
    }
    let zero_point = arg_i64(node, 2, 0) as f64;
    let qmin = node.args.get(3).and_then(Arg::as_i64);
    let qmax = node.args.get(4).and_then(Arg::as_i64);

    if scale != 1.0 {
        let scale_name = format!("{}_scale", node.name);
        let scaled_name = format!("{}_scaled", node.name);
        if push_filled_const_if_supported(
            &mut result,
            scale_name.clone(),
            output_shape.clone(),
            intermediate_ir_dtype,
            1.0 / scale,
        ) {
            let shift_name = push_shift_const(&mut result, &node.name);
            result.push_output_op(
                make_empty_tensor(
                    scaled_name.clone(),
                    output_shape.clone(),
                    intermediate_dtype,
                ),
                Op::Mul,
                Attribute::Mul(MulAttribute { shift: 0 }),
                vec![current_name, scale_name, shift_name],
            );
            current_name = scaled_name;
        }
    }

    let round_name = format!("{}_round", node.name);
    current_name = emit_bankers_round_with_fill(
        &mut result,
        &node.name,
        &round_name,
        current_name,
        output_shape.clone(),
        intermediate_dtype,
        intermediate_ir_dtype,
        make_empty_tensor(round_name.clone(), output_shape.clone(), intermediate_dtype),
        |value| value,
        "emit_quantize_per_tensor",
    );

    if zero_point != 0.0 {
        let zp_name = format!("{}_zp", node.name);
        let shifted_name = format!("{}_shifted", node.name);
        if push_filled_const_if_supported(
            &mut result,
            zp_name.clone(),
            output_shape.clone(),
            intermediate_ir_dtype,
            zero_point,
        ) {
            result.push_output_op(
                make_empty_tensor(
                    shifted_name.clone(),
                    output_shape.clone(),
                    intermediate_dtype,
                ),
                Op::Add,
                Attribute::Add(AddAttribute {}),
                vec![current_name, zp_name],
            );
            current_name = shifted_name;
        }
    }

    if let (Some(qmin), Some(qmax)) = (qmin, qmax) {
        if full_quant_bounds(output_dtype) != Some((qmin, qmax)) {
            let min_arg = Arg::Float(qmin as f64);
            let max_arg = Arg::Float(qmax as f64);
            let clamped_name = format!("{}_clamped", node.name);
            result.push_output_op(
                make_empty_tensor(
                    clamped_name.clone(),
                    output_shape.clone(),
                    intermediate_dtype,
                ),
                Op::Clamp,
                Attribute::Clamp(ClampAttribute {
                    min_val: clamp_bound_bytes_from_arg(
                        intermediate_ir_dtype,
                        Some(&min_arg),
                        true,
                    ),
                    max_val: clamp_bound_bytes_from_arg(
                        intermediate_ir_dtype,
                        Some(&max_arg),
                        false,
                    ),
                    nan_mode: NanPropagationMode::Propagate,
                }),
                vec![current_name],
            );
            current_name = clamped_name;
        }
    }

    result.push_output_op(
        ctx.make_output(),
        Op::Cast,
        Attribute::Cast(CastAttribute {}),
        vec![current_name],
    );
    result
}

pub(super) fn emit_dequantize_per_tensor(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    let Some(input_name) = ctx.input_name(0) else {
        return result;
    };

    let output_shape = ctx.shape();
    let ir_output_dtype = node.meta.val.dtype;
    let output_dtype = ir_dtype_to_tosa(ir_float_dtype_or_float32(ir_output_dtype));
    let output_ir_dtype = tosa_float_dtype_to_ir_or_float32(output_dtype);
    let scale = arg_f64(node, 1, 1.0);
    let zero_point = arg_i64(node, 2, 0) as f64;

    let cast_output_name = if zero_point == 0.0 && scale == 1.0 {
        node.name.clone()
    } else {
        format!("{}_cast", node.name)
    };
    result.push_output_op(
        make_empty_tensor(cast_output_name.clone(), output_shape.clone(), output_dtype),
        Op::Cast,
        Attribute::Cast(CastAttribute {}),
        vec![input_name],
    );

    let mut current_name = cast_output_name;
    if zero_point != 0.0 {
        let zp_name = format!("{}_zp", node.name);
        if push_filled_const_if_supported(
            &mut result,
            zp_name.clone(),
            output_shape.clone(),
            output_ir_dtype,
            zero_point,
        ) {
            let sub_output_name = if scale == 1.0 {
                node.name.clone()
            } else {
                format!("{}_sub", node.name)
            };
            result.push_output_op(
                make_empty_tensor(sub_output_name.clone(), output_shape.clone(), output_dtype),
                Op::Sub,
                Attribute::Sub(SubAttribute {}),
                vec![current_name, zp_name],
            );
            current_name = sub_output_name;
        }
    }

    if scale != 1.0 {
        let scale_name = format!("{}_scale", node.name);
        if push_filled_const_if_supported(
            &mut result,
            scale_name.clone(),
            output_shape.clone(),
            output_ir_dtype,
            scale,
        ) {
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
    push_zero_point_const(result, input_zp_name.clone(), input_dtype, 0);
    push_zero_point_const(result, output_zp_name.clone(), output_dtype, 0);

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
        let input_name = input_name(graph, node, 0)?;
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
    let mut result = EmitResult::default();
    assert!(
        push_filled_const_if_supported(
            &mut result,
            node.name.clone(),
            output_shape,
            dtype,
            fill_value,
        ),
        "{context}: unsupported dtype {:?} for node '{}'",
        dtype,
        node.name
    );
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
    let node = ctx.node;
    let output_shape = ctx.shape();
    let fill_value = arg_f64(node, 1, 0.0);
    emit_filled_const(node, output_shape, fill_value, "emit_full")
}

pub(super) fn emit_copy(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut inputs = ctx.input_names();
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
    let node = ctx.node;
    let output_shape = ctx.shape();
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
    let node = ctx.node;
    let mut output_shape = ctx.shape();
    if output_shape.is_empty() {
        output_shape.push(1);
    }
    let fill_value = arg_f64(node, 0, 0.0);
    emit_filled_const(node, output_shape, fill_value, "emit_scalar_tensor")
}

pub(super) fn emit_resize(ctx: &EmitContext) -> EmitResult {
    let Some(spec) = ParsedResizeSpec::parse(ctx.graph, ctx.node, ctx.store) else {
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

pub(super) fn emit_aten_permute(ctx: &EmitContext) -> EmitResult {
    let spec = ParsedShapeTransformSpec::parse(ctx.graph, ctx.store, ctx.node);
    let perms = ctx.arg_i32_list(1);
    if is_identity_permutation(&perms) {
        return ctx.emit_identity();
    }
    emit_shape_transform(
        spec,
        Op::Transpose,
        Attribute::Transpose(TransposeAttribute { perms }),
    )
}

/// If the node has a non-identity `tosa_dim_order`, insert a transpose op and
/// permute `multiples` to match. Returns the (possibly updated) input name,
/// input shape, and multiples.
fn maybe_insert_transpose(
    result: &mut EmitResult,
    node: &Node,
    padded_input_shape: &[i32],
    input_name: String,
    mut multiples: Vec<i32>,
    dtype: DType,
) -> (String, Vec<i32>, Vec<i32>) {
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
        if is_non_identity {
            let Some(transposed_shape) = permute_shape(padded_input_shape, &perms) else {
                return (input_name, padded_input_shape.to_vec(), multiples);
            };
            let transposed_name = format!("{}_input_layout", node.name);
            result.push_output_op(
                make_empty_tensor(transposed_name.clone(), transposed_shape.clone(), dtype),
                Op::Transpose,
                Attribute::Transpose(TransposeAttribute {
                    perms: perms.clone(),
                }),
                vec![input_name],
            );
            multiples = permute_shape(&multiples, &perms).unwrap_or(multiples);
            return (transposed_name, transposed_shape, multiples);
        }
    }
    (input_name, padded_input_shape.to_vec(), multiples)
}

/// Shared implementation for tile-like ops (`repeat` and `expand`).
///
/// Pads the input shape to `tile_rank` if needed, optionally transposes to
/// match the node's `tosa_dim_order`, then emits a TOSA `Tile` operator.
fn emit_tile_with_multiples(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    input_name: String,
    input_node: &Node,
    multiples: Vec<i32>,
    tile_rank: usize,
    align_equal_rank_layout: bool,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if multiples.is_empty() {
        return result;
    }

    let mut tile_multiples = multiples;
    let mut tile_input_name = input_name;
    let mut tile_input_shape = input_shape.clone();
    let mut rank_padded = false;
    let input_dtype = actual_tensor_dtype(graph, input_node);
    if input_shape.len() < tile_rank {
        rank_padded = true;
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
                input_dtype,
            ),
            Op::Reshape,
            Attribute::Reshape(ReshapeAttribute {}),
            vec![tile_input_name, reshape_shape_name],
        );
        tile_input_name = reshaped_name;
        tile_input_shape = padded_input_shape;
    }

    if rank_padded || align_equal_rank_layout {
        let (name, shape, mults) = maybe_insert_transpose(
            &mut result,
            node,
            &tile_input_shape,
            tile_input_name,
            tile_multiples,
            input_dtype,
        );
        tile_input_name = name;
        tile_input_shape = shape;
        tile_multiples = mults;
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
    push_const_shape(&mut result, multiples_name.clone(), tile_multiples);
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

fn emit_tile_like(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    multiples: Vec<i32>,
) -> EmitResult {
    let Some(input_name) = input_name(graph, node, 0) else {
        return EmitResult::default();
    };
    let Some(Arg::Node(input_id)) = node.args.first() else {
        return EmitResult::default();
    };
    let Some(input_node) = graph.node(*input_id) else {
        return EmitResult::default();
    };
    let tile_rank = multiples.len();
    emit_tile_with_multiples(
        graph, node, store, input_name, input_node, multiples, tile_rank, false,
    )
}

pub(super) fn emit_expand(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let Some(input_name) = ctx.input_name(0) else {
        return EmitResult::default();
    };
    let Some(Arg::Node(input_id)) = node.args.first() else {
        return EmitResult::default();
    };
    let Some(input_node) = graph.node(*input_id) else {
        return EmitResult::default();
    };

    let raw_output_shape = raw_i32_shape(node);
    let raw_input_shape = raw_i32_shape(input_node);
    let input_shape = actual_tensor_shape(graph, store, input_node);

    let input_is_layout_boundary = input_node.target == OpTarget::TosaTranspose
        && (input_node.name.ends_with("_input_layout")
            || input_node.name.ends_with("_output_layout"));
    let input_is_physical_equal_rank = input_shape.len() == raw_output_shape.len()
        && (input_is_layout_boundary || input_shape != raw_input_shape);

    let (target_shape, align_equal_rank_layout) = if input_is_physical_equal_rank {
        (actual_tensor_shape(graph, store, node), false)
    } else {
        // Compute multiples in logical (pre-transpose) order so that the shared
        // tile emitter can apply `tosa_dim_order` exactly once.
        (raw_output_shape, true)
    };
    if target_shape.is_empty() {
        return EmitResult::default();
    }
    let tile_rank = target_shape.len();

    // Pad input shape to tile_rank in logical order (matching emit_tile_with_multiples).
    let logical_input: Vec<i32> = if input_shape.len() < tile_rank {
        let mut padded = vec![1; tile_rank - input_shape.len()];
        padded.extend(input_shape.iter().copied());
        padded
    } else {
        input_shape
    };

    let multiples: Vec<i32> = logical_input
        .iter()
        .zip(target_shape.iter())
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

    emit_tile_with_multiples(
        graph,
        node,
        store,
        input_name,
        input_node,
        multiples,
        tile_rank,
        align_equal_rank_layout,
    )
}

pub(super) fn emit_repeat(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut multiples = ctx.arg_i32_list(1);
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
        Attribute::Const(ConstAttribute {}),
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
    let perms = transpose_perms(node);
    if is_identity_permutation(&perms) {
        return ctx.emit_identity();
    }
    if let Some(input_node) = first_input_node(graph, node) {
        let input_shape = actual_tensor_shape(graph, store, input_node);
        if transpose_only_moves_unit_axes(&input_shape, &perms) {
            let mut result = EmitResult::default();
            push_reshape_with_shape(
                &mut result,
                spec.input_names[0].clone(),
                spec.output_name,
                format!("{}_shape", node.name),
                spec.output_shape,
                spec.output_dtype,
            );
            return result;
        }
    }
    emit_shape_transform(
        spec,
        Op::Transpose,
        Attribute::Transpose(TransposeAttribute { perms }),
    )
}

fn is_identity_permutation(perms: &[i32]) -> bool {
    perms
        .iter()
        .enumerate()
        .all(|(axis, &permuted)| usize::try_from(permuted).ok() == Some(axis))
}

fn transpose_only_moves_unit_axes(input_shape: &[i32], perms: &[i32]) -> bool {
    if input_shape.len() != perms.len() {
        return false;
    }

    let non_unit_axes = input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| (*dim != 1).then_some(axis));
    let permuted_non_unit_axes = perms.iter().filter_map(|axis| {
        let axis = usize::try_from(*axis).ok()?;
        input_shape
            .get(axis)
            .and_then(|dim| (*dim != 1).then_some(axis))
    });

    non_unit_axes.eq(permuted_non_unit_axes)
}

fn is_explicit_layout_boundary(node: &Node) -> bool {
    node.target == OpTarget::TosaTranspose
        && (node.name.ends_with("_input_layout") || node.name.ends_with("_output_layout"))
}

fn is_pool_rank_drop_input(graph: &Graph, node: &Node) -> bool {
    let source = if is_getitem_node(node) {
        first_input_node(graph, node).unwrap_or(node)
    } else {
        node
    };
    op_semantics::is_avg_pool(&source.target) || op_semantics::is_max_pool(&source.target)
}

fn layout_metadata_source<'a>(graph: &'a Graph, node: &'a Node) -> &'a Node {
    if is_getitem_node(node) {
        first_input_node(graph, node).unwrap_or(node)
    } else {
        node
    }
}

pub(super) fn emit_reshape(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let out_shape = ctx.shape();
    let output_name = ctx.primary_output_name();
    let output_dtype = ctx.passthrough_dtype();

    if let Some(input_node) = first_input_node(graph, node) {
        let input_shape = actual_tensor_shape(graph, store, input_node);
        let layout_source = layout_metadata_source(graph, input_node);
        let raw_input_shape = raw_i32_shape(layout_source);
        if out_shape.len() >= input_shape.len()
            && !node.meta.retab_preserve_spatial_rank
            && !is_explicit_layout_boundary(input_node)
            && input_shape.len() == raw_input_shape.len()
            && input_shape != raw_input_shape
            && input_shape.iter().product::<i32>() == raw_input_shape.iter().product::<i32>()
        {
            let mut inverse_orders = Vec::new();
            if let Some(perms) = layout_source
                .meta
                .tosa_dim_order
                .as_ref()
                .and_then(|order| inverse_tosa_dim_order(order))
            {
                inverse_orders.push(perms);
            }
            if let Some(perms) =
                channel_last_order(input_shape.len()).and_then(|order| inverse_permutation(&order))
            {
                if !inverse_orders.iter().any(|existing| existing == &perms) {
                    inverse_orders.push(perms);
                }
            }
            for perms in inverse_orders {
                if permute_shape(&input_shape, &perms).as_deref()
                    == Some(raw_input_shape.as_slice())
                {
                    let transpose_name = format!("{}_pre_reshape_input_layout", node.name);
                    result.push_output_op(
                        make_empty_tensor(
                            transpose_name.clone(),
                            raw_input_shape.clone(),
                            output_dtype,
                        ),
                        Op::Transpose,
                        Attribute::Transpose(TransposeAttribute { perms }),
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
        }

        let needs_pool_rank_drop_fix = out_shape.len() == 3
            && input_shape.len() == 4
            && (input_shape.get(1) == Some(&1) || input_shape.get(2) == Some(&1))
            && is_pool_rank_drop_input(graph, input_node);

        if needs_pool_rank_drop_fix {
            let transpose_name = format!("{}_pre_reshape_nchw", node.name);
            let transpose_shape = vec![
                input_shape[0],
                input_shape[3],
                input_shape[1],
                input_shape[2],
            ];
            result.push_output_op(
                make_empty_tensor(transpose_name.clone(), transpose_shape, output_dtype),
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

    let shape_name = format!("{}_shape", node.name);
    push_const_shape(&mut result, shape_name.clone(), out_shape.clone());
    let mut reshape_inputs = input_names;
    reshape_inputs.push(shape_name.clone());
    result.push_output_op(
        make_empty_tensor(output_name.clone(), out_shape.clone(), output_dtype),
        Op::Reshape,
        Attribute::Reshape(ReshapeAttribute {}),
        reshape_inputs,
    );
    result
}

pub(super) fn emit_concat(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let mut input_names = ctx.input_names();
    let raw_axis = match node.args.get(1) {
        Some(Arg::Int(axis)) => *axis as i32,
        _ => kwarg_i32(node, "axis", 0),
    };
    // Map the cat axis through the OUTPUT node's dim_order, matching upstream
    // op_cat.py (`dim = output.dim_order.index(dim)`). Equivalent when operand
    // layouts are uniform (the norm), but the output is the canonical reference.
    let axis = map_axis_to_tosa_dim_order(graph, node, raw_axis);
    if crate::config::debug_cat_enabled() {
        fn dbg_shape(n: &crate::graph::ir::Node) -> Vec<i64> {
            n.meta.val.shape[..n.meta.val.shape_len as usize].to_vec()
        }
        eprintln!(
            "[CatDebug] {} raw_axis={} axis={} logical={:?} dim_order={:?} emitted={:?}",
            node.name,
            raw_axis,
            axis,
            dbg_shape(node),
            node.meta.tosa_dim_order,
            actual_tensor_shape(graph, ctx.store, node),
        );
        let ids = match node.args.first() {
            Some(Arg::Node(input_id)) => vec![*input_id],
            Some(Arg::NodeList(list)) => list.to_vec(),
            _ => vec![],
        };
        for id in ids {
            if let Some(n) = graph.node(id) {
                eprintln!(
                    "[CatDebug]   in {} target={:?} logical={:?} dim_order={:?} emitted={:?}",
                    n.name,
                    n.target,
                    dbg_shape(n),
                    n.meta.tosa_dim_order,
                    actual_tensor_shape(graph, ctx.store, n),
                );
            }
        }
    }
    let output_tensor = make_output_tensor(node);
    let output_shape = output_tensor.shape.clone();
    let input_nodes: Vec<&Node> = match node.args.first() {
        Some(Arg::Node(input_id)) => graph.node(*input_id).into_iter().collect(),
        Some(Arg::NodeList(ids)) => ids.iter().filter_map(|id| graph.node(*id)).collect(),
        _ => Vec::new(),
    };
    for (idx, input_node) in input_nodes.iter().enumerate().take(input_names.len()) {
        let input_shape = actual_tensor_shape(graph, ctx.store, input_node);
        if concat_shape_matches_output(&input_shape, &output_shape, axis as usize) {
            continue;
        }
        let Some((layout_shape, perms)) =
            concat_input_layout_transform(&input_shape, &output_shape, axis as usize, node)
        else {
            continue;
        };
        let layout_name = format!("{}_input{}_layout", node.name, idx);
        result.push_output_op(
            make_empty_tensor(
                layout_name.clone(),
                layout_shape,
                actual_tensor_dtype(graph, input_node),
            ),
            Op::Transpose,
            Attribute::Transpose(TransposeAttribute { perms }),
            vec![input_names[idx].clone()],
        );
        input_names[idx] = layout_name;
    }
    result.push_output_op(
        output_tensor,
        Op::Concat,
        Attribute::Concat(ConcatAttribute { axis }),
        input_names,
    );
    result
}

fn concat_shape_matches_output(shape: &[i32], output_shape: &[i32], axis: usize) -> bool {
    shape.len() == output_shape.len()
        && shape
            .iter()
            .zip(output_shape.iter())
            .enumerate()
            .all(|(idx, (&dim, &out_dim))| idx == axis || dim == out_dim)
}

fn concat_input_layout_transform(
    input_shape: &[i32],
    output_shape: &[i32],
    axis: usize,
    node: &Node,
) -> Option<(Vec<i32>, Vec<i32>)> {
    let mut orders = Vec::new();
    if let Some(order) = node.meta.tosa_dim_order.as_ref() {
        orders.push(order.iter().map(|&axis| axis as i32).collect::<Vec<_>>());
    }
    if let Some(order) = channel_last_order(input_shape.len()) {
        if !orders.iter().any(|existing| existing == &order) {
            orders.push(order);
        }
    }

    for perms in orders {
        let Some(shape) = permute_shape(input_shape, &perms) else {
            continue;
        };
        if concat_shape_matches_output(&shape, output_shape, axis) {
            return Some((shape, perms));
        }
    }

    None
}
