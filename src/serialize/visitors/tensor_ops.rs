use super::emitter::EmitContext;
use super::*;

pub fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaConstShape, emit_const_shape as EmitFn),
        (OpTarget::TosaConst, emit_const as EmitFn),
        (OpTarget::Full, emit_full),
        (OpTarget::Copy, emit_copy),
        (OpTarget::Eye, emit_eye),
        (OpTarget::ScalarTensor, emit_scalar_tensor),
        (OpTarget::TosaResize, emit_tosa_resize),
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
        (OpTarget::Cat, emit_concat),
        (OpTarget::Flip, emit_flip),
        (OpTarget::Zeros, emit_full),
    ]);
}

fn emit_const_shape(ctx: &EmitContext) -> EmitResult {
    let mut result = EmitResult::default();
    push_const_shape(
        &mut result,
        ctx.node.name.clone(),
        arg_i32_list(ctx.node, 0),
    );
    result
}

fn parse_resize_mode(node: &Node) -> ResizeMode {
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
                Attribute::Mul(MulAttribute {}),
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
                Attribute::Mul(MulAttribute {}),
                vec![current_name, scale_name, shift_name],
            );
        }
    }

    result
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
    let output_shape = ctx.shape();
    let fill_value = arg_f64(node, 0, 0.0);
    emit_filled_const(node, output_shape, fill_value, "emit_scalar_tensor")
}

/// rc4's `RESIZE(x, scale, offset, border, *, resize_mode)` (P2 design X4). The rewrite
/// already computed the parameters (`rewrite_upsample.py:45-155` keeps align_corners and
/// the nearest-mode offsets) into CONST_SHAPE operands, and an int8/int16 bilinear
/// rescale is its own RESCALE node, so this only serializes the node. The parameters go
/// through `push_const_shape`, which pools them into the operands' own CONST_SHAPE
/// tensors: those are pooled too, so naming them directly could dangle when two hold
/// the same values (offset and border are both [0, 0] for align_corners).
fn emit_tosa_resize(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let Some(input_name) = ctx.input_name(0) else {
        return result;
    };
    let mut inputs = vec![input_name];
    for (index, suffix) in [(1, "scales"), (2, "offset"), (3, "border")] {
        let name = format!("{}_{suffix}", node.name);
        push_const_shape(
            &mut result,
            name.clone(),
            shape_arg_i32_list(graph, node, index),
        );
        inputs.push(name);
    }
    // The val dtype, not the inherited output qparam: int8 bilinear produces int32, and
    // int16 bilinear is marked INT48 (`tosa/dialect/ops/resize.py:28-53`).
    let dtype = node
        .meta
        .tosa_special_dtype
        .map(ir_dtype_to_tosa)
        .unwrap_or_else(|| ir_dtype_to_tosa(node.meta.val.dtype));
    result.push_output_op(
        make_empty_tensor(ctx.primary_output_name(), ctx.shape(), dtype),
        Op::Resize,
        Attribute::Resize(ResizeAttribute {
            mode: parse_resize_mode(node),
        }),
        inputs,
    );
    result
}

pub(super) fn emit_aten_permute(ctx: &EmitContext) -> EmitResult {
    let spec = ParsedShapeTransformSpec::parse(ctx.graph, ctx.store, ctx.node);
    let perms = transpose_perms(ctx.node);
    emit_shape_transform(
        spec,
        Op::Transpose,
        Attribute::Transpose(TransposeAttribute { perms }),
    )
}

/// Shared implementation for tile-like ops (`repeat` and `expand`).
///
/// Pads the input shape to `tile_rank` if needed, then emits a TOSA `Tile` operator.
fn emit_tile_with_multiples(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    input_name: String,
    input_node: &Node,
    multiples: Vec<i32>,
    tile_rank: usize,
) -> EmitResult {
    let mut result = EmitResult::default();
    let input_shape = actual_tensor_shape(graph, store, input_node);
    if multiples.is_empty() {
        return result;
    }

    let tile_multiples = multiples;
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
            Attribute::Reshape(ReshapeAttribute {}),
            vec![tile_input_name, reshape_shape_name],
        );
        tile_input_name = reshaped_name;
        tile_input_shape = padded_input_shape;
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
        graph, node, store, input_name, input_node, multiples, tile_rank,
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

    let raw_input_shape = raw_i32_shape(input_node);
    let raw_output_shape = raw_i32_shape(node);
    let input_shape = actual_tensor_shape(graph, store, input_node);

    // Compute multiples in logical (pre-transpose) order so that
    // emit_tile_with_multiples can permute them during transpose.
    let target_shape = if raw_input_shape.len() == raw_output_shape.len() {
        actual_tensor_shape(graph, store, node)
    } else {
        raw_output_shape
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
        graph, node, store, input_name, input_node, multiples, tile_rank,
    )
}

pub(super) fn emit_repeat(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let multiples = ctx.arg_i32_list(1);
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

pub(super) fn emit_reshape(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let out_shape = ctx.shape();
    let output_name = ctx.primary_output_name();
    let output_dtype = ctx.passthrough_dtype();

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

/// Single-axis `aten.flip` -> TOSA REVERSE (upstream FlipVisitor). Multi-axis flips
/// are split into a chain of single-axis flips by `IrDecomposeFlipPass`.
pub(super) fn emit_flip(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let dims: Vec<i64> = match node.args.get(1) {
        Some(Arg::IntList(dims)) => dims.to_vec(),
        Some(Arg::Int(dim)) => vec![*dim],
        _ => vec![],
    };
    let [dim] = dims.as_slice() else {
        panic!(
            "emit_flip: '{}' expects exactly one flip dim after IrDecomposeFlipPass, got {:?}",
            node.name, dims
        );
    };
    let axis = normalize_axis(node, *dim as i32);
    let mut result = EmitResult::default();
    result.push_output_op(
        make_output_tensor(node),
        Op::Reverse,
        Attribute::Reverse(ReverseAttribute { axis }),
        ctx.input_names(),
    );
    result
}

pub(super) fn emit_concat(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let raw_axis = match node.args.get(1) {
        Some(Arg::Int(axis)) => *axis as i32,
        _ => kwarg_i32(node, "axis", 0),
    };
    // rc4's CONCAT visitor takes the axis as given (`op_tosa_concat.py`); a negative
    // one is counted from the back of the output.
    let axis = normalize_axis(node, raw_axis);
    if crate::config::debug_cat_enabled() {
        fn dbg_shape(n: &crate::graph::ir::Node) -> Vec<i64> {
            n.meta.val.shape[..n.meta.val.shape_len as usize].to_vec()
        }
        eprintln!(
            "[CatDebug] {} raw_axis={} axis={} logical={:?} emitted={:?}",
            node.name,
            raw_axis,
            axis,
            dbg_shape(node),
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
                    "[CatDebug]   in {} target={:?} logical={:?} emitted={:?}",
                    n.name,
                    n.target,
                    dbg_shape(n),
                    actual_tensor_shape(graph, ctx.store, n),
                );
            }
        }
    }
    result.push_output_op(
        make_output_tensor(node),
        Op::Concat,
        Attribute::Concat(ConcatAttribute { axis }),
        input_names,
    );
    result
}
