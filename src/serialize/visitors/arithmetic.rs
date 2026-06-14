use super::emitter::EmitContext;
use super::*;

pub fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaAdd, emit_add as EmitFn),
        (OpTarget::TosaSub, emit_sub),
        (OpTarget::TosaMul, emit_mul),
        (OpTarget::TosaClamp, emit_clamp),
        (OpTarget::Add, emit_add),
        (OpTarget::Sub, emit_sub),
        (OpTarget::Mul, emit_mul),
        (OpTarget::Clamp, emit_aten_clamp),
        (OpTarget::Hardtanh, emit_aten_clamp),
        (OpTarget::Relu, emit_aten_clamp),
        (OpTarget::Unsqueeze, emit_aten_shape_reshape),
        (OpTarget::Squeeze, emit_aten_shape_reshape),
    ]);
}

struct ParsedReshapeEmitterSpec {
    input_name: String,
    output_shape: Vec<i32>,
    output_dtype: DType,
}

impl ParsedReshapeEmitterSpec {
    // Parse once at the edge of the arithmetic/shape wrapper and emit from a
    // typed spec instead of repeatedly open-coding "first input + meta shape".
    fn parse(ctx: &EmitContext) -> Option<Self> {
        Some(Self {
            input_name: ctx.input_name(0)?,
            output_shape: ctx.shape(),
            output_dtype: ctx.dtype(),
        })
    }
}

pub(super) fn emit_aten_clamp(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();

    let clamp_dtype = node.meta.val.dtype;
    let relu_min = Arg::Float(0.0);
    let (min_arg, max_arg) = if matches!(&node.target, crate::graph::ir::OpTarget::Relu) {
        (Some(&relu_min), None)
    } else {
        (node.args.get(1), node.args.get(2))
    };
    let needs_extrema_lowering = clamp_dtype == crate::graph::ir::DType::Int32
        || matches!(min_arg, Some(Arg::Node(_)))
        || matches!(max_arg, Some(Arg::Node(_)));

    if needs_extrema_lowering {
        let input_name = ctx.input_name(0).unwrap_or_default();
        let output_shape = ctx.shape();
        let output_dtype = ir_dtype_to_tosa(clamp_dtype);
        let output_rank = output_shape.len();

        let min_input = resolve_clamp_bound_input(
            graph,
            node,
            min_arg,
            "min",
            output_rank,
            clamp_dtype,
            &mut result,
        );
        let max_input = resolve_clamp_bound_input(
            graph,
            node,
            max_arg,
            "max",
            output_rank,
            clamp_dtype,
            &mut result,
        );
        let has_min_input = min_input.is_some();
        let has_max_input = max_input.is_some();

        let mut current_name = input_name;
        let mut pending_max = max_input;

        if let Some((mut min_name, min_node)) = min_input {
            if let Some(bound_node) = min_node.and_then(|id| graph.node(id)) {
                maybe_materialize_clamp_bound_broadcast(
                    graph,
                    store,
                    node,
                    bound_node,
                    &mut min_name,
                    &mut result,
                    "min",
                );
            }

            let out_name = if pending_max.is_some() {
                format!("{}_clamp_min", node.name)
            } else {
                node.name.clone()
            };
            result.push_output_op(
                make_empty_tensor(out_name.clone(), output_shape.clone(), output_dtype),
                Op::Maximum,
                Attribute::Maximum(MaximumAttribute {
                    nan_mode: NanPropagationMode::Propagate,
                }),
                vec![current_name, min_name],
            );
            current_name = out_name;
        }

        if let Some((mut max_name, max_node)) = pending_max.take() {
            if let Some(bound_node) = max_node.and_then(|id| graph.node(id)) {
                maybe_materialize_clamp_bound_broadcast(
                    graph,
                    store,
                    node,
                    bound_node,
                    &mut max_name,
                    &mut result,
                    "max",
                );
            }

            result.push_output_op(
                make_empty_tensor(node.name.clone(), output_shape, output_dtype),
                Op::Minimum,
                Attribute::Minimum(MinimumAttribute {
                    nan_mode: NanPropagationMode::Propagate,
                }),
                vec![current_name.clone(), max_name],
            );
        }

        if !has_min_input && !has_max_input {
            result.push_output_op(
                ctx.make_output(),
                Op::Identity,
                Attribute::None,
                vec![current_name],
            );
        }

        return result;
    }

    let min_val = clamp_bound_bytes_from_arg(clamp_dtype, min_arg, true);
    let max_val = clamp_bound_bytes_from_arg(clamp_dtype, max_arg, false);
    let input_names = ctx.input_names();
    result.push_output_op(
        ctx.make_output(),
        Op::Clamp,
        Attribute::Clamp(ClampAttribute {
            min_val,
            max_val,
            nan_mode: NanPropagationMode::Propagate,
        }),
        input_names,
    );
    result
}

pub(super) fn emit_clamp(ctx: &EmitContext) -> EmitResult {
    let node = ctx.node;
    let mut result = EmitResult::default();
    let input_names = ctx.input_names();
    let clamp_dtype = node.meta.val.dtype;
    let min_arg = node
        .kwargs
        .iter()
        .find_map(|(k, v)| if k == "min_val" { Some(v) } else { None });
    let max_arg = node
        .kwargs
        .iter()
        .find_map(|(k, v)| if k == "max_val" { Some(v) } else { None });
    let min_val = clamp_bound_bytes_from_arg(clamp_dtype, min_arg, true);
    let max_val = clamp_bound_bytes_from_arg(clamp_dtype, max_arg, false);

    result.push_output_op(
        ctx.make_output(),
        Op::Clamp,
        Attribute::Clamp(ClampAttribute {
            min_val,
            max_val,
            nan_mode: NanPropagationMode::Propagate,
        }),
        input_names,
    );
    result
}

pub(super) fn emit_add(ctx: &EmitContext) -> EmitResult {
    ctx.emit_binary(Op::Add, Attribute::Add(AddAttribute {}))
}

pub(super) fn emit_sub(ctx: &EmitContext) -> EmitResult {
    ctx.emit_binary(Op::Sub, Attribute::Sub(SubAttribute {}))
}

pub(super) fn emit_mul(ctx: &EmitContext) -> EmitResult {
    let graph = ctx.graph;
    let node = ctx.node;
    let store = ctx.store;
    let mut result = EmitResult::default();
    let mut inputs = collect_binary_input_names(graph, store, node, &mut result);
    inputs.push(push_shift_const(&mut result, &node.name));
    result.push_output_op(
        ctx.make_output(),
        Op::Mul,
        Attribute::Mul(MulAttribute { shift: 0 }),
        inputs,
    );
    result
}

pub(super) fn emit_aten_shape_reshape(ctx: &EmitContext) -> EmitResult {
    let mut result = EmitResult::default();
    let spec = ParsedReshapeEmitterSpec::parse(ctx).unwrap_or_else(|| {
        panic!(
            "emit_aten_shape_reshape: missing input for node '{}'",
            ctx.name()
        )
    });
    push_reshape_with_shape_for_node(
        &mut result,
        Some(ctx.node),
        spec.input_name,
        ctx.node.name.clone(),
        format!("{}_shape", ctx.name()),
        spec.output_shape,
        spec.output_dtype,
    );
    result
}
