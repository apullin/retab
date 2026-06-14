use super::emitter::EmitContext;
use super::*;
use crate::graph::ir::Arg;

mod extrema;
mod mean;
mod sum_ops;

pub fn register(table: &mut EmitterTable) {
    for register in [
        extrema::register as fn(&mut EmitterTable),
        mean::register,
        sum_ops::register,
    ] {
        register(table);
    }
}

#[derive(Debug, Clone, Copy)]
struct ReduceAxis {
    raw_dim: i32,
    tosa_axis: i32,
}

#[derive(Debug)]
struct ParsedReduceSpec {
    input_name: String,
    input_shape: Vec<i32>,
    axes: Vec<ReduceAxis>,
    keepdim: bool,
}

#[derive(Debug)]
struct ParsedSingleAxisReduceSpec {
    input_names: Vec<String>,
    input_shape: Vec<i32>,
    raw_axis: i32,
    tosa_axis: i32,
    keepdim: bool,
    reduce_dtype: DType,
}

impl ParsedReduceSpec {
    // Parse reduction metadata once at the boundary. Downstream emitters can
    // then operate on typed axes instead of repeatedly validating raw args.
    fn parse(graph: &Graph, node: &Node, store: &TensorStore, keepdim_default: bool) -> Self {
        let input_name = collect_input_names(graph, node)
            .first()
            .cloned()
            .unwrap_or_default();
        let input_node = node
            .args
            .first()
            .and_then(Arg::as_node)
            .and_then(|id| graph.node(id));
        let input_shape = input_node
            .map(|n| actual_tensor_shape(graph, store, n))
            .unwrap_or_default();
        let keepdim = arg_bool(node, 2, keepdim_default);
        let mut dims = parse_reduce_dims(graph, node, store, input_node);
        dims.sort_unstable();
        dims.reverse();

        let axes = dims
            .into_iter()
            .map(|raw_dim| ReduceAxis {
                raw_dim,
                tosa_axis: input_node
                    .map(|n| map_axis_to_tosa_dim_order(graph, n, raw_dim))
                    .unwrap_or(raw_dim),
            })
            .collect();

        Self {
            input_name,
            input_shape,
            axes,
            keepdim,
        }
    }
}

impl ParsedSingleAxisReduceSpec {
    fn parse(graph: &Graph, node: &Node, store: &TensorStore) -> Self {
        let input_names = collect_input_names(graph, node);
        let input_node = match node.args.first() {
            Some(Arg::Node(input_id)) => graph.node(*input_id),
            _ => None,
        };

        let raw_axis = {
            let dims = arg_i32_list(node, 1);
            if let Some(axis) = dims.first().copied() {
                axis
            } else {
                kwarg_i32(node, "axis", 0)
            }
        };
        let tosa_axis = input_node
            .map(|input_node| map_axis_to_tosa_dim_order(graph, input_node, raw_axis))
            .unwrap_or(raw_axis);
        let keepdim = arg_bool(node, 2, true);
        let input_shape = input_node
            .map(|input_node| actual_tensor_shape(graph, store, input_node))
            .unwrap_or_default();
        let input_dtype = input_node
            .map(|n| passthrough_input_dtype(graph, n))
            .unwrap_or_else(|| node_output_dtype(node));
        let reduce_dtype = match input_dtype {
            DType::Int8 | DType::Int16 => DType::Int32,
            _ => node_output_dtype(node),
        };

        Self {
            input_names,
            input_shape,
            raw_axis,
            tosa_axis,
            keepdim,
            reduce_dtype,
        }
    }
}

fn parse_reduce_dims(
    graph: &Graph,
    node: &Node,
    store: &TensorStore,
    input_node: Option<&Node>,
) -> Vec<i32> {
    let mut dims = arg_i32_list(node, 1);
    if dims.is_empty() {
        if let Some(Arg::Int(dim)) = node.args.get(1) {
            dims.push(*dim as i32);
        }
    }
    if dims.is_empty() {
        dims = input_node
            .map(|n| {
                actual_tensor_shape(graph, store, n)
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| idx as i32)
                    .collect()
            })
            .unwrap_or_else(|| vec![0]);
    }
    dims
}

enum ReduceExtremaKind {
    Any,
    Max,
    Min,
}

/// Emit the dequantize-to-float chain: Cast -> Sub(zp) -> Mul(scale).
///
/// Given a quantised INT8/INT16 input, this appends up to three TOSA ops that
/// convert it to a floating-point representation:
///   1. Cast to `target_dtype`
///   2. Sub(zero_point)  – skipped when zero_point == 0
///   3. Mul(scale)       – skipped when scale == 1.0
///
/// Returns the name of the tensor produced by the last op in the chain.
fn emit_quant_to_float_chain(
    result: &mut EmitResult,
    node_name: &str,
    input_name: String,
    input_shape: &[i32],
    ir_dtype: crate::graph::ir::DType,
    target_dtype: DType,
    scale: f64,
    zero_point: f64,
    prefix: &str,
) -> String {
    // 1. Cast
    let cast_name = format!("{}_{}_input_cast", node_name, prefix);
    result.push_output_op(
        TosaTensor::new(&cast_name, input_shape.to_vec(), target_dtype),
        Op::Cast,
        Attribute::Cast(CastAttribute {}),
        vec![input_name],
    );
    let mut current = cast_name;

    // 2. Sub zero-point (when non-zero)
    if zero_point != 0.0 {
        let zp_name = format!("{}_{}_input_zp", node_name, prefix);
        if let Some(zp_tensor) =
            make_filled_const_tensor(zp_name.clone(), input_shape.to_vec(), ir_dtype, zero_point)
        {
            push_inline_const(result, zp_tensor);
            let sub_name = format!("{}_{}_input_sub", node_name, prefix);
            result.push_output_op(
                TosaTensor::new(&sub_name, input_shape.to_vec(), target_dtype),
                Op::Sub,
                Attribute::Sub(SubAttribute {}),
                vec![current, zp_name],
            );
            current = sub_name;
        }
    }

    // 3. Mul scale (when != 1.0)
    if (scale - 1.0).abs() > f64::EPSILON {
        let scale_name = format!("{}_{}_input_scale", node_name, prefix);
        if let Some(scale_tensor) =
            make_filled_const_tensor(scale_name.clone(), input_shape.to_vec(), ir_dtype, scale)
        {
            push_inline_const(result, scale_tensor);
            let shift_name = push_shift_const(result, &format!("{}_{}_input", node_name, prefix));
            let mul_name = format!("{}_{}_input_float", node_name, prefix);
            result.push_output_op(
                TosaTensor::new(&mul_name, input_shape.to_vec(), target_dtype),
                Op::Mul,
                Attribute::Mul(MulAttribute { shift: 0 }),
                vec![current, scale_name, shift_name],
            );
            current = mul_name;
        }
    }

    current
}

/// Emit a per-axis reduction loop, optionally followed by a reshape when
/// `keepdim` is false.
///
/// Each iteration collapses one axis of the current tensor via `op` +
/// `make_attr(tosa_axis)`.  The caller controls naming through:
///
///   - `intermediate_prefix` – used to build names for non-final reduce outputs
///     (e.g. `"reduce_axis"` → `"{node}_reduce_axis{raw_dim}"`).
///   - `final_name` – the tensor name to use for the last reduce output.
///   - `final_on_last` – if `true`, `final_name` is used unconditionally on
///     the last iteration; if `false`, it is only used when `keepdim` is set.
///
/// Returns `(last_tensor_name, last_shape)` so the caller can do any
/// post-processing (e.g. multiply by reciprocal for mean, reshape, etc.).
fn emit_multi_axis_reduce_loop(
    result: &mut EmitResult,
    node_name: &str,
    axes: &[ReduceAxis],
    keepdim: bool,
    input_name: String,
    input_shape: Vec<i32>,
    dtype: DType,
    op: Op,
    make_attr: impl Fn(i32) -> Attribute,
    intermediate_prefix: &str,
    final_name: String,
    final_on_last: bool,
) -> (String, Vec<i32>) {
    let mut current_input = input_name;
    let mut current_shape = input_shape;

    for (idx, axis) in axes.iter().enumerate() {
        let mut out_shape = current_shape.clone();
        if let Some(slot) = out_shape.get_mut(axis.tosa_axis as usize) {
            *slot = 1;
        }

        let is_last = idx == axes.len() - 1;
        let use_final = if final_on_last {
            is_last
        } else {
            is_last && keepdim
        };
        let out_name = if use_final {
            final_name.clone()
        } else {
            format!("{}_{}{}", node_name, intermediate_prefix, axis.raw_dim)
        };

        result.push_output_op(
            TosaTensor::new(&out_name, out_shape.clone(), dtype),
            op,
            make_attr(axis.tosa_axis),
            vec![current_input.clone()],
        );

        current_input = out_name;
        current_shape = out_shape;
    }

    (current_input, current_shape)
}
