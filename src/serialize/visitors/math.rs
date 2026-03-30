use super::*;
use super::emitter::EmitContext;

mod derived;
mod unary;

pub(super) use derived::{
    emit_aten_div, emit_aten_expm1, emit_aten_floor_divide, emit_aten_leaky_relu, emit_aten_log1p,
    emit_aten_sign, emit_aten_sqrt, emit_full_like,
};
pub(super) use unary::{emit_ceil, emit_floor, emit_negate, emit_round};

struct ParsedUnaryTensorSpec {
    input_name: String,
    output_shape: Vec<i32>,
    base_dtype: DType,
    ir_dtype: crate::graph::ir::DType,
}

impl ParsedUnaryTensorSpec {
    // Parse once at the module boundary, then emit from a typed view instead
    // of repeatedly fishing shape/dtype/name facts out of raw node args.
    fn parse(graph: &Graph, store: &TensorStore, node: &Node) -> Option<Self> {
        let input_name = collect_input_names(graph, node).first().cloned()?;
        Some(Self {
            input_name,
            output_shape: actual_tensor_shape(graph, store, node),
            base_dtype: node_output_dtype(node),
            ir_dtype: node.meta.val.dtype,
        })
    }
}

struct ParsedBinaryTensorSpec {
    lhs: String,
    rhs: String,
    output_shape: Vec<i32>,
    base_dtype: DType,
}

impl ParsedBinaryTensorSpec {
    fn parse(
        graph: &Graph,
        store: &TensorStore,
        node: &Node,
        result: &mut EmitResult,
    ) -> Option<Self> {
        let mut inputs = collect_input_names(graph, node);
        maybe_materialize_binary_broadcast(graph, store, node, result, &mut inputs);
        Some(Self {
            lhs: inputs.first()?.clone(),
            rhs: inputs.get(1)?.clone(),
            output_shape: actual_tensor_shape(graph, store, node),
            base_dtype: node_output_dtype(node),
        })
    }
}

fn is_integral_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Int8 | DType::Int16 | DType::Int32)
}

fn emit_identity(graph: &Graph, node: &Node) -> EmitResult {
    let mut result = EmitResult::default();
    let input_names = collect_input_names(graph, node);
    result.tensors.push(make_output_tensor(node));
    result.operators.push(TosaOperator::new(
        Op::Identity,
        Attribute::None,
        input_names,
        vec![node.name.clone()],
    ));
    result
}

fn push_shift_const(result: &mut EmitResult, name: &str) -> String {
    let shift_name = format!("{}_shift", name);
    push_inline_const(
        result,
        make_rank1_const_tensor(shift_name.clone(), DType::Int8, vec![0u8], 1),
    );
    shift_name
}
