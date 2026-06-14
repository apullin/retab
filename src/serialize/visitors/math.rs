use super::emitter::EmitContext;
use super::*;

mod derived;
mod unary;

pub fn register(table: &mut EmitterTable) {
    for register in [unary::register as fn(&mut EmitterTable), derived::register] {
        register(table);
    }
}

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
        let input_name = input_name(graph, node, 0)?;
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
        let inputs = collect_binary_input_names(graph, store, node, result);
        Some(Self {
            lhs: inputs.first()?.clone(),
            rhs: inputs.get(1)?.clone(),
            output_shape: actual_tensor_shape(graph, store, node),
            base_dtype: node_output_dtype(node),
        })
    }
}
