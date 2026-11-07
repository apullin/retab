use super::emitter::EmitContext;
use super::*;

mod conv;
mod pad;
mod pool;
mod rescale;
mod transpose_conv;

pub fn register(table: &mut EmitterTable) {
    for register in [
        conv::register as fn(&mut EmitterTable),
        pad::register,
        pool::register,
        rescale::register,
        transpose_conv::register,
    ] {
        register(table);
    }
}

fn push_input_weight_zp_tensors(
    result: &mut EmitResult,
    graph: &Graph,
    node: &Node,
    input_zp_val: i32,
    weight_zp_val: i32,
) -> (String, String) {
    let input_dtype = node_tensor_dtype(graph, node, 0);
    let weight_dtype = node_tensor_dtype(graph, node, 1);
    let input_zp_name = format!("{}_input_zp", node.name);
    push_zero_point_const(result, input_zp_name.clone(), input_dtype, input_zp_val);

    let weight_zp_name = format!("{}_weight_zp", node.name);
    push_zero_point_const(result, weight_zp_name.clone(), weight_dtype, weight_zp_val);

    (input_zp_name, weight_zp_name)
}

/// The input and weight zero points: the input qparams', else the `input_zp` and
/// `weight_zp` kwargs, else 0.
fn input_weight_zero_points(node: &Node) -> (i32, i32) {
    let input_zp =
        input_qparam_zero_point(node, 0).unwrap_or_else(|| kwarg_i32(node, "input_zp", 0));
    let weight_zp =
        input_qparam_zero_point(node, 1).unwrap_or_else(|| kwarg_i32(node, "weight_zp", 0));
    (input_zp, weight_zp)
}

fn extend_conv_inputs(
    mut input_names: Vec<String>,
    input_zp_name: String,
    weight_zp_name: String,
) -> Vec<String> {
    input_names.truncate(3);
    input_names.push(input_zp_name);
    input_names.push(weight_zp_name);
    input_names
}

fn prepare_conv_inputs(
    result: &mut EmitResult,
    graph: &Graph,
    node: &Node,
    mut input_names: Vec<String>,
    acc_type: DType,
    input_zp: i32,
    weight_zp: i32,
) -> Vec<String> {
    if matches!(input_names.len(), 2 | 4) {
        input_names.insert(
            2,
            push_zero_bias_tensor(result, node, synthetic_bias_dtype(node, acc_type)),
        );
    }
    if input_names.len() >= 5 {
        return input_names;
    }
    let (input_zp_name, weight_zp_name) =
        push_input_weight_zp_tensors(result, graph, node, input_zp, weight_zp);
    extend_conv_inputs(input_names, input_zp_name, weight_zp_name)
}

fn synthetic_bias_dtype(node: &Node, acc_type: DType) -> DType {
    let semantic_dtype = node_output_dtype(node);
    if node.meta.input_qparams.is_empty() && is_float_dtype(semantic_dtype) {
        semantic_dtype
    } else {
        acc_type
    }
}

/// A TOSA conv's output tensor. Its val is the TOSA output shape; only the shape comes
/// from it, the dtype stays the accumulator's, because the TOSA conv inherits the aten
/// conv's int8 output qparams (P2 design G1, critique A9).
fn conv_output_tensor(node: &Node) -> TosaTensor {
    let semantic_dtype = node_output_dtype(node);
    let dtype = if node.meta.input_qparams.is_empty() && is_float_dtype(semantic_dtype) {
        semantic_dtype
    } else {
        conv_accumulator_dtype(node)
    };
    TosaTensor::new(&node.name, raw_i32_shape(node), dtype)
}

/// A TOSA pool's output tensor: its val is the TOSA output shape (P2 design G1,
/// critique B1).
fn pool_output_tensor(node: &Node) -> TosaTensor {
    TosaTensor::new(&node.name, raw_i32_shape(node), node_output_dtype(node))
}

/// A TOSA conv's val is channels-last.
fn output_channels(node: &Node) -> i32 {
    node.meta.val.shape_slice().last().copied().unwrap_or(0) as i32
}

fn zero_bytes(dtype: DType, len: usize) -> Vec<u8> {
    vec![0; tosa_dtype_storage_size(dtype) * len]
}

fn push_zero_bias_tensor(result: &mut EmitResult, node: &Node, dtype: DType) -> String {
    let channels = output_channels(node).max(0) as usize;
    result.push_const(TosaTensor::with_data(
        &format!("{}_bias", node.name),
        vec![channels as i32],
        dtype,
        zero_bytes(dtype, channels),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{DType as IrDType, NodeMeta, TensorInfo};
    use smallvec::{smallvec, SmallVec};

    fn add(
        graph: &mut Graph,
        name: &str,
        op: OpType,
        target: OpTarget,
        args: SmallVec<[Arg; 4]>,
        shape: &[i64],
    ) -> NodeId {
        graph.create_node(
            op,
            target,
            args,
            Vec::new(),
            NodeMeta::new(TensorInfo::new(shape, IrDType::Int8)),
            name.into(),
        )
    }

    fn input(graph: &mut Graph, name: &str, shape: &[i64]) -> NodeId {
        add(
            graph,
            name,
            OpType::Placeholder,
            OpTarget::Other(name.into()),
            smallvec![],
            shape,
        )
    }

    fn list(values: &[i64]) -> Arg {
        Arg::IntList(values.iter().copied().collect())
    }

    /// `<target>(x', w, b, stride, pad, [1, 1])` on a channels-last input.
    fn tosa_conv(
        target: OpTarget,
        input_shape: &[i64],
        weight: &[i64],
        stride: &[i64],
        pad: &[i64],
        val: &[i64],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        let x = input(&mut graph, "x", input_shape);
        let w = input(&mut graph, "w", weight);
        let b = input(&mut graph, "b", &val[3..]);
        let conv = add(
            &mut graph,
            "conv",
            OpType::CallFunction,
            target,
            smallvec![
                Arg::Node(x),
                Arg::Node(w),
                Arg::Node(b),
                list(stride),
                list(pad),
                list(&[1, 1])
            ],
            val,
        );
        (graph, conv)
    }

    fn emit(graph: &Graph, id: NodeId) -> EmitResult {
        clear_shape_cache();
        emit_node(graph, id, graph.node(id).unwrap(), &TensorStore::new()).unwrap()
    }

    #[test]
    fn conv_pads_are_used_as_given() {
        // P2 design G2: rc4 already adjusted the pad, so [1, 1, 2, 2] stays as it is, and
        // the output shape is the val (G1).
        let (graph, conv) = tosa_conv(
            OpTarget::TosaConv2d,
            &[1, 6, 10, 3],
            &[8, 3, 5, 3],
            &[2, 2],
            &[1, 1, 2, 2],
            &[1, 3, 5, 8],
        );
        let result = emit(&graph, conv);
        let op = result.operators.last().unwrap();
        let Attribute::Conv2d(attr) = &op.attribute else {
            panic!("expected CONV2D, got {:?}", op.op);
        };
        assert_eq!(attr.pad, vec![1, 1, 2, 2]);
        assert_eq!(result.tensors.last().unwrap().shape, vec![1, 3, 5, 8]);
    }

    #[test]
    fn pool_outputs_are_the_val() {
        // P2 design G1: the output shape is the val, not recomputed from the arguments,
        // which would give [1, 4, 4, 4].
        let mut graph = Graph::new();
        let x = input(&mut graph, "x", &[1, 9, 9, 4]);
        let pool = add(
            &mut graph,
            "pool",
            OpType::CallFunction,
            OpTarget::TosaMaxPool2d,
            smallvec![
                Arg::Node(x),
                list(&[2, 2]),
                list(&[2, 2]),
                list(&[0, 0, 0, 0])
            ],
            &[1, 5, 5, 4],
        );
        let result = emit(&graph, pool);
        assert_eq!(result.operators.last().unwrap().op, Op::MaxPool2d);
        assert_eq!(result.tensors.last().unwrap().shape, vec![1, 5, 5, 4]);
    }
}
