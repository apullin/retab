use crate::graph::ir::{Arg, Node, OpTarget};
use crate::graph::Graph;
use crate::store::TensorStore;

use super::op_semantics;
use super::tensor_layout::{
    apply_tosa_dim_order, infer_shape_permutation, inverse_tosa_dim_order, node_tensor_dtype,
    permute_shape, raw_i32_shape,
};
use super::tosa_generated::{Attribute, Op, TosaOperator, TosaTensor, TransposeAttribute};
use super::visitors;

pub(super) fn append_output_names(graph: &Graph, arg: &Arg, output_names: &mut Vec<String>) {
    match arg {
        Arg::Node(id) => {
            if let Some(node) = graph.node(*id) {
                output_names.push(node.name.clone());
            }
        }
        Arg::NodeList(ids) => {
            for id in ids {
                if let Some(node) = graph.node(*id) {
                    output_names.push(node.name.clone());
                }
            }
        }
        _ => {}
    }
}

pub(super) fn append_normalized_output_names(
    graph: &Graph,
    store: &TensorStore,
    arg: &Arg,
    tensors: &mut Vec<TosaTensor>,
    operators: &mut Vec<TosaOperator>,
    output_names: &mut Vec<String>,
) {
    let mut append_one = |node: &Node| {
        let raw_shape = raw_i32_shape(node);
        let serialized_shape = apply_tosa_dim_order(node, raw_shape.clone());
        let actual_shape = visitors::actual_tensor_shape(graph, store, node);
        let preserves_non_identity_layout = has_non_identity_tosa_dim_order(node);
        let has_explicit_output_layout = op_semantics::is_tosa_transpose(&node.target);
        if let Some((output_shape, perms)) =
            control_flow_parent_output_layout(graph, node, &raw_shape, &serialized_shape)
        {
            push_output_layout_transpose(
                tensors,
                operators,
                output_names,
                node,
                output_shape,
                perms,
            );
            return;
        }
        // External outputs that still advertise a non-identity dim order are
        // intentionally user-visible channels-last tensors. Keep those in the
        // serializer's native layout so the runner can decode them using the
        // FX fake tensor metadata instead of fighting a hidden extra transpose.
        if raw_shape.len() != 4
            || actual_shape == raw_shape
            || preserves_non_identity_layout
            || has_explicit_output_layout
        {
            output_names.push(node.name.clone());
            return;
        }

        let Some(perms) =
            normalized_output_layout_perms(node, &raw_shape, &serialized_shape, &actual_shape)
        else {
            output_names.push(node.name.clone());
            return;
        };

        push_output_layout_transpose(tensors, operators, output_names, node, raw_shape, perms);
    };

    match arg {
        Arg::Node(id) => {
            if let Some(node) = graph.node(*id) {
                append_one(node);
            }
        }
        Arg::NodeList(ids) => {
            for id in ids {
                if let Some(node) = graph.node(*id) {
                    append_one(node);
                }
            }
        }
        _ => append_output_names(graph, arg, output_names),
    }
}

fn has_non_identity_tosa_dim_order(node: &Node) -> bool {
    node.meta
        .tosa_dim_order
        .as_ref()
        .map(|order| {
            order
                .iter()
                .enumerate()
                .any(|(idx, &axis)| axis as usize != idx)
        })
        .unwrap_or(false)
}

fn output_layout_name(node: &Node) -> String {
    format!("{}_output_layout", node.name)
}

fn push_output_layout_transpose(
    tensors: &mut Vec<TosaTensor>,
    operators: &mut Vec<TosaOperator>,
    output_names: &mut Vec<String>,
    node: &Node,
    output_shape: Vec<i32>,
    perms: Vec<i32>,
) {
    let output_name = output_layout_name(node);
    tensors.push(TosaTensor::new(
        &output_name,
        output_shape,
        node_tensor_dtype(node),
    ));
    operators.push(TosaOperator::new(
        Op::Transpose,
        Attribute::Transpose(TransposeAttribute { perms }),
        vec![node.name.clone()],
        vec![output_name.clone()],
    ));
    output_names.push(output_name);
}

fn control_flow_parent_output_layout(
    graph: &Graph,
    node: &Node,
    raw_shape: &[i32],
    serialized_shape: &[i32],
) -> Option<(Vec<i32>, Vec<i32>)> {
    if !node.meta.arm_control_flow_parent_output_user || raw_shape.len() != 4 {
        return None;
    }
    if is_cond_parent_output(graph, node) {
        return None;
    }
    let perms = node
        .meta
        .tosa_dim_order
        .as_ref()
        .and_then(|order| inverse_tosa_dim_order(order))?;
    let output_shape = permute_shape(serialized_shape, &perms)?;
    Some((output_shape, perms))
}

fn is_cond_parent_output(graph: &Graph, node: &Node) -> bool {
    let Some(Arg::Node(source_id)) = node.args.first() else {
        return false;
    };
    graph
        .node(*source_id)
        .is_some_and(|source| matches!(source.target, OpTarget::CondIf))
}

fn normalized_output_layout_perms(
    node: &Node,
    raw_shape: &[i32],
    serialized_shape: &[i32],
    actual_shape: &[i32],
) -> Option<Vec<i32>> {
    node.meta
        .tosa_dim_order
        .as_ref()
        .and_then(|order| {
            (serialized_shape == actual_shape)
                .then(|| inverse_tosa_dim_order(order))
                .flatten()
        })
        .or_else(|| infer_shape_permutation(actual_shape, raw_shape))
}
