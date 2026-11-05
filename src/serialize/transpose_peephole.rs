//! Block-level counterparts of two rc4 shared transforms (`backends/transforms`):
//! `FuseCascadedTransposeOrPermuteOps`, which drops a TRANSPOSE that undoes the TRANSPOSE
//! feeding it, and `ReplaceNopTransposeOrPermuteWithView`, which turns a TRANSPOSE that
//! only moves size-1 dims into a RESHAPE.
//!
//! RETAB's layout pass gives every rank-4 tensor a channels-last order and surrounds the
//! non-spatial consumers of those tensors with transposes. rc4's layout code never
//! creates them, so they are cleaned up here, after emission, where every shape is the
//! concrete TOSA shape.

use std::collections::{HashMap, HashSet};

use super::tosa_types::{
    Attribute, ConstShapeAttribute, Op, ReshapeAttribute, TosaOperator, TosaShape, TosaTensor,
};

fn transpose_perms(op: &TosaOperator) -> Option<&[i32]> {
    match (&op.op, &op.attribute) {
        (Op::Transpose, Attribute::Transpose(attr)) => Some(&attr.perms),
        _ => None,
    }
}

fn is_identity_composition(first: &[i32], second: &[i32]) -> bool {
    first.len() == second.len()
        && second
            .iter()
            .enumerate()
            .all(|(axis, &p)| first.get(p as usize) == Some(&(axis as i32)))
}

/// Upstream `_is_singleton_permutation`: the non-singleton axes keep their order.
fn is_singleton_permutation(input_shape: &[i32], perms: &[i32]) -> bool {
    if input_shape.len() != perms.len() {
        return false;
    }
    let kept: Vec<i32> = (0..input_shape.len() as i32)
        .filter(|&axis| input_shape[axis as usize] != 1)
        .collect();
    let permuted: Vec<i32> = perms
        .iter()
        .copied()
        .filter(|&axis| input_shape.get(axis as usize).is_some_and(|&d| d != 1))
        .collect();
    kept == permuted
}

fn use_counts(operators: &[TosaOperator], output_names: &[String]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for name in operators
        .iter()
        .flat_map(|op| op.inputs.iter())
        .chain(output_names)
    {
        *counts.entry(name.clone()).or_insert(0) += 1;
    }
    counts
}

/// A TRANSPOSE with identity perms only copies its input, so its readers read the
/// input. RETAB's layout islands leave these where a permute is a physical no-op.
fn drop_identity_transposes(
    operators: &mut Vec<TosaOperator>,
    input_names: &[String],
    output_names: &mut [String],
) -> HashSet<String> {
    let mut removed = HashSet::new();
    let mut index = 0;
    while index < operators.len() {
        let op = &operators[index];
        let identity = transpose_perms(op)
            .is_some_and(|perms| perms.iter().enumerate().all(|(axis, &p)| p == axis as i32));
        let (Some(source), Some(output)) = (op.inputs.first().cloned(), op.outputs.first().cloned())
        else {
            index += 1;
            continue;
        };
        // A block output must not alias a block input.
        if !identity || (input_names.contains(&source) && output_names.contains(&output)) {
            index += 1;
            continue;
        }
        for name in operators
            .iter_mut()
            .flat_map(|op| op.inputs.iter_mut())
            .chain(output_names.iter_mut())
        {
            if *name == output {
                *name = source.clone();
            }
        }
        operators.remove(index);
        removed.insert(output);
    }
    removed
}

/// A RESHAPE to the shape it already has only copies its input. Layout islands leave
/// these where a reshape of NHWC data lands on the same physical shape.
fn drop_identity_reshapes(
    tensors: &[TosaTensor],
    operators: &mut Vec<TosaOperator>,
    input_names: &[String],
    output_names: &mut [String],
) -> HashSet<String> {
    let tensor_shapes: HashMap<&str, &[i32]> = tensors
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();
    let mut removed = HashSet::new();
    let mut index = 0;
    while index < operators.len() {
        let op = &operators[index];
        let (Some(source), Some(output)) = (op.inputs.first().cloned(), op.outputs.first().cloned())
        else {
            index += 1;
            continue;
        };
        let same_shape = op.op == Op::Reshape
            && matches!(
                (tensor_shapes.get(source.as_str()), tensor_shapes.get(output.as_str())),
                (Some(a), Some(b)) if a == b
            );
        // A block output must not alias a block input.
        if !same_shape || (input_names.contains(&source) && output_names.contains(&output)) {
            index += 1;
            continue;
        }
        for name in operators
            .iter_mut()
            .flat_map(|op| op.inputs.iter_mut())
            .chain(output_names.iter_mut())
        {
            if *name == output {
                *name = source.clone();
            }
        }
        operators.remove(index);
        removed.insert(output);
    }
    removed
}

fn cancel_inverse_pairs(
    operators: &mut Vec<TosaOperator>,
    input_names: &[String],
    output_names: &mut [String],
) -> HashSet<String> {
    let mut removed = HashSet::new();
    loop {
        let producer: HashMap<&str, usize> = operators
            .iter()
            .enumerate()
            .flat_map(|(idx, op)| op.outputs.iter().map(move |out| (out.as_str(), idx)))
            .collect();
        let pair = operators.iter().enumerate().find_map(|(second, op)| {
            let second_perms = transpose_perms(op)?;
            let first = *producer.get(op.inputs.first()?.as_str())?;
            let first_perms = transpose_perms(&operators[first])?;
            if !is_identity_composition(first_perms, second_perms) {
                return None;
            }
            let source = operators[first].inputs.first()?.clone();
            let output = op.outputs.first()?.clone();
            // A block output must not alias a block input.
            if input_names.contains(&source) && output_names.contains(&output) {
                return None;
            }
            Some((first, second, source, output))
        });
        let Some((first, second, source, output)) = pair else {
            return removed;
        };
        for name in operators
            .iter_mut()
            .flat_map(|op| op.inputs.iter_mut())
            .chain(output_names.iter_mut())
        {
            if *name == output {
                *name = source.clone();
            }
        }
        removed.insert(output);
        let first_output = operators[first].outputs[0].clone();
        operators.remove(second);
        if !use_counts(operators, output_names).contains_key(&first_output) {
            operators.remove(first);
            removed.insert(first_output);
        }
    }
}

fn relabel_singleton_transposes(
    tensors: &[TosaTensor],
    shapes: &mut Vec<TosaShape>,
    operators: &mut Vec<TosaOperator>,
) -> usize {
    let tensor_shapes: HashMap<&str, &[i32]> = tensors
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();
    let mut rewritten = Vec::new();
    for (idx, op) in operators.iter().enumerate() {
        let Some(perms) = transpose_perms(op) else {
            continue;
        };
        let (Some(input), Some(output)) = (op.inputs.first(), op.outputs.first()) else {
            continue;
        };
        let (Some(input_shape), Some(output_shape)) = (
            tensor_shapes.get(input.as_str()),
            tensor_shapes.get(output.as_str()),
        ) else {
            continue;
        };
        if is_singleton_permutation(input_shape, perms) {
            rewritten.push((idx, input.clone(), output.clone(), output_shape.to_vec()));
        }
    }
    let count = rewritten.len();
    // One CONST_SHAPE per distinct shape, placed before its first use.
    let mut shape_names: HashMap<Vec<i32>, String> = HashMap::new();
    let mut inserted = 0;
    for (idx, input, output, output_shape) in rewritten {
        let shape_name = match shape_names.get(&output_shape) {
            Some(name) => name.clone(),
            None => {
                let name = format!("{output}_relabel_shape");
                shapes.push(TosaShape {
                    name: name.clone(),
                    rank: output_shape.len() as u32,
                    data: output_shape
                        .iter()
                        .flat_map(|&d| (d as i64).to_le_bytes())
                        .collect(),
                });
                operators.insert(
                    idx + inserted,
                    TosaOperator::new(
                        Op::ConstShape,
                        Attribute::ConstShape(ConstShapeAttribute {}),
                        vec![],
                        vec![name.clone()],
                    ),
                );
                inserted += 1;
                shape_names.insert(output_shape, name.clone());
                name
            }
        };
        operators[idx + inserted] = TosaOperator::new(
            Op::Reshape,
            Attribute::Reshape(ReshapeAttribute {}),
            vec![input, shape_name],
            vec![output],
        );
    }
    count
}

fn read_names(operators: &[TosaOperator]) -> HashSet<String> {
    operators
        .iter()
        .flat_map(|op| op.inputs.iter().cloned())
        .collect()
}

/// Drop the CONST_SHAPE ops, and their shapes, that only the removed RESHAPEs read.
fn drop_orphaned_const_shapes(
    shapes: &mut Vec<TosaShape>,
    operators: &mut Vec<TosaOperator>,
    read_before: &HashSet<String>,
) {
    let read_now = read_names(operators);
    let orphaned: HashSet<String> = read_before.difference(&read_now).cloned().collect();
    operators.retain(|op| {
        op.op != Op::ConstShape || !op.outputs.iter().any(|out| orphaned.contains(out))
    });
    shapes.retain(|shape| !orphaned.contains(&shape.name));
}

/// Remove redundant transposes from one block. Returns the number of TRANSPOSE ops
/// removed or rewritten.
pub(super) fn simplify_transposes(
    tensors: &mut Vec<TosaTensor>,
    shapes: &mut Vec<TosaShape>,
    operators: &mut Vec<TosaOperator>,
    input_names: &[String],
    output_names: &mut [String],
) -> usize {
    let mut removed = drop_identity_transposes(operators, input_names, output_names);
    removed.extend(cancel_inverse_pairs(operators, input_names, output_names));
    let relabeled = relabel_singleton_transposes(tensors, shapes, operators);
    let read_before = read_names(operators);
    removed.extend(drop_identity_reshapes(tensors, operators, input_names, output_names));
    tensors.retain(|t| !removed.contains(&t.name));
    drop_orphaned_const_shapes(shapes, operators, &read_before);
    removed.len() + relabeled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::tosa_types::{DType, TransposeAttribute};

    fn tensor(name: &str, shape: &[i32]) -> TosaTensor {
        TosaTensor::new(name, shape.to_vec(), DType::Int8)
    }

    #[test]
    fn same_shape_reshapes_and_their_shapes_are_dropped() {
        let tensors = vec![tensor("x", &[1, 1, 1, 8]), tensor("r", &[1, 1, 1, 8]), tensor("y", &[1, 1, 1, 8])];
        let mut shapes = vec![TosaShape { name: "s".into(), rank: 4, data: vec![] }];
        let mut operators = vec![
            TosaOperator::new(
                Op::ConstShape,
                Attribute::ConstShape(ConstShapeAttribute {}),
                vec![],
                vec!["s".into()],
            ),
            TosaOperator::new(
                Op::Reshape,
                Attribute::Reshape(ReshapeAttribute {}),
                vec!["x".into(), "s".into()],
                vec!["r".into()],
            ),
            TosaOperator::new(
                Op::Abs,
                Attribute::Abs(crate::serialize::tosa_types::AbsAttribute {}),
                vec!["r".into()],
                vec!["y".into()],
            ),
        ];
        let mut outputs = vec!["y".to_string()];
        let read_before = read_names(&operators);
        let removed = drop_identity_reshapes(&tensors, &mut operators, &[], &mut outputs);
        drop_orphaned_const_shapes(&mut shapes, &mut operators, &read_before);
        assert_eq!(removed, HashSet::from(["r".to_string()]));
        assert_eq!(operators.len(), 1);
        assert_eq!(operators[0].inputs, vec!["x".to_string()]);
        assert!(shapes.is_empty());
    }

    #[test]
    fn identity_transposes_are_dropped_unless_they_link_block_input_and_output() {
        let mut operators = vec![
            transpose("x", "t0", &[0, 1, 2, 3]),
            TosaOperator::new(
                Op::Abs,
                Attribute::Abs(crate::serialize::tosa_types::AbsAttribute {}),
                vec!["t0".into()],
                vec!["y".into()],
            ),
            transpose("x", "z", &[0, 1, 2, 3]),
        ];
        let mut outputs = vec!["y".to_string(), "z".to_string()];
        let removed = drop_identity_transposes(&mut operators, &["x".to_string()], &mut outputs);
        assert_eq!(removed, HashSet::from(["t0".to_string()]));
        assert_eq!(operators.len(), 2);
        assert_eq!(operators[0].inputs, vec!["x".to_string()]);
        assert_eq!(outputs, vec!["y".to_string(), "z".to_string()]);
    }

    fn transpose(input: &str, output: &str, perms: &[i32]) -> TosaOperator {
        TosaOperator::new(
            Op::Transpose,
            Attribute::Transpose(TransposeAttribute {
                perms: perms.to_vec(),
            }),
            vec![input.to_string()],
            vec![output.to_string()],
        )
    }

    #[test]
    fn detects_identity_compositions_and_singleton_moves() {
        assert!(is_identity_composition(&[0, 2, 3, 1], &[0, 3, 1, 2]));
        assert!(!is_identity_composition(&[0, 2, 3, 1], &[0, 2, 3, 1]));
        assert!(is_singleton_permutation(&[12, 3, 7, 1], &[0, 3, 1, 2]));
        assert!(!is_singleton_permutation(&[1, 2, 8, 1], &[0, 2, 3, 1]));
    }

    #[test]
    fn cancels_a_round_trip_and_rewires_the_block_output() {
        let mut tensors = vec![
            tensor("x", &[1, 12, 3, 7]),
            tensor("nhwc", &[1, 3, 7, 12]),
            tensor("nchw", &[1, 12, 3, 7]),
        ];
        let mut shapes = Vec::new();
        let mut operators = vec![
            transpose("x", "nhwc", &[0, 2, 3, 1]),
            transpose("nhwc", "nchw", &[0, 3, 1, 2]),
        ];
        let mut outputs = vec!["nchw".to_string()];
        let changed = simplify_transposes(
            &mut tensors,
            &mut shapes,
            &mut operators,
            &["y".to_string()],
            &mut outputs,
        );
        assert_eq!(changed, 2);
        assert!(operators.is_empty());
        assert_eq!(outputs, vec!["x".to_string()]);
        assert_eq!(tensors.len(), 1);
    }

    #[test]
    fn keeps_a_round_trip_whose_first_half_has_other_users() {
        let mut tensors = vec![
            tensor("x", &[1, 12, 3, 7]),
            tensor("nhwc", &[1, 3, 7, 12]),
            tensor("nchw", &[1, 12, 3, 7]),
        ];
        let mut operators = vec![
            transpose("x", "nhwc", &[0, 2, 3, 1]),
            transpose("nhwc", "nchw", &[0, 3, 1, 2]),
        ];
        let mut outputs = vec!["nchw".to_string(), "nhwc".to_string()];
        simplify_transposes(
            &mut tensors,
            &mut Vec::new(),
            &mut operators,
            &["y".to_string()],
            &mut outputs,
        );
        assert_eq!(operators.len(), 1);
        assert_eq!(outputs, vec!["x".to_string(), "nhwc".to_string()]);
    }

    #[test]
    fn rewrites_a_singleton_move_as_reshape() {
        let mut tensors = vec![tensor("x", &[12, 3, 7, 1]), tensor("y", &[12, 1, 3, 7])];
        let mut shapes = Vec::new();
        let mut operators = vec![transpose("x", "y", &[0, 3, 1, 2])];
        let mut outputs = vec!["y".to_string()];
        let changed =
            simplify_transposes(&mut tensors, &mut shapes, &mut operators, &[], &mut outputs);
        assert_eq!(changed, 1);
        assert_eq!(operators[0].op, Op::ConstShape);
        assert_eq!(operators[1].op, Op::Reshape);
        assert_eq!(operators[1].inputs, vec!["x", "y_relabel_shape"]);
        assert_eq!(shapes[0].rank, 4);
    }

    #[test]
    fn shares_one_const_shape_between_equal_relabels() {
        let mut tensors = vec![
            tensor("a", &[12, 3, 7, 1]),
            tensor("b", &[12, 1, 3, 7]),
            tensor("c", &[12, 3, 7, 1]),
            tensor("d", &[12, 1, 3, 7]),
        ];
        let mut shapes = Vec::new();
        let mut operators = vec![
            transpose("a", "b", &[0, 3, 1, 2]),
            transpose("c", "d", &[0, 3, 1, 2]),
        ];
        let mut outputs = vec!["b".to_string(), "d".to_string()];
        simplify_transposes(&mut tensors, &mut shapes, &mut operators, &[], &mut outputs);
        let ops: Vec<Op> = operators.iter().map(|op| op.op).collect();
        assert_eq!(ops, vec![Op::ConstShape, Op::Reshape, Op::Reshape]);
        assert_eq!(operators[2].inputs, vec!["c", "b_relabel_shape"]);
        assert_eq!(shapes.len(), 1);
    }
}
