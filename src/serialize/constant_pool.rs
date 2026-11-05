//! Per-block constant pooling, mirroring upstream `TosaSerializerWithConstantPool`
//! (backends/arm/tosa/constant_pool.py, ExecuTorch v1.6).
//!
//! Upstream pools every constant a node visitor adds with `addConst`: within one TOSA
//! basic block, a constant with the same dtype, shape and data as an earlier one is not
//! emitted again, and its users read the earlier tensor instead. Graph constants
//! (parameters, buffers, pass-materialized constants) go through `addUnpooledConst` so
//! their names stay addressable. RETAB mirrors the split: only constants a visitor
//! registers as poolable (`EmitResult::push_const`, `push_const_shape`) are pooled.

use std::collections::{HashMap, HashSet};

use super::tosa_generated::{DType, Op};
use super::visitors::EmitResult;

#[derive(Default)]
pub(crate) struct ConstantPool {
    tensors: HashMap<(DType, Vec<i32>, Vec<u8>), String>,
    shapes: HashMap<(u32, Vec<u8>), String>,
}

impl ConstantPool {
    /// Drop `result`'s poolable constants that the block already holds and point their
    /// users at the pooled copy. Generated constant names are local to one node's
    /// emission, so rewriting `result`'s own operators covers every user.
    pub(crate) fn absorb(&mut self, result: &mut EmitResult) {
        let poolable: HashSet<String> = std::mem::take(&mut result.poolable).into_iter().collect();
        if poolable.is_empty() {
            return;
        }
        let mut renames: HashMap<String, String> = HashMap::new();
        result.tensors.retain(|tensor| {
            if !poolable.contains(&tensor.name) {
                return true;
            }
            let key = (
                tensor.dtype,
                tensor.shape.clone(),
                tensor.data.clone().unwrap_or_default(),
            );
            match self.tensors.get(&key) {
                Some(pooled) => {
                    renames.insert(tensor.name.clone(), pooled.clone());
                    false
                }
                None => {
                    self.tensors.insert(key, tensor.name.clone());
                    true
                }
            }
        });
        result.shapes.retain(|shape| {
            if !poolable.contains(&shape.name) {
                return true;
            }
            let key = (shape.rank, shape.data.clone());
            match self.shapes.get(&key) {
                Some(pooled) => {
                    renames.insert(shape.name.clone(), pooled.clone());
                    false
                }
                None => {
                    self.shapes.insert(key, shape.name.clone());
                    true
                }
            }
        });
        if renames.is_empty() {
            return;
        }
        result.operators.retain(|op| {
            !(matches!(op.op, Op::Const | Op::ConstShape)
                && op.outputs.iter().all(|output| renames.contains_key(output)))
        });
        for op in &mut result.operators {
            for input in &mut op.inputs {
                if let Some(pooled) = renames.get(input) {
                    *input = pooled.clone();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::tosa_generated::{Attribute, ConstAttribute, TosaTensor};

    fn zp_result(node: &str, zp: i8) -> EmitResult {
        let mut result = EmitResult::default();
        let mut zp_tensor = TosaTensor::new(&format!("{node}_zp"), vec![1], DType::Int8);
        zp_tensor.data = Some(vec![zp as u8]);
        let zp_name = result.push_const(zp_tensor);
        result.push_op(
            Op::Identity,
            Attribute::Const(ConstAttribute {}),
            vec!["x".into(), zp_name],
            vec![format!("{node}_out")],
        );
        result
    }

    #[test]
    fn pools_identical_generated_constants_within_a_block() {
        let mut pool = ConstantPool::default();
        let mut first = zp_result("a", 0);
        let mut second = zp_result("b", 0);
        pool.absorb(&mut first);
        pool.absorb(&mut second);
        assert_eq!(first.tensors.len(), 1);
        assert!(second.tensors.is_empty());
        assert_eq!(
            second.operators.len(),
            1,
            "the duplicate CONST op is dropped"
        );
        assert_eq!(second.operators[0].inputs[1], "a_zp");
    }

    #[test]
    fn keeps_constants_with_different_data() {
        let mut pool = ConstantPool::default();
        let mut first = zp_result("a", 0);
        let mut second = zp_result("b", 3);
        pool.absorb(&mut first);
        pool.absorb(&mut second);
        assert_eq!(second.tensors.len(), 1);
        assert_eq!(second.operators[1].inputs[1], "b_zp");
    }

    #[test]
    fn leaves_unregistered_constants_alone() {
        let mut pool = ConstantPool::default();
        let mut first = zp_result("a", 0);
        let mut second = zp_result("b", 0);
        second.poolable.clear();
        pool.absorb(&mut first);
        pool.absorb(&mut second);
        assert_eq!(second.tensors.len(), 1);
    }
}
