//! IrPass port of the integer flow of ExecuTorch's DecomposeDivTensorModePass
//! (v1.6.0-rc4; upstream f11805c53 / #20536).
//!
//! `aten.div.Tensor_mode` on integer tensors with rounding mode `trunc` or `floor`
//! becomes TOSA INTDIV (which truncates) on int32 operands. Floor mode then subtracts one
//! where the remainder is nonzero and the operand signs differ; the quotient's own sign
//! can't be used because a truncated quotient of 0 may still need the correction.
//! Float division keeps the serializer's reciprocal lowering.
//!
//! Upstream takes this path only for INT-profile targets that aren't the U55 subset. On
//! U55 its partitioner keeps integer division on the CPU, so the case never reaches
//! preprocess and needs no profile check here.

use crate::fx_ir::{broadcast_shapes, IrArg, IrGraph, IrPass, IrRewriteCursor, NodeId, NodeMeta};
use crate::runner_logging::log_line;

pub struct IrDecomposeDivTensorModePass;

const INTDIV_TARGET: &str = "backend.tosa.INTDIV.default";
const INTEGER_DTYPES: &[&str] = &[
    "torch.uint8",
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
];

struct IntOperand {
    id: NodeId,
    shape: Vec<i64>,
}

fn integer_operand(graph: &IrGraph, arg: Option<&IrArg>) -> Option<IntOperand> {
    let id = arg?.as_node()?;
    let dtype = graph.node(id).meta.val_dtype()?;
    INTEGER_DTYPES.contains(&dtype).then_some(())?;
    Some(IntOperand {
        id,
        shape: graph.node_shape_vec(id)?,
    })
}

struct DivBuilder<'c, 'g> {
    cursor: &'c mut IrRewriteCursor<'g>,
    name: String,
    meta: NodeMeta,
}

impl DivBuilder<'_, '_> {
    fn node(
        &mut self,
        tag: &str,
        target: &str,
        inputs: &[NodeId],
        shape: &[i64],
        dtype: &str,
    ) -> NodeId {
        self.cursor.call_function(
            format!("{}_{tag}", self.name),
            target,
            inputs.iter().map(|&id| IrArg::Node(id)).collect(),
            self.meta.cloned_with_val(shape.to_vec(), dtype),
        )
    }

    fn cast(&mut self, tag: &str, source: NodeId, shape: &[i64], dtype: &str) -> NodeId {
        let meta = self.meta.cloned_with_val(shape.to_vec(), dtype);
        self.cursor
            .dtype_cast_if_needed(format!("{}_{tag}", self.name), source, dtype, meta)
    }

    /// Upstream `_correct_intdiv_floor`.
    fn correct_floor(
        &mut self,
        a: &IntOperand,
        b: &IntOperand,
        quotient: NodeId,
        shape: &[i64],
    ) -> NodeId {
        let (int32, boolean) = ("torch.int32", "torch.bool");
        let product = self.node(
            "floor_product",
            "aten.mul.Tensor",
            &[quotient, b.id],
            shape,
            int32,
        );
        let remainder = self.node(
            "floor_remainder",
            "aten.sub.Tensor",
            &[a.id, product],
            shape,
            int32,
        );
        let unit = vec![1i64; shape.len()];
        let zero_meta = self.meta.cloned_with_val(unit.clone(), int32);
        let zero = self.cursor.full_int(
            format!("{}_floor_zero", self.name),
            &unit,
            0,
            int32,
            zero_meta,
        );
        let remainder_is_zero = self.node(
            "floor_remainder_is_zero",
            "aten.eq.Tensor",
            &[remainder, zero],
            shape,
            boolean,
        );
        let remainder_nonzero = self.node(
            "floor_remainder_nonzero",
            "aten.logical_not.default",
            &[remainder_is_zero],
            shape,
            boolean,
        );
        let a_neg_shape =
            broadcast_shapes(&[unit.clone(), a.shape.clone()]).unwrap_or_else(|| shape.to_vec());
        let b_neg_shape =
            broadcast_shapes(&[unit.clone(), b.shape.clone()]).unwrap_or_else(|| shape.to_vec());
        let signs_shape = broadcast_shapes(&[a_neg_shape.clone(), b_neg_shape.clone()])
            .unwrap_or_else(|| shape.to_vec());
        let a_negative = self.node(
            "floor_lhs_negative",
            "aten.gt.Tensor",
            &[zero, a.id],
            &a_neg_shape,
            boolean,
        );
        let b_negative = self.node(
            "floor_rhs_negative",
            "aten.gt.Tensor",
            &[zero, b.id],
            &b_neg_shape,
            boolean,
        );
        let signs_differ = self.node(
            "floor_signs_differ",
            "aten.logical_xor.default",
            &[a_negative, b_negative],
            &signs_shape,
            boolean,
        );
        let needs_correction = self.node(
            "floor_needs_correction",
            "aten.logical_and.default",
            &[remainder_nonzero, signs_differ],
            shape,
            boolean,
        );
        let correction = self.cursor.dtype_cast(
            format!("{}_floor_correction", self.name),
            needs_correction,
            int32,
            self.meta.cloned_with_val(shape.to_vec(), int32),
        );
        self.node(
            "floor",
            "aten.sub.Tensor",
            &[quotient, correction],
            shape,
            int32,
        )
    }
}

fn decompose_one(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    if node.target != "aten.div.Tensor_mode" {
        return false;
    }
    let floor = match node.arg_or_kwarg_str(2, "rounding_mode") {
        Some("floor") => true,
        Some("trunc") => false,
        _ => return false,
    };
    let (Some(a), Some(b)) = (
        integer_operand(graph, node.args.first()),
        integer_operand(graph, node.args.get(1)),
    ) else {
        return false;
    };
    let (Some(shape), Some(output_dtype)) = (
        graph.node_shape_vec(id),
        node.meta.val_dtype().map(str::to_string),
    ) else {
        return false;
    };

    let name = node.name.clone();
    let meta = node.meta.clone().without_qparams();
    let mut cursor = graph.rewrite_cursor(id);
    let mut builder = DivBuilder {
        cursor: &mut cursor,
        name,
        meta,
    };
    let a = IntOperand {
        id: builder.cast("lhs_int32", a.id, &a.shape, "torch.int32"),
        shape: a.shape,
    };
    let b = IntOperand {
        id: builder.cast("rhs_int32", b.id, &b.shape, "torch.int32"),
        shape: b.shape,
    };
    let mut quotient = builder.node(
        "intdiv",
        INTDIV_TARGET,
        &[a.id, b.id],
        &shape,
        "torch.int32",
    );
    if floor {
        quotient = builder.correct_floor(&a, &b, quotient, &shape);
    }
    let result = builder.cast("out", quotient, &shape, &output_dtype);
    cursor.replace_anchor_with(result);
    true
}

impl IrPass for IrDecomposeDivTensorModePass {
    fn name(&self) -> &str {
        "IrDecomposeDivTensorMode"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(decompose_one);
        if modified {
            log_line(format_args!(
                "[IrDecomposeDivTensorMode] lowered integer div.Tensor_mode to INTDIV"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call, has_live_target, output, placeholder};
    use std::collections::HashMap;

    fn div_graph(lhs: &str, rhs: &str, rhs_shape: &[i64], mode: &str) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let a = placeholder(&mut graph, "a", &[4, 8], lhs);
        let b = placeholder(&mut graph, "b", rhs_shape, rhs);
        let div = call(
            &mut graph,
            "div",
            "aten.div.Tensor_mode",
            vec![IrArg::Node(a), IrArg::Node(b)],
            &[4, 8],
            lhs,
        );
        graph.node_mut(div).kwargs =
            HashMap::from([("rounding_mode".to_string(), IrArg::Str(mode.to_string()))]);
        output(&mut graph, div);
        (graph, div)
    }

    fn live_targets(graph: &IrGraph) -> Vec<String> {
        graph
            .live_call_function_ids()
            .into_iter()
            .map(|id| graph.node(id).target.clone())
            .collect()
    }

    #[test]
    fn int32_trunc_is_a_single_intdiv() {
        let (mut graph, div) = div_graph("torch.int32", "torch.int32", &[4, 8], "trunc");
        assert!(IrDecomposeDivTensorModePass.run(&mut graph));
        assert!(!graph.nodes[div].alive);
        assert_eq!(live_targets(&graph), vec![INTDIV_TARGET]);
        let out = graph.output_node().unwrap().first_node_arg().unwrap();
        assert_eq!(graph.node(out).meta.val_dtype(), Some("torch.int32"));
    }

    #[test]
    fn int8_floor_casts_corrects_and_casts_back() {
        let (mut graph, _) = div_graph("torch.int8", "torch.int8", &[], "floor");
        assert!(IrDecomposeDivTensorModePass.run(&mut graph));
        assert_eq!(
            live_targets(&graph),
            vec![
                "aten._to_copy.default",
                "aten._to_copy.default",
                INTDIV_TARGET,
                "aten.mul.Tensor",
                "aten.sub.Tensor",
                "aten.full.default",
                "aten.eq.Tensor",
                "aten.logical_not.default",
                "aten.gt.Tensor",
                "aten.gt.Tensor",
                "aten.logical_xor.default",
                "aten.logical_and.default",
                "aten._to_copy.default",
                "aten.sub.Tensor",
                "aten._to_copy.default",
            ]
        );
        let rhs_negative = graph.by_name("div_floor_rhs_negative").unwrap();
        assert_eq!(rhs_negative.meta.val_shape(), Some(&[1, 1][..]));
        let out = graph.output_node().unwrap().first_node_arg().unwrap();
        assert_eq!(graph.node(out).meta.val_dtype(), Some("torch.int8"));
    }

    #[test]
    fn leaves_float_and_unrounded_division_alone() {
        let (mut graph, _) = div_graph("torch.float32", "torch.float32", &[4, 8], "floor");
        assert!(!IrDecomposeDivTensorModePass.run(&mut graph));
        let (mut graph, _) = div_graph("torch.int8", "torch.int8", &[4, 8], "none");
        assert!(!IrDecomposeDivTensorModePass.run(&mut graph));
        assert!(has_live_target(&graph, "div", "aten.div.Tensor_mode"));
    }
}
