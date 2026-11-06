//! TABLE nodes that rc4 has already inserted when its phase-6 layout passes
//! run, but RETAB still holds as the original aten op.
//!
//! `InsertTableOpsPass` (`arm_pass_manager.py:708`) replaces each node whose
//! target is in `TableOps` and that has non-empty input and output qparams
//! (`backends/arm/_passes/insert_table_ops.py:36-75` and the quantized-only
//! check in its `call`). RETAB emits TABLE only at serialization, as a lowering
//! effect (`fx_insert_table_ops_ir`), so there is no IR node for the layout
//! passes to see; they treat such a node as the TABLE it will become, not as a
//! unary elementwise op.
//!
//! rc4 also decomposes round, sqrt and sign before phase 6 (`:611-629`), which
//! RETAB lowers in the serializer. That gap is closed in the pipeline instead:
//! explicit mode runs those decompositions ahead of phase 6 (APU-157), so the
//! passes stay faithful ports.

use crate::fx_ir::{IrNode, NodeOp};


/// `TableOps.unary_table_ops` and `special_table_ops`, as the venv's rc4
/// lists them, sorted.
pub(crate) const TABLE_OPS: &[&str] = &[
    "aten.acos.default",
    "aten.acosh.default",
    "aten.asin.default",
    "aten.asinh.default",
    "aten.atan.default",
    "aten.atanh.default",
    "aten.ceil.default",
    "aten.cos.default",
    "aten.cosh.default",
    "aten.elu.default",
    "aten.erf.default",
    "aten.erfinv.default",
    "aten.exp.default",
    "aten.expm1.default",
    "aten.floor.default",
    "aten.gelu.default",
    "aten.hardsigmoid.default",
    "aten.hardswish.default",
    "aten.leaky_relu.default",
    "aten.log.default",
    "aten.log10.default",
    "aten.log1p.default",
    "aten.pow.Tensor_Scalar",
    "aten.reciprocal.default",
    "aten.remainder.Scalar",
    "aten.round.default",
    "aten.rsqrt.default",
    "aten.sigmoid.default",
    "aten.silu.default",
    "aten.sin.default",
    "aten.sinh.default",
    "aten.softplus.default",
    "aten.tan.default",
    "aten.tanh.default",
];

/// Whether rc4 would already have replaced `node` with a TABLE.
pub(crate) fn is_pending_table(node: &IrNode) -> bool {
    node.op == NodeOp::CallFunction
        && TABLE_OPS.binary_search(&node.target.as_str()).is_ok()
        && node.meta.has_nonempty_input_and_output_qparams()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_op_list_is_rc4s_sorted_34() {
        assert_eq!(TABLE_OPS.len(), 34);
        assert!(TABLE_OPS.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
