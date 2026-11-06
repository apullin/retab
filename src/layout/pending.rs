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
//!
//! # A 16-bit table's RESCALE
//!
//! A 16-bit table whose values need a left shift becomes two nodes in rc4,
//! `TABLE(x, table) -> RESCALE(., out_dtype, [2**lshift], 0, 0)`
//! (insert_table_ops.py:400-417), and TABLE returns int32 for an int16 input
//! (tosa/dialect/ops/table.py:36-51). Layout passes can put a node between the
//! two: the Down propagation pass sinks a rank-changing view past the TABLE
//! and stops it at the RESCALE, which it may not cross. A pending table holds
//! both nodes in one, so it can't express that. [`split_table_rescales`] gives
//! such a pending table rc4's shape: the pending table keeps its args and
//! qparams but holds the TABLE's int32 value and is marked
//! [`SPLIT_RESCALE_KEY`], and a real per-tensor RESCALE node follows it,
//! marked [`TABLE_RESCALE_OF_KEY`]. [`merge_table_rescales`] folds a split
//! table back into one pending table wherever the RESCALE still reads only the
//! table and is the table's only user, at the RESCALE's place, which is how
//! the replay generator restores rc4's two nodes (`restore_pending_tables` in
//! tests/fixtures/layout_replay/gen_layout_replay.py). A split table that
//! survives a pass therefore marks a place where rc4 itself keeps the TABLE
//! and its RESCALE apart; whatever lowers it must emit the TABLE alone and
//! let the RESCALE node do the shift (the explicit-mode serializer's job once
//! it is wired; `fx_insert_table_ops_ir` plans the shifted table either way).

use std::collections::HashMap;

use crate::fx_insert_table_ops_ir::planned_table;
use crate::fx_ir::{IrArg, IrNode, NodeId, NodeMeta, NodeOp};
use crate::fx_meta::MetaValue;
use crate::layout::ctx::LayoutCtx;
use crate::layout::fx_node::all_input_nodes;


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

/// Meta key on a pending table whose RESCALE is its own node (see the module
/// docs): its value is the TABLE's, int32.
pub(crate) const SPLIT_RESCALE_KEY: &str = "retab_pending_table_split_rescale";

/// Meta key on the RESCALE [`split_table_rescales`] made: the name of its
/// pending table.
pub(crate) const TABLE_RESCALE_OF_KEY: &str = "retab_pending_table_rescale_of";

/// `exir_ops.backend.tosa.RESCALE.default` as RETAB's FX import spells it
/// (`normalize_fx_target`), which is how the RESCALE InsertTableOpsPass makes
/// reads in an imported graph.
const RESCALE_TARGET: &str = "tosa.RESCALE.default";

/// What TABLE returns for an int16 input (tosa/dialect/ops/table.py:36-51).
const TABLE_INT16_RESULT: &str = "torch.int32";

/// Whether `node` is a pending table whose RESCALE has been split off.
pub(crate) fn has_split_rescale(node: &IrNode) -> bool {
    node.meta.extra_bool(SPLIT_RESCALE_KEY) == Some(true)
}

/// The RESCALE rc4 puts after this pending table and RETAB still holds inside
/// it: `(left shift, output dtype)` for a 16-bit table with a non-zero shift
/// (insert_table_ops.py:400-417) that hasn't been split. A table RETAB can't
/// plan has none, as `fx_insert_table_ops_ir` emits no TABLE for it.
pub(crate) fn implicit_rescale(node: &IrNode) -> Option<(i64, &'static str)> {
    if !is_pending_table(node) || has_split_rescale(node) {
        return None;
    }
    let table = planned_table(node)?;
    (table.rescale_lshift != 0).then(|| (table.rescale_lshift, table.output.dtype.torch_str()))
}

/// Give every pending table with an implicit RESCALE rc4's two nodes, in graph
/// order, as InsertTableOpsPass does (insert_table_ops.py:400-421): the table
/// keeps its place, args and qparams and takes the TABLE's int32 value, and
/// `RESCALE(table, out_dtype, [2**lshift], 0, 0)` right after it holds the
/// table's old value and feeds its users, in their order. Returns the tables
/// split. The users of the graph's other nodes keep their order, so a ctx in
/// graph order, as after InsertTableOpsPass's retrace, stays so. O(N) to find
/// the tables, plus their users.
pub(crate) fn split_table_rescales(ctx: &mut LayoutCtx<'_>) -> Vec<NodeId> {
    let tables: Vec<(NodeId, i64, &'static str)> = ctx
        .order()
        .iter()
        .filter_map(|id| implicit_rescale(ctx.node(id)).map(|(lshift, dtype)| (id, lshift, dtype)))
        .collect();
    for &(table, lshift, out_dtype) in &tables {
        let node = ctx.node(table);
        let name = node.name.clone();
        let mut rescale_meta = NodeMeta::default();
        if let Some(facts) = node.meta.tensor_facts() {
            rescale_meta.set_tensor_facts(facts.clone());
        }
        if let Some(dim_order) = node.meta.val_dim_order() {
            rescale_meta.set_val_dim_order(dim_order.to_vec());
        }
        rescale_meta.set_extra(TABLE_RESCALE_OF_KEY, MetaValue::Str(name));
        let args = vec![
            IrArg::Node(table),
            IrArg::Dtype(out_dtype.to_string()),
            IrArg::List(vec![IrArg::Float(2f64.powi(lshift as i32))]),
            IrArg::Int(0),
            IrArg::Int(0),
        ];
        let users = ctx.users_in_fx_order(table);
        let rescale = ctx.inserting_after(table, |ctx| {
            ctx.call_function("tosa_rescale_default", RESCALE_TARGET, args, rescale_meta)
        });
        for user in users {
            ctx.replace_input_with(user, table, rescale);
        }
        let meta = ctx.meta_mut(table);
        meta.set_val_dtype_for_existing_shape(TABLE_INT16_RESULT);
        meta.set_extra(SPLIT_RESCALE_KEY, MetaValue::Bool(true));
    }
    tables.into_iter().map(|(table, _, _)| table).collect()
}

/// Fold each split table whose only user is its RESCALE, reading only the
/// table, back into one pending table that takes the RESCALE's place and
/// value; see the module docs. Returns how many tables stay split. O(N).
pub(crate) fn merge_table_rescales(ctx: &mut LayoutCtx<'_>) -> usize {
    let split: Vec<NodeId> = ctx
        .order()
        .iter()
        .filter(|&id| has_split_rescale(ctx.node(id)))
        .collect();
    let mut rescale_of: HashMap<String, NodeId> = HashMap::new();
    for id in ctx.order().iter() {
        if let Some(table) = ctx.node(id).meta.extra_str(TABLE_RESCALE_OF_KEY) {
            rescale_of.insert(table.to_string(), id);
        }
    }
    let mut kept = 0;
    for table in split {
        let Some(&rescale) = rescale_of.get(&ctx.node(table).name) else {
            kept += 1;
            continue;
        };
        if ctx.is_erased(rescale)
            || ctx.graph().live_user_ids(table) != [rescale]
            || all_input_nodes(ctx.node(rescale)) != [table]
        {
            kept += 1;
            continue;
        }
        ctx.move_before(table, rescale);
        let value = ctx.node(rescale).meta.clone();
        let meta = ctx.meta_mut(table);
        if let Some(facts) = value.tensor_facts() {
            meta.set_tensor_facts(facts.clone());
        }
        if let Some(dim_order) = value.val_dim_order() {
            meta.set_val_dim_order(dim_order.to_vec());
        }
        meta.extras.remove(SPLIT_RESCALE_KEY);
        ctx.replace_all_uses_with(rescale, table);
        ctx.erase_node(rescale);
    }
    kept
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
