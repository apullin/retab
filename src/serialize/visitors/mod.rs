// Per-op TOSA emission visitors.
//
// Each function takes the graph node, its metadata, and the tensor store, and
// returns the TOSA operator(s) + tensor(s) that should be added to the basic
// block being built.

mod arg_utils;
mod arithmetic;
mod broadcast;
mod const_utils;
mod control_flow;
pub(crate) mod dtype_quant;
mod elementwise;
mod emit_result;
mod emitter;
mod graph_utils;
mod indexing;
mod layout;
mod math;
mod matmul;
mod nn;
mod op_utils;
mod reductions;
pub(crate) mod shape_utils;
mod tensor_ops;

pub use emitter::*;

use super::tosa_generated::{
    AbsAttribute, AddAttribute, ArithmeticRightShiftAttribute, Attribute, AvgPool2dAttribute,
    BitwiseAndAttribute, BitwiseNotAttribute, BitwiseOrAttribute, BitwiseXorAttribute,
    CastAttribute, CeilAttribute, ClampAttribute, ConcatAttribute, ConstAttribute,
    ConstShapeAttribute, Conv2dAttribute, Conv3dAttribute, CosAttribute, DType,
    DepthwiseConv2dAttribute, EqualAttribute, ErfAttribute, ExpAttribute, FloorAttribute,
    GatherAttribute, GreaterAttribute, GreaterEqualAttribute, LogAttribute, LogicalAndAttribute,
    LogicalLeftShiftAttribute, LogicalNotAttribute, LogicalOrAttribute, LogicalXorAttribute,
    MatMulAttribute, MaxPool2dAttribute, MaximumAttribute, MinimumAttribute, MulAttribute,
    NanPropagationMode, NegateAttribute, Op, PadAttribute, PowAttribute, ReciprocalAttribute,
    ReduceAnyAttribute, ReduceMaxAttribute, ReduceMinAttribute, ReduceSumAttribute,
    RescaleAttribute, ReshapeAttribute, ResizeAttribute, ResizeMode, RoundingMode, RsqrtAttribute,
    ScatterAttribute, SelectAttribute, SigmoidAttribute, SinAttribute, SliceAttribute,
    SubAttribute, TableAttribute, TanhAttribute, TosaOperator, TosaShape, TosaTensor,
    TransposeAttribute, TransposeConv2dAttribute,
};
use crate::graph::ir::{Arg, Node, NodeId, OpTarget, OpType};
use crate::graph::Graph;
use crate::store::TensorStore;
use arg_utils::*;
use broadcast::*;
use const_utils::*;
pub(crate) use dtype_quant::ir_dtype_to_tosa;
use dtype_quant::{
    accumulator_output_dtype, actual_tensor_dtype, clamp_bound_bytes_from_arg,
    compute_multiplier_and_shift_with_width, conv_accumulator_dtype, encode_zp_for_dtype,
    has_only_tosa_rescale_users, is_float_dtype, matmul_input_dtype, node_input_zero_point,
    node_output_dtype, node_output_zero_point, node_tensor_dtype, normalize_rescale_zp,
    passthrough_input_dtype, single_qparam,
};
use emit_result::EmitResult;
use graph_utils::*;
use layout::*;
use op_utils::*;
pub(crate) use shape_utils::actual_tensor_shape;
use shape_utils::*;

fn build_emitter_table() -> EmitterTable {
    let mut table = EmitterTable::new();
    for register in [
        arithmetic::register as fn(&mut EmitterTable),
        tensor_ops::register,
        indexing::register,
        matmul::register,
        control_flow::register,
        elementwise::register,
        math::register,
        nn::register,
        reductions::register,
    ] {
        register(&mut table);
    }
    table
}

// Cached emitter dispatch table, built once per thread.
thread_local! {
    static EMITTER_TABLE: EmitterTable = build_emitter_table();
}

/// Dispatch to the correct per-op visitor based on the node's target.
pub fn emit_node(
    graph: &Graph,
    _node_id: NodeId,
    node: &Node,
    store: &TensorStore,
) -> Option<EmitResult> {
    if let Ok(probe) = std::env::var("ARM_TOSA_ENGINE_DEBUG_NODE") {
        if node.name.contains(&probe) {
            eprintln!(
                "[NodeProbe] serialize {} target={:?} raw_shape={:?} dim_order={:?}",
                node.name,
                node.target,
                &node.meta.val.shape[..node.meta.val.shape_len as usize],
                node.meta.tosa_dim_order,
            );
        }
    }
    // Handle the `getitem` special case first (dynamic Other variant).
    if let OpTarget::Other(target) = &node.target {
        if target.contains("getitem") {
            return Some(EmitResult::default());
        }
    }

    EMITTER_TABLE.with(|table| {
        if let Some(emitter) = table.get(&node.target) {
            let ctx = EmitContext { graph, node, store };
            Some(emitter(&ctx))
        } else {
            if crate::fx_graph::FxGraphModule::debug() {
                eprintln!(
                    "[emit_node] SKIPPED unhandled target: {:?} (node '{}')",
                    node.target, node.name
                );
            }
            None
        }
    })
}
