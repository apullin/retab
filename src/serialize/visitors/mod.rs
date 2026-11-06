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
mod rounding;
pub(crate) mod shape_utils;
mod tensor_ops;

pub use emitter::*;

use super::tensor_layout::permute_shape;
use super::tosa_types::{
    AbsAttribute, AddAttribute, ArgMaxAttribute, ArithmeticRightShiftAttribute, Attribute,
    AvgPool2dAttribute, BitwiseAndAttribute, BitwiseNotAttribute, BitwiseOrAttribute,
    BitwiseXorAttribute, CastAttribute, CeilAttribute, ClampAttribute, ConcatAttribute,
    ConstAttribute, ConstShapeAttribute, Conv2dAttribute, Conv3dAttribute, CosAttribute, DType,
    DepthwiseConv2dAttribute, EqualAttribute, ErfAttribute, ExpAttribute, FloorAttribute,
    GatherAttribute, GreaterAttribute, GreaterEqualAttribute, IntDivAttribute, LogAttribute, LogicalAndAttribute,
    LogicalLeftShiftAttribute, LogicalNotAttribute, LogicalOrAttribute, LogicalXorAttribute,
    MatMulAttribute, MaxPool2dAttribute, MaximumAttribute, MinimumAttribute, MulAttribute,
    NanPropagationMode, NegateAttribute, Op, PadAttribute, PowAttribute, ReciprocalAttribute,
    ReduceAnyAttribute, ReduceMaxAttribute, ReduceMinAttribute, ReduceSumAttribute,
    RescaleAttribute, ReshapeAttribute, ResizeAttribute, ResizeMode, ReverseAttribute,
    RoundingMode, RsqrtAttribute, ScatterAttribute, SelectAttribute, SigmoidAttribute,
    SinAttribute, SliceAttribute, SubAttribute, TableAttribute, TanhAttribute, TosaOperator,
    TosaShape, TosaTensor, TransposeAttribute, TransposeConv2dAttribute,
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
    has_only_tosa_rescale_users, input_qparam_zero_point, input_qparam_zero_point_or_kwarg,
    ir_float_dtype_or_float32, is_float_dtype, is_narrow_float_dtype, matmul_input_dtype,
    node_input_zero_point, node_output_dtype, node_output_zero_point, node_tensor_dtype,
    normalize_rescale_zp, output_qparam_zero_point, passthrough_input_dtype, single_qparam,
    tosa_dtype_storage_size, tosa_float_dtype_to_ir_or_float32,
};
pub(crate) use emit_result::EmitResult;
use graph_utils::*;
use layout::*;
use op_utils::*;
use rounding::*;
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
    if let Some(probe) = crate::config::debug_node_probe() {
        if node.name.contains(&probe) {
            eprintln!(
                "[NodeProbe] serialize {} target={:?} raw_shape={:?}",
                node.name,
                node.target,
                &node.meta.val.shape[..node.meta.val.shape_len as usize],
            );
        }
    }
    // Handle the `getitem` special case first (dynamic Other variant).
    if is_getitem_node(node) {
        return Some(EmitResult::default());
    }

    EMITTER_TABLE.with(|table| {
        if let Some(emitter) = table.get(&node.target) {
            let ctx = EmitContext { graph, node, store };
            Some(emitter(&ctx))
        } else {
            if crate::config::retab_debug_enabled() {
                eprintln!(
                    "[emit_node] SKIPPED unhandled target: {:?} (node '{}')",
                    node.target, node.name
                );
            }
            None
        }
    })
}
