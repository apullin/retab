//! Emitter context and dispatch types.
//!
//! Each TOSA emitter receives an `EmitContext` containing the graph, node, and
//! tensor store. The `EmitFn` type alias standardizes the emitter signature so
//! dispatch can use a `HashMap<OpTarget, EmitFn>` instead of a 117-arm match.

use std::collections::HashMap;

use crate::graph::ir::Node;
use crate::graph::ir::OpTarget;
use crate::graph::Graph;
use crate::store::TensorStore;

use super::emit_result::EmitResult;
use super::graph_utils::collect_input_names;
use super::op_utils::{emit_bool_binary, emit_direct_binary, emit_direct_unary, emit_identity};
use super::shape_utils::make_actual_output_tensor;
use crate::serialize::tosa_generated::{Attribute, Op, TosaTensor};

/// Everything an emitter needs to produce TOSA operators + tensors for one node.
pub struct EmitContext<'a> {
    pub graph: &'a Graph,
    pub node: &'a Node,
    pub store: &'a TensorStore,
}

impl<'a> EmitContext<'a> {
    /// Collect input tensor names from node args.
    pub fn input_names(&self) -> Vec<String> {
        collect_input_names(self.graph, self.node)
    }

    /// Create the output tensor with actual shape/dtype.
    pub fn output_tensor(&self) -> TosaTensor {
        make_actual_output_tensor(self.graph, self.store, self.node)
    }

    /// Emit a single unary op (1 input -> 1 output, same shape/dtype).
    pub fn emit_unary(&self, op: Op, attr: Attribute) -> EmitResult {
        emit_direct_unary(self.graph, self.store, self.node, op, attr)
    }

    /// Emit a single binary op (2 inputs -> 1 output, with broadcast).
    pub fn emit_binary(&self, op: Op, attr: Attribute) -> EmitResult {
        emit_direct_binary(self.graph, self.node, self.store, op, attr)
    }

    /// Emit a bool-output binary op.
    pub fn emit_bool_binary(&self, op: Op, attr: Attribute, swap_inputs: bool) -> EmitResult {
        emit_bool_binary(self.graph, self.node, self.store, op, attr, swap_inputs)
    }

    /// Emit an identity (passthrough) op.
    pub fn emit_identity(&self) -> EmitResult {
        emit_identity(self.graph, self.store, self.node)
    }
}

/// Emitter function signature: takes context, returns operators + tensors.
pub type EmitFn = fn(&EmitContext) -> EmitResult;

/// Shared dispatch table type for module-level registration.
pub type EmitterTable = HashMap<OpTarget, EmitFn>;
