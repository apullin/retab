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

use super::arg_utils::arg_i32_list;
use super::dtype_quant::{actual_tensor_dtype, node_output_dtype, passthrough_input_dtype};
use super::emit_result::EmitResult;
use super::graph_utils::{collect_input_names, input_name, primary_output_name};
use super::op_utils::{emit_bool_binary, emit_direct_binary, emit_direct_unary, emit_identity};
use super::shape_utils::{actual_tensor_shape, make_actual_output_tensor};
use crate::serialize::tosa_generated::{Attribute, DType, Op, TosaTensor};

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

    /// Resolve one input tensor name from node args.
    pub fn input_name(&self, index: usize) -> Option<String> {
        input_name(self.graph, self.node, index)
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

    // ---- Shape / dtype accessors ----

    /// Concrete TOSA tensor shape for this node.
    pub fn shape(&self) -> Vec<i32> {
        actual_tensor_shape(self.graph, self.store, self.node)
    }

    /// Semantic output dtype for this node.
    pub fn dtype(&self) -> DType {
        node_output_dtype(self.node)
    }

    /// Dtype inherited from the first input (passthrough).
    pub fn passthrough_dtype(&self) -> DType {
        passthrough_input_dtype(self.graph, self.node)
    }

    /// Structural dtype that accounts for accumulator / quantization rules.
    pub fn actual_dtype(&self) -> DType {
        actual_tensor_dtype(self.graph, self.node)
    }

    // ---- Node accessors ----

    /// Node name.
    pub fn name(&self) -> &str {
        &self.node.name
    }

    /// Primary output name (follows getitem convention).
    pub fn primary_output_name(&self) -> String {
        primary_output_name(self.graph, self.node)
    }

    // ---- Common arg helpers ----

    pub fn arg_i32_list(&self, idx: usize) -> Vec<i32> {
        arg_i32_list(self.node, idx)
    }

    // ---- Tensor creation shortcuts ----

    /// Create the output tensor using actual shape and dtype.
    pub fn make_output(&self) -> TosaTensor {
        make_actual_output_tensor(self.graph, self.store, self.node)
    }
}

/// Emitter function signature: takes context, returns operators + tensors.
pub type EmitFn = fn(&EmitContext) -> EmitResult;

/// Shared dispatch table type for module-level registration.
pub type EmitterTable = HashMap<OpTarget, EmitFn>;
