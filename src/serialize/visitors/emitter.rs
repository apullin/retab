//! Emitter context and dispatch types.
//!
//! Each TOSA emitter receives an `EmitContext` containing the graph, node, and
//! tensor store. The `EmitFn` type alias standardizes the emitter signature so
//! dispatch can use a `HashMap<OpTarget, EmitFn>` instead of a 117-arm match.

use crate::graph::ir::{Node, NodeId};
use crate::graph::Graph;
use crate::store::TensorStore;

use super::emit_result::EmitResult;

/// Everything an emitter needs to produce TOSA operators + tensors for one node.
pub struct EmitContext<'a> {
    pub graph: &'a Graph,
    pub node_id: NodeId,
    pub node: &'a Node,
    pub store: &'a TensorStore,
}

/// Emitter function signature: takes context, returns operators + tensors.
pub type EmitFn = fn(&EmitContext) -> EmitResult;
