//! Pure Rust IrPass decomposition of `aten.tril` into
//! arange + view + add(diagonal) + ge + where / bitwise_and.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::fx_target::is_tril_target;
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

/// Decompose a single tril node into primitive ops on the IrGraph.
/// Returns the final replacement NodeId, or None if the node cannot be decomposed.
fn decompose_one(graph: &mut IrGraph, id: NodeId) -> Option<NodeId> {
    let input_id = graph.nodes[id].first_node_arg()?;

    let diagonal = graph.nodes[id].arg_int_or(1, 0);

    let input_shape = graph.node_shape_vec(input_id)?;
    if input_shape.len() < 2 {
        return None;
    }
    let rows = input_shape[input_shape.len() - 2];
    let cols = input_shape[input_shape.len() - 1];
    let input_dtype = graph.node_dtype(input_id)?.to_string();
    let base_name = graph.nodes[id].name.clone();
    let base_meta = graph.nodes[id].meta.clone();

    // --- row indices: arange(0, rows) -> [rows], then view to [rows, 1] ---
    let row_arange = graph.add_anchored_call_function_with_val(
        id,
        format!("{base_name}_row_arange"),
        "aten.arange.start_step",
        vec![IrArg::Int(0), IrArg::Int(rows), IrArg::Int(1)],
        &base_meta,
        vec![rows],
        "torch.float32",
    );

    let row_view = graph.add_anchored_view_copy(
        id,
        format!("{base_name}_row_view"),
        row_arange,
        &[rows, 1],
        base_meta.cloned_with_val(vec![rows, 1], "torch.float32"),
    );

    // If diagonal != 0, add the offset
    let row_final = if diagonal != 0 {
        graph.add_node(
            NodeOp::CallFunction,
            format!("{base_name}_row_add_diag"),
            "aten.add.Scalar".to_string(),
            vec![IrArg::Node(row_view), IrArg::Int(diagonal)],
            base_meta.cloned_with_val(vec![rows, 1], "torch.float32"),
        )
    } else {
        row_view
    };

    // --- col indices: arange(0, cols) -> [cols], then view to [1, cols] ---
    let col_arange = graph.add_anchored_call_function_with_val(
        id,
        format!("{base_name}_col_arange"),
        "aten.arange.start_step",
        vec![IrArg::Int(0), IrArg::Int(cols), IrArg::Int(1)],
        &base_meta,
        vec![cols],
        "torch.float32",
    );

    let col_view = graph.add_anchored_view_copy(
        id,
        format!("{base_name}_col_view"),
        col_arange,
        &[1, cols],
        base_meta.cloned_with_val(vec![1, cols], "torch.float32"),
    );

    // --- mask: row >= col -> bool [rows, cols] ---
    let mask = graph.add_anchored_call_function_with_val(
        id,
        format!("{base_name}_ge"),
        "aten.ge.Tensor",
        vec![IrArg::Node(row_final), IrArg::Node(col_view)],
        &base_meta,
        vec![rows, cols],
        "torch.bool",
    );

    // --- result: bool input -> bitwise_and, else -> where(mask, input, zero) ---
    let is_bool = graph.nodes[input_id].meta.val_dtype_is(DType::Bool);

    let result = if is_bool {
        graph.add_node(
            NodeOp::CallFunction,
            format!("{base_name}_bitwise_and"),
            "aten.bitwise_and.Tensor".to_string(),
            vec![IrArg::Node(input_id), IrArg::Node(mask)],
            base_meta,
        )
    } else {
        // Create a scalar zero with the input dtype.
        // For quantized inputs, clamp the zero-point into [qmin, qmax].
        let (zero_val, zero_dtype) = if let Some(qp) = graph.nodes[id].meta.output_qparam() {
            let clamped = qp.zp.max(qp.qmin).min(qp.qmax);
            (clamped, qp.dtype_str.clone())
        } else {
            (0i64, input_dtype.clone())
        };

        let zero = graph.add_anchored_call_function(
            id,
            format!("{base_name}_zero"),
            "aten.scalar_tensor.default",
            vec![IrArg::Int(zero_val)],
            NodeMeta::tensor(Vec::<i64>::new(), &zero_dtype),
        );

        graph.add_node(
            NodeOp::CallFunction,
            format!("{base_name}_where"),
            "aten.where.self".to_string(),
            vec![IrArg::Node(mask), IrArg::Node(input_id), IrArg::Node(zero)],
            base_meta,
        )
    };

    Some(result)
}

pub struct IrDecomposeTrilPass;

impl IrPass for IrDecomposeTrilPass {
    fn name(&self) -> &str {
        "IrDecomposeTril"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            if !is_tril_target(&graph.nodes[id].target) {
                return false;
            }

            if let Some(new_id) = decompose_one(graph, id) {
                graph.replace_and_erase(id, new_id);
                return true;
            }
            false
        });

        if modified {
            log_line(format_args!("[IrDecomposeTril] decomposed tril ops"));
        }
        modified
    }
}
