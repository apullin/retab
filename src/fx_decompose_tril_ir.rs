//! Pure Rust IrPass decomposition of `aten.tril` into
//! arange + view + add(diagonal) + ge + where / bitwise_and.

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn is_tril(target: &str) -> bool {
    target.contains("aten.tril")
        || target.contains("aten_tril")
        || target.contains("aten.tril_copy")
        || target.contains("aten_tril_copy")
}

/// Build a NodeMeta with the given shape and dtype, cloning qparams from `base`.
fn meta_with_val(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta {
        shape,
        dtype: dtype.to_string(),
    });
    m
}

/// Decompose a single tril node into primitive ops on the IrGraph.
/// Returns the final replacement NodeId, or None if the node cannot be decomposed.
fn decompose_one(graph: &mut IrGraph, id: NodeId) -> Option<NodeId> {
    let input_id = graph.nodes[id].first_node_arg()?;

    let diagonal: i64 = match graph.nodes[id].args.get(1) {
        Some(IrArg::Int(d)) => *d,
        _ => 0,
    };

    let input_shape = graph.nodes[input_id].meta.val_shape()?.to_vec();
    if input_shape.len() < 2 {
        return None;
    }
    let rows = input_shape[input_shape.len() - 2];
    let cols = input_shape[input_shape.len() - 1];
    let input_dtype = graph.nodes[input_id].meta.val_dtype()?.to_string();
    let base_name = graph.nodes[id].name.clone();
    let base_meta = graph.nodes[id].meta.clone();

    // --- row indices: arange(0, rows) -> [rows], then view to [rows, 1] ---
    let row_arange = graph.add_node(
        NodeOp::CallFunction,
        format!("{base_name}_row_arange"),
        "aten.arange.start_step".to_string(),
        vec![IrArg::Int(0), IrArg::Int(rows), IrArg::Int(1)],
        meta_with_val(&base_meta, vec![rows], "torch.float32"),
    );
    graph.anchor_to(row_arange, id);

    let row_view = graph.add_node(
        NodeOp::CallFunction,
        format!("{base_name}_row_view"),
        "aten.view_copy.default".to_string(),
        vec![
            IrArg::Node(row_arange),
            IrArg::List(vec![IrArg::Int(rows), IrArg::Int(1)]),
        ],
        meta_with_val(&base_meta, vec![rows, 1], "torch.float32"),
    );
    graph.anchor_to(row_view, id);

    // If diagonal != 0, add the offset
    let row_final = if diagonal != 0 {
        graph.add_node(
            NodeOp::CallFunction,
            format!("{base_name}_row_add_diag"),
            "aten.add.Scalar".to_string(),
            vec![IrArg::Node(row_view), IrArg::Int(diagonal)],
            meta_with_val(&base_meta, vec![rows, 1], "torch.float32"),
        )
    } else {
        row_view
    };

    // --- col indices: arange(0, cols) -> [cols], then view to [1, cols] ---
    let col_arange = graph.add_node(
        NodeOp::CallFunction,
        format!("{base_name}_col_arange"),
        "aten.arange.start_step".to_string(),
        vec![IrArg::Int(0), IrArg::Int(cols), IrArg::Int(1)],
        meta_with_val(&base_meta, vec![cols], "torch.float32"),
    );
    graph.anchor_to(col_arange, id);

    let col_view = graph.add_node(
        NodeOp::CallFunction,
        format!("{base_name}_col_view"),
        "aten.view_copy.default".to_string(),
        vec![
            IrArg::Node(col_arange),
            IrArg::List(vec![IrArg::Int(1), IrArg::Int(cols)]),
        ],
        meta_with_val(&base_meta, vec![1, cols], "torch.float32"),
    );
    graph.anchor_to(col_view, id);

    // --- mask: row >= col -> bool [rows, cols] ---
    let mask = graph.add_node(
        NodeOp::CallFunction,
        format!("{base_name}_ge"),
        "aten.ge.Tensor".to_string(),
        vec![IrArg::Node(row_final), IrArg::Node(col_view)],
        meta_with_val(&base_meta, vec![rows, cols], "torch.bool"),
    );
    graph.anchor_to(mask, id);

    // --- result: bool input -> bitwise_and, else -> where(mask, input, zero) ---
    let is_bool = input_dtype.contains("bool");

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

        let zero = graph.add_node(
            NodeOp::CallFunction,
            format!("{base_name}_zero"),
            "aten.scalar_tensor.default".to_string(),
            vec![IrArg::Int(zero_val)],
            meta_with_val(&NodeMeta::default(), vec![], &zero_dtype),
        );
        graph.anchor_to(zero, id);

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
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !is_tril(&graph.nodes[id].target) {
                continue;
            }

            if let Some(new_id) = decompose_one(graph, id) {
                graph.replace_and_erase(id, new_id);
                modified = true;
            }
        }

        if modified {
            log_line(format_args!("[IrDecomposeTril] decomposed tril ops"));
        }
        modified
    }
}
