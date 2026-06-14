//! Control-flow IR metadata stamping and submodule boundary passes.

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrGraph, IrNode, IrPass, NodeId, RescaleRewrite};
use crate::fx_quant::QParam;
use crate::graph::ir::DType;

/// IrPass wrapper for the while-loop boundary Q/DQ strip (submodule IR pipeline).
pub struct IrStripWhileBoundaryQPass;
impl IrPass for IrStripWhileBoundaryQPass {
    fn name(&self) -> &str {
        "IrStripWhileBoundaryQ"
    }
    fn run(&self, graph: &mut IrGraph) -> bool {
        strip_while_boundary_qdq_ir(graph)
    }
}

/// IrPass: align each while_loop body output to its corresponding placeholder's
/// quantization domain by inserting a RESCALE (output[i] feeds back as input[i],
/// so their scale/zp must match). The IR analogue of FxInsertRescalePass::carried.
pub struct IrInsertCarriedOutputRescalePass;
impl IrPass for IrInsertCarriedOutputRescalePass {
    fn name(&self) -> &str {
        "IrInsertCarriedOutputRescale"
    }
    fn run(&self, graph: &mut IrGraph) -> bool {
        let ph_qps: Vec<Option<QParam>> = graph
            .placeholders()
            .map(|ph| ph.meta.output_qparam().or(ph.meta.input_qparam()).cloned())
            .collect();
        if ph_qps.iter().all(Option::is_none) {
            return false;
        }
        let Some(out_id) = graph.output_node().map(|n| n.id) else {
            return false;
        };
        let out_args = graph.node(out_id).args.clone();
        let val_ids: Vec<Option<NodeId>> = match out_args.first() {
            Some(IrArg::List(items)) => items
                .iter()
                .map(|a| match a {
                    IrArg::Node(id) => Some(*id),
                    _ => None,
                })
                .collect(),
            _ => out_args
                .iter()
                .map(|a| match a {
                    IrArg::Node(id) => Some(*id),
                    _ => None,
                })
                .collect(),
        };

        let mut modified = false;
        for (idx, ph_qp) in ph_qps.iter().enumerate() {
            let Some(ph_qp) = ph_qp else { continue };
            let Some(Some(src_id)) = val_ids.get(idx).copied() else {
                continue;
            };
            let Some(cur) = extract_effective_qparam_ir(graph, src_id) else {
                continue;
            };
            if cur.same_scale_zp(ph_qp) || cur.scale == 0.0 || ph_qp.scale == 0.0 {
                continue;
            }

            let meta = graph.node(src_id).meta.rescale_output_meta(ph_qp.clone());
            let src_name = graph.node(src_id).name.clone();
            let rescale = graph.add_after_rescale(
                src_id,
                RescaleRewrite::new(
                    format!("{src_name}_carried_rescale"),
                    src_id,
                    ph_qp.dtype_str.clone(),
                    vec![cur.scale / ph_qp.scale],
                    cur.zp,
                    ph_qp.zp,
                    meta,
                ),
            );
            replace_output_arg_use(graph, out_id, src_id, rescale);
            modified = true;
        }
        modified
    }
}

/// Replace a use of `old` with `new` in the output node's args (handles the
/// `output((v0, v1, ...),)` tuple-wrapped form).
fn replace_output_arg_use(graph: &mut IrGraph, out_id: NodeId, old: NodeId, new: NodeId) {
    let node = graph.node_mut(out_id);
    for arg in node.args.iter_mut() {
        match arg {
            IrArg::Node(id) if *id == old => *id = new,
            IrArg::List(items) => {
                for it in items.iter_mut() {
                    if let IrArg::Node(id) = it {
                        if *id == old {
                            *id = new;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Pure Rust: walk producer chain to find effective qparam.
pub(crate) fn extract_effective_qparam_ir(graph: &IrGraph, start: NodeId) -> Option<QParam> {
    let mut current = start;
    let mut seen = std::collections::HashSet::new();
    loop {
        if !seen.insert(current) {
            break;
        }
        let node = graph.node(current);
        if let Some(qp) = node.meta.output_qparam() {
            return Some(qp.clone());
        }
        if let Some(qp) = node.meta.input_qparam() {
            return Some(qp.clone());
        }
        // RESCALE reverse-engineering
        if let Some(qp) = reverse_engineer_rescale_ir(graph, node) {
            return Some(qp);
        }
        current = match node.first_node_arg() {
            Some(id) => id,
            None => break,
        };
    }
    None
}

/// Pure Rust: derive qparam from RESCALE args.
pub(crate) fn reverse_engineer_rescale_ir(graph: &IrGraph, node: &IrNode) -> Option<QParam> {
    if !node.is_rescale() || node.args.len() < 5 {
        return None;
    }
    let source_id = match node.args[0] {
        IrArg::Node(id) => id,
        _ => return None,
    };
    let source_qp = extract_effective_qparam_ir(graph, source_id)?;
    // args[2] = scale list, args[4] = output_zp
    let mult_val = match &node.args[2] {
        IrArg::List(items) if items.len() == 1 => match &items[0] {
            IrArg::Float(v) => *v,
            IrArg::Int(v) => *v as f64,
            _ => return None,
        },
        IrArg::Float(v) => *v,
        _ => return None,
    };
    if mult_val == 0.0 {
        return None;
    }
    let output_zp = match &node.args[4] {
        IrArg::Int(v) => *v,
        _ => return None,
    };
    // args[1] = dtype string (stored in target or as a string arg)
    let dtype_str = match &node.args[1] {
        IrArg::Str(s) => s.clone(),
        _ => {
            // dtype might be stored as a non-string Python object — fall back to target
            let s = format!("{:?}", node.args[1]);
            if s.contains("int") {
                s
            } else {
                return None;
            }
        }
    };
    let (qmin, qmax) = DType::quantized_integer_bounds_from_torch_str(&dtype_str)?;
    Some(QParam::per_tensor(
        source_qp.scale / mult_val,
        output_zp,
        qmin,
        qmax,
        dtype_str,
    ))
}

/// Pure Rust: stamp output node's input_qparams from placeholder qparams (loop-carried).
pub(crate) fn stamp_while_body_output_qparams_ir(graph: &mut IrGraph) -> usize {
    let ph_qparams: Vec<(usize, QParam)> = graph
        .placeholders()
        .enumerate()
        .filter_map(|(idx, ph)| {
            ph.meta
                .output_qparam()
                .or(ph.meta.input_qparam())
                .cloned()
                .map(|qp| (idx, qp))
        })
        .collect();

    if ph_qparams.is_empty() {
        return 0;
    }
    let count = ph_qparams.len();

    let out_id = match graph.output_node() {
        Some(n) => n.id,
        None => return 0,
    };
    graph
        .node_mut(out_id)
        .meta
        .set_input_qparams_from(ph_qparams);
    count
}

/// Pure Rust: strip boundary Q from placeholders, annotate DQ qparams on output.
pub(crate) fn strip_while_boundary_qdq_ir(graph: &mut IrGraph) -> bool {
    let mut modified = false;

    // Strip Q nodes from placeholders
    let ph_ids: Vec<NodeId> = graph.placeholders().map(|n| n.id).collect();
    for ph_id in ph_ids {
        let user_ids = graph.users(ph_id).to_vec();
        for q_id in user_ids {
            if !graph.nodes[q_id].alive || !graph.nodes[q_id].is_quantize() {
                continue;
            }
            // Stamp Q's qparams on placeholder
            if let Some(qp) = graph.nodes[q_id].parse_qdq_qparam() {
                graph.nodes[ph_id].meta.set_single_output_qparam(qp);
            }
            graph.replace_and_erase(q_id, ph_id);
            modified = true;
        }
    }

    // Annotate DQ qparams on output node (without stripping DQ)
    if let Some(out_id) = graph.output_node().map(|n| n.id) {
        // Get output arg node IDs
        let out_arg_ids: Vec<NodeId> = graph
            .node(out_id)
            .args
            .iter()
            .flat_map(|a| match a {
                IrArg::Node(id) => vec![*id],
                IrArg::List(items) => items
                    .iter()
                    .filter_map(|i| match i {
                        IrArg::Node(id) => Some(*id),
                        _ => None,
                    })
                    .collect(),
                _ => vec![],
            })
            .collect();

        let mut iqp_map: HashMap<usize, QParam> = graph
            .node(out_id)
            .meta
            .cloned_input_qparams()
            .unwrap_or_default();
        for (i, &arg_id) in out_arg_ids.iter().enumerate() {
            if !graph.nodes[arg_id].alive || !graph.nodes[arg_id].is_dequantize() {
                continue;
            }
            if let Some(qp) = graph.nodes[arg_id].parse_qdq_qparam() {
                iqp_map.insert(i, qp);
            }
        }
        if !iqp_map.is_empty() {
            graph.node_mut(out_id).meta.set_input_qparams(iqp_map);
            modified = true;
        }
    }

    modified
}
