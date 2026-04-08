//! Pure Rust IrPass: atanh, stack, unbind decompositions.
//! Mirrors FxDecomposeOpsPass (hardsigmoid is intentionally skipped).

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

fn meta_with(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta { shape, dtype: dtype.to_string() });
    m
}

fn is_atanh(t: &str) -> bool { t.contains("atanh") }
fn is_stack(t: &str) -> bool { t.contains("stack") && !t.contains("unstack") }
fn is_unbind(t: &str) -> bool { t.contains("unbind") }

/// Check if input is DQ and at least one user is Q (quantized envelope → skip atanh).
fn has_quantized_envelope(graph: &IrGraph, id: NodeId) -> bool {
    // Input must be dequantize
    let input_id = match graph.nodes[id].first_node_arg() { Some(n) => n, None => return false };
    if !graph.nodes[input_id].target.contains("dequantize_per_tensor") { return false; }
    // At least one user must be quantize
    graph.nodes.iter()
        .filter(|n| n.alive && n.op == NodeOp::CallFunction)
        .filter(|n| n.args.iter().any(|a| matches!(a, IrArg::Node(nid) if *nid == id)))
        .any(|n| n.target.contains("quantize_per_tensor") && !n.target.contains("dequantize"))
}

pub struct IrDecomposeOpsPass;

impl IrPass for IrDecomposeOpsPass {
    fn name(&self) -> &str { "IrDecomposeOps" }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            let target = graph.nodes[id].target.clone();

            if is_atanh(&target) {
                if has_quantized_envelope(graph, id) { continue; }
                if decompose_atanh(graph, id) { modified = true; }
            } else if is_stack(&target) {
                if decompose_stack(graph, id) { modified = true; }
            } else if is_unbind(&target) {
                if decompose_unbind(graph, id) { modified = true; }
            }
        }

        if modified {
            log_line(format_args!("[IrDecomposeOps] decomposed atanh/stack/unbind ops"));
        }
        modified
    }
}

/// atanh(x) = 0.5 * log((1+x) / (1-x))
fn decompose_atanh(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() { Some(n) => n, None => return false };
    let shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(), None => return false,
    };
    let dtype = graph.nodes[x_id].meta.val_dtype().unwrap_or("torch.float32").to_string();
    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();

    let one = graph.add_node(NodeOp::CallFunction, format!("{}_one", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(1.0)],
        meta_with(&meta, shape.clone(), &dtype));
    graph.anchor_to(one, id);

    let half = graph.add_node(NodeOp::CallFunction, format!("{}_half", name),
        "aten.full.default".to_string(),
        vec![i64_list(&shape), IrArg::Float(0.5)],
        meta_with(&meta, shape.clone(), &dtype));
    graph.anchor_to(half, id);

    let one_plus = graph.add_node(NodeOp::CallFunction, format!("{}_1px", name),
        "aten.add.Tensor".to_string(),
        vec![IrArg::Node(one), IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype));
    graph.anchor_to(one_plus, id);

    let one_minus = graph.add_node(NodeOp::CallFunction, format!("{}_1mx", name),
        "aten.sub.Tensor".to_string(),
        vec![IrArg::Node(one), IrArg::Node(x_id)],
        meta_with(&meta, shape.clone(), &dtype));
    graph.anchor_to(one_minus, id);

    let ratio = graph.add_node(NodeOp::CallFunction, format!("{}_ratio", name),
        "aten.div.Tensor".to_string(),
        vec![IrArg::Node(one_plus), IrArg::Node(one_minus)],
        meta_with(&meta, shape.clone(), &dtype));
    graph.anchor_to(ratio, id);

    let log_r = graph.add_node(NodeOp::CallFunction, format!("{}_log", name),
        "aten.log.default".to_string(),
        vec![IrArg::Node(ratio)],
        meta_with(&meta, shape.clone(), &dtype));
    graph.anchor_to(log_r, id);

    let result = graph.add_node(NodeOp::CallFunction, format!("{}_result", name),
        "aten.mul.Tensor".to_string(),
        vec![IrArg::Node(half), IrArg::Node(log_r)],
        meta);
    graph.replace_and_erase(id, result);
    true
}

/// stack(tensors, dim) → unsqueeze each + cat
fn decompose_stack(graph: &mut IrGraph, id: NodeId) -> bool {
    let args = &graph.nodes[id].args;
    if args.is_empty() { return false; }

    // args[0] = List of node refs, args[1] = dim (default 0)
    let tensor_ids: Vec<NodeId> = match &args[0] {
        IrArg::List(items) => items.iter().filter_map(|a| a.as_node()).collect(),
        _ => return false,
    };
    if tensor_ids.is_empty() { return false; }
    let dim = args.get(1).and_then(|a| a.as_int()).unwrap_or(0);

    let name = graph.nodes[id].name.clone();
    let meta = graph.nodes[id].meta.clone();
    let dtype = graph.nodes[tensor_ids[0]].meta.val_dtype()
        .unwrap_or("torch.float32").to_string();

    // Unsqueeze each tensor at dim
    let mut unsqueezed: Vec<NodeId> = Vec::new();
    for (i, &tid) in tensor_ids.iter().enumerate() {
        let t_shape = graph.nodes[tid].meta.val_shape()
            .map(|s| s.to_vec()).unwrap_or_default();
        let mut u_shape = t_shape;
        let pos = if dim < 0 { (dim + u_shape.len() as i64 + 1) as usize } else { dim as usize };
        u_shape.insert(pos, 1);

        let unsq = graph.add_node(NodeOp::CallFunction,
            format!("{}_unsq_{}", name, i),
            "aten.unsqueeze_copy.default".to_string(),
            vec![IrArg::Node(tid), IrArg::Int(dim)],
            meta_with(&meta, u_shape, &dtype));
        graph.anchor_to(unsq, id);
        unsqueezed.push(unsq);
    }

    // Cat along dim
    let cat_args = IrArg::List(unsqueezed.iter().map(|&n| IrArg::Node(n)).collect());
    let result = graph.add_node(NodeOp::CallFunction,
        format!("{}_cat", name),
        "aten.cat.default".to_string(),
        vec![cat_args, IrArg::Int(dim)],
        meta);
    graph.replace_and_erase(id, result);
    true
}

/// unbind(x, dim) → for each getitem user: slice + squeeze
fn decompose_unbind(graph: &mut IrGraph, id: NodeId) -> bool {
    let x_id = match graph.nodes[id].first_node_arg() { Some(n) => n, None => return false };
    let dim = graph.nodes[id].args.get(1).and_then(|a| a.as_int()).unwrap_or(0);

    let x_shape = match graph.nodes[x_id].meta.val_shape() {
        Some(s) => s.to_vec(), None => return false,
    };
    let rank = x_shape.len() as i64;
    let dim_norm = dim.rem_euclid(rank) as usize;
    let dtype = graph.nodes[x_id].meta.val_dtype().unwrap_or("torch.float32").to_string();
    let name = graph.nodes[id].name.clone();
    let orig_meta = graph.nodes[id].meta.clone();

    // Find getitem users
    let getitem_users: Vec<(NodeId, i64)> = graph.nodes.iter()
        .filter(|n| n.alive && n.target.contains("getitem"))
        .filter(|n| n.args.first().and_then(|a| a.as_node()) == Some(id))
        .map(|n| {
            let idx = n.args.get(1).and_then(|a| a.as_int()).unwrap_or(0);
            (n.id, idx)
        })
        .collect();

    if getitem_users.is_empty() { return false; }

    for (getitem_id, idx) in &getitem_users {
        // slice(x, dim, idx, idx+1)
        let mut slice_shape = x_shape.clone();
        slice_shape[dim_norm] = 1;
        let slice_id = graph.add_node(NodeOp::CallFunction,
            format!("{}_slice_{}", name, idx),
            "aten.slice_copy.Tensor".to_string(),
            vec![IrArg::Node(x_id), IrArg::Int(dim_norm as i64),
                 IrArg::Int(*idx), IrArg::Int(idx + 1)],
            meta_with(&orig_meta, slice_shape, &dtype));
        graph.anchor_to(slice_id, id);

        // squeeze(slice, dim)
        let mut sq_shape = x_shape.clone();
        sq_shape.remove(dim_norm);
        let squeeze_id = graph.add_node(NodeOp::CallFunction,
            format!("{}_sq_{}", name, idx),
            "aten.squeeze_copy.dims".to_string(),
            vec![IrArg::Node(slice_id), i64_list(&[dim_norm as i64])],
            meta_with(&orig_meta, sq_shape, &dtype));

        graph.replace_and_erase(*getitem_id, squeeze_id);
    }

    // Erase the unbind node itself (now has no users)
    graph.erase(id);
    true
}
