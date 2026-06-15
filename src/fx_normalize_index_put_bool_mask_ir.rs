//! Pure Rust IrPass version of NormalizeIndexPutBoolIndexTensorPass
//! (executorch master, backends/arm/_passes/normalize_index_put_bool_index_tensor_pass.py).
//!
//! The simple boolean-mask scatter `x[mask] = scalar`, i.e.
//!   index_put(destination, [mask], data, accumulate=False)
//! with a single bool mask and scalar data, lowers directly to a `where`:
//!   mask = view_copy(mask, mask_shape right-padded with 1s to destination rank)
//!   data = view_copy(data, [1]*destination_rank)
//!   out  = where(mask, data, destination)
//! Runs before IrRewriteIndexPut so the general SCATTER path only sees int indices.

#[cfg(test)]
use crate::fx_ir::NodeMeta;
use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeOp, TensorMetaStamp};
use crate::fx_target::is_index_put_family_target;
use crate::runner_logging::log_line;

fn view_copy(
    graph: &mut IrGraph,
    name: String,
    meta: &TensorMetaStamp<'_>,
    input_id: NodeId,
    shape: &[i64],
) -> NodeId {
    graph.add_view_copy(name, input_id, shape, meta.preserve_qparams(shape.to_vec()))
}

fn rewrite_one(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if !is_index_put_family_target(&node.target) || node.args.len() < 3 {
        return false;
    }
    let node_name = node.name.clone();
    let orig_meta = node.meta.clone();
    let dest = match node.args[0].as_node() {
        Some(n) => n,
        None => return false,
    };
    // indices is a list; the bool-mask case has exactly one Node entry.
    let indices = match &node.args[1] {
        IrArg::List(items) => items.clone(),
        _ => return false,
    };
    // Exactly one bool-mask Node; every other index entry must be None.
    let node_positions: Vec<(usize, NodeId)> = indices
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match a {
            IrArg::Node(n) => Some((i, *n)),
            _ => None,
        })
        .collect();
    if node_positions.len() != 1 {
        return false;
    }
    let (mask_pos, mask) = node_positions[0];
    if indices
        .iter()
        .enumerate()
        .any(|(i, a)| i != mask_pos && !matches!(a, IrArg::None))
    {
        return false;
    }
    let data = match node.arg_node(2) {
        Some(n) => n,
        None => return false,
    };
    let accumulate = node.arg_bool_or(3, false);

    // Only the supported simple case: single bool mask, no accumulate, scalar data.
    if accumulate || graph.node_dtype(mask) != Some("torch.bool") {
        return false;
    }
    let dest_shape = match graph.node_shape_vec(dest) {
        Some(s) => s,
        None => return false,
    };
    let mask_shape = match graph.node_shape_vec(mask) {
        Some(s) => s,
        None => return false,
    };
    let data_shape = match graph.node_shape_vec(data) {
        Some(s) => s,
        None => return false,
    };
    if data_shape.iter().product::<i64>() != 1 {
        return false;
    }
    let data_dtype = graph.node_dtype(data).unwrap_or("torch.int8").to_string();
    let mask_meta = orig_meta.tensor_stamp("torch.bool");
    let data_meta = orig_meta.tensor_stamp(&data_dtype);
    let dest_rank = dest_shape.len();

    let mut prev: Option<NodeId> = None;

    // Place the mask at its index position with singleton dims elsewhere so it
    // broadcasts against the destination (mask covers dims `mask_pos..`).
    let mask_node = {
        let mut target = vec![1_i64; dest_rank];
        for (i, &d) in mask_shape.iter().enumerate() {
            if mask_pos + i < dest_rank {
                target[mask_pos + i] = d;
            }
        }
        if target == mask_shape {
            mask
        } else {
            let v = view_copy(
                graph,
                format!("{}_mask_reshape", node_name),
                &mask_meta,
                mask,
                &target,
            );
            graph.anchor_to(v, id);
            prev = Some(v);
            v
        }
    };

    // Reshape scalar data to all-ones so it broadcasts to any destination shape.
    let data_node = if data_shape.len() != dest_rank {
        let ones = vec![1_i64; dest_rank];
        let v = view_copy(
            graph,
            format!("{}_data_reshape", node_name),
            &data_meta,
            data,
            &ones,
        );
        graph.anchor_to(v, id);
        if let Some(p) = prev {
            graph.anchor_after(v, p);
        }
        prev = Some(v);
        v
    } else {
        data
    };

    let result = graph.add_node(
        NodeOp::CallFunction,
        format!("{}_where", node_name),
        "aten.where.self".to_string(),
        vec![
            IrArg::Node(mask_node),
            IrArg::Node(data_node),
            IrArg::Node(dest),
        ],
        orig_meta,
    );
    if let Some(p) = prev {
        graph.anchor_after(result, p);
    }
    graph.replace_and_erase(id, result);
    true
}

pub struct IrNormalizeIndexPutBoolMaskPass;

impl IrPass for IrNormalizeIndexPutBoolMaskPass {
    fn name(&self) -> &str {
        "IrNormalizeIndexPutBoolMask"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(rewrite_one);
        if modified {
            log_line(format_args!(
                "[IrNormalizeIndexPutBoolMask] rewrote bool-mask index_put to where"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named<'a>(graph: &'a IrGraph, name: &str) -> &'a crate::fx_ir::IrNode {
        graph
            .nodes
            .iter()
            .find(|n| n.alive && n.name == name)
            .unwrap_or_else(|| panic!("missing node {name}"))
    }

    /// dest(dest_shape,int8) + mask(mask_shape,bool) + data(data_shape) index_put.
    fn graph_for(
        dest_shape: &[i64],
        mask_shape: &[i64],
        data_shape: &[i64],
        mask_dtype: &str,
        accumulate: bool,
    ) -> (IrGraph, NodeId, NodeId, NodeId) {
        let mut graph = IrGraph::empty();
        let dest = graph.add_node(
            NodeOp::Placeholder,
            "dest".into(),
            "dest".into(),
            vec![],
            NodeMeta::tensor(dest_shape, "torch.int8"),
        );
        let mask = graph.add_node(
            NodeOp::Placeholder,
            "mask".into(),
            "mask".into(),
            vec![],
            NodeMeta::tensor(mask_shape, mask_dtype),
        );
        let data = graph.add_node(
            NodeOp::Placeholder,
            "data".into(),
            "data".into(),
            vec![],
            NodeMeta::tensor(data_shape, "torch.int8"),
        );
        let mut args = vec![
            IrArg::Node(dest),
            IrArg::List(vec![IrArg::Node(mask)]),
            IrArg::Node(data),
        ];
        if accumulate {
            args.push(IrArg::Bool(true));
        }
        let ip = graph.add_node(
            NodeOp::CallFunction,
            "idxput".into(),
            "aten.index_put.default".into(),
            args,
            NodeMeta::tensor(dest_shape, "torch.int8"),
        );
        graph.add_node(
            NodeOp::Output,
            "output".into(),
            "output".into(),
            vec![IrArg::Node(ip)],
            NodeMeta::default(),
        );
        (graph, dest, mask, ip)
    }

    #[test]
    fn rewrites_bool_mask_scalar_to_where() {
        // dest [4,3], same-rank bool mask, scalar data [1] -> data reshaped to [1,1].
        let (mut graph, dest, mask, ip) = graph_for(&[4, 3], &[4, 3], &[1], "torch.bool", false);
        assert!(IrNormalizeIndexPutBoolMaskPass.run(&mut graph));
        assert!(!graph.nodes[ip].alive);

        let data_reshape = named(&graph, "idxput_data_reshape");
        assert_eq!(data_reshape.target, "aten.view_copy.default");
        assert_eq!(data_reshape.meta.val_shape().unwrap(), &[1, 1]);

        let wh = named(&graph, "idxput_where");
        assert_eq!(wh.target, "aten.where.self");
        assert_eq!(wh.args[0].as_node(), Some(mask)); // same-rank mask: not reshaped
        assert_eq!(wh.args[1].as_node(), Some(data_reshape.id));
        assert_eq!(wh.args[2].as_node(), Some(dest));
        assert_eq!(named(&graph, "output").args[0].as_node(), Some(wh.id));
    }

    #[test]
    fn pads_lower_rank_mask() {
        // dest [4,3], mask [4] (rank 1) -> reshaped to [4,1].
        let (mut graph, _dest, _mask, _ip) = graph_for(&[4, 3], &[4], &[1], "torch.bool", false);
        assert!(IrNormalizeIndexPutBoolMaskPass.run(&mut graph));
        assert_eq!(
            named(&graph, "idxput_mask_reshape")
                .meta
                .val_shape()
                .unwrap(),
            &[4, 1]
        );
    }

    #[test]
    fn skips_int_mask_and_accumulate() {
        // int index -> leave for the SCATTER path.
        let (mut g1, ..) = graph_for(&[4, 3], &[4, 3], &[1], "torch.int32", false);
        assert!(!IrNormalizeIndexPutBoolMaskPass.run(&mut g1));
        // accumulate=True -> unsupported, skip.
        let (mut g2, ..) = graph_for(&[4, 3], &[4, 3], &[1], "torch.bool", true);
        assert!(!IrNormalizeIndexPutBoolMaskPass.run(&mut g2));
    }
}
