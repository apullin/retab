//! Port of master NormalizeIndexPutNoneIndicesPass.
//!
//! `index_put(dest, [.., None, ..], data)` selects whole dims where the index is
//! None. retab's rewrite expands None -> arange(full range), which builds a
//! cartesian index product that doesn't match the data shape (invalid TOSA).
//! Instead, move the None-indexed dims to the back by permuting `dest` (and `data`),
//! run index_put with only the explicit indices, then permute the result back.
//! Trailing None indices need no permute — they are simply dropped (implicit).
//! Runs before IrRewriteIndexPut, which then handles the None-free index_put.

use crate::fx_ir::{IrArg, IrGraph, IrPass, NodeId, NodeMeta};
use crate::fx_target::is_index_put_default_target;
use crate::runner_logging::log_line;

fn permute_shape(shape: &[i64], order: &[usize]) -> Vec<i64> {
    order.iter().map(|&d| shape[d]).collect()
}

fn is_identity(order: &[usize]) -> bool {
    order.iter().enumerate().all(|(i, &d)| i == d)
}

/// Create a permute_copy node; returns (id, permuted shape).
#[allow(clippy::too_many_arguments)]
fn add_permute(
    graph: &mut IrGraph,
    anchor: NodeId,
    input_id: NodeId,
    input_shape: &[i64],
    input_meta: &NodeMeta,
    order: &[usize],
    name: String,
    prev: &mut Option<NodeId>,
) -> (NodeId, Vec<i64>) {
    let out_shape = permute_shape(input_shape, order);
    let dtype = input_meta.val_dtype_or_float32().to_string();
    let meta = input_meta.cloned_with_val(out_shape.clone(), &dtype);
    let order_arg = IrArg::List(order.iter().map(|&d| IrArg::Int(d as i64)).collect());
    let n = graph.add_anchored_call_function(
        anchor,
        name,
        "aten.permute_copy.default",
        vec![IrArg::Node(input_id), order_arg],
        meta,
    );
    if let Some(p) = *prev {
        graph.anchor_after(n, p);
    }
    *prev = Some(n);
    (n, out_shape)
}

/// data dim order per master `_get_data_dim_order`.
fn data_dim_order(explicit_dims: &[usize], destination_dim_order: &[usize]) -> Vec<usize> {
    let normalized_non_index = &destination_dim_order[explicit_dims.len()..];
    if explicit_dims.len() > 1 {
        // identity over the non-index dims
        (0..normalized_non_index.len()).collect()
    } else {
        // single explicit index: same order as destination
        destination_dim_order.to_vec()
    }
}

fn rewrite_one(graph: &mut IrGraph, id: NodeId) -> bool {
    let target = graph.nodes[id].target.clone();
    if !is_index_put_default_target(&target) {
        return false;
    }
    if graph.nodes[id].args.len() < 3 {
        return false;
    }
    let dest_id = match graph.nodes[id].args[0].as_node() {
        Some(n) => n,
        None => return false,
    };
    let indices_list = match &graph.nodes[id].args[1] {
        IrArg::List(items) => items.clone(),
        _ => return false,
    };
    // any None?
    if !indices_list.iter().any(|a| matches!(a, IrArg::None)) {
        return false;
    }
    let data_id = match graph.nodes[id].args[2].as_node() {
        Some(n) => n,
        None => return false,
    };
    let rest_args: Vec<IrArg> = graph.nodes[id].args[3..].to_vec();

    let dest_shape = match graph.node_shape_vec(dest_id) {
        Some(s) => s,
        None => return false,
    };
    let dest_rank = dest_shape.len();
    let dest_meta = graph.nodes[dest_id].meta.clone();
    let data_shape = match graph.node_shape_vec(data_id) {
        Some(s) => s,
        None => return false,
    };
    let data_meta = graph.nodes[data_id].meta.clone();
    let orig_meta = graph.nodes[id].meta.clone();
    let name = graph.nodes[id].name.clone();

    let explicit_dims: Vec<usize> = indices_list
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, IrArg::Node(_)))
        .map(|(i, _)| i)
        .collect();
    let none_dims: Vec<usize> = indices_list
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, IrArg::None))
        .map(|(i, _)| i)
        .collect();
    if explicit_dims.is_empty() {
        return false; // need at least one explicit index
    }
    let trailing_dims: Vec<usize> = (indices_list.len()..dest_rank).collect();
    let destination_dim_order: Vec<usize> = explicit_dims
        .iter()
        .chain(none_dims.iter())
        .chain(trailing_dims.iter())
        .copied()
        .collect();
    if destination_dim_order.len() != dest_rank {
        return false;
    }
    let needs_dest_permute = !is_identity(&destination_dim_order);

    let mut prev: Option<NodeId> = None;

    // permute destination
    let (dest_use, _dest_use_shape) = if needs_dest_permute {
        add_permute(
            graph,
            id,
            dest_id,
            &dest_shape,
            &dest_meta,
            &destination_dim_order,
            format!("{}_destperm", name),
            &mut prev,
        )
    } else {
        (dest_id, dest_shape.clone())
    };

    // permute (and maybe reshape) data
    let ddo = data_dim_order(&explicit_dims, &destination_dim_order);
    let needs_data_permute = !is_identity(&ddo);
    let data_use = if needs_data_permute {
        let aligned_rank = ddo.len();
        let (reshaped_id, reshaped_shape) = if data_shape.len() < aligned_rank {
            let mut padded = vec![1i64; aligned_rank - data_shape.len()];
            padded.extend_from_slice(&data_shape);
            let dtype = data_meta.val_dtype_or_float32().to_string();
            let meta = data_meta.cloned_with_val(padded.clone(), &dtype);
            let n = graph.add_anchored_view_copy(
                id,
                format!("{}_datareshape", name),
                data_id,
                &padded,
                meta,
            );
            if let Some(p) = prev {
                graph.anchor_after(n, p);
            }
            prev = Some(n);
            (n, padded)
        } else {
            (data_id, data_shape.clone())
        };
        let (p, _) = add_permute(
            graph,
            id,
            reshaped_id,
            &reshaped_shape,
            &data_meta,
            &ddo,
            format!("{}_dataperm", name),
            &mut prev,
        );
        p
    } else {
        data_id
    };

    // explicit indices only
    let explicit_indices: Vec<IrArg> = explicit_dims
        .iter()
        .map(|&i| indices_list[i].clone())
        .collect();

    // new index_put on the normalized operands
    let mut new_args = vec![
        IrArg::Node(dest_use),
        IrArg::List(explicit_indices),
        IrArg::Node(data_use),
    ];
    new_args.extend(rest_args);
    // result meta = permuted dest meta if permuted, else original
    let put_shape = permute_shape(&dest_shape, &destination_dim_order);
    let put_dtype = orig_meta.val_dtype_or_float32().to_string();
    let put_meta = if needs_dest_permute {
        orig_meta.cloned_with_val(put_shape.clone(), &put_dtype)
    } else {
        orig_meta.clone()
    };
    let new_put = graph.add_anchored_call_function(
        id,
        format!("{}_norm", name),
        "aten.index_put.default",
        new_args,
        put_meta,
    );
    if let Some(p) = prev {
        graph.anchor_after(new_put, p);
    }
    prev = Some(new_put);

    // permute back
    let final_id = if needs_dest_permute {
        let mut inv = vec![0usize; destination_dim_order.len()];
        for (new_dim, &orig_dim) in destination_dim_order.iter().enumerate() {
            inv[orig_dim] = new_dim;
        }
        let (p, _) = add_permute(
            graph,
            id,
            new_put,
            &put_shape,
            &orig_meta,
            &inv,
            format!("{}_invperm", name),
            &mut prev,
        );
        p
    } else {
        new_put
    };

    graph.replace_and_erase(id, final_id);
    true
}

pub struct IrNormalizeIndexPutNoneIndicesPass;

impl IrPass for IrNormalizeIndexPutNoneIndicesPass {
    fn name(&self) -> &str {
        "IrNormalizeIndexPutNoneIndices"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(rewrite_one);
        if modified {
            log_line(format_args!(
                "[IrNormalizeIndexPutNoneIndices] permuted None-indexed dims out of index_put"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeOp;

    /// Build index_put(dest, indices, data) where indices is given as a list of
    /// Option<NodeId> (None -> IrArg::None).
    fn build(
        dest_shape: &[i64],
        indices: &[Option<&[i64]>],
        data_shape: &[i64],
    ) -> (IrGraph, NodeId) {
        let mut g = IrGraph::empty();
        let dest = g.add_node(
            NodeOp::Placeholder,
            "dest".into(),
            "dest".into(),
            vec![],
            NodeMeta::tensor(dest_shape, "torch.int8"),
        );
        let mut idx_args = vec![];
        for (i, idx) in indices.iter().enumerate() {
            match idx {
                Some(s) => {
                    let n = g.add_node(
                        NodeOp::Placeholder,
                        format!("idx{i}"),
                        format!("idx{i}"),
                        vec![],
                        NodeMeta::tensor(*s, "torch.int8"),
                    );
                    idx_args.push(IrArg::Node(n));
                }
                None => idx_args.push(IrArg::None),
            }
        }
        let data = g.add_node(
            NodeOp::Placeholder,
            "data".into(),
            "data".into(),
            vec![],
            NodeMeta::tensor(data_shape, "torch.int8"),
        );
        let put = g.add_node(
            NodeOp::CallFunction,
            "ip".into(),
            "aten.index_put.default".into(),
            vec![IrArg::Node(dest), IrArg::List(idx_args), IrArg::Node(data)],
            NodeMeta::tensor(dest_shape, "torch.int8"),
        );
        g.add_node(
            NodeOp::Output,
            "output".into(),
            "output".into(),
            vec![IrArg::Node(put)],
            NodeMeta::default(),
        );
        (g, put)
    }

    fn live_with_target<'a>(g: &'a IrGraph, t: &str) -> Vec<&'a crate::fx_ir::IrNode> {
        g.nodes
            .iter()
            .filter(|n| n.alive && n.target == t)
            .collect()
    }

    #[test]
    fn trailing_none_is_dropped_no_permute() {
        // dest [4,3], indices (idx[3], None) -> drop trailing None, no permute.
        let (mut g, put) = build(&[4, 3], &[Some(&[3]), None], &[3, 3]);
        assert!(IrNormalizeIndexPutNoneIndicesPass.run(&mut g));
        assert!(!g.nodes[put].alive);
        // no permute created
        assert_eq!(live_with_target(&g, "aten.permute_copy.default").len(), 0);
        // new index_put with single explicit index, no None
        let puts = live_with_target(&g, "aten.index_put.default");
        assert_eq!(puts.len(), 1);
        if let IrArg::List(items) = &puts[0].args[1] {
            assert_eq!(items.len(), 1);
            assert!(matches!(items[0], IrArg::Node(_)));
        } else {
            panic!("indices not a list");
        }
    }

    #[test]
    fn leading_none_permutes() {
        // dest [5,3,4], indices (None, idx[2], None) -> explicit=[1], none=[0,2].
        // dest order = [1,0,2] (non-identity) -> permute + inverse permute.
        let (mut g, put) = build(&[5, 3, 4], &[None, Some(&[2]), None], &[2]);
        assert!(IrNormalizeIndexPutNoneIndicesPass.run(&mut g));
        assert!(!g.nodes[put].alive);
        // dest permute + inverse permute = 2 (data is single-explicit so may add 1 more)
        assert!(live_with_target(&g, "aten.permute_copy.default").len() >= 2);
        let puts = live_with_target(&g, "aten.index_put.default");
        assert_eq!(puts.len(), 1);
        if let IrArg::List(items) = &puts[0].args[1] {
            assert_eq!(items.len(), 1); // only the explicit index
        } else {
            panic!("indices not a list");
        }
    }

    #[test]
    fn no_none_is_skipped() {
        let (mut g, _put) = build(&[4, 3], &[Some(&[3])], &[3, 3]);
        assert!(!IrNormalizeIndexPutNoneIndicesPass.run(&mut g));
    }
}
