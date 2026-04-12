//! Pure Rust IrPass: select_scatter → arange + view + full + eq + unsqueeze + expand + where.

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

fn meta_with_shape(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta {
        shape,
        dtype: dtype.to_string(),
    });
    m
}

pub struct IrDecomposeSelectScatterPass;

impl IrPass for IrDecomposeSelectScatterPass {
    fn name(&self) -> &str {
        "IrDecomposeSelectScatter"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("select_scatter.default") {
                continue;
            }
            if graph.nodes[id].args.len() != 4 {
                continue;
            }

            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(n) => n,
                None => continue,
            };
            let src_id = match graph.nodes[id].args[1].as_node() {
                Some(n) => n,
                None => continue,
            };
            let dim = match graph.nodes[id].args[2].as_int() {
                Some(v) => v,
                None => continue,
            };
            let mut index = match graph.nodes[id].args[3].as_int() {
                Some(v) => v,
                None => continue,
            };

            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            if input_shape.is_empty() {
                continue;
            }
            let rank = input_shape.len() as i64;
            let dim_norm = dim.rem_euclid(rank) as usize;
            let dim_size = input_shape[dim_norm];
            if index < 0 {
                index += dim_size;
            }

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let input_dtype = graph.nodes[input_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();

            // 1. arange(0, dim_size) → shape [dim_size], dtype int32
            let arange_id = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_arange", node_name),
                "aten.arange.start_step".to_string(),
                vec![IrArg::Int(0), IrArg::Int(dim_size)],
                meta_with_shape(&orig_meta, vec![dim_size], "torch.int32"),
            );
            graph.anchor_to(arange_id, id);

            // 2. view(arange, [1, ..., dim_size, ..., 1])
            let mut mask_shape = vec![1i64; rank as usize];
            mask_shape[dim_norm] = dim_size;
            let viewed_arange = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_arange_view", node_name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(arange_id), i64_list(&mask_shape)],
                meta_with_shape(&orig_meta, mask_shape.clone(), "torch.int32"),
            );
            graph.anchor_to(viewed_arange, id);

            // 3. full([1], index, dtype=int32) — scalar index tensor
            let index_tensor = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_index_full", node_name),
                "aten.full.default".to_string(),
                vec![i64_list(&[1]), IrArg::Int(index)],
                meta_with_shape(&orig_meta, vec![1], "torch.int32"),
            );
            graph.anchor_to(index_tensor, id);

            // 4. eq(viewed_arange, index_tensor) → boolean mask
            let eq_mask = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_eq", node_name),
                "aten.eq.Tensor".to_string(),
                vec![IrArg::Node(viewed_arange), IrArg::Node(index_tensor)],
                meta_with_shape(&orig_meta, mask_shape, "torch.bool"),
            );
            graph.anchor_to(eq_mask, id);

            // 5. unsqueeze(src, dim) — restore dimension that select removed
            let src_shape = graph.nodes[src_id]
                .meta
                .val_shape()
                .map(|s| s.to_vec())
                .unwrap_or_default();
            let mut unsqueezed_shape = src_shape.clone();
            unsqueezed_shape.insert(dim_norm, 1);
            let mut unsqueeze_meta = graph.nodes[src_id].meta.clone();
            unsqueeze_meta.set_val(FakeTensorMeta {
                shape: unsqueezed_shape,
                dtype: input_dtype.clone(),
            });
            let unsqueeze_node = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_unsqueeze", node_name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(src_id), IrArg::Int(dim_norm as i64)],
                unsqueeze_meta,
            );
            graph.anchor_to(unsqueeze_node, id);

            // 6. expand(unsqueezed, input_shape)
            let mut expand_meta = graph.nodes[src_id].meta.clone();
            expand_meta.set_val(FakeTensorMeta {
                shape: input_shape.clone(),
                dtype: input_dtype,
            });
            let expand_node = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_expand", node_name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(unsqueeze_node), i64_list(&input_shape)],
                expand_meta,
            );
            graph.anchor_to(expand_node, id);

            // 7. where(mask, expanded_src, input) → result
            let where_node = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_where", node_name),
                "aten.where.self".to_string(),
                vec![
                    IrArg::Node(eq_mask),
                    IrArg::Node(expand_node),
                    IrArg::Node(input_id),
                ],
                orig_meta,
            );
            graph.anchor_to(where_node, id);

            graph.replace_and_erase(id, where_node);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeSelectScatter] decomposed select_scatter ops"
            ));
        }
        modified
    }
}
