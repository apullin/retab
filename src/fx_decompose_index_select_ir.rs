//! Pure Rust IrPass: index_select → view + unsqueeze + expand + GATHER + view.

use std::collections::HashMap;

use crate::fx_ir::{
    i64_list_arg as i64_list, FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp,
};
use crate::runner_logging::log_line;

fn meta_with(base: &NodeMeta, shape: Vec<i64>, dtype: &str) -> NodeMeta {
    let mut m = base.clone();
    m.set_val(FakeTensorMeta {
        shape,
        dtype: dtype.to_string(),
    });
    m
}

fn product(shape: &[i64]) -> i64 {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

pub struct IrDecomposeIndexSelectPass;

impl IrPass for IrDecomposeIndexSelectPass {
    fn name(&self) -> &str {
        "IrDecomposeIndexSelect"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id].target.contains("index_select.default") {
                continue;
            }
            if graph.nodes[id].args.len() != 3 {
                continue;
            }

            let x_id = match graph.nodes[id].args[0].as_node() {
                Some(n) => n,
                None => continue,
            };
            let dim = match graph.nodes[id].args[1].as_int() {
                Some(v) => v,
                None => continue,
            };
            let index_id = match graph.nodes[id].args[2].as_node() {
                Some(n) => n,
                None => continue,
            };

            let x_shape = match graph.nodes[x_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let index_shape = match graph.nodes[index_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            if x_shape.is_empty() || index_shape.len() != 1 {
                continue;
            }

            let x_rank = x_shape.len() as i64;
            let dim_norm = dim.rem_euclid(x_rank) as usize;
            let before: Vec<i64> = x_shape[..dim_norm].to_vec();
            let after: Vec<i64> = x_shape[dim_norm + 1..].to_vec();
            let n = product(&before);
            let k = x_shape[dim_norm];
            let c = product(&after);
            let w = index_shape[0];

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_dtype = graph.nodes[x_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();
            let idx_dtype = graph.nodes[index_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.int64")
                .to_string();

            let needs_bool_cast = x_dtype == "torch.bool";
            let gather_dtype = if needs_bool_cast {
                "torch.int8"
            } else {
                x_dtype.as_str()
            };

            let x_for_gather = if needs_bool_cast {
                let mut kwargs = HashMap::new();
                kwargs.insert("dtype".to_string(), IrArg::Dtype("torch.int8".to_string()));
                let cast = graph.add_anchored_call_function_with_kwargs(
                    id,
                    format!("{}_bool_to_i8", name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(x_id)],
                    kwargs,
                    meta_with(&orig_meta, x_shape.clone(), "torch.int8"),
                );
                cast
            } else {
                x_id
            };

            // Flatten x to (N, K, C)
            let values = graph.add_anchored_call_function(
                id,
                format!("{}_values", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(x_for_gather), i64_list(&[n, k, c])],
                meta_with(&orig_meta, vec![n, k, c], gather_dtype),
            );

            // Index: unsqueeze(0) → (1, W), then expand to (N, W)
            let idx_unsq = graph.add_anchored_call_function(
                id,
                format!("{}_idx_unsq", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(index_id), IrArg::Int(0)],
                meta_with(&orig_meta, vec![1, w], &idx_dtype),
            );

            let idx_exp = graph.add_anchored_call_function(
                id,
                format!("{}_idx_exp", name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(idx_unsq), i64_list(&[n, w])],
                meta_with(&orig_meta, vec![n, w], &idx_dtype),
            );

            // GATHER(values, indices) → (N, W, C)
            let gathered = graph.add_anchored_call_function(
                id,
                format!("{}_gather", name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(values), IrArg::Node(idx_exp)],
                meta_with(&orig_meta, vec![n, w, c], gather_dtype),
            );

            // Reshape to (...before, W, ...after)
            let out_shape: Vec<i64> = before
                .iter()
                .copied()
                .chain(std::iter::once(w))
                .chain(after.iter().copied())
                .collect();
            let result = graph.add_anchored_call_function(
                id,
                format!("{}_out", name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(gathered), i64_list(&out_shape)],
                meta_with(&orig_meta, out_shape.clone(), gather_dtype),
            );

            let result = if needs_bool_cast {
                let mut kwargs = HashMap::new();
                kwargs.insert("dtype".to_string(), IrArg::Dtype("torch.bool".to_string()));
                let cast = graph.add_anchored_call_function_with_kwargs(
                    id,
                    format!("{}_i8_to_bool", name),
                    "aten._to_copy.default".to_string(),
                    vec![IrArg::Node(result)],
                    kwargs,
                    meta_with(&orig_meta, out_shape, "torch.bool"),
                );
                cast
            } else {
                result
            };

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeIndexSelect] decomposed index_select ops"
            ));
        }
        modified
    }
}
