//! Pure Rust IrPass version of FxDecomposeRepeatInterleavePass.
//!
//! Decomposes `repeat_interleave.self_int(x, repeats, dim)` into:
//!   1. unsqueeze_copy(x, dim + 1)
//!   2. repeat(unsqueezed, multiples)  — multiples[dim+1] = repeats
//!   3. view_copy(repeated, final_shape) — final_shape[dim] *= repeats

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeOp};
use crate::runner_logging::log_line;

fn i64_list(vals: &[i64]) -> IrArg {
    IrArg::List(vals.iter().map(|&v| IrArg::Int(v)).collect())
}

pub struct IrDecomposeRepeatInterleavePass;

impl IrPass for IrDecomposeRepeatInterleavePass {
    fn name(&self) -> &str {
        "IrDecomposeRepeatInterleave"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            if !graph.nodes[id]
                .target
                .contains("repeat_interleave.self_int")
            {
                continue;
            }

            // args: (x, repeats, dim)
            if graph.nodes[id].args.len() < 3 {
                continue;
            }

            let input_id = match graph.nodes[id].args[0].as_node() {
                Some(nid) => nid,
                None => continue,
            };
            let repeats = match graph.nodes[id].args[1].as_int() {
                Some(r) if r >= 0 => r,
                _ => continue,
            };
            let dim_raw = match &graph.nodes[id].args[2] {
                IrArg::None => continue,
                IrArg::Int(d) => *d,
                _ => continue,
            };

            let input_shape = match graph.nodes[input_id].meta.val_shape() {
                Some(s) => s.to_vec(),
                None => continue,
            };
            let rank = input_shape.len() as i64;
            if rank == 0 {
                continue;
            }

            let dim = if dim_raw < 0 { dim_raw + rank } else { dim_raw };
            if !(0..rank).contains(&dim) {
                continue;
            }

            let dtype = graph.nodes[id]
                .meta
                .val_dtype()
                .unwrap_or("torch.float32")
                .to_string();
            let node_name = graph.nodes[id].name.clone();

            // 1. unsqueeze_copy(x, dim + 1)
            let unsqueeze_dim = dim + 1;
            let mut unsqueeze_shape = Vec::with_capacity(input_shape.len() + 1);
            for (i, &d) in input_shape.iter().enumerate() {
                if i as i64 == unsqueeze_dim {
                    unsqueeze_shape.push(1);
                }
                unsqueeze_shape.push(d);
            }
            if unsqueeze_dim as usize >= input_shape.len() {
                unsqueeze_shape.push(1);
            }

            let mut unsqueeze_meta = graph.nodes[id].meta.clone();
            unsqueeze_meta.set_val(FakeTensorMeta {
                shape: unsqueeze_shape.clone(),
                dtype: dtype.clone(),
            });
            let unsqueezed = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_unsqueeze", node_name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(input_id), IrArg::Int(unsqueeze_dim)],
                unsqueeze_meta,
            );
            graph.anchor_to(unsqueezed, id);

            // 2. repeat(unsqueezed, multiples) — multiples[unsqueeze_dim] = repeats
            let repeat_rank = (rank + 1) as usize;
            let mut multiples = vec![1i64; repeat_rank];
            multiples[unsqueeze_dim as usize] = repeats;

            let mut repeated_shape = unsqueeze_shape;
            repeated_shape[unsqueeze_dim as usize] = repeats;

            let mut repeat_meta = graph.nodes[id].meta.clone();
            repeat_meta.set_val(FakeTensorMeta {
                shape: repeated_shape,
                dtype: dtype.clone(),
            });
            let repeated = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_repeat", node_name),
                "aten.repeat.default".to_string(),
                vec![IrArg::Node(unsqueezed), i64_list(&multiples)],
                repeat_meta,
            );
            graph.anchor_to(repeated, id);

            // 3. view_copy(repeated, final_shape) — final_shape[dim] *= repeats
            let mut final_shape = input_shape;
            final_shape[dim as usize] *= repeats;

            let mut view_meta = graph.nodes[id].meta.clone();
            view_meta.set_val(FakeTensorMeta {
                shape: final_shape.clone(),
                dtype,
            });
            let result = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_view", node_name),
                "aten.view_copy.default".to_string(),
                vec![IrArg::Node(repeated), i64_list(&final_shape)],
                view_meta,
            );

            graph.replace_and_erase(id, result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeRepeatInterleave] decomposed repeat_interleave ops"
            ));
        }
        modified
    }
}
