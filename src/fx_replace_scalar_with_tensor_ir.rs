//! Pure Rust IrPass for scalar binary ops that must be rewritten to tensor form.

use std::collections::HashMap;

use crate::fx_ir::{FakeTensorMeta, IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp};
use crate::runner_logging::log_line;

fn replacement_target(target: &str) -> Option<&'static str> {
    if target.contains("aten.__lshift__.Scalar") {
        Some("aten.bitwise_left_shift.Tensor")
    } else if target.contains("aten.__rshift__.Scalar") {
        Some("aten.bitwise_right_shift.Tensor")
    } else {
        None
    }
}

fn scalar_arg(arg: &IrArg) -> Option<IrArg> {
    if let Some(value) = arg.as_int() {
        Some(IrArg::Int(value))
    } else {
        arg.as_float().map(IrArg::Float)
    }
}

fn tensor_meta(shape: &[i64], dtype: &str) -> NodeMeta {
    let mut meta = NodeMeta::default();
    meta.set_val(FakeTensorMeta {
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
    });
    meta
}

pub struct IrReplaceScalarWithTensorPass;

impl IrPass for IrReplaceScalarWithTensorPass {
    fn name(&self) -> &str {
        "IrReplaceScalarWithTensor"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        let ids: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for id in ids {
            if !graph.nodes[id].alive || graph.nodes[id].op != NodeOp::CallFunction {
                continue;
            }
            let target = graph.nodes[id].target.clone();
            let Some(new_target) = replacement_target(&target) else {
                continue;
            };
            if graph.nodes[id].args.len() != 2 {
                continue;
            }
            let node_name = graph.nodes[id].name.clone();
            let output_meta = graph.nodes[id].meta.clone();

            let lhs_id = match graph.nodes[id].args[0].as_node() {
                Some(value) => value,
                None => continue,
            };
            let scalar_value = match scalar_arg(&graph.nodes[id].args[1]) {
                Some(value) => value,
                None => continue,
            };
            let lhs_dtype = graph.nodes[lhs_id]
                .meta
                .val_dtype()
                .unwrap_or("torch.int64")
                .to_string();
            let shape = [1];
            let mut full_kwargs = HashMap::new();
            full_kwargs.insert("dtype".to_string(), IrArg::Dtype(lhs_dtype.clone()));

            let full = graph.add_node_with_kwargs(
                NodeOp::CallFunction,
                format!("{}_scalar_tensor", node_name),
                "aten.full.default".to_string(),
                vec![IrArg::List(vec![IrArg::Int(1)]), scalar_value],
                full_kwargs,
                tensor_meta(&shape, &lhs_dtype),
            );
            graph.anchor_to(full, id);

            let replacement = graph.add_node(
                NodeOp::CallFunction,
                format!("{}_tensor", node_name),
                new_target.to_string(),
                vec![IrArg::Node(lhs_id), IrArg::Node(full)],
                output_meta,
            );
            graph.anchor_to(replacement, id);

            graph.replace_and_erase(id, replacement);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrReplaceScalarWithTensor] rewrote scalar shift ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(shape: &[i64], dtype: &str) -> NodeMeta {
        let mut meta = NodeMeta::default();
        meta.set_val(FakeTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
        });
        meta
    }

    fn placeholder(graph: &mut IrGraph, name: &str, shape: &[i64], dtype: &str) -> NodeId {
        graph.add_node(
            NodeOp::Placeholder,
            name.to_string(),
            name.to_string(),
            Vec::new(),
            meta(shape, dtype),
        )
    }

    #[test]
    fn rewrites_left_shift_scalar_to_tensor_form() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", &[2, 3], "torch.int16");
        let shift = graph.add_node(
            NodeOp::CallFunction,
            "lshift".to_string(),
            "aten.__lshift__.Scalar".to_string(),
            vec![IrArg::Node(x), IrArg::Int(3)],
            meta(&[2, 3], "torch.int16"),
        );

        assert!(IrReplaceScalarWithTensorPass.run(&mut graph));
        assert!(!graph.nodes[shift].alive);

        let scalar_tensor = graph.by_name("lshift_scalar_tensor").unwrap();
        assert_eq!(scalar_tensor.target, "aten.full.default");
        assert_eq!(scalar_tensor.args[1].as_int(), Some(3));
        assert!(matches!(
            scalar_tensor.kwargs.get("dtype"),
            Some(IrArg::Dtype(dtype)) if dtype == "torch.int16"
        ));

        let replacement = graph.by_name("lshift_tensor").unwrap();
        assert_eq!(replacement.target, "aten.bitwise_left_shift.Tensor");
        assert_eq!(replacement.args[0].as_node(), Some(x));
        assert_eq!(replacement.args[1].as_node(), Some(scalar_tensor.id));
    }
}
