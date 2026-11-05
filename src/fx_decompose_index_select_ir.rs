//! Pure Rust IrPass: constant index_select → SLICE/CONCAT, otherwise GATHER.

use crate::fx_constant_tensor_eval::placeholder_is_constant_input;
use crate::fx_index_domain::{
    bool_stamp, gather_work_stamp_dtype_of_or_float32, index_stamp, node_index_dtype_or_default,
    slice_concat_for_indices,
};
use crate::fx_ir::{
    i64_list_arg as i64_list, shape_product, wrap_dim, IrArg, IrGraph, IrPass, NodeOp,
};
use crate::fx_target::is_index_select_target;
use crate::graph::ir::DType;
use crate::runner_logging::log_line;

pub struct IrDecomposeIndexSelectPass;

impl IrPass for IrDecomposeIndexSelectPass {
    fn name(&self) -> &str {
        "IrDecomposeIndexSelect"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_index_select_target(&graph.nodes[id].target) {
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

            let x_shape = match graph.node_shape_vec(x_id) {
                Some(s) => s,
                None => continue,
            };
            let index_shape = match graph.node_shape_vec(index_id) {
                Some(s) => s,
                None => continue,
            };
            if x_shape.is_empty() || index_shape.len() != 1 {
                continue;
            }

            let Some(dim_norm) = wrap_dim(dim, x_shape.len()) else {
                continue;
            };
            let index_node = &graph.nodes[index_id];
            if x_shape.iter().all(|&size| size >= 0)
                && (placeholder_is_constant_input(index_node) || index_node.op == NodeOp::GetAttr)
            {
                if let Some(indices) = index_node
                    .tensor
                    .as_ref()
                    .filter(|tensor| {
                        tensor.data.shape.len() == 1
                            && matches!(tensor.data.dtype, DType::Int32 | DType::Int64)
                    })
                    .and_then(|tensor| tensor.data.vec_i64())
                    .filter(|indices| !indices.is_empty())
                {
                    let dim_size = x_shape[dim_norm];
                    assert!(
                        indices.iter().all(|&index| index >= 0 && index < dim_size),
                        "index_select index out of range for dimension of size {dim_size}"
                    );
                    let result = slice_concat_for_indices(graph, id, x_id, dim_norm, &indices);
                    graph.replace_and_erase(id, result);
                    modified = true;
                    continue;
                }
            }
            let before: Vec<i64> = x_shape[..dim_norm].to_vec();
            let after: Vec<i64> = x_shape[dim_norm + 1..].to_vec();
            let n = shape_product(&before);
            let k = x_shape[dim_norm];
            let c = shape_product(&after);
            let w = index_shape[0];

            let name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let idx_dtype = node_index_dtype_or_default(graph, index_id);

            let x_meta = graph.nodes[x_id].meta.clone();
            let (gather_meta, needs_bool_cast) =
                gather_work_stamp_dtype_of_or_float32(&orig_meta, &x_meta);
            let index_meta = index_stamp(&orig_meta, &idx_dtype);
            let bool_meta = bool_stamp(&orig_meta);
            let mut cursor = graph.rewrite_cursor(id);

            let x_for_gather = if needs_bool_cast {
                cursor.dtype_cast(
                    format!("{}_bool_to_i8", name),
                    x_id,
                    gather_meta.dtype(),
                    gather_meta.preserve_qparams(x_shape.clone()),
                )
            } else {
                x_id
            };

            // Flatten x to (N, K, C)
            let values = cursor.view_copy(
                format!("{}_values", name),
                x_for_gather,
                &[n, k, c],
                gather_meta.preserve_qparams(vec![n, k, c]),
            );

            // Index: unsqueeze(0) -> (1, W), then expand to (N, W)
            let idx_unsq = cursor.call_function(
                format!("{}_idx_unsq", name),
                "aten.unsqueeze_copy.default".to_string(),
                vec![IrArg::Node(index_id), IrArg::Int(0)],
                index_meta.preserve_qparams(vec![1, w]),
            );

            let idx_exp = cursor.call_function(
                format!("{}_idx_exp", name),
                "aten.expand_copy.default".to_string(),
                vec![IrArg::Node(idx_unsq), i64_list(&[n, w])],
                index_meta.preserve_qparams(vec![n, w]),
            );

            // GATHER(values, indices) -> (N, W, C)
            let gathered = cursor.call_function(
                format!("{}_gather", name),
                "backend.tosa.GATHER.default".to_string(),
                vec![IrArg::Node(values), IrArg::Node(idx_exp)],
                gather_meta.preserve_qparams(vec![n, w, c]),
            );

            // Reshape to (...before, W, ...after)
            let out_shape: Vec<i64> = before
                .iter()
                .copied()
                .chain(std::iter::once(w))
                .chain(after.iter().copied())
                .collect();
            if needs_bool_cast {
                let result = cursor.view_copy(
                    format!("{}_out", name),
                    gathered,
                    &out_shape,
                    gather_meta.preserve_qparams(out_shape.clone()),
                );
                let final_result = cursor.dtype_cast(
                    format!("{}_i8_to_bool", name),
                    result,
                    "torch.bool",
                    bool_meta.preserve_qparams(out_shape),
                );
                cursor.replace_anchor_with(final_result);
            } else {
                let result = cursor.view_copy(
                    format!("{}_out", name),
                    gathered,
                    &out_shape,
                    gather_meta.preserve_qparams(out_shape.clone()),
                );
                cursor.replace_anchor_with(result);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{NodeMeta, QParamMap};
    use crate::lowering_effects::LoweringInputKind;
    use crate::store::TensorData;
    use crate::test_ir::{buffer, call, output, placeholder, qparam};

    fn constant_select_graph(indices: &[i64], dtype: &str) -> (IrGraph, usize, usize) {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [2, 5, 3], dtype);
        let index = buffer(
            &mut graph,
            "index",
            TensorData {
                data: indices
                    .iter()
                    .flat_map(|index| index.to_le_bytes())
                    .collect(),
                shape: vec![indices.len() as i64],
                dtype: DType::Int64,
            },
        );
        graph.nodes[index].meta = NodeMeta::tensor([indices.len() as i64], "torch.int64");
        let select = call(
            &mut graph,
            "select",
            "aten.index_select.default",
            vec![IrArg::Node(x), IrArg::Int(-2), IrArg::Node(index)],
            [2, indices.len() as i64, 3],
            dtype,
        );
        output(&mut graph, select);
        (graph, select, index)
    }

    #[test]
    fn contiguous_constant_indices_use_one_slice_without_bool_casts() {
        for indices in [&[2][..], &[1, 2, 3][..]] {
            let (mut graph, select, _) = constant_select_graph(indices, "torch.bool");
            assert!(IrDecomposeIndexSelectPass.run(&mut graph));
            let result = graph.replacement_of(select).unwrap();
            let slice = &graph.nodes[result];
            assert_eq!(slice.target, "aten.slice_copy.Tensor");
            assert_eq!(slice.arg_int(1), Some(1));
            assert_eq!(slice.arg_int(2), Some(indices[0]));
            assert_eq!(slice.arg_int(3), Some(indices[indices.len() - 1] + 1));
            assert_eq!(slice.meta.val_dtype(), Some("torch.bool"));
            assert_eq!(
                slice.meta.val_shape(),
                Some([2, indices.len() as i64, 3].as_slice())
            );
            assert_eq!(graph.live_call_function_ids(), vec![result]);
        }
    }

    #[test]
    fn reordered_repeated_indices_keep_slice_order_and_qparams() {
        let indices = [3, 1, 3];
        let (mut graph, select, _) = constant_select_graph(&indices, "torch.int8");
        let mut qparams = QParamMap::new();
        qparams.insert(0, qparam(0.2, 3, "torch.int8"));
        graph.nodes[select].meta.set_input_qparams(qparams.clone());
        graph.nodes[select].meta.set_output_qparams(qparams.clone());

        assert!(IrDecomposeIndexSelectPass.run(&mut graph));
        let cat = &graph.nodes[graph.replacement_of(select).unwrap()];
        assert_eq!(cat.target, "aten.cat.default");
        assert_eq!(cat.arg_int(1), Some(1));
        assert_eq!(cat.meta.val_shape(), Some([2, 3, 3].as_slice()));
        assert!(cat
            .meta
            .output_qparam()
            .unwrap()
            .same_scale_zp(qparams.get(&0).unwrap()));
        let IrArg::List(slices) = &cat.args[0] else {
            panic!("cat inputs");
        };
        assert_eq!(slices[0].as_node(), slices[2].as_node());
        assert_eq!(
            graph
                .live_nodes()
                .filter(|node| node.target == "aten.slice_copy.Tensor")
                .count(),
            2
        );
        for (slice, index) in slices.iter().zip(indices) {
            let slice = &graph.nodes[slice.as_node().unwrap()];
            assert_eq!(slice.arg_int(2), Some(index));
            assert_eq!(slice.arg_int(3), Some(index + 1));
            assert_eq!(slice.meta.val_shape(), Some([2, 1, 3].as_slice()));
            assert_eq!(slice.meta.val_dtype(), Some("torch.int8"));
            assert!(slice
                .meta
                .output_qparam()
                .unwrap()
                .same_scale_zp(qparams.get(&0).unwrap()));
        }
        assert!(!graph
            .live_nodes()
            .any(|node| node.target == "backend.tosa.GATHER.default"));
    }

    #[test]
    fn constant_sources_include_parameters_lifted_constants_and_attributes() {
        for kind in [
            LoweringInputKind::Parameter,
            LoweringInputKind::Buffer,
            LoweringInputKind::LiftedTensorConstant,
        ] {
            let (mut graph, _, index) = constant_select_graph(&[4, 0], "torch.int16");
            let tensor = graph.nodes[index].tensor.as_mut().unwrap();
            let crate::fx_ir::TensorSource::ExportedProgramInput(binding) = &mut tensor.source
            else {
                panic!("constant binding");
            };
            binding.input_kind = kind;
            tensor.data.dtype = DType::Int32;
            tensor.data.data = [4i32.to_le_bytes(), 0i32.to_le_bytes()].concat();
            graph.nodes[index].meta = NodeMeta::tensor([2], "torch.int32");
            assert!(IrDecomposeIndexSelectPass.run(&mut graph));
            assert!(graph
                .live_nodes()
                .any(|node| node.target == "aten.cat.default"));
        }
        let (mut graph, _, index) = constant_select_graph(&[1, 2], "torch.int32");
        graph.nodes[index].op = NodeOp::GetAttr;
        graph.nodes[index].tensor.as_mut().unwrap().source =
            crate::fx_ir::TensorSource::GraphModuleAttr;
        assert!(IrDecomposeIndexSelectPass.run(&mut graph));
        assert!(graph
            .live_nodes()
            .any(|node| node.target == "aten.slice_copy.Tensor"));
    }

    #[test]
    fn runtime_empty_and_symbolic_indices_keep_gather_path() {
        let (mut runtime, _, index) = constant_select_graph(&[1, 2], "torch.int8");
        runtime.nodes[index].tensor = None;
        let (empty, _, _) = constant_select_graph(&[], "torch.int8");
        let (mut symbolic, select, _) = constant_select_graph(&[1, 2], "torch.int8");
        let input = symbolic.nodes[select].first_node_arg().unwrap();
        symbolic.nodes[input].meta = NodeMeta::tensor([2, -1, 3], "torch.int8");
        for mut graph in [runtime, empty, symbolic] {
            assert!(IrDecomposeIndexSelectPass.run(&mut graph));
            assert!(graph
                .live_nodes()
                .any(|node| node.target == "backend.tosa.GATHER.default"));
            assert!(!graph
                .live_nodes()
                .any(|node| node.target == "aten.slice_copy.Tensor"));
        }
    }

    #[test]
    fn out_of_range_constant_indices_are_rejected_before_rewriting() {
        for indices in [&[-1][..], &[5][..]] {
            let (mut graph, select, _) = constant_select_graph(indices, "torch.int8");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                IrDecomposeIndexSelectPass.run(&mut graph)
            }));
            assert!(result.is_err());
            assert!(graph.nodes[select].alive);
            assert_eq!(graph.live_call_function_ids(), vec![select]);
        }
    }
}
