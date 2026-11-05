use std::collections::{BTreeMap, BTreeSet};

use crate::fx_ir;
use crate::StashedPipeline;

mod assembly;
mod captured_conv;
mod contracts;
mod metadata;

pub(crate) use assembly::append_subgraphs_from_ir;
pub(crate) use contracts::{
    apply_boundary_contracts, apply_pre_submodule_qparam_propagation_ir,
    snapshot_pre_pass_qparams_ir,
};
pub(crate) use metadata::{IrInsertCarriedOutputRescalePass, IrStripWhileBoundaryQPass};

#[derive(Clone, Copy)]
pub(crate) struct WhileBodyInputCounts {
    carried: usize,
    additional: usize,
}

fn qualify_region(parent_region: Option<&str>, local_name: &str) -> String {
    match parent_region {
        Some(prefix) if !local_name.contains("__") => format!("{prefix}__{local_name}"),
        _ => local_name.to_string(),
    }
}

pub(crate) fn region_name_from_ir_arg(
    ir: &fx_ir::IrGraph,
    arg: Option<&fx_ir::IrArg>,
    parent_region: Option<&str>,
) -> Option<String> {
    match arg {
        Some(fx_ir::IrArg::Str(name)) => Some(qualify_region(parent_region, name)),
        Some(fx_ir::IrArg::Node(id)) => ir
            .nodes
            .iter()
            .find(|node| node.id == *id)
            .map(|node| qualify_region(parent_region, &node.name)),
        _ => None,
    }
}

fn sequence_arg_len_ir(arg: Option<&fx_ir::IrArg>) -> usize {
    match arg {
        Some(fx_ir::IrArg::List(items)) => items.len(),
        Some(fx_ir::IrArg::None) | None => 0,
        Some(_) => 1,
    }
}

pub(crate) fn collect_region_metadata(
    parent_ir: &fx_ir::IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
) -> (BTreeSet<String>, BTreeMap<String, WhileBodyInputCounts>) {
    let mut while_cond_graph_names = BTreeSet::new();
    let mut while_body_input_counts = BTreeMap::new();

    for (parent_region, ir) in std::iter::once((None, parent_ir)).chain(
        submodules
            .iter()
            .map(|(name, (ir, _))| (Some(name.as_str()), ir)),
    ) {
        for node_id in ir.live_node_ids() {
            let node = ir.node(node_id);
            if !node.is_while_loop() {
                continue;
            }

            if let Some(cond_name) = region_name_from_ir_arg(ir, node.args.first(), parent_region) {
                while_cond_graph_names.insert(cond_name);
            }
            if let Some(body_name) = region_name_from_ir_arg(ir, node.args.get(1), parent_region) {
                while_body_input_counts.insert(
                    body_name,
                    WhileBodyInputCounts {
                        carried: sequence_arg_len_ir(node.args.get(2)),
                        additional: sequence_arg_len_ir(node.args.get(3)),
                    },
                );
            }
        }
    }

    (while_cond_graph_names, while_body_input_counts)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::fx_ir::{IrArg, IrGraph, NodeMeta, NodeOp};
    use crate::lowering_effects::LoweringEffectBatch;

    use super::*;

    fn add_while(
        graph: &mut IrGraph,
        name: &str,
        cond: &str,
        body: &str,
        carried: Vec<IrArg>,
        additional: Vec<IrArg>,
    ) {
        graph.add_node(
            NodeOp::CallFunction,
            name.to_string(),
            "while_loop".to_string(),
            vec![
                IrArg::Str(cond.to_string()),
                IrArg::Str(body.to_string()),
                IrArg::List(carried),
                IrArg::List(additional),
            ],
            NodeMeta::default(),
        );
    }

    #[test]
    fn nested_regions_join_with_double_underscore_like_stock() {
        assert_eq!(
            qualify_region(Some("outer"), "inner"),
            "outer__inner".to_string()
        );
        assert_eq!(qualify_region(None, "inner"), "inner".to_string());
        assert_eq!(
            qualify_region(Some("outer"), "outer__inner"),
            "outer__inner".to_string()
        );
    }

    #[test]
    fn nested_while_region_metadata_is_qualified_and_sorted() {
        let mut parent = IrGraph::empty();
        add_while(
            &mut parent,
            "outer_while",
            "z_parent_cond",
            "outer_body",
            vec![IrArg::Int(1)],
            vec![IrArg::Int(2), IrArg::Int(3)],
        );

        let mut inner = IrGraph::empty();
        add_while(
            &mut inner,
            "inner_while",
            "z_inner_cond",
            "a_inner_body",
            vec![IrArg::Int(4), IrArg::Int(5)],
            vec![IrArg::Int(6)],
        );

        let mut submodules = BTreeMap::new();
        submodules.insert(
            "outer_body".to_string(),
            (inner, LoweringEffectBatch::default()),
        );

        let (cond_regions, body_counts) = collect_region_metadata(&parent, &submodules);

        assert_eq!(
            cond_regions.into_iter().collect::<Vec<_>>(),
            vec![
                "outer_body__z_inner_cond".to_string(),
                "z_parent_cond".to_string(),
            ]
        );
        assert_eq!(
            body_counts.keys().cloned().collect::<Vec<_>>(),
            vec![
                "outer_body".to_string(),
                "outer_body__a_inner_body".to_string(),
            ]
        );
        let outer = body_counts.get("outer_body").unwrap();
        assert_eq!((outer.carried, outer.additional), (1, 2));
        let inner = body_counts.get("outer_body__a_inner_body").unwrap();
        assert_eq!((inner.carried, inner.additional), (2, 1));
    }

    #[test]
    fn region_boundaries_keep_logical_shapes_through_layout_transposes() {
        use crate::test_ir::{call, placeholder, qparam};

        for restore_logical_layout in [false, true] {
            let mut parent = IrGraph::empty();
            let x = placeholder(&mut parent, "x", [2, 3, 4, 5], "torch.int8");
            let qp = qparam(0.25, 3, "torch.int8");
            parent.node_mut(x).meta.set_both_qparams(qp.clone());
            parent.node_mut(x).meta.set_tosa_dim_order(vec![0, 1, 2, 3]);
            let mut input = call(
                &mut parent,
                "x_input_layout",
                "backend.tosa.TRANSPOSE.default",
                vec![IrArg::Node(x), crate::fx_ir::i64_list_arg(&[0, 2, 3, 1])],
                [2, 4, 5, 3],
                "torch.int8",
            );
            parent.node_mut(input).meta.set_both_qparams(qp.clone());
            parent
                .node_mut(input)
                .meta
                .set_tosa_dim_order(vec![0, 2, 3, 1]);
            let expected_order = if restore_logical_layout {
                input = call(
                    &mut parent,
                    "restore_layout",
                    "backend.tosa.TRANSPOSE.default",
                    vec![
                        IrArg::Node(input),
                        crate::fx_ir::i64_list_arg(&[0, 3, 1, 2]),
                    ],
                    [2, 3, 4, 5],
                    "torch.int8",
                );
                parent.node_mut(input).meta.set_both_qparams(qp.clone());
                parent
                    .node_mut(input)
                    .meta
                    .set_tosa_dim_order(vec![0, 1, 2, 3]);
                vec![0, 1, 2, 3]
            } else {
                vec![0, 2, 3, 1]
            };
            add_while(
                &mut parent,
                "loop",
                "cond",
                "body",
                vec![IrArg::Node(input)],
                Vec::new(),
            );
            let parent = (parent, LoweringEffectBatch::default());
            let mut region = IrGraph::empty();
            let ph = placeholder(&mut region, "arg", [2, 3, 4, 5], "torch.float32");
            assert!(
                apply_pre_submodule_qparam_propagation_ir(
                    Some("cond"),
                    &mut region,
                    Some(&parent),
                    &BTreeMap::new(),
                ) > 0
            );
            assert_eq!(region.node_shape(ph), Some([2, 3, 4, 5].as_slice()));
            assert_eq!(region.node(ph).meta.val_dtype(), Some("torch.int8"));
            assert_eq!(
                region.node(ph).meta.tosa_dim_order(),
                Some(expected_order.as_slice())
            );
            assert!(region
                .node(ph)
                .meta
                .output_qparam()
                .unwrap()
                .same_exact_affine_dtype(&qp));
        }
    }

    fn active_source_files(root: &Path) -> Vec<PathBuf> {
        fn visit(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    visit(&path, out);
                    continue;
                }
                let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
                    continue;
                };
                if matches!(ext, "rs" | "py") {
                    out.push(path);
                }
            }
        }

        let mut out = Vec::new();
        visit(&root.join("src"), &mut out);
        visit(&root.join("python/retab"), &mut out);
        out
    }

    #[test]
    fn active_source_has_no_legacy_python_boundary_symbols() {
        // `CARGO_MANIFEST_DIR` is only set under Cargo; under another build
        // system the crate source tree isn't available in the test sandbox, so
        // skip this scan.
        let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") else {
            return;
        };
        let root = Path::new(manifest_dir);
        let forbidden = [
            concat!("flush_", "ir"),
            concat!("write_", "back"),
            concat!("ingest_", "fx_graph_hierarchy"),
            concat!("collect_", "control_flow_subgraphs"),
            concat!("re-", "ingest"),
        ];

        let mut violations = Vec::new();
        for path in active_source_files(root) {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for needle in forbidden {
                if text.contains(needle) {
                    violations.push(format!("{} contains {needle}", path.display()));
                }
            }
        }

        assert!(violations.is_empty(), "{}", violations.join("\n"));
    }
}
