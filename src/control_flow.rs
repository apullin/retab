use std::collections::{BTreeMap, BTreeSet};

use crate::fx_ir;
use crate::StashedPipeline;

mod assembly;
mod captured_conv;
mod contracts;
mod metadata;

pub(crate) use assembly::append_subgraphs_from_ir;
// Explicit-mode serializer tests convert regions without linking Python (APU-135 S8).
#[cfg(test)]
pub(crate) use assembly::try_append_subgraphs_from_ir;
pub(crate) use contracts::{
    apply_boundary_contracts, apply_pre_submodule_qparam_propagation_ir,
    snapshot_pre_pass_qparams_ir, IrRecordControlFlowRescaleTargetsPass,
};
pub(crate) use metadata::{IrInsertCarriedOutputRescalePass, IrStripWhileBoundaryQPass};
// IrRewriteConv's tests pin how the boundary contracts read its nodes (APU-135 S3).
#[cfg(test)]
pub(crate) use metadata::extract_effective_qparam_ir;

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

/// Control-flow hierarchies for the contract tests below and the explicit-mode
/// serializer tests (APU-135 S8).
#[cfg(test)]
pub(crate) mod fixtures {
    use std::collections::BTreeMap;

    use crate::fx_ir::{IrArg, IrGraph};
    use crate::lowering_effects::LoweringEffectBatch;
    use crate::test_ir::{call, output, placeholder, qparam};
    use crate::StashedPipeline;

    /// The val of every tensor in [`rank4_hierarchy`].
    pub(crate) const RANK4: [i64; 4] = [2, 3, 2, 2];

    /// rc4's test_cond.py and test_while.py cases with rank-4 inputs, reduced to the
    /// boundary: `x` (int8, quantized) feeds a cond or a while loop whose rank-4 output,
    /// `getitem`, is the parent's output. Each region returns its input, `arg0_1`.
    pub(crate) fn rank4_hierarchy(
        while_loop: bool,
    ) -> (IrGraph, BTreeMap<String, StashedPipeline>) {
        let qp = qparam(0.25, 3, "torch.int8");
        let mut parent = IrGraph::empty();
        let x = placeholder(&mut parent, "x", RANK4, "torch.int8");
        parent.node_mut(x).meta.set_both_qparams(qp.clone());
        let (target, args, regions) = if while_loop {
            (
                "while_loop",
                vec![
                    IrArg::Str("cond_fn".into()),
                    IrArg::Str("body_fn".into()),
                    IrArg::List(vec![IrArg::Node(x)]),
                    IrArg::List(Vec::new()),
                ],
                ["cond_fn", "body_fn"],
            )
        } else {
            let pred = placeholder(&mut parent, "pred", [1], "torch.bool");
            (
                "torch.ops.higher_order.cond",
                vec![
                    IrArg::Node(pred),
                    IrArg::Str("true_graph_0".into()),
                    IrArg::Str("false_graph_0".into()),
                    IrArg::List(vec![IrArg::Node(x)]),
                ],
                ["true_graph_0", "false_graph_0"],
            )
        };
        let cf = call(&mut parent, "cf", target, args, RANK4, "torch.int8");
        assert!(parent.node(cf).is_control_flow(), "fixture: {target}");
        let getitem = call(
            &mut parent,
            "getitem",
            "getitem",
            vec![IrArg::Node(cf), IrArg::Int(0)],
            RANK4,
            "torch.int8",
        );
        parent.node_mut(getitem).meta.set_both_qparams(qp.clone());
        output(&mut parent, getitem);

        let submodules = regions
            .into_iter()
            .map(|name| {
                let mut region = IrGraph::empty();
                let arg = placeholder(&mut region, "arg0_1", RANK4, "torch.int8");
                region.node_mut(arg).meta.set_both_qparams(qp.clone());
                output(&mut region, arg);
                (name.to_string(), (region, LoweringEffectBatch::default()))
            })
            .collect();
        (parent, submodules)
    }
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

    /// A parent that runs a while loop over a layout TRANSPOSE of `x` (rank 4, int8,
    /// quantized at the returned qparam), laid out as RETAB's earlier dim-order model
    /// laid it out: the TRANSPOSE to NHWC, then, if `restore_logical_layout`, one back
    /// to NCHW. Returns the parent, the qparam and the val of the loop's operand.
    fn while_over_layout_transposes(
        restore_logical_layout: bool,
    ) -> (IrGraph, crate::fx_ir::QParam, Vec<i64>) {
        use crate::test_ir::{call, placeholder, qparam};

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
        if restore_logical_layout {
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
        }
        let operand_shape = parent.node_shape(input).unwrap().to_vec();
        add_while(
            &mut parent,
            "loop",
            "cond",
            "body",
            vec![IrArg::Node(input)],
            Vec::new(),
        );
        (parent, qp, operand_shape)
    }

    /// APU-135 P2 design §4.3.1: a region placeholder takes the val of the operand
    /// bound to it, as rc4's `_regularize_submodule` does, and the boundary writes no
    /// dim order, before the region's pipeline and at serialization alike. The parent
    /// is the dim-order fixture above, dim orders and TOSA TRANSPOSEs included, which
    /// RETAB's graphs no longer have: the boundary neither reconstructs a logical shape
    /// through a TRANSPOSE nor copies an operand's dim order.
    #[test]
    fn region_boundaries_take_operand_vals_without_dim_orders() {
        use crate::test_ir::{output, placeholder};

        let region_with_input = || {
            let mut region = IrGraph::empty();
            let ph = placeholder(&mut region, "arg", [2, 3, 4, 5], "torch.float32");
            output(&mut region, ph);
            (region, ph)
        };
        for restore_logical_layout in [false, true] {
            let (parent, qp, operand_shape) = while_over_layout_transposes(restore_logical_layout);
            let check = |region: &IrGraph, ph, stage: &str| {
                let meta = &region.node(ph).meta;
                assert_eq!(meta.val_shape(), Some(operand_shape.as_slice()), "{stage}");
                assert_eq!(meta.val_dtype(), Some("torch.int8"), "{stage}");
                assert_eq!(meta.tosa_dim_order(), None, "{stage}");
                assert_eq!(meta.val_dim_order(), None, "{stage}");
                assert!(
                    meta.output_qparam().unwrap().same_exact_affine_dtype(&qp),
                    "{stage}"
                );
            };

            // Before the region's pipeline runs.
            let stashed_parent = (parent, LoweringEffectBatch::default());
            let (mut region, ph) = region_with_input();
            assert!(
                apply_pre_submodule_qparam_propagation_ir(
                    Some("cond"),
                    &mut region,
                    Some(&stashed_parent),
                    &BTreeMap::new(),
                ) > 0
            );
            check(&region, ph, "pre-pass propagation");

            // At serialization.
            let (mut parent, _, _) = while_over_layout_transposes(restore_logical_layout);
            let mut submodules: BTreeMap<String, StashedPipeline> = ["cond", "body"]
                .into_iter()
                .map(|name| {
                    let (region, _) = region_with_input();
                    (name.to_string(), (region, LoweringEffectBatch::default()))
                })
                .collect();
            assert!(apply_boundary_contracts(&mut parent, &mut submodules) > 0);
            for (name, (region, _)) in &submodules {
                let ph = region.live_placeholder_ids()[0];
                check(region, ph, &format!("serialization, {name}"));
            }
        }
    }

    /// A parent's rank-4 cond or while output keeps its layout in its val, as rc4's
    /// does (APU-135 P2 design §4.3.1): no node in the hierarchy gets a dim order, which
    /// serialization would send to stock, and the rest of the contract still applies.
    #[test]
    fn rank4_control_flow_outputs_keep_their_vals_without_dim_orders() {
        use crate::test_ir::live_node;
        use fixtures::RANK4;

        for while_loop in [false, true] {
            let case = format!("while_loop={while_loop}");
            let (mut parent, mut submodules) = fixtures::rank4_hierarchy(while_loop);
            apply_boundary_contracts(&mut parent, &mut submodules);

            let getitem = &live_node(&parent, "getitem").meta;
            assert_eq!(getitem.val_shape(), Some(RANK4.as_slice()), "{case}");
            assert_eq!(
                getitem.extra_bool("arm_control_flow_parent_output_user"),
                None,
                "{case}"
            );
            let graphs = std::iter::once(("main", &parent)).chain(
                submodules
                    .iter()
                    .map(|(name, (region, _))| (name.as_str(), region)),
            );
            for (graph_name, graph) in graphs {
                for node in graph.live_nodes() {
                    assert_eq!(
                        node.meta.tosa_dim_order(),
                        None,
                        "{case}: {graph_name} {}",
                        node.name
                    );
                }
            }
            for (name, (region, _)) in &submodules {
                let arg = &live_node(region, "arg0_1").meta;
                assert_eq!(
                    arg.extra_bool("arm_control_flow_placeholder"),
                    Some(true),
                    "{case}: {name}"
                );
                assert_eq!(arg.val_shape(), Some(RANK4.as_slice()), "{case}");
            }
        }
    }

    /// Critique A2: why RETAB keeps IrNormalizeDelegateIOLayout out of the region
    /// pipelines (the `fx_pipeline` tests check that it does), though rc4 runs
    /// its pass on every region. rc4's does nothing there, because
    /// `_regularize_submodule` gives region IO the parent's post-pass vals, which the
    /// parent's normalization made contiguous. RETAB imports a region from the
    /// original edge submodule, and the boundary gives a placeholder the parent
    /// operand's val but not its dim order. So a channels-last export's branch input
    /// keeps the export's dim order (torch.export gives its val `(0, 2, 3, 1)`), and
    /// the pass would permute it where rc4's leaves it.
    #[test]
    fn region_inputs_keep_the_exports_dim_order_so_regions_skip_io_normalization() {
        use crate::fx_ir::IrPass;
        use crate::layout::normalize_delegate_io::IrNormalizeDelegateIOLayoutPass;
        use fixtures::RANK4;

        const CHANNELS_LAST: [usize; 4] = [0, 2, 3, 1];
        let (parent, mut submodules) = fixtures::rank4_hierarchy(false);
        let (mut region, _) = submodules.remove("true_graph_0").expect("a branch");
        let arg = region.live_placeholder_ids()[0];
        region
            .node_mut(arg)
            .meta
            .set_val_dim_order(CHANNELS_LAST.to_vec());

        let stashed_parent = (parent, LoweringEffectBatch::default());
        assert!(
            apply_pre_submodule_qparam_propagation_ir(
                Some("true_graph_0"),
                &mut region,
                Some(&stashed_parent),
                &BTreeMap::new(),
            ) > 0
        );
        assert_eq!(region.node_shape(arg), Some(RANK4.as_slice()));
        assert_eq!(
            region.node(arg).meta.val_dim_order(),
            Some(CHANNELS_LAST.as_slice())
        );

        assert!(IrNormalizeDelegateIOLayoutPass.run(&mut region));
        let arg = region.live_placeholder_ids()[0];
        assert_eq!(
            region.node_shape(arg),
            Some([2, 2, 2, 3].as_slice()),
            "the branch input would take the physical shape the parent does not pass"
        );
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
