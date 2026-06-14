use std::collections::{BTreeMap, BTreeSet};

use crate::fx_ir;
use crate::StashedPipeline;

mod assembly;
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
        Some(prefix) if !local_name.contains("::") => format!("{prefix}::{local_name}"),
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
        for node in ir.nodes.iter().filter(|node| node.alive) {
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
