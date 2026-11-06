//! Serializer tests for explicit layout mode's control flow (APU-135 P2 design S8). Each
//! test takes a hand-built parent and its regions through the boundary contracts,
//! ingest and serialization, as `ingest_and_serialize` does.

use std::collections::BTreeMap;

use super::collection::{check_explicit_layout_invariants, collect_regions, SerializedRegion};
use super::tosa_types::{Op, TosaTensor};
use super::visitors;
use crate::control_flow::fixtures::{rank4_hierarchy, RANK4};
use crate::fx_ir::IrGraph;
use crate::lowering_effects::LoweringEffectBatch;
use crate::pipeline_config::LayoutMode;
use crate::store::TensorStore;
use crate::StashedPipeline;

/// Serialize `parent` and its regions in `layout` as `ingest_and_serialize` does. In
/// explicit mode, also check D1 on every graph: no dim order, and every node serializes
/// at its val's shape.
fn serialize_hierarchy(
    (mut parent, mut submodules): (IrGraph, BTreeMap<String, StashedPipeline>),
    layout: LayoutMode,
) -> Result<Vec<SerializedRegion>, String> {
    crate::control_flow::apply_boundary_contracts(&mut parent, &mut submodules, layout);
    let mut store = TensorStore::new();
    let mut main = crate::ingest::from_ir::graph_from_ir(&parent, &mut store);
    main.layout = layout;
    crate::ingest::from_ir::apply_effects(&mut main, &mut store, &LoweringEffectBatch::default());
    crate::ingest::finalize::finalize_converted_graph(&mut main);
    let mut graphs = vec![("main".to_string(), main)];
    crate::control_flow::try_append_subgraphs_from_ir(
        &mut graphs,
        &mut store,
        &parent,
        &submodules,
        layout,
    )?;
    for (_, graph) in &mut graphs {
        graph.rebuild_user_lists();
    }
    visitors::shape_utils::clear_shape_cache();
    visitors::dtype_quant::clear_dtype_cache();
    if layout == LayoutMode::Explicit {
        for (name, graph) in &graphs {
            check_explicit_layout_invariants(name, graph, &store)?;
        }
    }
    collect_regions(&graphs, &store)
}

fn tensor<'a>(region: &'a SerializedRegion, name: &str) -> &'a TosaTensor {
    region
        .items
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("no tensor named {name} in {}", region.name))
}

fn count(region: &SerializedRegion, op: Op) -> usize {
    region.items.operators.iter().filter(|o| o.op == op).count()
}

/// S8 (design §4.3.1): a rank-4 cond or while output keeps its val in explicit mode, as
/// in rc4, where a region's IO takes the parent's vals. The boundary contracts used to
/// give it the NHWC dim order of dim-order mode's channels-last regions in either mode,
/// and explicit serialization sent every such graph to stock (`reject_dim_orders`).
/// Dim-order mode is unchanged: there the output is NHWC, and the main graph TRANSPOSEs
/// it back to the user's layout.
#[test]
fn rank4_control_flow_outputs_keep_their_vals_in_explicit_mode() {
    let logical: Vec<i32> = RANK4.iter().map(|&dim| dim as i32).collect();
    for while_loop in [false, true] {
        let op = if while_loop {
            Op::WhileLoop
        } else {
            Op::CondIf
        };

        let regions = serialize_hierarchy(rank4_hierarchy(while_loop), LayoutMode::Explicit)
            .unwrap_or_else(|error| panic!("{op:?}: {error}"));
        let main = &regions[0];
        assert_eq!(main.name, "main");
        assert_eq!(count(main, op), 1, "{op:?}");
        assert_eq!(tensor(main, "getitem").shape, logical, "{op:?}");
        assert_eq!(main.items.output_names, ["getitem"], "{op:?}");
        assert_eq!(regions.len(), 3, "{op:?}");
        for region in &regions {
            assert_eq!(count(region, Op::Transpose), 0, "{op:?} {}", region.name);
        }
        for region in &regions[1..] {
            let input = &region.items.input_names[0];
            assert_eq!(
                tensor(region, input).shape,
                logical,
                "{op:?} {}",
                region.name
            );
            assert_eq!(
                region.items.output_names,
                std::slice::from_ref(input),
                "{op:?}"
            );
        }

        let regions = serialize_hierarchy(rank4_hierarchy(while_loop), LayoutMode::DimOrder)
            .unwrap_or_else(|error| panic!("{op:?}: {error}"));
        let main = &regions[0];
        assert_eq!(tensor(main, "getitem").shape, [2, 2, 2, 3], "{op:?}");
        assert_eq!(count(main, Op::Transpose), 1, "{op:?}");
        assert_eq!(
            tensor(main, &main.items.output_names[0]).shape,
            logical,
            "{op:?}"
        );
    }
}
