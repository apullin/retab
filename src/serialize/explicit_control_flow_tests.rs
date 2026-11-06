//! Serializer tests for explicit layout mode's control flow (APU-135 P2 design S8). Each
//! test takes a hand-built parent and its regions through the boundary contracts,
//! ingest and serialization, as `ingest_and_serialize` does.

use std::collections::BTreeMap;

use super::collection::{check_explicit_layout_invariants, collect_regions, SerializedRegion};
use super::tosa_types::{Op, TosaTensor};
use super::visitors;
use crate::fx_ir::{IrArg, IrGraph};
use crate::lowering_effects::LoweringEffectBatch;
use crate::pipeline_config::LayoutMode;
use crate::store::TensorStore;
use crate::test_ir::{call, output, placeholder, qparam};
use crate::StashedPipeline;

const INT8: &str = "torch.int8";
const SHAPE: [i64; 4] = [2, 3, 2, 2];

/// rc4's test_cond.py and test_while.py cases with rank-4 inputs, reduced to the
/// boundary: `x` (int8, quantized) feeds a cond or a while loop whose rank-4 output,
/// `getitem`, is the graph's output. Each region returns its input.
fn rank4_hierarchy(while_loop: bool) -> (IrGraph, BTreeMap<String, StashedPipeline>) {
    let qp = qparam(0.25, 3, INT8);
    let mut parent = IrGraph::empty();
    let x = placeholder(&mut parent, "x", SHAPE, INT8);
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
    let cf = call(&mut parent, "cf", target, args, SHAPE, INT8);
    let getitem = call(
        &mut parent,
        "getitem",
        "getitem",
        vec![IrArg::Node(cf), IrArg::Int(0)],
        SHAPE,
        INT8,
    );
    parent.node_mut(getitem).meta.set_both_qparams(qp.clone());
    output(&mut parent, getitem);

    let submodules = regions
        .into_iter()
        .map(|name| {
            let mut region = IrGraph::empty();
            let arg = placeholder(&mut region, "arg0_1", SHAPE, INT8);
            region.node_mut(arg).meta.set_both_qparams(qp.clone());
            output(&mut region, arg);
            (name.to_string(), (region, LoweringEffectBatch::default()))
        })
        .collect();
    (parent, submodules)
}

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
    let logical: Vec<i32> = SHAPE.iter().map(|&dim| dim as i32).collect();
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
