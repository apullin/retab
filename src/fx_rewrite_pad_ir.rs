//! Pure Rust IrPass version of RewritePadPass.
//!
//! TOSA `PAD` is constant-only, so reflect/replicate/circular `aten.pad.default`
//! must be decomposed into `slice_copy` + `cat`. For each padded dim (innermost
//! outward) we gather `left` and `right` width-1 slices at mode-specific indices
//! and concatenate them around the original tensor:
//!   replicate: left=slice@0,           right=slice@(size-1)
//!   circular:  left=slice@(size-left+i),right=slice@i
//!   reflect:   left=slice@(left-i),     right=slice@(size-2-i)
//! Negative padding crops (upstream PRs #24108 / #24084): constant mode crops with a
//! `slice_copy` and keeps the non-negative remainder on the PAD node, which the
//! serializer emits as TOSA PAD; reflect/replicate pad from the original borders and
//! then crop; circular crops and then wraps the cropped input.

use crate::fx_ir::{
    i64_list_arg, IrArg, IrGraph, IrPass, IrRewriteCursor, NodeId, TensorMetaStamp,
};
use crate::fx_target::is_pad_target;
use crate::runner_logging::log_line;

#[allow(clippy::too_many_arguments)]
fn add_slice(
    cursor: &mut IrRewriteCursor<'_>,
    meta: &TensorMetaStamp<'_>,
    name: String,
    x_id: NodeId,
    x_shape: &[i64],
    dim: usize,
    idx: i64,
) -> NodeId {
    let mut out_shape = x_shape.to_vec();
    out_shape[dim] = 1;
    // slice_copy.Tensor(x, dim, idx, idx+1) — width-1 slice keeping the dim.
    cursor.slice_copy(
        name,
        x_id,
        dim as i64,
        idx,
        idx + 1,
        None,
        meta.preserve_qparams(out_shape),
    )
}

/// Pad `x` along `dim` by (left, right) per `mode`; returns (new node, new shape).
#[allow(clippy::too_many_arguments)]
fn pad_along_dim(
    cursor: &mut IrRewriteCursor<'_>,
    meta: &TensorMetaStamp<'_>,
    name_prefix: &str,
    counter: &mut usize,
    x_id: NodeId,
    x_shape: &[i64],
    dim: usize,
    left: i64,
    right: i64,
    mode: &str,
) -> (NodeId, Vec<i64>) {
    if left == 0 && right == 0 {
        return (x_id, x_shape.to_vec());
    }
    let size = x_shape[dim];

    let mut members: Vec<IrArg> = Vec::new();
    for i in 0..left {
        let idx = match mode {
            "replicate" => 0,
            "circular" => size - left + i,
            _ => left - i, // reflect (mode validated by caller)
        };
        let s = add_slice(
            cursor,
            meta,
            format!("{}_slice{}", name_prefix, *counter),
            x_id,
            x_shape,
            dim,
            idx,
        );
        *counter += 1;
        members.push(IrArg::Node(s));
    }
    members.push(IrArg::Node(x_id));
    for i in 0..right {
        let idx = match mode {
            "replicate" => size - 1,
            "circular" => i,
            _ => size - 2 - i, // reflect (mode validated by caller)
        };
        let s = add_slice(
            cursor,
            meta,
            format!("{}_slice{}", name_prefix, *counter),
            x_id,
            x_shape,
            dim,
            idx,
        );
        *counter += 1;
        members.push(IrArg::Node(s));
    }

    let mut out_shape = x_shape.to_vec();
    out_shape[dim] = size + left + right;
    let cat = cursor.call_function(
        format!("{}_cat{}", name_prefix, *counter),
        "aten.cat.default",
        vec![IrArg::List(members), IrArg::Int(dim as i64)],
        meta.preserve_qparams(out_shape.clone()),
    );
    *counter += 1;
    (cat, out_shape)
}

/// Crop `x` along `dim` by the negative parts of (left, right); returns (new node,
/// new shape). Negative padding removes elements, which TOSA PAD cannot express.
#[allow(clippy::too_many_arguments)]
fn crop_along_dim(
    cursor: &mut IrRewriteCursor<'_>,
    meta: &TensorMetaStamp<'_>,
    name_prefix: &str,
    counter: &mut usize,
    x_id: NodeId,
    x_shape: &[i64],
    dim: usize,
    left: i64,
    right: i64,
) -> (NodeId, Vec<i64>) {
    if left >= 0 && right >= 0 {
        return (x_id, x_shape.to_vec());
    }
    let start = (-left).max(0);
    let end = x_shape[dim] + right.min(0);
    let mut out_shape = x_shape.to_vec();
    out_shape[dim] = end - start;
    let cropped = cursor.slice_copy(
        format!("{}_crop{}", name_prefix, *counter),
        x_id,
        dim as i64,
        start,
        end,
        None,
        meta.preserve_qparams(out_shape.clone()),
    );
    *counter += 1;
    (cropped, out_shape)
}

/// False if (left, right) cannot be lowered along a dim of `size`. Checked in the
/// same order the rewrite applies padding and cropping for each mode.
fn pad_pair_supported(mode: &str, size: i64, left: i64, right: i64) -> bool {
    let (pad_left, pad_right) = (left.max(0), right.max(0));
    let (crop_left, crop_right) = ((-left).max(0), (-right).max(0));
    match mode {
        // Pad from the original borders first, then crop the padded tensor.
        "reflect" | "replicate" => {
            let reflect_ok = mode != "reflect" || (pad_left < size && pad_right < size);
            reflect_ok && size + pad_left + pad_right - crop_left - crop_right > 0
        }
        // Crop first; circular then wraps the cropped input.
        _ => {
            let cropped = size - crop_left - crop_right;
            cropped > 0 && (mode != "circular" || (pad_left <= cropped && pad_right <= cropped))
        }
    }
}

fn rewrite_one(graph: &mut IrGraph, id: NodeId) -> bool {
    let node = &graph.nodes[id];
    if !is_pad_target(&node.target) || node.args.len() < 2 {
        return false;
    }
    // constant_pad_nd has a numeric value at args[2] and always uses constant mode.
    let mode = node
        .arg_or_kwarg_str(2, "mode")
        .unwrap_or("constant")
        .to_string();
    if !matches!(
        mode.as_str(),
        "constant" | "reflect" | "replicate" | "circular"
    ) {
        return false;
    }
    let input_id = match node.args[0].as_node() {
        Some(n) => n,
        None => return false,
    };
    let pad: Vec<i64> = match node.args[1].as_int_list() {
        Some(p) if p.len() % 2 == 0 => p,
        _ => return false,
    };
    // A non-negative constant pad is serialized directly as TOSA PAD.
    if mode == "constant" && pad.iter().all(|&p| p >= 0) {
        return false;
    }
    let node_name = node.name.clone();
    let orig_meta = node.meta.clone();
    if orig_meta.val_dtype().is_none() {
        return false;
    }
    let meta = orig_meta.tensor_stamp_same_dtype_or_float32();
    let input_shape = match graph.logical_input_shape(id, input_id) {
        Some(s) => s,
        None => return false,
    };
    let rank = input_shape.len();
    if pad.len() / 2 > rank {
        return false;
    }
    // Validate up front so we never emit a partial chain.
    for (pair_idx, chunk) in pad.chunks(2).enumerate() {
        let size = input_shape[rank - 1 - pair_idx];
        if !pad_pair_supported(&mode, size, chunk[0], chunk[1]) {
            return false;
        }
    }

    let mut cur_id = input_id;
    let mut cur_shape = input_shape;
    let mut counter: usize = 0;
    let mut cursor = graph.rewrite_cursor(id);
    // F.pad pad pairs run innermost dim outward.
    for (pair_idx, chunk) in pad.chunks(2).enumerate() {
        let (left, right) = (chunk[0], chunk[1]);
        let dim = rank - 1 - pair_idx;
        let pad_dim = |cursor: &mut IrRewriteCursor<'_>,
                       counter: &mut usize,
                       x_id: NodeId,
                       x_shape: &[i64]| {
            pad_along_dim(
                cursor,
                &meta,
                &node_name,
                counter,
                x_id,
                x_shape,
                dim,
                left.max(0),
                right.max(0),
                &mode,
            )
        };
        // Reflect and replicate pad from the original borders, then crop. Circular
        // wraps the cropped input. Constant crops here and leaves the non-negative
        // remainder on the PAD node.
        if matches!(mode.as_str(), "reflect" | "replicate") {
            (cur_id, cur_shape) = pad_dim(&mut cursor, &mut counter, cur_id, &cur_shape);
        }
        (cur_id, cur_shape) = crop_along_dim(
            &mut cursor,
            &meta,
            &node_name,
            &mut counter,
            cur_id,
            &cur_shape,
            dim,
            left,
            right,
        );
        if mode == "circular" {
            (cur_id, cur_shape) = pad_dim(&mut cursor, &mut counter, cur_id, &cur_shape);
        }
    }

    if mode == "constant" {
        let mut args = graph.nodes[id].args.clone();
        args[0] = IrArg::Node(cur_id);
        args[1] = i64_list_arg(&pad.iter().map(|&p| p.max(0)).collect::<Vec<_>>());
        graph.set_args(id, args);
        return true;
    }
    if cur_id == input_id {
        return false; // nothing emitted (all-zero pad)
    }
    cursor.replace_anchor_with(cur_id);
    true
}

pub struct IrRewritePadPass;

impl IrPass for IrRewritePadPass {
    fn name(&self) -> &str {
        "IrRewritePad"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(rewrite_one);
        if modified {
            log_line(format_args!(
                "[IrRewritePad] rewrote padding with crops and slice+cat"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::i64_list_arg;
    use crate::fx_target::{is_cat_target, is_slice_copy_tensor_target};
    use crate::test_ir::{call, live_node as named, output, placeholder};

    #[test]
    fn fully_negative_constant_pad_lowers_to_one_slice() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [1, 2, 6, 8], "torch.int8");
        let pad = call(
            &mut graph,
            "pad",
            "aten.constant_pad_nd.default",
            vec![
                IrArg::Node(x),
                i64_list_arg(&[-1, -2, -1, 0]),
                IrArg::Int(5),
            ],
            [1, 2, 5, 5],
            "torch.int8",
        );
        let qp = crate::test_ir::qparam(0.25, 3, "torch.int8");
        graph.node_mut(pad).meta.set_both_qparams(qp.clone());
        output(&mut graph, pad);
        crate::fx_memory_format_ir::IrToTosaMemoryFormatPass.run(&mut graph);
        assert!(IrRewritePadPass.run(&mut graph));
        assert_eq!(graph.node(pad).arg_int_list(1), Some(vec![0, 0, 0, 0]));
        assert!(crate::fx_rewrite_slice_ir::IrRewriteSlicePass.run(&mut graph));
        assert!(crate::fx_fuse_consecutive_slices_ir::IrFuseConsecutiveSlicesPass.run(&mut graph));
        assert!(crate::fx_remove_noop_ir::IrRemoveNoopPass.run(&mut graph));
        assert!(!graph.node(pad).alive);
        let slices: Vec<_> = graph
            .live_nodes()
            .filter(|node| node.target == "backend.tosa.SLICE.default")
            .collect();
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].arg_int_list(1), Some(vec![0, 0, 1, 1]));
        assert_eq!(slices[0].arg_int_list(2), Some(vec![1, 2, 5, 5]));
        assert_eq!(slices[0].meta.input_qparam(), Some(&qp));
        assert_eq!(slices[0].meta.output_qparam(), Some(&qp));
    }

    #[test]
    fn constant_negative_pad_crops_then_keeps_clamped_pad() {
        // x [1,4,16], pad (-2, 3): crop 2 from the start, then PAD (0, 3).
        let (mut graph, pad) = pad_graph(&[1, 4, 16], &[-2, 3], "constant");
        assert!(IrRewritePadPass.run(&mut graph));
        assert!(graph.nodes[pad].alive);
        let crop = graph.nodes[pad].arg_node(0).unwrap();
        assert!(is_slice_copy_tensor_target(&graph.nodes[crop].target));
        assert_eq!(graph.node_shape_vec(crop), Some(vec![1, 4, 14]));
        assert_eq!(graph.nodes[crop].arg_int(2), Some(2));
        assert_eq!(graph.nodes[crop].arg_int(3), Some(16));
        assert_eq!(graph.nodes[pad].arg_int_list(1), Some(vec![0, 3]));
    }

    #[test]
    fn constant_pad_nd_negative_pad_preserves_fill_and_qparams() {
        for value in [None, Some(IrArg::Int(0)), Some(IrArg::Int(5))] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [1, 1, 4, 4], "torch.int8");
            let qp = crate::test_ir::qparam(0.25, 3, "torch.int8");
            let mut args = vec![IrArg::Node(x), i64_list_arg(&[-1, 1])];
            args.extend(value.clone());
            let pad = call(
                &mut graph,
                "pad",
                "aten.constant_pad_nd.default",
                args,
                [1, 1, 4, 4],
                "torch.int8",
            );
            graph.node_mut(pad).meta.set_both_qparams(qp.clone());
            output(&mut graph, pad);

            assert!(IrRewritePadPass.run(&mut graph));
            let crop = graph.node(pad).arg_node(0).unwrap();
            assert!(is_slice_copy_tensor_target(&graph.node(crop).target));
            assert_eq!(graph.node(crop).arg_int(1), Some(3));
            assert_eq!(graph.node(crop).arg_int(2), Some(1));
            assert_eq!(graph.node(crop).arg_int(3), Some(4));
            assert_eq!(graph.node_shape_vec(crop), Some(vec![1, 1, 4, 3]));
            assert_eq!(graph.node(pad).arg_int_list(1), Some(vec![0, 1]));
            assert_eq!(
                graph.node(pad).args.get(2).and_then(IrArg::as_int),
                value.as_ref().and_then(IrArg::as_int)
            );
            assert!(graph
                .node(crop)
                .meta
                .output_qparam()
                .unwrap()
                .same_exact_affine_dtype(&qp));
            assert!(graph
                .node(pad)
                .meta
                .output_qparam()
                .unwrap()
                .same_exact_affine_dtype(&qp));
        }
    }

    #[test]
    fn non_negative_constant_pad_is_left_to_the_serializer() {
        let (mut graph, _) = pad_graph(&[1, 4, 16], &[1, 2], "constant");
        assert!(!IrRewritePadPass.run(&mut graph));
    }

    #[test]
    fn reflect_pads_from_original_borders_then_crops() {
        // x [1,3,6], reflect pad (3, -2): pad 3 on the left from the original
        // borders (cat of width 9), then crop 2 from the end -> width 7.
        let (mut graph, pad) = pad_graph(&[1, 3, 6], &[3, -2], "reflect");
        assert!(IrRewritePadPass.run(&mut graph));
        assert!(!graph.nodes[pad].alive);
        let out = graph.output_node().unwrap().first_node_arg().unwrap();
        assert!(is_slice_copy_tensor_target(&graph.nodes[out].target));
        assert_eq!(graph.node_shape_vec(out), Some(vec![1, 3, 7]));
        let padded = graph.nodes[out].arg_node(0).unwrap();
        assert!(is_cat_target(&graph.nodes[padded].target));
        assert_eq!(graph.node_shape_vec(padded), Some(vec![1, 3, 9]));
    }

    #[test]
    fn replicate_can_crop_more_than_the_original_width_after_padding() {
        // x [1,3,6], replicate pad (-6, 2): pad 2 on the right first (width 8), then
        // crop 6 from the start -> width 2.
        let (mut graph, _) = pad_graph(&[1, 3, 6], &[-6, 2], "replicate");
        assert!(IrRewritePadPass.run(&mut graph));
        let out = graph.output_node().unwrap().first_node_arg().unwrap();
        assert!(is_slice_copy_tensor_target(&graph.nodes[out].target));
        assert_eq!(graph.node_shape_vec(out), Some(vec![1, 3, 2]));
        assert_eq!(graph.nodes[out].arg_int(2), Some(6));
        assert_eq!(graph.nodes[out].arg_int(3), Some(8));
    }

    #[test]
    fn circular_crops_then_wraps_the_cropped_input() {
        // x [1,8], circular pad (-2, 1): crop 2 from the start (width 6), then wrap
        // one element taken from the cropped tensor's start -> width 7.
        let (mut graph, _) = pad_graph(&[1, 8], &[-2, 1], "circular");
        assert!(IrRewritePadPass.run(&mut graph));
        let out = graph.output_node().unwrap().first_node_arg().unwrap();
        assert!(is_cat_target(&graph.nodes[out].target));
        assert_eq!(graph.node_shape_vec(out), Some(vec![1, 7]));
        let IrArg::List(members) = &graph.nodes[out].args[0] else {
            panic!("cat members");
        };
        let cropped = members[0].as_node().unwrap();
        assert_eq!(graph.node_shape_vec(cropped), Some(vec![1, 6]));
        let wrap = members[1].as_node().unwrap();
        assert_eq!(graph.nodes[wrap].arg_node(0), Some(cropped));
    }

    fn pad_graph(in_shape: &[i64], pad: &[i64], mode: &str) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", in_shape, "torch.int8");
        // output shape: last dim grows by pad[0]+pad[1] (single-dim pad)
        let mut out = in_shape.to_vec();
        *out.last_mut().unwrap() += pad[0] + pad[1];
        let pad_args = vec![IrArg::Node(x), i64_list_arg(pad), IrArg::Str(mode.into())];
        let p = call(
            &mut graph,
            "pad",
            "aten.pad.default",
            pad_args,
            out.as_slice(),
            "torch.int8",
        );
        output(&mut graph, p);
        (graph, p)
    }

    #[test]
    fn reflect_pad_last_dim() {
        // x [1,4] reflect pad (2,2) on last dim -> [1,8].
        let (mut graph, pad) = pad_graph(&[1, 4], &[2, 2], "reflect");
        assert!(IrRewritePadPass.run(&mut graph));
        assert!(!graph.nodes[pad].alive);
        // 2 left + 2 right = 4 slice_copy nodes + 1 cat.
        let slices: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.alive && is_slice_copy_tensor_target(&n.target))
            .collect();
        assert_eq!(slices.len(), 4);
        // reflect left indices: left-i for i in 0..2 -> [2,1]; right: size-2-i -> [2,3].
        let cat = named(&graph, "pad_cat4");
        assert!(is_cat_target(&cat.target));
        assert_eq!(cat.meta.val_shape().unwrap(), &[1, 8]);
        // cat list has 5 members (2 left + x + 2 right)
        if let IrArg::List(items) = &cat.args[0] {
            assert_eq!(items.len(), 5);
        } else {
            panic!("cat arg0 not a list");
        }
        assert_eq!(named(&graph, "output").args[0].as_node(), Some(cat.id));
    }

    #[test]
    fn circular_pad_last_dim() {
        // x [1,4] circular pad (1,1) -> [1,6]; left=slice@(size-1)=3, right=slice@0.
        let (mut graph, _p) = pad_graph(&[1, 4], &[1, 1], "circular");
        assert!(IrRewritePadPass.run(&mut graph));
        let slices: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.alive && is_slice_copy_tensor_target(&n.target))
            .collect();
        assert_eq!(slices.len(), 2);
    }

    #[test]
    fn skips_constant_mode() {
        let (mut graph, _p) = pad_graph(&[1, 4], &[2, 2], "constant");
        assert!(!IrRewritePadPass.run(&mut graph));
    }
}
