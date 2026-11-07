//! Port of rc4's RewriteConvPass
//! (`backends/arm/_passes/rewrite_conv_pass.py`, "RCP" below with its line
//! numbers), APU-135 P2 S3 and S4 (APU-155).
//!
//! The pass replaces every `aten.convolution.default` with a TOSA convolution
//! in channels-last layout, with every layout change an explicit node (design
//! §2.1-2.7, §2.11; critique A6, A9, B5, B10, B12):
//!
//! ```text
//! before c: x' = permute_copy(x, PRE)              [conv1d: then view_copy(x', [N, 1, L, C])]
//! after c:  t  = backend.tosa.<OP>(x', W', b', ...) inherits c's qparams (A16W8: INT48)
//!           r  = backend.tosa.RESCALE(t, out dtype, scales, 0, out zp)   (quantized only)
//!           [conv1d: view_copy(r, [N, Lout, Co])]
//!           y  = permute_copy(r, POST)              the uses of c move here; c is erased
//! ```
//!
//! | kind (dispatch order, RCP:939-1141) | OP | PRE / POST | W' |
//! |---|---|---|---|
//! | transposed: 2-D, undilated, ungrouped | TRANSPOSE_CONV2D | NHWC / NCHW | IOHW permuted (1,2,3,0): `{c}_weight_ohwi` |
//! | spatial rank 1, depthwise | DEPTHWISE_CONV2D | (0,2,1) + view / view + (0,2,1) | permute (1,2,0), reshape (1,K,C,M): `_hwicm` |
//! | spatial rank 1 | CONV2D | as above | permute (0,2,1), reshape (O,1,K,I): `_ohwi` |
//! | rank 5 (groups must be 1) | CONV3D | NDHWC / NCDHW | permute (0,2,3,4,1): `_odhwi` |
//! | depthwise (rc4's rule, Cin = 1 included) | DEPTHWISE_CONV2D | NHWC / NCHW | permute (2,3,0,1), reshape (KH,KW,C,M): `_hwicm` |
//! | otherwise | CONV2D | NHWC / NCHW | permute (0,2,3,1): `_ohwi` |
//!
//! rc4's depthwise rule (`_is_depthwise_conv`, RCP:133-145) is a rank-3 or
//! rank-4 input whose channel count equals the groups and divides the output
//! channels, so a `Conv2d(1, Co, k)` (and `Linear(1, Co)`) is depthwise too.
//!
//! **Constants.** rc4 runs the pass after ComputeConstantOpsAOT and
//! FuseConstantArgs (`arm_pass_manager.py:699-700`), so a weight is always a
//! constant placeholder there; the pipeline runs `IrMaterializeConstantsPass`
//! at that slot (`layout::constants`), so a weight RETAB's planners can
//! evaluate is a constant node here too and the pass reads its tensor. The
//! pre-permuted weight is a new constant of the source's kind, inserted
//! directly after the source (RCP:368-376), carrying its special dtype (INT4,
//! RCP:377-378). A folded source is the `{node}_fused_const` placeholder
//! `IrMaterializeConstantsPass` put where FuseConstantArgs puts it, so the
//! order is rc4's `[fused_const, {c}_weight_*]` (critique A6). A conv without
//! a bias gets a zero bias, an int32 parameter inserted before the first
//! placeholder (RCP:160-195), so several end up in reverse conv order. The
//! sources stay, unused, as rc4 leaves them; ingest drops unused constants. A conv whose tensor inputs are all constants is
//! left alone (design §2.2): rc4 folds it at `:700`, and RETAB folds it when a
//! planner can (`IrMaterializeConstantsPass`, or the effect keyed by its name);
//! one no planner folds reaches the serializer as an aten conv, and its
//! aten spatial-op guard (`reject_aten_spatial_ops`) hands the graph to stock.
//!
//! **Dtypes** (design F2). RETAB keeps a quantized activation's val float32
//! and its dtype in qparams, so rc4's fake-dtype tests read qparams here
//! (`layout::dtype`): a conv is quantized when it has input qparams, and its
//! activation dtype is input qparam 0's. The created nodes record rc4's
//! dtypes: the TOSA convolution is int32 for an int8 or int16 activation
//! (critique A9: never its inherited int8 output qparam), the RESCALE takes
//! its output dtype, and a layout copy its producer's val dtype.
//!
//! **Output RESCALE** (RCP:464-509). `scales[i] = (s_in * s_w[i]) / s_out` in
//! f64, one per weight scale as rc4 zips them (normally one per output channel;
//! an A8W4 transposed conv's weight is quantized per input channel, and rc4
//! keeps those scales); the output qparams are the conv's own, else those of a
//! sole `view_copy` or `clamp` user (RCP:420-450). An int16 activation (A16W8) marks the TOSA convolution and
//! its bias INT48 (RCP:414-418, :856).
//!
//! **A16W8 branches** (RCP:665-816, :1234-1269; design §2.8). An A16W8 conv's
//! RESCALE narrows the INT48 accumulator to the conv's int16 output, but rc4
//! keeps the consumers that take int32 on a widened path from the
//! accumulator instead. These are the RESCALEs to int32 that
//! `IrInsertInt32Rescale` (rc4's InsertRescaleInt32Pass) put in front of
//! int32 compute, such as a residual add:
//!
//! ```text
//! direct, U = RESCALE(c, int32, s_U, zp, zp_U):
//!     U becomes RESCALE(t, int32, scales * s_U, 0, zp_U), in place, kwargs kept,
//!     and U's users read a permute(U, POST) inserted directly after it
//! behind a permute user P of c:
//!     after t: RESCALE(t, int32, scales, 0, zp) -> permute(·, POST) -> P' = permute(·, P's dims)
//!     P's int32 RESCALE users read P' instead; P goes when nothing else reads it
//! ```
//!
//! `scales * s_U` is `_combine_rescale_scales` (RCP:511-531): elementwise, a
//! single factor repeating. The conv's other users read the int16 path (`y`
//! above); when none are left, its RESCALE and permute are erased
//! (RCP:1260-1266). An A16W8 conv1d never branches (RCP:1236-1248): its
//! consumers stay behind the int16 RESCALE, view and permute.
//!
//! **U55** (RCP:583-663, :1273-1280). On the U55 subset (`is_u55_subset`, from
//! the TOSA spec), after every conv is rewritten, each A16W8 accumulator of
//! spatial rank 2 or 3 with two or more users is separated, so that every
//! TOSA convolution feeds one RESCALE. Its users, taken in graph order, must
//! all be RESCALEs. A RESCALE whose only user is a permute is a "head", and a
//! head whose RESCALE and permute sign like an earlier head's (target, the
//! args after the input, kwargs) merges into it. Every RESCALE left after the
//! first (critique A4: head or not) then reads its own clone of the
//! accumulator, inserted directly before it, and the accumulator and its
//! clones carry `do_not_fuse_duplicate`, which keeps `IrFuseDuplicateUsers`
//! from merging them again (design F7). The generic profile keeps duplicate
//! heads, as rc4 does.
//!
//! **U55 permute splits** (design F8). rc4 judges whatever permutes are
//! alive at `arm_pass_manager.py:736`, after this pass, including the ones
//! this pass creates. `IrDecomposePermuteForU55` does the same on U55,
//! asking Python's Vela probe, so this pass leaves the split to it.
//!
//! **Regions** (APU-135 S8; design §2.9, §4.3). A region's constants are
//! `get_attr` nodes (`GraphScope::Region`). A weight or bias the region
//! captures from the parent graph arrives as a branch input. rc4 resolves it
//! to the parent's constant here, through the boundary RESCALEs it inserted at
//! `arm_pass_manager.py:595` (`_resolve_control_flow_constant`, RCP:223-326);
//! RETAB inserts those RESCALEs at serialization
//! (`control_flow::contracts::apply_boundary_contracts`), after the region
//! pipeline. So a captured weight gets its layout change as runtime ops on the
//! branch input, `permute_copy(W, dims)` and for HWCM and conv1d a
//! `view_copy(·, shape)`, and a captured bias stays as it is; captured_conv
//! folds both into region constants after the boundary
//! contracts (`control_flow::captured_conv`). A captured operand is a branch
//! input read directly or through its boundary quantize and data-movement
//! copies, the chain captured_conv can resolve (`captured_branch_input`).
//!
//! **Control-flow outputs.** A conv reading a cond or while output whose
//! getitem has no output qparams, directly or through copies without input
//! qparams, such as IrSizeAdjustInput's slices, requests a stock
//! fallback when the cond or while carries no recorded target for that output
//! (see `control_flow_hazard`); RETAB records the targets before
//! IrFoldQdq, so in its pipeline that leaves only outputs no quantize
//! followed. FP8 is out of scope (design F1).
//!
//! **Stock fallbacks** (design §2.11, critique B5, B10). Where rc4 raises, the
//! pass requests a stock fallback (`stock_fallback::request`) and stops, so
//! stock reports its own error: a transposed conv that is not 2-D or is
//! dilated (RCP:939-947), a grouped CONV3D (RCP:147-158), a stride remainder
//! larger than the padding (RCP:126-129), a weight that is not a constant
//! (RCP:243-251) or, in a region, a captured one, a quantized conv whose bias
//! is not int32 or whose weight is not int8, or whose input or output qparams
//! are missing or per-channel (RCP:434-438, :471-474), dtypes rc4's TOSA
//! kernels reject (`tosa/dialect/ops/conv2d.py:17-88`), and an A16W8 conv's
//! direct INT32 RESCALE whose input zero point is not the conv's output zero
//! point, or whose scales don't combine with the conv's (RCP:511-531,
//! :711-716). So does anything the port cannot rewrite faithfully: a grouped
//! conv `IrDecomposeGroupedConv` declined (B5), the raw
//! `aten.conv{1,2,3}d`/`conv_transpose2d` forms (design §2.2), a conv whose
//! rewritten output shape would not be its own (B10), and an INT32 RESCALE or
//! permute user whose arguments don't parse. Each conv is planned before the
//! graph changes, so the convs before the fallback stay rewritten and the one
//! that requested it stays as it was; a fallback also skips the U55 step,
//! which rc4 never reaches after raising.

use std::collections::HashMap;

use crate::fx_constant_tensor_eval::permute_tensor_data;
use crate::fx_conv_args::{ConvArgKind, ConvArgs};
use crate::fx_fuse_duplicate_users_ir::node_signature_from;
use crate::fx_ir::{
    i64_list_arg, IrArg, IrGraph, IrNode, IrPass, MetaValue, NodeId, NodeMeta, NodeOp, QParam,
    QParamMap,
};
use crate::fx_meta::TensorFacts;
use crate::fx_target::{
    is_layout_passthrough_target, is_slice_tensor_target, noop_target, NoopTarget,
};
use crate::graph::ir::DType;
use crate::layout::constants::{constant_kind, is_constant};
use crate::layout::ctx::{GraphScope, InsertPoint, LayoutCtx, NodeOrder};
use crate::layout::dtype::{
    conv_output_dtype, is_quantized_conv, set_special_dtype, value_dtype, SpecialDtype,
    SPECIAL_DTYPE_KEY,
};
use crate::layout::meta;
use crate::layout::shape::tosa::{
    conv2d_shape, conv3d_shape, depthwise_conv2d_shape, transpose_conv2d_shape,
};
use crate::layout::shape::{expand_around_channel, permute_shape};
use crate::lowering_effects::LoweringInputKind;
use crate::pipeline_config::PipelineConfig;
use crate::store::TensorData;

const CONVOLUTION: &str = "aten.convolution.default";
const CONV_TRANSPOSE2D_INPUT: &str = "aten.conv_transpose2d.input";
const PERMUTE: &str = "aten.permute_copy.default";
const VIEW: &str = "aten.view_copy.default";
const CLAMP: &str = "aten.clamp.default";
const RESCALE: &str = "backend.tosa.RESCALE.default";
const TOSA_CONV2D: &str = "backend.tosa.CONV2D.default";
const TOSA_DEPTHWISE_CONV2D: &str = "backend.tosa.DEPTHWISE_CONV2D.default";
const TOSA_TRANSPOSE_CONV2D: &str = "backend.tosa.TRANSPOSE_CONV2D.default";
const TOSA_CONV3D: &str = "backend.tosa.CONV3D.default";

const INT8: &str = "torch.int8";
const INT16: &str = "torch.int16";
const INT32: &str = "torch.int32";

/// `DO_NOT_FUSE_DUPLICATE_META_KEY`
/// (`backends/transforms/fuse_duplicate_users_pass.py:17`), which
/// `IrFuseDuplicateUsers` honours.
const DO_NOT_FUSE_DUPLICATE: &str = "do_not_fuse_duplicate";

/// `NHWC_ORDER` and `NHWC_INVERSE_ORDER` (`backends/arm/constants.py`).
const NHWC: [usize; 4] = [0, 2, 3, 1];
const NCHW: [usize; 4] = [0, 3, 1, 2];
/// `ODHWI_ORDER` and `ODHWI_INVERSE_ORDER`.
const NDHWC: [usize; 5] = [0, 2, 3, 4, 1];
const NCDHW: [usize; 5] = [0, 4, 1, 2, 3];
/// A conv1d's input and output permutation (RCP:1012-1013).
const NLC: [usize; 3] = [0, 2, 1];
/// `OHWI_ORDER`, the transposed conv's IOHW weight to OHWI.
const TRANSPOSED_OHWI: [usize; 4] = [1, 2, 3, 0];
/// `HWCM_ORDER`.
const HWCM: [usize; 4] = [2, 3, 0, 1];
/// The conv1d weights' permutations (RCP:1042, :1056).
const CONV1D_HWCM: [usize; 3] = [1, 2, 0];
const CONV1D_OHWI: [usize; 3] = [0, 2, 1];

/// rc4's RewriteConvPass; see the module docs.
pub(crate) struct IrRewriteConvPass {
    /// `get_context_spec().is_U55_subset` (RCP:1273): whether A16W8
    /// accumulators feeding several RESCALEs are separated, one TOSA
    /// convolution per RESCALE (design §2.8). rc4 reads it from the lowering
    /// context's TOSA spec; [`IrRewriteConvPass::new`] reads RETAB's copy of
    /// that spec, `PipelineConfig::is_u55_subset`.
    pub(crate) is_u55_subset: bool,
    /// The pipeline's region flag: constants are `get_attr` nodes
    /// (`GraphScope::Region`), and weights and biases captured from the parent
    /// graph are left for captured_conv (design §2.9; see the module docs).
    pub(crate) in_region: bool,
}

impl IrRewriteConvPass {
    /// The pass for a pipeline lowering to `config`'s TOSA spec: the U55
    /// separation runs on the U55 subset (`TOSA-1.0+INT+...+u55`, which
    /// `PipelineConfig::from_tosa_spec` records as `is_u55_subset`, as rc4's
    /// `TosaSpecification.is_U55_subset` does).
    pub(crate) fn new(config: &PipelineConfig, in_region: bool) -> Self {
        Self {
            is_u55_subset: config.is_u55_subset,
            in_region,
        }
    }
}

impl IrPass for IrRewriteConvPass {
    fn name(&self) -> &str {
        "IrRewriteConv"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        // RCP:882-887: a graph without a convolution is left alone.
        if !graph
            .live_nodes()
            .any(|node| node.op == NodeOp::CallFunction && conv_form(&node.target).is_some())
        {
            return false;
        }
        let scope = GraphScope::of_region(self.in_region);
        let mut ctx = LayoutCtx::with_scope(graph, scope);
        let mut changed = false;
        let mut fell_back = false;
        // RCP:881: the TOSA convolutions of A16W8 convs of spatial rank 2 or 3,
        // for the U55 step.
        let mut a16w8_tosa_ops = Vec::new();
        // RCP:882 walks `graph.nodes` while it edits; the nodes it inserts come
        // after the conv it rewrites and are never convs, so a snapshot visits
        // the same convs.
        for id in ctx.order().to_vec() {
            let node = ctx.node(id);
            if node.op != NodeOp::CallFunction {
                continue;
            }
            let planned = match conv_form(&node.target) {
                None => continue,
                Some(ConvForm::Raw) => Err(format!(
                    "{} reaches RewriteConvPass's slot; only aten.convolution is rewritten",
                    node.target
                )),
                Some(ConvForm::Convolution) => plan(ctx.graph(), ctx.order(), scope, id),
            };
            match planned {
                Ok(Some(rewrite)) => {
                    a16w8_tosa_ops.extend(apply(&mut ctx, rewrite));
                    changed = true;
                }
                Ok(None) => {}
                Err(reason) => {
                    let name = &ctx.node(id).name;
                    crate::stock_fallback::request(format!("IrRewriteConv: {name}: {reason}"));
                    fell_back = true;
                    break;
                }
            }
        }
        // RCP:1273-1280, with the graph order taken once, after the last conv.
        if self.is_u55_subset && !fell_back && !a16w8_tosa_ops.is_empty() {
            let node_order = graph_order(ctx.order());
            for tosa_op in a16w8_tosa_ops {
                separate_u55_a16w8_output_rescales(&mut ctx, tosa_op, &node_order);
            }
        }
        if changed {
            ctx.finish();
        }
        changed
    }
}

/// `{node: index for index, node in enumerate(graph_module.graph.nodes)}`
/// (RCP:1274-1276): each node's position in `order`.
pub(crate) fn graph_order(order: &NodeOrder) -> HashMap<NodeId, usize> {
    order
        .iter()
        .enumerate()
        .map(|(position, id)| (id, position))
        .collect()
}

/// How a conv target reaches the pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConvForm {
    /// `aten.convolution.default`, the only form RCP matches (RCP:883-887).
    Convolution,
    /// `aten.conv{1,2,3}d.default` and `aten.conv_transpose2d.input`, which an
    /// edge graph never holds; only `IrDecomposeGroupedConv` re-creates one,
    /// for a raw source (design §2.2).
    Raw,
}

fn conv_form(target: &str) -> Option<ConvForm> {
    match target {
        CONVOLUTION => Some(ConvForm::Convolution),
        "aten.conv1d.default"
        | "aten.conv2d.default"
        | "aten.conv3d.default"
        | CONV_TRANSPOSE2D_INPUT => Some(ConvForm::Raw),
        _ => None,
    }
}

/// `_adjust_pad_if_needed` (RCP:75-131) for static sizes: the trailing pad of
/// one axis, reduced by `(input + 2 * pad - dilation * (kernel - 1) - 1) mod
/// stride` so that TOSA's output size divides exactly. rc4 raises when the
/// remainder exceeds the pad (SizeAdjustInputPass should have sliced the
/// input), and so does this; rc4's symbolic remainders have no counterpart in
/// RETAB's static IR.
pub(crate) fn adjust_pad_if_needed(
    input_len: i64,
    kernel: i64,
    stride: i64,
    pad: i64,
    dilation: i64,
) -> Result<i64, String> {
    if stride < 1 {
        return Err(format!("stride {stride} is not positive"));
    }
    let numerator = i128::from(input_len) + 2 * i128::from(pad)
        - i128::from(dilation) * (i128::from(kernel) - 1)
        - 1;
    // Python's `%` with a positive divisor, which `rem_euclid` is.
    let remainder = numerator.rem_euclid(i128::from(stride));
    if remainder > i128::from(pad) {
        return Err(
            "This case should be handled by SizeAdjustInputPass, is it enabled?".to_string(),
        );
    }
    Ok(pad - remainder as i64)
}

/// `_is_depthwise_conv` (RCP:133-145) on a conv's input shape, `groups` and
/// output channels: a rank-3 or rank-4 input whose channel count equals the
/// groups and divides the output channels.
fn is_depthwise(input_shape: &[i64], groups: i64, out_channels: i64) -> bool {
    matches!(input_shape.len(), 3 | 4) && {
        let in_channels = input_shape[1];
        in_channels == groups && in_channels != 0 && out_channels % in_channels == 0
    }
}

/// `_get_effective_conv_input_qparams` (RCP:452-462): the conv's input
/// qparams, with a missing activation (0) or weight (1) entry taken from input
/// qparam 0 of that argument, as a VGF annotation can leave it on an input
/// view.
fn effective_input_qparams(graph: &IrGraph, conv: &IrNode) -> QParamMap {
    let mut qparams = conv.meta.cloned_input_qparams().unwrap_or_default();
    for index in [0, 1] {
        if qparams.get(&index).is_some() {
            continue;
        }
        let Some(arg) = conv.args.get(index).and_then(IrArg::as_node) else {
            continue;
        };
        if let Some(qparam) = graph.node(arg).meta.input_qparam_at(0) {
            qparams.insert(index, qparam.clone());
        }
    }
    qparams
}

/// `_get_effective_output_qparams` (RCP:420-450): the conv's output qparams,
/// else those of its sole user when that is a `view_copy` or a `clamp` that
/// has them (a quantized activation that was not fused). rc4 raises otherwise.
fn effective_output_qparams<'g>(graph: &'g IrGraph, conv: &'g IrNode) -> Option<&'g QParamMap> {
    let nonempty = |node: &'g IrNode| node.meta.output_qparams().filter(|map| !map.is_empty());
    if let Some(qparams) = nonempty(conv) {
        return Some(qparams);
    }
    let [user] = graph.live_user_ids(conv.id)[..] else {
        return None;
    };
    let user = graph.node(user);
    if matches!(user.target.as_str(), VIEW | CLAMP) {
        return nonempty(user);
    }
    None
}

/// `_is_direct_int32_rescale` (RCP:533-541): a RESCALE to int32.
fn is_direct_int32_rescale(node: &IrNode) -> bool {
    node.op == NodeOp::CallFunction
        && node.is_rescale()
        && matches!(node.args.get(1), Some(IrArg::Dtype(dtype) | IrArg::Str(dtype)) if dtype == INT32)
}

fn is_permute(node: &IrNode) -> bool {
    node.op == NodeOp::CallFunction && node.target == PERMUTE
}

/// `id`'s live users in graph order. rc4 walks `node.users`, the order the
/// uses were made in, which the retrace closing every rc4 pass makes graph
/// order; RewriteConvPass adds no use of a conv it hasn't rewritten yet.
fn users_in_order(graph: &IrGraph, order: &NodeOrder, id: NodeId) -> Vec<NodeId> {
    let mut users = graph.live_user_ids(id);
    order.sort(&mut users);
    users
}

/// `_get_direct_int32_rescale_users` (RCP:543-547): the INT32 RESCALEs among
/// `id`'s users.
fn direct_int32_rescale_users(graph: &IrGraph, order: &NodeOrder, id: NodeId) -> Vec<NodeId> {
    users_in_order(graph, order, id)
        .into_iter()
        .filter(|&user| is_direct_int32_rescale(graph.node(user)))
        .collect()
}

/// `_combine_rescale_scales` (RCP:511-531): the products of two RESCALEs'
/// factors, elementwise, a single factor standing for every channel; rc4
/// raises for an empty list or two lists of different lengths above one.
pub(crate) fn combine_rescale_scales(lhs: &[f64], rhs: &[f64]) -> Result<Vec<f64>, String> {
    if lhs.is_empty() || rhs.is_empty() {
        return Err("Cannot combine empty rescale factors.".to_string());
    }
    if lhs.len() != 1 && rhs.len() != 1 && lhs.len() != rhs.len() {
        return Err(format!(
            "Cannot combine rescales with incompatible scale counts: {} and {}.",
            lhs.len(),
            rhs.len()
        ));
    }
    let count = lhs.len().max(rhs.len());
    Ok((0..count)
        .map(|index| lhs[index % lhs.len()] * rhs[index % rhs.len()])
        .collect())
}

/// A RESCALE's scales, input zero point and output zero point (`args[2..=4]`).
fn rescale_params(node: &IrNode) -> Option<(Vec<f64>, i64, i64)> {
    let scales = match node.args.get(2)? {
        IrArg::List(items) => items
            .iter()
            .map(IrArg::as_numeric_f64)
            .collect::<Option<Vec<f64>>>()?,
        _ => return None,
    };
    let input_zp = node.args.get(3)?.as_int()?;
    let output_zp = node.args.get(4)?.as_int()?;
    Some((scales, input_zp, output_zp))
}

/// A direct INT32 user's retargeting (RCP:690-748).
struct DirectInt32User {
    rescale: NodeId,
    /// `_combine_rescale_scales(conv scales, user scales)`.
    scales: Vec<f64>,
    output_zp: i64,
}

/// A permute user of the conv that INT32 RESCALEs read (RCP:750-816).
struct PermutedInt32Users {
    permute: NodeId,
    /// The INT32 RESCALE users, which move to the widened branch.
    rescales: Vec<NodeId>,
    /// The widened copy's value shape: the conv's, permuted by the user's dims.
    shape: Vec<i64>,
}

/// `_insert_a16w8_output_branches` (RCP:665-816) for one A16W8 conv of
/// spatial rank 2 or 3, planned before the graph changes. Both lists may be
/// empty; rc4 still records the conv's TOSA op for the U55 step.
struct Int32Branches {
    direct: Vec<DirectInt32User>,
    permuted: Vec<PermutedInt32Users>,
    /// The conv RESCALE's scales and output zero point (RCP:701-702), which
    /// the widened branches reuse.
    conv_scales: Vec<f64>,
    conv_output_zp: i64,
}

/// Plan the A16W8 branches of `conv`, whose default RESCALE is `rescale` and
/// whose value has shape `conv_shape`; `Err` where rc4 raises (RCP:711-716,
/// :511-531) or a user's arguments don't parse.
fn plan_int32_branches(
    graph: &IrGraph,
    order: &NodeOrder,
    conv: NodeId,
    rescale: &RescaleRewrite,
    conv_shape: &[i64],
) -> Result<Int32Branches, String> {
    let mut direct = Vec::new();
    for user in direct_int32_rescale_users(graph, order, conv) {
        let node = graph.node(user);
        let (user_scales, input_zp, output_zp) = rescale_params(node)
            .ok_or_else(|| format!("INT32 RESCALE {} has malformed arguments", node.name))?;
        if input_zp != rescale.output_zp {
            return Err(format!(
                "Cannot combine convolution and INT32 rescales with different intermediate zero \
                 points: {} and {input_zp}.",
                rescale.output_zp
            ));
        }
        direct.push(DirectInt32User {
            rescale: user,
            scales: combine_rescale_scales(&rescale.scales, &user_scales)?,
            output_zp,
        });
    }
    // `_get_permute_int32_rescale_users` (RCP:549-561).
    let mut permuted = Vec::new();
    for user in users_in_order(graph, order, conv) {
        let node = graph.node(user);
        if !is_permute(node) {
            continue;
        }
        let rescales = direct_int32_rescale_users(graph, order, user);
        if rescales.is_empty() {
            continue;
        }
        let shape = node
            .arg_int_list(1)
            .and_then(|dims| permute_shape(conv_shape, &dims).ok())
            .ok_or_else(|| format!("permute {} does not permute the conv's value", node.name))?;
        permuted.push(PermutedInt32Users {
            permute: user,
            rescales,
            shape,
        });
    }
    Ok(Int32Branches {
        direct,
        permuted,
        conv_scales: rescale.scales.clone(),
        conv_output_zp: rescale.output_zp,
    })
}

/// A per-tensor qparam's scale: rc4's `get_scale_per_tensor` raises for a
/// per-channel one.
fn per_tensor(qparam: &QParam, what: &str) -> Result<(f64, i64), String> {
    if qparam.is_per_channel() {
        return Err(format!("the {what} qparam is per-channel"));
    }
    Ok((qparam.scale, qparam.zp))
}

fn is_float(dtype: &str) -> bool {
    matches!(dtype, "torch.float32" | "torch.float16" | "torch.bfloat16")
}

fn to_i64(dims: &[usize]) -> Vec<i64> {
    dims.iter().map(|&dim| dim as i64).collect()
}

/// Why a conv that `plan` could rewrite must still go to stock when the
/// boundary contracts cannot read past the nodes this pass creates (APU-135
/// S8; design §4.3). rc4 inserts its control-flow boundary RESCALEs before
/// RewriteConvPass (APM:595); RETAB's boundary contracts run at
/// serialization, after the whole pipeline
/// (`control_flow::contracts::apply_boundary_contracts`). A cond or while
/// output's target qparam comes from its getitem's output qparams, else from
/// the input qparams of the getitem's direct consumers
/// (`infer_output_qparams_map_ir`); the conv was that consumer, its input
/// permute has none, as rc4's does, and the region's output RESCALE would be
/// lost. This holds in a region too, for the output of a cond it holds.
///
/// (A region output's own qparam is safe: RETAB records it from the
/// region's output dequantize, as rc4's FoldAndAnnotateQParamsPass does, so
/// the output contract does not reverse engineer the rewrite's qparam-less
/// RESCALE; `contracts::rewrite_output_dequantize_boundaries_ir`.)
///
/// The lookup also fails when the conv reads such a getitem through
/// data-movement copies without input qparams (`control_flow_output_read`).
/// IrSizeAdjustInput builds its slices with emptied qparam maps, as rc4's
/// `create_node(from_node=...)` does, and `IrDecomposeLinear`'s input views
/// and `IrDecomposeGroupedConv`'s group slices have none. The contract then
/// finds no target whether or not the conv is rewritten. But the aten conv
/// sends the graph to stock at serialization (the serializer's aten spatial-op
/// guard, `reject_aten_spatial_ops`), and the rewritten graph would serialize
/// without the output RESCALE. So these convs fall back too, unless the cond
/// or while carries a recorded target for the output: RETAB records them
/// before IrFoldQdq, from the quantize after each getitem, which
/// is where rc4's InsertControlFlowRescalesPass target comes from at APM:595
/// (`control_flow::IrRecordControlFlowRescaleTargetsPass`), and the contract
/// takes a recorded target whatever the graph shows by then.
fn control_flow_hazard(graph: &IrGraph, input: NodeId) -> Option<String> {
    let (getitem, copies) = control_flow_output_read(graph, input)?;
    let getitem = &graph.node(getitem).name;
    Some(if copies.is_empty() {
        format!(
            "it reads control-flow output {getitem}, whose boundary contract takes its qparam \
             from the conv, which the rewrite replaces with a qparam-less permute (APU-135 S8)"
        )
    } else {
        let copies: Vec<&str> = copies
            .iter()
            .map(|&copy| graph.node(copy).name.as_str())
            .collect();
        format!(
            "it reads control-flow output {getitem} through copies without input qparams ({}), \
             so the output's boundary contract finds no qparam, and the aten conv is what sends \
             the graph to stock (APU-135 S8)",
            copies.join(", ")
        )
    })
}

/// The cond or while output the conv reads with no qparam on the way for the
/// output's boundary contract: a getitem without output qparams, read directly
/// or through data-movement copies without input qparams. The contract looks
/// only at the getitem's own output qparams and at its direct consumers' input
/// qparams, so it can read none of these nodes, and the conv's input qparams
/// leave with the rewrite. Returns the getitem and the copies, from the
/// getitem's consumer to the conv's input.
fn control_flow_output_read(graph: &IrGraph, input: NodeId) -> Option<(NodeId, Vec<NodeId>)> {
    let mut copies = Vec::new();
    let mut current = input;
    // A copy chain cannot be longer than the graph; the bound keeps a
    // malformed (cyclic) graph from looping.
    for _ in 0..=graph.nodes.len() {
        let node = graph.node(current);
        if node.is_getitem() {
            let unqualified = node.meta.output_qparam().is_none()
                && node
                    .first_node_arg()
                    .is_some_and(|source| graph.node(source).is_control_flow())
                && !crate::layout::control_flow_guard::output_target_recorded(graph, node);
            copies.reverse();
            return unqualified.then_some((current, copies));
        }
        if !copies_its_input(node) || node.meta.input_qparam().is_some() {
            return None;
        }
        copies.push(current);
        current = node.first_node_arg()?;
    }
    None
}

/// A data-movement copy: its value holds its first input's elements unchanged,
/// so they keep that input's quantization. These are permutes, views and other
/// rank changes, slices, and clones.
fn copies_its_input(node: &IrNode) -> bool {
    let target = node.target.as_str();
    node.op == NodeOp::CallFunction
        && (is_layout_passthrough_target(target)
            || is_slice_tensor_target(target)
            || noop_target(target) == Some(NoopTarget::PassthroughFirstArg))
}

/// The conv's weight, as RCP finds it (`_rewrite_weight`, RCP:328-378).
enum WeightOperand<'g> {
    /// A constant. In the parent it must be a placeholder: rc4 raises for
    /// anything else, including a `get_attr` (`get_constant_placeholder_kind`).
    /// In a region only a `get_attr` constant survives ingest as a constant
    /// (design F4).
    Constant(&'g IrNode),
    /// A region weight captured from the parent graph, with its value: rc4
    /// resolves it to the parent's constant (`_resolve_control_flow_constant`,
    /// RCP:223-261); here it gets runtime layout ops, which captured_conv folds
    /// (see the module docs).
    Captured(TensorFacts),
}

/// Classify the conv's weight; `Err` is a stock fallback's reason.
fn weight_operand(
    graph: &IrGraph,
    scope: GraphScope,
    weight: NodeId,
) -> Result<WeightOperand<'_>, String> {
    let node = graph.node(weight);
    match scope {
        GraphScope::Parent if is_constant(node) && node.op == NodeOp::Placeholder => {
            Ok(WeightOperand::Constant(node))
        }
        GraphScope::Parent => Err(format!(
            "weight {} is not a constant placeholder, which RewriteConvPass requires",
            node.name
        )),
        GraphScope::Region if is_constant(node) && node.op == NodeOp::GetAttr => {
            Ok(WeightOperand::Constant(node))
        }
        GraphScope::Region if captured_branch_input(graph, weight).is_some() => node
            .meta
            .tensor_facts()
            .cloned()
            .map(WeightOperand::Captured)
            .ok_or_else(|| format!("captured weight {} has no value", node.name)),
        // RCP:243-251: `_resolve_control_flow_constant` raises.
        GraphScope::Region => Err(format!(
            "weight {} is neither a region constant nor captured from the parent graph, which \
             RewriteConvPass requires",
            node.name
        )),
    }
}

/// The branch input a region operand captures from the parent graph, when it
/// does: the operand is a placeholder, which ingest makes a runtime input
/// bound to the parent's operand, data or not (`control_flow/assembly.rs`),
/// read directly or through its boundary quantize, which the boundary
/// contracts turn into the boundary RESCALE (or drop), a RESCALE, and the
/// data-movement copies RETAB puts on a weight (a Linear's weight view). These
/// are the chains captured_conv folds after the contracts
/// (`control_flow::captured_conv`), as rc4's `_resolve_control_flow_constant`
/// follows RESCALEs and branch inputs (RCP:223-261).
fn captured_branch_input(graph: &IrGraph, operand: NodeId) -> Option<NodeId> {
    let mut current = operand;
    // A chain cannot be longer than the graph; the bound keeps a malformed
    // (cyclic) graph from looping.
    for _ in 0..=graph.nodes.len() {
        let node = graph.node(current);
        if node.op == NodeOp::Placeholder {
            return Some(current);
        }
        let passes_through = node.op == NodeOp::CallFunction
            && (node.is_quantize()
                || node.is_rescale()
                || matches!(node.target.as_str(), PERMUTE | VIEW));
        if !passes_through {
            return None;
        }
        current = node.first_node_arg()?;
    }
    None
}

/// The weight the TOSA convolution reads.
enum WeightRewrite {
    /// The pre-permuted constant (RCP:328-403), directly after `source`.
    Constant {
        /// The source weight, which the new constant follows.
        source: NodeId,
        name_suffix: &'static str,
        kind: LoweringInputKind,
        data: TensorData,
        /// The source's `tosa_special_dtype` (RCP:377-378).
        special: Option<MetaValue>,
    },
    /// A captured region weight's layout change as runtime ops after
    /// `source` (design §2.9): `permute_copy(source, dims)`, then for HWCM and
    /// conv1d `view_copy(·, reshaped.shape)`. The last is named as rc4 names
    /// the constant, `{c}_weight_<suffix>`.
    Runtime {
        source: NodeId,
        name_suffix: &'static str,
        dims: Vec<i64>,
        permuted: TensorFacts,
        reshaped: Option<TensorFacts>,
    },
}

enum BiasRewrite {
    /// The conv's own bias.
    Keep(NodeId),
    /// `_add_bias`'s zeros (RCP:160-195).
    Zeros(TensorData),
    /// A region bias captured from the parent graph, which stays: captured_conv
    /// requantizes it into the int32 bias, and marks that
    /// INT48 for an int16 activation, after the boundary contracts
    /// (`_rewrite_control_flow_bias`, RCP:263-326).
    Captured(NodeId),
}

/// RCP:818-877's RESCALE.
struct RescaleRewrite {
    dtype: String,
    scales: Vec<f64>,
    output_zp: i64,
}

/// Everything one conv's rewrite creates, computed before the graph changes.
struct ConvRewrite {
    conv: NodeId,
    input: NodeId,
    /// The conv's meta with the effective input qparams (RCP:903-906): the
    /// source of every created node's meta.
    source: NodeMeta,
    /// RCP:408-418: the bias and, through `int48`, the TOSA op are INT48.
    int16_activation: bool,
    weight: WeightRewrite,
    bias: BiasRewrite,
    pre_dims: Vec<i64>,
    permuted_input: TensorFacts,
    /// Conv1d's `[N, 1, L, C]` view (RCP:1025-1032).
    input_view: Option<TensorFacts>,
    target: &'static str,
    /// The arguments after `(x', W', b')`.
    params: Vec<IrArg>,
    tosa: TensorFacts,
    /// RCP:856: A16W8's accumulator is INT48.
    int48: bool,
    rescale: Option<RescaleRewrite>,
    /// Conv1d's `[N, Lout, Co]` view (RCP:1200-1217).
    squeeze: Option<Vec<i64>>,
    post_dims: Vec<i64>,
    output_shape: Vec<i64>,
    /// RCP:1234-1267: an A16W8 conv of spatial rank 2 or 3 branches.
    a16w8_branches: Option<Int32Branches>,
}

/// One kind's layout facts (RCP:939-1141).
struct Kind {
    target: &'static str,
    pre: &'static [usize],
    post: &'static [usize],
    weight_dims: &'static [usize],
    weight_reshape: Option<Vec<i64>>,
    name_suffix: &'static str,
}

/// Plan the rewrite of `conv` (RCP:889-1271 without the graph edits), or
/// `None` for a conv to leave alone; `Err` is a stock fallback's reason.
/// `order` is the graph order, for walking the conv's users as rc4 does.
fn plan(
    graph: &IrGraph,
    order: &NodeOrder,
    scope: GraphScope,
    conv: NodeId,
) -> Result<Option<ConvRewrite>, String> {
    let node = graph.node(conv);
    let args = ConvArgs::parse_convolution_default(node)
        .filter(|args| args.kind == ConvArgKind::ConvolutionDefault && args.bias_well_formed)
        .ok_or_else(|| "malformed convolution arguments".to_string())?;

    // Design §2.2: rc4 folds a conv of constants at `:700`, before RCP.
    let constant = |id: NodeId| is_constant(graph.node(id));
    if constant(args.input) && constant(args.weight) && args.bias.is_none_or(constant) {
        return Ok(None);
    }

    let facts = |id: NodeId, what: &str| {
        graph
            .node(id)
            .meta
            .tensor_facts()
            .cloned()
            .ok_or_else(|| format!("the {what} has no value"))
    };
    let input = facts(args.input, "input")?;
    let output = facts(conv, "convolution")?;
    let rank = input.shape.len();
    if !(3..=5).contains(&rank) || output.shape.len() != rank {
        return Err(format!(
            "a rank-{rank} input and a rank-{} output are not a 1-D, 2-D or 3-D convolution",
            output.shape.len()
        ));
    }
    let spatial_rank = rank - 2;
    let weight_operand = weight_operand(graph, scope, args.weight)?;
    let weight_name = &graph.node(args.weight).name;
    let (weight_shape, weight_dtype) = match &weight_operand {
        WeightOperand::Constant(weight_node) => {
            let data = &weight_node
                .tensor
                .as_ref()
                .expect("a constant carries its data")
                .data;
            if weight_node
                .meta
                .val_shape()
                .is_some_and(|shape| shape != data.shape.as_slice())
            {
                return Err(format!(
                    "weight {weight_name}'s value {:?} is not its data's shape {:?}",
                    weight_node.meta.val_shape(),
                    data.shape
                ));
            }
            (data.shape.clone(), data.dtype.torch_str().to_owned())
        }
        // The weight as the branch reads it: the boundary quantize gives it
        // the dtype the parent's constant is requantized into.
        WeightOperand::Captured(value) => (
            value.shape.clone(),
            value_dtype(graph, args.weight)
                .ok_or_else(|| format!("captured weight {weight_name} has no dtype"))?,
        ),
    };
    if weight_shape.len() != rank {
        return Err(format!(
            "weight {weight_name} of shape {weight_shape:?} does not fit a rank-{rank} convolution"
        ));
    }

    // Step 1 (RCP:903-906, :405-412).
    let quantized = is_quantized_conv(&node.meta);
    let mut source = node.meta.clone();
    let input_qparams = quantized.then(|| effective_input_qparams(graph, node));
    if let Some(qparams) = &input_qparams {
        source.set_input_qparams(qparams.clone());
    }
    let activation = input_qparams
        .as_ref()
        .and_then(|qparams| qparams.get(&0))
        .map(|qparam| qparam.dtype_str.clone())
        .filter(|dtype| !dtype.is_empty())
        .or_else(|| value_dtype(graph, args.input))
        .ok_or_else(|| "the input has no dtype".to_string())?;
    let int16_activation = activation == INT16;

    // The output dtype of rc4's TOSA kernels and the operand dtypes they
    // require (`validate_conv2d_args_dtypes`, `conv2d.py:17-88`).
    let int_activation = matches!(activation.as_str(), INT8 | INT16);
    let tosa_dtype = match (quantized, int_activation) {
        (true, true) => INT32,
        (false, false) if is_float(&activation) => {
            conv_output_dtype(&activation).expect("a float conv input has an output dtype")
        }
        _ => {
            return Err(format!(
                "a {} conv reading a {activation} input, which rc4's TOSA kernels reject",
                if quantized { "quantized" } else { "float" }
            ))
        }
    };
    let expected_weight = if quantized { INT8 } else { activation.as_str() };
    if weight_dtype != expected_weight {
        return Err(format!(
            "a {weight_dtype} weight with a {activation} input, which rc4's TOSA kernels reject"
        ));
    }

    // Step 2 (RCP:908-917).
    let stride = expand_around_channel(&args.stride, spatial_rank);
    let dilation = expand_around_channel(&args.dilation, spatial_rank);
    let padding = expand_around_channel(&args.padding, spatial_rank);
    if [&stride, &dilation, &padding]
        .iter()
        .any(|param| param.len() != spatial_rank)
    {
        return Err("stride, padding or dilation does not match the spatial rank".to_string());
    }
    let out_channels = output.shape[1];
    let depthwise = is_depthwise(&input.shape, args.groups, out_channels);
    // Critique B5: rc4's DecomposeGroupedConvPass leaves only depthwise
    // rank-3/4 convs grouped (`decompose_grouped_conv_pass.py:254-263`); a
    // grouped conv `IrDecomposeGroupedConv` declined would lose its groups.
    if args.groups != 1 && (args.transposed || !depthwise) {
        return Err(format!(
            "a grouped (groups={}) {}conv that rc4 decomposes before RewriteConvPass",
            args.groups,
            if args.transposed { "transposed " } else { "" }
        ));
    }

    // Step 3 (RCP:919-933).
    let bias = match args.bias {
        None => {
            let dtype = if quantized {
                DType::Int32
            } else {
                DType::from_torch_str(&output.dtype)
            };
            if dtype.torch_str() != tosa_dtype {
                return Err(format!(
                    "a {} zero bias for a {tosa_dtype} accumulator",
                    dtype.torch_str()
                ));
            }
            let size = usize::try_from(out_channels).map_err(|_| "negative channels")?;
            BiasRewrite::Zeros(TensorData {
                data: vec![0; size * dtype.storage_size_bytes()],
                shape: vec![out_channels],
                dtype,
            })
        }
        Some(bias)
            if scope == GraphScope::Region
                && !(is_constant(graph.node(bias)) && graph.node(bias).op == NodeOp::GetAttr) =>
        {
            // RCP:925-931 requantizes a captured bias through the boundary
            // RESCALEs, which RETAB inserts at serialization: captured_conv does
            // it then, and sends a quantized conv whose bias
            // is still not int32 to stock, as rc4's TOSA kernel rejects it.
            if captured_branch_input(graph, bias).is_none() {
                return Err(format!(
                    "bias {} is neither a region constant nor captured from the parent graph",
                    graph.node(bias).name
                ));
            }
            BiasRewrite::Captured(bias)
        }
        Some(bias) => {
            let bias_node = graph.node(bias);
            let bias_dtype = value_dtype(graph, bias)
                .ok_or_else(|| format!("bias {} has no dtype", bias_node.name))?;
            if bias_dtype != tosa_dtype {
                // RCP:925-931: a quantized conv's RESCALE'd control-flow bias
                // is requantized there; any other one rc4's TOSA kernel rejects.
                return Err(if quantized && bias_node.is_rescale() {
                    format!(
                        "bias {} is a RESCALE, which only a control-flow branch holds",
                        bias_node.name
                    )
                } else {
                    format!(
                        "a {bias_dtype} bias for a {tosa_dtype} accumulator, which rc4's TOSA \
                         kernels reject"
                    )
                });
            }
            BiasRewrite::Keep(bias)
        }
    };

    // Step 4: the kind (RCP:939-1141).
    let (kind, params, input_view_shape) = if args.transposed {
        if spatial_rank != 2 {
            return Err(
                "Only 2D transpose convolutions are supported in the Arm backend.".to_string(),
            );
        }
        if dilation.iter().any(|&d| d != 1) {
            return Err(
                "Transpose convolutions with dilation are not supported in the Arm backend."
                    .to_string(),
            );
        }
        let output_padding = expand_around_channel(&args.output_padding, spatial_rank);
        let [op0, op1] = output_padding[..] else {
            return Err("output padding does not match the spatial rank".to_string());
        };
        let out_pad = [
            -padding[0],
            -padding[0] + op0,
            -padding[1],
            -padding[1] + op1,
        ];
        let kind = Kind {
            target: TOSA_TRANSPOSE_CONV2D,
            pre: &NHWC,
            post: &NCHW,
            weight_dims: &TRANSPOSED_OHWI,
            weight_reshape: None,
            name_suffix: "ohwi",
        };
        (
            kind,
            vec![i64_list_arg(&out_pad), i64_list_arg(&stride)],
            None,
        )
    } else {
        // RCP:987-1004: duplicate each pad, then trim each trailing one.
        let mut pad: Vec<i64> = padding.iter().flat_map(|&p| [p, p]).collect();
        for axis in 0..spatial_rank {
            pad[2 * axis + 1] = adjust_pad_if_needed(
                input.shape[axis + 2],
                weight_shape[axis + 2],
                stride[axis],
                pad[2 * axis + 1],
                dilation[axis],
            )?;
        }
        match spatial_rank {
            1 => {
                let [n, c, l] = input.shape[..] else {
                    unreachable!("rank 3")
                };
                let kernel = weight_shape[2];
                let kind = if depthwise {
                    Kind {
                        target: TOSA_DEPTHWISE_CONV2D,
                        pre: &NLC,
                        post: &NLC,
                        weight_dims: &CONV1D_HWCM,
                        weight_reshape: Some(vec![1, kernel, c, weight_shape[0] / c]),
                        name_suffix: "hwicm",
                    }
                } else {
                    Kind {
                        target: TOSA_CONV2D,
                        pre: &NLC,
                        post: &NLC,
                        weight_dims: &CONV1D_OHWI,
                        weight_reshape: Some(vec![weight_shape[0], 1, kernel, weight_shape[1]]),
                        name_suffix: "ohwi",
                    }
                };
                let params = vec![
                    i64_list_arg(&[1, stride[0]]),
                    i64_list_arg(&[0, 0, pad[0], pad[1]]),
                    i64_list_arg(&[1, dilation[0]]),
                ];
                (kind, params, Some(vec![n, 1, l, c]))
            }
            3 => {
                let kind = Kind {
                    target: TOSA_CONV3D,
                    pre: &NDHWC,
                    post: &NCDHW,
                    weight_dims: &NDHWC,
                    weight_reshape: None,
                    name_suffix: "odhwi",
                };
                let params = vec![
                    i64_list_arg(&stride),
                    i64_list_arg(&pad),
                    i64_list_arg(&dilation),
                ];
                (kind, params, None)
            }
            _ => {
                let kind = if depthwise {
                    let c = input.shape[1];
                    Kind {
                        target: TOSA_DEPTHWISE_CONV2D,
                        pre: &NHWC,
                        post: &NCHW,
                        weight_dims: &HWCM,
                        weight_reshape: Some(vec![
                            weight_shape[2],
                            weight_shape[3],
                            c,
                            weight_shape[0] / c,
                        ]),
                        name_suffix: "hwicm",
                    }
                } else {
                    Kind {
                        target: TOSA_CONV2D,
                        pre: &NHWC,
                        post: &NCHW,
                        weight_dims: &NHWC,
                        weight_reshape: None,
                        name_suffix: "ohwi",
                    }
                };
                let params = vec![
                    i64_list_arg(&stride),
                    i64_list_arg(&pad),
                    i64_list_arg(&dilation),
                ];
                (kind, params, None)
            }
        }
    };

    // RCP:361-364: permute, reshape, `.contiguous()`; the bytes keep their
    // dtype, and the output channels stay outermost (OHWI) or C*M ordered
    // (HWCM), so per-channel qparams still line up. A captured weight gets the
    // same permute and reshape as runtime ops.
    let shape_error = |error: crate::layout::shape::ShapeError| error.to_string();
    let weight_dims = to_i64(kind.weight_dims);
    let permuted_weight_shape = permute_shape(&weight_shape, &weight_dims).map_err(shape_error)?;
    if let Some(shape) = &kind.weight_reshape {
        if shape.iter().product::<i64>() != permuted_weight_shape.iter().product::<i64>() {
            return Err(format!(
                "weight {weight_name} of shape {weight_shape:?} cannot take the shape {shape:?}"
            ));
        }
    }
    let tosa_weight_shape = kind
        .weight_reshape
        .clone()
        .unwrap_or_else(|| permuted_weight_shape.clone());

    // The TOSA op's value (RCP:1152-1160): rc4's fake kernels.
    let pre_dims = to_i64(kind.pre);
    let permuted_shape = permute_shape(&input.shape, &pre_dims).map_err(shape_error)?;
    let tosa_input = input_view_shape.as_deref().unwrap_or(&permuted_shape);
    let list = |index: usize| params[index].as_int_list().expect("an int list");
    let tosa_shape = match kind.target {
        TOSA_TRANSPOSE_CONV2D => {
            transpose_conv2d_shape(tosa_input, &tosa_weight_shape, &list(0), &list(1))
        }
        TOSA_DEPTHWISE_CONV2D => {
            depthwise_conv2d_shape(tosa_input, &tosa_weight_shape, &list(0), &list(1), &list(2))
        }
        TOSA_CONV3D => conv3d_shape(tosa_input, &tosa_weight_shape, &list(0), &list(1), &list(2)),
        _ => conv2d_shape(tosa_input, &tosa_weight_shape, &list(0), &list(1), &list(2)),
    }
    .map_err(shape_error)?;

    // Step 5: the output conversion (RCP:818-877).
    let rescale = if quantized {
        let qparams = input_qparams.as_ref().expect("a quantized conv's qparams");
        let (input_scale, _) = per_tensor(
            qparams
                .get(&0)
                .ok_or("the quantized conv has no activation qparam")?,
            "activation",
        )?;
        let weight = qparams
            .get(&1)
            .ok_or("the quantized conv has no weight qparam")?;
        let output_qparam = effective_output_qparams(graph, node)
            .and_then(|qparams| qparams.get(&0))
            .ok_or("RewriteConvPass: No output quantization parameter found")?;
        let (output_scale, output_zp) = per_tensor(output_qparam, "output")?;
        if output_qparam.dtype_str.is_empty() {
            return Err("the output qparam has no dtype".to_string());
        }
        let weight_scales = if weight.is_per_channel() {
            weight.per_channel_scales.clone()
        } else {
            vec![weight.scale]
        };
        // RCP:481-488, with exactly this association. rc4 zips the weight's
        // scales with the cycled input and output scales, so the RESCALE gets
        // one scale per weight scale, which need not be one per output channel:
        // rc4's quantizer leaves an int4 transposed conv's per-channel weight on
        // IOHW axis 0 (`_adjust_weight_qspec_for_conv_transpose`,
        // `quantization_annotator.py:136-138`), so its scales are per input
        // channel. rc4 emits them as they are, and TOSA's reference model
        // rejects the RESCALE (rc4 xfails those op tests,
        // `test_transpose_conv2d.py:497-501`).
        let scales = weight_scales
            .iter()
            .map(|&weight_scale| (input_scale * weight_scale) / output_scale)
            .collect();
        Some(RescaleRewrite {
            dtype: output_qparam.dtype_str.clone(),
            scales,
            output_zp,
        })
    } else {
        None
    };
    let a16w8 = quantized && int16_activation;

    // Steps 6-7 (RCP:1200-1232) and critique B10: the chain must give back
    // the conv's own value.
    let squeeze = (spatial_rank == 1).then(|| {
        let mut shape = tosa_shape.clone();
        shape.remove(1);
        shape
    });
    let post_dims = to_i64(kind.post);
    let output_shape = permute_shape(squeeze.as_deref().unwrap_or(&tosa_shape), &post_dims)
        .map_err(shape_error)?;
    if output_shape != output.shape {
        return Err(format!(
            "the rewrite would give {output_shape:?}, not the convolution's {:?}",
            output.shape
        ));
    }
    if let Some(reason) = control_flow_hazard(graph, args.input) {
        return Err(reason);
    }
    // Step 8 (RCP:1234-1249): an A16W8 conv1d narrows to int16 instead of
    // branching.
    let a16w8_branches = if a16w8 && spatial_rank != 1 {
        let rescale = rescale.as_ref().expect("an A16W8 conv is quantized");
        Some(plan_int32_branches(
            graph,
            order,
            conv,
            rescale,
            &output.shape,
        )?)
    } else {
        None
    };

    let weight = match weight_operand {
        WeightOperand::Constant(weight_node) => {
            let data = &weight_node
                .tensor
                .as_ref()
                .expect("a constant carries its data")
                .data;
            let mut rewritten = permute_tensor_data(data, kind.weight_dims)
                .ok_or_else(|| format!("cannot permute weight {weight_name}"))?;
            rewritten.shape = tosa_weight_shape;
            WeightRewrite::Constant {
                source: args.weight,
                name_suffix: kind.name_suffix,
                kind: constant_kind(weight_node),
                data: rewritten,
                special: weight_node.meta.extras.get(SPECIAL_DTYPE_KEY).cloned(),
            }
        }
        // Each layout copy takes its producer's dtype (`layout::shape`).
        WeightOperand::Captured(value) => WeightRewrite::Runtime {
            source: args.weight,
            name_suffix: kind.name_suffix,
            dims: weight_dims,
            permuted: TensorFacts::new(permuted_weight_shape, value.dtype.clone()),
            reshaped: kind
                .weight_reshape
                .map(|shape| TensorFacts::new(shape, value.dtype.clone())),
        },
    };

    Ok(Some(ConvRewrite {
        conv,
        input: args.input,
        int16_activation,
        weight,
        bias,
        permuted_input: TensorFacts::new(permuted_shape, input.dtype.clone()),
        input_view: input_view_shape.map(|shape| TensorFacts::new(shape, input.dtype.clone())),
        pre_dims,
        target: kind.target,
        params,
        tosa: TensorFacts::new(tosa_shape, tosa_dtype),
        int48: a16w8,
        rescale,
        squeeze,
        post_dims,
        output_shape,
        a16w8_branches,
        source,
    }))
}

/// Names for the created call nodes, `{c}_<suffix>` (critique B12), so none
/// looks like a sibling aten conv and none reuses a name an effect may hold
/// (design F9; `LayoutCtx::unique_name` adds `_N` to a taken one).
fn layout_names(rank: usize) -> (&'static str, &'static str) {
    match rank {
        3 => ("nlc", "ncl"),
        5 => ("ndhwc", "ncdhw"),
        _ => ("nhwc", "nchw"),
    }
}

/// Make the graph edits `plan` computed (RCP:919-1271), in rc4's positions.
/// Returns the TOSA convolution of an A16W8 conv of spatial rank 2 or 3,
/// which RCP:1267 records for the U55 step.
fn apply(ctx: &mut LayoutCtx<'_>, rewrite: ConvRewrite) -> Option<NodeId> {
    let ConvRewrite {
        conv,
        input,
        source,
        int16_activation,
        weight,
        bias,
        pre_dims,
        permuted_input,
        input_view,
        target,
        params,
        tosa,
        int48,
        rescale,
        squeeze,
        post_dims,
        output_shape,
        a16w8_branches,
    } = rewrite;
    let name = ctx.node(conv).name.clone();
    let (input_layout, output_layout) = layout_names(pre_dims.len());

    // The bias (RCP:919-933, :414-418): zeros before the first placeholder,
    // the only position from which constants stay ahead of every user input.
    let (bias, int48_bias) = match bias {
        BiasRewrite::Keep(bias) => (bias, int16_activation),
        BiasRewrite::Zeros(data) => {
            let first_placeholder = ctx
                .order()
                .iter()
                .find(|&id| ctx.node(id).op == NodeOp::Placeholder);
            let point = first_placeholder.map_or(InsertPoint::Start, InsertPoint::Before);
            let zeros = ctx.with_insert_point(point, |ctx| {
                ctx.create_constant(format!("{name}_bias"), LoweringInputKind::Parameter, data)
            });
            (zeros, int16_activation)
        }
        // captured_conv marks the int32 bias it makes (RCP:323-325).
        BiasRewrite::Captured(bias) => (bias, false),
    };
    if int48_bias {
        set_special_dtype(ctx.meta_mut(bias), SpecialDtype::Int48);
    }

    // The input permute, and conv1d's view, directly before the conv.
    let tosa_input = ctx.inserting_before(conv, |ctx| {
        let permuted = ctx.call_function(
            format!("{name}_{input_layout}"),
            PERMUTE,
            vec![IrArg::Node(input), i64_list_arg(&pre_dims)],
            meta::from_node(&source, permuted_input),
        );
        match input_view {
            None => permuted,
            Some(view) => {
                let size = i64_list_arg(&view.shape);
                ctx.call_function(
                    format!("{name}_nhwc"),
                    VIEW,
                    vec![IrArg::Node(permuted), size],
                    meta::from_node(&source, view),
                )
            }
        }
    });

    // The pre-permuted weight, directly after its source (RCP:368-378), or a
    // captured weight's runtime layout ops there.
    let weight_operand = match weight {
        WeightRewrite::Constant {
            source: weight_source,
            name_suffix,
            kind,
            data,
            special,
        } => {
            let constant = ctx.inserting_after(weight_source, |ctx| {
                ctx.create_constant(format!("{name}_weight_{name_suffix}"), kind, data)
            });
            if let Some(special) = special {
                ctx.meta_mut(constant).set_extra(SPECIAL_DTYPE_KEY, special);
            }
            constant
        }
        WeightRewrite::Runtime {
            source: weight_source,
            name_suffix,
            dims,
            permuted,
            reshaped,
        } => {
            let last = format!("{name}_weight_{name_suffix}");
            let permute_name = if reshaped.is_some() {
                format!("{name}_weight_permute")
            } else {
                last.clone()
            };
            let permute = ctx.inserting_after(weight_source, |ctx| {
                ctx.call_function(
                    permute_name,
                    PERMUTE,
                    vec![IrArg::Node(weight_source), i64_list_arg(&dims)],
                    meta::from_node(&source, permuted),
                )
            });
            match reshaped {
                None => permute,
                Some(view) => {
                    let size = i64_list_arg(&view.shape);
                    ctx.inserting_after(permute, |ctx| {
                        ctx.call_function(
                            last,
                            VIEW,
                            vec![IrArg::Node(permute), size],
                            meta::from_node(&source, view),
                        )
                    })
                }
            }
        }
    };

    // The TOSA op directly after the conv, inheriting its qparams
    // (RCP:1175-1183), then the RESCALE, conv1d's view and the output permute,
    // each directly after the one before.
    let mut tosa_meta = meta::inheriting_qparams(&source, tosa.clone());
    if int48 {
        set_special_dtype(&mut tosa_meta, SpecialDtype::Int48);
    }
    let mut args = vec![
        IrArg::Node(tosa_input),
        IrArg::Node(weight_operand),
        IrArg::Node(bias),
    ];
    args.extend(params);
    let tosa_op = ctx.inserting_after(conv, |ctx| {
        ctx.call_function(format!("{name}_tosa"), target, args, tosa_meta)
    });
    let conversion = match rescale {
        None => tosa_op,
        Some(RescaleRewrite {
            dtype,
            scales,
            output_zp,
        }) => {
            let val = TensorFacts::new(tosa.shape.clone(), dtype.clone());
            ctx.inserting_after(tosa_op, |ctx| {
                ctx.call_function(
                    format!("{name}_rescale"),
                    RESCALE,
                    vec![
                        IrArg::Node(tosa_op),
                        IrArg::Dtype(dtype),
                        IrArg::List(scales.into_iter().map(IrArg::Float).collect()),
                        IrArg::Int(0),
                        IrArg::Int(output_zp),
                    ],
                    meta::from_node(&source, val),
                )
            })
        }
    };
    // RCP:1204-1214, :1226-1229: each layout copy takes its producer's dtype
    // and special dtype.
    let copy_meta = |ctx: &LayoutCtx<'_>, producer: NodeId, shape: Vec<i64>| {
        let producer = &ctx.node(producer).meta;
        let dtype = producer
            .val_dtype()
            .expect("a created node has a value")
            .to_owned();
        let mut created = meta::from_node(&source, TensorFacts::new(shape, dtype));
        crate::layout::dtype::copy_special_dtype(producer, &mut created);
        created
    };
    let squeeze_view = squeeze.map(|shape| {
        let created = copy_meta(ctx, conversion, shape.clone());
        ctx.inserting_after(conversion, |ctx| {
            ctx.call_function(
                format!("{name}_squeeze"),
                VIEW,
                vec![IrArg::Node(conversion), i64_list_arg(&shape)],
                created,
            )
        })
    });
    let post_input = squeeze_view.unwrap_or(conversion);
    let created = copy_meta(ctx, post_input, output_shape.clone());
    let output = ctx.inserting_after(post_input, |ctx| {
        ctx.call_function(
            format!("{name}_{output_layout}"),
            PERMUTE,
            vec![IrArg::Node(post_input), i64_list_arg(&post_dims)],
            created,
        )
    });

    // RCP:1234-1269.
    let recorded = match a16w8_branches {
        None => {
            ctx.replace_all_uses_with(conv, output);
            None
        }
        Some(branches) => {
            let layout = OutputLayout {
                name: &name,
                suffix: output_layout,
                dims: &post_dims,
                accumulator_shape: &tosa.shape,
                conv_shape: &output_shape,
            };
            insert_a16w8_output_branches(ctx, &source, tosa_op, &layout, branches);
            // The users left read the int16 path; without any, it goes.
            if ctx.graph().has_live_users(conv) {
                ctx.replace_all_uses_with(conv, output);
            } else {
                ctx.erase_node(output);
                if let Some(view) = squeeze_view {
                    ctx.erase_node(view);
                }
                ctx.erase_node(conversion);
            }
            Some(tosa_op)
        }
    };
    // RCP:1271. The source weight stays, as rc4 leaves it.
    ctx.erase_node(conv);
    recorded
}

/// What the A16W8 branches' layout permutes need: `_insert_layout_permute`'s
/// `post_permute_dims` (RCP:564-581) and the names and values around them.
struct OutputLayout<'a> {
    /// The conv's name, which created names extend (critique B12).
    name: &'a str,
    /// The output permute's name suffix (`nchw`, `ncdhw`).
    suffix: &'a str,
    dims: &'a [i64],
    /// The TOSA convolution's shape, which an INT32 RESCALE of it keeps.
    accumulator_shape: &'a [i64],
    /// The conv's shape, which a layout permute gives back.
    conv_shape: &'a [i64],
}

/// `_insert_a16w8_output_branches` (RCP:665-816): move the INT32 consumers of
/// the conv whose rewrite created `tosa_op` onto widened branches of the
/// accumulator, as `plan_int32_branches` planned them. Every created node is
/// `create_node(from_node=...)`'s, with emptied qparam maps: the RESCALE and
/// permutes from the conv (`source`), a permute copy from the permute it
/// copies.
fn insert_a16w8_output_branches(
    ctx: &mut LayoutCtx<'_>,
    source: &NodeMeta,
    tosa_op: NodeId,
    layout: &OutputLayout<'_>,
    branches: Int32Branches,
) {
    let Int32Branches {
        direct,
        permuted,
        conv_scales,
        conv_output_zp,
    } = branches;
    // RESCALE's fake kernel keeps its input's shape; a layout permute gives
    // back the conv's (`permute_fake_tensor_metadata`).
    let accumulator_val = || TensorFacts::new(layout.accumulator_shape.to_vec(), INT32);
    let layout_val = || TensorFacts::new(layout.conv_shape.to_vec(), INT32);
    let layout_permute = |ctx: &mut LayoutCtx<'_>, input: NodeId, name: String| {
        ctx.inserting_after(input, |ctx| {
            ctx.call_function(
                name,
                PERMUTE,
                vec![IrArg::Node(input), i64_list_arg(layout.dims)],
                meta::from_node(source, layout_val()),
            )
        })
    };

    // RCP:690-748: retarget each direct INT32 user to the accumulator with
    // the combined scales, where it stands, and give its users the conv's
    // layout back through a permute directly after it. The users are taken
    // before the permute exists, which reads the RESCALE too.
    for user in direct {
        let consumers = ctx.graph().live_user_ids(user.rescale);
        ctx.set_args(
            user.rescale,
            vec![
                IrArg::Node(tosa_op),
                IrArg::Dtype(INT32.to_string()),
                IrArg::List(user.scales.into_iter().map(IrArg::Float).collect()),
                IrArg::Int(0),
                IrArg::Int(user.output_zp),
            ],
        );
        meta::assign_val(ctx.meta_mut(user.rescale), accumulator_val());
        let name = format!("{}_{}", ctx.node(user.rescale).name, layout.suffix);
        let permute = layout_permute(ctx, user.rescale, name);
        for consumer in consumers {
            ctx.replace_input_with(consumer, user.rescale, permute);
        }
    }

    // RCP:750-816: an INT32 path behind a permute user rescales the
    // accumulator to the conv's domain in int32, directly after it, and
    // repeats the user permute on that, since the permute may move the
    // per-channel axis the two RESCALEs' scales follow.
    for user in permuted {
        let widened = ctx.inserting_after(tosa_op, |ctx| {
            ctx.call_function(
                format!("{}_rescale_int32", layout.name),
                RESCALE,
                vec![
                    IrArg::Node(tosa_op),
                    IrArg::Dtype(INT32.to_string()),
                    IrArg::List(conv_scales.iter().copied().map(IrArg::Float).collect()),
                    IrArg::Int(0),
                    IrArg::Int(conv_output_zp),
                ],
                meta::from_node(source, accumulator_val()),
            )
        });
        let widened_output = layout_permute(
            ctx,
            widened,
            format!("{}_rescale_int32_{}", layout.name, layout.suffix),
        );
        let permute = ctx.node(user.permute);
        let name = format!("{}_int32", permute.name);
        let dims = permute.args[1].clone();
        let created = meta::from_node(&permute.meta, TensorFacts::new(user.shape, INT32));
        let widened_permute = ctx.inserting_after(widened_output, |ctx| {
            ctx.call_function(
                name,
                PERMUTE,
                vec![IrArg::Node(widened_output), dims],
                created,
            )
        });
        for rescale in user.rescales {
            ctx.replace_input_with(rescale, user.permute, widened_permute);
        }
        if !ctx.graph().has_live_users(user.permute) {
            ctx.erase_node(user.permute);
        }
    }
}

/// `_deduplicate_a16w8_output_rescales` (RCP:583-631) on accumulator
/// `tosa_op`, with `node_order` the graph order taken before the U55 step:
/// merge each RESCALE-to-permute head into the earliest head with the same
/// signatures, and return the RESCALE users left, in graph order. `None`, with
/// nothing changed, when a user is missing from `node_order` or is not a
/// RESCALE, a shape rc4's rewrite never makes.
///
/// A head is a RESCALE whose only user is a permute; any other RESCALE stays
/// as it is. Two heads merge when their RESCALEs and their permutes sign alike
/// apart from their inputs (`build_node_signature(..., positional_arg_start=1)`:
/// target, the remaining args, trailing positional flags included, and the
/// kwargs): the later permute's users move to the earlier one, which comes
/// directly after its RESCALE and so before them, and the later head goes.
pub(crate) fn deduplicate_a16w8_output_rescales(
    ctx: &mut LayoutCtx<'_>,
    tosa_op: NodeId,
    node_order: &HashMap<NodeId, usize>,
) -> Option<Vec<NodeId>> {
    let mut rescale_users = ctx.graph().live_user_ids(tosa_op);
    if rescale_users
        .iter()
        .any(|user| !node_order.contains_key(user))
    {
        return None;
    }
    rescale_users.sort_by_key(|user| node_order[user]);
    if rescale_users
        .iter()
        .any(|&user| !ctx.node(user).is_rescale())
    {
        return None;
    }
    let mut unique: HashMap<(String, String), NodeId> = HashMap::new();
    let mut deduplicated = Vec::new();
    for rescale in rescale_users {
        let layout_permute = match ctx.graph().live_user_ids(rescale)[..] {
            [output] if is_permute(ctx.node(output)) => output,
            _ => {
                deduplicated.push(rescale);
                continue;
            }
        };
        let signature = (
            node_signature_from(ctx.graph(), rescale, 1),
            node_signature_from(ctx.graph(), layout_permute, 1),
        );
        match unique.get(&signature) {
            Some(&canonical) => {
                ctx.replace_all_uses_with(layout_permute, canonical);
                ctx.erase_node(layout_permute);
                ctx.erase_node(rescale);
            }
            None => {
                unique.insert(signature, layout_permute);
                deduplicated.push(rescale);
            }
        }
    }
    Some(deduplicated)
}

/// `_separate_u55_a16w8_output_rescales` (RCP:633-663): give every RESCALE
/// user of accumulator `tosa_op` after the first its own copy of the TOSA
/// convolution, once duplicate heads are merged
/// ([`deduplicate_a16w8_output_rescales`]). A copy is `create_node(...,
/// from_node=tosa_op, inherit_qparams=True)`: the same target, args and
/// kwargs, and the accumulator's meta (qparams, INT48), inserted directly
/// before its RESCALE. The accumulator and its copies are marked
/// `do_not_fuse_duplicate`.
pub(crate) fn separate_u55_a16w8_output_rescales(
    ctx: &mut LayoutCtx<'_>,
    tosa_op: NodeId,
    node_order: &HashMap<NodeId, usize>,
) {
    if ctx.graph().live_user_ids(tosa_op).len() < 2 {
        return;
    }
    let Some(rescale_users) = deduplicate_a16w8_output_rescales(ctx, tosa_op, node_order) else {
        return;
    };
    if rescale_users.len() < 2 {
        return;
    }
    ctx.meta_mut(tosa_op)
        .set_extra(DO_NOT_FUSE_DUPLICATE, MetaValue::Bool(true));
    for &rescale in &rescale_users[1..] {
        let node = ctx.node(tosa_op);
        let name = format!("{}_clone", node.name);
        let target = node.target.clone();
        let args = node.args.clone();
        let kwargs = node.kwargs.clone();
        let val = node
            .meta
            .tensor_facts()
            .cloned()
            .expect("a TOSA convolution has a value");
        let mut created = meta::inheriting_qparams(&node.meta, val);
        created.set_extra(DO_NOT_FUSE_DUPLICATE, MetaValue::Bool(true));
        let clone = ctx.inserting_before(rescale, |ctx| {
            ctx.create_node(NodeOp::CallFunction, name, target, args, kwargs, created)
        });
        ctx.replace_input_with(rescale, tosa_op, clone);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// rc4's static `_adjust_pad_if_needed` vectors
    /// (`test_rewrite_conv_pass.py:1002-1027`).
    #[test]
    fn adjust_pad_if_needed_matches_rc4s_static_tests() {
        // test_rewrite_conv_adjust_pad_if_needed_static_allows_negative_padding_until_later_validation
        let error = adjust_pad_if_needed(6, 2, 3, 0, 1).unwrap_err();
        assert!(error.contains("SizeAdjustInputPass"), "{error}");
        // ..._static_positive_padding_stays_non_negative
        assert_eq!(adjust_pad_if_needed(8, 2, 3, 2, 1), Ok(1));
        // ..._static_exact_remainder_matches_pad
        assert_eq!(adjust_pad_if_needed(6, 1, 3, 1, 1), Ok(0));
    }

    /// The design's pad table (§2.3, A1 §7.1): k3 s2 p1 on 8 trims the
    /// trailing pad, k3 s3 p2 on 10 and 11 trims one axis only, and a dilated
    /// or exact conv keeps both pads. A numerator below zero (an input shorter
    /// than the receptive field) takes Python's non-negative modulo.
    #[test]
    fn adjust_pad_if_needed_follows_pythons_modulo() {
        assert_eq!(adjust_pad_if_needed(8, 3, 2, 1, 1), Ok(0));
        assert_eq!(adjust_pad_if_needed(10, 3, 3, 2, 1), Ok(0));
        assert_eq!(adjust_pad_if_needed(11, 3, 3, 2, 1), Ok(2));
        assert_eq!(adjust_pad_if_needed(8, 3, 1, 2, 2), Ok(2));
        assert_eq!(adjust_pad_if_needed(9, 3, 2, 1, 1), Ok(1));
        // (1 + 0 - 2 - 1) = -2; -2 % 3 == 1 in Python.
        assert!(adjust_pad_if_needed(1, 3, 3, 0, 1).is_err());
        assert_eq!(adjust_pad_if_needed(1, 3, 2, 0, 1), Ok(0));
        assert!(adjust_pad_if_needed(8, 3, 0, 1, 1).is_err());
    }

    /// `_is_depthwise_conv`: Cin == groups dividing Co, for rank 3 and 4 only.
    #[test]
    fn depthwise_follows_rc4s_rule() {
        assert!(is_depthwise(&[1, 4, 8, 8], 4, 8));
        assert!(is_depthwise(&[1, 1, 8, 8], 1, 4), "Cin = 1 counts");
        assert!(is_depthwise(&[1, 1, 9], 1, 4), "conv1d too");
        assert!(!is_depthwise(&[1, 1, 4, 6, 6], 1, 4), "rank 5 never");
        assert!(!is_depthwise(&[1, 4, 8, 8], 2, 6), "grouped");
        assert!(!is_depthwise(&[1, 4, 8, 8], 4, 6), "Co % Cin");
        assert!(!is_depthwise(&[1, 3, 8, 8], 1, 4));
    }
}
