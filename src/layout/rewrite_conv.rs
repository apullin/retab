//! Explicit mode's port of rc4's RewriteConvPass
//! (`backends/arm/_passes/rewrite_conv_pass.py`, "RCP" below with its line
//! numbers), APU-135 P2 S3 (APU-155).
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
//! constant placeholder there; explicit mode runs `IrMaterializeConstantsPass`
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
//! explicit-mode guard hands the graph to stock.
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
//! f64, one per output channel for a per-channel weight; the output qparams
//! are the conv's own, else those of a sole `view_copy` or `clamp` user
//! (RCP:420-450). An int16 activation (A16W8) marks the TOSA convolution and
//! its bias INT48 (RCP:414-418, :856).
//!
//! **Not here yet.** A16W8's INT32 branching and U55 separation (design §2.8)
//! are S4's: an A16W8 conv of spatial rank 2 or 3 that feeds an INT32 RESCALE,
//! directly or through a permute, requests a stock fallback until then, and
//! without such consumers no accumulator gets a second user, so the U55 step
//! has nothing to separate. Control flow is S8's (design §2.9, §4.3): RETAB
//! applies the boundary contracts at serialization, and two of their qparam
//! lookups misread the qparam-less nodes rc4's rewrite creates (see
//! `control_flow_hazard`), so every region conv requests a stock fallback, and
//! so does a parent conv reading a cond or while output whose getitem has no
//! output qparams. The region rules S8 builds on are in place: constants are
//! `get_attr` nodes (`GraphScope::Region`), and a weight or bias captured from
//! the parent (a placeholder) is reported as such. FP8 is out of scope (design
//! F1).
//!
//! **Stock fallbacks** (design §2.11, critique B5, B10). Where rc4 raises, the
//! pass requests a stock fallback (`stock_fallback::request`) and stops, so
//! stock reports its own error: a transposed conv that is not 2-D or is
//! dilated (RCP:939-947), a grouped CONV3D (RCP:147-158), a stride remainder
//! larger than the padding (RCP:126-129), a weight that is not a constant
//! (RCP:243-251), a quantized conv whose bias is not int32 or whose weight is
//! not int8, or whose input or output qparams are missing or per-channel
//! (RCP:434-438, :471-474), and dtypes rc4's TOSA kernels reject
//! (`tosa/dialect/ops/conv2d.py:17-88`). So does anything the port cannot
//! rewrite faithfully: a grouped conv `IrDecomposeGroupedConv` declined (B5),
//! the raw `aten.conv{1,2,3}d`/`conv_transpose2d` forms (design §2.2), and a
//! conv whose rewritten output shape would not be its own (B10). Each conv is
//! planned before the graph changes, so the convs before the fallback stay
//! rewritten and the one that requested it stays as it was.

use crate::fx_constant_tensor_eval::permute_tensor_data;
use crate::fx_conv_args::{ConvArgKind, ConvArgs};
use crate::fx_ir::{
    i64_list_arg, IrArg, IrGraph, IrNode, IrPass, MetaValue, NodeId, NodeMeta, NodeOp, QParam,
    QParamMap,
};
use crate::fx_meta::TensorFacts;
use crate::graph::ir::DType;
use crate::layout::constants::{constant_kind, is_constant};
use crate::layout::ctx::{GraphScope, InsertPoint, LayoutCtx};
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

/// rc4's RewriteConvPass for explicit layout mode; see the module docs.
pub(crate) struct IrRewriteConvPass {
    /// `get_context_spec().is_U55_subset` (RCP:1273): whether A16W8
    /// accumulators get one TOSA convolution per RESCALE user. Only the INT32
    /// branching gives an accumulator a second user, and that is S4's (design
    /// §2.8); until then the flag has nothing to act on.
    pub(crate) is_u55_subset: bool,
    /// The pipeline's region flag: constants are `get_attr` nodes
    /// (`GraphScope::Region`). Until S8 every region conv falls back to stock
    /// (design §2.9; see `control_flow_hazard`).
    pub(crate) in_region: bool,
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
                Some(ConvForm::Convolution) => plan(ctx.graph(), scope, id),
            };
            match planned {
                Ok(Some(rewrite)) => {
                    apply(&mut ctx, rewrite);
                    changed = true;
                }
                Ok(None) => {}
                Err(reason) => {
                    let name = &ctx.node(id).name;
                    crate::stock_fallback::request(format!("IrRewriteConv: {name}: {reason}"));
                    break;
                }
            }
        }
        // RCP:1273-1280 (U55 separation) acts on A16W8 accumulators with two or
        // more users, which only S4's branching creates; see `is_u55_subset`.
        if changed {
            ctx.finish();
        }
        changed
    }
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

/// Whether the conv feeds an INT32 RESCALE directly or through a permute
/// (`_get_direct_int32_rescale_users`, `_get_permute_int32_rescale_users`,
/// RCP:543-561): the consumers A16W8's branching keeps on an INT32 path.
fn has_int32_consumers(graph: &IrGraph, conv: NodeId) -> bool {
    graph.live_user_ids(conv).into_iter().any(|user| {
        let node = graph.node(user);
        is_direct_int32_rescale(node)
            || (node.op == NodeOp::CallFunction
                && node.target == PERMUTE
                && graph
                    .live_user_ids(user)
                    .into_iter()
                    .any(|consumer| is_direct_int32_rescale(graph.node(consumer))))
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

/// Why a conv that `plan` could rewrite must still go to stock until S8 makes
/// control flow explicit-aware (design §2.9, §4.3). rc4 inserts its
/// control-flow boundary RESCALEs before RewriteConvPass (APM:595); RETAB's
/// boundary contracts run at serialization, after the whole pipeline
/// (`control_flow::contracts::apply_boundary_contracts`), and two of their
/// qparam lookups misread the nodes this pass creates, which carry no qparams
/// as rc4's do:
///
/// - A region output's own qparam comes from `extract_effective_qparam_ir`,
///   which walks a qparam-less output permute to the RESCALE and reverse
///   engineers it (`reverse_engineer_rescale_ir`) as a rescale of the TOSA
///   conv's inherited output qparam: for a per-tensor weight it computes
///   `s_out^2 / (s_in * s_w)`, not `s_out`, and the output contract inserts a
///   wrong RESCALE (cumsum's per-tensor kernel in a cond branch is one). So in
///   a region every conv falls back, after the reasons specific to regions
///   (captured operands).
/// - A cond or while output's target qparam comes from its getitem's output
///   qparams, else from the input qparams of the getitem's direct consumers
///   (`infer_output_qparams_map_ir`); the conv was that consumer, its input
///   permute has none, and the region's output RESCALE would be lost.
fn control_flow_hazard(graph: &IrGraph, scope: GraphScope, input: NodeId) -> Option<String> {
    if scope == GraphScope::Region {
        return Some(
            "region convs wait for the explicit-mode boundary contracts and captured_conv \
             (APU-135 S8): the output contract misreads the rewrite's qparam-less RESCALE"
                .to_string(),
        );
    }
    let node = graph.node(input);
    let reads_control_flow = node.is_getitem()
        && node.meta.output_qparam().is_none()
        && node
            .first_node_arg()
            .is_some_and(|source| graph.node(source).is_control_flow());
    reads_control_flow.then(|| {
        format!(
            "it reads control-flow output {}, whose boundary contract takes its qparam from the \
             conv, which the rewrite replaces with a qparam-less permute (APU-135 S8)",
            node.name
        )
    })
}

/// The conv's weight, as RCP resolves it (`_resolve_control_flow_constant`,
/// RCP:223-261, without control flow). In the parent it must be a constant
/// placeholder: rc4 raises for anything else, including a `get_attr`
/// (`get_constant_placeholder_kind`). In a region only a `get_attr` constant
/// survives ingest as a constant; a placeholder is the parent's operand.
fn constant_weight(graph: &IrGraph, scope: GraphScope, weight: NodeId) -> Result<&IrNode, String> {
    let node = graph.node(weight);
    let expected = match scope {
        GraphScope::Parent => NodeOp::Placeholder,
        GraphScope::Region => NodeOp::GetAttr,
    };
    if is_constant(node) && node.op == expected {
        return Ok(node);
    }
    Err(match scope {
        GraphScope::Parent => format!(
            "weight {} is not a constant placeholder, which RewriteConvPass requires",
            node.name
        ),
        GraphScope::Region => format!(
            "weight {} is captured from the parent graph; region convs with captured weights \
             need runtime weight layout ops and the explicit captured_conv (APU-135 S8)",
            node.name
        ),
    })
}

/// The pre-permuted weight constant (RCP:328-403).
struct WeightRewrite {
    /// The source weight, which the new constant follows.
    source: NodeId,
    name_suffix: &'static str,
    kind: LoweringInputKind,
    data: TensorData,
    /// The source's `tosa_special_dtype` (RCP:377-378).
    special: Option<MetaValue>,
}

enum BiasRewrite {
    /// The conv's own bias.
    Keep(NodeId),
    /// `_add_bias`'s zeros (RCP:160-195).
    Zeros(TensorData),
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
    /// RCP:1234-1267: an A16W8 conv of spatial rank 2 or 3.
    a16w8_branching: bool,
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
fn plan(graph: &IrGraph, scope: GraphScope, conv: NodeId) -> Result<Option<ConvRewrite>, String> {
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
    let weight_node = constant_weight(graph, scope, args.weight)?;
    let weight_data = &weight_node
        .tensor
        .as_ref()
        .expect("a constant carries its data")
        .data;
    if weight_data.shape.len() != rank
        || weight_node
            .meta
            .val_shape()
            .is_some_and(|shape| shape != weight_data.shape.as_slice())
    {
        return Err(format!(
            "weight {} of shape {:?} does not fit a rank-{rank} convolution",
            weight_node.name, weight_data.shape
        ));
    }
    let weight_shape = weight_data.shape.clone();

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
    let weight_dtype = weight_data.dtype.torch_str();
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
        Some(bias) => {
            let bias_node = graph.node(bias);
            if scope == GraphScope::Region
                && !(is_constant(bias_node) && bias_node.op == NodeOp::GetAttr)
            {
                return Err(format!(
                    "bias {} is captured from the parent graph; region conv biases need the \
                     explicit captured_conv (APU-135 S8)",
                    bias_node.name
                ));
            }
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
    // (HWCM), so per-channel qparams still line up.
    let mut rewritten = permute_tensor_data(weight_data, kind.weight_dims)
        .ok_or_else(|| format!("cannot permute weight {}", weight_node.name))?;
    if let Some(shape) = &kind.weight_reshape {
        if shape.iter().product::<i64>() != rewritten.shape.iter().product::<i64>() {
            return Err(format!(
                "weight {} of shape {weight_shape:?} cannot take the shape {shape:?}",
                weight_node.name
            ));
        }
        rewritten.shape.clone_from(shape);
    }

    // The TOSA op's value (RCP:1152-1160): rc4's fake kernels.
    let pre_dims = to_i64(kind.pre);
    let shape_error = |error: crate::layout::shape::ShapeError| error.to_string();
    let permuted_shape = permute_shape(&input.shape, &pre_dims).map_err(shape_error)?;
    let tosa_input = input_view_shape.as_deref().unwrap_or(&permuted_shape);
    let list = |index: usize| params[index].as_int_list().expect("an int list");
    let tosa_shape = match kind.target {
        TOSA_TRANSPOSE_CONV2D => {
            transpose_conv2d_shape(tosa_input, &rewritten.shape, &list(0), &list(1))
        }
        TOSA_DEPTHWISE_CONV2D => {
            depthwise_conv2d_shape(tosa_input, &rewritten.shape, &list(0), &list(1), &list(2))
        }
        TOSA_CONV3D => conv3d_shape(tosa_input, &rewritten.shape, &list(0), &list(1), &list(2)),
        _ => conv2d_shape(tosa_input, &rewritten.shape, &list(0), &list(1), &list(2)),
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
        // A per-channel RESCALE needs one scale per output channel.
        let channels = *tosa_shape.last().expect("a rank-4 or rank-5 accumulator");
        if weight_scales.len() != 1 && weight_scales.len() as i64 != channels {
            return Err(format!(
                "{} weight scales for {channels} output channels",
                weight_scales.len()
            ));
        }
        // RCP:481-488, with exactly this association.
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
    // RCP:1238: an A16W8 conv1d narrows to int16 instead of branching.
    let a16w8_branching = a16w8 && spatial_rank != 1;
    if a16w8_branching && has_int32_consumers(graph, conv) {
        return Err(
            "an A16W8 conv feeding INT32 consumers, whose INT48 branching (design §2.8) is \
             APU-135 S4's"
                .to_string(),
        );
    }

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
    if let Some(reason) = control_flow_hazard(graph, scope, args.input) {
        return Err(reason);
    }

    Ok(Some(ConvRewrite {
        conv,
        input: args.input,
        int16_activation,
        weight: WeightRewrite {
            source: args.weight,
            name_suffix: kind.name_suffix,
            kind: constant_kind(weight_node),
            data: rewritten,
            special: weight_node.meta.extras.get(SPECIAL_DTYPE_KEY).cloned(),
        },
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
        a16w8_branching,
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
fn apply(ctx: &mut LayoutCtx<'_>, rewrite: ConvRewrite) {
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
        a16w8_branching,
    } = rewrite;
    let name = ctx.node(conv).name.clone();
    let (input_layout, output_layout) = layout_names(pre_dims.len());

    // The bias (RCP:919-933, :414-418): zeros before the first placeholder,
    // the only position from which constants stay ahead of every user input.
    let bias = match bias {
        BiasRewrite::Keep(bias) => bias,
        BiasRewrite::Zeros(data) => {
            let first_placeholder = ctx
                .order()
                .iter()
                .find(|&id| ctx.node(id).op == NodeOp::Placeholder);
            let point = first_placeholder.map_or(InsertPoint::Start, InsertPoint::Before);
            ctx.with_insert_point(point, |ctx| {
                ctx.create_constant(format!("{name}_bias"), LoweringInputKind::Parameter, data)
            })
        }
    };
    if int16_activation {
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

    // The pre-permuted weight, directly after its source (RCP:368-378).
    let weight_constant = ctx.inserting_after(weight.source, |ctx| {
        ctx.create_constant(
            format!("{name}_weight_{}", weight.name_suffix),
            weight.kind,
            weight.data,
        )
    });
    if let Some(special) = weight.special {
        ctx.meta_mut(weight_constant)
            .set_extra(SPECIAL_DTYPE_KEY, special);
    }

    // The TOSA op directly after the conv, inheriting its qparams
    // (RCP:1175-1183), then the RESCALE, conv1d's view and the output permute,
    // each directly after the one before.
    let mut tosa_meta = meta::inheriting_qparams(&source, tosa.clone());
    if int48 {
        set_special_dtype(&mut tosa_meta, SpecialDtype::Int48);
    }
    let mut args = vec![
        IrArg::Node(tosa_input),
        IrArg::Node(weight_constant),
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
    let created = copy_meta(ctx, post_input, output_shape);
    let output = ctx.inserting_after(post_input, |ctx| {
        ctx.call_function(
            format!("{name}_{output_layout}"),
            PERMUTE,
            vec![IrArg::Node(post_input), i64_list_arg(&post_dims)],
            created,
        )
    });

    // RCP:1234-1269. Planning sent an A16W8 conv with INT32 consumers to
    // stock, so S4's branching has nothing to do here yet; a conv left
    // without users then drops its INT16 path (RCP:1260-1266).
    if a16w8_branching && !ctx.graph().has_live_users(conv) {
        ctx.erase_node(output);
        if let Some(view) = squeeze_view {
            ctx.erase_node(view);
        }
        ctx.erase_node(conversion);
    } else {
        ctx.replace_all_uses_with(conv, output);
    }
    // RCP:1271. The source weight stays, as rc4 leaves it.
    ctx.erase_node(conv);
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
