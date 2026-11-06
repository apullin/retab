//! Port of the control-flow part of upstream RewriteConvPass (ExecuTorch v1.6.0-rc4):
//! `_resolve_control_flow_constant`, `_rewrite_control_flow_bias` and the
//! control-flow case of `_rewrite_weight`.
//!
//! A conv weight or bias that a cond branch captures from the enclosing graph arrives
//! as a branch input, requantized at the boundary like an activation. Both are traced
//! back through the boundary RESCALEs and branch inputs, across nested conds, to the
//! constant in the outermost graph, and replaced with branch-local constants:
//!
//! - The bias is dequantized at the scale the RESCALE chain implies and requantized
//!   into the INT32 bias at input_scale * weight_scale that TOSA CONV2D needs.
//! - The weight becomes a constant in the conv's weight layout, followed by the same
//!   RESCALE chain. RESCALE is elementwise, so it commutes with the layout change, and
//!   the weight needs no runtime RESHAPE or TRANSPOSE.
//!
//! This runs after the boundary contracts have created those RESCALEs. The branch
//! inputs stay, unused, so the branch keeps the cond's operand list.
//!
//! `rewrite_explicit_captured_conv_operands` (APU-135 S8) matches the TOSA convs
//! IrRewriteConv makes, whose captured weight it gave rc4's layout change as runtime
//! `permute_copy`/`view_copy` ops; it applies them to the data, and sends what rc4
//! raises for to stock.

use std::collections::{BTreeMap, BTreeSet};

use crate::fx_constant_tensor_eval::permute_tensor_data;
use crate::fx_conv_args::ConvArgs;
use crate::fx_ir::{
    IrArg, IrGraph, IrNode, MetaValue, NodeId, NodeMeta, NodeOp, NodeTensor, TensorSource,
};
use crate::fx_target::{is_permute_target, is_view_like_target};
use crate::graph::ir::DType;
use crate::layout::dtype::{
    is_quantized_conv, set_special_dtype, value_dtype, SpecialDtype, SPECIAL_DTYPE_KEY,
};
use crate::layout::shape::{mark_contiguous, tosa_op_name};
use crate::store::TensorData;
use crate::StashedPipeline;

use super::contracts::node_ids_from_ir_arg;
use super::{collect_region_metadata, region_name_from_ir_arg};

/// The graph that invokes a cond branch (`None` = parent) and the operands bound to the
/// branch's inputs, in placeholder order.
type Invocation = (Option<String>, Vec<NodeId>);

fn cond_invocations(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
) -> BTreeMap<String, Invocation> {
    let graphs = std::iter::once((None, parent)).chain(
        submodules
            .iter()
            .map(|(name, (ir, _))| (Some(name.as_str()), ir)),
    );
    let mut invocations = BTreeMap::new();
    for (region, ir) in graphs {
        for node in ir.live_nodes().filter(|node| node.is_cond()) {
            let operands = node_ids_from_ir_arg(node.args.last());
            for branch in [1, 2] {
                if let Some(name) = region_name_from_ir_arg(ir, node.args.get(branch), region) {
                    invocations.insert(name, (region.map(str::to_string), operands.clone()));
                }
            }
        }
    }
    invocations
}

/// A boundary RESCALE on the path from a captured constant into a branch.
struct BoundaryRescale {
    target: String,
    args: Vec<IrArg>,
    meta: NodeMeta,
}

impl BoundaryRescale {
    fn of(node: &IrNode) -> Self {
        Self {
            target: node.target.clone(),
            args: node.args.clone(),
            meta: node.meta.clone(),
        }
    }

    /// `(scale ratio, input zero point)`.
    fn params(&self) -> Option<(f64, i64)> {
        let IrArg::List(scales) = self.args.get(2)? else {
            return None;
        };
        let IrArg::Int(input_zp) = self.args.get(3)? else {
            return None;
        };
        Some((scales.first()?.as_float()?, *input_zp))
    }
}

struct CapturedConstant {
    data: TensorData,
    /// The constant's `tosa_special_dtype` (an INT4 weight's), which a rewritten
    /// weight carries on (RCP:377-378).
    special: Option<MetaValue>,
    /// Boundary RESCALEs from the branch outwards.
    rescales: Vec<BoundaryRescale>,
}

impl CapturedConstant {
    fn rescale_params(&self) -> Option<Vec<(f64, i64)>> {
        self.rescales.iter().map(BoundaryRescale::params).collect()
    }
}

/// Where [`walk_captured_constant`] stopped short of a constant: the graph (`None` =
/// parent) and the node it could not follow.
struct Unresolved {
    region: Option<String>,
    node: NodeId,
}

impl Unresolved {
    /// Critique B7: the while-loop region (in `while_regions`) at one of whose inputs
    /// the walk stopped, if it stopped at one. A while loop's captures are carried
    /// inputs, which RETAB does not resolve (see
    /// [`rewrite_explicit_captured_conv_operands`]).
    fn while_loop_input<'a>(
        &'a self,
        submodules: &BTreeMap<String, StashedPipeline>,
        while_regions: &BTreeSet<String>,
    ) -> Option<&'a str> {
        let region = self.region.as_deref()?;
        let (ir, _) = submodules.get(region)?;
        (while_regions.contains(region) && ir.node(self.node).op == NodeOp::Placeholder)
            .then_some(region)
    }
}

/// The stock fallback's reason for `conv`'s `operand` (its weight or bias) `name`, which
/// while-loop region `region` captures (critique B7).
fn while_capture_reason(operand: &str, name: &str, conv: &IrNode, region: &str) -> String {
    format!(
        "{operand} {name} of {} is captured by while-loop region {region}, whose captured \
         constants RETAB does not resolve (critique B7)",
        conv.name
    )
}

/// Upstream `_resolve_control_flow_constant`: follow RESCALEs and branch inputs outwards
/// until a node with constant data is reached.
fn resolve_captured_constant(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    invocations: &BTreeMap<String, Invocation>,
    region: &str,
    start: NodeId,
) -> Option<CapturedConstant> {
    walk_captured_constant(parent, submodules, invocations, region, start).ok()
}

/// [`resolve_captured_constant`], saying where the walk stopped when it finds no
/// constant: at a node that is neither a RESCALE, a constant nor the input of a branch
/// a cond invokes.
fn walk_captured_constant(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    invocations: &BTreeMap<String, Invocation>,
    region: &str,
    start: NodeId,
) -> Result<CapturedConstant, Unresolved> {
    let mut region = Some(region.to_string());
    let mut current = start;
    let mut rescales = Vec::new();
    let stop = |region: &Option<String>, node| Unresolved {
        region: region.clone(),
        node,
    };
    for _ in 0..256 {
        let ir = match &region {
            None => parent,
            Some(name) => match submodules.get(name) {
                Some((ir, _)) => ir,
                None => return Err(stop(&region, current)),
            },
        };
        let node = ir.node(current);
        if node.is_rescale() {
            rescales.push(BoundaryRescale::of(node));
            current = node
                .first_node_arg()
                .ok_or_else(|| stop(&region, current))?;
            continue;
        }
        if let Some(tensor) = &node.tensor {
            return Ok(CapturedConstant {
                data: tensor.data.clone(),
                special: node.meta.extras.get(SPECIAL_DTYPE_KEY).cloned(),
                rescales,
            });
        }
        let operand = (node.op == NodeOp::Placeholder)
            .then(|| invocations.get(region.as_deref()?))
            .flatten()
            .and_then(|(enclosing, operands)| {
                let index = ir
                    .live_placeholder_ids()
                    .iter()
                    .position(|&id| id == current)?;
                Some((enclosing, *operands.get(index)?))
            });
        let Some((enclosing, operand)) = operand else {
            return Err(stop(&region, current));
        };
        current = operand;
        region = enclosing.clone();
    }
    Err(stop(&region, current))
}

fn decode_integers(data: &TensorData) -> Option<Vec<f32>> {
    let values = match data.dtype {
        DType::Int8 => data.data.iter().map(|&b| b as i8 as f32).collect(),
        DType::Int16 => data
            .data
            .as_chunks::<2>().0.iter()
            .map(|c| i16::from_le_bytes(*c) as f32)
            .collect(),
        DType::Int32 => data
            .data
            .as_chunks::<4>().0.iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        _ => return None,
    };
    Some(values)
}

/// Upstream `_rewrite_control_flow_bias` arithmetic, given the stored bias and its
/// boundary RESCALEs `(scale ratio, input zero point)` from the branch outwards. The
/// tensor math runs in float32, like the torch ops upstream, so ties round the same way.
fn requantize_bias(
    data: &TensorData,
    rescales: &[(f64, i64)],
    bias_scale: f64,
    input_scale: f64,
    weight_scales: &[f64],
) -> Option<Vec<i32>> {
    let values = decode_integers(data)?;
    let boundary_scale = rescales
        .iter()
        .rev()
        .fold(1.0f64, |acc, &(ratio, _)| acc * ratio);
    let boundary_zp = rescales.last()?.1 as f32;
    let source_scale = (boundary_scale * bias_scale) as f32;
    let input_scale = input_scale as f32;
    if weight_scales.len() != 1 && weight_scales.len() != values.len() {
        return None;
    }
    Some(
        values
            .iter()
            .enumerate()
            .map(|(channel, &q)| {
                let weight_scale = weight_scales[channel.min(weight_scales.len() - 1)] as f32;
                let float_bias = (q - boundary_zp) * source_scale;
                let rounded = (float_bias / (input_scale * weight_scale)).round_ties_even();
                (rounded as f64).clamp(i32::MIN as f64, i32::MAX as f64) as i32
            })
            .collect(),
    )
}

struct BiasRewrite {
    region: String,
    conv: NodeId,
    old_bias: NodeId,
    name: String,
    values: Vec<i32>,
    /// RCP:323-325: an int16 activation's INT32 bias is TOSA's INT48.
    int48: bool,
}

/// `_rewrite_control_flow_bias` for `conv`'s bias `bias`, a TOSA convolution's.
fn plan_captured_bias_rewrite(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    invocations: &BTreeMap<String, Invocation>,
    region: &str,
    conv: &IrNode,
    bias: NodeId,
    int48: bool,
) -> Option<BiasRewrite> {
    let ir = &submodules.get(region)?.0;
    // Upstream only rewrites a bias that reaches the conv through a boundary RESCALE.
    if !ir.node(bias).is_rescale() {
        return None;
    }
    let captured = resolve_captured_constant(parent, submodules, invocations, region, bias)?;
    let input_qp = conv.meta.input_qparam_at(ConvArgs::INPUT_INDEX)?;
    let weight_qp = conv.meta.input_qparam_at(ConvArgs::WEIGHT_INDEX)?;
    let bias_qp = conv.meta.input_qparam_at(ConvArgs::BIAS_INDEX)?;
    let weight_scales = if weight_qp.is_per_channel() {
        weight_qp.per_channel_scales.clone()
    } else {
        vec![weight_qp.scale]
    };
    let values = requantize_bias(
        &captured.data,
        &captured.rescale_params()?,
        bias_qp.scale,
        input_qp.scale,
        &weight_scales,
    )?;
    Some(BiasRewrite {
        region: region.to_string(),
        conv: conv.id,
        old_bias: bias,
        name: format!("{}_bias_int32", conv.name),
        values,
        int48,
    })
}

fn apply_bias_rewrite(ir: &mut IrGraph, rewrite: BiasRewrite) {
    let channels = rewrite.values.len() as i64;
    let mut meta = NodeMeta::tensor(vec![channels], "torch.int32");
    if rewrite.int48 {
        set_special_dtype(&mut meta, SpecialDtype::Int48);
    }
    let constant = ir.add_node(
        NodeOp::GetAttr,
        rewrite.name.clone(),
        rewrite.name,
        Vec::new(),
        meta,
    );
    ir.node_mut(constant).tensor = Some(NodeTensor {
        source: TensorSource::GraphModuleAttr,
        data: TensorData {
            data: rewrite
                .values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            shape: vec![channels],
            dtype: DType::Int32,
        },
    });
    ir.set_arg(rewrite.conv, ConvArgs::BIAS_INDEX, IrArg::Node(constant));
    erase_dead_operand_chain(ir, rewrite.old_bias);
}

/// Erase the layout ops and RESCALEs that fed a replaced conv operand, back to the
/// first node something else still uses: views and the `permute_copy` IrRewriteConv
/// puts on a captured weight (design §4.3). The branch input itself stays.
fn erase_dead_operand_chain(ir: &mut IrGraph, start: NodeId) {
    let mut dead = Some(start);
    while let Some(id) = dead {
        let node = ir.node(id);
        let removable = node.is_rescale()
            || is_view_like_target(&node.target)
            || is_permute_target(&node.target);
        if !removable || !ir.live_user_ids(id).is_empty() {
            break;
        }
        dead = node.first_node_arg();
        ir.erase(id);
    }
}

/// The TOSA convolutions explicit mode's IrRewriteConv makes. Their weight is
/// argument 1 and their bias argument 2, as `aten.convolution`'s.
fn is_tosa_conv(node: &IrNode) -> bool {
    node.op == NodeOp::CallFunction
        && matches!(
            tosa_op_name(&node.target),
            Some("CONV2D" | "DEPTHWISE_CONV2D" | "TRANSPOSE_CONV2D" | "CONV3D")
        )
}

/// A layout op on the path from a captured weight to the TOSA conv reading it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LayoutOp {
    Permute(Vec<usize>),
    /// A view, to the shape of its value.
    Reshape(Vec<i64>),
}

/// Explicit mode's `_rewrite_weight` for a captured weight (RCP:328-403).
struct ExplicitWeightRewrite {
    region: String,
    conv: NodeId,
    old_weight: NodeId,
    name: String,
    /// The captured constant, in the conv's weight layout.
    data: TensorData,
    special: Option<MetaValue>,
    /// The boundary RESCALEs to rebuild after the constant, in data-flow order.
    rescales: Vec<BoundaryRescale>,
}

/// Plan the rewrite of `conv`'s weight: `None` when it is a region constant
/// already, `Err` (a stock fallback's reason) when it does not resolve to a
/// constant, which is where rc4 raises (RCP:243-251), or when a while loop
/// captures it (`while_regions`, see [`rewrite_explicit_captured_conv_operands`]).
///
/// IrRewriteConv gave a captured weight its layout change as `permute_copy`
/// and `view_copy` ops, and the boundary contracts have put the boundary
/// RESCALEs next to the branch input. The walk from the conv collects both, to
/// the first node that is neither, which must be the branch input (or a
/// constant), and `walk_captured_constant` follows it out to the parent's
/// constant, collecting the outer regions' RESCALEs. The layout ops apply to
/// the data innermost first. The RESCALEs, all per-tensor, are rebuilt after
/// the constant in data-flow order: the outermost region's first, this
/// region's last.
fn plan_explicit_weight_rewrite(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    invocations: &BTreeMap<String, Invocation>,
    while_regions: &BTreeSet<String>,
    region: &str,
    conv: &IrNode,
) -> Result<Option<ExplicitWeightRewrite>, String> {
    let ir = &submodules
        .get(region)
        .ok_or_else(|| format!("no region {region}"))?
        .0;
    let weight = conv
        .args
        .get(ConvArgs::WEIGHT_INDEX)
        .and_then(IrArg::as_node)
        .ok_or_else(|| format!("{} has no weight node", conv.name))?;
    let weight_name = &ir.node(weight).name;
    let mut layout_ops = Vec::new();
    let mut region_rescales = Vec::new();
    let mut current = weight;
    // A chain cannot be longer than the graph; the bound keeps a malformed
    // (cyclic) graph from looping.
    for _ in 0..=ir.nodes.len() {
        let node = ir.node(current);
        if is_permute_target(&node.target) {
            let dims = node
                .arg_int_list(1)
                .and_then(|dims| {
                    dims.iter()
                        .map(|&dim| crate::fx_ir::normalize_dim(dim, dims.len()))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| format!("permute {} has no dims", node.name))?;
            layout_ops.push(LayoutOp::Permute(dims));
        } else if is_view_like_target(&node.target) {
            let shape = node
                .meta
                .val_shape()
                .ok_or_else(|| format!("view {} has no value", node.name))?;
            layout_ops.push(LayoutOp::Reshape(shape.to_vec()));
        } else if node.is_rescale() {
            region_rescales.push(BoundaryRescale::of(node));
        } else {
            break;
        }
        current = node
            .first_node_arg()
            .ok_or_else(|| format!("{} has no input", node.name))?;
    }
    if current == weight && ir.node(weight).tensor.is_some() {
        return Ok(None);
    }
    let source = ir.node(current);
    let captured = walk_captured_constant(parent, submodules, invocations, region, current)
        .map_err(|stop| {
            let Some(loop_region) = stop.while_loop_input(submodules, while_regions) else {
                return format!(
                    "weight {weight_name} of {} does not resolve to a constant of an enclosing \
                     cond's graph",
                    conv.name
                );
            };
            while_capture_reason("weight", weight_name, conv, loop_region)
        })?;
    if source
        .meta
        .val_shape()
        .is_some_and(|shape| shape != captured.data.shape.as_slice())
    {
        return Err(format!(
            "weight {weight_name} reads {} of shape {:?}, which resolves to a constant of shape \
             {:?}",
            source.name,
            source.meta.val_shape(),
            captured.data.shape
        ));
    }
    let mut data = captured.data;
    for op in layout_ops.iter().rev() {
        data = match op {
            LayoutOp::Permute(dims) => permute_tensor_data(&data, dims)
                .ok_or_else(|| format!("cannot permute weight {weight_name} by {dims:?}"))?,
            LayoutOp::Reshape(shape) => {
                if shape.iter().product::<i64>() != data.shape.iter().product::<i64>() {
                    return Err(format!(
                        "weight {weight_name} of shape {:?} cannot take the shape {shape:?}",
                        data.shape
                    ));
                }
                TensorData {
                    shape: shape.clone(),
                    ..data
                }
            }
        };
    }
    if let Some(shape) = ir.node(weight).meta.val_shape() {
        if shape != data.shape.as_slice() {
            return Err(format!(
                "weight {weight_name} has the shape {shape:?}, its layout ops give {:?}",
                data.shape
            ));
        }
    }
    let mut rescales = captured.rescales;
    rescales.reverse();
    region_rescales.reverse();
    rescales.extend(region_rescales);
    // A per-channel RESCALE would not commute with the layout change.
    if let Some(rescale) = rescales.iter().find(
        |rescale| !matches!(rescale.args.get(2), Some(IrArg::List(scales)) if scales.len() == 1),
    ) {
        return Err(format!(
            "weight {weight_name} reaches the conv through a RESCALE without one scale ({:?})",
            rescale.args.get(2)
        ));
    }
    Ok(Some(ExplicitWeightRewrite {
        region: region.to_string(),
        conv: conv.id,
        old_weight: weight,
        name: format!("{weight_name}_captured"),
        data,
        special: captured.special,
        rescales,
    }))
}

/// `name`, or `name_<n>` with the first `n` no node, live or erased, has.
fn unique_name(ir: &IrGraph, name: String) -> String {
    if !ir.contains_name(&name) {
        return name;
    }
    (1..)
        .map(|suffix| format!("{name}_{suffix}"))
        .find(|candidate| !ir.contains_name(candidate))
        .expect("an unused suffix")
}

/// The region constant's val is its physical shape, as every val is; the
/// rebuilt RESCALEs keep their arguments and meta with that shape, and the
/// conv reads the last one.
fn apply_explicit_weight_rewrite(ir: &mut IrGraph, rewrite: ExplicitWeightRewrite) {
    let shape = rewrite.data.shape.clone();
    let name = unique_name(ir, rewrite.name);
    let mut meta = NodeMeta::tensor(shape.clone(), rewrite.data.dtype.torch_str());
    mark_contiguous(&mut meta);
    if let Some(special) = rewrite.special {
        meta.set_extra(SPECIAL_DTYPE_KEY, special);
    }
    let constant = ir.add_constant_node(
        NodeOp::GetAttr,
        name.clone(),
        name.clone(),
        NodeTensor {
            source: TensorSource::GraphModuleAttr,
            data: rewrite.data,
        },
        meta,
    );
    let mut weight = constant;
    for (index, rescale) in rewrite.rescales.into_iter().enumerate() {
        let mut args = rescale.args;
        args[0] = IrArg::Node(weight);
        let mut meta = rescale
            .meta
            .cloned_with_same_dtype_or(shape.clone(), "torch.int8");
        mark_contiguous(&mut meta);
        let rescale_name = unique_name(ir, format!("{name}_rescale{index}"));
        weight = ir.add_call_function(rescale_name, rescale.target, args, meta);
    }
    ir.set_arg(rewrite.conv, ConvArgs::WEIGHT_INDEX, IrArg::Node(weight));
    erase_dead_operand_chain(ir, rewrite.old_weight);
}

/// Critique A10: rc4's TOSA convolution kernels require an int32 bias for a
/// quantized conv (`backends/arm/tosa/dialect/ops/conv2d.py:46-50`).
/// RewriteConvPass keeps a bias whose dtype is int32, constant or runtime, and
/// hands any other to `_rewrite_control_flow_bias` (RCP:925-933), which
/// requantizes only a boundary RESCALE of a constant. A captured bias whose
/// boundary qparams match reaches the conv without a boundary RESCALE, so it
/// is not requantized and stays the branch's int8 input; rc4 leaves it too,
/// and the kernel raises.
fn non_int32_bias(ir: &IrGraph, conv: &IrNode) -> Option<String> {
    if !is_quantized_conv(&conv.meta) {
        return None;
    }
    let bias = conv
        .args
        .get(ConvArgs::BIAS_INDEX)
        .and_then(IrArg::as_node)?;
    let node = ir.node(bias);
    let dtype = match &node.tensor {
        Some(tensor) => Some(tensor.data.dtype.torch_str().to_owned()),
        None => value_dtype(ir, bias),
    };
    (dtype.as_deref() != Some("torch.int32")).then(|| {
        format!(
            "{}'s bias {} is {}, not int32",
            conv.name,
            node.name,
            dtype.as_deref().unwrap_or("untyped")
        )
    })
}

/// Critique B7 for a bias: the fallback reason when `conv`'s bias, behind any
/// RESCALEs, is the input of a while loop's region (in `while_regions`).
fn while_captured_bias(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    invocations: &BTreeMap<String, Invocation>,
    while_regions: &BTreeSet<String>,
    region: &str,
    conv: &IrNode,
) -> Option<String> {
    let bias = conv
        .args
        .get(ConvArgs::BIAS_INDEX)
        .and_then(IrArg::as_node)?;
    let stop = walk_captured_constant(parent, submodules, invocations, region, bias).err()?;
    let loop_region = stop.while_loop_input(submodules, while_regions)?;
    let name = &submodules.get(region)?.0.node(bias).name;
    Some(while_capture_reason("bias", name, conv, loop_region))
}

/// Explicit layout mode's captured_conv (APU-135 S8; P2 design §4.3). Returns
/// the number of rewritten operands, or requests a stock fallback.
///
/// IrRewriteConv leaves a region conv's captured operands for this step: a
/// weight as runtime layout ops on the branch input, a bias as the branch
/// input. After the boundary contracts have inserted the boundary RESCALEs, it
/// does what rc4's RewriteConvPass does in a branch (RCP:223-403), on every
/// TOSA conv of every region:
///
/// - The weight becomes a region constant (`get_attr`) in TOSA's layout, the
///   parent's constant with the layout ops applied, followed by the boundary
///   RESCALEs: RESCALE is elementwise, so it commutes with the layout change
///   (`plan_explicit_weight_rewrite`, `_rewrite_weight`).
/// - A bias that reaches the conv through a boundary RESCALE is requantized
///   into the int32 bias TOSA needs, marked INT48 for an int16 activation
///   (`_rewrite_control_flow_bias`, RCP:263-326).
///
/// Where rc4 raises, the graph goes to stock (`stock_fallback::request`): a
/// weight that does not resolve to a constant, and a quantized conv whose bias
/// is not int32 afterwards (`non_int32_bias`, critique A10). Nothing changes
/// then. The request reaches Python through `ingest_and_serialize`'s check
/// right after the contracts (critique B6).
///
/// Captures of a while loop's regions go to stock too (critique B7), a weight's
/// or a non-int32 bias's: only cond invocations are followed. rc4 resolves
/// while-loop captures as well, through the operand `_regularize_submodule`
/// records on every region placeholder (`tosa/backend.py:222-238`), so stock
/// gives such a delegate exactly rc4's outcome. Resolving them here would take
/// while semantics rc4 does not model: torch.export makes a body's captured
/// parameters carried inputs, which the body hands back through a clone, and a
/// carried input holds its initial value on every iteration only if each one
/// passes it through unchanged. rc4 resolves any carried input to its initial
/// value; RETAB would have to prove the pass-through. For that form rc4 then
/// raises anyway: RewriteConvPass cannot put the rewritten weight ahead of the
/// region's user inputs. No rc4 op test has a conv in a while body.
pub(crate) fn rewrite_explicit_captured_conv_operands(
    parent: &IrGraph,
    submodules: &mut BTreeMap<String, StashedPipeline>,
) -> u32 {
    let invocations = cond_invocations(parent, submodules);
    let (while_conds, while_bodies) = collect_region_metadata(parent, submodules);
    let while_regions: BTreeSet<String> = while_conds
        .into_iter()
        .chain(while_bodies.into_keys())
        .collect();
    let mut biases = Vec::new();
    let mut weights = Vec::new();
    for (region, (ir, _)) in submodules.iter() {
        for conv in ir.live_nodes().filter(|node| is_tosa_conv(node)) {
            if let Some(bias) = conv.args.get(ConvArgs::BIAS_INDEX).and_then(IrArg::as_node) {
                let int48 = conv
                    .meta
                    .input_qparam_at(ConvArgs::INPUT_INDEX)
                    .is_some_and(|qparam| qparam.dtype_str == "torch.int16");
                biases.extend(plan_captured_bias_rewrite(
                    parent,
                    submodules,
                    &invocations,
                    region,
                    conv,
                    bias,
                    int48,
                ));
            }
            match plan_explicit_weight_rewrite(
                parent,
                submodules,
                &invocations,
                &while_regions,
                region,
                conv,
            ) {
                Ok(rewrite) => weights.extend(rewrite),
                Err(reason) => {
                    crate::stock_fallback::request(format!(
                        "captured_conv: region {region}: {reason}"
                    ));
                    return 0;
                }
            }
        }
    }
    // Every region conv's bias must be int32 once the captured ones are
    // requantized; check before changing anything.
    let rewritten_biases: Vec<(&str, NodeId)> = biases
        .iter()
        .map(|rewrite| (rewrite.region.as_str(), rewrite.conv))
        .collect();
    for (region, (ir, _)) in submodules.iter() {
        for conv in ir.live_nodes().filter(|node| is_tosa_conv(node)) {
            if rewritten_biases.contains(&(region.as_str(), conv.id)) {
                continue;
            }
            if let Some(reason) = non_int32_bias(ir, conv) {
                let reason = while_captured_bias(
                    parent,
                    submodules,
                    &invocations,
                    &while_regions,
                    region,
                    conv,
                )
                .unwrap_or_else(|| format!("{reason}, which rc4's TOSA kernels reject"));
                crate::stock_fallback::request(format!("captured_conv: region {region}: {reason}"));
                return 0;
            }
        }
    }
    let count = (biases.len() + weights.len()) as u32;
    for rewrite in biases {
        if let Some((ir, _)) = submodules.get_mut(&rewrite.region) {
            apply_bias_rewrite(ir, rewrite);
        }
    }
    for rewrite in weights {
        if let Some((ir, _)) = submodules.get_mut(&rewrite.region) {
            apply_explicit_weight_rewrite(ir, rewrite);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int8_constant(values: &[i8]) -> TensorData {
        TensorData {
            data: values.iter().map(|&v| v as u8).collect(),
            shape: vec![values.len() as i64],
            dtype: DType::Int8,
        }
    }

    #[test]
    fn requantizes_at_the_outermost_stored_scale() {
        // One boundary: parent scale 0.02 (zp 3) requantized to the branch scale 0.05.
        let data = int8_constant(&[13, -7, 3]);
        let values =
            requantize_bias(&data, &[(0.02 / 0.05, 3)], 0.05, 0.1, &[0.01, 0.02, 0.04]).unwrap();
        // (q - 3) * 0.02 / (0.1 * w)
        assert_eq!(values, vec![200, -100, 0]);
    }

    #[test]
    fn composes_nested_boundaries_outer_to_inner() {
        // Parent scale 0.02, zp 1 -> outer branch 0.04 -> inner branch 0.08. The inner
        // RESCALE is found first; the zero point comes from the outermost one.
        let data = int8_constant(&[11]);
        let rescales = [(0.04 / 0.08, 5), (0.02 / 0.04, 1)];
        let values = requantize_bias(&data, &rescales, 0.08, 0.5, &[0.1]).unwrap();
        assert_eq!(values, vec![4]);
    }

    #[test]
    fn rejects_mismatched_weight_scales() {
        let data = int8_constant(&[1, 2, 3]);
        assert!(requantize_bias(&data, &[(1.0, 0)], 1.0, 1.0, &[1.0, 1.0]).is_none());
    }

    // -- Explicit layout mode (APU-135 S8) -----------------------------------

    use crate::fx_ir::i64_list_arg;
    use crate::layout::dtype::special_dtype;
    use crate::lowering_effects::LoweringEffectBatch;
    use crate::test_ir::{buffer, call, call_with_meta, live_node, output, placeholder, qparam};

    const COND: &str = "torch.ops.higher_order.cond";
    const WHILE_LOOP: &str = "torch.ops.higher_order.while_loop";
    const RESCALE: &str = "backend.tosa.RESCALE.default";
    const PERMUTE: &str = "aten.permute_copy.default";
    const VIEW: &str = "aten.view_copy.default";
    const CONV2D: &str = "backend.tosa.CONV2D.default";
    const I8: &str = "torch.int8";
    const I16: &str = "torch.int16";

    /// The parent's int8 conv1d weight, `[O, I, K] = [4, 2, 3]` holding
    /// `o * 6 + i * 3 + k - 12`.
    fn weight_data() -> TensorData {
        TensorData {
            data: (0..24).map(|value: i32| (value - 12) as i8 as u8).collect(),
            shape: vec![4, 2, 3],
            dtype: DType::Int8,
        }
    }

    /// `weight_data` in OHWI, `[4, 1, 3, 2]`: permuted by (0, 2, 1), then
    /// viewed, as IrRewriteConv's conv1d layout ops do.
    fn weight_ohwi_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        for o in 0..4 {
            for k in 0..3 {
                for i in 0..2 {
                    bytes.push((o * 6 + i * 3 + k - 12) as i8 as u8);
                }
            }
        }
        bytes
    }

    fn bias_data() -> TensorData {
        int8_constant(&[13, -7, 3, 0])
    }

    /// A boundary RESCALE as the contracts insert one: one scale.
    fn boundary_rescale(
        graph: &mut IrGraph,
        name: &str,
        input: NodeId,
        scale: f64,
        output_zp: i64,
    ) -> NodeId {
        let shape = graph.node_shape_vec(input).expect("a value");
        call(
            graph,
            name,
            RESCALE,
            vec![
                IrArg::Node(input),
                IrArg::Dtype(I8.into()),
                IrArg::List(vec![IrArg::Float(scale)]),
                IrArg::Int(0),
                IrArg::Int(output_zp),
            ],
            shape,
            I8,
        )
    }

    /// `target`'s arguments for branches `branches` over `operands`: a cond's
    /// `(pred, then, else, operands)`, a while loop's `(cond, body, carried,
    /// additional)`.
    fn control_flow_args(target: &str, branches: [&str; 2], operands: Vec<NodeId>) -> Vec<IrArg> {
        let operands = IrArg::List(operands.into_iter().map(IrArg::Node).collect());
        let [first, second] = branches.map(|name| IrArg::Str(name.to_string()));
        if target == COND {
            vec![IrArg::Int(1), first, second, operands]
        } else {
            vec![first, second, IrArg::List(Vec::new()), operands]
        }
    }

    /// The parent: a cond or while loop `target` over an input and the int8
    /// weight and bias constants.
    fn parent_graph(target: &str, branches: [&str; 2]) -> IrGraph {
        let mut parent = IrGraph::empty();
        let x = placeholder(&mut parent, "x", [1, 2, 8], I8);
        let w = buffer(&mut parent, "w", weight_data());
        let b = buffer(&mut parent, "b", bias_data());
        let control_flow = call_with_meta(
            &mut parent,
            "control_flow",
            target,
            control_flow_args(target, branches, vec![x, w, b]),
            NodeMeta::default(),
        );
        let getitem = call(
            &mut parent,
            "getitem",
            "<built-in function getitem>",
            vec![IrArg::Node(control_flow), IrArg::Int(0)],
            [1, 4, 8],
            I8,
        );
        output(&mut parent, getitem);
        parent
    }

    /// A region as explicit IrRewriteConv leaves a conv1d over captured
    /// operands, after the boundary contracts: branch inputs `x_ph`, `w_ph` and
    /// `b_ph`; the weight's and bias's boundary RESCALEs, of `weight_scale` and
    /// `bias_scale` (`None` leaves the operand without one); the weight's
    /// runtime layout ops, `permute_copy(0, 2, 1)` then `view_copy [4, 1, 3,
    /// 2]`; and `conv_tosa`, with an `activation` input qparam.
    fn conv1d_region(
        weight_scale: Option<f64>,
        bias_scale: Option<f64>,
        activation: &str,
    ) -> IrGraph {
        let mut region = IrGraph::empty();
        let x = placeholder(&mut region, "x_ph", [1, 2, 8], activation);
        let w = placeholder(&mut region, "w_ph", [4, 2, 3], I8);
        let b = placeholder(&mut region, "b_ph", [4], I8);
        let weight = weight_scale.map_or(w, |scale| {
            boundary_rescale(&mut region, "w_rescale", w, scale, -1)
        });
        let bias = bias_scale.map_or(b, |scale| {
            boundary_rescale(&mut region, "b_rescale", b, scale, -3)
        });
        let x_nlc = call(
            &mut region,
            "conv_nlc",
            PERMUTE,
            vec![IrArg::Node(x), i64_list_arg(&[0, 2, 1])],
            [1, 8, 2],
            activation,
        );
        let x_nhwc = call(
            &mut region,
            "conv_nhwc",
            VIEW,
            vec![IrArg::Node(x_nlc), i64_list_arg(&[1, 1, 8, 2])],
            [1, 1, 8, 2],
            activation,
        );
        let w_permute = call(
            &mut region,
            "conv_weight_permute",
            PERMUTE,
            vec![IrArg::Node(weight), i64_list_arg(&[0, 2, 1])],
            [4, 3, 2],
            I8,
        );
        let w_ohwi = call(
            &mut region,
            "conv_weight_ohwi",
            VIEW,
            vec![IrArg::Node(w_permute), i64_list_arg(&[4, 1, 3, 2])],
            [4, 1, 3, 2],
            I8,
        );
        let mut meta = NodeMeta::tensor(vec![1, 1, 8, 4], "torch.int32");
        meta.set_input_qparams_from([
            (0, qparam(0.02, 0, activation)),
            (1, qparam(0.005, -1, I8)),
            (2, qparam(0.01, -3, I8)),
        ]);
        meta.set_single_output_qparam(qparam(0.04, 7, I8));
        let conv = call_with_meta(
            &mut region,
            "conv_tosa",
            CONV2D,
            vec![
                IrArg::Node(x_nhwc),
                IrArg::Node(w_ohwi),
                IrArg::Node(bias),
                i64_list_arg(&[1, 1]),
                i64_list_arg(&[0, 0, 1, 1]),
                i64_list_arg(&[1, 1]),
            ],
            meta,
        );
        output(&mut region, conv);
        region
    }

    fn regions(entries: Vec<(&str, IrGraph)>) -> BTreeMap<String, StashedPipeline> {
        entries
            .into_iter()
            .map(|(name, graph)| (name.to_string(), (graph, LoweringEffectBatch::default())))
            .collect()
    }

    /// Run the explicit captured_conv; returns its count and the fallback it
    /// requested.
    fn rewrite_explicitly(
        parent: &IrGraph,
        submodules: &mut BTreeMap<String, StashedPipeline>,
    ) -> (u32, Option<String>) {
        crate::stock_fallback::take();
        let count = rewrite_explicit_captured_conv_operands(parent, submodules);
        (count, crate::stock_fallback::take())
    }

    fn is_live(graph: &IrGraph, name: &str) -> bool {
        graph
            .nodes
            .iter()
            .any(|node| node.alive && node.name == name)
    }

    fn scale_args(node: &IrNode) -> String {
        format!("{:?}", &node.args[1..])
    }

    /// The conv's operand `index` must be a region constant (`get_attr` with
    /// graph-module data, as ingest keeps it in a region); returns its data.
    fn region_constant(graph: &IrGraph, node: NodeId) -> &TensorData {
        let node = graph.node(node);
        assert_eq!(node.op, NodeOp::GetAttr, "{}", node.name);
        let tensor = node.tensor.as_ref().expect("a constant");
        assert!(matches!(tensor.source, TensorSource::GraphModuleAttr));
        &tensor.data
    }

    /// Design §4.3: a captured conv1d weight in a cond branch becomes a region
    /// constant in OHWI, `[4, 1, 3, 2]`, followed by its boundary RESCALE (RCP's
    /// `_rewrite_weight`), and its boundary-rescaled bias the int32 bias at
    /// `s_in * s_w` (`_rewrite_control_flow_bias`). The layout ops and the
    /// RESCALEs that fed them go; the branch inputs stay.
    #[test]
    fn explicit_captured_operands_become_region_constants() {
        let parent = parent_graph(COND, ["true_graph", "false_graph"]);
        let mut submodules = regions(vec![(
            "true_graph",
            conv1d_region(Some(4.0), Some(7.0), I8),
        )]);
        let rescale_args = scale_args(live_node(&submodules["true_graph"].0, "w_rescale"));
        assert_eq!(rewrite_explicitly(&parent, &mut submodules), (2, None));
        let region = &submodules["true_graph"].0;
        let conv = live_node(region, "conv_tosa");

        let rescale = region.node(conv.arg_node(1).expect("a weight"));
        assert_eq!(rescale.target, RESCALE);
        assert_eq!(rescale.name, "conv_weight_ohwi_captured_rescale0");
        assert_eq!(scale_args(rescale), rescale_args);
        assert_eq!(region.node_shape(rescale.id), Some([4, 1, 3, 2].as_slice()));
        let constant = rescale.first_node_arg().expect("an input");
        assert_eq!(region.node(constant).name, "conv_weight_ohwi_captured");
        let data = region_constant(region, constant);
        assert_eq!(
            (data.shape.as_slice(), data.dtype),
            ([4, 1, 3, 2].as_slice(), DType::Int8)
        );
        assert_eq!(data.data, weight_ohwi_bytes());
        assert_eq!(region.node_shape(constant), Some([4, 1, 3, 2].as_slice()));

        for gone in [
            "conv_weight_ohwi",
            "conv_weight_permute",
            "w_rescale",
            "b_rescale",
        ] {
            assert!(!is_live(region, gone), "{gone}");
        }
        for kept in ["w_ph", "b_ph"] {
            assert!(is_live(region, kept), "{kept}");
        }

        let bias = conv.arg_node(2).expect("a bias");
        assert_eq!(region.node(bias).name, "conv_tosa_bias_int32");
        let rewritten_bias = region_constant(region, bias);
        let expected = requantize_bias(&bias_data(), &[(7.0, 0)], 0.01, 0.02, &[0.005])
            .expect("requantizable");
        let values: Vec<i32> = rewritten_bias
            .data
            .chunks(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("4 bytes")))
            .collect();
        assert_eq!(values, expected);
        assert_eq!(special_dtype(&region.node(bias).meta), None);
    }

    /// RCP:323-325: an int16 activation's int32 bias is INT48 to TOSA.
    #[test]
    fn explicit_captured_biases_of_int16_activations_are_int48() {
        let parent = parent_graph(COND, ["true_graph", "false_graph"]);
        let mut submodules = regions(vec![(
            "true_graph",
            conv1d_region(Some(4.0), Some(7.0), I16),
        )]);
        assert_eq!(rewrite_explicitly(&parent, &mut submodules), (2, None));
        let region = &submodules["true_graph"].0;
        let bias = live_node(region, "conv_tosa").arg_node(2).expect("a bias");
        assert_eq!(
            special_dtype(&region.node(bias).meta),
            Some(SpecialDtype::Int48)
        );
    }

    /// `test_nested_cond_conv1d_tosa_INT`'s shape: the weight crosses two cond
    /// boundaries, each with its RESCALE. The constant gets both, in data-flow
    /// order: the outer branch's first.
    #[test]
    fn explicit_nested_captured_weights_rebuild_every_boundary_outermost_first() {
        let parent = parent_graph(COND, ["outer_true", "outer_false"]);
        let mut outer = IrGraph::empty();
        let x = placeholder(&mut outer, "x_ph", [1, 2, 8], I8);
        let w = placeholder(&mut outer, "w_ph", [4, 2, 3], I8);
        let b = placeholder(&mut outer, "b_ph", [4], I8);
        let w_outer = boundary_rescale(&mut outer, "w_outer_rescale", w, 2.0, 5);
        let b_outer = boundary_rescale(&mut outer, "b_outer_rescale", b, 3.0, 2);
        let inner_cond = call_with_meta(
            &mut outer,
            "inner_cond",
            COND,
            control_flow_args(
                COND,
                ["inner_true", "inner_false"],
                vec![x, w_outer, b_outer],
            ),
            NodeMeta::default(),
        );
        let getitem = call(
            &mut outer,
            "inner_getitem",
            "<built-in function getitem>",
            vec![IrArg::Node(inner_cond), IrArg::Int(0)],
            [1, 4, 8],
            I8,
        );
        output(&mut outer, getitem);
        let outer_args = scale_args(live_node(&outer, "w_outer_rescale"));
        let inner = conv1d_region(Some(4.0), Some(7.0), I8);
        let inner_args = scale_args(live_node(&inner, "w_rescale"));
        let mut submodules = regions(vec![
            ("outer_true", outer),
            ("outer_true__inner_true", inner),
        ]);
        assert_eq!(rewrite_explicitly(&parent, &mut submodules), (2, None));

        let region = &submodules["outer_true__inner_true"].0;
        let conv = live_node(region, "conv_tosa");
        let last = region.node(conv.arg_node(1).expect("a weight"));
        let first = region.node(last.first_node_arg().expect("an input"));
        assert_eq!(
            (first.name.as_str(), last.name.as_str()),
            (
                "conv_weight_ohwi_captured_rescale0",
                "conv_weight_ohwi_captured_rescale1"
            )
        );
        assert_eq!(scale_args(first), outer_args);
        assert_eq!(scale_args(last), inner_args);
        let data = region_constant(region, first.first_node_arg().expect("an input"));
        assert_eq!(data.data, weight_ohwi_bytes());
        // The outer branch keeps its RESCALEs: the inner cond reads them.
        assert!(is_live(&submodules["outer_true"].0, "w_outer_rescale"));

        let bias = region.node(conv.arg_node(2).expect("a bias"));
        let expected = requantize_bias(&bias_data(), &[(7.0, 0), (3.0, 0)], 0.01, 0.02, &[0.005])
            .expect("requantizable");
        let values: Vec<i32> = region_constant(region, bias.id)
            .data
            .chunks(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("4 bytes")))
            .collect();
        assert_eq!(values, expected);
    }

    /// `conv1d_region`, with the conv reading a region constant weight in OHWI
    /// instead of the captured one.
    fn conv1d_region_with_constant_weight(bias_scale: Option<f64>) -> IrGraph {
        let mut region = conv1d_region(Some(4.0), bias_scale, I8);
        let conv = live_node(&region, "conv_tosa").id;
        let weight = region.add_constant_node(
            NodeOp::GetAttr,
            "conv_weight_constant".into(),
            "conv_weight_constant".into(),
            NodeTensor {
                source: TensorSource::GraphModuleAttr,
                data: TensorData {
                    data: weight_ohwi_bytes(),
                    shape: vec![4, 1, 3, 2],
                    dtype: DType::Int8,
                },
            },
            NodeMeta::tensor(vec![4, 1, 3, 2], I8),
        );
        region.set_arg(conv, 1, IrArg::Node(weight));
        region
    }

    /// Where rc4 raises the graph goes to stock, and nothing changes: a weight
    /// read through a runtime op; and a captured bias whose boundary needed no
    /// RESCALE, which stays an int8 branch input that rc4's TOSA kernel rejects
    /// (critique A10). So do a while body's captured weight and captured int8
    /// bias, which only cond invocations would resolve (critique B7).
    #[test]
    fn explicit_captured_conv_falls_back_where_rc4_raises() {
        struct Case {
            label: &'static str,
            parent: IrGraph,
            region: &'static str,
            graph: IrGraph,
            reason: &'static str,
        }
        let mut int8_constant_bias = conv1d_region(Some(4.0), Some(7.0), I8);
        let conv = live_node(&int8_constant_bias, "conv_tosa").id;
        let constant = int8_constant_bias.add_constant_node(
            NodeOp::GetAttr,
            "conv_bias".into(),
            "conv_bias".into(),
            NodeTensor {
                source: TensorSource::GraphModuleAttr,
                data: bias_data(),
            },
            NodeMeta::tensor(vec![4], I8),
        );
        int8_constant_bias.set_arg(conv, 2, IrArg::Node(constant));
        let mut through_runtime_op = conv1d_region(Some(4.0), Some(7.0), I8);
        let permute = live_node(&through_runtime_op, "conv_weight_permute").id;
        let rescale = live_node(&through_runtime_op, "w_rescale").id;
        let negated = call(
            &mut through_runtime_op,
            "w_neg",
            "aten.neg.default",
            vec![IrArg::Node(rescale)],
            [4, 2, 3],
            I8,
        );
        through_runtime_op.set_arg(permute, 0, IrArg::Node(negated));
        let cases = [
            Case {
                label: "a while body's captured weight",
                parent: parent_graph(WHILE_LOOP, ["loop_cond", "loop_body"]),
                region: "loop_body",
                graph: conv1d_region(Some(4.0), Some(7.0), I8),
                reason: "region loop_body: weight conv_weight_ohwi of conv_tosa is captured by \
                         while-loop region loop_body, whose captured constants RETAB does not \
                         resolve (critique B7)",
            },
            Case {
                label: "a while body's captured bias",
                parent: parent_graph(WHILE_LOOP, ["loop_cond", "loop_body"]),
                region: "loop_body",
                graph: conv1d_region_with_constant_weight(Some(7.0)),
                reason: "region loop_body: bias b_rescale of conv_tosa is captured by while-loop \
                         region loop_body, whose captured constants RETAB does not resolve \
                         (critique B7)",
            },
            Case {
                label: "a weight read through a runtime op",
                parent: parent_graph(COND, ["true_graph", "false_graph"]),
                region: "true_graph",
                graph: through_runtime_op,
                reason: "weight conv_weight_ohwi of conv_tosa does not resolve",
            },
            Case {
                label: "an int8 bias without a boundary RESCALE",
                parent: parent_graph(COND, ["true_graph", "false_graph"]),
                region: "true_graph",
                graph: conv1d_region(Some(4.0), None, I8),
                reason: "conv_tosa's bias b_ph is torch.int8, not int32, which rc4's TOSA kernels \
                         reject",
            },
            Case {
                label: "a region constant bias holding int8",
                parent: parent_graph(COND, ["true_graph", "false_graph"]),
                region: "true_graph",
                graph: int8_constant_bias,
                reason: "conv_tosa's bias conv_bias is torch.int8, not int32",
            },
        ];
        for case in cases {
            let label = case.label;
            let before = crate::layout::replay::snapshot(&case.graph).expect("a valid graph");
            let mut submodules = regions(vec![(case.region, case.graph)]);
            let (count, reason) = rewrite_explicitly(&case.parent, &mut submodules);
            assert_eq!(count, 0, "{label}");
            let reason = reason.unwrap_or_else(|| panic!("{label}: no fallback"));
            assert!(reason.starts_with("captured_conv: "), "{label}: {reason}");
            assert!(reason.contains(case.reason), "{label}: {reason}");
            assert_eq!(
                crate::layout::replay::snapshot(&submodules[case.region].0).expect("a valid graph"),
                before,
                "{label}"
            );
        }
    }

    /// rc4's RewriteConvPass keeps a quantized conv's bias whose dtype is int32,
    /// runtime or constant (RCP:925-933), so captured_conv keeps it too: a cond
    /// branch's captured weight is rewritten, and its int32 branch input stays
    /// the bias. (`explicit_region_constants_stay` keeps a constant one.)
    #[test]
    fn explicit_captured_conv_keeps_a_runtime_int32_bias() {
        let parent = parent_graph(COND, ["true_graph", "false_graph"]);
        let mut region = conv1d_region(Some(4.0), None, I8);
        let b = live_node(&region, "b_ph").id;
        region
            .node_mut(b)
            .meta
            .set_tensor_facts(crate::fx_meta::TensorFacts::new([4], "torch.int32"));
        let mut submodules = regions(vec![("true_graph", region)]);
        assert_eq!(rewrite_explicitly(&parent, &mut submodules), (1, None));
        let region = &submodules["true_graph"].0;
        let conv = live_node(region, "conv_tosa");
        assert_eq!(conv.arg_node(2), Some(b));
        let weight = region.node(conv.arg_node(1).expect("a weight"));
        assert_eq!(weight.name, "conv_weight_ohwi_captured_rescale0");
    }

    /// Critique B6: a fallback captured_conv requests at serialization reaches
    /// the caller. `ingest_and_serialize` runs the contracts through
    /// `apply_control_flow_contracts`, which returns the request for it to raise
    /// as `RETAB_STOCK_FALLBACK:`; nothing after it would read the request.
    #[test]
    fn contracts_return_captured_convs_fallback_requests() {
        crate::stock_fallback::take();
        let mut parent = parent_graph(WHILE_LOOP, ["loop_cond", "loop_body"]);
        let mut submodules = regions(vec![("loop_body", conv1d_region(Some(4.0), Some(7.0), I8))]);
        let reason = crate::apply_control_flow_contracts(&mut parent, &mut submodules)
            .expect_err("a stock fallback");
        assert!(
            reason.starts_with(
                "captured_conv: region loop_body: weight conv_weight_ohwi of conv_tosa is \
                 captured by while-loop region loop_body"
            ),
            "{reason}"
        );
        assert_eq!(crate::stock_fallback::take(), None, "the request is taken");

        let mut parent = parent_graph(COND, ["true_graph", "false_graph"]);
        let mut submodules = regions(vec![(
            "true_graph",
            conv1d_region(Some(4.0), Some(7.0), I8),
        )]);
        assert_eq!(
            crate::apply_control_flow_contracts(&mut parent, &mut submodules),
            Ok(())
        );
        let region = &submodules["true_graph"].0;
        let weight = live_node(region, "conv_tosa")
            .arg_node(1)
            .expect("a weight");
        assert_eq!(
            region.node(weight).name,
            "conv_weight_ohwi_captured_rescale0"
        );
    }

    /// A region conv IrRewriteConv gave region constants (its own pre-permuted
    /// weight and an int32 bias) has nothing captured: it stays as it is.
    #[test]
    fn explicit_region_constants_stay() {
        let parent = parent_graph(COND, ["true_graph", "false_graph"]);
        let mut region = conv1d_region(Some(4.0), Some(7.0), I8);
        let conv = live_node(&region, "conv_tosa").id;
        let constant = |data: TensorData| NodeTensor {
            source: TensorSource::GraphModuleAttr,
            data,
        };
        let weight = region.add_constant_node(
            NodeOp::GetAttr,
            "conv_weight_ohwi_constant".into(),
            "conv_weight_ohwi_constant".into(),
            constant(TensorData {
                data: weight_ohwi_bytes(),
                shape: vec![4, 1, 3, 2],
                dtype: DType::Int8,
            }),
            NodeMeta::tensor(vec![4, 1, 3, 2], I8),
        );
        let bias = region.add_constant_node(
            NodeOp::GetAttr,
            "conv_bias".into(),
            "conv_bias".into(),
            constant(TensorData {
                data: vec![0; 16],
                shape: vec![4],
                dtype: DType::Int32,
            }),
            NodeMeta::tensor(vec![4], "torch.int32"),
        );
        region.set_arg(conv, 1, IrArg::Node(weight));
        region.set_arg(conv, 2, IrArg::Node(bias));
        let before = crate::layout::replay::snapshot(&region).expect("a valid graph");
        let mut submodules = regions(vec![("true_graph", region)]);
        assert_eq!(rewrite_explicitly(&parent, &mut submodules), (0, None));
        assert_eq!(
            crate::layout::replay::snapshot(&submodules["true_graph"].0).expect("a valid graph"),
            before
        );
    }
}
