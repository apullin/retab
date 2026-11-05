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

use std::collections::BTreeMap;

use crate::fx_conv_args::ConvArgs;
use crate::fx_ir::{IrArg, IrGraph, IrNode, NodeId, NodeMeta, NodeOp, NodeTensor, TensorSource};
use crate::fx_target::{is_tosa_transpose_target, is_view_like_target};
use crate::graph::ir::DType;
use crate::store::TensorData;
use crate::StashedPipeline;

use super::contracts::node_ids_from_ir_arg;
use super::region_name_from_ir_arg;

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
    /// Boundary RESCALEs from the branch outwards.
    rescales: Vec<BoundaryRescale>,
}

impl CapturedConstant {
    fn rescale_params(&self) -> Option<Vec<(f64, i64)>> {
        self.rescales.iter().map(BoundaryRescale::params).collect()
    }
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
    let mut region = Some(region.to_string());
    let mut current = start;
    let mut rescales = Vec::new();
    for _ in 0..256 {
        let ir = match &region {
            None => parent,
            Some(name) => &submodules.get(name)?.0,
        };
        let node = ir.node(current);
        if node.is_rescale() {
            rescales.push(BoundaryRescale::of(node));
            current = node.first_node_arg()?;
            continue;
        }
        if let Some(tensor) = &node.tensor {
            return Some(CapturedConstant {
                data: tensor.data.clone(),
                rescales,
            });
        }
        if node.op != NodeOp::Placeholder {
            return None;
        }
        let (enclosing, operands) = invocations.get(region.as_deref()?)?;
        let index = ir
            .live_placeholder_ids()
            .iter()
            .position(|&id| id == current)?;
        current = *operands.get(index)?;
        region = enclosing.clone();
    }
    None
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
}

fn plan_bias_rewrite(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    invocations: &BTreeMap<String, Invocation>,
    region: &str,
    conv: &IrNode,
) -> Option<BiasRewrite> {
    let ir = &submodules.get(region)?.0;
    let bias = ConvArgs::parse(conv)?.bias?;
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
    })
}

fn apply_bias_rewrite(ir: &mut IrGraph, rewrite: BiasRewrite) {
    let channels = rewrite.values.len() as i64;
    let constant = ir.add_node(
        NodeOp::GetAttr,
        rewrite.name.clone(),
        rewrite.name,
        Vec::new(),
        NodeMeta::tensor(vec![channels], "torch.int32"),
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
/// first node something else still uses. The branch input itself stays.
fn erase_dead_operand_chain(ir: &mut IrGraph, start: NodeId) {
    let mut dead = Some(start);
    while let Some(id) = dead {
        let node = ir.node(id);
        let removable = node.is_rescale()
            || is_view_like_target(&node.target)
            || is_tosa_transpose_target(&node.target);
        if !removable || !ir.live_user_ids(id).is_empty() {
            break;
        }
        dead = node.first_node_arg();
        ir.erase(id);
    }
}

struct WeightRewrite {
    region: String,
    conv: NodeId,
    old_weight: NodeId,
    name: String,
    /// The captured constant, reshaped to the weight's logical (OIHW) shape.
    data: TensorData,
    /// The physical layout: the TRANSPOSE the branch ran on the weight, or the
    /// operand's own dim order when it had none.
    dim_order: Option<Vec<usize>>,
    /// The boundary RESCALEs to re-apply, outermost first.
    rescales: Vec<BoundaryRescale>,
}

/// Upstream `_rewrite_weight` for a weight a cond branch captures. In the branch the
/// conv reads `TRANSPOSE(reshape*(RESCALE*(branch input)))`, where the reshape is the
/// conv1d unsqueeze and the TRANSPOSE the layout change to OHWI.
fn plan_weight_rewrite(
    parent: &IrGraph,
    submodules: &BTreeMap<String, StashedPipeline>,
    invocations: &BTreeMap<String, Invocation>,
    region: &str,
    conv: &IrNode,
) -> Option<WeightRewrite> {
    let ir = &submodules.get(region)?.0;
    let args = ConvArgs::parse(conv)?;
    // Plain convs only. The serializer lays out depthwise and transposed weights
    // differently, and reads a conv with one input channel as depthwise.
    if args.groups != 1 || args.transposed {
        return None;
    }
    let operand = ir.node(args.weight);
    let (logical, dim_order) = if is_tosa_transpose_target(&operand.target) {
        let perm = operand.arg_int_list(1)?;
        let order = perm
            .iter()
            .map(|&axis| usize::try_from(axis).ok())
            .collect::<Option<Vec<_>>>()?;
        (operand.first_node_arg()?, Some(order))
    } else {
        (args.weight, operand.meta.cloned_tosa_dim_order())
    };
    let shape = ir.node(logical).meta.val_shape()?.to_vec();
    if shape.get(1).is_none_or(|&in_channels| in_channels < 2)
        || dim_order.as_ref().is_some_and(|order| order.len() != shape.len())
    {
        return None;
    }
    let mut source = logical;
    while is_view_like_target(&ir.node(source).target) {
        source = ir.node(source).first_node_arg()?;
    }
    let source_node = ir.node(source);
    if source_node.tensor.is_some()
        || !(source_node.is_rescale() || source_node.op == NodeOp::Placeholder)
    {
        return None;
    }
    let captured = resolve_captured_constant(parent, submodules, invocations, region, source)?;
    let numel = usize::try_from(shape.iter().product::<i64>()).ok()?;
    if numel * captured.data.dtype.storage_size_bytes() != captured.data.data.len() {
        return None;
    }
    let mut rescales = captured.rescales;
    rescales.reverse();
    Some(WeightRewrite {
        region: region.to_string(),
        conv: conv.id,
        old_weight: args.weight,
        name: format!("{}_captured_weight", conv.name),
        data: TensorData {
            data: captured.data.data,
            shape,
            dtype: captured.data.dtype,
        },
        dim_order,
        rescales,
    })
}

/// The new constant has the logical shape and the TRANSPOSE as its dim order, so the
/// serializer writes it in OHWI. The re-applied RESCALEs keep that shape and order, as
/// elementwise ops on an NHWC tensor do, and the conv reads the last one.
fn apply_weight_rewrite(ir: &mut IrGraph, rewrite: WeightRewrite) {
    let shape = rewrite.data.shape.clone();
    let mut constant_meta = NodeMeta::tensor(shape.clone(), rewrite.data.dtype.torch_str());
    constant_meta.replace_tosa_dim_order(rewrite.dim_order.clone());
    let constant = ir.add_node(
        NodeOp::GetAttr,
        rewrite.name.clone(),
        rewrite.name.clone(),
        Vec::new(),
        constant_meta,
    );
    ir.node_mut(constant).tensor = Some(NodeTensor {
        source: TensorSource::GraphModuleAttr,
        data: rewrite.data,
    });
    let mut weight = constant;
    for (index, rescale) in rewrite.rescales.into_iter().enumerate() {
        let mut args = rescale.args;
        args[0] = IrArg::Node(weight);
        let mut meta = rescale.meta.cloned_with_same_dtype_or(shape.clone(), "torch.int8");
        meta.replace_tosa_dim_order(rewrite.dim_order.clone());
        weight = ir.add_call_function(
            format!("{}_rescale{index}", rewrite.name),
            rescale.target,
            args,
            meta,
        );
    }
    ir.set_arg(rewrite.conv, ConvArgs::WEIGHT_INDEX, IrArg::Node(weight));
    erase_dead_operand_chain(ir, rewrite.old_weight);
}

/// Replace every captured conv weight and boundary-rescaled conv bias in the cond
/// branches with branch-local constants. Returns the number of rewritten operands.
pub(crate) fn rewrite_captured_conv_operands(
    parent: &IrGraph,
    submodules: &mut BTreeMap<String, StashedPipeline>,
) -> u32 {
    let invocations = cond_invocations(parent, submodules);
    let mut biases = Vec::new();
    let mut weights = Vec::new();
    for (region, (ir, _)) in submodules.iter() {
        for conv in ir
            .live_nodes()
            .filter(|node| ConvArgs::parse(node).is_some())
        {
            biases.extend(plan_bias_rewrite(parent, submodules, &invocations, region, conv));
            weights.extend(plan_weight_rewrite(parent, submodules, &invocations, region, conv));
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
            apply_weight_rewrite(ir, rewrite);
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
}
