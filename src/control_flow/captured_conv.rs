//! Port of the control-flow part of upstream RewriteConvPass (ExecuTorch v1.6.0-rc4):
//! `_resolve_control_flow_constant` and `_rewrite_control_flow_bias`.
//!
//! When a cond branch captures a conv bias from the enclosing graph, the bias arrives as
//! a branch input. The quantizer treats that input as an activation, so the branch sees
//! an int8 tensor requantized at the boundary rather than the INT32 bias at
//! input_scale * weight_scale that TOSA CONV2D needs. The bias is traced back through
//! the boundary RESCALEs and branch inputs, across nested conds, to the constant in the
//! outermost graph. It is then dequantized at the scale the RESCALE chain implies and
//! requantized into a branch-local INT32 constant. This runs after the boundary
//! contracts have created those RESCALEs.

use std::collections::BTreeMap;

use crate::fx_conv_args::ConvArgs;
use crate::fx_ir::{IrArg, IrGraph, IrNode, NodeId, NodeMeta, NodeOp, NodeTensor, TensorSource};
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

/// `(scale ratio, input zero point)` of a boundary RESCALE.
fn rescale_params(node: &IrNode) -> Option<(f64, i64)> {
    let IrArg::List(scales) = node.args.get(2)? else {
        return None;
    };
    let IrArg::Int(input_zp) = node.args.get(3)? else {
        return None;
    };
    Some((scales.first()?.as_float()?, *input_zp))
}

struct CapturedConstant {
    data: TensorData,
    /// Boundary RESCALEs from the branch outwards.
    rescales: Vec<(f64, i64)>,
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
            rescales.push(rescale_params(node)?);
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
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        DType::Int32 => data
            .data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        _ => return None,
    };
    Some(values)
}

/// Upstream `_rewrite_control_flow_bias` arithmetic. The tensor math runs in float32,
/// like the torch ops upstream, so ties round the same way.
fn requantize_bias(
    captured: &CapturedConstant,
    bias_scale: f64,
    input_scale: f64,
    weight_scales: &[f64],
) -> Option<Vec<i32>> {
    let values = decode_integers(&captured.data)?;
    let boundary_scale = captured
        .rescales
        .iter()
        .rev()
        .fold(1.0f64, |acc, &(ratio, _)| acc * ratio);
    let boundary_zp = captured.rescales.last()?.1 as f32;
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
    let values = requantize_bias(&captured, bias_qp.scale, input_qp.scale, &weight_scales)?;
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

    let mut dead = Some(rewrite.old_bias);
    while let Some(id) = dead {
        let node = ir.node(id);
        if !node.is_rescale() || !ir.live_user_ids(id).is_empty() {
            break;
        }
        dead = node.first_node_arg();
        ir.erase(id);
    }
}

/// Rewrite every captured, boundary-rescaled conv bias in the cond branches to an INT32
/// constant. Returns the number of rewritten convs.
pub(crate) fn rewrite_captured_conv_biases(
    parent: &IrGraph,
    submodules: &mut BTreeMap<String, StashedPipeline>,
) -> u32 {
    let invocations = cond_invocations(parent, submodules);
    let mut rewrites = Vec::new();
    for (region, (ir, _)) in submodules.iter() {
        for conv in ir
            .live_nodes()
            .filter(|node| ConvArgs::parse(node).is_some())
        {
            rewrites.extend(plan_bias_rewrite(
                parent,
                submodules,
                &invocations,
                region,
                conv,
            ));
        }
    }
    let count = rewrites.len() as u32;
    for rewrite in rewrites {
        if let Some((ir, _)) = submodules.get_mut(&rewrite.region) {
            apply_bias_rewrite(ir, rewrite);
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
        let captured = CapturedConstant {
            data: int8_constant(&[13, -7, 3]),
            rescales: vec![(0.02 / 0.05, 3)],
        };
        let values = requantize_bias(&captured, 0.05, 0.1, &[0.01, 0.02, 0.04]).unwrap();
        // (q - 3) * 0.02 / (0.1 * w)
        assert_eq!(values, vec![200, -100, 0]);
    }

    #[test]
    fn composes_nested_boundaries_outer_to_inner() {
        // Parent scale 0.02, zp 1 -> outer branch 0.04 -> inner branch 0.08. The inner
        // RESCALE is found first; the zero point comes from the outermost one.
        let captured = CapturedConstant {
            data: int8_constant(&[11]),
            rescales: vec![(0.04 / 0.08, 5), (0.02 / 0.04, 1)],
        };
        let values = requantize_bias(&captured, 0.08, 0.5, &[0.1]).unwrap();
        assert_eq!(values, vec![4]);
    }

    #[test]
    fn rejects_mismatched_weight_scales() {
        let captured = CapturedConstant {
            data: int8_constant(&[1, 2, 3]),
            rescales: vec![(1.0, 0)],
        };
        assert!(requantize_bias(&captured, 1.0, 1.0, &[1.0, 1.0]).is_none());
    }
}
