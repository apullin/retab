use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Insert TOSA RESCALE nodes between quantized operators whose output qparams
/// differ from the next operator's input qparams.
pub struct InsertRescalePass;

impl Pass for InsertRescalePass {
    fn name(&self) -> &str {
        "InsertRescale"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            // Get output qparams for this node
            let out_qparams = match graph.node(node_id) {
                Some(n) if !n.meta.output_qparams.is_empty() => {
                    n.meta.output_qparams.clone()
                }
                _ => continue,
            };

            // Get the output qparam at position 0
            let out_qp = match out_qparams.iter().find(|(p, _)| *p == 0) {
                Some((_, qp)) => *qp,
                None => continue,
            };

            // Get all users of this node
            let users: Vec<NodeId> = graph
                .node(node_id)
                .map(|n| n.users.to_vec())
                .unwrap_or_default();

            for user_id in users {
                // Find the input qparam on the user that corresponds to this node
                let user_input_qp = {
                    let user = match graph.node(user_id) {
                        Some(n) => n,
                        None => continue,
                    };

                    // Find which arg position references node_id
                    let pos = user
                        .args
                        .iter()
                        .enumerate()
                        .find_map(|(i, arg)| match arg {
                            Arg::Node(id) if *id == node_id => Some(i as u32),
                            _ => None,
                        });

                    let pos = match pos {
                        Some(p) => p,
                        None => continue,
                    };

                    // Get the input qparam at that position
                    user.meta
                        .input_qparams
                        .iter()
                        .find(|(p, _)| *p == pos)
                        .map(|(_, qp)| *qp)
                };

                let in_qp = match user_input_qp {
                    Some(qp) => qp,
                    None => continue,
                };

                // Check if rescale is needed (different scale or zero_point)
                if (out_qp.scale - in_qp.scale).abs() < 1e-12
                    && out_qp.zero_point == in_qp.zero_point
                {
                    continue;
                }

                // Compute the combined scale ratio
                let scale_ratio = out_qp.scale / in_qp.scale;
                let (multiplier, shift) = compute_multiplier_and_shift(scale_ratio);

                // Create const nodes for multiplier, shift, input_zp, output_zp
                let node_name_base = graph
                    .node(node_id)
                    .map(|n| n.name.clone())
                    .unwrap_or_default();
                let rescale_name = format!("{}_rescale", node_name_base);

                let meta_val = graph
                    .node(node_id)
                    .map(|n| n.meta.val)
                    .unwrap_or_default();

                // Create multiplier const
                let mult_name = format!("{}_multiplier", rescale_name);
                let mult_data = multiplier.to_le_bytes().to_vec();
                store.insert(
                    mult_name.clone(),
                    TensorData {
                        data: mult_data,
                        shape: vec![1],
                        dtype: DType::Int32,
                    },
                );
                let mult_id = graph.insert_before(
                    user_id,
                    OpType::Placeholder,
                    OpTarget::TosaConst,
                    smallvec![],
                    vec![],
                    NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
                    mult_name,
                );

                // Create shift const
                let shift_name = format!("{}_shift", rescale_name);
                let shift_data = vec![shift as u8];
                store.insert(
                    shift_name.clone(),
                    TensorData {
                        data: shift_data,
                        shape: vec![1],
                        dtype: DType::Int8,
                    },
                );
                let shift_id = graph.insert_before(
                    user_id,
                    OpType::Placeholder,
                    OpTarget::TosaConst,
                    smallvec![],
                    vec![],
                    NodeMeta::new(TensorInfo::new(&[1], DType::Int8)),
                    shift_name,
                );

                // Create the RESCALE node
                // Args: [input, multiplier, shift]
                // input_zp and output_zp stored as kwargs
                let rescale_meta = NodeMeta::new(TensorInfo::new(
                    meta_val.shape_slice(),
                    in_qp.dtype,
                ));
                let rescale_id = graph.insert_before(
                    user_id,
                    OpType::CallFunction,
                    OpTarget::TosaRescale,
                    smallvec![
                        Arg::Node(node_id),
                        Arg::Node(mult_id),
                        Arg::Node(shift_id),
                    ],
                    vec![
                        ("input_zp".into(), Arg::Int(out_qp.zero_point)),
                        ("output_zp".into(), Arg::Int(in_qp.zero_point)),
                    ],
                    rescale_meta,
                    rescale_name,
                );

                shape::propagate_shape(graph, rescale_id);

                // Rewire: replace this specific use of node_id in user with rescale_id
                if let Some(user) = graph.node_mut(user_id) {
                    for arg in &mut user.args {
                        if let Arg::Node(id) = arg {
                            if *id == node_id {
                                *id = rescale_id;
                                break; // Only replace first occurrence
                            }
                        }
                    }
                }

                // Update user lists
                if let Some(rescale_node) = graph.node_mut(rescale_id) {
                    if !rescale_node.users.contains(&user_id) {
                        rescale_node.users.push(user_id);
                    }
                }
                if let Some(src_node) = graph.node_mut(node_id) {
                    src_node.users.retain(|uid| *uid != user_id);
                    if !src_node.users.contains(&rescale_id) {
                        src_node.users.push(rescale_id);
                    }
                }

                modified = true;
            }
        }

        modified
    }
}

// ---------------------------------------------------------------------------
// Scale-to-multiplier/shift conversion
// ---------------------------------------------------------------------------

/// Convert a floating-point scale to a fixed-point (multiplier, shift) pair
/// matching the Python `_compute_multiplier_and_shift` behavior.
fn compute_multiplier_and_shift(scale: f64) -> (i32, i8) {
    if scale == 0.0 {
        return (0, 0);
    }
    let (mantissa, exponent) = frexp(scale);
    let shift = 31 - exponent;
    let multiplier = (mantissa * (1i64 << 31) as f64).round() as i32;
    (multiplier, shift as i8)
}

/// Decompose a float into mantissa and exponent such that
/// `mantissa * 2^exponent == x` where `0.5 <= |mantissa| < 1.0`.
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i32 - 1022;
    let mantissa =
        f64::from_bits((bits & 0x800F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000);
    (mantissa, exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frexp_basic() {
        let (m, e) = frexp(1.0);
        assert!((m - 0.5).abs() < 1e-10);
        assert_eq!(e, 1);
    }

    #[test]
    fn test_frexp_zero() {
        let (m, e) = frexp(0.0);
        assert_eq!(m, 0.0);
        assert_eq!(e, 0);
    }

    #[test]
    fn test_compute_multiplier_shift() {
        let (mult, shift) = compute_multiplier_and_shift(0.5);
        assert!(mult > 0);
        assert!(shift >= 0);
    }
}
