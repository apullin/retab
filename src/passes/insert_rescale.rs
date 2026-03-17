use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Insert TOSA RESCALE nodes where requantization is needed.
///
/// Two cases:
/// 1. **Accumulator rescale**: Conv-like ops (Conv2d, DepthwiseConv2d, etc.)
///    produce int32 accumulators that must be rescaled to int8. The scale ratio
///    is `(input_scale * weight_scale) / output_scale`.
/// 2. **Mismatch rescale**: Any node whose `output_qparams` differ from its
///    consumer's `input_qparams` needs a RESCALE between them.
pub struct InsertRescalePass;

impl Pass for InsertRescalePass {
    fn name(&self) -> &str {
        "InsertRescale"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        if debug {
            let mut with_out_qp = 0;
            let mut with_in_qp = 0;
            for &nid in &all_ids {
                if let Some(n) = graph.node(nid) {
                    if !n.meta.output_qparams.is_empty() {
                        with_out_qp += 1;
                    }
                    if !n.meta.input_qparams.is_empty() {
                        with_in_qp += 1;
                    }
                }
            }
            eprintln!(
                "[InsertRescale] checking {} nodes: {} with output_qparams, {} with input_qparams",
                all_ids.len(),
                with_out_qp,
                with_in_qp,
            );
        }

        for node_id in all_ids {
            let node = match graph.node(node_id) {
                Some(n) => n,
                None => continue,
            };

            // Skip nodes without output_qparams
            if node.meta.output_qparams.is_empty() {
                continue;
            }

            let out_qp = match node.meta.output_qparams.iter().find(|(p, _)| *p == 0) {
                Some((_, qp)) => *qp,
                None => continue,
            };

            // Case 1: Accumulator-producing ops always need a rescale (int32 -> int8)
            if is_accumulator_op(&node.target) {
                if insert_accumulator_rescale(graph, store, node_id, &out_qp, debug) {
                    modified = true;
                }
                continue;
            }

            // Case 2: Mismatch rescale — output qparams differ from consumer input qparams
            let users: Vec<NodeId> = graph
                .node(node_id)
                .map(|n| n.users.to_vec())
                .unwrap_or_default();

            for user_id in users {
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

                if debug {
                    let name = graph
                        .node(node_id)
                        .map(|n| n.name.clone())
                        .unwrap_or_default();
                    eprintln!(
                        "[InsertRescale] mismatch rescale after '{}': out_scale={}, in_scale={}",
                        name, out_qp.scale, in_qp.scale,
                    );
                }

                // Compute the combined scale ratio
                let scale_ratio = out_qp.scale / in_qp.scale;

                let rescale_id = create_rescale_node(
                    graph,
                    store,
                    node_id,
                    user_id,
                    scale_ratio,
                    out_qp.zero_point,
                    in_qp.zero_point,
                    in_qp.dtype,
                );

                // Set output_qparams on the rescale node so downstream passes
                // can detect further mismatches.
                if let Some(rescale_node) = graph.node_mut(rescale_id) {
                    rescale_node.meta.output_qparams.push((0, in_qp));
                }

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
// Accumulator-op detection
// ---------------------------------------------------------------------------

/// Returns true for ops that produce int32 accumulator outputs and always
/// need a RESCALE to requantize back to int8.
fn is_accumulator_op(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Conv2d
            | OpTarget::DepthwiseConv2d
            | OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::TosaTransposeConv2d
            | OpTarget::TosaFullyConnected
            | OpTarget::Linear
            | OpTarget::Addmm
            | OpTarget::Matmul
            | OpTarget::Bmm
            | OpTarget::TosaMatmul
    )
}

/// Insert a RESCALE node after an accumulator-producing op.
///
/// The accumulator is int32 with effective scale = input_scale * weight_scale.
/// The output should be int8 with the node's output_qparams scale/zp.
/// Scale ratio = (input_scale * weight_scale) / output_scale.
fn insert_accumulator_rescale(
    graph: &mut Graph,
    store: &mut TensorStore,
    node_id: NodeId,
    out_qp: &QuantArgs,
    debug: bool,
) -> bool {
    // Compute accumulator scale from input_qparams
    let (input_scale, weight_scale) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        let is = node
            .meta
            .input_qparams
            .iter()
            .find(|(p, _)| *p == 0)
            .map(|(_, qa)| qa.scale)
            .unwrap_or(1.0);

        let ws = node
            .meta
            .input_qparams
            .iter()
            .find(|(p, _)| *p == 1)
            .map(|(_, qa)| qa.scale)
            .unwrap_or(1.0);

        (is, ws)
    };

    let accumulator_scale = input_scale * weight_scale;
    let output_scale = out_qp.scale;

    if output_scale == 0.0 {
        return false;
    }

    let scale_ratio = accumulator_scale / output_scale;

    if debug {
        let name = graph
            .node(node_id)
            .map(|n| n.name.clone())
            .unwrap_or_default();
        eprintln!(
            "[InsertRescale] accumulator rescale after '{}': input_s={}, weight_s={}, out_s={}, ratio={}",
            name, input_scale, weight_scale, output_scale, scale_ratio,
        );
    }

    // Get users before inserting (we will rewire all of them to go through the rescale)
    let users: Vec<NodeId> = graph
        .node(node_id)
        .map(|n| n.users.to_vec())
        .unwrap_or_default();

    if users.is_empty() {
        return false;
    }

    // Insert the rescale node after the accumulator op, before the first user.
    // We use the first user as the insertion reference point.
    let first_user = users[0];

    // input_zp for the accumulator is 0 (conv accumulator is always zero-centered)
    let input_zp = 0i64;
    let output_zp = out_qp.zero_point;

    let rescale_id = create_rescale_node(
        graph,
        store,
        node_id,
        first_user,
        scale_ratio,
        input_zp,
        output_zp,
        out_qp.dtype,
    );

    // Rewire ALL users of the accumulator op to use the rescale output instead
    for user_id in &users {
        if let Some(user) = graph.node_mut(*user_id) {
            for arg in &mut user.args {
                if let Arg::Node(id) = arg {
                    if *id == node_id {
                        *id = rescale_id;
                    }
                }
            }
        }

        // Update rescale's user list
        if let Some(rescale_node) = graph.node_mut(rescale_id) {
            if !rescale_node.users.contains(user_id) {
                rescale_node.users.push(*user_id);
            }
        }
    }

    // Set output_qparams on the rescale node so that a subsequent
    // InsertRescalePass run can detect mismatches between this rescale's
    // output and its consumers' input_qparams.
    if let Some(rescale_node) = graph.node_mut(rescale_id) {
        rescale_node.meta.output_qparams.push((0, *out_qp));
    }

    // Update the accumulator op's user list: remove all old users, add rescale
    if let Some(src_node) = graph.node_mut(node_id) {
        src_node.users.clear();
        src_node.users.push(rescale_id);
        // Clear output_qparams so a subsequent InsertRescalePass run
        // (e.g., after rewrite_conv in phase2) doesn't double-insert.
        src_node.meta.output_qparams.clear();
    }

    true
}

// ---------------------------------------------------------------------------
// Rescale node creation helper
// ---------------------------------------------------------------------------

/// Monotonically increasing counter to disambiguate rescale node names.
static RESCALE_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Create a RESCALE node with its multiplier/shift const nodes, inserted
/// before `ref_id` in the graph order. Returns the NodeId of the new RESCALE.
fn create_rescale_node(
    graph: &mut Graph,
    store: &mut TensorStore,
    input_id: NodeId,
    ref_id: NodeId,
    scale_ratio: f64,
    input_zp: i64,
    output_zp: i64,
    output_dtype: DType,
) -> NodeId {
    let (multiplier, shift) = compute_multiplier_and_shift(scale_ratio);
    let counter = RESCALE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let node_name_base = graph
        .node(input_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();
    let rescale_name = format!("{}_rescale_{}", node_name_base, counter);

    let meta_val = graph
        .node(input_id)
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
        ref_id,
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
        ref_id,
        OpType::Placeholder,
        OpTarget::TosaConst,
        smallvec![],
        vec![],
        NodeMeta::new(TensorInfo::new(&[1], DType::Int8)),
        shift_name,
    );

    // Create the RESCALE node
    let rescale_meta = NodeMeta::new(TensorInfo::new(meta_val.shape_slice(), output_dtype));
    let rescale_id = graph.insert_before(
        ref_id,
        OpType::CallFunction,
        OpTarget::TosaRescale,
        smallvec![
            Arg::Node(input_id),
            Arg::Node(mult_id),
            Arg::Node(shift_id),
        ],
        vec![
            ("input_zp".into(), Arg::Int(input_zp)),
            ("output_zp".into(), Arg::Int(output_zp)),
        ],
        rescale_meta,
        rescale_name,
    );

    shape::propagate_shape(graph, rescale_id);

    rescale_id
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

    #[test]
    fn test_is_accumulator_op() {
        assert!(is_accumulator_op(&OpTarget::Conv2d));
        assert!(is_accumulator_op(&OpTarget::DepthwiseConv2d));
        assert!(is_accumulator_op(&OpTarget::TosaConv2d));
        assert!(is_accumulator_op(&OpTarget::Linear));
        assert!(!is_accumulator_op(&OpTarget::Relu));
        assert!(!is_accumulator_op(&OpTarget::Add));
        assert!(!is_accumulator_op(&OpTarget::TosaClamp));
    }
}
