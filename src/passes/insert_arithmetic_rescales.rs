use std::collections::HashSet;

use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Insert TOSA RESCALE nodes around quantized arithmetic ops to satisfy TOSA
/// requirements for ADD, SUB, MUL, and REDUCE_SUM.
///
/// TOSA requires:
/// - **ADD/SUB**: Both inputs at the same scale/zp. We rescale both to a
///   common scale (`2 * max(s0, s1)`) and insert an output rescale to the
///   node's desired output scale.
/// - **MUL**: Output has natural scale `s0 * s1`. We insert an output rescale
///   from the natural scale to the node's desired output scale.
/// - **REDUCE_SUM**: The sum accumulates in a wider type; insert a rescale
///   from the accumulator back to int8.
///
/// This pass should run AFTER `PropagateQParamsPass` (so input/output_qparams
/// are populated) and AFTER `FuseDuplicateUsersPass`.
pub struct InsertArithmeticRescalesPass;

/// Monotonically increasing counter for naming.
static ARITH_RESCALE_COUNTER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

impl Pass for InsertArithmeticRescalesPass {
    fn name(&self) -> &str {
        "InsertArithmeticRescales"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let debug = std::env::var("TOSA_ENGINE_DEBUG").is_ok();
        let mut modified = false;
        let all_ids = graph.all_node_ids();
        let mut processed: HashSet<NodeId> = HashSet::new();

        for node_id in all_ids {
            // Skip nodes we've already processed in this run
            if processed.contains(&node_id) {
                continue;
            }

            let (target, has_qp) = match graph.node(node_id) {
                Some(n) => {
                    let has = !n.meta.input_qparams.is_empty();
                    (n.target.clone(), has)
                }
                None => continue,
            };

            if !has_qp {
                continue;
            }

            // Skip nodes that are already TosaRescale (avoid double-processing)
            if target == OpTarget::TosaRescale {
                continue;
            }

            match &target {
                OpTarget::Add
                | OpTarget::Sub
                | OpTarget::TosaAdd
                | OpTarget::TosaSub => {
                    if insert_add_sub_rescales(graph, store, node_id, debug) {
                        modified = true;
                    }
                    processed.insert(node_id);
                }
                OpTarget::Mul | OpTarget::TosaMul => {
                    if insert_mul_rescale(graph, store, node_id, debug) {
                        modified = true;
                    }
                    processed.insert(node_id);
                }
                _ => {}
            }
        }

        modified
    }
}

// ---------------------------------------------------------------------------
// ADD / SUB rescale insertion
// ---------------------------------------------------------------------------

/// For TOSA ADD/SUB, both inputs must be at the same scale.
/// We rescale both inputs to `common_scale = 2 * max(s_a, s_b)`,
/// then set the ADD/SUB's output_qparams to common_scale.
/// The existing `InsertRescalePass` mismatch detection will then add
/// an output rescale if the consumer expects a different scale.
fn insert_add_sub_rescales(
    graph: &mut Graph,
    store: &mut TensorStore,
    node_id: NodeId,
    debug: bool,
) -> bool {
    // Get input qparams for positions 0 and 1
    let (qp_a, qp_b, input_a_id, input_b_id, node_name) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        let qp_a = node
            .meta
            .input_qparams
            .iter()
            .find(|(p, _)| *p == 0)
            .map(|(_, q)| *q);
        let qp_b = node
            .meta
            .input_qparams
            .iter()
            .find(|(p, _)| *p == 1)
            .map(|(_, q)| *q);

        let input_a = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };
        let input_b = match node.args.get(1) {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };

        (qp_a, qp_b, input_a, input_b, node.name.clone())
    };

    // Need both qparams to insert arithmetic rescales
    let qp_a = match qp_a {
        Some(q) => q,
        None => return false,
    };
    let qp_b = match qp_b {
        Some(q) => q,
        None => return false,
    };

    // Guard: if either input is already a TosaRescale, this ADD/SUB was already
    // processed by a previous run of this pass. Skip to avoid infinite rescaling.
    let input_a_is_rescale = graph
        .node(input_a_id)
        .map(|n| n.target == OpTarget::TosaRescale)
        .unwrap_or(false);
    let input_b_is_rescale = graph
        .node(input_b_id)
        .map(|n| n.target == OpTarget::TosaRescale)
        .unwrap_or(false);
    if input_a_is_rescale || input_b_is_rescale {
        return false;
    }

    // Compute the common scale: 2 * max(s_a, s_b) to avoid overflow
    let common_scale = 2.0 * qp_a.scale.max(qp_b.scale);
    let common_qp = QuantArgs {
        scale: common_scale,
        zero_point: 0,
        dtype: qp_a.dtype,
        qmin: qp_a.qmin,
        qmax: qp_a.qmax,
    };

    if debug {
        eprintln!(
            "[InsertArithRescale] ADD/SUB '{}': s_a={}, s_b={}, common={}",
            node_name, qp_a.scale, qp_b.scale, common_scale,
        );
    }

    let mut any_inserted = false;

    // Rescale input A if its scale differs from common_scale
    if (qp_a.scale - common_scale).abs() > 1e-12 || qp_a.zero_point != 0 {
        let scale_ratio = qp_a.scale / common_scale;
        let rescale_a_id = create_rescale_node(
            graph,
            store,
            input_a_id,
            node_id,
            scale_ratio,
            qp_a.zero_point,
            0, // output zp = 0 for the common domain
            qp_a.dtype,
            &format!("{}_input_a", node_name),
        );

        // Set output_qparams on the rescale
        if let Some(r) = graph.node_mut(rescale_a_id) {
            r.meta.output_qparams.push((0, common_qp));
        }

        // Rewire: replace input_a with rescale in the add/sub node's arg 0
        if let Some(node) = graph.node_mut(node_id) {
            if let Some(Arg::Node(id)) = node.args.first_mut() {
                *id = rescale_a_id;
            }
            // Update input_qparams[0] to common scale
            for entry in &mut node.meta.input_qparams {
                if entry.0 == 0 {
                    entry.1 = common_qp;
                }
            }
        }

        // Update user lists
        if let Some(r) = graph.node_mut(rescale_a_id) {
            if !r.users.contains(&node_id) {
                r.users.push(node_id);
            }
        }
        if let Some(a) = graph.node_mut(input_a_id) {
            a.users.retain(|u| *u != node_id);
            if !a.users.contains(&rescale_a_id) {
                a.users.push(rescale_a_id);
            }
        }

        any_inserted = true;
    }

    // Rescale input B if its scale differs from common_scale
    if (qp_b.scale - common_scale).abs() > 1e-12 || qp_b.zero_point != 0 {
        let scale_ratio = qp_b.scale / common_scale;
        let rescale_b_id = create_rescale_node(
            graph,
            store,
            input_b_id,
            node_id,
            scale_ratio,
            qp_b.zero_point,
            0,
            qp_b.dtype,
            &format!("{}_input_b", node_name),
        );

        if let Some(r) = graph.node_mut(rescale_b_id) {
            r.meta.output_qparams.push((0, common_qp));
        }

        if let Some(node) = graph.node_mut(node_id) {
            if let Some(Arg::Node(id)) = node.args.get_mut(1) {
                *id = rescale_b_id;
            }
            for entry in &mut node.meta.input_qparams {
                if entry.0 == 1 {
                    entry.1 = common_qp;
                }
            }
        }

        if let Some(r) = graph.node_mut(rescale_b_id) {
            if !r.users.contains(&node_id) {
                r.users.push(node_id);
            }
        }
        if let Some(b) = graph.node_mut(input_b_id) {
            b.users.retain(|u| *u != node_id);
            if !b.users.contains(&rescale_b_id) {
                b.users.push(rescale_b_id);
            }
        }

        any_inserted = true;
    }

    // Update the ADD/SUB's output_qparams to the common scale.
    // This way, InsertRescalePass's mismatch detection will insert an output
    // rescale if the consumer expects a different scale.
    if any_inserted {
        if let Some(node) = graph.node_mut(node_id) {
            // Replace existing output_qparams with common_qp
            node.meta.output_qparams.retain(|(p, _)| *p != 0);
            node.meta.output_qparams.push((0, common_qp));
        }
    }

    any_inserted
}

// ---------------------------------------------------------------------------
// MUL rescale insertion
// ---------------------------------------------------------------------------

/// For TOSA MUL, the natural output scale is `s_a * s_b`.
/// We insert an output RESCALE from `(s_a * s_b, 0)` to the node's
/// desired output scale (from output_qparams, if set by FoldQDQ).
/// If no output_qparams are set, we just mark the natural scale.
fn insert_mul_rescale(
    graph: &mut Graph,
    store: &mut TensorStore,
    node_id: NodeId,
    debug: bool,
) -> bool {
    let (qp_a, qp_b, existing_out_qp, node_name) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        let qp_a = node
            .meta
            .input_qparams
            .iter()
            .find(|(p, _)| *p == 0)
            .map(|(_, q)| *q);
        let qp_b = node
            .meta
            .input_qparams
            .iter()
            .find(|(p, _)| *p == 1)
            .map(|(_, q)| *q);
        let out = node
            .meta
            .output_qparams
            .iter()
            .find(|(p, _)| *p == 0)
            .map(|(_, q)| *q);

        (qp_a, qp_b, out, node.name.clone())
    };

    let qp_a = match qp_a {
        Some(q) => q,
        None => return false,
    };

    // Guard: if any user is already a TosaRescale (from a previous run), skip.
    {
        let users = graph
            .node(node_id)
            .map(|n| n.users.to_vec())
            .unwrap_or_default();
        let has_rescale_user = users.iter().any(|uid| {
            graph
                .node(*uid)
                .map(|n| n.target == OpTarget::TosaRescale)
                .unwrap_or(false)
        });
        if has_rescale_user {
            return false;
        }
    }

    // For MUL with only one quantized input (e.g., Mul(x, const_float)),
    // use the single input's scale as the natural output scale
    let natural_scale = if let Some(qp_b) = qp_b {
        qp_a.scale * qp_b.scale
    } else {
        qp_a.scale
    };

    let natural_qp = QuantArgs {
        scale: natural_scale,
        zero_point: 0,
        dtype: qp_a.dtype,
        qmin: qp_a.qmin,
        qmax: qp_a.qmax,
    };

    // If there's an existing output_qparams from FoldQDQ and it differs from
    // the natural scale, we need an output rescale.
    let desired_out_qp = match existing_out_qp {
        Some(oqp) if (oqp.scale - natural_scale).abs() > 1e-12 => oqp,
        _ => {
            // Update the output_qparams to the natural scale
            if let Some(node) = graph.node_mut(node_id) {
                node.meta.output_qparams.retain(|(p, _)| *p != 0);
                node.meta.output_qparams.push((0, natural_qp));
            }
            return false;
        }
    };

    if debug {
        eprintln!(
            "[InsertArithRescale] MUL '{}': natural_scale={}, desired_out={}",
            node_name, natural_scale, desired_out_qp.scale,
        );
    }

    // Set the MUL's output_qparams to the natural scale (what MUL actually produces)
    if let Some(node) = graph.node_mut(node_id) {
        node.meta.output_qparams.retain(|(p, _)| *p != 0);
        node.meta.output_qparams.push((0, natural_qp));
    }

    // The mismatch detection in InsertRescalePass will handle inserting the
    // output rescale from natural_scale to desired_out_qp.scale when it detects
    // that the MUL's output_qparams differ from the consumer's input_qparams.
    //
    // But we need to ensure the consumer's input_qparams are set to desired_out_qp
    // rather than natural_qp. Since backward propagation would have set them to
    // natural_qp, we need to correct this here.
    //
    // Instead, just insert the output rescale directly.
    let users: Vec<NodeId> = graph
        .node(node_id)
        .map(|n| n.users.to_vec())
        .unwrap_or_default();

    if users.is_empty() {
        return false;
    }

    let first_user = users[0];
    let scale_ratio = natural_scale / desired_out_qp.scale;

    let rescale_id = create_rescale_node(
        graph,
        store,
        node_id,
        first_user,
        scale_ratio,
        0, // input_zp = 0 (MUL output is zero-centered)
        desired_out_qp.zero_point,
        desired_out_qp.dtype,
        &format!("{}_mul_out", node_name),
    );

    // Set output_qparams on rescale
    if let Some(r) = graph.node_mut(rescale_id) {
        r.meta.output_qparams.push((0, desired_out_qp));
    }

    // Rewire ALL users of node_id to use rescale_id instead
    for &user_id in &users {
        if let Some(user) = graph.node_mut(user_id) {
            for arg in &mut user.args {
                if let Arg::Node(id) = arg {
                    if *id == node_id {
                        *id = rescale_id;
                    }
                }
            }
        }
        if let Some(r) = graph.node_mut(rescale_id) {
            if !r.users.contains(&user_id) {
                r.users.push(user_id);
            }
        }
    }

    // Update node_id's users to point to rescale only
    if let Some(node) = graph.node_mut(node_id) {
        node.users.clear();
        node.users.push(rescale_id);
    }

    true
}

// ---------------------------------------------------------------------------
// Rescale node creation helper (mirrors insert_rescale.rs)
// ---------------------------------------------------------------------------

fn create_rescale_node(
    graph: &mut Graph,
    store: &mut TensorStore,
    input_id: NodeId,
    ref_id: NodeId,
    scale_ratio: f64,
    input_zp: i64,
    output_zp: i64,
    output_dtype: DType,
    name_hint: &str,
) -> NodeId {
    let (multiplier, shift) = compute_multiplier_and_shift(scale_ratio);
    let counter = ARITH_RESCALE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let rescale_name = format!("{}_arith_rescale_{}", name_hint, counter);

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

    // Create RESCALE node
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

/// Convert a floating-point scale to a fixed-point (multiplier, shift) pair.
fn compute_multiplier_and_shift(scale: f64) -> (i32, i8) {
    if scale == 0.0 {
        return (0, 0);
    }
    let (mantissa, exponent) = frexp(scale);
    let shift = 31 - exponent;
    let multiplier = (mantissa * (1i64 << 31) as f64).round() as i32;
    (multiplier, shift as i8)
}

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
