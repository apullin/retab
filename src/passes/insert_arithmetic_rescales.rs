use std::collections::HashSet;

use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Insert TOSA RESCALE nodes around quantized INT32-accumulator ops to match
/// the Python `InsertRescaleInt32Pass` behavior.
///
/// The Python pass handles these op categories:
///
/// - **ADD/SUB**: Rescale both inputs to `2 * max(s_a, s_b)` scale, output
///   at the same scale. `InsertRescalePass` mismatch detection adds the
///   output rescale to the consumer's expected scale.
/// - **MUL**: Input scales unchanged; output has natural scale `s_a * s_b`.
///   Insert output rescale to the desired output scale.
/// - **SUM**: Input rescaled from int8 to int32 (same scale, just dtype
///   upcast). Output rescaled from int32 to int8.
/// - **ABS**: Same as SUM — identity rescale in, identity rescale out.
/// - **MAXIMUM/MINIMUM**: Both inputs rescaled to `min(s_a, s_b)`, output
///   at the same min scale.
/// - **Comparisons (eq, ge, gt, le, lt)**: Inputs rescaled to
///   `min(s_a, s_b)`. Output is bool — no output rescale needed.
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
                OpTarget::Sum | OpTarget::TosaReduceSum => {
                    if insert_sum_rescales(graph, store, node_id, debug) {
                        modified = true;
                    }
                    processed.insert(node_id);
                }
                OpTarget::Abs | OpTarget::TosaAbs => {
                    if insert_abs_rescales(graph, store, node_id, debug) {
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
// SUM rescale insertion
// ---------------------------------------------------------------------------

/// For TOSA SUM/ABS, the computation happens in int32.
/// Per the Python `InsertRescaleInt32Pass`:
/// - Input rescale: int8 -> int32, same scale (identity upcast, removes zp)
/// - Output rescale: int32 -> int8, from int32 scale to the desired output scale
///
/// Both input AND output rescales are always inserted (matching Python behavior),
/// because the op needs to run in int32 domain even when input and output scales
/// are the same.
fn insert_sum_rescales(
    graph: &mut Graph,
    store: &mut TensorStore,
    node_id: NodeId,
    debug: bool,
) -> bool {
    let (qp_input, input_id, existing_out_qp, node_name) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        let qp = node
            .meta
            .input_qparams
            .iter()
            .find(|(p, _)| *p == 0)
            .map(|(_, q)| *q);
        let input = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };
        let out = node
            .meta
            .output_qparams
            .iter()
            .find(|(p, _)| *p == 0)
            .map(|(_, q)| *q);

        (qp, input, out, node.name.clone())
    };

    let qp = match qp_input {
        Some(q) if q.dtype == DType::Int8 || q.dtype == DType::Int16 => q,
        _ => return false,
    };

    // Guard: if input is already a rescale, skip
    let input_is_rescale = graph
        .node(input_id)
        .map(|n| n.target == OpTarget::TosaRescale)
        .unwrap_or(false);
    if input_is_rescale {
        return false;
    }

    if debug {
        eprintln!(
            "[InsertArithRescale] SUM/ABS '{}': input_scale={}, input_zp={}",
            node_name, qp.scale, qp.zero_point,
        );
    }

    // The int32 domain qparams: same scale, zero zp
    let int32_qp = QuantArgs {
        scale: qp.scale,
        zero_point: 0,
        dtype: qp.dtype,
        qmin: i32::MIN as i64,
        qmax: i32::MAX as i64,
    };

    // Insert input rescale: int8 -> int32 (scale ratio = 1.0, but zp changes)
    let rescale_in_id = create_rescale_node(
        graph,
        store,
        input_id,
        node_id,
        1.0, // identity scale: old_scale / new_scale = s/s
        qp.zero_point, // input_zp: original zp
        0,              // output_zp: 0 (int32 domain)
        qp.dtype,
        &format!("{}_int32_in", node_name),
    );

    // Set output_qparams on input rescale
    if let Some(r) = graph.node_mut(rescale_in_id) {
        r.meta.output_qparams.push((0, int32_qp));
    }

    // Rewire: replace input in the node's arg 0
    if let Some(node) = graph.node_mut(node_id) {
        if let Some(Arg::Node(id)) = node.args.first_mut() {
            *id = rescale_in_id;
        }
        for entry in &mut node.meta.input_qparams {
            if entry.0 == 0 {
                entry.1 = int32_qp;
            }
        }
    }

    // Update user lists for input rescale
    if let Some(r) = graph.node_mut(rescale_in_id) {
        if !r.users.contains(&node_id) {
            r.users.push(node_id);
        }
    }
    if let Some(a) = graph.node_mut(input_id) {
        a.users.retain(|u| *u != node_id);
        if !a.users.contains(&rescale_in_id) {
            a.users.push(rescale_in_id);
        }
    }

    // Determine the desired output qparams. Use existing output_qparams if
    // available (set by fold_qdq or propagation), otherwise fall back to the
    // original input qparams (the int8 domain values before upcast).
    let desired_out_qp = existing_out_qp.unwrap_or(qp);

    // Set the node's output_qparams to the int32 domain scale.
    if let Some(node) = graph.node_mut(node_id) {
        node.meta.output_qparams.retain(|(p, _)| *p != 0);
        node.meta.output_qparams.push((0, int32_qp));
    }

    // Always insert output rescale: int32 -> int8.
    // The output rescale converts from the int32 accumulator domain back to
    // int8 at the desired output scale.
    let users: Vec<NodeId> = graph
        .node(node_id)
        .map(|n| n.users.to_vec())
        .unwrap_or_default();

    if users.is_empty() {
        return true; // Input rescale was still inserted
    }

    let first_user = users[0];
    // Scale ratio: int32_scale / desired_out_scale = input_scale / desired_out_scale
    let out_rescale_scale = int32_qp.scale / desired_out_qp.scale;

    let rescale_out_id = create_rescale_node(
        graph,
        store,
        node_id,
        first_user,
        out_rescale_scale,
        0,                        // input_zp: 0 (int32 domain)
        desired_out_qp.zero_point, // output_zp
        desired_out_qp.dtype,
        &format!("{}_int32_out", node_name),
    );

    if let Some(r) = graph.node_mut(rescale_out_id) {
        r.meta.output_qparams.push((0, desired_out_qp));
    }

    // Rewire ALL users to use the output rescale
    for &user_id in &users {
        if let Some(user) = graph.node_mut(user_id) {
            for arg in &mut user.args {
                if let Arg::Node(id) = arg {
                    if *id == node_id {
                        *id = rescale_out_id;
                    }
                }
            }
        }
        if let Some(r) = graph.node_mut(rescale_out_id) {
            if !r.users.contains(&user_id) {
                r.users.push(user_id);
            }
        }
    }

    if let Some(node) = graph.node_mut(node_id) {
        node.users.clear();
        node.users.push(rescale_out_id);
    }

    true
}

// ---------------------------------------------------------------------------
// ABS rescale insertion
// ---------------------------------------------------------------------------

/// For ABS, the Python rescales input from int8 to int32 (same scale,
/// just dtype upcast) and then rescales output back to int8.
fn insert_abs_rescales(
    graph: &mut Graph,
    store: &mut TensorStore,
    node_id: NodeId,
    debug: bool,
) -> bool {
    // ABS is identical to SUM in rescale behavior: identity in, identity out
    insert_sum_rescales(graph, store, node_id, debug)
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
