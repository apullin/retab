use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;
use smallvec::smallvec;

// ---------------------------------------------------------------------------
// Helper: placeholder meta and name counter
// ---------------------------------------------------------------------------

fn placeholder_meta() -> NodeMeta {
    NodeMeta::new(TensorInfo::default())
}

// =========================================================================
// DecomposeSoftmaxPass
// =========================================================================
//
// softmax(x, dim) →
//   max_val = Amax(x, [dim], keepdim=true)
//   shifted = Sub(x, max_val)
//   exps    = Exp(shifted)
//   sum_exp = Sum(exps, [dim], keepdim=true)
//   recip   = Reciprocal(sum_exp)
//   result  = Mul(exps, recip)

pub struct DecomposeSoftmaxPass;

impl Pass for DecomposeSoftmaxPass {
    fn name(&self) -> &str {
        "DecomposeSoftmax"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Softmax);
        if targets.is_empty() {
            return false;
        }
        for id in targets {
            decompose_softmax(graph, id);
        }
        true
    }
}

fn decompose_softmax(graph: &mut Graph, sm_id: NodeId) {
    let (x_id, dim, output_info) = {
        let node = match graph.node(sm_id) {
            Some(n) => n,
            None => return,
        };
        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };
        let dim = match node.args.get(1) {
            Some(Arg::Int(d)) => *d,
            _ => return,
        };
        (x_id, dim, node.meta.val)
    };

    let dims_sv: smallvec::SmallVec<[i64; 6]> = smallvec![dim];

    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("softmax_decomp_{suffix}_{counter}")
    };

    // 1. max_val = Amax(x, [dim], keepdim=true)
    let max_id = graph.insert_before(
        sm_id,
        OpType::CallFunction,
        OpTarget::Amax,
        smallvec![
            Arg::Node(x_id),
            Arg::IntList(dims_sv.clone()),
            Arg::Bool(true),
        ],
        vec![],
        placeholder_meta(),
        next_name("amax"),
    );
    crate::shape::propagate_shape(graph, max_id);

    // 2. shifted = Sub(x, max_val)
    let shifted_id = graph.insert_before(
        sm_id,
        OpType::CallFunction,
        OpTarget::Sub,
        smallvec![Arg::Node(x_id), Arg::Node(max_id)],
        vec![],
        placeholder_meta(),
        next_name("sub"),
    );
    crate::shape::propagate_shape(graph, shifted_id);

    // 3. exps = Exp(shifted)
    let exp_id = graph.insert_before(
        sm_id,
        OpType::CallFunction,
        OpTarget::Exp,
        smallvec![Arg::Node(shifted_id)],
        vec![],
        placeholder_meta(),
        next_name("exp"),
    );
    crate::shape::propagate_shape(graph, exp_id);

    // 4. sum_exp = Sum(exps, [dim], keepdim=true)
    let sum_id = graph.insert_before(
        sm_id,
        OpType::CallFunction,
        OpTarget::Sum,
        smallvec![
            Arg::Node(exp_id),
            Arg::IntList(dims_sv),
            Arg::Bool(true),
        ],
        vec![],
        placeholder_meta(),
        next_name("sum"),
    );
    crate::shape::propagate_shape(graph, sum_id);

    // 5. recip = Reciprocal(sum_exp)
    let recip_id = graph.insert_before(
        sm_id,
        OpType::CallFunction,
        OpTarget::Reciprocal,
        smallvec![Arg::Node(sum_id)],
        vec![],
        placeholder_meta(),
        next_name("reciprocal"),
    );
    crate::shape::propagate_shape(graph, recip_id);

    // 6. result = Mul(exps, recip)
    let mul_id = graph.insert_before(
        sm_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(exp_id), Arg::Node(recip_id)],
        vec![],
        placeholder_meta(),
        next_name("mul"),
    );
    crate::shape::propagate_shape(graph, mul_id);

    if let Some(n) = graph.node_mut(mul_id) {
        n.meta.val = output_info;
    }

    graph.replace_all_uses(sm_id, mul_id);
    graph.erase_node(sm_id);
}

// =========================================================================
// DecomposeGeluPass
// =========================================================================
//
// gelu(x) ≈ x * 0.5 * (1 + erf(x / sqrt(2)))
//
// Since erf is not a TOSA primitive, use the tanh approximation:
//   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Decomposed as:
//   x3       = Mul(Mul(x, x), x)
//   scaled   = Mul(x3, Full(0.044715))
//   inner    = Add(x, scaled)
//   coeff    = Full(sqrt(2/pi))       // 0.7978845608
//   inner2   = Mul(inner, coeff)
//   t        = Tanh(inner2)
//   one      = Full(1.0)
//   t_plus_1 = Add(t, one)
//   half     = Full(0.5)
//   half_x   = Mul(x, half)
//   result   = Mul(half_x, t_plus_1)

pub struct DecomposeGeluPass;

impl Pass for DecomposeGeluPass {
    fn name(&self) -> &str {
        "DecomposeGelu"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Gelu);
        if targets.is_empty() {
            return false;
        }
        for id in targets {
            decompose_gelu(graph, id);
        }
        true
    }
}

fn decompose_gelu(graph: &mut Graph, gelu_id: NodeId) {
    let (x_id, output_info, dtype, rank) = {
        let node = match graph.node(gelu_id) {
            Some(n) => n,
            None => return,
        };
        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };
        let input_info = match graph.node(x_id) {
            Some(n) => n.meta.val,
            None => return,
        };
        (x_id, node.meta.val, input_info.dtype, input_info.shape_len as usize)
    };

    let scalar_shape: smallvec::SmallVec<[i64; 6]> = (0..rank).map(|_| 1i64).collect();

    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("gelu_decomp_{suffix}_{counter}")
    };

    let make_const = |graph: &mut Graph, ref_id: NodeId, val: f64, name: String| -> NodeId {
        let id = graph.insert_before(
            ref_id,
            OpType::CallFunction,
            OpTarget::Full,
            smallvec![Arg::IntList(scalar_shape.clone()), Arg::Float(val)],
            vec![("dtype".into(), Arg::Dtype(dtype))],
            placeholder_meta(),
            name,
        );
        crate::shape::propagate_shape(graph, id);
        id
    };

    // x2 = Mul(x, x)
    let x2_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(x_id), Arg::Node(x_id)],
        vec![],
        placeholder_meta(),
        next_name("x_sq"),
    );
    crate::shape::propagate_shape(graph, x2_id);

    // x3 = Mul(x2, x)
    let x3_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(x2_id), Arg::Node(x_id)],
        vec![],
        placeholder_meta(),
        next_name("x_cube"),
    );
    crate::shape::propagate_shape(graph, x3_id);

    // coeff_044715 = Full(0.044715)
    let c_id = make_const(graph, gelu_id, 0.044715, next_name("c_044715"));

    // scaled = Mul(x3, coeff_044715)
    let scaled_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(x3_id), Arg::Node(c_id)],
        vec![],
        placeholder_meta(),
        next_name("scaled"),
    );
    crate::shape::propagate_shape(graph, scaled_id);

    // inner = Add(x, scaled)
    let inner_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(x_id), Arg::Node(scaled_id)],
        vec![],
        placeholder_meta(),
        next_name("inner"),
    );
    crate::shape::propagate_shape(graph, inner_id);

    // sqrt_2_over_pi = Full(0.7978845608)
    let sqrt2pi_id = make_const(graph, gelu_id, 0.7978845608028654, next_name("sqrt2pi"));

    // inner2 = Mul(inner, sqrt_2_over_pi)
    let inner2_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(inner_id), Arg::Node(sqrt2pi_id)],
        vec![],
        placeholder_meta(),
        next_name("inner2"),
    );
    crate::shape::propagate_shape(graph, inner2_id);

    // t = Tanh(inner2)
    let tanh_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Tanh,
        smallvec![Arg::Node(inner2_id)],
        vec![],
        placeholder_meta(),
        next_name("tanh"),
    );
    crate::shape::propagate_shape(graph, tanh_id);

    // one = Full(1.0)
    let one_id = make_const(graph, gelu_id, 1.0, next_name("one"));

    // t_plus_1 = Add(t, one)
    let tp1_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(tanh_id), Arg::Node(one_id)],
        vec![],
        placeholder_meta(),
        next_name("t_plus_1"),
    );
    crate::shape::propagate_shape(graph, tp1_id);

    // half = Full(0.5)
    let half_id = make_const(graph, gelu_id, 0.5, next_name("half"));

    // half_x = Mul(x, half)
    let hx_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(x_id), Arg::Node(half_id)],
        vec![],
        placeholder_meta(),
        next_name("half_x"),
    );
    crate::shape::propagate_shape(graph, hx_id);

    // result = Mul(half_x, t_plus_1)
    let result_id = graph.insert_before(
        gelu_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(hx_id), Arg::Node(tp1_id)],
        vec![],
        placeholder_meta(),
        next_name("result"),
    );
    crate::shape::propagate_shape(graph, result_id);

    if let Some(n) = graph.node_mut(result_id) {
        n.meta.val = output_info;
    }

    graph.replace_all_uses(gelu_id, result_id);
    graph.erase_node(gelu_id);
}

// =========================================================================
// DecomposeBatchNormPass
// =========================================================================
//
// batch_norm(x, weight, bias, running_mean, running_var, training, momentum, eps)
// → (x - mean) / sqrt(var + eps) * weight + bias
//
// For inference (no_training): mean and var come from running_mean/running_var.

pub struct DecomposeBatchNormPass;

impl Pass for DecomposeBatchNormPass {
    fn name(&self) -> &str {
        "DecomposeBatchNorm"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::BatchNorm);
        if targets.is_empty() {
            return false;
        }
        for id in targets {
            decompose_batchnorm(graph, id);
        }
        true
    }
}

fn decompose_batchnorm(graph: &mut Graph, bn_id: NodeId) {
    // batch_norm args: [input, weight, bias, running_mean, running_var, ...]
    // For _native_batch_norm_legit_no_training:
    //   args: [input, weight, bias, running_mean, running_var, momentum, eps]
    let (x_id, weight_id, bias_id, mean_id, var_id, epsilon, input_info, output_info) = {
        let node = match graph.node(bn_id) {
            Some(n) => n,
            None => return,
        };

        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let weight_id = match node.args.get(1) {
            Some(Arg::Node(id)) => Some(*id),
            Some(Arg::OptionalNode(opt)) => *opt,
            _ => None,
        };

        let bias_id = match node.args.get(2) {
            Some(Arg::Node(id)) => Some(*id),
            Some(Arg::OptionalNode(opt)) => *opt,
            _ => None,
        };

        let mean_id = match node.args.get(3) {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let var_id = match node.args.get(4) {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        // eps is typically arg[6] or arg[7] depending on variant
        let epsilon = node
            .args
            .iter()
            .rev()
            .find_map(|a| {
                if let Arg::Float(f) = a {
                    Some(*f)
                } else {
                    None
                }
            })
            .unwrap_or(1e-5);

        let input_info = match graph.node(x_id) {
            Some(n) => n.meta.val,
            None => return,
        };

        (
            x_id,
            weight_id,
            bias_id,
            mean_id,
            var_id,
            epsilon,
            input_info,
            node.meta.val,
        )
    };

    let rank = input_info.shape_len as usize;
    let dtype = input_info.dtype;

    // Shape for broadcasting mean/var/weight/bias: [1, C, 1, 1] for 4D input
    let in_shape = input_info.shape_slice();
    let channel_shape: Vec<i64> = (0..rank)
        .map(|i| if i == 1 { in_shape[1] } else { 1 })
        .collect();
    let channel_shape_sv: smallvec::SmallVec<[i64; 6]> = channel_shape.iter().copied().collect();

    let eps_shape: smallvec::SmallVec<[i64; 6]> = (0..rank).map(|_| 1i64).collect();

    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("batchnorm_decomp_{suffix}_{counter}")
    };

    // 1. Reshape mean to [1, C, 1, 1]
    let mean_reshaped = graph.insert_before(
        bn_id,
        OpType::CallFunction,
        OpTarget::Reshape,
        smallvec![Arg::Node(mean_id), Arg::IntList(channel_shape_sv.clone())],
        vec![],
        placeholder_meta(),
        next_name("reshape_mean"),
    );
    crate::shape::propagate_shape(graph, mean_reshaped);

    // 2. diff = Sub(x, mean)
    let diff_id = graph.insert_before(
        bn_id,
        OpType::CallFunction,
        OpTarget::Sub,
        smallvec![Arg::Node(x_id), Arg::Node(mean_reshaped)],
        vec![],
        placeholder_meta(),
        next_name("sub_mean"),
    );
    crate::shape::propagate_shape(graph, diff_id);

    // 3. Reshape var to [1, C, 1, 1]
    let var_reshaped = graph.insert_before(
        bn_id,
        OpType::CallFunction,
        OpTarget::Reshape,
        smallvec![Arg::Node(var_id), Arg::IntList(channel_shape_sv.clone())],
        vec![],
        placeholder_meta(),
        next_name("reshape_var"),
    );
    crate::shape::propagate_shape(graph, var_reshaped);

    // 4. eps_tensor = Full(eps_shape, eps)
    let eps_tensor = graph.insert_before(
        bn_id,
        OpType::CallFunction,
        OpTarget::Full,
        smallvec![Arg::IntList(eps_shape), Arg::Float(epsilon)],
        vec![("dtype".into(), Arg::Dtype(dtype))],
        placeholder_meta(),
        next_name("full_eps"),
    );
    crate::shape::propagate_shape(graph, eps_tensor);

    // 5. var_eps = Add(var, eps)
    let var_eps = graph.insert_before(
        bn_id,
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(var_reshaped), Arg::Node(eps_tensor)],
        vec![],
        placeholder_meta(),
        next_name("add_eps"),
    );
    crate::shape::propagate_shape(graph, var_eps);

    // 6. inv_std = Rsqrt(var_eps)
    let inv_std = graph.insert_before(
        bn_id,
        OpType::CallFunction,
        OpTarget::Rsqrt,
        smallvec![Arg::Node(var_eps)],
        vec![],
        placeholder_meta(),
        next_name("rsqrt"),
    );
    crate::shape::propagate_shape(graph, inv_std);

    // 7. normalized = Mul(diff, inv_std)
    let normalized = graph.insert_before(
        bn_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(diff_id), Arg::Node(inv_std)],
        vec![],
        placeholder_meta(),
        next_name("mul_norm"),
    );
    crate::shape::propagate_shape(graph, normalized);

    let mut final_id = normalized;

    // 8. If weight: reshape weight to [1, C, 1, 1], mul
    if let Some(w_id) = weight_id {
        let w_reshaped = graph.insert_before(
            bn_id,
            OpType::CallFunction,
            OpTarget::Reshape,
            smallvec![Arg::Node(w_id), Arg::IntList(channel_shape_sv.clone())],
            vec![],
            placeholder_meta(),
            next_name("reshape_weight"),
        );
        crate::shape::propagate_shape(graph, w_reshaped);

        let mul_w = graph.insert_before(
            bn_id,
            OpType::CallFunction,
            OpTarget::Mul,
            smallvec![Arg::Node(final_id), Arg::Node(w_reshaped)],
            vec![],
            placeholder_meta(),
            next_name("mul_weight"),
        );
        crate::shape::propagate_shape(graph, mul_w);
        final_id = mul_w;
    }

    // 9. If bias: reshape bias to [1, C, 1, 1], add
    if let Some(b_id) = bias_id {
        let b_reshaped = graph.insert_before(
            bn_id,
            OpType::CallFunction,
            OpTarget::Reshape,
            smallvec![Arg::Node(b_id), Arg::IntList(channel_shape_sv)],
            vec![],
            placeholder_meta(),
            next_name("reshape_bias"),
        );
        crate::shape::propagate_shape(graph, b_reshaped);

        let add_b = graph.insert_before(
            bn_id,
            OpType::CallFunction,
            OpTarget::Add,
            smallvec![Arg::Node(final_id), Arg::Node(b_reshaped)],
            vec![],
            placeholder_meta(),
            next_name("add_bias"),
        );
        crate::shape::propagate_shape(graph, add_b);
        final_id = add_b;
    }

    if let Some(n) = graph.node_mut(final_id) {
        n.meta.val = output_info;
    }

    graph.replace_all_uses(bn_id, final_id);
    graph.erase_node(bn_id);
}

// =========================================================================
// DecomposeGroupNormPass
// =========================================================================
//
// group_norm(x, num_groups, weight, bias, eps) →
//   Reshape x from (N, C, *) to (N, G, C/G, *)
//   Compute mean and var over the (C/G, *) dims
//   Normalize: (x - mean) / sqrt(var + eps)
//   Reshape back, apply weight and bias
//
// Simplified: treat each group as a LayerNorm-like computation.
// Since we decompose into Mean/Var/Sub/Mul which further decompose,
// we reshape → normalize → reshape back.

pub struct DecomposeGroupNormPass;

impl Pass for DecomposeGroupNormPass {
    fn name(&self) -> &str {
        "DecomposeGroupNorm"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::GroupNorm);
        if targets.is_empty() {
            return false;
        }
        for id in targets {
            decompose_groupnorm(graph, id);
        }
        true
    }
}

fn decompose_groupnorm(graph: &mut Graph, gn_id: NodeId) {
    // group_norm args: [input, num_groups, weight, bias, eps]
    let (x_id, num_groups, weight_id, bias_id, epsilon, input_info, output_info) = {
        let node = match graph.node(gn_id) {
            Some(n) => n,
            None => return,
        };

        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let num_groups = match node.args.get(1) {
            Some(Arg::Int(g)) => *g,
            _ => return,
        };

        let weight_id = match node.args.get(2) {
            Some(Arg::Node(id)) => Some(*id),
            Some(Arg::OptionalNode(opt)) => *opt,
            _ => None,
        };

        let bias_id = match node.args.get(3) {
            Some(Arg::Node(id)) => Some(*id),
            Some(Arg::OptionalNode(opt)) => *opt,
            _ => None,
        };

        let epsilon = match node.args.get(4) {
            Some(Arg::Float(f)) => *f,
            _ => 1e-5,
        };

        let input_info = match graph.node(x_id) {
            Some(n) => n.meta.val,
            None => return,
        };

        (
            x_id,
            num_groups,
            weight_id,
            bias_id,
            epsilon,
            input_info,
            node.meta.val,
        )
    };

    let in_shape = input_info.shape_slice();
    let rank = in_shape.len();
    if rank < 2 {
        return;
    }

    let n = in_shape[0];
    let c = in_shape[1];
    let channels_per_group = c / num_groups;
    let dtype = input_info.dtype;

    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("groupnorm_decomp_{suffix}_{counter}")
    };

    // Build grouped shape: (N, G, C/G, *spatial)
    // For 4D: (N, G, C/G, H, W) — but TensorInfo only supports up to 4 dims.
    // Strategy: reshape to (N*G, C/G, H, W) and do mean/var on dims [1, 2, 3]
    // then reshape back. But 4D limit applies.
    //
    // Simpler approach for 4D (N,C,H,W):
    //   reshape to (N*G, C/G*H*W) — 2D
    //   mean/var on dim [1], keepdim=true
    //   normalize, reshape back to (N,C,H,W)
    //   apply weight/bias

    // Compute spatial product for dims >= 2
    let spatial_product: i64 = in_shape[2..].iter().product();
    let flat_features = channels_per_group * spatial_product;

    // Reshape to (N*G, C/G * H * W)
    let grouped_shape: smallvec::SmallVec<[i64; 6]> =
        smallvec![n * num_groups, flat_features];

    let reshape1_id = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Reshape,
        smallvec![Arg::Node(x_id), Arg::IntList(grouped_shape)],
        vec![],
        placeholder_meta(),
        next_name("reshape_grouped"),
    );
    crate::shape::propagate_shape(graph, reshape1_id);

    // mean on dim [1], keepdim=true
    let reduce_dims: smallvec::SmallVec<[i64; 6]> = smallvec![1];

    let mean_id = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Mean,
        smallvec![
            Arg::Node(reshape1_id),
            Arg::IntList(reduce_dims.clone()),
            Arg::Bool(true),
        ],
        vec![],
        placeholder_meta(),
        next_name("mean"),
    );
    crate::shape::propagate_shape(graph, mean_id);

    // diff = Sub(reshaped, mean)
    let diff_id = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Sub,
        smallvec![Arg::Node(reshape1_id), Arg::Node(mean_id)],
        vec![],
        placeholder_meta(),
        next_name("sub"),
    );
    crate::shape::propagate_shape(graph, diff_id);

    // var on dim [1], correction=0, keepdim=true
    let var_id = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Var,
        smallvec![Arg::Node(reshape1_id), Arg::IntList(reduce_dims)],
        vec![
            ("correction".into(), Arg::Int(0)),
            ("keepdim".into(), Arg::Bool(true)),
        ],
        placeholder_meta(),
        next_name("var"),
    );
    crate::shape::propagate_shape(graph, var_id);

    // eps_tensor
    let eps_shape: smallvec::SmallVec<[i64; 6]> = smallvec![1, 1];
    let eps_id = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Full,
        smallvec![Arg::IntList(eps_shape), Arg::Float(epsilon)],
        vec![("dtype".into(), Arg::Dtype(dtype))],
        placeholder_meta(),
        next_name("full_eps"),
    );
    crate::shape::propagate_shape(graph, eps_id);

    // var_eps = Add(var, eps)
    let var_eps = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(var_id), Arg::Node(eps_id)],
        vec![],
        placeholder_meta(),
        next_name("add_eps"),
    );
    crate::shape::propagate_shape(graph, var_eps);

    // inv_std = Rsqrt(var_eps)
    let inv_std = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Rsqrt,
        smallvec![Arg::Node(var_eps)],
        vec![],
        placeholder_meta(),
        next_name("rsqrt"),
    );
    crate::shape::propagate_shape(graph, inv_std);

    // normalized = Mul(diff, inv_std)
    let normalized = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(diff_id), Arg::Node(inv_std)],
        vec![],
        placeholder_meta(),
        next_name("mul_norm"),
    );
    crate::shape::propagate_shape(graph, normalized);

    // Reshape back to original shape
    let orig_shape_sv: smallvec::SmallVec<[i64; 6]> = in_shape.iter().copied().collect();
    let reshape_back = graph.insert_before(
        gn_id,
        OpType::CallFunction,
        OpTarget::Reshape,
        smallvec![Arg::Node(normalized), Arg::IntList(orig_shape_sv)],
        vec![],
        placeholder_meta(),
        next_name("reshape_back"),
    );
    crate::shape::propagate_shape(graph, reshape_back);

    let mut final_id = reshape_back;

    // Apply weight: channel-wise, shape [1, C, 1, 1]
    if let Some(w_id) = weight_id {
        let weight_shape: Vec<i64> = (0..rank)
            .map(|i| if i == 1 { c } else { 1 })
            .collect();
        let weight_shape_sv: smallvec::SmallVec<[i64; 6]> = weight_shape.iter().copied().collect();

        let w_reshaped = graph.insert_before(
            gn_id,
            OpType::CallFunction,
            OpTarget::Reshape,
            smallvec![Arg::Node(w_id), Arg::IntList(weight_shape_sv.clone())],
            vec![],
            placeholder_meta(),
            next_name("reshape_weight"),
        );
        crate::shape::propagate_shape(graph, w_reshaped);

        let mul_w = graph.insert_before(
            gn_id,
            OpType::CallFunction,
            OpTarget::Mul,
            smallvec![Arg::Node(final_id), Arg::Node(w_reshaped)],
            vec![],
            placeholder_meta(),
            next_name("mul_weight"),
        );
        crate::shape::propagate_shape(graph, mul_w);
        final_id = mul_w;

        if let Some(b_id) = bias_id {
            let b_reshaped = graph.insert_before(
                gn_id,
                OpType::CallFunction,
                OpTarget::Reshape,
                smallvec![Arg::Node(b_id), Arg::IntList(weight_shape_sv)],
                vec![],
                placeholder_meta(),
                next_name("reshape_bias"),
            );
            crate::shape::propagate_shape(graph, b_reshaped);

            let add_b = graph.insert_before(
                gn_id,
                OpType::CallFunction,
                OpTarget::Add,
                smallvec![Arg::Node(final_id), Arg::Node(b_reshaped)],
                vec![],
                placeholder_meta(),
                next_name("add_bias"),
            );
            crate::shape::propagate_shape(graph, add_b);
            final_id = add_b;
        }
    }

    if let Some(n) = graph.node_mut(final_id) {
        n.meta.val = output_info;
    }

    graph.replace_all_uses(gn_id, final_id);
    graph.erase_node(gn_id);
}

// =========================================================================
// DecomposeLinearPass
// =========================================================================
//
// Following the Python reference: decompose into Conv2D with reshapes.
//   linear(x, weight, bias) →
//     x_reshaped   = Reshape(x, [N, Ci, 1, 1])
//     w_reshaped   = Reshape(weight, [Co, Ci, 1, 1])
//     conv         = Conv2d(x_reshaped, w_reshaped, bias, stride=[1,1], pad=[0,0], dil=[1,1])
//     result       = Reshape(conv, output_shape)

pub struct DecomposeLinearPass;

impl Pass for DecomposeLinearPass {
    fn name(&self) -> &str {
        "DecomposeLinear"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Linear);
        if targets.is_empty() {
            return false;
        }
        for id in targets {
            decompose_linear(graph, id);
        }
        true
    }
}

fn decompose_linear(graph: &mut Graph, lin_id: NodeId) {
    let (x_id, weight_id, bias_id, input_info, weight_info, output_info, lin_input_qparams, lin_output_qparams) = {
        let node = match graph.node(lin_id) {
            Some(n) => n,
            None => return,
        };

        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let weight_id = match node.args.get(1) {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let bias_id = match node.args.get(2) {
            Some(Arg::Node(id)) => Some(*id),
            Some(Arg::OptionalNode(opt)) => *opt,
            _ => None,
        };

        let input_info = match graph.node(x_id) {
            Some(n) => n.meta.val,
            None => return,
        };

        let weight_info = match graph.node(weight_id) {
            Some(n) => n.meta.val,
            None => return,
        };

        // Capture qparams from the original linear node (set by fold_qdq)
        let iqp = node.meta.input_qparams.clone();
        let oqp = node.meta.output_qparams.clone();

        (x_id, weight_id, bias_id, input_info, weight_info, node.meta.val, iqp, oqp)
    };

    let in_shape = input_info.shape_slice();
    let w_shape = weight_info.shape_slice();

    if in_shape.is_empty() || w_shape.len() < 2 {
        return;
    }

    let ci = in_shape[in_shape.len() - 1];
    let co = w_shape[0];

    // Batch = product of all dims except last
    let batch: i64 = in_shape[..in_shape.len() - 1].iter().product();

    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("linear_decomp_{suffix}_{counter}")
    };

    // Reshape input to (N, Ci, 1, 1).
    // Optimization: if the input is already a Reshape producing a 2D tensor
    // (e.g., from nn.Flatten), modify that reshape's target shape in-place
    // to produce [N, Ci, 1, 1] directly, avoiding an extra reshape op.
    let x_shape_4d: smallvec::SmallVec<[i64; 6]> = smallvec![batch, ci, 1, 1];
    let input_is_reshape_2d = graph.node(x_id).map(|n| {
        (n.target == OpTarget::Reshape || n.target == OpTarget::View
            || n.target == OpTarget::TosaReshape || n.target == OpTarget::Flatten)
            && n.meta.val.shape_len == 2
    }).unwrap_or(false);

    let x_reshaped = if input_is_reshape_2d {
        // Modify existing reshape to output [N, Ci, 1, 1] instead of [N, Ci]
        if let Some(node) = graph.node_mut(x_id) {
            // Update the shape argument (arg 1) to the 4D shape
            if node.args.len() >= 2 {
                node.args[1] = Arg::IntList(x_shape_4d);
            }
        }
        crate::shape::propagate_shape(graph, x_id);
        x_id
    } else {
        let x_reshaped = graph.insert_before(
            lin_id,
            OpType::CallFunction,
            OpTarget::Reshape,
            smallvec![Arg::Node(x_id), Arg::IntList(x_shape_4d)],
            vec![],
            placeholder_meta(),
            next_name("reshape_input"),
        );
        crate::shape::propagate_shape(graph, x_reshaped);
        x_reshaped
    };

    // Reshape weight to (Co, Ci, 1, 1)
    let w_shape_4d: smallvec::SmallVec<[i64; 6]> = smallvec![co, ci, 1, 1];
    let w_reshaped = graph.insert_before(
        lin_id,
        OpType::CallFunction,
        OpTarget::Reshape,
        smallvec![Arg::Node(weight_id), Arg::IntList(w_shape_4d)],
        vec![],
        placeholder_meta(),
        next_name("reshape_weight"),
    );
    crate::shape::propagate_shape(graph, w_reshaped);

    // Conv2d(x_reshaped, w_reshaped, bias, stride=[1,1], padding=[0,0], dilation=[1,1])
    let bias_arg = match bias_id {
        Some(id) => Arg::Node(id),
        None => Arg::OptionalNode(None),
    };

    let conv_meta = {
        let mut m = placeholder_meta();
        // Inherit qparams from the original linear node (matches Python's
        // inherit_qparams=True in create_node for the decomposed conv).
        m.input_qparams = lin_input_qparams;
        m.output_qparams = lin_output_qparams;
        m
    };
    let conv_id = graph.insert_before(
        lin_id,
        OpType::CallFunction,
        OpTarget::Conv2d,
        smallvec![
            Arg::Node(x_reshaped),
            Arg::Node(w_reshaped),
            bias_arg,
            Arg::IntList(smallvec![1, 1]),  // stride
            Arg::IntList(smallvec![0, 0]),  // padding
            Arg::IntList(smallvec![1, 1]),  // dilation
        ],
        vec![],
        conv_meta,
        next_name("conv2d"),
    );
    crate::shape::propagate_shape(graph, conv_id);

    // Reshape output back to original output shape
    let out_shape = output_info.shape_slice();
    let out_shape_sv: smallvec::SmallVec<[i64; 6]> = out_shape.iter().copied().collect();
    let result_id = graph.insert_before(
        lin_id,
        OpType::CallFunction,
        OpTarget::Reshape,
        smallvec![Arg::Node(conv_id), Arg::IntList(out_shape_sv)],
        vec![],
        placeholder_meta(),
        next_name("reshape_output"),
    );
    crate::shape::propagate_shape(graph, result_id);

    if let Some(n) = graph.node_mut(result_id) {
        n.meta.val = output_info;
    }

    graph.replace_all_uses(lin_id, result_id);
    graph.erase_node(lin_id);
}

// =========================================================================
// DecomposeAddmmPass
// =========================================================================
//
// addmm(bias, x, weight) → matmul(x, weight) + bias

pub struct DecomposeAddmmPass;

impl Pass for DecomposeAddmmPass {
    fn name(&self) -> &str {
        "DecomposeAddmm"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::Addmm);
        if targets.is_empty() {
            return false;
        }
        for id in targets {
            decompose_addmm(graph, id);
        }
        true
    }
}

fn decompose_addmm(graph: &mut Graph, addmm_id: NodeId) {
    let (bias_id, x_id, weight_id, output_info) = {
        let node = match graph.node(addmm_id) {
            Some(n) => n,
            None => return,
        };

        let bias_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let x_id = match node.args.get(1) {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        let weight_id = match node.args.get(2) {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        (bias_id, x_id, weight_id, node.meta.val)
    };

    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("addmm_decomp_{suffix}_{counter}")
    };

    // matmul(x, weight)
    let mm_id = graph.insert_before(
        addmm_id,
        OpType::CallFunction,
        OpTarget::Matmul,
        smallvec![Arg::Node(x_id), Arg::Node(weight_id)],
        vec![],
        placeholder_meta(),
        next_name("matmul"),
    );
    crate::shape::propagate_shape(graph, mm_id);

    // result = Add(matmul, bias)
    let add_id = graph.insert_before(
        addmm_id,
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(mm_id), Arg::Node(bias_id)],
        vec![],
        placeholder_meta(),
        next_name("add_bias"),
    );
    crate::shape::propagate_shape(graph, add_id);

    if let Some(n) = graph.node_mut(add_id) {
        n.meta.val = output_info;
    }

    graph.replace_all_uses(addmm_id, add_id);
    graph.erase_node(addmm_id);
}
