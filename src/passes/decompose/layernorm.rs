use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;
use smallvec::smallvec;

/// Decompose `LayerNorm(x, normalized_shape, weight, bias, eps)` into:
///   mean = Mean(x, dims, keepdim=true)
///   diff = Sub(x, mean)
///   var  = Var(x, dims, correction=0, keepdim=true)
///   eps_tensor = Full(eps_shape, eps)
///   var_eps = Add(var, eps_tensor)
///   inv_std = Rsqrt(var_eps)
///   normalized = Mul(diff, inv_std)
///   if weight: result = Add(Mul(normalized, Reshape(weight, shape)), Reshape(bias, shape))
pub struct DecomposeLayerNormPass;

impl Pass for DecomposeLayerNormPass {
    fn name(&self) -> &str {
        "DecomposeLayerNorm"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let targets: Vec<NodeId> = graph.nodes_by_target(&OpTarget::LayerNorm);
        if targets.is_empty() {
            return false;
        }

        for ln_id in targets {
            decompose_one(graph, ln_id);
        }
        true
    }
}

fn decompose_one(graph: &mut Graph, ln_id: NodeId) {
    // Extract info from the LayerNorm node before mutating graph.
    let (x_id, normalized_shape, weight_id, bias_id, epsilon, input_info, output_info) = {
        let node = match graph.node(ln_id) {
            Some(n) => n,
            None => return,
        };

        let x_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return,
        };

        // normalized_shape: IntList at arg[1]
        let normalized_shape: Vec<i64> = match node.args.get(1) {
            Some(Arg::IntList(v)) => v.to_vec(),
            _ => return,
        };

        // weight: OptionalNode or Node at arg[2]
        let weight_id = match node.args.get(2) {
            Some(Arg::Node(id)) => Some(*id),
            Some(Arg::OptionalNode(opt)) => *opt,
            _ => None,
        };

        // bias: OptionalNode or Node at arg[3]
        let bias_id = match node.args.get(3) {
            Some(Arg::Node(id)) => Some(*id),
            Some(Arg::OptionalNode(opt)) => *opt,
            _ => None,
        };

        // eps: Float at arg[4], default to f32 machine epsilon
        let epsilon = match node.args.get(4) {
            Some(Arg::Float(f)) => *f,
            _ => f32::EPSILON as f64,
        };

        let input_info = match graph.node(x_id) {
            Some(n) => n.meta.val,
            None => return,
        };

        let output_info = node.meta.val;

        (
            x_id,
            normalized_shape,
            weight_id,
            bias_id,
            epsilon,
            input_info,
            output_info,
        )
    };

    let rank = input_info.shape_len as usize;
    let n_dims = normalized_shape.len();
    // dims to reduce: the last n_dims dimensions
    let dims: Vec<i64> = (0..n_dims).map(|i| (rank - n_dims + i) as i64).collect();
    let dims_sv: smallvec::SmallVec<[i64; 6]> = dims.iter().copied().collect();

    let in_shape = input_info.shape_slice();

    // Shape for epsilon tensor: all 1s with same rank
    let eps_shape: Vec<i64> = vec![1; rank];
    let eps_shape_sv: smallvec::SmallVec<[i64; 6]> = eps_shape.iter().copied().collect();

    // Shape for weight/bias reshape: 1 for non-reduced dims, original size for reduced dims
    let weights_reshaped_shape: Vec<i64> = (0..rank)
        .map(|i| {
            if dims.contains(&(i as i64)) {
                in_shape[i]
            } else {
                1
            }
        })
        .collect();
    let reshape_shape_sv: smallvec::SmallVec<[i64; 6]> =
        weights_reshaped_shape.iter().copied().collect();

    let dtype = input_info.dtype;
    let mut counter = 0u32;
    let mut next_name = |suffix: &str| -> String {
        counter += 1;
        format!("layernorm_decomp_{suffix}_{counter}")
    };

    let placeholder_meta = || NodeMeta::new(TensorInfo::default());

    // 1. mean = Mean(x, dims, keepdim=true)
    let mean_id = graph.insert_before(
        ln_id,
        OpType::CallFunction,
        OpTarget::Mean,
        smallvec![
            Arg::Node(x_id),
            Arg::IntList(dims_sv.clone()),
            Arg::Bool(true),
        ],
        vec![],
        placeholder_meta(),
        next_name("mean"),
    );
    crate::shape::propagate_shape(graph, mean_id);

    // 2. diff = Sub(x, mean)
    let diff_id = graph.insert_before(
        ln_id,
        OpType::CallFunction,
        OpTarget::Sub,
        smallvec![Arg::Node(x_id), Arg::Node(mean_id)],
        vec![],
        placeholder_meta(),
        next_name("sub"),
    );
    crate::shape::propagate_shape(graph, diff_id);

    // 3. var = Var(x, dims, correction=0, keepdim=true)
    let var_id = graph.insert_before(
        ln_id,
        OpType::CallFunction,
        OpTarget::Var,
        smallvec![Arg::Node(x_id), Arg::IntList(dims_sv.clone())],
        vec![
            ("correction".into(), Arg::Int(0)),
            ("keepdim".into(), Arg::Bool(true)),
        ],
        placeholder_meta(),
        next_name("var"),
    );
    crate::shape::propagate_shape(graph, var_id);

    // 4. eps_tensor = Full(eps_shape, eps)
    let eps_id = graph.insert_before(
        ln_id,
        OpType::CallFunction,
        OpTarget::Full,
        smallvec![Arg::IntList(eps_shape_sv), Arg::Float(epsilon)],
        vec![("dtype".into(), Arg::Dtype(dtype))],
        placeholder_meta(),
        next_name("full_eps"),
    );
    crate::shape::propagate_shape(graph, eps_id);

    // 5. var_eps = Add(var, eps_tensor)
    let var_eps_id = graph.insert_before(
        ln_id,
        OpType::CallFunction,
        OpTarget::Add,
        smallvec![Arg::Node(var_id), Arg::Node(eps_id)],
        vec![],
        placeholder_meta(),
        next_name("add_eps"),
    );
    crate::shape::propagate_shape(graph, var_eps_id);

    // 6. inv_std = Rsqrt(var_eps)
    let inv_std_id = graph.insert_before(
        ln_id,
        OpType::CallFunction,
        OpTarget::Rsqrt,
        smallvec![Arg::Node(var_eps_id)],
        vec![],
        placeholder_meta(),
        next_name("rsqrt"),
    );
    crate::shape::propagate_shape(graph, inv_std_id);

    // 7. normalized = Mul(diff, inv_std)
    let normalized_id = graph.insert_before(
        ln_id,
        OpType::CallFunction,
        OpTarget::Mul,
        smallvec![Arg::Node(diff_id), Arg::Node(inv_std_id)],
        vec![],
        placeholder_meta(),
        next_name("mul_norm"),
    );
    crate::shape::propagate_shape(graph, normalized_id);

    // 8. If weight: reshape weight, mul, then if bias: reshape bias, add
    let mut final_id = normalized_id;

    if let Some(w_id) = weight_id {
        // Reshape weight to broadcastable shape
        let w_reshaped_id = graph.insert_before(
            ln_id,
            OpType::CallFunction,
            OpTarget::Reshape,
            smallvec![Arg::Node(w_id), Arg::IntList(reshape_shape_sv.clone())],
            vec![],
            placeholder_meta(),
            next_name("reshape_weight"),
        );
        crate::shape::propagate_shape(graph, w_reshaped_id);

        let mul_w_id = graph.insert_before(
            ln_id,
            OpType::CallFunction,
            OpTarget::Mul,
            smallvec![Arg::Node(normalized_id), Arg::Node(w_reshaped_id)],
            vec![],
            placeholder_meta(),
            next_name("mul_weight"),
        );
        crate::shape::propagate_shape(graph, mul_w_id);
        final_id = mul_w_id;
    }

    if let Some(b_id) = bias_id {
        // Reshape bias to broadcastable shape
        let b_reshaped_id = graph.insert_before(
            ln_id,
            OpType::CallFunction,
            OpTarget::Reshape,
            smallvec![Arg::Node(b_id), Arg::IntList(reshape_shape_sv)],
            vec![],
            placeholder_meta(),
            next_name("reshape_bias"),
        );
        crate::shape::propagate_shape(graph, b_reshaped_id);

        let add_b_id = graph.insert_before(
            ln_id,
            OpType::CallFunction,
            OpTarget::Add,
            smallvec![Arg::Node(final_id), Arg::Node(b_reshaped_id)],
            vec![],
            placeholder_meta(),
            next_name("add_bias"),
        );
        crate::shape::propagate_shape(graph, add_b_id);
        final_id = add_b_id;
    }

    // Propagate the output dtype/qparams from the original LN node
    if let Some(final_node) = graph.node_mut(final_id) {
        final_node.meta.val = output_info;
    }

    graph.replace_all_uses(ln_id, final_id);
    graph.erase_node(ln_id);
}
