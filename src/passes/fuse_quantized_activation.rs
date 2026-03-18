use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;

/// Fuse quantized activations (relu, hardtanh with min=0) into their predecessor.
///
/// This pass operates on the **pre-fold** graph where Q/DQ nodes are still
/// present. It mirrors the Python `FuseQuantizedActivationPass` exactly:
///
/// 1. Find relu (or hardtanh with min_val=0) nodes.
/// 2. Check that the single user is a QuantizePerTensor node with `zp == qmin`
///    (which is always true for symmetric int8: zp=-128, qmin=-128).
/// 3. Check that the relu's input is a fuseable op (conv, linear).
/// 4. If all conditions are met, bypass (remove) the relu node. The Q node
///    then directly follows the conv, and `FoldAndAnnotateQParamsPass` will
///    fold it as the conv's `output_qparams`.
///
/// This avoids creating separate CLAMP ops for relu.
pub struct FuseQuantizedActivationPass;

impl Pass for FuseQuantizedActivationPass {
    fn name(&self) -> &str {
        "FuseQuantizedActivation"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let node = match graph.node(node_id) {
                Some(n) => n,
                None => continue,
            };

            // Check if this is a fuseable activation (relu or hardtanh with min=0)
            let is_fuseable_activation = match &node.target {
                OpTarget::Relu => true,
                OpTarget::Hardtanh => {
                    let min_val = match node.args.get(1) {
                        Some(Arg::Float(f)) => *f,
                        Some(Arg::Int(i)) => *i as f64,
                        _ => -1.0,
                    };
                    min_val == 0.0
                }
                _ => continue,
            };

            if !is_fuseable_activation {
                continue;
            }

            // Check that the single user is a QuantizePerTensor node
            if node.users.len() != 1 {
                continue;
            }
            let q_node_id = node.users[0];
            let is_quantized_with_qmin_zp = match graph.node(q_node_id) {
                Some(q_node) if q_node.target == OpTarget::QuantizePerTensor => {
                    // Extract zp and qmin from Q node args:
                    // args: [input, scale, zero_point, quant_min, quant_max, dtype]
                    let zp = match q_node.args.get(2) {
                        Some(Arg::Int(i)) => *i,
                        _ => 0,
                    };
                    let qmin = match q_node.args.get(3) {
                        Some(Arg::Int(i)) => *i,
                        _ => -128,
                    };
                    zp == qmin
                }
                _ => false,
            };

            if !is_quantized_with_qmin_zp {
                continue;
            }

            // Check that the input is a fuseable op (conv or linear)
            let input_id = match node.args.first() {
                Some(Arg::Node(id)) => *id,
                _ => continue,
            };

            let is_fuseable_input = match graph.node(input_id) {
                Some(input_node) => {
                    matches!(
                        input_node.target,
                        OpTarget::Conv2d | OpTarget::DepthwiseConv2d | OpTarget::Linear
                    ) && input_node.users.len() == 1
                }
                None => false,
            };

            if !is_fuseable_input {
                continue;
            }

            // All conditions met: bypass the relu node
            graph.bypass_node(node_id);
            modified = true;
        }

        modified
    }
}
