use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;

/// Convert `Relu` and `Hardtanh` nodes to `Clamp`.
///
/// - Relu(x) -> Clamp(x, min=0.0, max=f32::MAX)
/// - Hardtanh(x, min_val, max_val) -> Clamp(x, min=min_val, max=max_val)
pub struct ConvertToClampPass;

impl Pass for ConvertToClampPass {
    fn name(&self) -> &str {
        "ConvertToClamp"
    }

    fn run(&self, graph: &mut Graph, _store: &mut TensorStore) -> bool {
        let mut modified = false;

        let all_ids = graph.all_node_ids();
        for id in all_ids {
            let node = match graph.node(id) {
                Some(n) => n,
                None => continue,
            };

            match &node.target {
                OpTarget::Relu => {
                    // Relu: clamp(x, min=0, max=MAX)
                    // For quantized int8 models: relu clips to [0, 127]
                    // For FP32: relu clips to [0, f32::MAX]
                    let is_int8 = node.meta.val.dtype == DType::Int8;
                    let min_val: i64 = 0;
                    let max_val: i64 = if is_int8 { 127 } else { i32::MAX as i64 };
                    if let Some(node) = graph.node_mut(id) {
                        node.target = OpTarget::TosaClamp;
                        // Keep first arg (input), add min/max
                        while node.args.len() > 1 {
                            node.args.pop();
                        }
                        node.kwargs = vec![
                            ("min_val".into(), Arg::Int(min_val)),
                            ("max_val".into(), Arg::Int(max_val)),
                        ];
                        modified = true;
                    }
                }
                OpTarget::Hardtanh => {
                    // Hardtanh args: [input, min_val, max_val]
                    let min_val = match node.args.get(1) {
                        Some(Arg::Float(f)) => *f,
                        Some(Arg::Int(i)) => *i as f64,
                        _ => -1.0,
                    };
                    let max_val = match node.args.get(2) {
                        Some(Arg::Float(f)) => *f,
                        Some(Arg::Int(i)) => *i as f64,
                        _ => 1.0,
                    };
                    if let Some(node) = graph.node_mut(id) {
                        node.target = OpTarget::TosaClamp;
                        while node.args.len() > 1 {
                            node.args.pop();
                        }
                        node.kwargs = vec![
                            ("min_val".into(), Arg::Int(min_val as i64)),
                            ("max_val".into(), Arg::Int(max_val as i64)),
                        ];
                        modified = true;
                    }
                }
                OpTarget::Clamp => {
                    // Already an ATen Clamp — convert to TosaClamp
                    if let Some(node) = graph.node_mut(id) {
                        node.target = OpTarget::TosaClamp;
                        modified = true;
                    }
                }
                _ => {}
            }
        }

        modified
    }
}
