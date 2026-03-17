use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::TensorStore;

/// If a Relu/Clamp/Hardtanh immediately follows a quantized op (one with
/// output_qparams), fuse the clamp range into the quantization range by
/// adjusting qmin/qmax in the output_qparams, then bypass the activation node.
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

            // Check if this is an activation node
            let (clamp_min, clamp_max) = match &node.target {
                OpTarget::Relu => (Some(0.0_f64), None),
                OpTarget::Clamp | OpTarget::Hardtanh | OpTarget::TosaClamp => {
                    let cmin = node
                        .args
                        .get(1)
                        .and_then(|a| match a {
                            Arg::Float(f) => Some(*f),
                            Arg::Int(i) => Some(*i as f64),
                            _ => None,
                        })
                        .or_else(|| {
                            node.kwargs.iter().find_map(|(k, v)| {
                                if k == "min" || k == "min_val" {
                                    match v {
                                        Arg::Float(f) => Some(*f),
                                        Arg::Int(i) => Some(*i as f64),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            })
                        });
                    let cmax = node
                        .args
                        .get(2)
                        .and_then(|a| match a {
                            Arg::Float(f) => Some(*f),
                            Arg::Int(i) => Some(*i as f64),
                            _ => None,
                        })
                        .or_else(|| {
                            node.kwargs.iter().find_map(|(k, v)| {
                                if k == "max" || k == "max_val" {
                                    match v {
                                        Arg::Float(f) => Some(*f),
                                        Arg::Int(i) => Some(*i as f64),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            })
                        });
                    (cmin, cmax)
                }
                _ => continue,
            };

            // Get the input node
            let input_id = match node.args.first() {
                Some(Arg::Node(id)) => *id,
                _ => continue,
            };

            // Check if the input has output_qparams
            let out_qparams = match graph.node(input_id) {
                Some(n) if !n.meta.output_qparams.is_empty() => {
                    n.meta.output_qparams.clone()
                }
                _ => continue,
            };

            // Find the output qparam at position 0
            let (qp_idx, qp) = match out_qparams
                .iter()
                .enumerate()
                .find(|(_, (p, _))| *p == 0)
            {
                Some((i, (_, qp))) => (i, *qp),
                None => continue,
            };

            // Convert clamp bounds to quantized domain
            let new_qmin = if let Some(cmin) = clamp_min {
                let quantized_min = ((cmin / qp.scale) + qp.zero_point as f64).round() as i64;
                qp.qmin.max(quantized_min)
            } else {
                qp.qmin
            };

            let new_qmax = if let Some(cmax) = clamp_max {
                let quantized_max = ((cmax / qp.scale) + qp.zero_point as f64).round() as i64;
                qp.qmax.min(quantized_max)
            } else {
                qp.qmax
            };

            // Update the producer's output qparams
            if let Some(producer) = graph.node_mut(input_id) {
                if let Some(entry) = producer.meta.output_qparams.get_mut(qp_idx) {
                    entry.1.qmin = new_qmin;
                    entry.1.qmax = new_qmax;
                }
            }

            // Bypass the activation node
            graph.bypass_node(node_id);
            modified = true;
        }

        modified
    }
}
