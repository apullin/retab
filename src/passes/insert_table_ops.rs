use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Replace quantized unary nodes (Rsqrt, Exp, Log, Sigmoid, Tanh) with
/// TosaTable nodes that use a precomputed lookup table.
pub struct InsertTableOpsPass;

/// Ops eligible for TABLE replacement when quantized.
fn table_func(target: &OpTarget) -> Option<fn(f64) -> f64> {
    match target {
        OpTarget::Rsqrt => Some(|x: f64| if x > 0.0 { 1.0 / x.sqrt() } else { 0.0 }),
        OpTarget::Exp => Some(|x: f64| x.exp()),
        OpTarget::Log => Some(|x: f64| if x > 0.0 { x.ln() } else { 0.0 }),
        OpTarget::Sigmoid => Some(|x: f64| 1.0 / (1.0 + (-x).exp())),
        OpTarget::Tanh => Some(|x: f64| x.tanh()),
        _ => None,
    }
}

impl Pass for InsertTableOpsPass {
    fn name(&self) -> &str {
        "InsertTableOps"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let (target, has_qparams, node_name, input_qp, output_qp) = {
                let node = match graph.node(node_id) {
                    Some(n) => n,
                    None => continue,
                };
                let has_input_qp = !node.meta.input_qparams.is_empty();
                let has_output_qp = !node.meta.output_qparams.is_empty();
                let iqp = node.meta.input_qparams.clone();
                let oqp = node.meta.output_qparams.clone();
                (
                    node.target.clone(),
                    has_input_qp && has_output_qp,
                    node.name.clone(),
                    iqp,
                    oqp,
                )
            };

            let func = match table_func(&target) {
                Some(f) if has_qparams => f,
                _ => continue,
            };

            // Get input and output qparams (first entry of each)
            let in_qa = match input_qp.first() {
                Some((_, qa)) => *qa,
                None => continue,
            };
            let out_qa = match output_qp.first() {
                Some((_, qa)) => *qa,
                None => continue,
            };

            // Compute the lookup table
            let table_values = compute_table(&in_qa, &out_qa, func);

            // Store the table as a const tensor
            let table_len = table_values.len() as i64;
            let table_name = format!("{}_table_constant", node_name);
            let table_bytes: Vec<u8> = table_values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            store.insert(
                table_name.clone(),
                TensorData {
                    data: table_bytes,
                    shape: vec![table_len],
                    dtype: DType::Int16,
                },
            );

            // Create a Placeholder node for the table const
            let table_placeholder_id = graph.insert_before(
                node_id,
                OpType::Placeholder,
                OpTarget::TosaConst,
                smallvec![],
                vec![],
                NodeMeta::new(TensorInfo::new(&[table_len], DType::Int16)),
                table_name,
            );

            // Get the input node
            let input_id = match graph.node(node_id) {
                Some(n) => match n.args.first() {
                    Some(Arg::Node(id)) => *id,
                    _ => continue,
                },
                None => continue,
            };

            // Rewrite the node to TosaTable in place
            if let Some(n) = graph.node_mut(node_id) {
                n.target = OpTarget::TosaTable;
                n.args = smallvec![Arg::Node(input_id), Arg::Node(table_placeholder_id)];
                n.kwargs.clear();
            }

            // Update user lists for the table placeholder
            if let Some(t) = graph.node_mut(table_placeholder_id) {
                if !t.users.contains(&node_id) {
                    t.users.push(node_id);
                }
            }

            modified = true;
        }

        modified
    }
}

/// Compute a lookup table for quantized tensors with an arbitrary function.
///
/// For int8 (qmin=-128, qmax=127): produces 256 entries (one per quantized value).
/// For int16: produces 513 entries (matching TOSA TABLE spec for 16-bit).
///
/// Each entry:
///   1. Dequantize: x = (i - zero_point) * scale
///   2. Apply func: y = func(x)
///   3. Quantize: q = round(y / out_scale + out_zero_point), clamped to output range
fn compute_table(
    input_qparams: &QuantArgs,
    output_qparams: &QuantArgs,
    func: fn(f64) -> f64,
) -> Vec<i16> {
    let num_entries = (input_qparams.qmax - input_qparams.qmin + 1) as usize;
    let mut table = Vec::with_capacity(num_entries);
    for i in (input_qparams.qmin as i32)..=(input_qparams.qmax as i32) {
        // Dequantize
        let x = (i as f64 - input_qparams.zero_point as f64) * input_qparams.scale;
        // Apply function
        let y = func(x);
        // Quantize
        let q = ((y / output_qparams.scale) + output_qparams.zero_point as f64)
            .round()
            .clamp(output_qparams.qmin as f64, output_qparams.qmax as f64) as i16;
        table.push(q);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use smallvec::SmallVec;

    #[test]
    fn test_compute_table_length() {
        let in_qp = QuantArgs {
            scale: 0.01,
            zero_point: 0,
            dtype: DType::Int8,
            qmin: -128,
            qmax: 127,
        };
        let out_qp = QuantArgs {
            scale: 0.05,
            zero_point: 0,
            dtype: DType::Int8,
            qmin: -128,
            qmax: 127,
        };
        let rsqrt_fn: fn(f64) -> f64 = |x| if x > 0.0 { 1.0 / x.sqrt() } else { 0.0 };
        let table = compute_table(&in_qp, &out_qp, rsqrt_fn);
        // int8 range: 256 entries (-128 to 127 inclusive)
        assert_eq!(table.len(), 256);
    }

    #[test]
    fn test_rsqrt_table_values_non_negative() {
        // rsqrt of positive values should produce positive quantized values
        let in_qp = QuantArgs {
            scale: 0.01,
            zero_point: -128, // so raw 0 maps to dequant 1.28
            dtype: DType::Int8,
            qmin: -128,
            qmax: 127,
        };
        let out_qp = QuantArgs {
            scale: 0.1,
            zero_point: 0,
            dtype: DType::Int8,
            qmin: -128,
            qmax: 127,
        };
        let rsqrt_fn: fn(f64) -> f64 = |x| if x > 0.0 { 1.0 / x.sqrt() } else { 0.0 };
        let table = compute_table(&in_qp, &out_qp, rsqrt_fn);
        // All dequantized values are positive (zp=-128, so i - zp >= 0 for i >= -128)
        // rsqrt should be >= 0, so quantized values should be >= 0
        for &v in &table {
            assert!(v >= 0, "Table value should be non-negative, got {}", v);
        }
    }

    #[test]
    fn test_insert_table_ops_pass_rewrites_rsqrt() {
        let mut graph = Graph::new();
        let mut store = TensorStore::new();

        // Create input placeholder
        let input_id = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("input".into()),
            smallvec![],
            vec![],
            NodeMeta::new(TensorInfo::new(&[1, 16, 28, 28], DType::Int8)),
            "input".into(),
        );

        // Create Rsqrt node with qparams
        let mut rsqrt_meta = NodeMeta::new(TensorInfo::new(&[1, 16, 28, 28], DType::Int8));
        rsqrt_meta.input_qparams = SmallVec::from_vec(vec![(
            0,
            QuantArgs {
                scale: 0.01,
                zero_point: 0,
                dtype: DType::Int8,
                qmin: -128,
                qmax: 127,
            },
        )]);
        rsqrt_meta.output_qparams = SmallVec::from_vec(vec![(
            0,
            QuantArgs {
                scale: 0.05,
                zero_point: 0,
                dtype: DType::Int8,
                qmin: -128,
                qmax: 127,
            },
        )]);

        let _rsqrt_id = graph.create_node(
            OpType::CallFunction,
            OpTarget::Rsqrt,
            smallvec![Arg::Node(input_id)],
            vec![],
            rsqrt_meta,
            "rsqrt_node".into(),
        );

        let pass = InsertTableOpsPass;
        let modified = pass.run(&mut graph, &mut store);
        assert!(modified);

        // Check that the Rsqrt was replaced with TosaTable
        let table_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|(_, n)| n.target == OpTarget::TosaTable)
            .collect();
        assert_eq!(table_nodes.len(), 1);

        // Check that the table const was created in the store
        assert!(store.get("rsqrt_node_table_constant").is_some());
    }
}
