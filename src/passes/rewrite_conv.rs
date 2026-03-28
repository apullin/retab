use crate::graph::ir::*;
use crate::graph::Graph;
use crate::passes::Pass;
use crate::shape;
use crate::store::{TensorData, TensorStore};
use smallvec::smallvec;

/// Rewrite ATen Conv2d / Linear / Addmm nodes to TOSA Conv2d / DepthwiseConv2d.
///
/// For Conv2d:
///   - If groups == in_channels (depthwise): -> TosaDepthwiseConv2d
///   - Else: -> TosaConv2d
///
/// For Linear / Addmm:
///   - Rewrite as TosaConv2d (1x1 conv interpretation)
pub struct RewriteConvPass;

impl Pass for RewriteConvPass {
    fn name(&self) -> &str {
        "RewriteConv"
    }

    fn run(&self, graph: &mut Graph, store: &mut TensorStore) -> bool {
        let mut modified = false;
        let all_ids = graph.all_node_ids();

        for node_id in all_ids {
            let target = match graph.node(node_id) {
                Some(n) => n.target.clone(),
                None => continue,
            };

            match target {
                OpTarget::Conv2d => {
                    if rewrite_conv2d(graph, store, node_id) {
                        modified = true;
                    }
                }
                OpTarget::Linear | OpTarget::Addmm => {
                    if rewrite_linear(graph, store, node_id) {
                        modified = true;
                    }
                }
                _ => {}
            }
        }

        modified
    }
}

/// Rewrite a Conv2d node to TosaConv2d or TosaDepthwiseConv2d.
fn rewrite_conv2d(graph: &mut Graph, store: &mut TensorStore, node_id: NodeId) -> bool {
    // Read node info
    let (input_id, weight_id, bias_arg, stride, padding, dilation, groups) = {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => return false,
        };

        let input_id = match node.args.first() {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };
        let weight_id = match node.args.get(1) {
            Some(Arg::Node(id)) => *id,
            _ => return false,
        };
        let bias_arg = node.args.get(2).cloned().unwrap_or(Arg::None);

        let stride = match node.args.get(3) {
            Some(Arg::IntList(v)) => v.to_vec(),
            _ => vec![1, 1],
        };
        let padding = match node.args.get(4) {
            Some(Arg::IntList(v)) => v.to_vec(),
            _ => vec![0, 0],
        };
        let dilation = match node.args.get(5) {
            Some(Arg::IntList(v)) => v.to_vec(),
            _ => vec![1, 1],
        };
        let groups = match node.args.get(6) {
            Some(Arg::Int(g)) => *g,
            _ => 1,
        };

        (
            input_id, weight_id, bias_arg, stride, padding, dilation, groups,
        )
    };

    // Determine if depthwise
    let in_channels = graph
        .node(input_id)
        .map(|n| {
            let s = n.meta.val.shape_slice();
            if s.len() >= 2 {
                s[1]
            } else {
                0
            }
        })
        .unwrap_or(0);

    // Match the Python reference: depthwise when in_channels == groups and
    // out_channels is a multiple of in_channels.
    let out_channels = graph
        .node(weight_id)
        .map(|n| n.meta.val.shape_slice()[0])
        .unwrap_or(0);
    let is_depthwise =
        in_channels > 0 && groups == in_channels && out_channels % in_channels == 0;

    // Create input_zp and weight_zp const nodes
    let input_qparams = graph
        .node(node_id)
        .map(|n| n.meta.input_qparams.clone())
        .unwrap_or_default();

    let input_zp_val = input_qparams
        .iter()
        .find(|(p, _)| *p == 0)
        .map(|(_, qa)| qa.zero_point)
        .unwrap_or(0);

    let weight_zp_val = input_qparams
        .iter()
        .find(|(p, _)| *p == 1)
        .map(|(_, qa)| qa.zero_point)
        .unwrap_or(0);

    let node_name = graph
        .node(node_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();

    // Create input_zp const
    let izp_name = format!("{}_input_zp", node_name);
    store.insert(
        izp_name.clone(),
        TensorData {
            data: (input_zp_val as i32).to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: DType::Int32,
        },
    );
    let izp_id = graph.insert_before(
        node_id,
        OpType::Placeholder,
        OpTarget::TosaConst,
        smallvec![],
        vec![],
        NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
        izp_name,
    );

    // Create weight_zp const
    let wzp_name = format!("{}_weight_zp", node_name);
    store.insert(
        wzp_name.clone(),
        TensorData {
            data: (weight_zp_val as i32).to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: DType::Int32,
        },
    );
    let wzp_id = graph.insert_before(
        node_id,
        OpType::Placeholder,
        OpTarget::TosaConst,
        smallvec![],
        vec![],
        NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
        wzp_name,
    );

    // Build TOSA conv2d padding attribute: [pad_top, pad_bottom, pad_left, pad_right]
    let pad_h = padding.first().copied().unwrap_or(0);
    let pad_w = padding.get(1).copied().unwrap_or(pad_h);
    let tosa_padding = smallvec![pad_h, pad_h, pad_w, pad_w];

    let stride_sv: smallvec::SmallVec<[i64; 6]> = stride.iter().copied().collect();
    let dilation_sv: smallvec::SmallVec<[i64; 6]> = dilation.iter().copied().collect();

    let new_target = if is_depthwise {
        OpTarget::TosaDepthwiseConv2d
    } else {
        OpTarget::TosaConv2d
    };

    // Ensure we have a bias node. If bias_arg is Arg::None, create a zero bias.
    let actual_bias_arg = if let Arg::Node(_) = &bias_arg {
        bias_arg
    } else {
        // Create zero bias const. For quantized conv, bias storage is int32
        // with shape [out_channels], but int16-activation convs require the
        // TOSA-visible bias contract to be INT48.
        let out_channels = graph
            .node(weight_id)
            .map(|n| n.meta.val.shape_slice()[0])
            .unwrap_or(1);
        let input_dtype = graph
            .node(input_id)
            .map(|n| n.meta.val.dtype)
            .unwrap_or(DType::Int32);
        let input_contract_dtype = input_qparams
            .iter()
            .find(|(p, _)| *p == 0)
            .map(|(_, qa)| qa.dtype)
            .unwrap_or(input_dtype);
        let bias_name = format!("{}_zero_bias", node_name);
        let bias_data = vec![0u8; (out_channels as usize) * 4]; // int32 zeros
        store.insert(
            bias_name.clone(),
            TensorData {
                data: bias_data,
                shape: vec![out_channels],
                dtype: DType::Int32,
            },
        );
        let mut bias_meta = NodeMeta::new(TensorInfo::new(&[out_channels], DType::Int32));
        if input_contract_dtype == DType::Int16 {
            bias_meta.tosa_special_dtype = Some(DType::Int48);
        }
        let bias_id = graph.insert_before(
            node_id,
            OpType::Placeholder,
            OpTarget::TosaConst,
            smallvec![],
            vec![],
            bias_meta,
            bias_name,
        );
        Arg::Node(bias_id)
    };

    // Build args: [input, weight, bias, input_zp, weight_zp]
    let new_args: smallvec::SmallVec<[Arg; 4]> = smallvec![
        Arg::Node(input_id),
        Arg::Node(weight_id),
        actual_bias_arg,
        Arg::Node(izp_id),
        Arg::Node(wzp_id),
    ];

    // Store conv attributes as kwargs
    let new_kwargs = vec![
        ("pad".into(), Arg::IntList(tosa_padding)),
        ("stride".into(), Arg::IntList(stride_sv)),
        ("dilation".into(), Arg::IntList(dilation_sv)),
    ];

    // Mutate the node in place
    if let Some(node) = graph.node_mut(node_id) {
        node.target = new_target;
        node.args = new_args;
        node.kwargs = new_kwargs;
    }

    shape::propagate_shape(graph, node_id);

    true
}

/// Rewrite a Linear or Addmm node to TosaConv2d (1x1 convolution).
fn rewrite_linear(graph: &mut Graph, store: &mut TensorStore, node_id: NodeId) -> bool {
    let node = match graph.node(node_id) {
        Some(n) => n,
        None => return false,
    };

    // For Addmm: args are [bias, input, weight]
    // For Linear: args are [input, weight, bias?]
    let (input_id, weight_id, bias_arg) = match &node.target {
        OpTarget::Addmm => {
            let bias_id = match node.args.first() {
                Some(Arg::Node(id)) => Some(*id),
                _ => None,
            };
            let input_id = match node.args.get(1) {
                Some(Arg::Node(id)) => *id,
                _ => return false,
            };
            let weight_id = match node.args.get(2) {
                Some(Arg::Node(id)) => *id,
                _ => return false,
            };
            let ba = match bias_id {
                Some(id) => Arg::Node(id),
                None => Arg::None,
            };
            (input_id, weight_id, ba)
        }
        OpTarget::Linear => {
            let input_id = match node.args.first() {
                Some(Arg::Node(id)) => *id,
                _ => return false,
            };
            let weight_id = match node.args.get(1) {
                Some(Arg::Node(id)) => *id,
                _ => return false,
            };
            let ba = node.args.get(2).cloned().unwrap_or(Arg::None);
            (input_id, weight_id, ba)
        }
        _ => return false,
    };

    let node_name = node.name.clone();
    let input_qparams = node.meta.input_qparams.clone();

    let input_zp_val = input_qparams
        .iter()
        .find(|(p, _)| *p == 0)
        .map(|(_, qa)| qa.zero_point)
        .unwrap_or(0);

    let weight_zp_val = input_qparams
        .iter()
        .find(|(p, _)| *p == 1)
        .map(|(_, qa)| qa.zero_point)
        .unwrap_or(0);

    // Create zp consts
    let izp_name = format!("{}_input_zp", node_name);
    store.insert(
        izp_name.clone(),
        TensorData {
            data: (input_zp_val as i32).to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: DType::Int32,
        },
    );
    let izp_id = graph.insert_before(
        node_id,
        OpType::Placeholder,
        OpTarget::TosaConst,
        smallvec![],
        vec![],
        NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
        izp_name,
    );

    let wzp_name = format!("{}_weight_zp", node_name);
    store.insert(
        wzp_name.clone(),
        TensorData {
            data: (weight_zp_val as i32).to_le_bytes().to_vec(),
            shape: vec![1],
            dtype: DType::Int32,
        },
    );
    let wzp_id = graph.insert_before(
        node_id,
        OpType::Placeholder,
        OpTarget::TosaConst,
        smallvec![],
        vec![],
        NodeMeta::new(TensorInfo::new(&[1], DType::Int32)),
        wzp_name,
    );

    // Ensure we have a bias node.
    let actual_bias_arg = if let Arg::Node(_) = &bias_arg {
        bias_arg
    } else {
        let out_features = graph
            .node(weight_id)
            .map(|n| n.meta.val.shape_slice()[0])
            .unwrap_or(1);
        let bias_name = format!("{}_zero_bias", node_name);
        let bias_data = vec![0u8; (out_features as usize) * 4];
        store.insert(
            bias_name.clone(),
            TensorData {
                data: bias_data,
                shape: vec![out_features],
                dtype: DType::Int32,
            },
        );
        let bias_id = graph.insert_before(
            node_id,
            OpType::Placeholder,
            OpTarget::TosaConst,
            smallvec![],
            vec![],
            NodeMeta::new(TensorInfo::new(&[out_features], DType::Int32)),
            bias_name,
        );
        Arg::Node(bias_id)
    };

    // Args: [input, weight, bias, input_zp, weight_zp]
    let new_args: smallvec::SmallVec<[Arg; 4]> = smallvec![
        Arg::Node(input_id),
        Arg::Node(weight_id),
        actual_bias_arg,
        Arg::Node(izp_id),
        Arg::Node(wzp_id),
    ];

    // Linear as 1x1 conv: stride=1, padding=0, dilation=1
    let new_kwargs = vec![
        ("pad".into(), Arg::IntList(smallvec![0, 0, 0, 0])),
        ("stride".into(), Arg::IntList(smallvec![1, 1])),
        ("dilation".into(), Arg::IntList(smallvec![1, 1])),
    ];

    if let Some(node) = graph.node_mut(node_id) {
        node.target = OpTarget::TosaConv2d;
        node.args = new_args;
        node.kwargs = new_kwargs;
    }

    shape::propagate_shape(graph, node_id);

    true
}
