//! Pure Rust IrPass: adaptive_avg_pool2d → avg_pool2d or slice + pool + cat.

use crate::fx_ir::{i64_list_arg as i64_list, IrArg, IrGraph, IrPass, NodeId};
use crate::fx_target::is_adaptive_avg_pool2d_target;
use crate::runner_logging::log_line;

pub struct IrDecomposeAdaptiveAvgPoolPass;

impl IrPass for IrDecomposeAdaptiveAvgPoolPass {
    fn name(&self) -> &str {
        "IrDecomposeAdaptiveAvgPool"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            if !is_adaptive_avg_pool2d_target(&graph.nodes[id].target) {
                continue;
            }

            let x_id = match graph.nodes[id].first_node_arg() {
                Some(n) => n,
                None => continue,
            };
            let input_shape = match graph.node_shape(x_id) {
                Some(s) if s.len() == 4 => s.to_vec(),
                _ => continue,
            };
            let (n_batch, c_ch, input_h, input_w) = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            );

            let (output_h, output_w) = match &graph.nodes[id].args.get(1) {
                Some(IrArg::List(items)) if items.len() == 2 => {
                    let (Some(h), Some(w)) = (items[0].as_int(), items[1].as_int()) else {
                        continue;
                    };
                    (h, w)
                }
                _ => continue,
            };
            if input_h <= 0 || input_w <= 0 || output_h <= 0 || output_w <= 0 {
                continue;
            }

            let node_name = graph.nodes[id].name.clone();
            let orig_meta = graph.nodes[id].meta.clone();
            let x_meta = graph.nodes[x_id].meta.clone();
            let meta = orig_meta.tensor_stamp_with_dtype_of_or_float32(&x_meta);

            let mut cursor = graph.rewrite_cursor(id);
            let (stride_h, stride_w) = (input_h / output_h, input_w / output_w);
            let (rem_h, rem_w) = (input_h % output_h, input_w % output_w);
            if rem_h <= 1
                && rem_w <= 1
                && (1..=3).contains(&stride_h)
                && (1..=3).contains(&stride_w)
            {
                let pooled = cursor.call_function(
                    format!("{}_pool", node_name),
                    "aten.avg_pool2d.default",
                    vec![
                        IrArg::Node(x_id),
                        i64_list(&[stride_h + rem_h, stride_w + rem_w]),
                        i64_list(&[stride_h, stride_w]),
                        i64_list(&[0, 0]),
                    ],
                    meta.preserve_qparams(vec![n_batch, c_ch, output_h, output_w]),
                );
                cursor.replace_anchor_with(pooled);
                modified = true;
                continue;
            }

            let mut rows: Vec<NodeId> = Vec::new();
            for out_i in 0..output_h {
                let mut row: Vec<NodeId> = Vec::new();
                for out_j in 0..output_w {
                    let start_h = out_i * input_h / output_h;
                    let end_h = (out_i + 1) * input_h / output_h
                        + if ((out_i + 1) * input_h) % output_h != 0 {
                            1
                        } else {
                            0
                        };
                    let start_w = out_j * input_w / output_w;
                    let end_w = (out_j + 1) * input_w / output_w
                        + if ((out_j + 1) * input_w) % output_w != 0 {
                            1
                        } else {
                            0
                        };
                    let kh = end_h - start_h;
                    let kw = end_w - start_w;

                    // slice_copy(x, dim=2, start_h, end_h)
                    let slice_h = cursor.slice_copy(
                        format!("{}_sh_{}_{}", node_name, out_i, out_j),
                        x_id,
                        2,
                        start_h,
                        end_h,
                        None,
                        meta.drop_qparams(vec![n_batch, c_ch, kh, input_w]),
                    );

                    // slice_copy(slice_h, dim=3, start_w, end_w)
                    let slice_hw = cursor.slice_copy(
                        format!("{}_sw_{}_{}", node_name, out_i, out_j),
                        slice_h,
                        3,
                        start_w,
                        end_w,
                        None,
                        meta.drop_qparams(vec![n_batch, c_ch, kh, kw]),
                    );

                    // avg_pool2d(slice_hw, [kh,kw], [1,1], [0,0])
                    let pooled = cursor.call_function(
                        format!("{}_pool_{}_{}", node_name, out_i, out_j),
                        "aten.avg_pool2d.default",
                        vec![
                            IrArg::Node(slice_hw),
                            i64_list(&[kh, kw]), // kernel
                            i64_list(&[1, 1]),   // stride
                            i64_list(&[0, 0]),   // padding
                        ],
                        meta.preserve_qparams(vec![n_batch, c_ch, 1, 1]),
                    );

                    row.push(pooled);
                }

                // Cat columns along W (dim=3) if multiple
                if row.len() == 1 {
                    rows.push(row[0]);
                } else {
                    let cat_args = IrArg::List(row.iter().map(|&n| IrArg::Node(n)).collect());
                    let row_cat = cursor.call_function(
                        format!("{}_cat_w_{}", node_name, out_i),
                        "aten.cat.default",
                        vec![cat_args, IrArg::Int(3)],
                        meta.drop_qparams(vec![n_batch, c_ch, 1, output_w]),
                    );
                    rows.push(row_cat);
                }
            }

            // Cat rows along H (dim=2) if multiple
            let result = if rows.len() == 1 {
                rows[0]
            } else {
                let cat_args = IrArg::List(rows.iter().map(|&n| IrArg::Node(n)).collect());
                cursor.call_function(
                    format!("{}_cat_h", node_name),
                    "aten.cat.default",
                    vec![cat_args, IrArg::Int(2)],
                    meta.drop_qparams(vec![n_batch, c_ch, output_h, output_w]),
                )
            };

            cursor.replace_anchor_with(result);
            modified = true;
        }

        if modified {
            log_line(format_args!(
                "[IrDecomposeAdaptiveAvgPool] decomposed adaptive pool ops"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{NodeMeta, QParam};
    use crate::test_ir::{call_with_meta, live_node, output, placeholder};

    fn pool_graph(input_shape: [i64; 4], output_size: [i64; 2]) -> (IrGraph, NodeId) {
        let mut graph = IrGraph::empty();
        let input = placeholder(&mut graph, "x", input_shape, "torch.int8");
        let mut meta = NodeMeta::tensor(
            [
                input_shape[0],
                input_shape[1],
                output_size[0],
                output_size[1],
            ],
            "torch.int8",
        );
        meta.set_unary_input_output_qparams(
            QParam::per_tensor(0.5, 3, -128, 127, "torch.int8".into()),
            QParam::per_tensor(0.25, -2, -128, 127, "torch.int8".into()),
        );
        let pool = call_with_meta(
            &mut graph,
            "pool",
            "aten._adaptive_avg_pool2d.default",
            vec![IrArg::Node(input), i64_list(&output_size)],
            meta,
        );
        output(&mut graph, pool);
        (graph, pool)
    }

    #[test]
    fn directly_representable_pools_use_rc4_kernel_and_stride() {
        for (input, output_size, kernel, stride) in [
            ([1, 3, 8, 8], [4, 4], [2, 2], [2, 2]),
            ([1, 3, 10, 16], [3, 5], [4, 4], [3, 3]),
            ([1, 3, 7, 7], [3, 3], [3, 3], [2, 2]),
            ([1, 3, 9, 15], [3, 5], [3, 3], [3, 3]),
        ] {
            let (mut graph, pool) = pool_graph(input, output_size);
            assert!(IrDecomposeAdaptiveAvgPoolPass.run(&mut graph));
            let pooled = live_node(&graph, "pool_pool");
            assert_eq!(graph.replacement_of(pool), Some(pooled.id));
            assert_eq!(pooled.first_node_arg(), Some(0));
            assert_eq!(pooled.arg_int_list(1), Some(kernel.to_vec()));
            assert_eq!(pooled.arg_int_list(2), Some(stride.to_vec()));
            assert_eq!(pooled.arg_int_list(3), Some(vec![0, 0]));
            assert_eq!(pooled.meta.input_qparam().unwrap().zp, 3);
            assert_eq!(pooled.meta.output_qparam().unwrap().scale, 0.25);
            assert_eq!(graph.live_call_function_ids().len(), 1);
        }
    }

    #[test]
    fn fallback_bins_cover_nonuniform_and_upsampling_cases() {
        for (input, output_size, h_bins, w_bins) in [
            (
                [1, 3, 7, 7],
                [4, 4],
                vec![(0, 2), (1, 4), (3, 6), (5, 7)],
                vec![(0, 2), (1, 4), (3, 6), (5, 7)],
            ),
            (
                [1, 3, 9, 13],
                [2, 3],
                vec![(0, 5), (4, 9)],
                vec![(0, 5), (4, 9), (8, 13)],
            ),
            (
                [1, 3, 17, 17],
                [5, 5],
                vec![(0, 4), (3, 7), (6, 11), (10, 14), (13, 17)],
                vec![(0, 4), (3, 7), (6, 11), (10, 14), (13, 17)],
            ),
            (
                [1, 3, 6, 6],
                [7, 7],
                vec![(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)],
                vec![(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)],
            ),
        ] {
            let (mut graph, pool) = pool_graph(input, output_size);
            assert!(IrDecomposeAdaptiveAvgPoolPass.run(&mut graph));
            assert!(!graph.node(pool).alive);
            for (i, &(start_h, end_h)) in h_bins.iter().enumerate() {
                for (j, &(start_w, end_w)) in w_bins.iter().enumerate() {
                    let sh = live_node(&graph, &format!("pool_sh_{i}_{j}"));
                    let sw = live_node(&graph, &format!("pool_sw_{i}_{j}"));
                    assert_eq!((sh.arg_int(2), sh.arg_int(3)), (Some(start_h), Some(end_h)));
                    assert_eq!((sw.arg_int(2), sw.arg_int(3)), (Some(start_w), Some(end_w)));
                    let pooled = live_node(&graph, &format!("pool_pool_{i}_{j}"));
                    assert_eq!(
                        pooled.arg_int_list(1),
                        Some(vec![end_h - start_h, end_w - start_w])
                    );
                    assert_eq!(pooled.arg_int_list(2), Some(vec![1, 1]));
                    assert_eq!(pooled.meta.val_shape(), Some([1, 3, 1, 1].as_slice()));
                    assert_eq!(pooled.meta.input_qparam().unwrap().zp, 3);
                    assert_eq!(pooled.meta.output_qparam().unwrap().scale, 0.25);
                }
            }
            for id in graph.live_call_function_ids() {
                let node = graph.node(id);
                if node.target == "aten.slice_copy.Tensor" || node.target == "aten.cat.default" {
                    assert!(node.meta.input_qparams().is_none());
                    assert!(node.meta.output_qparams().is_none());
                }
            }
        }
    }

    #[test]
    fn leaves_nonstatic_or_invalid_sizes_untouched() {
        let (mut graph, pool) = pool_graph([1, 3, 8, 8], [0, 4]);
        assert!(!IrDecomposeAdaptiveAvgPoolPass.run(&mut graph));
        assert!(graph.node(pool).alive);
        graph.set_arg(pool, 1, IrArg::List(vec![IrArg::None, IrArg::Int(4)]));
        assert!(!IrDecomposeAdaptiveAvgPoolPass.run(&mut graph));
        assert!(graph.node(pool).alive);
    }
}
