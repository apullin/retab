use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass};
use crate::runner_logging::log_line;

const U55_MAX_POOL_DIM: i64 = 65536;
const U55_MAX_POOL_KERNEL_WIDTH: i64 = 256;

fn can_decompose_large_stride_maxpool2d(
    shape: &[i64],
    kernel: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
) -> bool {
    if shape.len() != 4
        || kernel != stride
        || stride.0.max(stride.1) <= 3
        || padding != (0, 0)
        || dilation != (1, 1)
        || ceil_mode
        || !shape[1..]
            .iter()
            .all(|dim| (1..=U55_MAX_POOL_DIM).contains(dim))
        || !(1..=U55_MAX_POOL_KERNEL_WIDTH).contains(&kernel.0)
        || !(1..=U55_MAX_POOL_KERNEL_WIDTH).contains(&kernel.1)
        || shape[2] < kernel.0
        || shape[3] < kernel.1
    {
        return false;
    }
    let windows = (shape[2] / kernel.0) * (shape[3] / kernel.1);
    windows * kernel.0 <= U55_MAX_POOL_DIM && windows <= U55_MAX_POOL_DIM
}

pub struct IrDecomposeLargeStrideMaxPool2dForU55Pass;

impl IrPass for IrDecomposeLargeStrideMaxPool2dForU55Pass {
    fn name(&self) -> &str {
        "IrDecomposeLargeStrideMaxPool2dForU55"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let modified = graph.rewrite_live_call_functions(|graph, id| {
            let node = &graph.nodes[id];
            let with_indices = node.target == "aten.max_pool2d_with_indices.default";
            if node.target != "aten.max_pool2d.default" && !with_indices {
                return false;
            }
            let mut value_users = Vec::new();
            let mut unused_indices = Vec::new();
            if with_indices {
                for user in graph.live_user_ids(id) {
                    if !graph.nodes[user].is_getitem() {
                        return false;
                    }
                    match graph.nodes[user].arg_int(1) {
                        Some(0) => value_users.push(user),
                        Some(1) if graph.users(user).is_empty() => unused_indices.push(user),
                        _ => return false,
                    }
                }
                if value_users.is_empty() {
                    return false;
                }
            }
            let Some(input) = node.first_node_arg() else {
                return false;
            };
            let Some(shape) = graph.node_shape_vec(input) else {
                return false;
            };
            let Some(kernel) = node.arg_int_pair_or_scalar(1) else {
                return false;
            };
            let stride = node.arg_int_pair_or_scalar(2).unwrap_or(kernel);
            let padding = node.arg_int_pair_or_scalar(3).unwrap_or((0, 0));
            let dilation = node.arg_int_pair_or_scalar(4).unwrap_or((1, 1));
            let ceil_mode = node.arg_bool_or(5, false);
            if !can_decompose_large_stride_maxpool2d(
                &shape, kernel, stride, padding, dilation, ceil_mode,
            ) {
                return false;
            }

            let (n, c, height, width) = (shape[0], shape[1], shape[2], shape[3]);
            let (kh, kw) = kernel;
            let oh = height / kh;
            let ow = width / kw;
            let name = node.name.clone();
            let orig_meta = value_users
                .first()
                .map(|&user| graph.nodes[user].meta.clone())
                .unwrap_or_else(|| node.meta.clone());
            let input_meta = graph.nodes[input].meta.clone();
            let meta = orig_meta.tensor_stamp_with_dtype_of_or_float32(&input_meta);
            let result = {
                let mut cursor = graph.rewrite_cursor(id);
                let mut input = input;
                let mut cropped_shape = shape;
                if oh * kh != height {
                    cropped_shape[2] = oh * kh;
                    input = cursor.slice_copy(
                        format!("{name}_crop_h"),
                        input,
                        2,
                        0,
                        oh * kh,
                        None,
                        meta.drop_qparams(cropped_shape.clone()),
                    );
                }
                if ow * kw != width {
                    cropped_shape[3] = ow * kw;
                    input = cursor.slice_copy(
                        format!("{name}_crop_w"),
                        input,
                        3,
                        0,
                        ow * kw,
                        None,
                        meta.drop_qparams(cropped_shape),
                    );
                }
                let blocked = cursor.view_copy(
                    format!("{name}_blocks"),
                    input,
                    &[n, c, oh, kh, ow, kw],
                    meta.drop_qparams(vec![n, c, oh, kh, ow, kw]),
                );
                let permuted = cursor.permute_copy(
                    format!("{name}_permute"),
                    blocked,
                    &[0, 1, 2, 4, 3, 5],
                    meta.drop_qparams(vec![n, c, oh, ow, kh, kw]),
                );
                let rows = cursor.view_copy(
                    format!("{name}_rows"),
                    permuted,
                    &[n, c, oh * ow * kh, kw],
                    meta.drop_qparams(vec![n, c, oh * ow * kh, kw]),
                );
                let row_max = cursor.call_function(
                    format!("{name}_row_max"),
                    "aten.max_pool2d.default",
                    vec![
                        IrArg::Node(rows),
                        i64_list_arg(&[1, kw]),
                        i64_list_arg(&[1, 1]),
                        i64_list_arg(&[0, 0]),
                        i64_list_arg(&[1, 1]),
                        IrArg::Bool(false),
                    ],
                    meta.drop_qparams(vec![n, c, oh * ow * kh, 1]),
                );
                let columns = cursor.view_copy(
                    format!("{name}_columns"),
                    row_max,
                    &[n, c, oh * ow, kh],
                    meta.drop_qparams(vec![n, c, oh * ow, kh]),
                );
                let column_max = cursor.call_function(
                    format!("{name}_column_max"),
                    "aten.max_pool2d.default",
                    vec![
                        IrArg::Node(columns),
                        i64_list_arg(&[1, kh]),
                        i64_list_arg(&[1, 1]),
                        i64_list_arg(&[0, 0]),
                        i64_list_arg(&[1, 1]),
                        IrArg::Bool(false),
                    ],
                    meta.drop_qparams(vec![n, c, oh * ow, 1]),
                );
                cursor.view_copy(
                    format!("{name}_out"),
                    column_max,
                    &[n, c, oh, ow],
                    meta.preserve_qparams(vec![n, c, oh, ow]),
                )
            };
            if with_indices {
                for user in value_users {
                    graph.replace_and_erase(user, result);
                }
                for user in unused_indices {
                    graph.erase(user);
                }
                graph.erase(id);
            } else {
                graph.replace_and_erase(id, result);
            }
            true
        });
        if modified {
            log_line(format_args!(
                "[IrDecomposeLargeStrideMaxPool2dForU55] decomposed non-overlapping pools"
            ));
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{call, live_node, placeholder, qparam};

    #[test]
    fn accepts_only_the_stock_u55_kernel_and_dimension_limits() {
        for (shape, kernel, expected) in [
            (vec![1, 3, 13, 17], (5, 5), true),
            (vec![1, 2, 11, 23], (4, 7), true),
            (vec![1, 3, 1, 17], (1, 5), true),
            (vec![1, 1, 256, 256], (256, 256), true),
            (vec![1, 1, 257, 1], (257, 1), false),
            (vec![1, 1, 1, 257], (1, 257), false),
            (vec![1, 1, 65537, 25], (5, 5), false),
            (vec![1, 65537, 25, 25], (5, 5), false),
            (vec![1, 1, 1028, 1028], (4, 4), false),
            (vec![1, 1, -1, 25], (5, 5), false),
            (vec![1, 1, 4, 25], (5, 5), false),
            (vec![1, 1, 25, 25], (3, 3), false),
            (vec![1, 1, 25, 25], (0, 5), false),
        ] {
            assert_eq!(
                can_decompose_large_stride_maxpool2d(&shape, kernel, kernel, (0, 0), (1, 1), false),
                expected,
                "{shape:?}, {kernel:?}"
            );
        }
        for (stride, padding, dilation, ceil_mode) in [
            ((4, 4), (0, 0), (1, 1), false),
            ((5, 5), (1, 0), (1, 1), false),
            ((5, 5), (0, 0), (2, 1), false),
            ((5, 5), (0, 0), (1, 1), true),
        ] {
            assert!(!can_decompose_large_stride_maxpool2d(
                &[1, 3, 13, 17],
                (5, 5),
                stride,
                padding,
                dilation,
                ceil_mode
            ));
        }
    }

    #[test]
    fn crops_partial_windows_and_reduces_each_kernel_at_stride_one() {
        for (shape, kernel, output_shape) in [
            ([1, 3, 13, 17], [5, 5], [1, 3, 2, 3]),
            ([1, 2, 11, 23], [4, 7], [1, 2, 2, 3]),
            ([1, 3, 1, 17], [1, 5], [1, 3, 1, 3]),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", shape, "torch.int8");
            let pool = call(
                &mut graph,
                "pool",
                "aten.max_pool2d.default",
                vec![IrArg::Node(x), i64_list_arg(&kernel), i64_list_arg(&[])],
                output_shape,
                "torch.int8",
            );
            graph.nodes[pool]
                .meta
                .set_both_qparams(qparam(0.1, 0, "torch.int8"));
            assert!(IrDecomposeLargeStrideMaxPool2dForU55Pass.run(&mut graph));
            let oh = output_shape[2];
            let ow = output_shape[3];
            let blocks = live_node(&graph, "pool_blocks");
            assert_eq!(
                blocks.meta.val_shape(),
                Some([shape[0], shape[1], oh, kernel[0], ow, kernel[1]].as_slice())
            );
            let permuted = live_node(&graph, "pool_permute");
            assert_eq!(permuted.arg_int_list(1), Some(vec![0, 1, 2, 4, 3, 5]));
            for (name, expected_kernel, expected_shape) in [
                (
                    "pool_row_max",
                    vec![1, kernel[1]],
                    vec![shape[0], shape[1], oh * ow * kernel[0], 1],
                ),
                (
                    "pool_column_max",
                    vec![1, kernel[0]],
                    vec![shape[0], shape[1], oh * ow, 1],
                ),
            ] {
                let reduction = live_node(&graph, name);
                assert_eq!(reduction.arg_int_list(1), Some(expected_kernel));
                assert_eq!(reduction.arg_int_list(2), Some(vec![1, 1]));
                assert_eq!(reduction.arg_int_list(3), Some(vec![0, 0]));
                assert_eq!(reduction.arg_int_list(4), Some(vec![1, 1]));
                assert_eq!(reduction.arg_bool(5), Some(false));
                assert_eq!(reduction.meta.val_shape(), Some(expected_shape.as_slice()));
            }
            let result = graph.replacement_of(pool).unwrap();
            assert_eq!(graph.node_shape(result), Some(output_shape.as_slice()));
            for node in graph
                .live_nodes()
                .filter(|node| node.op == crate::fx_ir::NodeOp::CallFunction)
            {
                assert_eq!(node.meta.val_dtype(), Some("torch.int8"));
                assert_eq!(node.meta.has_nonempty_qparam_metadata(), node.id == result);
                if node.target == "aten.slice_copy.Tensor" {
                    let dim = node.arg_int(1).unwrap() as usize;
                    assert_eq!(node.arg_int(2), Some(0));
                    assert_eq!(node.arg_int(3), Some(output_shape[dim] * kernel[dim - 2]));
                }
            }
            assert!(!IrDecomposeLargeStrideMaxPool2dForU55Pass.run(&mut graph));
        }
    }

    #[test]
    fn divisible_inputs_and_scalar_attributes_need_no_crop() {
        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [1, 2, 10, 15], "torch.int16");
        call(
            &mut graph,
            "pool",
            "aten.max_pool2d.default",
            vec![
                IrArg::Node(x),
                IrArg::Int(5),
                IrArg::Int(5),
                IrArg::Int(0),
                IrArg::Int(1),
                IrArg::Bool(false),
            ],
            [1, 2, 2, 3],
            "torch.int16",
        );
        assert!(IrDecomposeLargeStrideMaxPool2dForU55Pass.run(&mut graph));
        assert!(!graph
            .live_nodes()
            .any(|node| node.target == "aten.slice_copy.Tensor"));
    }

    #[test]
    fn overlapping_and_tuple_output_pools_are_unchanged() {
        for (target, kernel, stride) in [
            ("aten.max_pool2d.default", 6, 4),
            ("aten.max_pool2d.default", 2, 2),
            ("aten.max_pool2d_with_indices.default", 5, 5),
        ] {
            let mut graph = IrGraph::empty();
            let x = placeholder(&mut graph, "x", [1, 2, 15, 15], "torch.int8");
            call(
                &mut graph,
                "pool",
                target,
                vec![IrArg::Node(x), IrArg::Int(kernel), IrArg::Int(stride)],
                [1, 2, 3, 3],
                "torch.int8",
            );
            assert!(!IrDecomposeLargeStrideMaxPool2dForU55Pass.run(&mut graph));
        }
    }

    #[test]
    fn value_only_tuple_pool_replaces_all_value_users_and_drops_unused_indices() {
        use crate::test_ir::output;

        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [1, 3, 13, 17], "torch.int16");
        let pool = call(
            &mut graph,
            "pool",
            "aten.max_pool2d_with_indices.default",
            vec![IrArg::Node(x), IrArg::Int(5), IrArg::Int(5)],
            [],
            "torch.int16",
        );
        let value = call(
            &mut graph,
            "value",
            "operator.getitem",
            vec![IrArg::Node(pool), IrArg::Int(0)],
            [1, 3, 2, 3],
            "torch.int16",
        );
        let duplicate_value = call(
            &mut graph,
            "duplicate_value",
            "operator.getitem",
            vec![IrArg::Node(pool), IrArg::Int(0)],
            [1, 3, 2, 3],
            "torch.int16",
        );
        let index = call(
            &mut graph,
            "indices",
            "operator.getitem",
            vec![IrArg::Node(pool), IrArg::Int(1)],
            [1, 3, 2, 3],
            "torch.int64",
        );
        graph.nodes[value]
            .meta
            .set_both_qparams(qparam(0.1, 0, "torch.int16"));
        output(&mut graph, value);
        assert!(IrDecomposeLargeStrideMaxPool2dForU55Pass.run(&mut graph));
        let result = graph.replacement_of(value).unwrap();
        assert_eq!(graph.replacement_of(duplicate_value), Some(result));
        assert_eq!(graph.node_shape(result), Some([1, 3, 2, 3].as_slice()));
        assert!(graph.nodes[result].meta.has_nonempty_qparam_metadata());
        assert!(!graph.nodes[pool].alive && !graph.nodes[index].alive);
        assert_eq!(
            graph
                .live_nodes()
                .filter(|node| node.target == "aten.max_pool2d.default")
                .count(),
            2
        );
    }

    #[test]
    fn tuple_pool_with_used_indices_is_unchanged() {
        use crate::test_ir::output;

        let mut graph = IrGraph::empty();
        let x = placeholder(&mut graph, "x", [1, 3, 13, 17], "torch.int8");
        let pool = call(
            &mut graph,
            "pool",
            "aten.max_pool2d_with_indices.default",
            vec![IrArg::Node(x), IrArg::Int(5), IrArg::Int(5)],
            [],
            "torch.int8",
        );
        call(
            &mut graph,
            "value",
            "operator.getitem",
            vec![IrArg::Node(pool), IrArg::Int(0)],
            [1, 3, 2, 3],
            "torch.int8",
        );
        let indices = call(
            &mut graph,
            "indices",
            "operator.getitem",
            vec![IrArg::Node(pool), IrArg::Int(1)],
            [1, 3, 2, 3],
            "torch.int64",
        );
        output(&mut graph, indices);
        assert!(!IrDecomposeLargeStrideMaxPool2dForU55Pass.run(&mut graph));
        assert!(graph.nodes[pool].alive && graph.nodes[indices].alive);
    }
}
