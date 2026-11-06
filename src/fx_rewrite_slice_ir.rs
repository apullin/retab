use crate::fx_fuse_consecutive_slices_ir::static_slice;
use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, IrPass};
use crate::fx_target::is_slice_copy_tensor_target;

pub struct IrRewriteSlicePass;

impl IrPass for IrRewriteSlicePass {
    fn name(&self) -> &str {
        "IrRewriteSlice"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut modified = false;
        for id in graph.topological_live_node_ids() {
            if !is_slice_copy_tensor_target(&graph.node(id).target) {
                continue;
            }
            let Some(slice) = static_slice(graph, id) else {
                continue;
            };
            graph.set_args(
                id,
                vec![
                    IrArg::Node(slice.input),
                    i64_list_arg(&slice.start),
                    i64_list_arg(&slice.size),
                ],
            );
            let node = graph.node_mut(id);
            node.target = "backend.tosa.SLICE.default".into();
            node.meta = node.meta.cloned_with_same_dtype_or_float32(slice.size);
            modified = true;
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{f32_call, f32_placeholder, output};

    #[test]
    fn rewrites_static_slice_bounds_of_a_transposed_input() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [1, 3, 8, 10]);
        let nhwc = f32_call(
            &mut graph,
            "nhwc",
            "backend.tosa.TRANSPOSE.default",
            vec![IrArg::Node(x), i64_list_arg(&[0, 2, 3, 1])],
            [1, 8, 10, 3],
        );
        let sliced = f32_call(
            &mut graph,
            "slice",
            "aten.slice_copy.Tensor",
            vec![
                IrArg::Node(nhwc),
                IrArg::Int(1),
                IrArg::Int(-6),
                IrArg::Int(i64::MAX),
            ],
            [1, 6, 10, 3],
        );
        output(&mut graph, sliced);
        assert!(IrRewriteSlicePass.run(&mut graph));
        assert_eq!(graph.node(sliced).target, "backend.tosa.SLICE.default");
        assert_eq!(graph.node(sliced).arg_int_list(1), Some(vec![0, 2, 0, 0]));
        assert_eq!(graph.node(sliced).arg_int_list(2), Some(vec![1, 6, 10, 3]));
    }

    #[test]
    fn keeps_empty_dynamic_and_strided_slices() {
        for (start, end, step) in [
            (IrArg::Int(1), IrArg::Int(1), 1),
            (IrArg::None, IrArg::None, 2),
        ] {
            let mut graph = IrGraph::empty();
            let x = f32_placeholder(&mut graph, "x", [2, 3]);
            let slice = f32_call(
                &mut graph,
                "slice",
                "aten.slice_copy.Tensor",
                vec![IrArg::Node(x), IrArg::Int(1), start, end, IrArg::Int(step)],
                [2, 0],
            );
            output(&mut graph, slice);
            assert!(!IrRewriteSlicePass.run(&mut graph));
        }
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 3]);
        let bound = f32_placeholder(&mut graph, "bound", []);
        let sliced = f32_call(
            &mut graph,
            "slice",
            "aten.slice_copy.Tensor",
            vec![
                IrArg::Node(x),
                IrArg::Int(1),
                IrArg::Int(0),
                IrArg::Node(bound),
            ],
            [2, 3],
        );
        output(&mut graph, sliced);
        assert!(!IrRewriteSlicePass.run(&mut graph));
    }
}
