//! Graph-building DSL for layout tests.
//!
//! A test names targets, arguments and input shapes. Each call gets its value
//! facts from `shape`'s local refresh, the way rc4's layout tests attach
//! `meta["val"]` to hand-built FX graphs (for example
//! `backends/transforms/test/test_propagate_view_copy_permute_pass.py`), so
//! fixtures cannot drift from the shapes the passes will compute. Assertions
//! read live call targets in graph order.

use std::collections::HashMap;

use crate::fx_ir::{i64_list_arg, IrArg, IrGraph, NodeId, NodeMeta, NodeOp};
use crate::layout::shape::refreshed_meta;

pub(crate) const PERMUTE: &str = "aten.permute_copy.default";
pub(crate) const VIEW: &str = "aten.view_copy.default";
pub(crate) const SLICE: &str = "aten.slice_copy.Tensor";
pub(crate) const CAT: &str = "aten.cat.default";
pub(crate) const MEAN: &str = "aten.mean.dim";
pub(crate) const SUM: &str = "aten.sum.dim_IntList";
pub(crate) const RESCALE: &str = "backend.tosa.RESCALE.default";
pub(crate) const CLONE: &str = "dim_order_ops._clone_dim_order.default";
pub(crate) const TO_COPY: &str = "aten._to_copy.default";
pub(crate) const ADD: &str = "aten.add.Tensor";
pub(crate) const ADD_INPLACE: &str = "aten.add_.Tensor";
pub(crate) const MUL: &str = "aten.mul.Tensor";
pub(crate) const ABS: &str = "aten.abs.default";
pub(crate) const NEG: &str = "aten.neg.default";
pub(crate) const RELU: &str = "aten.relu.default";
pub(crate) const SIGMOID: &str = "aten.sigmoid.default";
pub(crate) const CLAMP_TENSOR: &str = "aten.clamp.Tensor";
pub(crate) const WHERE: &str = "aten.where.self";
pub(crate) const MASKED_FILL: &str = "aten.masked_fill.Scalar";

/// A tensor operand.
pub(crate) fn node(id: NodeId) -> IrArg {
    IrArg::Node(id)
}

/// An `int[]` argument.
pub(crate) fn ints(values: &[i64]) -> IrArg {
    i64_list_arg(values)
}

/// An `IrGraph` built for a layout test.
pub(crate) struct LayoutGraph {
    graph: IrGraph,
    name_uses: HashMap<String, usize>,
}

impl LayoutGraph {
    pub(crate) fn new() -> Self {
        Self {
            graph: IrGraph::empty(),
            name_uses: HashMap::new(),
        }
    }

    /// A float32 graph input.
    pub(crate) fn input(&mut self, name: &str, shape: &[i64]) -> NodeId {
        self.typed_input(name, shape, "torch.float32")
    }

    /// A graph input of `dtype`.
    pub(crate) fn typed_input(&mut self, name: &str, shape: &[i64], dtype: &str) -> NodeId {
        let name = self.fresh_name(name);
        self.graph.add_node(
            NodeOp::Placeholder,
            name.clone(),
            name,
            Vec::new(),
            NodeMeta::tensor(shape.to_vec(), dtype),
        )
    }

    /// A call whose meta comes from the local refresh. Panics if no rule
    /// covers it or its arguments are invalid.
    pub(crate) fn call(&mut self, target: &str, args: Vec<IrArg>) -> NodeId {
        self.call_with_kwargs(target, args, &[])
    }

    pub(crate) fn call_with_kwargs(
        &mut self,
        target: &str,
        args: Vec<IrArg>,
        kwargs: &[(&str, IrArg)],
    ) -> NodeId {
        let id = self.add_call(target, args, kwargs);
        self.refresh(id);
        id
    }

    /// A call left with empty meta, for tests of the refresh itself.
    pub(crate) fn add_unrefreshed(&mut self, target: &str, args: Vec<IrArg>) -> NodeId {
        self.add_call(target, args, &[])
    }

    pub(crate) fn permute(&mut self, input: NodeId, dims: &[i64]) -> NodeId {
        self.call(PERMUTE, vec![node(input), ints(dims)])
    }

    pub(crate) fn view(&mut self, input: NodeId, size: &[i64]) -> NodeId {
        self.call(VIEW, vec![node(input), ints(size)])
    }

    pub(crate) fn slice(&mut self, input: NodeId, dim: i64, start: i64, end: i64) -> NodeId {
        self.call(
            SLICE,
            vec![
                node(input),
                IrArg::Int(dim),
                IrArg::Int(start),
                IrArg::Int(end),
            ],
        )
    }

    pub(crate) fn cat(&mut self, inputs: &[NodeId], dim: i64) -> NodeId {
        let tensors = IrArg::List(inputs.iter().map(|&input| node(input)).collect());
        self.call(CAT, vec![tensors, IrArg::Int(dim)])
    }

    /// A per-tensor RESCALE with zero points of zero.
    pub(crate) fn rescale(&mut self, input: NodeId, dtype: &str, scale: f64) -> NodeId {
        self.call(
            RESCALE,
            vec![
                node(input),
                IrArg::Dtype(dtype.to_string()),
                IrArg::List(vec![IrArg::Float(scale)]),
                IrArg::Int(0),
                IrArg::Int(0),
            ],
        )
    }

    /// The graph output, holding `outputs` in a list as exported graphs do.
    pub(crate) fn output(&mut self, outputs: &[NodeId]) -> NodeId {
        let results = IrArg::List(outputs.iter().map(|&output| node(output)).collect());
        self.graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![results],
            NodeMeta::default(),
        )
    }

    /// Re-derive `id`'s meta from its current operands.
    pub(crate) fn refresh(&mut self, id: NodeId) {
        let meta = refreshed_meta(&self.graph, id).unwrap_or_else(|error| {
            let node = self.graph.node(id);
            panic!("cannot refresh {} ({}): {error}", node.name, node.target)
        });
        self.graph.node_mut(id).meta = meta;
    }

    /// Point `user`'s uses of `old` at `new`.
    pub(crate) fn rewire(&mut self, user: NodeId, old: NodeId, new: NodeId) {
        assert!(
            self.graph.replace_input_use(user, old, new),
            "{} does not use {}",
            self.graph.node(user).name,
            self.graph.node(old).name
        );
    }

    /// The name of `id`, which survives the renumbering `LayoutCtx::finish` does.
    pub(crate) fn name(&self, id: NodeId) -> String {
        self.graph().node(id).name.clone()
    }

    /// The current id of the node named `name`. A pass that runs in a
    /// `LayoutCtx` renumbers the graph, so tests re-resolve the ids they held.
    pub(crate) fn id(&self, name: &str) -> NodeId {
        self.graph()
            .by_name(name)
            .unwrap_or_else(|| panic!("no node named {name}"))
            .id
    }

    pub(crate) fn graph(&self) -> &IrGraph {
        &self.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut IrGraph {
        &mut self.graph
    }

    pub(crate) fn shape(&self, id: NodeId) -> &[i64] {
        self.graph
            .node_shape(id)
            .unwrap_or_else(|| panic!("{} has no shape", self.graph.node(id).name))
    }

    pub(crate) fn dtype(&self, id: NodeId) -> &str {
        self.graph
            .node_dtype(id)
            .unwrap_or_else(|| panic!("{} has no dtype", self.graph.node(id).name))
    }

    /// Live call targets in graph order (`IrGraph::topological_live_node_ids`).
    pub(crate) fn targets(&self) -> Vec<&str> {
        self.live_calls()
            .map(|id| self.graph.node(id).target.as_str())
            .collect()
    }

    pub(crate) fn assert_targets(&self, expected: &[&str]) {
        assert_eq!(self.targets(), expected);
    }

    pub(crate) fn count(&self, target: &str) -> usize {
        self.targets()
            .into_iter()
            .filter(|&live| live == target)
            .count()
    }

    /// Asserts the numbers of live permute and view copies.
    pub(crate) fn assert_counts(&self, permutes: usize, views: usize) {
        assert_eq!(
            (self.count(PERMUTE), self.count(VIEW)),
            (permutes, views),
            "(permutes, views) in {:?}",
            self.targets()
        );
    }

    /// The dims of every live permute, in graph order.
    pub(crate) fn permute_dims(&self) -> Vec<Vec<i64>> {
        self.live_calls()
            .map(|id| self.graph.node(id))
            .filter(|node| node.target == PERMUTE)
            .map(|node| {
                node.arg_int_list(1)
                    .unwrap_or_else(|| panic!("{} has no dims", node.name))
            })
            .collect()
    }

    pub(crate) fn assert_permute_dims(&self, expected: &[&[i64]]) {
        assert_eq!(self.permute_dims(), expected);
    }

    fn live_calls(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.graph
            .topological_live_node_ids()
            .into_iter()
            .filter(|&id| self.graph.node(id).op == NodeOp::CallFunction)
    }

    fn add_call(&mut self, target: &str, args: Vec<IrArg>, kwargs: &[(&str, IrArg)]) -> NodeId {
        let name = self.fresh_name(&op_name(target));
        let kwargs = kwargs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect();
        self.graph
            .add_call_function_with_kwargs(name, target, args, kwargs, NodeMeta::default())
    }

    /// FX-style unique names: `relu`, `relu_1`, `relu_2`, ...
    fn fresh_name(&mut self, base: &str) -> String {
        let uses = self.name_uses.entry(base.to_string()).or_insert(0);
        let name = match *uses {
            0 => base.to_string(),
            n => format!("{base}_{n}"),
        };
        *uses += 1;
        name
    }
}

/// The op segment of a target, e.g. `permute_copy` for
/// `aten.permute_copy.default`.
fn op_name(target: &str) -> String {
    target
        .rsplit('.')
        .nth(1)
        .unwrap_or(target)
        .to_ascii_lowercase()
}
