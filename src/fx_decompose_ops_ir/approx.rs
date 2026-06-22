use crate::fx_ir::{
    i64_list_arg as i64_list, IrArg, IrGraph, IrRewriteCursor, NodeId, NodeMeta, TensorMetaStamp,
};

pub(super) struct MetaFactory<'a> {
    tensor: TensorMetaStamp<'a>,
    bool_tensor: TensorMetaStamp<'a>,
}

impl<'a> MetaFactory<'a> {
    fn new(base: &'a NodeMeta, dtype: &'a str) -> Self {
        Self {
            tensor: base.tensor_stamp(dtype),
            bool_tensor: base.tensor_stamp("torch.bool"),
        }
    }

    pub(super) fn from_tensor_stamp(tensor: TensorMetaStamp<'a>) -> Self {
        Self {
            bool_tensor: tensor.with_dtype("torch.bool"),
            tensor,
        }
    }

    pub(super) fn tensor(&self, shape: &[i64]) -> NodeMeta {
        self.tensor.preserve_qparams(shape.to_vec())
    }

    fn bool(&self, shape: &[i64]) -> NodeMeta {
        self.bool_tensor.preserve_qparams(shape.to_vec())
    }
}

#[derive(Clone)]
pub(super) struct ApproxCtx {
    anchor_id: NodeId,
    name: String,
    pub(super) shape: Vec<i64>,
    dtype: String,
    meta: NodeMeta,
}

impl ApproxCtx {
    pub(super) fn from_input(graph: &IrGraph, anchor_id: NodeId, input_id: NodeId) -> Option<Self> {
        Some(Self {
            anchor_id,
            name: graph.nodes[anchor_id].name.clone(),
            shape: graph.node_shape_vec(input_id)?,
            dtype: graph.node_dtype_or_float32(input_id).to_string(),
            meta: graph.nodes[anchor_id].meta.clone(),
        })
    }

    pub(super) fn from_first_input(graph: &IrGraph, anchor_id: NodeId) -> Option<(NodeId, Self)> {
        let input_id = graph.nodes[anchor_id].first_node_arg()?;
        let ctx = Self::from_input(graph, anchor_id, input_id)?;
        Some((input_id, ctx))
    }

    fn dtype(&self) -> &str {
        &self.dtype
    }

    pub(super) fn with_meta(&self, meta: NodeMeta) -> Self {
        Self {
            anchor_id: self.anchor_id,
            name: self.name.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            meta,
        }
    }

    fn metas(&self) -> MetaFactory<'_> {
        MetaFactory::new(&self.meta, &self.dtype)
    }

    fn tensor_meta(&self, shape: &[i64]) -> NodeMeta {
        self.metas().tensor(shape)
    }

    fn bool_meta(&self, shape: &[i64]) -> NodeMeta {
        self.metas().bool(shape)
    }

    fn op_name(&self, suffix: &str) -> String {
        format!("{}_{}", self.name, suffix)
    }

    pub(super) fn full(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        value: f64,
    ) -> NodeId {
        self.full_shape(cursor, suffix, value, &self.shape)
    }

    pub(super) fn scalar_shape(&self) -> Vec<i64> {
        vec![1; self.shape.len()]
    }

    pub(super) fn full_scalar(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        value: f64,
    ) -> NodeId {
        let shape = self.scalar_shape();
        self.full_shape(cursor, suffix, value, &shape)
    }

    pub(super) fn full_shape(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        value: f64,
        shape: &[i64],
    ) -> NodeId {
        cursor.full_float(
            self.op_name(suffix),
            shape,
            value,
            self.dtype(),
            self.tensor_meta(shape),
        )
    }

    fn tensor(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
    ) -> NodeId {
        self.tensor_shape(cursor, suffix, target, args, &self.shape)
    }

    fn tensor_nodes(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
    ) -> NodeId {
        self.tensor(
            cursor,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
        )
    }

    pub(super) fn tensor_scalar(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
    ) -> NodeId {
        let shape = self.scalar_shape();
        self.tensor_shape(cursor, suffix, target, args, &shape)
    }

    pub(super) fn tensor_shape(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
        shape: &[i64],
    ) -> NodeId {
        cursor.call_function(self.op_name(suffix), target, args, self.tensor_meta(shape))
    }

    pub(super) fn tensor_nodes_shape(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
        shape: &[i64],
    ) -> NodeId {
        self.tensor_shape(
            cursor,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
            shape,
        )
    }

    fn bool(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
    ) -> NodeId {
        self.bool_shape(cursor, suffix, target, args, &self.shape)
    }

    fn bool_nodes(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
    ) -> NodeId {
        self.bool(
            cursor,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
        )
    }

    fn bool_shape(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
        shape: &[i64],
    ) -> NodeId {
        cursor.call_function(self.op_name(suffix), target, args, self.bool_meta(shape))
    }

    fn output(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
    ) -> NodeId {
        cursor.call_function(self.op_name(suffix), target, args, self.meta.clone())
    }

    pub(super) fn output_nodes(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
    ) -> NodeId {
        self.output(
            cursor,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
        )
    }

    fn replace_output(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        args: Vec<IrArg>,
    ) -> NodeId {
        let replacement = self.output(cursor, suffix, target, args);
        cursor.replace_anchor_with(replacement)
    }

    pub(super) fn replace_nodes(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        nodes: &[NodeId],
    ) -> NodeId {
        self.replace_output(
            cursor,
            suffix,
            target,
            nodes.iter().copied().map(IrArg::Node).collect(),
        )
    }

    pub(super) fn view_shape(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        source: NodeId,
        shape: &[i64],
    ) -> NodeId {
        cursor.view_copy(self.op_name(suffix), source, shape, self.tensor_meta(shape))
    }

    pub(super) fn reshape_shape(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        source: NodeId,
        shape: &[i64],
    ) -> NodeId {
        self.tensor_shape(
            cursor,
            suffix,
            target,
            vec![IrArg::Node(source), i64_list(shape)],
            shape,
        )
    }

    pub(super) fn add(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        lhs: NodeId,
        rhs: NodeId,
    ) -> NodeId {
        self.tensor_nodes(cursor, suffix, "aten.add.Tensor", &[lhs, rhs])
    }

    pub(super) fn sub(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        lhs: NodeId,
        rhs: NodeId,
    ) -> NodeId {
        self.tensor_nodes(cursor, suffix, "aten.sub.Tensor", &[lhs, rhs])
    }

    pub(super) fn mul(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        lhs: NodeId,
        rhs: NodeId,
    ) -> NodeId {
        self.tensor_nodes(cursor, suffix, "aten.mul.Tensor", &[lhs, rhs])
    }

    pub(super) fn div(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        lhs: NodeId,
        rhs: NodeId,
    ) -> NodeId {
        self.tensor_nodes(cursor, suffix, "aten.div.Tensor", &[lhs, rhs])
    }

    pub(super) fn unary(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        target: &str,
        input: NodeId,
    ) -> NodeId {
        self.tensor_nodes(cursor, suffix, target, &[input])
    }

    pub(super) fn neg(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        input: NodeId,
    ) -> NodeId {
        self.unary(cursor, suffix, "aten.neg.default", input)
    }

    pub(super) fn exp(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        input: NodeId,
    ) -> NodeId {
        self.unary(cursor, suffix, "aten.exp.default", input)
    }

    pub(super) fn sqrt(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        input: NodeId,
    ) -> NodeId {
        self.unary(cursor, suffix, "aten.sqrt.default", input)
    }

    pub(super) fn log(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        input: NodeId,
    ) -> NodeId {
        self.unary(cursor, suffix, "aten.log.default", input)
    }

    pub(super) fn abs(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        input: NodeId,
    ) -> NodeId {
        self.unary(cursor, suffix, "aten.abs.default", input)
    }

    pub(super) fn reciprocal(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        input: NodeId,
    ) -> NodeId {
        self.unary(cursor, suffix, "aten.reciprocal.default", input)
    }

    pub(super) fn tanh(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        input: NodeId,
    ) -> NodeId {
        self.unary(cursor, suffix, "aten.tanh.default", input)
    }

    pub(super) fn erf(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        input: NodeId,
    ) -> NodeId {
        self.unary(cursor, suffix, "aten.erf.default", input)
    }

    pub(super) fn ge(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        lhs: NodeId,
        rhs: NodeId,
    ) -> NodeId {
        self.bool_nodes(cursor, suffix, "aten.ge.Tensor", &[lhs, rhs])
    }

    pub(super) fn gt(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        lhs: NodeId,
        rhs: NodeId,
    ) -> NodeId {
        self.bool_nodes(cursor, suffix, "aten.gt.Tensor", &[lhs, rhs])
    }

    pub(super) fn lt(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        lhs: NodeId,
        rhs: NodeId,
    ) -> NodeId {
        self.bool_nodes(cursor, suffix, "aten.lt.Tensor", &[lhs, rhs])
    }

    pub(super) fn eq(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        lhs: NodeId,
        rhs: NodeId,
    ) -> NodeId {
        self.bool_nodes(cursor, suffix, "aten.eq.Tensor", &[lhs, rhs])
    }

    pub(super) fn where_self(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        suffix: &str,
        condition: NodeId,
        when_true: NodeId,
        when_false: NodeId,
    ) -> NodeId {
        self.tensor_nodes(
            cursor,
            suffix,
            "aten.where.self",
            &[condition, when_true, when_false],
        )
    }

    pub(super) fn poly(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        prefix: &str,
        coefficients: &[f64],
        variable_id: NodeId,
    ) -> Option<NodeId> {
        self.poly_shape(cursor, prefix, coefficients, variable_id, &self.shape)
    }

    fn poly_shape(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        prefix: &str,
        coefficients: &[f64],
        variable_id: NodeId,
        shape: &[i64],
    ) -> Option<NodeId> {
        let first = *coefficients.first()?;
        let mut result = self.full_shape(cursor, &format!("{prefix}_c0"), first, shape);
        let mut current_var = variable_id;

        for (idx, coeff) in coefficients.iter().enumerate().skip(1) {
            let coeff_id = self.full_shape(cursor, &format!("{prefix}_c{idx}"), *coeff, shape);
            let term = self.tensor_shape(
                cursor,
                &format!("{prefix}_term{idx}"),
                "aten.mul.Tensor",
                vec![IrArg::Node(current_var), IrArg::Node(coeff_id)],
                shape,
            );
            let sum = self.tensor_shape(
                cursor,
                &format!("{prefix}_sum{idx}"),
                "aten.add.Tensor",
                vec![IrArg::Node(result), IrArg::Node(term)],
                shape,
            );
            result = sum;

            if idx + 1 < coefficients.len() {
                current_var = self.tensor_shape(
                    cursor,
                    &format!("{prefix}_pow{idx}"),
                    "aten.mul.Tensor",
                    vec![IrArg::Node(current_var), IrArg::Node(current_var)],
                    shape,
                );
            }
        }

        Some(result)
    }

    pub(super) fn horner_nodes(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        prefix: &str,
        variable_id: NodeId,
        coefficients: &[NodeId],
    ) -> Option<NodeId> {
        self.horner_nodes_shape(cursor, prefix, variable_id, coefficients, &self.shape)
    }

    fn horner_nodes_shape(
        &self,
        cursor: &mut IrRewriteCursor<'_>,
        prefix: &str,
        variable_id: NodeId,
        coefficients: &[NodeId],
        shape: &[i64],
    ) -> Option<NodeId> {
        let (&last, rest) = coefficients.split_last()?;
        let mut acc = last;

        for (step, coeff_id) in rest.iter().rev().enumerate() {
            let mul = self.tensor_shape(
                cursor,
                &format!("{prefix}_mul{step}"),
                "aten.mul.Tensor",
                vec![IrArg::Node(acc), IrArg::Node(variable_id)],
                shape,
            );
            acc = self.tensor_shape(
                cursor,
                &format!("{prefix}_add{step}"),
                "aten.add.Tensor",
                vec![IrArg::Node(mul), IrArg::Node(*coeff_id)],
                shape,
            );
        }

        Some(acc)
    }
}
