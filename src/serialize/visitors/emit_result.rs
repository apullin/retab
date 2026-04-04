use super::*;

/// Result of visiting a single graph node: zero or more TOSA operators and
/// tensors (including const data tensors and shape entries) to append to the
/// basic block.
#[derive(Debug, Default)]
pub struct EmitResult {
    pub operators: Vec<TosaOperator>,
    pub tensors: Vec<TosaTensor>,
    pub shapes: Vec<TosaShape>,
}

impl EmitResult {
    pub fn push_op(
        &mut self,
        op: Op,
        attribute: Attribute,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) {
        self.operators
            .push(TosaOperator::new(op, attribute, inputs, outputs));
    }

    pub fn push_output_op(
        &mut self,
        output: TosaTensor,
        op: Op,
        attribute: Attribute,
        inputs: Vec<String>,
    ) {
        let output_name = output.name.clone();
        self.tensors.push(output);
        self.push_op(op, attribute, inputs, vec![output_name]);
    }

    pub fn push_const(&mut self, tensor: TosaTensor) -> String {
        let name = tensor.name.clone();
        self.tensors.push(tensor);
        self.push_op(
            Op::Const,
            Attribute::Const(ConstAttribute {}),
            vec![],
            vec![name.clone()],
        );
        name
    }

    // ── Chainable builder helpers for multi-op decompositions ──────────

    /// Emit a unary op: one input -> one output (same shape).
    pub fn unary(
        &mut self,
        name: String,
        shape: Vec<i32>,
        dtype: DType,
        op: Op,
        attr: Attribute,
        input: String,
    ) -> &mut Self {
        self.push_output_op(TosaTensor::new(&name, shape, dtype), op, attr, vec![input]);
        self
    }

    /// Emit a binary op: two inputs -> one output (same shape as first).
    pub fn binary(
        &mut self,
        name: String,
        shape: Vec<i32>,
        dtype: DType,
        op: Op,
        attr: Attribute,
        lhs: String,
        rhs: String,
    ) -> &mut Self {
        self.push_output_op(
            TosaTensor::new(&name, shape, dtype),
            op,
            attr,
            vec![lhs, rhs],
        );
        self
    }

    /// Emit a ternary op (e.g. MUL with shift, SELECT): three inputs -> one output.
    pub fn ternary(
        &mut self,
        name: String,
        shape: Vec<i32>,
        dtype: DType,
        op: Op,
        attr: Attribute,
        a: String,
        b: String,
        c: String,
    ) -> &mut Self {
        self.push_output_op(
            TosaTensor::new(&name, shape, dtype),
            op,
            attr,
            vec![a, b, c],
        );
        self
    }
}
