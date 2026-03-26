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
        self.operators.push(TosaOperator {
            op,
            attribute,
            inputs,
            outputs,
            location: None,
        });
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
            Attribute::Const(ConstAttribute),
            vec![],
            vec![name.clone()],
        );
        name
    }
}
