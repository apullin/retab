use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    register_unary!(table;
        OpTarget::TosaTable   => Op::Table,      Attribute::Table(TableAttribute {});
        OpTarget::TosaSigmoid => Op::Sigmoid,    Attribute::Sigmoid(SigmoidAttribute {});
        OpTarget::TosaTanh    => Op::Tanh,       Attribute::Tanh(TanhAttribute {});
        OpTarget::TosaExp     => Op::Exp,        Attribute::Exp(ExpAttribute {});
        OpTarget::TosaLog     => Op::Log,        Attribute::Log(LogAttribute {});
        OpTarget::TosaAbs     => Op::Abs,        Attribute::Abs(AbsAttribute {});
        OpTarget::TosaReciprocal => Op::Reciprocal, Attribute::Reciprocal(ReciprocalAttribute {});
        OpTarget::TosaErf     => Op::Erf,        Attribute::Erf(ErfAttribute {});
        OpTarget::TosaCos     => Op::Cos,        Attribute::Cos(CosAttribute {});
        OpTarget::TosaSin     => Op::Sin,        Attribute::Sin(SinAttribute {});
        OpTarget::TosaRsqrt   => Op::Rsqrt,      Attribute::Rsqrt(RsqrtAttribute {});
        OpTarget::Sigmoid     => Op::Sigmoid,    Attribute::Sigmoid(SigmoidAttribute {});
        OpTarget::Tanh        => Op::Tanh,       Attribute::Tanh(TanhAttribute {});
        OpTarget::Exp         => Op::Exp,        Attribute::Exp(ExpAttribute {});
        OpTarget::Log         => Op::Log,        Attribute::Log(LogAttribute {});
        OpTarget::Abs         => Op::Abs,        Attribute::Abs(AbsAttribute {});
        OpTarget::Reciprocal  => Op::Reciprocal, Attribute::Reciprocal(ReciprocalAttribute {});
        OpTarget::Rsqrt       => Op::Rsqrt,      Attribute::Rsqrt(RsqrtAttribute {});
        OpTarget::Sin         => Op::Sin,        Attribute::Sin(SinAttribute {});
        OpTarget::Cos         => Op::Cos,        Attribute::Cos(CosAttribute {});
    );
    table.extend([
        (OpTarget::TosaCast, emit_cast as EmitFn),
        (OpTarget::To, emit_cast),
    ]);
}

fn emit_cast(ctx: &EmitContext) -> EmitResult {
    if ctx.passthrough_dtype() == ctx.actual_dtype() {
        ctx.emit_identity()
    } else {
        ctx.emit_unary(Op::Cast, Attribute::Cast(CastAttribute {}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ir::{
        Arg, DType as IrDType, NodeMeta as GraphNodeMeta, OpTarget, OpType, TensorInfo,
    };
    use crate::graph::Graph;
    use crate::store::TensorStore;
    use smallvec::smallvec;

    fn meta(shape: &[i64], dtype: IrDType) -> GraphNodeMeta {
        GraphNodeMeta::new(TensorInfo::new(shape, dtype))
    }

    #[test]
    fn identity_cast_serializes_as_identity() {
        let mut graph = Graph::new();
        let x = graph.create_node(
            OpType::Placeholder,
            OpTarget::Other("x".to_string()),
            smallvec![],
            Vec::new(),
            meta(&[2, 3], IrDType::Int32),
            "x".to_string(),
        );
        let cast = graph.create_node(
            OpType::CallFunction,
            OpTarget::To,
            smallvec![Arg::Node(x)],
            Vec::new(),
            meta(&[2, 3], IrDType::Int32),
            "x_to_i32".to_string(),
        );
        let store = TensorStore::default();
        let ctx = EmitContext {
            graph: &graph,
            node: graph.node(cast).expect("cast node"),
            store: &store,
        };

        let result = emit_cast(&ctx);

        assert_eq!(result.operators.len(), 1);
        assert_eq!(result.operators[0].op, Op::Identity);
        assert_eq!(result.operators[0].inputs, vec!["x".to_string()]);
        assert_eq!(result.operators[0].outputs, vec!["x_to_i32".to_string()]);
    }
}
