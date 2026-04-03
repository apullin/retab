use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    register_unary!(table;
        OpTarget::TosaTable   => Op::Table,      Attribute::Table(TableAttribute);
        OpTarget::TosaSigmoid => Op::Sigmoid,    Attribute::Sigmoid(SigmoidAttribute);
        OpTarget::TosaTanh    => Op::Tanh,       Attribute::Tanh(TanhAttribute);
        OpTarget::TosaExp     => Op::Exp,        Attribute::Exp(ExpAttribute);
        OpTarget::TosaLog     => Op::Log,        Attribute::Log(LogAttribute);
        OpTarget::TosaAbs     => Op::Abs,        Attribute::Abs(AbsAttribute);
        OpTarget::TosaReciprocal => Op::Reciprocal, Attribute::Reciprocal(ReciprocalAttribute);
        OpTarget::TosaErf     => Op::Erf,        Attribute::Erf(ErfAttribute);
        OpTarget::TosaCos     => Op::Cos,        Attribute::Cos(CosAttribute);
        OpTarget::TosaSin     => Op::Sin,        Attribute::Sin(SinAttribute);
        OpTarget::TosaRsqrt   => Op::Rsqrt,      Attribute::Rsqrt(RsqrtAttribute);
        OpTarget::TosaCast    => Op::Cast,        Attribute::Cast(CastAttribute);
        OpTarget::Sigmoid     => Op::Sigmoid,    Attribute::Sigmoid(SigmoidAttribute);
        OpTarget::Tanh        => Op::Tanh,       Attribute::Tanh(TanhAttribute);
        OpTarget::Exp         => Op::Exp,        Attribute::Exp(ExpAttribute);
        OpTarget::Log         => Op::Log,        Attribute::Log(LogAttribute);
        OpTarget::Abs         => Op::Abs,        Attribute::Abs(AbsAttribute);
        OpTarget::Reciprocal  => Op::Reciprocal, Attribute::Reciprocal(ReciprocalAttribute);
        OpTarget::Rsqrt       => Op::Rsqrt,      Attribute::Rsqrt(RsqrtAttribute);
        OpTarget::Sin         => Op::Sin,        Attribute::Sin(SinAttribute);
        OpTarget::Cos         => Op::Cos,        Attribute::Cos(CosAttribute);
        OpTarget::To          => Op::Cast,        Attribute::Cast(CastAttribute);
    );
}
