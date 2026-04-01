use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaTable, emit_table as EmitFn),
        (OpTarget::TosaSigmoid, emit_sigmoid),
        (OpTarget::TosaTanh, emit_tanh),
        (OpTarget::TosaExp, emit_exp),
        (OpTarget::TosaLog, emit_log),
        (OpTarget::TosaAbs, emit_abs),
        (OpTarget::TosaReciprocal, emit_reciprocal),
        (OpTarget::TosaErf, emit_erf),
        (OpTarget::TosaCos, emit_cos),
        (OpTarget::TosaSin, emit_sin),
        (OpTarget::TosaRsqrt, emit_rsqrt),
        (OpTarget::TosaCast, emit_cast),
        (OpTarget::Sigmoid, emit_sigmoid),
        (OpTarget::Tanh, emit_tanh),
        (OpTarget::Exp, emit_exp),
        (OpTarget::Log, emit_log),
        (OpTarget::Abs, emit_abs),
        (OpTarget::Reciprocal, emit_reciprocal),
        (OpTarget::Rsqrt, emit_rsqrt),
        (OpTarget::Sin, emit_sin),
        (OpTarget::Cos, emit_cos),
        (OpTarget::To, emit_cast),
    ]);
}

direct_unary_emitters! {
    pub(in crate::serialize::visitors) fn emit_table => (Op::Table, Attribute::Table(TableAttribute));
    pub(in crate::serialize::visitors) fn emit_sigmoid => (Op::Sigmoid, Attribute::Sigmoid(SigmoidAttribute));
    pub(in crate::serialize::visitors) fn emit_tanh => (Op::Tanh, Attribute::Tanh(TanhAttribute));
    pub(in crate::serialize::visitors) fn emit_exp => (Op::Exp, Attribute::Exp(ExpAttribute));
    pub(in crate::serialize::visitors) fn emit_log => (Op::Log, Attribute::Log(LogAttribute));
    pub(in crate::serialize::visitors) fn emit_abs => (Op::Abs, Attribute::Abs(AbsAttribute));
    pub(in crate::serialize::visitors) fn emit_reciprocal => (Op::Reciprocal, Attribute::Reciprocal(ReciprocalAttribute));
    pub(in crate::serialize::visitors) fn emit_erf => (Op::Erf, Attribute::Erf(ErfAttribute));
    pub(in crate::serialize::visitors) fn emit_cos => (Op::Cos, Attribute::Cos(CosAttribute));
    pub(in crate::serialize::visitors) fn emit_sin => (Op::Sin, Attribute::Sin(SinAttribute));
    pub(in crate::serialize::visitors) fn emit_rsqrt => (Op::Rsqrt, Attribute::Rsqrt(RsqrtAttribute));
    pub(in crate::serialize::visitors) fn emit_cast => (Op::Cast, Attribute::Cast(CastAttribute));
}
