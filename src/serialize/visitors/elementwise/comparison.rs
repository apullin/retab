use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;
    table.extend([
        (OpTarget::TosaEqual, emit_equal as EmitFn),
        (OpTarget::TosaGreater, emit_greater),
        (OpTarget::TosaGreaterEqual, emit_greater_equal),
        (OpTarget::TosaMaximum, emit_maximum),
        (OpTarget::TosaMinimum, emit_minimum),
        (OpTarget::Gt, emit_aten_gt),
        (OpTarget::Ge, emit_aten_ge),
        (OpTarget::Lt, emit_aten_lt),
        (OpTarget::Le, emit_aten_le),
        (OpTarget::Eq, emit_aten_eq),
        (OpTarget::Pow, emit_pow),
    ]);
}

bool_binary_emitters! {
    pub(in crate::serialize::visitors) fn emit_equal => (Op::Equal, Attribute::Equal(EqualAttribute), false);
    pub(in crate::serialize::visitors) fn emit_greater => (Op::Greater, Attribute::Greater(GreaterAttribute), false);
    pub(in crate::serialize::visitors) fn emit_greater_equal => (Op::GreaterEqual, Attribute::GreaterEqual(GreaterEqualAttribute), false);
    pub(in crate::serialize::visitors) fn emit_aten_eq => (Op::Equal, Attribute::Equal(EqualAttribute), false);
    pub(in crate::serialize::visitors) fn emit_aten_gt => (Op::Greater, Attribute::Greater(GreaterAttribute), false);
    pub(in crate::serialize::visitors) fn emit_aten_ge => (Op::GreaterEqual, Attribute::GreaterEqual(GreaterEqualAttribute), false);
    pub(in crate::serialize::visitors) fn emit_aten_lt => (Op::Greater, Attribute::Greater(GreaterAttribute), true);
    pub(in crate::serialize::visitors) fn emit_aten_le => (Op::GreaterEqual, Attribute::GreaterEqual(GreaterEqualAttribute), true);
}

direct_binary_emitters! {
    pub(in crate::serialize::visitors) fn emit_pow => (Op::Pow, Attribute::Pow(PowAttribute));
    pub(in crate::serialize::visitors) fn emit_maximum => (
        Op::Maximum,
        Attribute::Maximum(MaximumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        })
    );
    pub(in crate::serialize::visitors) fn emit_minimum => (
        Op::Minimum,
        Attribute::Minimum(MinimumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        })
    );
}
