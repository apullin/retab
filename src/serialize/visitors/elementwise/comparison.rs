use super::*;

pub(super) fn register(table: &mut EmitterTable) {
    use crate::graph::ir::OpTarget;

    // Bool-output binary comparisons
    register_bool_binary!(table;
        OpTarget::TosaEqual        => Op::Equal,        Attribute::Equal(EqualAttribute {}),        false;
        OpTarget::TosaGreater      => Op::Greater,      Attribute::Greater(GreaterAttribute {}),      false;
        OpTarget::TosaGreaterEqual => Op::GreaterEqual,  Attribute::GreaterEqual(GreaterEqualAttribute {}), false;
        OpTarget::Eq               => Op::Equal,        Attribute::Equal(EqualAttribute {}),        false;
        OpTarget::Gt               => Op::Greater,      Attribute::Greater(GreaterAttribute {}),      false;
        OpTarget::Ge               => Op::GreaterEqual,  Attribute::GreaterEqual(GreaterEqualAttribute {}), false;
        OpTarget::Lt               => Op::Greater,      Attribute::Greater(GreaterAttribute {}),      true;
        OpTarget::Le               => Op::GreaterEqual,  Attribute::GreaterEqual(GreaterEqualAttribute {}), true;
    );

    // Direct binary ops
    register_binary!(table;
        OpTarget::Pow         => Op::Pow, Attribute::Pow(PowAttribute {});
        OpTarget::Maximum     => Op::Maximum, Attribute::Maximum(MaximumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        });
        OpTarget::Minimum     => Op::Minimum, Attribute::Minimum(MinimumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        });
        OpTarget::TosaMaximum => Op::Maximum, Attribute::Maximum(MaximumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        });
        OpTarget::TosaMinimum => Op::Minimum, Attribute::Minimum(MinimumAttribute {
            nan_mode: NanPropagationMode::Propagate,
        });
    );
}
