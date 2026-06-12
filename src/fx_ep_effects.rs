use crate::fx_fuse_constant_args::ZeroInputConstantKind;
use crate::graph::ir::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxEpInputKind {
    Parameter,
    Buffer,
    LiftedTensorConstant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxDeleteConstantPlaceholderEffect {
    pub placeholder_name: String,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxCastInt64InputToInt32Effect {
    pub placeholder_name: String,
    pub input_kind: FxEpInputKind,
    pub target_name: Option<String>,
    pub allow_missing: bool,
    pub update_placeholder_meta: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxMaterializeZeroInputConstantEffect {
    pub node_name: String,
    pub kind: ZeroInputConstantKind,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxFuseConstantNodeEffect {
    pub node_name: String,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxMaterializeConstantNodeEffect {
    pub node_name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxMaterializeZeroInputConstantNodeEffect {
    pub node_name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxTableTensorSpec {
    pub scale: f64,
    pub zp: i64,
    pub qmin: i64,
    pub qmax: i64,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxInsertTableOpEffect {
    pub node_name: String,
    pub kind: String,
    pub scalar_arg: Option<f64>,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxInsertPrecomputedTableOpEffect {
    pub node_name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub table_dtype: DType,
    pub input: FxTableTensorSpec,
    pub output: FxTableTensorSpec,
    pub rescale_lshift: i64,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FxEpEffect {
    DeleteConstantPlaceholder(FxDeleteConstantPlaceholderEffect),
    CastInt64InputToInt32(FxCastInt64InputToInt32Effect),
    MaterializeZeroInputConstant(FxMaterializeZeroInputConstantEffect),
    FuseConstantNode(FxFuseConstantNodeEffect),
    MaterializeConstantNode(FxMaterializeConstantNodeEffect),
    MaterializeZeroInputConstantNode(FxMaterializeZeroInputConstantNodeEffect),
    InsertTableOp(FxInsertTableOpEffect),
    InsertPrecomputedTableOp(FxInsertPrecomputedTableOpEffect),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FxEpEffectBatch {
    pub effects: Vec<FxEpEffect>,
}

impl FxEpEffectBatch {
    pub fn push(&mut self, effect: FxEpEffect) {
        self.effects.push(effect);
    }
}
