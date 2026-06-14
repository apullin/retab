use crate::fx_fuse_constant_args::ZeroInputConstantKind;
use crate::graph::ir::DType;
use crate::store::TensorData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweringInputKind {
    Parameter,
    Buffer,
    LiftedTensorConstant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteConstantPlaceholderEffect {
    pub placeholder_name: String,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CastInt64InputToInt32Effect {
    pub placeholder_name: String,
    pub input_kind: LoweringInputKind,
    pub target_name: Option<String>,
    pub allow_missing: bool,
    pub update_placeholder_meta: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializeZeroInputConstantEffect {
    pub node_name: String,
    pub kind: ZeroInputConstantKind,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuseConstantNodeEffect {
    pub node_name: String,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaterializeTensorNodeEffect {
    pub node_name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub allow_missing: bool,
}

impl MaterializeTensorNodeEffect {
    pub fn from_tensor(node_name: String, tensor: TensorData, allow_missing: bool) -> Self {
        Self {
            node_name,
            data: tensor.data,
            shape: tensor.shape,
            dtype: tensor.dtype,
            allow_missing,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableTensorSpec {
    pub scale: f64,
    pub zp: i64,
    pub qmin: i64,
    pub qmax: i64,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InsertTableOpEffect {
    pub node_name: String,
    pub kind: String,
    pub scalar_arg: Option<f64>,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InsertPrecomputedTableOpEffect {
    pub node_name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub table_dtype: DType,
    pub input: TableTensorSpec,
    pub output: TableTensorSpec,
    pub rescale_lshift: i64,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoweringEffect {
    DeleteConstantPlaceholder(DeleteConstantPlaceholderEffect),
    CastInt64InputToInt32(CastInt64InputToInt32Effect),
    MaterializeZeroInputConstant(MaterializeZeroInputConstantEffect),
    FuseConstantNode(FuseConstantNodeEffect),
    MaterializeTensorNode(MaterializeTensorNodeEffect),
    InsertTableOp(InsertTableOpEffect),
    InsertPrecomputedTableOp(InsertPrecomputedTableOpEffect),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoweringEffectBatch {
    pub items: Vec<LoweringEffect>,
}

impl LoweringEffectBatch {
    pub fn push(&mut self, effect: LoweringEffect) {
        self.items.push(effect);
    }
}
