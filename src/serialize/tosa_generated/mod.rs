#![allow(dead_code)]

// Compatibility re-export module: bridges tosa-rs types to the API
// that the rest of retab expects. Where types are identical they are
// re-exported directly; where they differ thin wrappers preserve the
// existing call-site contract.

// ---------------------------------------------------------------------------
// Enums -- identical, re-export directly
// ---------------------------------------------------------------------------
pub use tosa_rs::{DType, NanPropagationMode, Op, ResizeMode, RoundingMode};

// ---------------------------------------------------------------------------
// Attribute structs -- re-export those whose fields match exactly
// ---------------------------------------------------------------------------
pub use tosa_rs::{
    AvgPool2dAttribute, ClampAttribute, ConcatAttribute, CondIfAttribute, Conv2dAttribute,
    Conv3dAttribute, DepthwiseConv2dAttribute, MaxPool2dAttribute, MaximumAttribute,
    MinimumAttribute, ReduceAnyAttribute, ReduceMaxAttribute, ReduceMinAttribute,
    ReduceSumAttribute, RescaleAttribute, ResizeAttribute, TransposeAttribute,
    TransposeConv2dAttribute,
};

// ---------------------------------------------------------------------------
// Empty attribute structs -- tosa-rs uses `struct Foo {}` (braced) but
// retab call sites use unit-struct syntax (`FooAttribute`). Define
// local unit structs so existing code compiles unchanged.
// ---------------------------------------------------------------------------
macro_rules! unit_attr {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Debug, Clone)]
            pub struct $name;

            impl From<$name> for tosa_rs::$name {
                fn from(_: $name) -> Self { tosa_rs::$name {} }
            }
        )*
    };
}

unit_attr!(
    AddAttribute,
    SubAttribute,
    ReshapeAttribute,
    ConstAttribute,
    ConstShapeAttribute,
    TableAttribute,
    SigmoidAttribute,
    TanhAttribute,
    ExpAttribute,
    LogAttribute,
    NegateAttribute,
    AbsAttribute,
    CeilAttribute,
    FloorAttribute,
    ReciprocalAttribute,
    ErfAttribute,
    CosAttribute,
    SinAttribute,
    RsqrtAttribute,
    MatMulAttribute,
    PadAttribute,
    SliceAttribute,
    GatherAttribute,
    ScatterAttribute,
    SelectAttribute,
    CastAttribute,
    EqualAttribute,
    GreaterAttribute,
    GreaterEqualAttribute,
    BitwiseNotAttribute,
    BitwiseAndAttribute,
    BitwiseOrAttribute,
    BitwiseXorAttribute,
    LogicalLeftShiftAttribute,
    LogicalRightShiftAttribute,
    ArithmeticRightShiftAttribute,
    LogicalNotAttribute,
    LogicalAndAttribute,
    LogicalOrAttribute,
    LogicalXorAttribute,
    PowAttribute,
    IntDivAttribute,
);

// ---------------------------------------------------------------------------
// MulAttribute -- retab carries a `shift` field for U55/Vela compat;
// tosa-rs MulAttribute is empty (TOSA 1.0 puts shift in an input tensor).
// Keep the local struct so serialization code can read `a.shift`.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MulAttribute {
    pub shift: i32,
}

// ---------------------------------------------------------------------------
// WhileLoopAttribute -- retab uses cond_branch / body_branch;
// tosa-rs uses cond_graph / body_graph. Thin wrapper preserves field names.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct WhileLoopAttribute {
    pub cond_branch: String,
    pub body_branch: String,
}

// ---------------------------------------------------------------------------
// Structural types -- re-export those that are layout-compatible
// ---------------------------------------------------------------------------
pub use tosa_rs::{OpLocation, SoftwareVersion, TosaRegion, TosaShape};

// ---------------------------------------------------------------------------
// Version -- same fields, but retab default is (1,0,0,false) while
// tosa-rs default is (-1,-1,-1,true). Re-export the struct, override Default.
// ---------------------------------------------------------------------------
/// Version table -- wraps tosa_rs::Version with retab's default.
#[derive(Debug, Clone)]
pub struct Version {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
    pub draft: bool,
}

impl Default for Version {
    fn default() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
            draft: false,
        }
    }
}

impl From<Version> for tosa_rs::Version {
    fn from(v: Version) -> Self {
        tosa_rs::Version {
            major: v.major,
            minor: v.minor,
            patch: v.patch,
            draft: v.draft,
        }
    }
}

impl From<tosa_rs::Version> for Version {
    fn from(v: tosa_rs::Version) -> Self {
        Version {
            major: v.major,
            minor: v.minor,
            patch: v.patch,
            draft: v.draft,
        }
    }
}

// ---------------------------------------------------------------------------
// TosaTensor -- retab uses Vec<u8>/String where tosa-rs uses
// Option<Vec<u8>>/Option<String>, plus tosa-rs has offset/size fields.
// Keep a local struct with convenience constructors.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct TosaTensor {
    pub name: String,
    pub shape: Vec<i32>,
    pub dtype: DType,
    pub data: Vec<u8>,
    pub variable: bool,
    pub is_unranked: bool,
    pub variable_name: String,
}

impl TosaTensor {
    /// Empty output tensor (no data, not variable, not unranked).
    pub fn new(name: String, shape: Vec<i32>, dtype: DType) -> Self {
        Self {
            name,
            shape,
            dtype,
            data: Vec::new(),
            variable: false,
            is_unranked: false,
            variable_name: String::new(),
        }
    }

    /// Const tensor with inline data.
    pub fn with_data(name: String, shape: Vec<i32>, dtype: DType, data: Vec<u8>) -> Self {
        Self {
            data,
            ..Self::new(name, shape, dtype)
        }
    }
}

impl From<TosaTensor> for tosa_rs::TosaTensor {
    fn from(t: TosaTensor) -> Self {
        tosa_rs::TosaTensor {
            name: t.name,
            shape: t.shape,
            dtype: t.dtype,
            data: if t.data.is_empty() { None } else { Some(t.data) },
            variable: t.variable,
            is_unranked: t.is_unranked,
            variable_name: if t.variable_name.is_empty() {
                None
            } else {
                Some(t.variable_name)
            },
            offset: 0,
            size: 0,
        }
    }
}

impl From<tosa_rs::TosaTensor> for TosaTensor {
    fn from(t: tosa_rs::TosaTensor) -> Self {
        TosaTensor {
            name: t.name,
            shape: t.shape,
            dtype: t.dtype,
            data: t.data.unwrap_or_default(),
            variable: t.variable,
            is_unranked: t.is_unranked,
            variable_name: t.variable_name.unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------
// TosaOperator -- same fields but keep local so it uses our Attribute/Op.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct TosaOperator {
    pub op: Op,
    pub attribute: Attribute,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub location: Option<OpLocation>,
}

impl TosaOperator {
    /// Create an operator with no location metadata (the common case).
    pub fn new(op: Op, attribute: Attribute, inputs: Vec<String>, outputs: Vec<String>) -> Self {
        Self {
            op,
            attribute,
            inputs,
            outputs,
            location: None,
        }
    }
}

// ---------------------------------------------------------------------------
// TosaGraph -- same fields but uses our local Version.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct TosaGraph {
    pub version: Version,
    pub regions: Vec<TosaRegion>,
    pub software_version: Option<SoftwareVersion>,
}

// ---------------------------------------------------------------------------
// Attribute union -- local copy so we can include our MulAttribute (with
// shift) and WhileLoopAttribute (with cond_branch/body_branch).
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub enum Attribute {
    None,
    Conv2d(Conv2dAttribute),
    Conv3d(Conv3dAttribute),
    DepthwiseConv2d(DepthwiseConv2dAttribute),
    Rescale(RescaleAttribute),
    MaxPool2d(MaxPool2dAttribute),
    Transpose(TransposeAttribute),
    Reshape(ReshapeAttribute),
    Clamp(ClampAttribute),
    Const(ConstAttribute),
    ConstShape(ConstShapeAttribute),
    Add(AddAttribute),
    Sub(SubAttribute),
    Mul(MulAttribute),
    AvgPool2d(AvgPool2dAttribute),
    ReduceSum(ReduceSumAttribute),
    Table(TableAttribute),
    Sigmoid(SigmoidAttribute),
    Tanh(TanhAttribute),
    Exp(ExpAttribute),
    Log(LogAttribute),
    Negate(NegateAttribute),
    Abs(AbsAttribute),
    Ceil(CeilAttribute),
    Floor(FloorAttribute),
    Reciprocal(ReciprocalAttribute),
    Erf(ErfAttribute),
    Cos(CosAttribute),
    Sin(SinAttribute),
    Rsqrt(RsqrtAttribute),
    MatMul(MatMulAttribute),
    Concat(ConcatAttribute),
    Pad(PadAttribute),
    Slice(SliceAttribute),
    Gather(GatherAttribute),
    Scatter(ScatterAttribute),
    Resize(ResizeAttribute),
    Select(SelectAttribute),
    Cast(CastAttribute),
    Equal(EqualAttribute),
    Greater(GreaterAttribute),
    GreaterEqual(GreaterEqualAttribute),
    TransposeConv2d(TransposeConv2dAttribute),
    CondIf(CondIfAttribute),
    WhileLoop(WhileLoopAttribute),
    Maximum(MaximumAttribute),
    Minimum(MinimumAttribute),
    ReduceAny(ReduceAnyAttribute),
    ReduceMax(ReduceMaxAttribute),
    ReduceMin(ReduceMinAttribute),
    BitwiseNot(BitwiseNotAttribute),
    BitwiseAnd(BitwiseAndAttribute),
    BitwiseOr(BitwiseOrAttribute),
    BitwiseXor(BitwiseXorAttribute),
    LogicalLeftShift(LogicalLeftShiftAttribute),
    LogicalRightShift(LogicalRightShiftAttribute),
    ArithmeticRightShift(ArithmeticRightShiftAttribute),
    LogicalNot(LogicalNotAttribute),
    LogicalAnd(LogicalAndAttribute),
    LogicalOr(LogicalOrAttribute),
    LogicalXor(LogicalXorAttribute),
    Pow(PowAttribute),
    IntDiv(IntDivAttribute),
}

impl Attribute {
    /// Returns the union type byte used in the TOSA flatbuffer.
    /// 0 = NONE; indices match the union order in tosa.fbs.
    pub fn union_type(&self) -> u8 {
        match self {
            Attribute::None => 0,
            Attribute::Conv2d(_) => 3,
            Attribute::Conv3d(_) => 4,
            Attribute::DepthwiseConv2d(_) => 5,
            Attribute::Rescale(_) => 66,
            Attribute::MaxPool2d(_) => 8,
            Attribute::Transpose(_) => 61,
            Attribute::Reshape(_) => 57,
            Attribute::Clamp(_) => 11,
            Attribute::Const(_) => 67,
            Attribute::ConstShape(_) => 75,
            Attribute::Add(_) => 15,
            Attribute::Sub(_) => 30,
            Attribute::Mul(_) => 28,
            Attribute::AvgPool2d(_) => 2,
            Attribute::ReduceSum(_) => 54,
            Attribute::Table(_) => 31,
            Attribute::Sigmoid(_) => 13,
            Attribute::Tanh(_) => 14,
            Attribute::Exp(_) => 37,
            Attribute::Log(_) => 39,
            Attribute::Negate(_) => 41,
            Attribute::Abs(_) => 32,
            Attribute::Ceil(_) => 34,
            Attribute::Floor(_) => 38,
            Attribute::Reciprocal(_) => 42,
            Attribute::Erf(_) => 12,
            Attribute::Cos(_) => 36,
            Attribute::Sin(_) => 44,
            Attribute::Rsqrt(_) => 43,
            Attribute::MatMul(_) => 7,
            Attribute::Concat(_) => 55,
            Attribute::Pad(_) => 56,
            Attribute::Slice(_) => 59,
            Attribute::Gather(_) => 62,
            Attribute::Scatter(_) => 63,
            Attribute::Resize(_) => 64,
            Attribute::Select(_) => 45,
            Attribute::Cast(_) => 65,
            Attribute::Equal(_) => 46,
            Attribute::Greater(_) => 47,
            Attribute::GreaterEqual(_) => 48,
            Attribute::TransposeConv2d(_) => 10,
            Attribute::CondIf(_) => 70,
            Attribute::WhileLoop(_) => 71,
            Attribute::Maximum(_) => 26,
            Attribute::Minimum(_) => 27,
            Attribute::ReduceAny(_) => 50,
            Attribute::ReduceMax(_) => 51,
            Attribute::ReduceMin(_) => 52,
            Attribute::BitwiseNot(_) => 33,
            Attribute::BitwiseAnd(_) => 17,
            Attribute::BitwiseOr(_) => 18,
            Attribute::BitwiseXor(_) => 19,
            Attribute::LogicalLeftShift(_) => 22,
            Attribute::LogicalRightShift(_) => 23,
            Attribute::ArithmeticRightShift(_) => 16,
            Attribute::LogicalNot(_) => 40,
            Attribute::LogicalAnd(_) => 21,
            Attribute::LogicalOr(_) => 24,
            Attribute::LogicalXor(_) => 25,
            Attribute::Pow(_) => 29,
            Attribute::IntDiv(_) => 20,
        }
    }
}
