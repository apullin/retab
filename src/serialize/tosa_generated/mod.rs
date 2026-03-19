// Manual Rust definitions for the subset of TOSA 1.0 flatbuffer types
// we need for Phase 1 serialization. These mirror the schema at
// tosa-tools/serialization/schema/tosa.fbs.
//
// When flatc codegen is available, this file can be replaced by the
// generated output.

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// TOSA data type (matches DType:uint32 in the schema).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DType {
    Unknown = 0,
    Bool = 1,
    Int4 = 2,
    Int8 = 3,
    Int16 = 4,
    Int32 = 5,
    Int48 = 6,
    Fp32 = 7,
    Fp16 = 8,
    Bf16 = 9,
    Shape = 10,
    Fp8E4M3 = 11,
    Fp8E5M2 = 12,
    Fp6E2M3 = 13,
    Fp6E3M2 = 14,
    Fp4E2M1 = 15,
    Fp8Ue8M0 = 16,
    Int64 = 17,
    Mxint8 = 18,
}

/// TOSA operator enum (matches Op:uint32 in the schema).
/// Only the operators needed for Phase 1 are given named variants;
/// the rest can be represented via their raw u32 value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Op {
    Unknown = 0,
    ArgMax = 1,
    AvgPool2d = 2,
    Conv2d = 3,
    Conv3d = 4,
    DepthwiseConv2d = 5,
    Fft2d = 6,
    Matmul = 7,
    MaxPool2d = 8,
    Rfft2d = 9,
    TransposeConv2d = 10,
    Clamp = 11,
    Erf = 12,
    Sigmoid = 13,
    Tanh = 14,
    Add = 15,
    ArithmeticRightShift = 16,
    BitwiseAnd = 17,
    BitwiseOr = 18,
    BitwiseXor = 19,
    IntDiv = 20,
    LogicalAnd = 21,
    LogicalLeftShift = 22,
    LogicalRightShift = 23,
    LogicalOr = 24,
    LogicalXor = 25,
    Maximum = 26,
    Minimum = 27,
    Mul = 28,
    Pow = 29,
    Sub = 30,
    Table = 31,
    Abs = 32,
    BitwiseNot = 33,
    Ceil = 34,
    Clz = 35,
    Cos = 36,
    Exp = 37,
    Floor = 38,
    Log = 39,
    LogicalNot = 40,
    Negate = 41,
    Reciprocal = 42,
    Rsqrt = 43,
    Sin = 44,
    Select = 45,
    Equal = 46,
    Greater = 47,
    GreaterEqual = 48,
    ReduceAll = 49,
    ReduceAny = 50,
    ReduceMax = 51,
    ReduceMin = 52,
    ReduceProduct = 53,
    ReduceSum = 54,
    Concat = 55,
    Pad = 56,
    Reshape = 57,
    Reverse = 58,
    Slice = 59,
    Tile = 60,
    Transpose = 61,
    Gather = 62,
    Scatter = 63,
    Resize = 64,
    Cast = 65,
    Rescale = 66,
    Const = 67,
    Identity = 68,
    Custom = 69,
    CondIf = 70,
    WhileLoop = 71,
    Variable = 72,
    VariableWrite = 73,
    VariableRead = 74,
    ConstShape = 75,
}

/// Rounding mode used by RESCALE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RoundingMode {
    Unknown = 0,
    SingleRound = 1,
    InexactRound = 2,
    DoubleRound = 3,
}

/// NaN propagation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum NanPropagationMode {
    Unknown = 0,
    Propagate = 1,
    Ignore = 2,
}

// ---------------------------------------------------------------------------
// Attribute types (tables in the schema)
// ---------------------------------------------------------------------------

/// Conv2dAttribute — pad, stride, dilation, local_bound, acc_type.
#[derive(Debug, Clone)]
pub struct Conv2dAttribute {
    pub pad: Vec<i32>,
    pub stride: Vec<i32>,
    pub dilation: Vec<i32>,
    pub local_bound: bool,
    pub acc_type: DType,
}

/// DepthwiseConv2dAttribute — same fields as Conv2d.
#[derive(Debug, Clone)]
pub struct DepthwiseConv2dAttribute {
    pub pad: Vec<i32>,
    pub stride: Vec<i32>,
    pub dilation: Vec<i32>,
    pub local_bound: bool,
    pub acc_type: DType,
}

/// RescaleAttribute.
#[derive(Debug, Clone)]
pub struct RescaleAttribute {
    pub scale32: bool,
    pub rounding_mode: RoundingMode,
    pub per_channel: bool,
    pub input_unsigned: bool,
    pub output_unsigned: bool,
}

/// MaxPool2dAttribute.
#[derive(Debug, Clone)]
pub struct MaxPool2dAttribute {
    pub kernel: Vec<i32>,
    pub stride: Vec<i32>,
    pub pad: Vec<i32>,
    pub nan_mode: NanPropagationMode,
}

/// TransposeAttribute — permutation vector.
#[derive(Debug, Clone)]
pub struct TransposeAttribute {
    pub perms: Vec<i32>,
}

/// ReshapeAttribute — empty in the schema.
#[derive(Debug, Clone)]
pub struct ReshapeAttribute;

/// ClampAttribute — min/max stored as raw bytes.
#[derive(Debug, Clone)]
pub struct ClampAttribute {
    pub min_val: Vec<u8>,
    pub max_val: Vec<u8>,
    pub nan_mode: NanPropagationMode,
}

/// ConstAttribute — empty; value is in the output TosaTensor.
#[derive(Debug, Clone)]
pub struct ConstAttribute;

/// ConstShapeAttribute — empty; value is in the output TosaTensor.
#[derive(Debug, Clone)]
pub struct ConstShapeAttribute;

/// AddAttribute — empty.
#[derive(Debug, Clone)]
pub struct AddAttribute;

/// SubAttribute — empty.
#[derive(Debug, Clone)]
pub struct SubAttribute;

/// MulAttribute — empty in TOSA 1.0 (shift is a third input tensor).
#[derive(Debug, Clone)]
pub struct MulAttribute;

/// AvgPool2dAttribute — pooling parameters.
/// In TOSA 1.0, input_zp and output_zp are separate input tensors,
/// not part of the attribute table.
#[derive(Debug, Clone)]
pub struct AvgPool2dAttribute {
    pub kernel: Vec<i32>,
    pub stride: Vec<i32>,
    pub pad: Vec<i32>,
    pub acc_type: DType,
}

/// ReduceSumAttribute — reduction axis.
#[derive(Debug, Clone)]
pub struct ReduceSumAttribute {
    pub axis: i32,
}

/// TableAttribute — empty (table data is an input tensor).
#[derive(Debug, Clone)]
pub struct TableAttribute;

/// SigmoidAttribute — empty.
#[derive(Debug, Clone)]
pub struct SigmoidAttribute;

/// TanhAttribute — empty.
#[derive(Debug, Clone)]
pub struct TanhAttribute;

/// ExpAttribute — empty.
#[derive(Debug, Clone)]
pub struct ExpAttribute;

/// LogAttribute — empty.
#[derive(Debug, Clone)]
pub struct LogAttribute;

/// NegateAttribute — empty.
#[derive(Debug, Clone)]
pub struct NegateAttribute;

/// AbsAttribute — empty.
#[derive(Debug, Clone)]
pub struct AbsAttribute;

/// CeilAttribute — empty.
#[derive(Debug, Clone)]
pub struct CeilAttribute;

/// FloorAttribute — empty.
#[derive(Debug, Clone)]
pub struct FloorAttribute;

/// ReciprocalAttribute — empty.
#[derive(Debug, Clone)]
pub struct ReciprocalAttribute;

/// ErfAttribute — empty.
#[derive(Debug, Clone)]
pub struct ErfAttribute;

/// CosAttribute — empty.
#[derive(Debug, Clone)]
pub struct CosAttribute;

/// SinAttribute — empty.
#[derive(Debug, Clone)]
pub struct SinAttribute;

/// RsqrtAttribute — empty.
#[derive(Debug, Clone)]
pub struct RsqrtAttribute;

/// MatMulAttribute — empty.
#[derive(Debug, Clone)]
pub struct MatMulAttribute;

/// ConcatAttribute — axis for concatenation.
#[derive(Debug, Clone)]
pub struct ConcatAttribute {
    pub axis: i32,
}

/// PadAttribute — empty (padding comes from input shapes).
#[derive(Debug, Clone)]
pub struct PadAttribute;

/// SliceAttribute — empty (start/size come from input shapes).
#[derive(Debug, Clone)]
pub struct SliceAttribute;

/// GatherAttribute — empty.
#[derive(Debug, Clone)]
pub struct GatherAttribute;

/// SelectAttribute — empty.
#[derive(Debug, Clone)]
pub struct SelectAttribute;

/// CastAttribute — empty.
#[derive(Debug, Clone)]
pub struct CastAttribute;

/// EqualAttribute — empty.
#[derive(Debug, Clone)]
pub struct EqualAttribute;

/// GreaterAttribute — empty.
#[derive(Debug, Clone)]
pub struct GreaterAttribute;

/// GreaterEqualAttribute — empty.
#[derive(Debug, Clone)]
pub struct GreaterEqualAttribute;

/// TransposeConv2dAttribute — deconvolution parameters.
#[derive(Debug, Clone)]
pub struct TransposeConv2dAttribute {
    pub out_pad: Vec<i32>,
    pub stride: Vec<i32>,
    pub local_bound: bool,
    pub acc_type: DType,
}

// ---------------------------------------------------------------------------
// Attribute union
// ---------------------------------------------------------------------------

/// Union of all operator attributes we support.
#[derive(Debug, Clone)]
pub enum Attribute {
    None,
    Conv2d(Conv2dAttribute),
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
    Select(SelectAttribute),
    Cast(CastAttribute),
    Equal(EqualAttribute),
    Greater(GreaterAttribute),
    GreaterEqual(GreaterEqualAttribute),
    TransposeConv2d(TransposeConv2dAttribute),
}

impl Attribute {
    /// Returns the union type byte used in the TOSA flatbuffer.
    /// 0 = NONE; indices match the union order in tosa.fbs.
    pub fn union_type(&self) -> u8 {
        match self {
            Attribute::None => 0,
            Attribute::Conv2d(_) => 3,          // Conv2dAttribute
            Attribute::DepthwiseConv2d(_) => 5, // DepthwiseConv2dAttribute
            Attribute::Rescale(_) => 66,        // RescaleAttribute
            Attribute::MaxPool2d(_) => 8,       // MaxPool2dAttribute
            Attribute::Transpose(_) => 61,      // TransposeAttribute
            Attribute::Reshape(_) => 57,        // ReshapeAttribute
            Attribute::Clamp(_) => 11,          // ClampAttribute
            Attribute::Const(_) => 67,          // ConstAttribute
            Attribute::ConstShape(_) => 75,     // ConstShapeAttribute
            Attribute::Add(_) => 15,            // AddAttribute
            Attribute::Sub(_) => 30,            // SubAttribute
            Attribute::Mul(_) => 28,            // MulAttribute
            Attribute::AvgPool2d(_) => 2,       // AvgPool2dAttribute
            Attribute::ReduceSum(_) => 54,      // ReduceSumAttribute
            Attribute::Table(_) => 31,          // TableAttribute
            Attribute::Sigmoid(_) => 13,        // SigmoidAttribute
            Attribute::Tanh(_) => 14,           // TanhAttribute
            Attribute::Exp(_) => 37,            // ExpAttribute
            Attribute::Log(_) => 39,            // LogAttribute
            Attribute::Negate(_) => 41,         // NegateAttribute
            Attribute::Abs(_) => 32,            // AbsAttribute
            Attribute::Ceil(_) => 34,           // CeilAttribute
            Attribute::Floor(_) => 38,          // FloorAttribute
            Attribute::Reciprocal(_) => 42,     // ReciprocalAttribute
            Attribute::Erf(_) => 12,            // ErfAttribute
            Attribute::Cos(_) => 36,            // CosAttribute
            Attribute::Sin(_) => 44,            // SinAttribute
            Attribute::Rsqrt(_) => 43,          // RsqrtAttribute
            Attribute::MatMul(_) => 7,          // MatMulAttribute
            Attribute::Concat(_) => 55,         // ConcatAttribute
            Attribute::Pad(_) => 56,            // PadAttribute
            Attribute::Slice(_) => 59,          // SliceAttribute
            Attribute::Gather(_) => 62,         // GatherAttribute
            Attribute::Select(_) => 45,         // SelectAttribute
            Attribute::Cast(_) => 65,           // CastAttribute
            Attribute::Equal(_) => 46,          // EqualAttribute
            Attribute::Greater(_) => 47,        // GreaterAttribute
            Attribute::GreaterEqual(_) => 48,   // GreaterEqualAttribute
            Attribute::TransposeConv2d(_) => 10, // TransposeConv2dAttribute
        }
    }
}

// ---------------------------------------------------------------------------
// Structural tables
// ---------------------------------------------------------------------------

/// Version table.
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

/// SoftwareVersion table.
#[derive(Debug, Clone)]
pub struct SoftwareVersion {
    pub major: i32,
    pub minor: i32,
    pub micro: i32,
    pub modifier: String,
}

impl Default for SoftwareVersion {
    fn default() -> Self {
        Self {
            major: 0,
            minor: 1,
            micro: 0,
            modifier: String::from("arm-tosa-engine"),
        }
    }
}

/// TosaTensor — a named tensor with shape, dtype, and optional inline data.
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

/// TosaShape — named shape value.
#[derive(Debug, Clone)]
pub struct TosaShape {
    pub name: String,
    pub rank: u32,
    pub data: Vec<u8>,
}

/// OpLocation — opaque source location.
#[derive(Debug, Clone)]
pub struct OpLocation {
    pub text: String,
}

/// TosaOperator — one op node in a basic block.
#[derive(Debug, Clone)]
pub struct TosaOperator {
    pub op: Op,
    pub attribute: Attribute,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub location: Option<OpLocation>,
}

/// TosaBasicBlock — contains operators and their tensors.
#[derive(Debug, Clone)]
pub struct TosaBasicBlock {
    pub name: String,
    pub operators: Vec<TosaOperator>,
    pub tensors: Vec<TosaTensor>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub shapes: Vec<TosaShape>,
}

/// TosaRegion — contains basic blocks.
#[derive(Debug, Clone)]
pub struct TosaRegion {
    pub name: String,
    pub blocks: Vec<TosaBasicBlock>,
}

/// TosaGraph — the root flatbuffer table.
#[derive(Debug, Clone)]
pub struct TosaGraph {
    pub version: Version,
    pub regions: Vec<TosaRegion>,
    pub software_version: Option<SoftwareVersion>,
}
