
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
    ArgMaxAttribute, AvgPool2dAttribute, ClampAttribute, ConcatAttribute, CondIfAttribute,
    Conv2dAttribute, Conv3dAttribute, DepthwiseConv2dAttribute, MaxPool2dAttribute,
    MaximumAttribute, MinimumAttribute, MulAttribute, ReduceAnyAttribute, ReduceMaxAttribute,
    ReduceMinAttribute, ReduceSumAttribute, RescaleAttribute, ResizeAttribute, ReverseAttribute,
    TransposeAttribute, TransposeConv2dAttribute, WhileLoopAttribute,
};

// ---------------------------------------------------------------------------
// Empty attribute structs -- re-exported from tosa-rs (braced struct syntax).
// ---------------------------------------------------------------------------
#[allow(unused_imports)]
pub use tosa_rs::{
    AbsAttribute, AddAttribute, ArithmeticRightShiftAttribute, BitwiseAndAttribute,
    BitwiseNotAttribute, BitwiseOrAttribute, BitwiseXorAttribute, CastAttribute, CeilAttribute,
    ConstAttribute, ConstShapeAttribute, CosAttribute, EqualAttribute, ErfAttribute, ExpAttribute,
    FloorAttribute, GatherAttribute, GreaterAttribute, GreaterEqualAttribute, IntDivAttribute,
    LogAttribute, LogicalAndAttribute, LogicalLeftShiftAttribute, LogicalNotAttribute,
    LogicalOrAttribute, LogicalRightShiftAttribute, LogicalXorAttribute, MatMulAttribute,
    NegateAttribute, PadAttribute, PowAttribute, ReciprocalAttribute, ReshapeAttribute,
    RsqrtAttribute, ScatterAttribute, SelectAttribute, SigmoidAttribute, SinAttribute,
    SliceAttribute, SubAttribute, TableAttribute, TanhAttribute,
};

// ---------------------------------------------------------------------------
// Structural types -- re-export those that are layout-compatible
// ---------------------------------------------------------------------------
pub use tosa_rs::TosaShape;

// ---------------------------------------------------------------------------
// TosaTensor -- re-exported directly from tosa-rs.
// ---------------------------------------------------------------------------
pub use tosa_rs::TosaTensor;

// ---------------------------------------------------------------------------
// TosaOperator -- same fields but keep local so it uses our Attribute/Op.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct TosaOperator {
    pub op: Op,
    pub attribute: Attribute,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl TosaOperator {
    pub fn new(op: Op, attribute: Attribute, inputs: Vec<String>, outputs: Vec<String>) -> Self {
        Self {
            op,
            attribute,
            inputs,
            outputs,
        }
    }
}

// ---------------------------------------------------------------------------
// Attribute union -- re-exported directly from tosa-rs.
// ---------------------------------------------------------------------------
pub use tosa_rs::Attribute;
