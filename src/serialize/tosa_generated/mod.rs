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
// Attribute union -- re-exported directly from tosa-rs.
// ---------------------------------------------------------------------------
pub use tosa_rs::Attribute;
