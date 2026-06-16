use slotmap::new_key_type;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// NodeId — slotmap key
// ---------------------------------------------------------------------------
new_key_type! {
    pub struct NodeId;
}

// ---------------------------------------------------------------------------
// OpType
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    Placeholder,
    CallFunction,
    Output,
}

// ---------------------------------------------------------------------------
// OpTarget — covers ATen inputs, quantized ops, TOSA outputs, and an escape hatch
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpTarget {
    // ATen operators
    Conv2d,
    Relu,
    Add,
    Sub,
    Mul,
    Permute,
    Reshape,
    Full,
    AvgPool2d,
    MaxPool2d,
    Sum,
    Rsqrt,
    LayerNorm,
    BatchNorm,
    Linear,
    Matmul,
    Softmax,
    Gelu,
    Sigmoid,
    Tanh,
    Clamp,
    Mean,
    Cat,
    Slice,
    Unsqueeze,
    Squeeze,
    Flatten,
    Hardtanh,
    Var,
    FullLike,
    Copy,
    Eye,
    Pad,
    ConvTranspose2d,
    DepthwiseConv2d,
    Addmm,
    Bmm,
    Embedding,
    Exp,
    Log,
    Neg,
    Reciprocal,
    Abs,
    Pow,
    Div,
    Floor,
    Ceil,
    GroupNorm,
    Amax,
    Select,
    SelectCopy,
    IndexSelect,
    Where,
    FloorDivide,
    Arange,
    Expand,
    Contiguous,
    View,
    To,
    LeakyRelu,
    Sqrt,
    Log1p,
    Expm1,
    Remainder,
    Sign,
    Round,

    // Quantized
    QuantizePerTensor,
    DequantizePerTensor,
    DequantizePerChannel,

    // TOSA operators
    TosaConv2d,
    TosaDepthwiseConv2d,
    TosaTransposeConv2d,
    TosaAdd,
    TosaMul,
    TosaSub,
    TosaRescale,
    TosaResize,
    TosaReshape,
    TosaTranspose,
    TosaAvgPool2d,
    TosaMaxPool2d,
    TosaClamp,
    TosaReduceSum,
    TosaTable,
    TosaConcat,
    TosaSlice,
    TosaPad,
    TosaFullyConnected,
    TosaConst,
    TosaConv3d,
    TosaMatmul,
    TosaSigmoid,
    TosaTanh,
    TosaExp,
    TosaLog,
    TosaNeg,
    TosaAbs,
    TosaCeil,
    TosaFloor,
    TosaReciprocal,
    TosaReduceMax,
    TosaGather,
    TosaScatter,
    TosaSelect,
    TosaEqual,
    TosaGreater,
    TosaGreaterEqual,
    TosaErf,
    TosaCos,
    TosaSin,
    TosaRsqrt,
    TosaCast,
    TosaNegate,
    TosaReduceMin,
    TosaReduceAny,
    TosaMaximum,
    TosaMinimum,

    // Aliases for control flow (used by visitors)
    CondIf,
    WhileLoop,

    // ATen ops that may appear in post-pass graphs
    Amin,
    Any,

    // Bitwise ops
    BitwiseNot,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseLeftShift,
    BitwiseRightShift,

    // Logical ops
    LogicalNot,
    LogicalAnd,
    LogicalOr,
    LogicalXor,

    // Comparison ops (ATen)
    Gt,
    Ge,
    Lt,
    Le,
    Eq,

    // Trig / math ops
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Elu,
    Hardswish,
    Silu,

    // Additional ops
    AdaptiveAvgPool2d,
    UpsampleNearest2d,
    UpsampleBilinear2d,
    Conv1d,
    ScalarTensor,
    Zeros,
    Repeat,
    Cumsum,
    Erfinv,
    UnfoldCopy,
    SplitWithSizesCopy,
    IndexTensor,

    // Escape hatch
    Other(String),
}

// ---------------------------------------------------------------------------
// DType
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Float8E4M3,
    Float8E5M2,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
    Int4,
    Int48,
}

impl DType {
    pub fn from_torch_str(dtype_str: &str) -> Self {
        let dtype_str = dtype_str.to_ascii_lowercase();
        match dtype_str.as_str() {
            s if s.contains("float8_e4m3") || s.contains("fp8e4m3") => DType::Float8E4M3,
            s if s.contains("float8_e5m2") || s.contains("fp8e5m2") => DType::Float8E5M2,
            s if s.contains("float32") => DType::Float32,
            s if s.contains("bfloat16") => DType::BFloat16,
            s if s.contains("float16") => DType::Float16,
            s if s.contains("int8") => DType::Int8,
            s if s.contains("int16") => DType::Int16,
            s if s.contains("int32") => DType::Int32,
            s if s.contains("int64") => DType::Int64,
            s if s.contains("bool") => DType::Bool,
            _ => DType::Float32,
        }
    }

    pub fn torch_str_is_float_like(dtype_str: &str) -> bool {
        let dtype_str = dtype_str.to_ascii_lowercase();
        dtype_str.contains("float") || dtype_str.contains("fp") || dtype_str.contains("bf")
    }

    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::Int4 | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int48 | DType::Int64
        )
    }

    pub fn is_float(self) -> bool {
        matches!(
            self,
            DType::Float32
                | DType::Float16
                | DType::BFloat16
                | DType::Float8E4M3
                | DType::Float8E5M2
        )
    }

    pub fn is_narrow_float(self) -> bool {
        matches!(
            self,
            DType::Float16 | DType::BFloat16 | DType::Float8E4M3 | DType::Float8E5M2
        )
    }

    pub fn is_quantized_integer(self) -> bool {
        matches!(self, DType::Int8 | DType::Int16 | DType::Int32)
    }

    pub fn storage_size_bytes(self) -> usize {
        match self {
            DType::Bool | DType::Int4 | DType::Int8 | DType::Float8E4M3 | DType::Float8E5M2 => 1,
            DType::Int16 | DType::Float16 | DType::BFloat16 => 2,
            DType::Int32 | DType::Float32 => 4,
            DType::Int48 => 6,
            DType::Int64 => 8,
        }
    }

    pub fn quantized_integer_bounds(self) -> Option<(i64, i64)> {
        match self {
            DType::Int8 => Some((-128, 127)),
            DType::Int16 => Some((-32768, 32767)),
            DType::Int32 => Some((i32::MIN as i64, i32::MAX as i64)),
            _ => None,
        }
    }

    pub fn quantized_integer_bounds_from_torch_str(dtype_str: &str) -> Option<(i64, i64)> {
        let dtype_str = dtype_str.to_ascii_lowercase();
        if !dtype_str.contains("int") {
            return None;
        }
        DType::from_torch_str(&dtype_str)
            .quantized_integer_bounds()
            .or(Some((i32::MIN as i64, i32::MAX as i64)))
    }

    pub fn torch_str(self) -> &'static str {
        match self {
            DType::Float32 => "torch.float32",
            DType::Float16 => "torch.float16",
            DType::BFloat16 => "torch.bfloat16",
            DType::Float8E4M3 => "torch.float8_e4m3fn",
            DType::Float8E5M2 => "torch.float8_e5m2",
            DType::Int8 => "torch.int8",
            DType::Int16 => "torch.int16",
            DType::Int32 => "torch.int32",
            DType::Int64 => "torch.int64",
            DType::Bool => "torch.bool",
            DType::Int4 => "torch.int8",
            DType::Int48 => "torch.int64",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn dtype_helpers_preserve_torch_string_domains() {
        assert!(DType::torch_str_is_float_like("torch.float16"));
        assert!(DType::torch_str_is_float_like("torch.bf16"));
        assert_eq!(
            DType::from_torch_str("torch.float8_e4m3fn"),
            DType::Float8E4M3
        );
        assert_eq!(
            DType::from_torch_str("torch.float8_e5m2"),
            DType::Float8E5M2
        );
        assert!(DType::Float8E4M3.is_float());
        assert!(DType::Float8E5M2.is_narrow_float());
        assert_eq!(DType::Float8E4M3.storage_size_bytes(), 1);
        assert_eq!(DType::BFloat16.storage_size_bytes(), 2);
        assert_eq!(DType::Int48.storage_size_bytes(), 6);
        assert_eq!(DType::Float8E4M3.torch_str(), "torch.float8_e4m3fn");
        assert!(!DType::torch_str_is_float_like("torch.int8"));

        assert_eq!(DType::Int8.quantized_integer_bounds(), Some((-128, 127)));
        assert_eq!(
            DType::quantized_integer_bounds_from_torch_str("torch.int16"),
            Some((-32768, 32767))
        );
        assert_eq!(
            DType::quantized_integer_bounds_from_torch_str("torch.int4"),
            Some((i32::MIN as i64, i32::MAX as i64))
        );
        assert_eq!(
            DType::quantized_integer_bounds_from_torch_str("torch.float32"),
            None
        );
    }
}

// ---------------------------------------------------------------------------
// TensorInfo — lightweight tensor metadata (Copy)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorInfo {
    pub shape: [i64; 6],
    pub shape_len: u8,
    pub dtype: DType,
}

impl TensorInfo {
    pub fn new(shape: &[i64], dtype: DType) -> Self {
        let mut arr = [0i64; 6];
        let len = shape.len().min(6);
        arr[..len].copy_from_slice(&shape[..len]);
        Self {
            shape: arr,
            shape_len: len as u8,
            dtype,
        }
    }

    pub fn shape_slice(&self) -> &[i64] {
        &self.shape[..self.shape_len as usize]
    }
}

// ---------------------------------------------------------------------------
// QuantArgs — quantization parameters (Copy)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantArgs {
    pub scale: f64,
    pub zero_point: i64,
    pub dtype: DType,
    pub qmin: i64,
    pub qmax: i64,
}

// ---------------------------------------------------------------------------
// Arg — operand for a node
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    Node(NodeId),
    NodeList(SmallVec<[NodeId; 4]>),
    Int(i64),
    Float(f64),
    Bool(bool),
    IntList(SmallVec<[i64; 6]>),
    FloatList(SmallVec<[f64; 4]>),
    Dtype(DType),
    None,
    String(String),
}

impl Arg {
    /// Return all NodeIds referenced by this argument.
    pub fn node_refs(&self) -> SmallVec<[NodeId; 4]> {
        match self {
            Arg::Node(id) => smallvec::smallvec![*id],
            Arg::NodeList(ids) => ids.clone(),
            _ => SmallVec::new(),
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Arg::Float(value) => *value,
            Arg::Int(value) => *value as f64,
            Arg::Bool(value) => i32::from(*value) as f64,
            _ => 0.0,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Arg::Int(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Arg::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_node(&self) -> Option<NodeId> {
        match self {
            Arg::Node(id) => Some(*id),
            _ => None,
        }
    }

    pub fn as_int_list(&self) -> Option<&[i64]> {
        match self {
            Arg::IntList(values) => Some(values.as_slice()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// NodeMeta — per-node metadata
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
pub struct NodeMeta {
    pub val: TensorInfo,
    pub input_qparams: SmallVec<[(u32, QuantArgs); 2]>,
    pub output_qparams: SmallVec<[(u32, QuantArgs); 1]>,
    pub tosa_dim_order: Option<SmallVec<[u8; 6]>>,
    pub tosa_special_dtype: Option<DType>,
    pub arm_control_flow_parent_output_user: bool,
    pub preserve_scalar_shape: bool,
}

impl NodeMeta {
    pub fn new(val: TensorInfo) -> Self {
        Self {
            val,
            input_qparams: SmallVec::new(),
            output_qparams: SmallVec::new(),
            tosa_dim_order: None,
            tosa_special_dtype: None,
            arm_control_flow_parent_output_user: false,
            preserve_scalar_shape: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Node — a single operation in the graph
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Node {
    pub op: OpType,
    pub target: OpTarget,
    pub args: SmallVec<[Arg; 4]>,
    pub kwargs: Vec<(String, Arg)>,
    pub meta: NodeMeta,
    pub users: SmallVec<[NodeId; 4]>,
    pub name: String,
}
