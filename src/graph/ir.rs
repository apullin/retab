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
        match dtype_str {
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

    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::Int4 | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int48 | DType::Int64
        )
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
