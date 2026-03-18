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

// ---------------------------------------------------------------------------
// TensorInfo — lightweight tensor metadata (Copy)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorInfo {
    pub shape: [i64; 4],
    pub shape_len: u8,
    pub dtype: DType,
}

impl TensorInfo {
    pub fn new(shape: &[i64], dtype: DType) -> Self {
        let mut arr = [0i64; 4];
        let len = shape.len().min(4);
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
    OptionalNode(Option<NodeId>),
    OptionalInt(Option<i64>),
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
            Arg::OptionalNode(Some(id)) => smallvec::smallvec![*id],
            _ => SmallVec::new(),
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
}

impl NodeMeta {
    pub fn new(val: TensorInfo) -> Self {
        Self {
            val,
            input_qparams: SmallVec::new(),
            output_qparams: SmallVec::new(),
            tosa_dim_order: None,
            tosa_special_dtype: None,
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
