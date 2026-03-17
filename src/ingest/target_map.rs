use crate::graph::ir::OpTarget;

/// Map a Python target string (from `str(node.target)`) to an `OpTarget` variant.
///
/// ATen targets look like `"aten.conv2d.default"`, quantized decomposed targets
/// look like `"quantized_decomposed.quantize_per_tensor.default"`. We match on
/// substrings so we tolerate different overload suffixes.
pub fn parse_op_target(target_str: &str) -> OpTarget {
    // Quantized decomposed ops — check first to avoid false positives with aten
    if target_str.contains("quantized_decomposed.quantize_per_tensor") {
        return OpTarget::QuantizePerTensor;
    }
    if target_str.contains("quantized_decomposed.dequantize_per_tensor") {
        return OpTarget::DequantizePerTensor;
    }
    if target_str.contains("quantized_decomposed.dequantize_per_channel") {
        return OpTarget::DequantizePerChannel;
    }

    // ATen ops — order matters for ops whose names are substrings of others
    // (e.g. "conv2d" vs "conv_transpose2d")
    if target_str.contains("aten.conv_transpose2d") || target_str.contains("aten.convolution") && target_str.contains("transpose") {
        return OpTarget::ConvTranspose2d;
    }
    if target_str.contains("aten.conv2d") || target_str.contains("aten.convolution") {
        return OpTarget::Conv2d;
    }
    if target_str.contains("aten.relu") {
        return OpTarget::Relu;
    }
    if target_str.contains("aten.add.") {
        return OpTarget::Add;
    }
    if target_str.contains("aten.sub.") {
        return OpTarget::Sub;
    }
    if target_str.contains("aten.mul.") {
        return OpTarget::Mul;
    }
    if target_str.contains("aten.div.") {
        return OpTarget::Div;
    }
    if target_str.contains("aten.permute") {
        return OpTarget::Permute;
    }
    if target_str.contains("aten.reshape") {
        return OpTarget::Reshape;
    }
    if target_str.contains("aten.full.") {
        return OpTarget::Full;
    }
    if target_str.contains("aten.avg_pool2d") {
        return OpTarget::AvgPool2d;
    }
    if target_str.contains("aten.max_pool2d") {
        return OpTarget::MaxPool2d;
    }
    if target_str.contains("aten.sum.") {
        return OpTarget::Sum;
    }
    if target_str.contains("aten.rsqrt") {
        return OpTarget::Rsqrt;
    }
    if target_str.contains("aten.layer_norm") {
        return OpTarget::LayerNorm;
    }
    if target_str.contains("aten.batch_norm") || target_str.contains("aten._native_batch_norm") {
        return OpTarget::BatchNorm;
    }
    if target_str.contains("aten.linear") {
        return OpTarget::Linear;
    }
    if target_str.contains("aten.matmul") || target_str.contains("aten.mm.") {
        return OpTarget::Matmul;
    }
    if target_str.contains("aten.bmm") {
        return OpTarget::Bmm;
    }
    if target_str.contains("aten.addmm") {
        return OpTarget::Addmm;
    }
    if target_str.contains("aten.softmax") || target_str.contains("aten._softmax") {
        return OpTarget::Softmax;
    }
    if target_str.contains("aten.gelu") {
        return OpTarget::Gelu;
    }
    if target_str.contains("aten.sigmoid") {
        return OpTarget::Sigmoid;
    }
    if target_str.contains("aten.tanh") {
        return OpTarget::Tanh;
    }
    if target_str.contains("aten.clamp") {
        return OpTarget::Clamp;
    }
    if target_str.contains("aten.hardtanh") {
        return OpTarget::Hardtanh;
    }
    if target_str.contains("aten.mean") {
        return OpTarget::Mean;
    }
    if target_str.contains("aten.cat") {
        return OpTarget::Cat;
    }
    if target_str.contains("aten.slice") {
        return OpTarget::Slice;
    }
    if target_str.contains("aten.select") {
        return OpTarget::Select;
    }
    if target_str.contains("aten.unsqueeze") {
        return OpTarget::Unsqueeze;
    }
    if target_str.contains("aten.squeeze") {
        return OpTarget::Squeeze;
    }
    if target_str.contains("aten.flatten") {
        return OpTarget::Flatten;
    }
    if target_str.contains("aten.pad") {
        return OpTarget::Pad;
    }
    if target_str.contains("aten.embedding") {
        return OpTarget::Embedding;
    }
    if target_str.contains("aten.exp.") || target_str.contains("aten.exp2") {
        return OpTarget::Exp;
    }
    if target_str.contains("aten.log.") || target_str.contains("aten.log2") {
        return OpTarget::Log;
    }
    if target_str.contains("aten.neg.") {
        return OpTarget::Neg;
    }
    if target_str.contains("aten.reciprocal") {
        return OpTarget::Reciprocal;
    }
    if target_str.contains("aten.abs.") {
        return OpTarget::Abs;
    }
    if target_str.contains("aten.pow.") {
        return OpTarget::Pow;
    }
    if target_str.contains("aten.floor.") {
        return OpTarget::Floor;
    }
    if target_str.contains("aten.ceil.") {
        return OpTarget::Ceil;
    }
    if target_str.contains("aten.expand") {
        return OpTarget::Expand;
    }
    if target_str.contains("aten.contiguous") {
        return OpTarget::Contiguous;
    }
    if target_str.contains("aten.view") || target_str.contains("aten._unsafe_view") {
        return OpTarget::View;
    }
    if target_str.contains("aten.to.") || target_str.contains("aten._to_copy") {
        return OpTarget::To;
    }
    if target_str.contains("aten.var.") {
        return OpTarget::Var;
    }
    if target_str.contains("aten.full_like") {
        return OpTarget::FullLike;
    }

    OpTarget::Other(target_str.to_string())
}
