//! Shape rules for the TOSA ops the explicit-mode rewrites create (APU-135
//! P2, design S0): ports of rc4's fake kernels in
//! `backends/arm/tosa/dialect/ops/` for CONV2D, DEPTHWISE_CONV2D,
//! TRANSPOSE_CONV2D, CONV3D, MAX_POOL2D, AVG_POOL2D and RESIZE. RESCALE's rule
//! is in `shape.rs`.
//!
//! Each rule gives the kernel's output shape and dtype and makes the kernel's
//! shape checks: ranks, argument lengths, RESIZE's parameter validation, and
//! a negative output extent, which the kernel's `torch.empty` rejects. Where
//! rc4 indexes an argument without checking it (CONV2D's stride, pad and
//! dilation; the 2-D convolutions' ranks), the rule requires what the schema
//! states, and a stride a kernel divides by must be positive. Checks that only
//! judge lowering legality are not ported, as for RESCALE: the TOSA spec's
//! profile and extension support, a convolution's weight and bias dtypes, a
//! pool's input dtype, AVG_POOL2D's zero-point and accumulator dtypes. A pass
//! that creates these nodes checks what it needs from RETAB's representation
//! (design §2.5, critique B10).
//!
//! **Input dtype.** The convolution and RESIZE kernels pick their output dtype
//! from their input's, which RETAB's IR keeps in qparams for a quantized
//! tensor (design F2). The TOSA ops inherit or copy the source node's qparams
//! (design §2.7, §3), so a rule reads the node's input qparam 0 when it has
//! one and the input's val dtype otherwise, as `_is_int16_activation_conv`
//! does (`rewrite_conv_pass.py:408-412`; [`dtype::operand_dtype`]).
//!
//! **Arguments.** rc4's argument order, including AVG_POOL2D's zero points
//! before the kernel (`avg_pool2d.py:92-104`, critique B1). RESIZE's scale,
//! offset and border are `CONST_SHAPE` nodes in rc4's graphs
//! (`rewrite_upsample.py:205-230`) or literal lists; a rule reads a
//! `CONST_SHAPE` through the operand lookup's shape values
//! (`IrGraph::constant_shape`).
//!
//! The expected values in the tests are rc4's kernels run under
//! `TosaLoweringContext` and `FakeTensorMode` (TOSA-1.0+INT, +int16, +FP,
//! +FP+bf16) with rc4 installed.
//!
//! [`dtype::operand_dtype`]: crate::layout::dtype::operand_dtype

use super::{required_int_list, Operands, ShapeError};
use crate::fx_ir::IrArg;
use crate::fx_meta::TensorFacts;
use crate::layout::dtype::{conv_output_dtype, input_qparam_dtype, resize_output_dtype};

/// A TOSA op with a rule here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TosaOp {
    Conv2d,
    DepthwiseConv2d,
    TransposeConv2d,
    Conv3d,
    MaxPool2d,
    AvgPool2d,
    Resize,
}

impl TosaOp {
    pub(crate) fn of(target: &str) -> Option<Self> {
        Some(match tosa_op_name(target)? {
            "CONV2D" => Self::Conv2d,
            "DEPTHWISE_CONV2D" => Self::DepthwiseConv2d,
            "TRANSPOSE_CONV2D" => Self::TransposeConv2d,
            "CONV3D" => Self::Conv3d,
            "MAX_POOL2D" => Self::MaxPool2d,
            "AVG_POOL2D" => Self::AvgPool2d,
            "RESIZE" => Self::Resize,
            _ => return None,
        })
    }
}

/// The op name of a TOSA dialect target: `CONV2D` for
/// `backend.tosa.CONV2D.default`, which RETAB's IR also spells
/// `tosa.CONV2D.default`.
pub(crate) fn tosa_op_name(target: &str) -> Option<&str> {
    target
        .strip_prefix("backend.tosa.")
        .or_else(|| target.strip_prefix("tosa."))?
        .strip_suffix(".default")
}

/// TOSA's limit on RESIZE input and output dimensions (`resize_utils.py:12`).
const MAX_RESIZE_DIMENSION: i64 = 16384;
/// `_MAX_RESIZE_SCALE_NUMERATOR` (`resize_utils.py:13`).
const MAX_RESIZE_SCALE_NUMERATOR: i64 = 1 << 11;
/// `_MAX_SCALE` (`resize_utils.py:14`). The 8K level's 256 (`:15`, `:38-39`)
/// is not modelled: a rule has no spec, and Ethos-U specs never enable `8k`.
const MAX_SCALE: i64 = 2048;
const INT16_MIN: i64 = -(1 << 15);
const INT16_MAX: i64 = (1 << 15) - 1;

/// An output extent: non-negative, and in `i64`.
fn extent(value: i128) -> Result<i64, ShapeError> {
    if value < 0 {
        return Err(ShapeError::InvalidArg("negative output extent"));
    }
    i64::try_from(value).map_err(|_| ShapeError::InvalidArg("output extent overflows"))
}

fn rank<const N: usize>(shape: &[i64], what: &'static str) -> Result<[i64; N], ShapeError> {
    shape.try_into().map_err(|_| ShapeError::InvalidArg(what))
}

fn values<const N: usize>(list: &[i64], what: &'static str) -> Result<[i64; N], ShapeError> {
    list.try_into().map_err(|_| ShapeError::InvalidArg(what))
}

/// `conv_output_dim` (`conv2d.py:91-105`): `(in + pads - receptive field)`
/// floor-divided by the stride, plus one.
pub(crate) fn conv_output_dim(
    input: i64,
    kernel: i64,
    stride: i64,
    pad_before: i64,
    pad_after: i64,
    dilation: i64,
) -> Result<i64, ShapeError> {
    if stride < 1 {
        return Err(ShapeError::InvalidArg(
            "convolution stride must be positive",
        ));
    }
    let receptive_field = i128::from(dilation) * (i128::from(kernel) - 1) + 1;
    let span = i128::from(input) + i128::from(pad_before) + i128::from(pad_after) - receptive_field;
    extent(span.div_euclid(i128::from(stride)) + 1)
}

/// CONV2D (`conv2d.py:108-139`): an NHWC input, an OHWI weight, `stride[2]`,
/// `pad[4]` as (top, bottom, left, right) and `dilation[2]`.
pub(crate) fn conv2d_shape(
    input: &[i64],
    weight: &[i64],
    stride: &[i64],
    pad: &[i64],
    dilation: &[i64],
) -> Result<Vec<i64>, ShapeError> {
    let [n, h, w, _] = rank::<4>(input, "convolution input must be rank 4")?;
    let [c_out, kh, kw, _] = rank::<4>(weight, "convolution weight must be rank 4")?;
    let Conv2dParams {
        stride,
        pad,
        dilation,
    } = conv2d_params(stride, pad, dilation)?;
    Ok(vec![
        n,
        conv_output_dim(h, kh, stride[0], pad[0], pad[1], dilation[0])?,
        conv_output_dim(w, kw, stride[1], pad[2], pad[3], dilation[1])?,
        c_out,
    ])
}

/// DEPTHWISE_CONV2D (`depthwise_conv2d.py:19-49`): CONV2D's arguments with an
/// HWCM weight, giving `C * M` output channels.
pub(crate) fn depthwise_conv2d_shape(
    input: &[i64],
    weight: &[i64],
    stride: &[i64],
    pad: &[i64],
    dilation: &[i64],
) -> Result<Vec<i64>, ShapeError> {
    let [n, h, w, _] = rank::<4>(input, "convolution input must be rank 4")?;
    let [kh, kw, c, m] = rank::<4>(weight, "convolution weight must be rank 4")?;
    let Conv2dParams {
        stride,
        pad,
        dilation,
    } = conv2d_params(stride, pad, dilation)?;
    Ok(vec![
        n,
        conv_output_dim(h, kh, stride[0], pad[0], pad[1], dilation[0])?,
        conv_output_dim(w, kw, stride[1], pad[2], pad[3], dilation[1])?,
        extent(i128::from(c) * i128::from(m))?,
    ])
}

/// A 2-D convolution's `stride[2]`, `pad[4]` and `dilation[2]`.
struct Conv2dParams {
    stride: [i64; 2],
    pad: [i64; 4],
    dilation: [i64; 2],
}

fn conv2d_params(
    stride: &[i64],
    pad: &[i64],
    dilation: &[i64],
) -> Result<Conv2dParams, ShapeError> {
    Ok(Conv2dParams {
        stride: values(stride, "convolution stride must have 2 values")?,
        pad: values(pad, "convolution pad must have 4 values")?,
        dilation: values(dilation, "convolution dilation must have 2 values")?,
    })
}

/// TRANSPOSE_CONV2D (`transpose_conv2d.py:16-58`): an NHWC input, an OHWI
/// weight, `out_pad[4]` and `stride[2]`; each spatial output is
/// `(in - 1) * stride + out_pad before + out_pad after + kernel`.
pub(crate) fn transpose_conv2d_shape(
    input: &[i64],
    weight: &[i64],
    out_pad: &[i64],
    stride: &[i64],
) -> Result<Vec<i64>, ShapeError> {
    let out_pad = values::<4>(out_pad, "TRANSPOSE_CONV2D out_pad must have 4 values")?;
    let stride = values::<2>(stride, "TRANSPOSE_CONV2D stride must have 2 values")?;
    let [n, h, w, _] = rank::<4>(input, "convolution input must be rank 4")?;
    let [c_out, kh, kw, _] = rank::<4>(weight, "convolution weight must be rank 4")?;
    let spatial = |input: i64, stride: i64, before: i64, after: i64, kernel: i64| {
        extent(
            (i128::from(input) - 1) * i128::from(stride)
                + i128::from(before)
                + i128::from(after)
                + i128::from(kernel),
        )
    };
    Ok(vec![
        n,
        spatial(h, stride[0], out_pad[0], out_pad[1], kh)?,
        spatial(w, stride[1], out_pad[2], out_pad[3], kw)?,
        c_out,
    ])
}

/// CONV3D (`conv3d.py:21-69`): an NDHWC input, an ODHWI weight, `stride[3]`,
/// `pad[6]` and `dilation[3]`. rc4 checks both ranks.
pub(crate) fn conv3d_shape(
    input: &[i64],
    weight: &[i64],
    stride: &[i64],
    pad: &[i64],
    dilation: &[i64],
) -> Result<Vec<i64>, ShapeError> {
    let (Ok([n, d, h, w, _]), Ok([c_out, kd, kh, kw, _])) =
        (<[i64; 5]>::try_from(input), <[i64; 5]>::try_from(weight))
    else {
        return Err(ShapeError::InvalidArg(
            "CONV3D input and weight must be rank 5",
        ));
    };
    let stride = values::<3>(stride, "CONV3D stride must have 3 values")?;
    let pad = values::<6>(pad, "CONV3D pad must have 6 values")?;
    let dilation = values::<3>(dilation, "CONV3D dilation must have 3 values")?;
    Ok(vec![
        n,
        conv_output_dim(d, kd, stride[0], pad[0], pad[1], dilation[0])?,
        conv_output_dim(h, kh, stride[1], pad[2], pad[3], dilation[1])?,
        conv_output_dim(w, kw, stride[2], pad[4], pad[5], dilation[2])?,
        c_out,
    ])
}

/// MAX_POOL2D and AVG_POOL2D's output (`compute_pool2d_output_shape`,
/// `_common.py:105-134`): an NHWC input, `kernel[2]`, `stride[2]` and `pad[4]`
/// as (top, bottom, left, right); each spatial output is
/// `floor((in + pads - kernel) / stride) + 1`. Both kernels require rank 4
/// (`max_pool2d.py:85-86`, `avg_pool2d.py:128-129`).
pub(crate) fn pool2d_shape(
    input: &[i64],
    kernel: &[i64],
    stride: &[i64],
    pad: &[i64],
) -> Result<Vec<i64>, ShapeError> {
    let [n, h, w, c] = rank::<4>(input, "pool input must be rank 4")?;
    let kernel = values::<2>(kernel, "pool kernel must have 2 values")?;
    let stride = values::<2>(stride, "pool stride must have 2 values")?;
    let pad = values::<4>(pad, "pool pad must have 4 values")?;
    let spatial = |input: i64, before: i64, after: i64, kernel: i64, stride: i64| {
        if stride < 1 {
            return Err(ShapeError::InvalidArg("pool stride must be positive"));
        }
        let span = i128::from(input) + i128::from(before) + i128::from(after) - i128::from(kernel);
        extent(span.div_euclid(i128::from(stride)) + 1)
    };
    Ok(vec![
        n,
        spatial(h, pad[0], pad[1], kernel[0], stride[0])?,
        spatial(w, pad[2], pad[3], kernel[1], stride[1])?,
        c,
    ])
}

/// RESIZE's output height and width for an input of `input_hw`
/// (`resize.py:71-113`), after rc4's validation of the parameters
/// (`get_tosa_resize_validation_error`, `resize_utils.py:290-320`, without an
/// expected output) and of the output (`:88-103`). `scale` is
/// `[y_n, y_d, x_n, x_d]` in TOSA's doubled-rational form; `offset` and
/// `border` are `[y, x]`. The argument lengths are checked first; rc4's
/// unpacking checks them as it reaches each list.
pub(crate) fn resize_output_hw(
    input_hw: [i64; 2],
    scale: &[i64],
    offset: &[i64],
    border: &[i64],
) -> Result<[i64; 2], ShapeError> {
    let [y_n, y_d, x_n, x_d] = values::<4>(scale, "RESIZE scale must have 4 values")?;
    let [offset_y, offset_x] = values::<2>(offset, "RESIZE offset must have 2 values")?;
    let [border_y, border_x] = values::<2>(border, "RESIZE border must have 2 values")?;
    let invalid = ShapeError::InvalidArg;

    // `_validate_dimensions` (`:62-86`).
    if input_hw.iter().any(|&dim| dim >= MAX_RESIZE_DIMENSION) {
        return Err(invalid("RESIZE dimensions must be less than 16384"));
    }
    // `_validate_scale` (`:109-161`), without the exact-boundary allowance.
    let int16 = |value: &i64| (INT16_MIN..=INT16_MAX).contains(value);
    if !scale.iter().all(int16) {
        return Err(invalid("RESIZE scale must be in int16 range"));
    }
    if scale.iter().any(|&value| value <= 0) {
        return Err(invalid("RESIZE scale values must be positive"));
    }
    if y_n > MAX_SCALE * y_d || x_n > MAX_SCALE * x_d {
        return Err(invalid(
            "RESIZE scale ratio must be at most MAX_SCALE (2048)",
        ));
    }
    if y_n > MAX_RESIZE_SCALE_NUMERATOR || x_n > MAX_RESIZE_SCALE_NUMERATOR {
        return Err(invalid("RESIZE scale numerator must be at most 2048"));
    }
    if y_d >= 16 * y_n || x_d >= 16 * x_n {
        return Err(invalid("RESIZE downscale must be greater than 1/16"));
    }
    // `_validate_offset` (`:192-214`).
    if !(-y_n..16 * y_n).contains(&offset_y) || !(-x_n..16 * x_n).contains(&offset_x) {
        return Err(invalid("RESIZE offset out of range"));
    }
    // `_validate_border` (`:217-245`).
    if !border.iter().all(int16) {
        return Err(invalid("RESIZE border must be in int16 range"));
    }
    if !(-16 * y_n..y_n).contains(&border_y) || !(-16 * x_n..x_n).contains(&border_x) {
        return Err(invalid("RESIZE border out of range"));
    }
    // `calculate_tosa_resize_output_hw` (`:270-287`): upscale by the
    // numerator, apply offset and border, downscale by the denominator.
    let output = |input: i64, n: i64, d: i64, offset: i64, border: i64| {
        let upscaled =
            (i128::from(input) - 1) * i128::from(n) - i128::from(offset) + i128::from(border);
        upscaled.div_euclid(i128::from(d)) + 1
    };
    let output_hw = [
        output(input_hw[0], y_n, y_d, offset_y, border_y),
        output(input_hw[1], x_n, x_d, offset_x, border_x),
    ];
    // `get_tosa_resize_output_hw_validation_error` (`:88-103`).
    if output_hw.iter().any(|&dim| dim <= 0) {
        return Err(invalid("RESIZE output dimensions must be positive"));
    }
    if output_hw
        .iter()
        .any(|&dim| dim >= i128::from(MAX_RESIZE_DIMENSION))
    {
        return Err(invalid("RESIZE dimensions must be less than 16384"));
    }
    Ok(output_hw.map(|dim| dim as i64))
}

/// RESIZE (`resize.py:71-118`): an NHWC input resized to
/// [`resize_output_hw`].
pub(crate) fn resize_shape(
    input: &[i64],
    scale: &[i64],
    offset: &[i64],
    border: &[i64],
) -> Result<Vec<i64>, ShapeError> {
    let [n, h, w, c] = rank::<4>(input, "RESIZE input must be rank 4")?;
    let [oh, ow] = resize_output_hw([h, w], scale, offset, border)?;
    Ok(vec![n, oh, ow, c])
}

impl Operands<'_, '_> {
    pub(super) fn tosa(&self, op: TosaOp) -> Result<TensorFacts, ShapeError> {
        match op {
            TosaOp::Conv2d | TosaOp::DepthwiseConv2d | TosaOp::Conv3d => self.tosa_conv(op),
            TosaOp::TransposeConv2d => self.tosa_transpose_conv2d(),
            TosaOp::MaxPool2d => self.tosa_max_pool2d(),
            TosaOp::AvgPool2d => self.tosa_avg_pool2d(),
            TosaOp::Resize => self.tosa_resize(),
        }
    }

    /// The input dtype rc4's kernel sees; see the module docs.
    fn kernel_input_dtype(&self, input: &TensorFacts) -> String {
        input_qparam_dtype(&self.node.meta, 0)
            .unwrap_or(&input.dtype)
            .to_owned()
    }

    fn int_list_arg(
        &self,
        index: usize,
        name: &str,
        what: &'static str,
    ) -> Result<Vec<i64>, ShapeError> {
        required_int_list(self.arg(index, name), what)
    }

    /// CONV2D, DEPTHWISE_CONV2D and CONV3D:
    /// `(input, weight, bias, stride, pad, dilation)`.
    fn tosa_conv(&self, op: TosaOp) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "input")?;
        let weight = self.tensor(1, "weight")?;
        let stride = self.int_list_arg(3, "stride", "convolution stride")?;
        let pad = self.int_list_arg(4, "pad", "convolution pad")?;
        let dilation = self.int_list_arg(5, "dilation", "convolution dilation")?;
        let shape = match op {
            TosaOp::Conv2d => conv2d_shape(&input.shape, &weight.shape, &stride, &pad, &dilation),
            TosaOp::DepthwiseConv2d => {
                depthwise_conv2d_shape(&input.shape, &weight.shape, &stride, &pad, &dilation)
            }
            _ => conv3d_shape(&input.shape, &weight.shape, &stride, &pad, &dilation),
        }?;
        Ok(TensorFacts::new(shape, self.conv_output_dtype(input)?))
    }

    /// TRANSPOSE_CONV2D: `(input, weight, bias, out_pad, stride)`.
    fn tosa_transpose_conv2d(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "input")?;
        let weight = self.tensor(1, "weight")?;
        let out_pad = self.int_list_arg(3, "out_pad", "TRANSPOSE_CONV2D out_pad")?;
        let stride = self.int_list_arg(4, "stride", "TRANSPOSE_CONV2D stride")?;
        let dtype = self.conv_output_dtype(input)?;
        let shape = transpose_conv2d_shape(&input.shape, &weight.shape, &out_pad, &stride)?;
        Ok(TensorFacts::new(shape, dtype))
    }

    fn conv_output_dtype(&self, input: &TensorFacts) -> Result<&'static str, ShapeError> {
        conv_output_dtype(&self.kernel_input_dtype(input)).ok_or(ShapeError::Incompatible(
            "unsupported convolution input dtype",
        ))
    }

    /// MAX_POOL2D (`max_pool2d.py:55-75`): `(input, kernel, stride, pad)`; the
    /// output keeps the input dtype.
    fn tosa_max_pool2d(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "input")?;
        let kernel = self.int_list_arg(1, "kernel", "pool kernel")?;
        let stride = self.int_list_arg(2, "stride", "pool stride")?;
        let pad = self.int_list_arg(3, "pad", "pool pad")?;
        let shape = pool2d_shape(&input.shape, &kernel, &stride, &pad)?;
        Ok(TensorFacts::new(shape, self.kernel_input_dtype(input)))
    }

    /// AVG_POOL2D (`avg_pool2d.py:92-133`): `(input, input_zp, output_zp,
    /// kernel, stride, pad, acc_type)`; the output keeps the input dtype.
    fn tosa_avg_pool2d(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "input")?;
        self.tensor(1, "input_zp")?;
        self.tensor(2, "output_zp")?;
        let kernel = self.int_list_arg(3, "kernel", "pool kernel")?;
        let stride = self.int_list_arg(4, "stride", "pool stride")?;
        let pad = self.int_list_arg(5, "pad", "pool pad")?;
        if !matches!(
            self.arg(6, "acc_type"),
            Some(IrArg::Dtype(_) | IrArg::Str(_))
        ) {
            return Err(ShapeError::InvalidArg("AVG_POOL2D acc_type"));
        }
        let shape = pool2d_shape(&input.shape, &kernel, &stride, &pad)?;
        Ok(TensorFacts::new(shape, self.kernel_input_dtype(input)))
    }

    /// RESIZE: `(input, scale_factors, offset, border, *, resize_mode)`.
    fn tosa_resize(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "input")?;
        if input.shape.len() != 4 {
            return Err(ShapeError::InvalidArg("RESIZE input must be rank 4"));
        }
        let bilinear = match self.node.kwarg("resize_mode").and_then(IrArg::as_str) {
            Some("nearest") => false,
            Some("bilinear") => true,
            _ => {
                return Err(ShapeError::InvalidArg(
                    "RESIZE resize_mode must be nearest or bilinear",
                ))
            }
        };
        let dtype = resize_output_dtype(&self.kernel_input_dtype(input), bilinear)
            .ok_or(ShapeError::Incompatible("unsupported RESIZE input dtype"))?;
        let scale = self.shape_values(1, "scale_factors")?;
        let offset = self.shape_values(2, "offset")?;
        let border = self.shape_values(3, "border")?;
        let shape = resize_shape(&input.shape, &scale, &offset, &border)?;
        Ok(TensorFacts::new(shape, dtype))
    }

    /// A `SymInt[]` argument: a literal list, or the `int[]` a `CONST_SHAPE`
    /// operand holds.
    fn shape_values(&self, index: usize, name: &'static str) -> Result<Vec<i64>, ShapeError> {
        match self.arg(index, name) {
            Some(IrArg::Node(id)) => {
                (self.shape_values_of)(*id).ok_or(ShapeError::MissingOperand(name))
            }
            Some(arg) => arg.as_int_list().ok_or(ShapeError::InvalidArg(name)),
            None => Err(ShapeError::MissingOperand(name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeId;
    use crate::layout::shape::{has_local_rule, infer_facts, refreshed_meta};
    use crate::layout::testing::*;
    use crate::test_ir::qparam;

    const I8: &str = "torch.int8";
    const I16: &str = "torch.int16";
    const I32: &str = "torch.int32";
    const F32: &str = "torch.float32";
    const F16: &str = "torch.float16";
    const BF16: &str = "torch.bfloat16";

    fn invalid(what: &'static str) -> Result<Vec<i64>, ShapeError> {
        Err(ShapeError::InvalidArg(what))
    }

    /// `(input, weight, stride, pad, dilation, expected)` for a 2-D convolution.
    type Conv2dCase = (
        &'static [i64],
        &'static [i64],
        [i64; 2],
        [i64; 4],
        [i64; 2],
        [i64; 4],
    );
    /// `(input, weight, out_pad, stride, expected)`.
    type TransposeConv2dCase = (&'static [i64], &'static [i64], [i64; 4], [i64; 2], [i64; 4]);
    /// `(input, weight, stride, pad, dilation, expected)` for CONV3D.
    type Conv3dCase = (
        &'static [i64],
        &'static [i64],
        [i64; 3],
        [i64; 6],
        [i64; 3],
        [i64; 5],
    );
    /// `(input, kernel, stride, pad, expected)`.
    type PoolCase = (&'static [i64], [i64; 2], [i64; 2], [i64; 4], [i64; 4]);
    /// `(input, scale, offset, border, expected)`.
    type ResizeCase = ([i64; 4], [i64; 4], [i64; 2], [i64; 2], [i64; 4]);
    /// `(input, scale, offset, border, error)`.
    type ResizeErrorCase = (
        &'static [i64],
        &'static [i64],
        &'static [i64],
        &'static [i64],
        &'static str,
    );

    // rc4's kernels, case by case: (input, weight, args..., expected).

    #[test]
    fn conv2d_matches_rc4() {
        #[rustfmt::skip]
        let cases: [Conv2dCase; 7] = [
            (&[1, 8, 20, 20], &[8, 2, 5, 5], [1, 1], [2, 2, 2, 2], [1, 1], [1, 11, 20, 8]),
            (&[1, 8, 20, 20], &[4, 2, 5, 5], [2, 2], [2, 2, 2, 2], [1, 1], [1, 6, 10, 4]),
            (&[2, 9, 9, 3], &[5, 3, 3, 3], [2, 2], [1, 0, 1, 0], [1, 1], [2, 4, 4, 5]),
            (&[1, 7, 10, 4], &[6, 3, 2, 4], [1, 3], [0, 1, 2, 0], [2, 1], [1, 4, 4, 6]),
            (&[1, 1, 12, 3], &[4, 1, 3, 3], [1, 2], [0, 0, 1, 0], [1, 2], [1, 1, 5, 4]),
            (&[1, 5, 5, 2], &[3, 3, 3, 2], [1, 1], [1, 1, 1, 1], [1, 1], [1, 5, 5, 3]),
            (&[1, 5, 5, 2], &[3, 3, 3, 2], [3, 3], [0, 0, 0, 0], [1, 1], [1, 1, 1, 3]),
        ];
        for (input, weight, stride, pad, dilation, expected) in cases {
            assert_eq!(
                conv2d_shape(input, weight, &stride, &pad, &dilation),
                Ok(expected.to_vec()),
                "{input:?} {weight:?}"
            );
        }
    }

    #[test]
    fn depthwise_conv2d_matches_rc4() {
        #[rustfmt::skip]
        let cases: [Conv2dCase; 5] = [
            (&[1, 9, 9, 4], &[3, 3, 4, 2], [2, 2], [1, 1, 1, 1], [1, 1], [1, 5, 5, 8]),
            (&[1, 1, 10, 3], &[1, 3, 3, 1], [1, 2], [0, 0, 1, 0], [1, 1], [1, 1, 5, 3]),
            // A KW = 1 HWCM weight (design §2.4).
            (&[1, 6, 6, 4], &[3, 1, 4, 2], [2, 2], [1, 0, 0, 0], [1, 1], [1, 3, 3, 8]),
            (&[2, 8, 7, 3], &[2, 3, 3, 1], [1, 2], [1, 1, 0, 1], [3, 1], [2, 7, 3, 3]),
            (&[1, 4, 4, 1], &[1, 1, 1, 6], [1, 1], [0, 0, 0, 0], [1, 1], [1, 4, 4, 6]),
        ];
        for (input, weight, stride, pad, dilation, expected) in cases {
            assert_eq!(
                depthwise_conv2d_shape(input, weight, &stride, &pad, &dilation),
                Ok(expected.to_vec()),
                "{input:?} {weight:?}"
            );
        }
    }

    #[test]
    fn transpose_conv2d_matches_rc4() {
        #[rustfmt::skip]
        let cases: [TransposeConv2dCase; 4] = [
            (&[1, 4, 4, 3], &[6, 3, 3, 3], [-1, -1, -1, -1], [2, 2], [1, 7, 7, 6]),
            (&[1, 4, 4, 3], &[6, 3, 3, 3], [-1, 0, -1, 0], [2, 2], [1, 8, 8, 6]),
            (&[2, 3, 5, 2], &[4, 2, 3, 2], [0, 0, -1, 0], [1, 2], [2, 4, 10, 4]),
            (&[1, 2, 3, 2], &[1, 4, 1, 2], [0, 0, 0, 0], [3, 1], [1, 7, 3, 1]),
        ];
        for (input, weight, out_pad, stride, expected) in cases {
            assert_eq!(
                transpose_conv2d_shape(input, weight, &out_pad, &stride),
                Ok(expected.to_vec()),
                "{input:?} {out_pad:?}"
            );
        }
        let (input, weight) = ([1, 4, 4, 3], [6, 3, 3, 3]);
        assert_eq!(
            transpose_conv2d_shape(&input, &weight, &[-1, -1, -1], &[2, 2]),
            invalid("TRANSPOSE_CONV2D out_pad must have 4 values")
        );
        assert_eq!(
            transpose_conv2d_shape(&input, &weight, &[-1, -1, -1, -1], &[2]),
            invalid("TRANSPOSE_CONV2D stride must have 2 values")
        );
    }

    #[test]
    fn conv3d_matches_rc4() {
        #[rustfmt::skip]
        let cases: [Conv3dCase; 3] = [
            (&[1, 4, 8, 8, 3], &[5, 2, 3, 3, 3], [1, 2, 2], [0, 0, 1, 1, 1, 1], [1, 1, 1], [1, 3, 4, 4, 5]),
            (&[2, 6, 5, 7, 2], &[3, 3, 2, 3, 2], [2, 1, 3], [1, 0, 0, 1, 2, 2], [1, 2, 1], [2, 3, 4, 3, 3]),
            (&[1, 3, 3, 3, 1], &[2, 1, 1, 1, 1], [1, 1, 1], [0; 6], [1, 1, 1], [1, 3, 3, 3, 2]),
        ];
        for (input, weight, stride, pad, dilation, expected) in cases {
            assert_eq!(
                conv3d_shape(input, weight, &stride, &pad, &dilation),
                Ok(expected.to_vec()),
                "{input:?} {weight:?}"
            );
        }
        assert_eq!(
            conv3d_shape(
                &[1, 8, 8, 3],
                &[5, 3, 3, 3],
                &[1, 1, 1],
                &[0; 6],
                &[1, 1, 1]
            ),
            invalid("CONV3D input and weight must be rank 5")
        );
    }

    #[test]
    fn pools_match_rc4() {
        #[rustfmt::skip]
        let cases: [PoolCase; 7] = [
            // MAX_POOL2D
            (&[1, 8, 8, 3], [2, 2], [2, 2], [0, 0, 0, 0], [1, 4, 4, 3]),
            (&[1, 7, 9, 5], [3, 3], [2, 2], [1, 1, 1, 1], [1, 4, 5, 5]),
            (&[2, 6, 5, 4], [3, 2], [1, 3], [1, 0, 0, 1], [2, 5, 2, 4]),
            (&[1, 5, 5, 2], [5, 5], [1, 1], [2, 2, 2, 2], [1, 5, 5, 2]),
            // AVG_POOL2D
            (&[2, 7, 9, 5], [3, 3], [1, 2], [1, 1, 1, 0], [2, 7, 4, 5]),
            (&[1, 6, 6, 2], [3, 2], [3, 2], [0, 0, 0, 0], [1, 2, 3, 2]),
            (&[1, 5, 7, 4], [4, 4], [3, 3], [1, 2, 2, 1], [1, 2, 3, 4]),
        ];
        for (input, kernel, stride, pad, expected) in cases {
            assert_eq!(
                pool2d_shape(input, &kernel, &stride, &pad),
                Ok(expected.to_vec()),
                "{input:?} {kernel:?}"
            );
        }
        assert_eq!(
            pool2d_shape(&[8, 8, 3], &[2, 2], &[2, 2], &[0; 4]),
            invalid("pool input must be rank 4")
        );
        assert_eq!(
            pool2d_shape(&[1, 8, 8, 3], &[2], &[2, 2], &[0; 4]),
            invalid("pool kernel must have 2 values")
        );
        assert_eq!(
            pool2d_shape(&[1, 8, 8, 3], &[2, 2], &[0, 2], &[0; 4]),
            invalid("pool stride must be positive")
        );
    }

    /// RESIZE's parameters for each case come from rc4's
    /// `RewriteUpsamplePass.get_resize_parameters_1d` (`rewrite_upsample.py:45-155`).
    #[test]
    fn resize_matches_rc4() {
        #[rustfmt::skip]
        let cases: [ResizeCase; 9] = [
            // nearest 5 -> 8
            ([1, 5, 5, 3], [16, 10, 16, 10], [-8, -8], [-2, -2], [1, 8, 8, 3]),
            // nearest 2x with scale factor 2.0
            ([1, 4, 6, 2], [4, 2, 4, 2], [-1, -1], [1, 1], [1, 8, 12, 2]),
            // bilinear align_corners 5 -> 9
            ([1, 5, 5, 3], [4, 2, 4, 2], [0, 0], [0, 0], [1, 9, 9, 3]),
            // bilinear (3, 4) -> (6, 7)
            ([2, 3, 4, 1], [4, 2, 14, 8], [-1, -3], [1, 3], [2, 6, 7, 1]),
            // nearest (3, 3) -> (7, 5)
            ([1, 3, 3, 2], [14, 6, 10, 6], [-7, -5], [1, -1], [1, 7, 5, 2]),
            // nearest with scale factors (1.5, 2.0)
            ([1, 4, 5, 3], [6, 4, 4, 2], [-3, -1], [-1, 1], [1, 6, 10, 3]),
            // bilinear downscale (6, 6) -> (3, 2)
            ([1, 6, 6, 3], [2, 4, 2, 6], [1, 2], [-1, -2], [1, 3, 2, 3]),
            // just above TOSA's 1/16 downscale bound
            ([1, 5, 5, 3], [2, 31, 2, 2], [0, 0], [0, 0], [1, 1, 5, 3]),
            // an output just under the dimension limit
            ([1, 8192, 5, 3], [4, 2, 2, 2], [0, 0], [0, 0], [1, 16383, 5, 3]),
        ];
        for (input, scale, offset, border, expected) in cases {
            assert_eq!(
                resize_shape(&input, &scale, &offset, &border),
                Ok(expected.to_vec()),
                "{input:?} {scale:?}"
            );
        }
    }

    /// Each validation branch rc4 raises on, with the inputs that reach it.
    #[test]
    fn resize_rejects_what_rc4_rejects() {
        #[rustfmt::skip]
        let cases: [ResizeErrorCase; 14] = [
            (&[5, 5, 3], &[16, 10, 16, 10], &[-8, -8], &[-2, -2], "RESIZE input must be rank 4"),
            (&[1, 16384, 5, 3], &[2, 2, 2, 2], &[0, 0], &[0, 0], "RESIZE dimensions must be less than 16384"),
            (&[1, 8193, 5, 3], &[4, 2, 2, 2], &[0, 0], &[0, 0], "RESIZE dimensions must be less than 16384"),
            (&[1, 5, 5, 3], &[40000, 2, 2, 2], &[0, 0], &[0, 0], "RESIZE scale must be in int16 range"),
            (&[1, 5, 5, 3], &[0, 10, 16, 10], &[-8, -8], &[-2, -2], "RESIZE scale values must be positive"),
            (&[1, 5, 5, 3], &[2049, 1, 2, 2], &[0, 0], &[0, 0], "RESIZE scale ratio must be at most MAX_SCALE (2048)"),
            (&[1, 5, 5, 3], &[4096, 2, 16, 10], &[0, -8], &[0, -2], "RESIZE scale numerator must be at most 2048"),
            (&[1, 5, 5, 3], &[3000, 10, 16, 10], &[-8, -8], &[-2, -2], "RESIZE scale numerator must be at most 2048"),
            (&[1, 5, 5, 3], &[2, 32, 2, 2], &[0, 0], &[0, 0], "RESIZE downscale must be greater than 1/16"),
            (&[1, 5, 5, 3], &[16, 10, 16, 10], &[-17, -8], &[-2, -2], "RESIZE offset out of range"),
            (&[1, 5, 5, 3], &[16, 10, 16, 10], &[-8, 256], &[-2, -2], "RESIZE offset out of range"),
            (&[1, 5, 5, 3], &[2, 2, 2, 2], &[0, 0], &[40000, 0], "RESIZE border must be in int16 range"),
            (&[1, 5, 5, 3], &[16, 10, 16, 10], &[-8, -8], &[16, -2], "RESIZE border out of range"),
            (&[1, 5, 5, 3], &[16, 10, 16, 10], &[-8, -8], &[-2, -257], "RESIZE border out of range"),
        ];
        for (input, scale, offset, border, error) in cases {
            assert_eq!(
                resize_shape(input, scale, offset, border),
                invalid(error),
                "{input:?} {scale:?} {offset:?} {border:?}"
            );
        }
        // Borders inside their range that still leave no output (rc4: -12, -11).
        for (scale, offset, border) in [
            ([16, 10, 16, 10], [-8, -8], [-200, -2]),
            ([2, 2, 2, 2], [0, 0], [-32, 0]),
        ] {
            assert_eq!(
                resize_shape(&[1, 5, 5, 3], &scale, &offset, &border),
                invalid("RESIZE output dimensions must be positive")
            );
        }
        assert_eq!(
            resize_shape(&[1, 5, 5, 3], &[16, 10, 16], &[-8, -8], &[-2, -2]),
            invalid("RESIZE scale must have 4 values")
        );
    }

    #[test]
    fn convolution_arguments_are_checked() {
        let (input, weight) = ([1, 8, 8, 3], [4, 3, 3, 3]);
        assert_eq!(
            conv2d_shape(&input, &weight, &[1], &[0; 4], &[1, 1]),
            invalid("convolution stride must have 2 values")
        );
        assert_eq!(
            conv2d_shape(&input, &weight, &[1, 1], &[0; 2], &[1, 1]),
            invalid("convolution pad must have 4 values")
        );
        assert_eq!(
            conv2d_shape(&input, &weight, &[0, 1], &[0; 4], &[1, 1]),
            invalid("convolution stride must be positive")
        );
        assert_eq!(
            conv2d_shape(&[1, 8, 3], &weight, &[1, 1], &[0; 4], &[1, 1]),
            invalid("convolution input must be rank 4")
        );
        assert_eq!(
            depthwise_conv2d_shape(&input, &[3, 3, 3], &[1, 1], &[0; 4], &[1, 1]),
            invalid("convolution weight must be rank 4")
        );
        // A kernel larger than the padded input: torch.empty rejects the size.
        assert_eq!(
            conv2d_shape(&[1, 2, 2, 3], &[4, 5, 5, 3], &[2, 2], &[0; 4], &[1, 1]),
            invalid("negative output extent")
        );
        assert_eq!(
            conv_output_dim(2, 5, 2, 0, 0, 1),
            Err(ShapeError::InvalidArg("negative output extent"))
        );
        assert_eq!(conv_output_dim(4, 5, 1, 0, 0, 1), Ok(0));
    }

    #[test]
    fn target_names_cover_both_spellings() {
        assert_eq!(tosa_op_name("backend.tosa.CONV2D.default"), Some("CONV2D"));
        assert_eq!(tosa_op_name("tosa.RESIZE.default"), Some("RESIZE"));
        assert_eq!(tosa_op_name("aten.convolution.default"), None);
        assert_eq!(tosa_op_name("backend.tosa.CONV2D.out"), None);
        assert_eq!(TosaOp::of(TOSA_AVG_POOL2D), Some(TosaOp::AvgPool2d));
        assert_eq!(
            TosaOp::of("tosa.DEPTHWISE_CONV2D.default"),
            Some(TosaOp::DepthwiseConv2d)
        );
        assert_eq!(TosaOp::of(RESCALE), None);
        for target in [
            TOSA_CONV2D,
            TOSA_DEPTHWISE_CONV2D,
            TOSA_TRANSPOSE_CONV2D,
            TOSA_CONV3D,
            TOSA_MAX_POOL2D,
            TOSA_AVG_POOL2D,
            TOSA_RESIZE,
        ] {
            assert!(has_local_rule(target), "{target}");
        }
        assert!(!has_local_rule(TOSA_CONST_SHAPE));
    }

    // The rules on graphs built with the layout test DSL.

    fn conv_args(x: NodeId, w: NodeId, b: Option<NodeId>, tail: &[&[i64]]) -> Vec<IrArg> {
        let mut args = vec![node(x), node(w), b.map_or(IrArg::None, node)];
        args.extend(tail.iter().map(|list| ints(list)));
        args
    }

    #[test]
    fn dsl_builds_tosa_convolutions() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 8, 20, 20], I8);
        let w = g.typed_input("w", &[8, 2, 5, 5], I8);
        let b = g.typed_input("b", &[8], I32);
        let conv = g.call(
            TOSA_CONV2D,
            conv_args(x, w, Some(b), &[&[1, 1], &[2, 2, 2, 2], &[1, 1]]),
        );
        assert_eq!((g.shape(conv), g.dtype(conv)), (&[1, 11, 20, 8][..], I32));
        let no_bias = g.call(
            TOSA_CONV2D,
            conv_args(x, w, None, &[&[2, 2], &[2, 2, 2, 2], &[1, 1]]),
        );
        assert_eq!(g.shape(no_bias), &[1, 6, 10, 8]);

        let hwcm = g.typed_input("hwcm", &[3, 3, 20, 2], I8);
        let dw = g.call(
            TOSA_DEPTHWISE_CONV2D,
            conv_args(x, hwcm, None, &[&[1, 1], &[1, 1, 1, 1], &[1, 1]]),
        );
        assert_eq!((g.shape(dw), g.dtype(dw)), (&[1, 8, 20, 40][..], I32));

        let ohwi = g.typed_input("ohwi", &[6, 3, 3, 20], F32);
        let fx = g.typed_input("fx", &[1, 4, 4, 20], F32);
        let tconv = g.call(
            TOSA_TRANSPOSE_CONV2D,
            conv_args(fx, ohwi, None, &[&[-1, -1, -1, -1], &[2, 2]]),
        );
        assert_eq!((g.shape(tconv), g.dtype(tconv)), (&[1, 7, 7, 6][..], F32));

        let x3 = g.typed_input("x3", &[1, 4, 8, 8, 3], I16);
        let w3 = g.typed_input("w3", &[5, 2, 3, 3, 3], I8);
        let conv3d = g.call(
            TOSA_CONV3D,
            conv_args(x3, w3, None, &[&[1, 2, 2], &[0, 0, 1, 1, 1, 1], &[1, 1, 1]]),
        );
        assert_eq!(
            (g.shape(conv3d), g.dtype(conv3d)),
            (&[1, 3, 4, 4, 5][..], I32)
        );

        let half = g.typed_input("half", &[1, 5, 5, 2], F16);
        let half_w = g.typed_input("half_w", &[3, 3, 3, 2], F16);
        let bf = g.typed_input("bf", &[1, 5, 5, 2], BF16);
        let bf_w = g.typed_input("bf_w", &[3, 3, 3, 2], BF16);
        let half_conv = g.call(
            TOSA_CONV2D,
            conv_args(half, half_w, None, &[&[3, 3], &[0; 4], &[1, 1]]),
        );
        let bf_conv = g.call(
            TOSA_CONV2D,
            conv_args(bf, bf_w, None, &[&[1, 1], &[1, 1, 1, 1], &[1, 1]]),
        );
        assert_eq!(
            (g.shape(half_conv), g.dtype(half_conv)),
            (&[1, 1, 1, 3][..], F16)
        );
        assert_eq!(
            (g.shape(bf_conv), g.dtype(bf_conv)),
            (&[1, 5, 5, 3][..], BF16)
        );
    }

    /// RETAB keeps a quantized producer's val in float32 (design F2); the TOSA
    /// convolution inherits the convolution's input qparams, so its rule sees
    /// rc4's int8 input and gives int32.
    #[test]
    fn convolution_input_dtype_comes_from_inherited_qparams() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 8, 8, 3]);
        let w = g.typed_input("w", &[4, 3, 3, 3], I8);
        let args = conv_args(x, w, None, &[&[1, 1], &[0; 4], &[1, 1]]);
        let conv = g.add_unrefreshed(TOSA_CONV2D, args.clone());
        assert_eq!(infer_facts(g.graph(), conv).unwrap().dtype, F32);
        g.graph_mut()
            .node_mut(conv)
            .meta
            .set_input_qparams_from([(0, qparam(0.1, 0, I16)), (1, qparam(0.2, 0, I8))]);
        assert_eq!(
            infer_facts(g.graph(), conv).unwrap(),
            TensorFacts::new(vec![1, 6, 6, 4], I32)
        );
        // A recorded dtype survives a refresh, as for every op but a copy.
        g.graph_mut()
            .node_mut(conv)
            .meta
            .set_tensor_facts(TensorFacts::new(vec![1, 1, 1, 1], "torch.int64"));
        let meta = refreshed_meta(g.graph(), conv).unwrap();
        assert_eq!(meta.val_shape(), Some(&[1, 6, 6, 4][..]));
        assert_eq!(meta.val_dtype(), Some("torch.int64"));
        assert!(meta.input_qparams().is_some());

        let int32 = g.typed_input("int32", &[1, 8, 8, 3], I32);
        let bad = g.add_unrefreshed(
            TOSA_CONV2D,
            conv_args(int32, w, None, &[&[1, 1], &[0; 4], &[1, 1]]),
        );
        assert_eq!(
            infer_facts(g.graph(), bad),
            Err(ShapeError::Incompatible(
                "unsupported convolution input dtype"
            ))
        );
    }

    #[test]
    fn dsl_builds_tosa_pools() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 8, 8, 3], I8);
        let max = g.call(
            TOSA_MAX_POOL2D,
            vec![node(x), ints(&[2, 2]), ints(&[2, 2]), ints(&[0; 4])],
        );
        assert_eq!((g.shape(max), g.dtype(max)), (&[1, 4, 4, 3][..], I8));

        let in_zp = g.typed_input("in_zp", &[1], I8);
        let out_zp = g.typed_input("out_zp", &[1], I8);
        let avg_args = |kernel: &[i64]| {
            vec![
                node(x),
                node(in_zp),
                node(out_zp),
                ints(kernel),
                ints(&[3, 3]),
                ints(&[1, 2, 2, 1]),
                IrArg::Dtype(I32.into()),
            ]
        };
        let avg = g.call(TOSA_AVG_POOL2D, avg_args(&[4, 4]));
        assert_eq!((g.shape(avg), g.dtype(avg)), (&[1, 3, 3, 3][..], I8));

        // Critique B1: AVG_POOL2D's kernel is argument 3, after the zero points.
        let short_kernel = g.add_unrefreshed(TOSA_AVG_POOL2D, avg_args(&[4]));
        assert_eq!(
            infer_facts(g.graph(), short_kernel),
            Err(ShapeError::InvalidArg("pool kernel must have 2 values"))
        );
        let no_acc = g.add_unrefreshed(TOSA_AVG_POOL2D, avg_args(&[4, 4])[..6].to_vec());
        assert_eq!(
            infer_facts(g.graph(), no_acc),
            Err(ShapeError::InvalidArg("AVG_POOL2D acc_type"))
        );

        // A pool copies the pool's qparams (design §3.1): its input dtype is
        // the input qparam's, whatever the input's val says.
        let float_x = g.input("float_x", &[1, 8, 8, 3]);
        let pool = g.add_unrefreshed(
            TOSA_MAX_POOL2D,
            vec![node(float_x), ints(&[2, 2]), ints(&[2, 2]), ints(&[0; 4])],
        );
        g.graph_mut()
            .node_mut(pool)
            .meta
            .set_single_input_qparam(qparam(0.1, 0, I8));
        assert_eq!(
            infer_facts(g.graph(), pool).unwrap(),
            TensorFacts::new(vec![1, 4, 4, 3], I8)
        );
    }

    /// rc4's graphs pass RESIZE's parameters as `CONST_SHAPE` nodes
    /// (`rewrite_upsample.py:205-230`).
    #[test]
    fn dsl_builds_resizes_from_const_shapes_or_lists() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 5, 5, 3], I8);
        let scale = g.const_shape(&[16, 10, 16, 10]);
        let offset = g.const_shape(&[-8, -8]);
        let border = g.const_shape(&[-2, -2]);
        let mode = |name: &str| [("resize_mode", IrArg::Str(name.into()))];
        let nearest = g.call_with_kwargs(
            TOSA_RESIZE,
            vec![node(x), node(scale), node(offset), node(border)],
            &mode("nearest"),
        );
        assert_eq!(
            (g.shape(nearest), g.dtype(nearest)),
            (&[1, 8, 8, 3][..], I8)
        );
        let bilinear = g.call_with_kwargs(
            TOSA_RESIZE,
            vec![node(x), node(scale), node(offset), node(border)],
            &mode("bilinear"),
        );
        assert_eq!(g.dtype(bilinear), I32);
        let literal = g.call_with_kwargs(
            TOSA_RESIZE,
            vec![node(x), ints(&[4, 2, 4, 2]), ints(&[0, 0]), ints(&[0, 0])],
            &mode("bilinear"),
        );
        assert_eq!(
            (g.shape(literal), g.dtype(literal)),
            (&[1, 9, 9, 3][..], I32)
        );

        let x16 = g.typed_input("x16", &[1, 3, 3, 2], I16);
        let int16 = g.call_with_kwargs(
            TOSA_RESIZE,
            vec![
                node(x16),
                ints(&[4, 2, 4, 2]),
                ints(&[-1, -1]),
                ints(&[1, 1]),
            ],
            &mode("bilinear"),
        );
        assert_eq!((g.shape(int16), g.dtype(int16)), (&[1, 6, 6, 2][..], I16));

        let area = g.add_unrefreshed(
            TOSA_RESIZE,
            vec![node(x), node(scale), node(offset), node(border)],
        );
        assert_eq!(
            infer_facts(g.graph(), area),
            Err(ShapeError::InvalidArg(
                "RESIZE resize_mode must be nearest or bilinear"
            ))
        );
        // A node operand that is not a CONST_SHAPE has no shape values.
        let not_a_shape = g.add_unrefreshed(
            TOSA_RESIZE,
            vec![node(x), node(x), node(offset), node(border)],
        );
        g.graph_mut()
            .node_mut(not_a_shape)
            .kwargs
            .insert("resize_mode".into(), IrArg::Str("nearest".into()));
        assert_eq!(
            infer_facts(g.graph(), not_a_shape),
            Err(ShapeError::MissingOperand("scale_factors"))
        );
        // The CONST_SHAPEs hold int[] values, not tensors.
        assert!(g.graph().node(scale).meta.tensor_facts().is_none());
    }

    /// A planner's hypothetical node has no graph to read `CONST_SHAPE`s from,
    /// but literal lists work.
    #[test]
    fn planners_can_infer_a_hypothetical_resize() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 5, 5, 3], I8);
        let resize = g.call_with_kwargs(
            TOSA_RESIZE,
            vec![
                node(x),
                ints(&[16, 10, 16, 10]),
                ints(&[-8, -8]),
                ints(&[-2, -2]),
            ],
            &[("resize_mode", IrArg::Str("nearest".into()))],
        );
        let moved = TensorFacts::new(vec![2, 5, 5, 3], I8);
        let planned = crate::layout::shape::infer_node_facts(g.graph().node(resize), &|id| {
            (id == x).then_some(&moved)
        })
        .unwrap();
        assert_eq!(planned.shape, vec![2, 8, 8, 3]);
    }
}
