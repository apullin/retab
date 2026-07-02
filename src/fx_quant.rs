use pyo3::prelude::*;

use crate::fx_py_attr::attrs;
use crate::graph::ir::DType;

/// Quantization parameters extracted once from Python QuantArgs objects.
#[derive(Debug, Clone)]
pub(crate) struct QParam {
    pub scale: f64,
    pub zp: i64,
    pub qmin: i64,
    pub qmax: i64,
    pub dtype_str: String,
    /// Per-channel scales (empty for per-tensor quantization).
    pub per_channel_scales: Vec<f64>,
    /// Per-channel zero points (empty for per-tensor quantization).
    pub per_channel_zps: Vec<i64>,
    /// Quantization axis for per-channel (default 0).
    pub axis: i64,
}

impl QParam {
    /// Extract from a Python QuantArgs object (or anything with scale/zp/dtype attrs).
    pub(crate) fn from_py(qp: &Bound<'_, PyAny>) -> Self {
        let qp_attrs = attrs(qp);
        let per_channel = qp_attrs.or_value("per_channel", false);
        let (per_channel_scales, per_channel_zps) = if per_channel {
            let scales = qp_attrs.or_value("scale", Vec::<f64>::new());
            let zps = qp_attrs.first_or(&["zp", "zero_point"], Vec::<i64>::new());
            (scales, zps)
        } else {
            (Vec::new(), Vec::new())
        };
        let axis = qp_attrs.or_value("axis", 0);
        Self {
            scale: qp_attrs.or_value("scale", 1.0),
            zp: qp_attrs.first_or(&["zp", "zero_point"], 0),
            qmin: qp_attrs.or_value("qmin", -128),
            qmax: qp_attrs.or_value("qmax", 127),
            dtype_str: qp_attrs.display_optional("dtype").unwrap_or_default(),
            per_channel_scales,
            per_channel_zps,
            axis,
        }
    }

    /// Create a per-tensor QParam (most common case).
    pub fn per_tensor(scale: f64, zp: i64, qmin: i64, qmax: i64, dtype_str: String) -> Self {
        Self {
            scale,
            zp,
            qmin,
            qmax,
            dtype_str,
            per_channel_scales: Vec::new(),
            per_channel_zps: Vec::new(),
            axis: 0,
        }
    }

    /// Is this per-channel quantization?
    pub fn is_per_channel(&self) -> bool {
        !self.per_channel_scales.is_empty()
    }

    pub fn dtype(&self) -> DType {
        DType::from_torch_str(&self.dtype_str)
    }

    pub fn is_integer_dtype(&self) -> bool {
        self.dtype().is_integer()
    }

    pub fn is_quantized_integer_dtype(&self) -> bool {
        self.dtype().is_quantized_integer()
    }

    pub(crate) fn is_unsigned_u8(&self) -> bool {
        self.qmin == 0 && self.qmax == 255 || self.dtype_str.to_ascii_lowercase().contains("uint8")
    }

    pub fn supports_axis0_channel_slice(&self, end: usize) -> bool {
        !self.is_per_channel()
            || (self.axis == 0
                && self.per_channel_scales.len() >= end
                && (self.per_channel_zps.is_empty() || self.per_channel_zps.len() >= end))
    }

    /// Slice output-channel qparams for grouped conv decomposition.
    pub fn slice_axis0_channels(&self, start: usize, end: usize) -> Option<Self> {
        if start > end || !self.supports_axis0_channel_slice(end) {
            return None;
        }
        let mut sliced = self.clone();
        if self.is_per_channel() {
            sliced.per_channel_scales = self.per_channel_scales[start..end].to_vec();
            sliced.per_channel_zps = if self.per_channel_zps.is_empty() {
                Vec::new()
            } else {
                self.per_channel_zps[start..end].to_vec()
            };
        }
        Some(sliced)
    }

    pub(crate) fn same_scale_zp(&self, other: &Self) -> bool {
        (self.scale - other.scale).abs() < 1e-10 && self.zp == other.zp
    }

    pub(crate) fn same_exact_scale_zp(&self, other: &Self) -> bool {
        self.scale == other.scale && self.zp == other.zp
    }

    pub(crate) fn same_exact_affine_dtype(&self, other: &Self) -> bool {
        self.same_exact_scale_zp(other) && self.dtype_str == other.dtype_str
    }

    pub(crate) fn same_per_tensor_exact_affine_dtype(&self, other: &Self) -> bool {
        !self.is_per_channel() && !other.is_per_channel() && self.same_exact_affine_dtype(other)
    }

    pub(crate) fn same_exact_affine_qrange_dtype(&self, other: &Self) -> bool {
        self.same_exact_affine_dtype(other) && self.qmin == other.qmin && self.qmax == other.qmax
    }
}

impl From<&Bound<'_, PyAny>> for QParam {
    fn from(qp: &Bound<'_, PyAny>) -> Self {
        Self::from_py(qp)
    }
}
