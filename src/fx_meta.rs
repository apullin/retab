//! Typed FX metadata imported into the Rust lowering IR.
//!
//! This module owns tensor, quantization, layout, and passthrough metadata
//! facts. Graph topology and rewrite mechanics live in `fx_ir`.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::ops::{Index, IndexMut};

use pyo3::prelude::*;

use crate::fx_py_attr::{attrs, items, meta as meta_view};
use crate::fx_quant::QParam;
use crate::graph::ir::DType;

/// Lossless tensor value facts imported from FX metadata.
///
/// The dtype stays in torch spelling so the FX-side IR can round-trip boundary
/// data exactly; typed decisions should use `dtype_kind`.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorFacts {
    pub shape: Vec<i64>,
    pub dtype: String,
}

impl TensorFacts {
    pub fn new(shape: impl Into<Vec<i64>>, dtype: impl Into<String>) -> Self {
        Self {
            shape: shape.into(),
            dtype: dtype.into(),
        }
    }

    pub fn from_py(val: &Bound<'_, PyAny>) -> Option<Self> {
        let val_attrs = attrs(val);
        let shape: Vec<i64> = val_attrs.optional("shape")?;
        let dtype = val_attrs.display_optional("dtype")?;
        Some(Self::new(shape, dtype))
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn dtype_kind(&self) -> DType {
        DType::from_torch_str(&self.dtype)
    }
}

/// Untyped meta value for passthrough of custom keys.
#[derive(Debug, Clone)]
pub enum MetaValue {
    Bool(bool),
    Str(String),
}

#[derive(Debug, Clone, Default)]
struct QuantFacts {
    input_qparams: Option<QParamMap>,
    output_qparams: Option<QParamMap>,
    pre_pass_arg_qparams: Option<QParamMap>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct QParamMap {
    entries: BTreeMap<usize, QParam>,
}

impl QParamMap {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn from_pairs(pairs: impl IntoIterator<Item = (usize, QParam)>) -> Self {
        Self {
            entries: pairs.into_iter().collect(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn get(&self, index: &usize) -> Option<&QParam> {
        self.entries.get(index)
    }

    pub(crate) fn get_mut(&mut self, index: &usize) -> Option<&mut QParam> {
        self.entries.get_mut(index)
    }

    pub(crate) fn insert(&mut self, index: usize, qparam: QParam) -> Option<QParam> {
        self.entries.insert(index, qparam)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&usize, &QParam)> {
        self.entries.iter()
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &QParam> {
        self.entries.values()
    }

    pub(crate) fn cloned_pairs(&self) -> Vec<(usize, QParam)> {
        self.entries
            .iter()
            .map(|(&index, qparam)| (index, qparam.clone()))
            .collect()
    }
}

impl From<HashMap<usize, QParam>> for QParamMap {
    fn from(map: HashMap<usize, QParam>) -> Self {
        Self::from_pairs(map)
    }
}

impl FromIterator<(usize, QParam)> for QParamMap {
    fn from_iter<T: IntoIterator<Item = (usize, QParam)>>(iter: T) -> Self {
        Self::from_pairs(iter)
    }
}

impl<'a> IntoIterator for &'a QParamMap {
    type Item = (&'a usize, &'a QParam);
    type IntoIter = std::collections::btree_map::Iter<'a, usize, QParam>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl Index<&usize> for QParamMap {
    type Output = QParam;

    fn index(&self, index: &usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl IndexMut<&usize> for QParamMap {
    fn index_mut(&mut self, index: &usize) -> &mut Self::Output {
        self.entries.get_mut(index).expect("missing qparam index")
    }
}

#[derive(Debug, Clone, Default)]
struct LayoutFacts {
    val_dim_order: Option<Vec<usize>>,
    tosa_dim_order: Option<Vec<usize>>,
    original_dim_orders: Option<Vec<Vec<usize>>>,
    tosa_spatial_rank: Option<usize>,
}

/// Metadata fields extracted from `node.meta`. The common fields are typed;
/// rare fields stay in the Python meta dict and are not touched.
#[derive(Debug, Clone, Default)]
pub struct NodeMeta {
    tensor: Option<TensorFacts>,
    quant: QuantFacts,
    layout: LayoutFacts,
    /// Arbitrary extra meta keys for passthrough (skip_size_adjust, arm_constant_placeholder, etc).
    pub extras: HashMap<String, MetaValue>,
}

fn extract_qparam_dict(py_dict: &Bound<'_, PyAny>) -> Option<QParamMap> {
    if py_dict.is_none() || !py_dict.is_truthy().unwrap_or(false) {
        return None;
    }
    let mut map = HashMap::new();
    for (key, val) in items(py_dict).typed_key_pairs::<usize>() {
        map.insert(key, QParam::from_py(&val));
    }
    if map.is_empty() {
        None
    } else {
        Some(QParamMap::from(map))
    }
}

impl NodeMeta {
    pub fn tensor(shape: impl Into<Vec<i64>>, dtype: impl Into<String>) -> Self {
        let mut meta = Self::default();
        meta.set_tensor_facts(TensorFacts::new(shape, dtype));
        meta
    }

    pub fn tensor_stamp<'a>(&'a self, dtype: impl Into<Cow<'a, str>>) -> TensorMetaStamp<'a> {
        TensorMetaStamp {
            base: self,
            dtype: dtype.into(),
        }
    }

    pub fn tensor_stamp_with_dtype_of_or<'a>(
        &'a self,
        dtype_source: &'a Self,
        default: &'a str,
    ) -> TensorMetaStamp<'a> {
        self.tensor_stamp(dtype_source.val_dtype_or(default))
    }

    pub fn tensor_stamp_with_dtype_of_or_float32<'a>(
        &'a self,
        dtype_source: &'a Self,
    ) -> TensorMetaStamp<'a> {
        self.tensor_stamp_with_dtype_of_or(dtype_source, "torch.float32")
    }

    pub fn tensor_stamp_same_dtype_or_float32(&self) -> TensorMetaStamp<'_> {
        self.tensor_stamp_with_dtype_of_or_float32(self)
    }

    pub fn tensor_stamp_same_dtype_or_dtype_of_or_float32<'a>(
        &'a self,
        dtype_source: &'a Self,
    ) -> TensorMetaStamp<'a> {
        match self.val_dtype() {
            Some(dtype) => self.tensor_stamp(dtype),
            None => self.tensor_stamp_with_dtype_of_or_float32(dtype_source),
        }
    }

    /// Extract typed metadata from a Python FX node.
    pub fn from_fx(node: &Bound<'_, PyAny>) -> PyResult<Self> {
        let meta = attrs(node).object("meta")?;
        let meta_attrs = meta_view(&meta);

        let input_qparams = meta_attrs
            .value("input_qparams")
            .and_then(|v| extract_qparam_dict(&v));

        let output_qparams = meta_attrs
            .value("output_qparams")
            .and_then(|v| extract_qparam_dict(&v));

        let val_obj = meta_attrs.value("val");
        let tensor = val_obj.as_ref().and_then(|v| TensorFacts::from_py(v));
        let val_dim_order = val_obj
            .as_ref()
            .and_then(|v| attrs(v).call_optional("dim_order"));

        let tosa_dim_order = meta_attrs.optional("tosa_dim_order");

        let original_dim_orders = meta_attrs.optional("original_dim_orders");

        let tosa_spatial_rank = meta_attrs.optional("tosa_spatial_rank");

        let pre_pass_arg_qparams = meta_attrs
            .value("_pre_pass_arg_qparams")
            .and_then(|v| extract_qparam_dict(&v));

        let mut extras = HashMap::new();
        for key in [
            "arm_constant_placeholder",
            "arm_control_flow_placeholder",
            "arm_control_flow_parent_output_user",
            "do_not_fuse_duplicate",
        ] {
            if let Some(b) = meta_attrs.optional::<bool>(key) {
                extras.insert(key.into(), MetaValue::Bool(b));
            }
        }
        if let Some(b) = meta_attrs.optional::<bool>("skip_size_adjust") {
            extras.insert("skip_size_adjust".into(), MetaValue::Bool(b));
        }
        if let Some(custom) = meta_attrs.value("custom") {
            if let Some(ann_info) = items(&custom).key("_arm_annotation_info") {
                if let Some(b) = items(&ann_info).optional_key::<bool>("quantized") {
                    extras.insert("arm_quantized_annotation".into(), MetaValue::Bool(b));
                }
            }
        }
        if let Some(s) = meta_attrs.display_optional("tosa_special_dtype") {
            extras.insert("tosa_special_dtype".into(), MetaValue::Str(s));
        }

        Ok(Self {
            tensor,
            quant: QuantFacts {
                input_qparams,
                output_qparams,
                pre_pass_arg_qparams,
            },
            layout: LayoutFacts {
                val_dim_order,
                tosa_dim_order,
                original_dim_orders,
                tosa_spatial_rank,
            },
            extras,
        })
    }

    pub fn val_shape(&self) -> Option<&[i64]> {
        self.tensor_facts().map(|v| v.shape.as_slice())
    }

    pub fn val_dtype(&self) -> Option<&str> {
        self.tensor_facts().map(|v| v.dtype.as_str())
    }

    pub fn val_dtype_kind(&self) -> Option<DType> {
        self.tensor_facts().map(TensorFacts::dtype_kind)
    }

    pub fn val_dtype_is(&self, dtype: DType) -> bool {
        self.val_dtype_kind() == Some(dtype)
    }

    pub fn val_dtype_or<'a>(&'a self, default: &'a str) -> &'a str {
        self.val_dtype().unwrap_or(default)
    }

    pub fn val_dtype_or_float32(&self) -> &str {
        self.val_dtype_or("torch.float32")
    }

    pub fn rank(&self) -> Option<usize> {
        self.tensor.as_ref().map(|v| v.rank())
    }

    pub fn val_dim_order(&self) -> Option<&[usize]> {
        self.layout.val_dim_order.as_deref()
    }

    pub fn tosa_dim_order(&self) -> Option<&[usize]> {
        self.layout.tosa_dim_order.as_deref()
    }

    pub fn cloned_tosa_dim_order(&self) -> Option<Vec<usize>> {
        self.layout.tosa_dim_order.clone()
    }

    pub fn original_dim_orders(&self) -> Option<&[Vec<usize>]> {
        self.layout.original_dim_orders.as_deref()
    }

    pub fn set_output_qparams(&mut self, map: impl Into<QParamMap>) {
        self.quant.output_qparams = Some(map.into());
    }

    pub fn set_input_qparams(&mut self, map: impl Into<QParamMap>) {
        self.quant.input_qparams = Some(map.into());
    }

    pub fn set_output_qparams_from(&mut self, pairs: impl IntoIterator<Item = (usize, QParam)>) {
        self.set_output_qparams(QParamMap::from_pairs(pairs));
    }

    pub fn set_input_qparams_from(&mut self, pairs: impl IntoIterator<Item = (usize, QParam)>) {
        self.set_input_qparams(QParamMap::from_pairs(pairs));
    }

    pub fn set_empty_input_qparams(&mut self) {
        self.set_input_qparams(QParamMap::new());
    }

    pub fn set_single_output_qparam(&mut self, qp: QParam) {
        self.set_output_qparam_at(0, qp);
    }

    pub fn set_output_qparam_at(&mut self, index: usize, qp: QParam) {
        let mut map = HashMap::new();
        map.insert(index, qp);
        self.set_output_qparams(map);
    }

    pub fn set_single_input_qparam(&mut self, qp: QParam) {
        self.set_input_qparam_at(0, qp);
    }

    pub fn set_input_qparam_at(&mut self, index: usize, qp: QParam) {
        let mut map = HashMap::new();
        map.insert(index, qp);
        self.set_input_qparams(map);
    }

    pub fn set_both_qparams(&mut self, qp: QParam) {
        self.set_single_output_qparam(qp.clone());
        self.set_single_input_qparam(qp);
    }

    pub fn set_unary_input_output_qparams(&mut self, input_qparam: QParam, output_qparam: QParam) {
        self.set_single_input_qparam(input_qparam);
        self.set_single_output_qparam(output_qparam);
    }

    pub fn set_binary_input_output_qparams(
        &mut self,
        lhs_qparam: QParam,
        rhs_qparam: QParam,
        output_qparam: QParam,
    ) {
        self.set_input_qparams_from([(0, lhs_qparam), (1, rhs_qparam)]);
        self.set_single_output_qparam(output_qparam);
    }

    pub fn set_pre_pass_arg_qparams(&mut self, map: impl Into<QParamMap>) {
        self.quant.pre_pass_arg_qparams = Some(map.into());
    }

    pub fn clear_output_qparams(&mut self) {
        self.quant.output_qparams = None;
    }

    pub fn clear_input_qparams(&mut self) {
        self.quant.input_qparams = None;
    }

    pub fn set_tosa_dim_order(&mut self, order: Vec<usize>) {
        self.layout.tosa_dim_order = Some(order);
    }

    pub fn set_val_dim_order(&mut self, order: Vec<usize>) {
        self.layout.val_dim_order = Some(order);
    }

    pub fn set_original_dim_orders(&mut self, orders: Vec<Vec<usize>>) {
        self.layout.original_dim_orders = Some(orders);
    }

    pub fn set_tosa_spatial_rank(&mut self, rank: usize) -> bool {
        if self.layout.tosa_spatial_rank == Some(rank) {
            return false;
        }
        self.layout.tosa_spatial_rank = Some(rank);
        true
    }

    pub fn tosa_spatial_rank_or(&self, default: usize) -> usize {
        self.layout.tosa_spatial_rank.unwrap_or(default)
    }

    pub fn tosa_spatial_rank(&self) -> Option<usize> {
        self.layout.tosa_spatial_rank
    }

    pub fn copy_dim_order_facts_from(&mut self, other: &Self) {
        self.layout.val_dim_order = other.layout.val_dim_order.clone();
        self.layout.tosa_dim_order = other.layout.tosa_dim_order.clone();
        self.layout.original_dim_orders = other.layout.original_dim_orders.clone();
    }

    pub fn tensor_facts(&self) -> Option<&TensorFacts> {
        self.tensor.as_ref()
    }

    pub fn set_tensor_facts(&mut self, facts: TensorFacts) {
        self.tensor = Some(facts);
    }

    pub fn set_val(&mut self, facts: TensorFacts) {
        self.set_tensor_facts(facts);
    }

    pub fn set_tensor_facts_for_existing_shape(&mut self, dtype: impl Into<String>) -> bool {
        let Some(shape) = self.val_shape().map(|shape| shape.to_vec()) else {
            return false;
        };
        self.set_tensor_facts(TensorFacts::new(shape, dtype));
        true
    }

    pub fn set_val_dtype_for_existing_shape(&mut self, dtype: impl Into<String>) -> bool {
        self.set_tensor_facts_for_existing_shape(dtype)
    }

    pub fn cloned_with_tensor_facts(&self, facts: TensorFacts) -> Self {
        let mut meta = self.clone();
        meta.set_tensor_facts(facts);
        meta
    }

    pub fn cloned_with_val(&self, shape: impl Into<Vec<i64>>, dtype: impl Into<String>) -> Self {
        self.cloned_with_tensor_facts(TensorFacts::new(shape, dtype))
    }

    pub fn cloned_with_same_dtype_or(&self, shape: impl Into<Vec<i64>>, default: &str) -> Self {
        self.cloned_with_val(shape, self.val_dtype_or(default))
    }

    pub fn cloned_with_same_dtype_or_float32(&self, shape: impl Into<Vec<i64>>) -> Self {
        self.cloned_with_same_dtype_or(shape, "torch.float32")
    }

    pub fn cloned_with_dtype_of_or(
        &self,
        shape: impl Into<Vec<i64>>,
        dtype_source: &Self,
        default: &str,
    ) -> Self {
        self.cloned_with_val(shape, dtype_source.val_dtype_or(default))
    }

    pub fn cloned_with_dtype_of_or_float32(
        &self,
        shape: impl Into<Vec<i64>>,
        dtype_source: &Self,
    ) -> Self {
        self.cloned_with_dtype_of_or(shape, dtype_source, "torch.float32")
    }

    pub fn without_qparams(mut self) -> Self {
        self.clear_all_qparams();
        self
    }

    pub fn output_qparam(&self) -> Option<&QParam> {
        self.quant.output_qparams.as_ref()?.get(&0)
    }

    pub fn input_qparam(&self) -> Option<&QParam> {
        self.input_qparam_at(0)
    }

    pub fn input_qparam_at(&self, index: usize) -> Option<&QParam> {
        self.quant.input_qparams.as_ref()?.get(&index)
    }

    pub fn input_qparams(&self) -> Option<&QParamMap> {
        self.quant.input_qparams.as_ref()
    }

    pub fn output_qparams(&self) -> Option<&QParamMap> {
        self.quant.output_qparams.as_ref()
    }

    pub fn pre_pass_arg_qparams(&self) -> Option<&QParamMap> {
        self.quant.pre_pass_arg_qparams.as_ref()
    }

    pub fn cloned_input_qparams(&self) -> Option<QParamMap> {
        self.quant.input_qparams.clone()
    }

    pub fn cloned_nonempty_input_qparams_with(
        &self,
        extra_qparams: impl IntoIterator<Item = (usize, QParam)>,
    ) -> Option<QParamMap> {
        let mut qparams = self.cloned_input_qparams()?;
        if qparams.is_empty() {
            return None;
        }
        for (index, qparam) in extra_qparams {
            qparams.insert(index, qparam);
        }
        Some(qparams)
    }

    pub fn cloned_output_qparams(&self) -> Option<QParamMap> {
        self.quant.output_qparams.clone()
    }

    pub fn has_qparam_metadata(&self) -> bool {
        self.quant.input_qparams.is_some() || self.quant.output_qparams.is_some()
    }

    pub fn has_nonempty_qparam_metadata(&self) -> bool {
        self.quant
            .input_qparams
            .as_ref()
            .is_some_and(|qparams| !qparams.is_empty())
            || self
                .quant
                .output_qparams
                .as_ref()
                .is_some_and(|qparams| !qparams.is_empty())
    }

    pub fn has_nonempty_input_and_output_qparams(&self) -> bool {
        self.quant
            .input_qparams
            .as_ref()
            .is_some_and(|qparams| !qparams.is_empty())
            && self
                .quant
                .output_qparams
                .as_ref()
                .is_some_and(|qparams| !qparams.is_empty())
    }

    pub fn sorted_input_qparams(&self) -> Vec<(usize, QParam)> {
        self.quant
            .input_qparams
            .as_ref()
            .map(QParamMap::cloned_pairs)
            .unwrap_or_default()
    }

    pub fn copy_input_qparams_from(&mut self, other: &Self) -> bool {
        let Some(iqp) = other.quant.input_qparams.clone() else {
            return false;
        };
        self.set_input_qparams(iqp);
        true
    }

    pub fn copy_output_qparams_from(&mut self, other: &Self) -> bool {
        let Some(oqp) = other.quant.output_qparams.clone() else {
            return false;
        };
        self.set_output_qparams(oqp);
        true
    }

    pub fn copy_qparams_from(&mut self, other: &Self) {
        self.copy_input_qparams_from(other);
        self.copy_output_qparams_from(other);
    }

    pub fn clear_all_qparams(&mut self) {
        self.clear_input_qparams();
        self.clear_output_qparams();
    }

    pub fn clear_tosa_special_dtype(&mut self) -> bool {
        self.extras.remove("tosa_special_dtype").is_some()
    }

    pub fn clear_tosa_dim_order_if_rank_mismatch(&mut self, rank: usize) -> bool {
        if self
            .layout
            .tosa_dim_order
            .as_ref()
            .is_some_and(|order| order.len() != rank)
        {
            self.layout.tosa_dim_order = None;
            true
        } else {
            false
        }
    }

    pub fn set_empty_qparam_maps(&mut self) {
        self.set_input_qparams(QParamMap::new());
        self.set_output_qparams(QParamMap::new());
    }

    pub fn rescale_output_meta(&self, output_qparam: QParam) -> Self {
        self.rescale_output_meta_with_shape(self, output_qparam)
    }

    pub fn rescale_output_meta_with_shape(
        &self,
        shape_source: &Self,
        output_qparam: QParam,
    ) -> Self {
        let mut meta = self.clone();
        meta.set_empty_input_qparams();
        meta.set_single_output_qparam(output_qparam.clone());
        meta.clear_tosa_special_dtype();
        if let Some(shape) = shape_source.val_shape() {
            meta.set_tensor_facts(TensorFacts::new(shape.to_vec(), output_qparam.dtype_str));
        }
        meta
    }

    pub fn rescale_meta_with_input_qparam(
        &self,
        shape_source: &Self,
        input_qparam: QParam,
        output_qparam: QParam,
    ) -> Self {
        let mut meta = self.rescale_output_meta_with_shape(shape_source, output_qparam);
        meta.set_single_input_qparam(input_qparam);
        meta
    }

    pub fn set_extra(&mut self, key: impl Into<String>, value: MetaValue) {
        self.extras.insert(key.into(), value);
    }

    pub fn extra_bool(&self, key: &str) -> Option<bool> {
        match self.extras.get(key) {
            Some(MetaValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn extra_str(&self, key: &str) -> Option<&str> {
        match self.extras.get(key) {
            Some(MetaValue::Str(v)) => Some(v.as_str()),
            _ => None,
        }
    }
}

pub struct TensorMetaStamp<'a> {
    base: &'a NodeMeta,
    dtype: Cow<'a, str>,
}

impl<'a> TensorMetaStamp<'a> {
    pub fn dtype(&self) -> &str {
        self.dtype.as_ref()
    }

    pub fn with_dtype(&self, dtype: impl Into<Cow<'a, str>>) -> TensorMetaStamp<'a> {
        self.base.tensor_stamp(dtype)
    }

    fn facts(&self, shape: impl Into<Vec<i64>>) -> TensorFacts {
        TensorFacts::new(shape, self.dtype.as_ref())
    }

    pub fn fresh_tensor(&self, shape: impl Into<Vec<i64>>) -> NodeMeta {
        NodeMeta::tensor(shape, self.dtype.as_ref())
    }

    pub fn preserve_qparams(&self, shape: impl Into<Vec<i64>>) -> NodeMeta {
        self.base.cloned_with_tensor_facts(self.facts(shape))
    }

    pub fn drop_qparams(&self, shape: impl Into<Vec<i64>>) -> NodeMeta {
        self.base
            .cloned_with_tensor_facts(self.facts(shape))
            .without_qparams()
    }

    pub fn empty_qparam_maps(&self, shape: impl Into<Vec<i64>>) -> NodeMeta {
        let mut meta = self.base.cloned_with_tensor_facts(self.facts(shape));
        meta.set_empty_qparam_maps();
        meta
    }

    pub fn plain_tensor(&self, shape: impl Into<Vec<i64>>) -> NodeMeta {
        let mut meta = self.base.cloned_with_tensor_facts(self.facts(shape));
        meta.set_empty_input_qparams();
        meta.clear_output_qparams();
        meta.clear_tosa_special_dtype();
        meta
    }
}
