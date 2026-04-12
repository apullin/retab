use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::fx_fuse_constant_args::{self, ZeroInputConstantKind};
use crate::fx_graph::{meta_val, FxGraphModule};
use crate::graph::ir::DType;

/// Rust-planned, Python-applied ExportedProgram bookkeeping effects.
///
/// These effects intentionally stay narrow:
/// - constant-placeholder deletion
/// - graph-signature mapping edits
/// - int64 -> int32 state-tensor / FakeTensor updates
///
/// The intent is to let Rust IR passes plan EP mutations without hard-wiring
/// those passes to Python bookkeeping details.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxEpInputKind {
    Parameter,
    Buffer,
    LiftedTensorConstant,
}

impl FxEpInputKind {
    pub fn graph_signature_attr(self) -> &'static str {
        match self {
            Self::Parameter => "inputs_to_parameters",
            Self::Buffer => "inputs_to_buffers",
            Self::LiftedTensorConstant => "inputs_to_lifted_tensor_constants",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxEpTensorDType {
    Int32,
    Int64,
}

impl FxEpTensorDType {
    fn torch_attr(self) -> &'static str {
        match self {
            Self::Int32 => "int32",
            Self::Int64 => "int64",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Int32 => "torch.int32",
            Self::Int64 => "torch.int64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxEpGraphSignatureMappingUpdate {
    UpsertTarget,
    DropEntry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxDeleteConstantPlaceholderEffect {
    pub placeholder_name: String,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxGraphSignatureMappingEffect {
    pub placeholder_name: String,
    pub input_kind: FxEpInputKind,
    pub update: FxEpGraphSignatureMappingUpdate,
    pub target_name: Option<String>,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxUpdatePlaceholderTensorDtypeEffect {
    pub placeholder_name: String,
    pub from_dtype: Option<FxEpTensorDType>,
    pub to_dtype: FxEpTensorDType,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxCastInt64InputToInt32Effect {
    pub placeholder_name: String,
    pub input_kind: FxEpInputKind,
    pub target_name: Option<String>,
    pub allow_missing: bool,
    pub update_placeholder_meta: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxMaterializeZeroInputConstantEffect {
    pub node_name: String,
    pub kind: ZeroInputConstantKind,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxFuseConstantNodeEffect {
    pub node_name: String,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxMaterializeConstantNodeEffect {
    pub node_name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxMaterializeZeroInputConstantNodeEffect {
    pub node_name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxInsertTableOpEffect {
    pub node_name: String,
    pub kind: String,
    pub scalar_arg: Option<f64>,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FxInsertPrecomputedTableOpEffect {
    pub node_name: String,
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub table_dtype: DType,
    pub input_scale: f64,
    pub input_zp: i64,
    pub input_qmin: i64,
    pub input_qmax: i64,
    pub input_dtype: DType,
    pub output_scale: f64,
    pub output_zp: i64,
    pub output_qmin: i64,
    pub output_qmax: i64,
    pub output_dtype: DType,
    pub rescale_lshift: i64,
    pub allow_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FxEpEffect {
    DeleteConstantPlaceholder(FxDeleteConstantPlaceholderEffect),
    UpdateGraphSignatureMapping(FxGraphSignatureMappingEffect),
    UpdatePlaceholderTensorDtype(FxUpdatePlaceholderTensorDtypeEffect),
    CastInt64InputToInt32(FxCastInt64InputToInt32Effect),
    MaterializeZeroInputConstant(FxMaterializeZeroInputConstantEffect),
    FuseConstantNode(FxFuseConstantNodeEffect),
    MaterializeConstantNode(FxMaterializeConstantNodeEffect),
    MaterializeZeroInputConstantNode(FxMaterializeZeroInputConstantNodeEffect),
    InsertTableOp(FxInsertTableOpEffect),
    InsertPrecomputedTableOp(FxInsertPrecomputedTableOpEffect),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FxEpApplyStats {
    pub applied: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FxEpEffectBatch {
    pub effects: Vec<FxEpEffect>,
}

impl FxEpEffectBatch {
    pub fn push(&mut self, effect: FxEpEffect) {
        self.effects.push(effect);
    }

    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<FxEpApplyStats> {
        let mut stats = FxEpApplyStats::default();
        for effect in &self.effects {
            if effect.apply(ctx)? {
                stats.applied += 1;
            } else {
                stats.skipped += 1;
            }
        }
        Ok(stats)
    }
}

pub struct FxEpApplyContext<'py> {
    gm: FxGraphModule<'py>,
    exported_program: Bound<'py, PyAny>,
    torch: Bound<'py, PyAny>,
}

impl<'py> FxEpApplyContext<'py> {
    pub fn new(
        graph_module: &Bound<'py, PyAny>,
        exported_program: &Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        let py = graph_module.py();
        Ok(Self {
            gm: FxGraphModule::new(graph_module),
            exported_program: exported_program.clone(),
            torch: py.import_bound("torch")?.into_any(),
        })
    }

    pub fn gm(&self) -> &FxGraphModule<'py> {
        &self.gm
    }

    pub fn exported_program(&self) -> &Bound<'py, PyAny> {
        &self.exported_program
    }

    fn torch_dtype(&self, dtype: FxEpTensorDType) -> PyResult<Bound<'py, PyAny>> {
        self.torch.getattr(dtype.torch_attr())
    }
}

impl FxEpEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        match self {
            Self::DeleteConstantPlaceholder(effect) => effect.apply(ctx),
            Self::UpdateGraphSignatureMapping(effect) => effect.apply(ctx),
            Self::UpdatePlaceholderTensorDtype(effect) => effect.apply(ctx),
            Self::CastInt64InputToInt32(effect) => effect.apply(ctx),
            Self::MaterializeZeroInputConstant(effect) => effect.apply(ctx),
            Self::FuseConstantNode(effect) => effect.apply(ctx),
            Self::MaterializeConstantNode(effect) => effect.apply(ctx),
            Self::MaterializeZeroInputConstantNode(effect) => effect.apply(ctx),
            Self::InsertTableOp(effect) => effect.apply(ctx),
            Self::InsertPrecomputedTableOp(effect) => effect.apply(ctx),
        }
    }
}

impl FxDeleteConstantPlaceholderEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        if let Some(delete_by_name) =
            import_delete_constant_placeholder_by_name(ctx.exported_program.py())?
        {
            return delete_by_name
                .call1((
                    ctx.exported_program(),
                    self.placeholder_name.as_str(),
                    self.allow_missing,
                ))?
                .extract();
        }

        let Some(node) = find_node_by_name(ctx.gm(), &self.placeholder_name)? else {
            return if self.allow_missing {
                Ok(false)
            } else {
                Err(PyKeyError::new_err(format!(
                    "missing placeholder node '{}'",
                    self.placeholder_name
                )))
            };
        };
        let delete_helper = import_delete_constant_placeholder(ctx.exported_program.py())?;
        delete_helper.call1((ctx.exported_program(), &node))?;
        Ok(true)
    }
}

impl FxGraphSignatureMappingEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let py = ctx.exported_program.py();
        let mapping = graph_signature_mapping(ctx.exported_program(), self.input_kind)?;
        match self.update {
            FxEpGraphSignatureMappingUpdate::DropEntry => {
                if mapping.get_item(&self.placeholder_name).is_err() {
                    return if self.allow_missing {
                        Ok(false)
                    } else {
                        Err(PyKeyError::new_err(format!(
                            "missing graph-signature entry '{}' in {}",
                            self.placeholder_name,
                            self.input_kind.graph_signature_attr()
                        )))
                    };
                }
                mapping.call_method1("pop", (&self.placeholder_name, py.None()))?;
                Ok(true)
            }
            FxEpGraphSignatureMappingUpdate::UpsertTarget => {
                let target_name = self.target_name.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "missing target_name for graph-signature upsert '{}'",
                        self.placeholder_name
                    ))
                })?;
                if let Ok(existing) = mapping.get_item(&self.placeholder_name) {
                    let existing: String = existing.extract()?;
                    if existing == *target_name {
                        return Ok(false);
                    }
                }
                mapping.set_item(&self.placeholder_name, target_name)?;
                Ok(true)
            }
        }
    }
}

impl FxUpdatePlaceholderTensorDtypeEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let Some(node) = find_node_by_name(ctx.gm(), &self.placeholder_name)? else {
            return if self.allow_missing {
                Ok(false)
            } else {
                Err(PyKeyError::new_err(format!(
                    "missing placeholder node '{}'",
                    self.placeholder_name
                )))
            };
        };
        let meta = node.getattr("meta")?;
        let Some(val) = meta_val(&meta, "val") else {
            return Ok(false);
        };
        if let Some(expected_from) = self.from_dtype {
            let current_dtype = val.getattr("dtype")?;
            let expected_dtype = ctx.torch_dtype(expected_from)?;
            if !current_dtype.eq(&expected_dtype)? {
                return Ok(false);
            }
        }
        let target_dtype = ctx.torch_dtype(self.to_dtype)?;
        let current_dtype = val.getattr("dtype")?;
        if current_dtype.eq(&target_dtype)? {
            return Ok(false);
        }
        let new_val = val.call_method1("to", (&target_dtype,))?;
        meta.set_item("val", &new_val)?;
        Ok(true)
    }
}

impl FxCastInt64InputToInt32Effect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let target_name = match self.target_name.clone().map(Ok).unwrap_or_else(|| {
            resolve_graph_signature_target(
                ctx.exported_program(),
                self.input_kind,
                &self.placeholder_name,
            )
        }) {
            Ok(name) => name,
            Err(err) if self.allow_missing => return Ok(false),
            Err(err) => return Err(err),
        };

        let container =
            exported_tensor_container(ctx.exported_program(), self.input_kind, &target_name)?;
        let tensor = match container.get_item(&target_name) {
            Ok(tensor) => tensor,
            Err(err) if self.allow_missing => return Ok(false),
            Err(err) => return Err(err),
        };

        let int64_dtype = ctx.torch_dtype(FxEpTensorDType::Int64)?;
        let int32_dtype = ctx.torch_dtype(FxEpTensorDType::Int32)?;
        let tensor_dtype = tensor.getattr("dtype")?;
        let mut modified = false;

        if tensor_dtype.eq(&int64_dtype)? {
            ensure_tensor_fits_i32(&ctx.torch, &tensor, &self.placeholder_name)?;
            let tensor_i32 = tensor.call_method1("to", (&int32_dtype,))?;
            container.set_item(&target_name, &tensor_i32)?;
            modified = true;
        } else if !tensor_dtype.eq(&int32_dtype)? {
            return Err(PyRuntimeError::new_err(format!(
                "input '{}' expected {} or {}, found {}",
                self.placeholder_name,
                FxEpTensorDType::Int64.display_name(),
                FxEpTensorDType::Int32.display_name(),
                tensor_dtype.str()?.to_str()?
            )));
        }

        if self.update_placeholder_meta {
            let meta_modified = FxUpdatePlaceholderTensorDtypeEffect {
                placeholder_name: self.placeholder_name.clone(),
                from_dtype: None,
                to_dtype: FxEpTensorDType::Int32,
                allow_missing: self.allow_missing,
            }
            .apply(ctx)?;
            modified |= meta_modified;
        }

        Ok(modified)
    }
}

impl FxMaterializeZeroInputConstantEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let Some(node) = find_node_by_name(ctx.gm(), &self.node_name)? else {
            return if self.allow_missing {
                Ok(false)
            } else {
                Err(PyKeyError::new_err(format!(
                    "missing zero-input constant node '{}'",
                    self.node_name
                )))
            };
        };
        fx_fuse_constant_args::fold_zero_input_constant(ctx.gm(), &node, self.kind)
    }
}

impl FxFuseConstantNodeEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let Some(node) = find_node_by_name(ctx.gm(), &self.node_name)? else {
            return if self.allow_missing {
                Ok(false)
            } else {
                Err(PyKeyError::new_err(format!(
                    "missing constant-fold node '{}'",
                    self.node_name
                )))
            };
        };
        import_fuse_constant_node_helper(ctx.exported_program.py())?
            .call1((ctx.gm().gm_ref(), ctx.exported_program(), &node))?
            .extract()
    }
}

impl FxMaterializeConstantNodeEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let Some(node) = find_node_by_name(ctx.gm(), &self.node_name)? else {
            return if self.allow_missing {
                Ok(false)
            } else {
                Err(PyKeyError::new_err(format!(
                    "missing constant-materialize node '{}'",
                    self.node_name
                )))
            };
        };
        let helpers = ctx.exported_program.py().import_bound("fx_helpers")?;
        let py_shape = pyo3::types::PyList::new_bound(ctx.exported_program.py(), &self.shape);
        let py_bytes = PyBytes::new_bound(ctx.exported_program.py(), &self.data);
        helpers.call_method1(
            "materialize_constant_node",
            (
                ctx.gm().gm_ref(),
                ctx.exported_program(),
                &node,
                py_bytes,
                py_shape,
                ir_dtype_to_torch_name(self.dtype),
            ),
        )?;
        Ok(true)
    }
}

impl FxMaterializeZeroInputConstantNodeEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let Some(node) = find_node_by_name(ctx.gm(), &self.node_name)? else {
            return if self.allow_missing {
                Ok(false)
            } else {
                Err(PyKeyError::new_err(format!(
                    "missing zero-input constant-materialize node '{}'",
                    self.node_name
                )))
            };
        };
        let helpers = ctx.exported_program.py().import_bound("fx_helpers")?;
        let py_shape = pyo3::types::PyList::new_bound(ctx.exported_program.py(), &self.shape);
        let py_bytes = PyBytes::new_bound(ctx.exported_program.py(), &self.data);
        helpers.call_method1(
            "materialize_zero_input_constant_node",
            (
                ctx.gm().gm_ref(),
                &node,
                py_bytes,
                py_shape,
                ir_dtype_to_torch_name(self.dtype),
            ),
        )?;
        Ok(true)
    }
}

impl FxInsertTableOpEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let Some(node) = find_node_by_name(ctx.gm(), &self.node_name)? else {
            return if self.allow_missing {
                Ok(false)
            } else {
                Err(PyKeyError::new_err(format!(
                    "missing table-op node '{}'",
                    self.node_name
                )))
            };
        };
        let helpers = ctx.exported_program.py().import_bound("fx_helpers")?;
        helpers.call_method1(
            "insert_table_op",
            (
                ctx.gm().gm_ref(),
                &node,
                self.kind.as_str(),
                self.scalar_arg,
            ),
        )?;
        Ok(true)
    }
}

impl FxInsertPrecomputedTableOpEffect {
    pub fn apply<'py>(&self, ctx: &FxEpApplyContext<'py>) -> PyResult<bool> {
        let Some(node) = find_node_by_name(ctx.gm(), &self.node_name)? else {
            return if self.allow_missing {
                Ok(false)
            } else {
                Err(PyKeyError::new_err(format!(
                    "missing precomputed table-op node '{}'",
                    self.node_name
                )))
            };
        };
        let helpers = ctx.exported_program.py().import_bound("fx_helpers")?;
        let py_shape = pyo3::types::PyList::new_bound(ctx.exported_program.py(), &self.shape);
        let py_bytes = PyBytes::new_bound(ctx.exported_program.py(), &self.data);
        let input_qspec = (
            self.input_scale,
            self.input_zp,
            self.input_qmin,
            self.input_qmax,
            ir_dtype_to_torch_name(self.input_dtype),
        );
        let output_qspec = (
            self.output_scale,
            self.output_zp,
            self.output_qmin,
            self.output_qmax,
            ir_dtype_to_torch_name(self.output_dtype),
        );
        helpers.call_method1(
            "insert_precomputed_table_op",
            (
                ctx.gm().gm_ref(),
                &node,
                py_bytes,
                py_shape,
                ir_dtype_to_torch_name(self.table_dtype),
                input_qspec,
                output_qspec,
                self.rescale_lshift,
            ),
        )?;
        Ok(true)
    }
}

fn find_node_by_name<'py>(
    gm: &FxGraphModule<'py>,
    target_name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    for node in gm.nodes()? {
        if FxGraphModule::node_name(&node)? == target_name {
            return Ok(Some(node));
        }
    }
    Ok(None)
}

fn ir_dtype_to_torch_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "torch.float32",
        DType::Float16 => "torch.float16",
        DType::BFloat16 => "torch.bfloat16",
        DType::Int8 => "torch.int8",
        DType::Int16 => "torch.int16",
        DType::Int32 => "torch.int32",
        DType::Int64 => "torch.int64",
        DType::Bool => "torch.bool",
        DType::Int4 => "torch.int8",
        DType::Int48 => "torch.int32",
    }
}

fn import_delete_constant_placeholder<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(helpers) = py.import_bound("fx_helpers") {
        if let Ok(helper) = helpers.getattr("delete_constant_placeholder") {
            return Ok(helper);
        }
    }
    py.import_bound("executorch.backends.transforms.utils")?
        .getattr("delete_constant_placeholder")
}

fn import_delete_constant_placeholder_by_name<'py>(
    py: Python<'py>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(helpers) = py.import_bound("fx_helpers") {
        if let Ok(helper) = helpers.getattr("delete_constant_placeholder_by_name") {
            return Ok(Some(helper));
        }
    }
    Ok(None)
}

fn import_fuse_constant_node_helper<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(helpers) = py.import_bound("fx_helpers") {
        if let Ok(helper) = helpers.getattr("fuse_constant_node") {
            return Ok(helper);
        }
    }
    Err(PyRuntimeError::new_err(
        "fx_helpers.fuse_constant_node helper is unavailable",
    ))
}

fn graph_signature_mapping<'py>(
    exported_program: &Bound<'py, PyAny>,
    input_kind: FxEpInputKind,
) -> PyResult<Bound<'py, PyAny>> {
    exported_program
        .getattr("graph_signature")?
        .getattr(input_kind.graph_signature_attr())
}

fn resolve_graph_signature_target<'py>(
    exported_program: &Bound<'py, PyAny>,
    input_kind: FxEpInputKind,
    placeholder_name: &str,
) -> PyResult<String> {
    graph_signature_mapping(exported_program, input_kind)?
        .get_item(placeholder_name)?
        .extract()
}

fn exported_tensor_container<'py>(
    exported_program: &Bound<'py, PyAny>,
    input_kind: FxEpInputKind,
    target_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match input_kind {
        FxEpInputKind::Parameter => exported_program.getattr("state_dict"),
        FxEpInputKind::LiftedTensorConstant => exported_program.getattr("constants"),
        FxEpInputKind::Buffer => {
            let graph_signature = exported_program.getattr("graph_signature")?;
            let non_persistent = graph_signature.getattr("non_persistent_buffers")?;
            let is_non_persistent: bool = non_persistent
                .call_method1("__contains__", (target_name,))?
                .extract()?;
            if is_non_persistent {
                exported_program.getattr("constants")
            } else {
                exported_program.getattr("state_dict")
            }
        }
    }
}

fn ensure_tensor_fits_i32(
    torch: &Bound<'_, PyAny>,
    tensor: &Bound<'_, PyAny>,
    placeholder_name: &str,
) -> PyResult<()> {
    let min_val: i64 = torch
        .call_method1("min", (tensor,))?
        .call_method0("item")?
        .extract()?;
    let max_val: i64 = torch
        .call_method1("max", (tensor,))?
        .call_method0("item")?
        .extract()?;
    let i32_min: i64 = -2_147_483_648;
    let i32_max: i64 = 2_147_483_647;
    if min_val < i32_min || max_val > i32_max {
        return Err(PyRuntimeError::new_err(format!(
            "input '{}' has values outside int32 range: min={}, max={}",
            placeholder_name, min_val, max_val
        )));
    }
    Ok(())
}
