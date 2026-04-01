use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct decomposition of ops that TOSA cannot represent directly.
///
/// Each op is decomposed into primitive ATen ops via PyO3 calls to the FX graph
/// API. No Python helper functions are used for graph manipulation -- all node
/// creation goes through `graph.call_function`.
///
/// Supported decompositions:
///   - atanh(x)       -> 0.5 * log((1+x)/(1-x))
///   - hardsigmoid(x) -> clamp((x + 3) / 6, 0, 1)
///   - stack          -> unsqueeze each + cat
///   - unbind         -> slice + squeeze for each element
///
/// Note: `masked_fill` is handled separately by `FxDecomposeMaskedFillPass`
/// (runs after fold_qdq so the fill value can be quantized correctly).
pub struct FxDecomposeOpsPass;

fn is_atanh(target: &str) -> bool {
    target.contains("atanh.default")
}

fn is_hardsigmoid(target: &str) -> bool {
    target.contains("hardsigmoid.default")
}

fn is_stack(target: &str) -> bool {
    target.contains("stack.default")
}

fn is_unbind(target: &str) -> bool {
    target.contains("unbind.int") || target.contains("unbind_copy.int")
}

fn is_dequantize(target: &str) -> bool {
    target.contains("dequantize_per_tensor") || target.contains("dequantize_per_channel")
}

fn is_quantize(target: &str) -> bool {
    target.contains("quantize_per_tensor") || target.contains("quantize_per_channel")
}

fn has_quantized_envelope(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    let Some(input) = args.first() else {
        return Ok(false);
    };
    if !FxGraphModule::is_node(input) {
        return Ok(false);
    }
    let input_target = FxGraphModule::node_target_str(input)?;
    if !is_dequantize(&input_target) {
        return Ok(false);
    }

    let users = FxGraphModule::node_users(node)?;
    Ok(users.iter().any(|user| {
        FxGraphModule::node_target_str(user)
            .map(|target| is_quantize(&target))
            .unwrap_or(false)
    }))
}

enum DecompKind {
    Atanh,
    Stack,
    Unbind,
}

/// Resolve an ATen op target, trying exir_ops.edge.aten first, then torch.ops.aten.
pub fn resolve_aten_op<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let parts: Vec<&str> = name.split('.').collect();
    let op_name = parts[0];
    let overload = if parts.len() > 1 { parts[1] } else { "default" };

    let result = py
        .import_bound("executorch.exir.dialects._ops")
        .and_then(|m| m.getattr("ops"))
        .and_then(|ops| ops.getattr("edge"))
        .and_then(|edge| edge.getattr("aten"))
        .and_then(|aten| aten.getattr(op_name))
        .and_then(|op| op.getattr(overload));

    match result {
        Ok(op) => Ok(op),
        Err(_) => py
            .import_bound("torch")?
            .getattr("ops")?
            .getattr("aten")?
            .getattr(op_name)?
            .getattr(overload),
    }
}

/// Resolve the TOSA GATHER op target: `backend.tosa.GATHER.default`.
pub fn resolve_tosa_gather_op<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.eval_bound(
        "getattr(getattr(getattr(getattr(__import__('executorch.exir.dialects._ops', fromlist=['ops']), 'ops'), 'backend'), 'tosa'), 'GATHER').default",
        None,
        None,
    )
}

/// Convert a Rust f64 into a Bound Python float.
pub fn py_float<'py>(py: Python<'py>, v: f64) -> PyResult<Bound<'py, PyAny>> {
    let builtins = py.import_bound("builtins")?;
    builtins.call_method1("float", (v,))
}

/// Convert a Rust i64 into a Bound Python int.
pub fn py_int<'py>(py: Python<'py>, v: i64) -> PyResult<Bound<'py, PyAny>> {
    let builtins = py.import_bound("builtins")?;
    builtins.call_method1("int", (v,))
}

/// Fluent builder for inserting `call_function` nodes before a reference node.
///
/// # Examples
///
/// ```ignore
/// // Simple case — meta copied from ref_node, qparams cleared:
/// let n = NodeInserter::before(py, &graph, &node).call(&op, &args)?;
///
/// // Custom meta source:
/// let n = NodeInserter::before(py, &graph, &node)
///     .meta_from(&other)
///     .call(&op, &args)?;
///
/// // Keep qparams and pass kwargs:
/// let n = NodeInserter::before(py, &graph, &node)
///     .keep_qparams()
///     .call_with_kwargs(&op, &args, &kw)?;
/// ```
#[derive(Clone)]
pub struct NodeInserter<'py> {
    py: Python<'py>,
    graph: Bound<'py, PyAny>,
    ref_node: Bound<'py, PyAny>,
    meta_src: Bound<'py, PyAny>,
    clear_qparams: bool,
}

impl<'py> NodeInserter<'py> {
    pub fn before(
        py: Python<'py>,
        graph: &Bound<'py, PyAny>,
        ref_node: &Bound<'py, PyAny>,
    ) -> Self {
        Self {
            py,
            graph: graph.clone(),
            ref_node: ref_node.clone(),
            meta_src: ref_node.clone(),
            clear_qparams: true,
        }
    }

    /// Copy meta from a specific source node (default: ref_node).
    pub fn meta_from(mut self, src: &Bound<'py, PyAny>) -> Self {
        self.meta_src = src.clone();
        self
    }

    /// Keep qparams from the meta source (default: clear them).
    pub fn keep_qparams(mut self) -> Self {
        self.clear_qparams = false;
        self
    }

    /// Insert a call_function node with the given target and args.
    pub fn call(
        &self,
        target: &Bound<'py, PyAny>,
        args: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call_impl(target, args, None)
    }

    /// Insert a call_function node with args and kwargs.
    pub fn call_with_kwargs(
        &self,
        target: &Bound<'py, PyAny>,
        args: &Bound<'py, PyAny>,
        kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call_impl(target, args, Some(kwargs))
    }

    fn call_impl(
        &self,
        target: &Bound<'py, PyAny>,
        args: &Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inserting = self.graph.call_method1("inserting_before", (&self.ref_node,))?;
        inserting.call_method0("__enter__")?;
        let new_node = if let Some(kw) = kwargs {
            self.graph.call_method1("call_function", (target, args, kw))?
        } else {
            self.graph.call_method1("call_function", (target, args))?
        };
        let copy_mod = self.py.import_bound("copy")?;
        let meta_copy = copy_mod.call_method1("copy", (self.meta_src.getattr("meta")?,))?;
        if self.clear_qparams {
            let _ = meta_copy.call_method1("__setitem__", ("input_qparams", PyDict::new_bound(self.py)));
            let _ = meta_copy.call_method1("__setitem__", ("output_qparams", PyDict::new_bound(self.py)));
        }
        new_node.setattr("meta", &meta_copy)?;
        let _ = refresh_fake_tensor(self.py, &new_node);
        inserting.call_method("__exit__", (self.py.None(), self.py.None(), self.py.None()), None)?;
        Ok(new_node)
    }
}

/// Propagate qparams from a source node to a target node.
/// Sets input_qparams and output_qparams on the target's meta.
/// For intermediate nodes in a decomposition, use the source's qparams.
pub fn propagate_qparams<'py>(src: &Bound<'py, PyAny>, dst: &Bound<'py, PyAny>) -> PyResult<()> {
    let src_meta = src.getattr("meta")?;
    let dst_meta = dst.getattr("meta")?;

    // Copy input_qparams if available
    if let Ok(iqp) = src_meta.call_method1("get", ("input_qparams",)) {
        if !iqp.is_none() {
            dst_meta.set_item("input_qparams", &iqp)?;
        }
    }
    // Copy output_qparams if available
    if let Ok(oqp) = src_meta.call_method1("get", ("output_qparams",)) {
        if !oqp.is_none() {
            dst_meta.set_item("output_qparams", &oqp)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FxDecomposer — fluent DSL for writing op decompositions
// ---------------------------------------------------------------------------

/// Shorthand for a node reference in the FX graph.
pub type N<'py> = Bound<'py, PyAny>;

/// Fluent DSL for decomposing a single FX node into primitive ATen ops.
///
/// Wraps a [`NodeInserter`] and the decomposition context (graph, gm, original
/// node) so that every emitted op is a single method call instead of 5-7 lines
/// of resolve + build-tuple + call.
///
/// # Example
/// ```ignore
/// let d = FxDecomposer::new(py, &graph, gm, &node);
/// let sum = d.add(&a, &b)?;
/// d.replace(&sum)?;
/// ```
pub struct FxDecomposer<'py> {
    py: Python<'py>,
    graph: Bound<'py, PyAny>,
    ins: NodeInserter<'py>,
    original: Bound<'py, PyAny>,
    gm: &'py FxGraphModule<'py>,
}

impl<'py> FxDecomposer<'py> {
    pub fn new(
        py: Python<'py>,
        graph: &Bound<'py, PyAny>,
        gm: &'py FxGraphModule<'py>,
        node: &Bound<'py, PyAny>,
    ) -> Self {
        Self {
            py,
            graph: graph.clone(),
            ins: NodeInserter::before(py, graph, node),
            original: node.clone(),
            gm,
        }
    }

    // -- internal helpers ---------------------------------------------------

    fn resolve(&self, op_name: &str) -> PyResult<Bound<'py, PyAny>> {
        resolve_aten_op(self.py, op_name)
    }

    /// Return an inserter whose meta source is overridden to `src`.
    fn ins_from(&self, src: &Bound<'py, PyAny>) -> NodeInserter<'py> {
        self.ins.clone().meta_from(src)
    }

    /// Build a `PyList` of i64 values (used for shapes, dims, repeats).
    fn py_list(&self, vals: &[i64]) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new_bound(self.py, vals))
    }

    /// Build kwargs dict with a single `dtype` entry.
    fn dtype_kwargs(&self, dtype: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        let kw = PyDict::new_bound(self.py);
        kw.set_item("dtype", dtype)?;
        Ok(kw)
    }

    // =====================================================================
    // Shape ops
    // =====================================================================

    /// `view_copy.default(x, shape)`
    pub fn view(&self, x: &N<'py>, shape: &[i64]) -> PyResult<N<'py>> {
        let op = self.resolve("view_copy.default")?;
        let shape_list = self.py_list(shape)?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref(), shape_list.as_ref()]);
        self.ins_from(x).call(&op, args.as_ref())
    }

    /// `unsqueeze_copy.default(x, dim)`
    pub fn unsqueeze(&self, x: &N<'py>, dim: i64) -> PyResult<N<'py>> {
        let op = self.resolve("unsqueeze_copy.default")?;
        let dim_py = py_int(self.py, dim)?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref(), dim_py.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `permute_copy.default(x, dims)`
    pub fn permute(&self, x: &N<'py>, dims: &[i64]) -> PyResult<N<'py>> {
        let op = self.resolve("permute_copy.default")?;
        let dims_list = self.py_list(dims)?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref(), dims_list.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `expand_copy.default(x, shape)`
    pub fn expand(&self, x: &N<'py>, shape: &[i64]) -> PyResult<N<'py>> {
        let op = self.resolve("expand_copy.default")?;
        let shape_list = self.py_list(shape)?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref(), shape_list.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `squeeze_copy.dims(x, [dim])`
    pub fn squeeze(&self, x: &N<'py>, dim: i64) -> PyResult<N<'py>> {
        let op = self.resolve("squeeze_copy.dims")?;
        let dims_list = self.py_list(&[dim])?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref(), dims_list.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    // =====================================================================
    // Arithmetic ops
    // =====================================================================

    /// `add.Tensor(a, b)`
    pub fn add(&self, a: &N<'py>, b: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("add.Tensor")?;
        let args = PyTuple::new_bound(self.py, &[a.as_ref(), b.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `sub.Tensor(a, b)`
    pub fn sub(&self, a: &N<'py>, b: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("sub.Tensor")?;
        let args = PyTuple::new_bound(self.py, &[a.as_ref(), b.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `mul.Tensor(a, b)`
    pub fn mul(&self, a: &N<'py>, b: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("mul.Tensor")?;
        let args = PyTuple::new_bound(self.py, &[a.as_ref(), b.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `div.Tensor(a, b)`
    pub fn div(&self, a: &N<'py>, b: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("div.Tensor")?;
        let args = PyTuple::new_bound(self.py, &[a.as_ref(), b.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `add.Scalar(x, val)`
    pub fn add_scalar(&self, x: &N<'py>, val: i64) -> PyResult<N<'py>> {
        let op = self.resolve("add.Scalar")?;
        let val_py = py_int(self.py, val)?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref(), val_py.as_ref()]);
        self.ins_from(x).call(&op, args.as_ref())
    }

    // =====================================================================
    // Constant creation ops
    // =====================================================================

    /// `full.default(shape, val, dtype=dtype)`
    pub fn full(
        &self,
        shape: &[i64],
        val: f64,
        dtype: &Bound<'py, PyAny>,
    ) -> PyResult<N<'py>> {
        let op = self.resolve("full.default")?;
        let shape_list = self.py_list(shape)?;
        let val_py = py_float(self.py, val)?;
        let args = PyTuple::new_bound(self.py, &[shape_list.as_ref(), val_py.as_ref()]);
        let kw = self.dtype_kwargs(dtype)?;
        self.ins.call_with_kwargs(&op, args.as_ref(), &kw)
    }

    /// `arange.start_step(start, end, dtype=dtype)`
    pub fn arange(
        &self,
        start: i64,
        end: i64,
        dtype: &Bound<'py, PyAny>,
    ) -> PyResult<N<'py>> {
        let op = self.resolve("arange.start_step")?;
        let start_py = start.into_py(self.py).into_bound(self.py);
        let end_py = end.into_py(self.py).into_bound(self.py);
        let args = PyTuple::new_bound(self.py, &[start_py.as_ref(), end_py.as_ref()]);
        let kw = self.dtype_kwargs(dtype)?;
        self.ins.call_with_kwargs(&op, args.as_ref(), &kw)
    }

    /// `scalar_tensor.default(val, dtype=dtype)`
    pub fn scalar_tensor(
        &self,
        val: &Bound<'py, PyAny>,
        dtype: &Bound<'py, PyAny>,
    ) -> PyResult<N<'py>> {
        let op = self.resolve("scalar_tensor.default")?;
        let args = PyTuple::new_bound(self.py, &[val.as_ref()]);
        let kw = self.dtype_kwargs(dtype)?;
        self.ins.call_with_kwargs(&op, args.as_ref(), &kw)
    }

    /// Build a quantization-aware zero scalar for use as the `where` false-
    /// branch in tril/triu decompositions.  When the node has output qparams
    /// the zero-point is clamped to [qmin, qmax] and the quantized dtype is
    /// used; otherwise the input tensor's dtype is forwarded.
    pub fn scalar_tensor_quantized(
        &self,
        node: &N<'py>,
        input_tensor: &N<'py>,
        debug: bool,
    ) -> PyResult<N<'py>> {
        let builtins = self.py.import_bound("builtins")?;
        let kw = PyDict::new_bound(self.py);

        if crate::fx_utils::is_quantized(node) {
            if let Some(qparams) = crate::fx_utils::get_output_qparam(node) {
                let zp: i64 = qparams
                    .getattr("zp")
                    .or_else(|_| qparams.getattr("zero_point"))
                    .and_then(|v| v.extract())
                    .unwrap_or(0);
                let qmin: i64 = qparams
                    .getattr("qmin")
                    .and_then(|v| v.extract())
                    .unwrap_or(-128);
                let qmax: i64 = qparams
                    .getattr("qmax")
                    .and_then(|v| v.extract())
                    .unwrap_or(127);
                let output_dtype = qparams.getattr("dtype")?;
                let clamped = zp.max(qmin).min(qmax);
                if debug {
                    eprintln!(
                        "[FxDecomposer] quantizing zero fill -> {} (zp={}, qmin={}, qmax={})",
                        clamped, zp, qmin, qmax
                    );
                }
                kw.set_item("dtype", output_dtype)?;
                let val = builtins.call_method1("int", (clamped,))?;
                let op = self.resolve("scalar_tensor.default")?;
                let args = PyTuple::new_bound(self.py, &[val.as_ref()]);
                return self.ins.call_with_kwargs(&op, args.as_ref(), &kw);
            }
        }

        let input_dtype = input_tensor
            .getattr("meta")?
            .get_item("val")?
            .getattr("dtype")?;
        kw.set_item("dtype", &input_dtype)?;
        let val = builtins.call_method1("int", (0,))?;
        let op = self.resolve("scalar_tensor.default")?;
        let args = PyTuple::new_bound(self.py, &[val.as_ref()]);
        self.ins.call_with_kwargs(&op, args.as_ref(), &kw)
    }

    // =====================================================================
    // Selection / comparison ops
    // =====================================================================

    /// `where.self(cond, on_true, on_false)`
    pub fn where_(
        &self,
        cond: &N<'py>,
        on_true: &N<'py>,
        on_false: &N<'py>,
    ) -> PyResult<N<'py>> {
        let op = self.resolve("where.self")?;
        let args = PyTuple::new_bound(
            self.py,
            &[cond.as_ref(), on_true.as_ref(), on_false.as_ref()],
        );
        self.ins.call(&op, args.as_ref())
    }

    /// `slice_copy.Tensor(x, dim, start, end)`
    pub fn slice(&self, x: &N<'py>, dim: i64, start: i64, end: i64) -> PyResult<N<'py>> {
        let op = self.resolve("slice_copy.Tensor")?;
        let dim_py = py_int(self.py, dim)?;
        let start_py = py_int(self.py, start)?;
        let end_py = py_int(self.py, end)?;
        let args = PyTuple::new_bound(
            self.py,
            &[x.as_ref(), dim_py.as_ref(), start_py.as_ref(), end_py.as_ref()],
        );
        self.ins.call(&op, args.as_ref())
    }

    /// `eq.Tensor(a, b)`
    pub fn eq(&self, a: &N<'py>, b: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("eq.Tensor")?;
        let args = PyTuple::new_bound(self.py, &[a.as_ref(), b.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `ge.Tensor(a, b)`
    pub fn ge(&self, a: &N<'py>, b: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("ge.Tensor")?;
        let args = PyTuple::new_bound(self.py, &[a.as_ref(), b.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    // =====================================================================
    // Other ops
    // =====================================================================

    /// `cat.default([tensors], dim)`
    pub fn cat(&self, tensors: &[&N<'py>], dim: i64) -> PyResult<N<'py>> {
        let op = self.resolve("cat.default")?;
        let dim_py = py_int(self.py, dim)?;
        let tensor_list = PyList::new_bound(self.py, tensors);
        let args = PyTuple::new_bound(self.py, &[tensor_list.as_ref(), dim_py.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `_to_copy.default(x, dtype=dtype)`
    pub fn cast(&self, x: &N<'py>, dtype: &Bound<'py, PyAny>) -> PyResult<N<'py>> {
        let op = self.resolve("_to_copy.default")?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref()]);
        let kw = self.dtype_kwargs(dtype)?;
        self.ins.call_with_kwargs(&op, args.as_ref(), &kw)
    }

    /// `log.default(x)`
    pub fn log(&self, x: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("log.default")?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `reciprocal.default(x)`
    pub fn reciprocal(&self, x: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("reciprocal.default")?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `repeat.default(x, repeats)`
    pub fn repeat(&self, x: &N<'py>, repeats: &[i64]) -> PyResult<N<'py>> {
        let op = self.resolve("repeat.default")?;
        let repeats_list = self.py_list(repeats)?;
        let args = PyTuple::new_bound(self.py, &[x.as_ref(), repeats_list.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `bitwise_and.Tensor(a, b)`
    pub fn bitwise_and(&self, a: &N<'py>, b: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("bitwise_and.Tensor")?;
        let args = PyTuple::new_bound(self.py, &[a.as_ref(), b.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `clamp.default(x, min, max)`
    pub fn clamp(&self, x: &N<'py>, min_val: f64, max_val: f64) -> PyResult<N<'py>> {
        let op = self.resolve("clamp.default")?;
        let min_py = py_float(self.py, min_val)?;
        let max_py = py_float(self.py, max_val)?;
        let args = PyTuple::new_bound(
            self.py,
            &[x.as_ref(), min_py.as_ref(), max_py.as_ref()],
        );
        self.ins.call(&op, args.as_ref())
    }

    /// `mm.default(a, b)`
    pub fn mm(&self, a: &N<'py>, b: &N<'py>) -> PyResult<N<'py>> {
        let op = self.resolve("mm.default")?;
        let args = PyTuple::new_bound(self.py, &[a.as_ref(), b.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `full.default(shape, val)` — like [`full`] but without a dtype kwarg.
    ///
    /// Useful for broadcast scalars in arithmetic decompositions where the
    /// dtype is inferred by the downstream op.
    pub fn full_scalar(&self, rank: usize, val: f64) -> PyResult<N<'py>> {
        let op = self.resolve("full.default")?;
        let shape: Vec<i64> = vec![1; rank];
        let shape_tuple = PyTuple::new_bound(self.py, shape);
        let val_py = py_float(self.py, val)?;
        let args = PyTuple::new_bound(self.py, &[shape_tuple.as_ref(), val_py.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    /// `convolution.default(input, weight, bias, stride, padding, dilation,
    ///   transposed, output_padding, groups)`
    pub fn convolution(
        &self,
        input: &N<'py>,
        weight: &N<'py>,
        bias: Option<&N<'py>>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        transposed: bool,
        output_padding: &[i64],
        groups: i64,
    ) -> PyResult<N<'py>> {
        let op = self.resolve("convolution.default")?;
        let bias_py: Bound<'py, PyAny> = match bias {
            Some(b) => b.clone(),
            None => self.py.None().into_bound(self.py),
        };
        let stride_py = self.py_list(stride)?;
        let padding_py = self.py_list(padding)?;
        let dilation_py = self.py_list(dilation)?;
        let transposed_py = transposed.into_py(self.py).into_bound(self.py);
        let output_padding_py = self.py_list(output_padding)?;
        let groups_py = groups.into_py(self.py).into_bound(self.py);
        let args = PyTuple::new_bound(
            self.py,
            &[
                input.as_ref(),
                weight.as_ref(),
                bias_py.as_ref(),
                stride_py.as_ref(),
                padding_py.as_ref(),
                dilation_py.as_ref(),
                transposed_py.as_ref(),
                output_padding_py.as_ref(),
                groups_py.as_ref(),
            ],
        );
        self.ins.clone().keep_qparams().call(&op, args.as_ref())
    }

    /// `full.default(shape, val, dtype=dtype, device=device)` — variant that
    /// accepts optional device kwargs in addition to dtype.
    pub fn full_with_device(
        &self,
        shape: &[i64],
        val: f64,
        dtype: Option<&Bound<'py, PyAny>>,
        device: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<N<'py>> {
        let op = self.resolve("full.default")?;
        let shape_list = self.py_list(shape)?;
        let val_py = py_float(self.py, val)?;
        let args = PyTuple::new_bound(self.py, &[shape_list.as_ref(), val_py.as_ref()]);
        let kw = PyDict::new_bound(self.py);
        if let Some(dt) = dtype {
            kw.set_item("dtype", dt)?;
        }
        if let Some(dev) = device {
            kw.set_item("device", dev)?;
        }
        self.ins.call_with_kwargs(&op, args.as_ref(), &kw)
    }

    /// `full.default(shape, val, dtype=dtype)` where val is an integer.
    pub fn full_int(
        &self,
        shape: &[i64],
        val: i64,
        dtype: &Bound<'py, PyAny>,
    ) -> PyResult<N<'py>> {
        let op = self.resolve("full.default")?;
        let shape_tuple = PyTuple::new_bound(self.py, shape);
        let val_py = val.into_py(self.py).into_bound(self.py);
        let args = PyTuple::new_bound(self.py, &[shape_tuple.as_ref(), val_py.as_ref()]);
        let kw = self.dtype_kwargs(dtype)?;
        self.ins.call_with_kwargs(&op, args.as_ref(), &kw)
    }

    // =====================================================================
    // TOSA backend ops
    // =====================================================================

    /// `backend.tosa.GATHER.default(values, indices)`
    pub fn gather(&self, values: &N<'py>, indices: &N<'py>) -> PyResult<N<'py>> {
        let op = resolve_tosa_gather_op(self.py)?;
        let args = PyTuple::new_bound(self.py, &[values.as_ref(), indices.as_ref()]);
        self.ins.call(&op, args.as_ref())
    }

    // =====================================================================
    // Utilities
    // =====================================================================

    /// Access the Python interpreter token.
    pub fn py(&self) -> Python<'py> {
        self.py
    }

    /// Access the underlying inserter for advanced use cases.
    pub fn inserter(&self) -> &NodeInserter<'py> {
        &self.ins
    }

    /// The original node being decomposed.
    pub fn original(&self) -> &Bound<'py, PyAny> {
        &self.original
    }

    /// Propagate qparams from the original node to `dst`.
    pub fn propagate_qparams(&self, dst: &N<'py>) -> PyResult<()> {
        propagate_qparams(&self.original, dst)
    }

    /// Set `meta["val"]` on `node` to a new empty tensor with the given shape,
    /// inheriting dtype from `template`.
    pub fn set_meta_val_shape(
        &self,
        template: &N<'py>,
        node: &N<'py>,
        shape: &[i64],
    ) -> PyResult<()> {
        let template_meta = template.getattr("meta")?;
        let template_val = template_meta.get_item("val")?;
        if !template_val.hasattr("new_empty")? {
            return Ok(());
        }
        let dtype = template_val.getattr("dtype")?;
        let kwargs = PyDict::new_bound(self.py);
        kwargs.set_item("dtype", dtype)?;
        let new_val = template_val.call_method("new_empty", (shape.to_vec(),), Some(&kwargs))?;
        node.getattr("meta")?.set_item("val", new_val)?;
        Ok(())
    }

    // =====================================================================
    // Finalize: replace the original node with the decomposition result
    // =====================================================================

    /// Copy meta from the original node to `result`, refresh its FakeTensor,
    /// replace all uses of the original node, then erase it.
    pub fn replace(&self, result: &N<'py>) -> PyResult<bool> {
        crate::fx_utils::copy_meta_and_refresh(self.py, &self.original, result)?;
        FxGraphModule::replace_all_uses_with(&self.original, result)?;
        self.gm.erase_node(&self.original)?;
        Ok(true)
    }
}

/// Try to refresh a node's meta["val"] from its target and args.
fn refresh_fake_tensor(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<()> {
    let helpers = py.import_bound("fx_helpers")?;
    helpers.call_method1("refresh_node_fake_tensor", (node,))?;
    Ok(())
}

impl FxPass for FxDecomposeOpsPass {
    fn name(&self) -> &str {
        "FxDecomposeOps"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let debug = FxGraphModule::debug();
        let mut modified = false;

        let all_nodes = gm.nodes()?;
        let mut candidates: Vec<(Bound<'_, PyAny>, DecompKind)> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;

            if is_atanh(&target) {
                if has_quantized_envelope(node)? {
                    continue;
                }
                candidates.push((node.clone(), DecompKind::Atanh));
            } else if is_hardsigmoid(&target) {
                // Skip hardsigmoid decomposition for quantized graphs —
                // InsertTableOpsPass handles it via TABLE lookup instead.
                // The arithmetic decomposition (x+3)/6 causes INT32/FP32
                // dtype mismatches in the quantized pipeline.
            } else if is_stack(&target) {
                candidates.push((node.clone(), DecompKind::Stack));
            } else if is_unbind(&target) {
                candidates.push((node.clone(), DecompKind::Unbind));
            }
        }

        if candidates.is_empty() {
            return Ok(false);
        }

        let graph = gm.gm_ref().getattr("graph")?;

        for (node, kind) in &candidates {
            match kind {
                DecompKind::Atanh => {
                    decompose_atanh(py, &graph, gm, node, debug)?;
                    modified = true;
                }
                DecompKind::Stack => {
                    decompose_stack(py, &graph, gm, node, debug)?;
                    modified = true;
                }
                DecompKind::Unbind => {
                    decompose_unbind(py, &graph, gm, node, debug)?;
                    modified = true;
                }
            }
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

/// atanh(x) = 0.5 * log((1+x) / (1-x))
fn decompose_atanh<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }
    let x = &args[0];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!("[DecomposeOps] decomposing atanh: {}", name);
    }

    let add_op = resolve_aten_op(py, "add.Tensor")?;
    let sub_op = resolve_aten_op(py, "sub.Tensor")?;
    let div_op = resolve_aten_op(py, "div.Tensor")?;
    let log_op = resolve_aten_op(py, "log.default")?;
    let mul_op = resolve_aten_op(py, "mul.Tensor")?;
    let full_op = resolve_aten_op(py, "full.default")?;

    // Get input shape for scalar constants
    let input_shape = FxGraphModule::node_shape(x).unwrap_or_else(|_| vec![1]);
    let shape_tuple = PyTuple::new_bound(py, input_shape.iter().copied());

    let ins = NodeInserter::before(py, graph, node);

    // Create constant 1.0
    let one_val = py_float(py, 1.0)?;
    let one_args = PyTuple::new_bound(py, &[shape_tuple.as_ref(), one_val.as_ref()]);
    let one = ins.call(&full_op, one_args.as_ref())?;

    // Create constant 0.5
    let half_val = py_float(py, 0.5)?;
    let half_args = PyTuple::new_bound(py, &[shape_tuple.as_ref(), half_val.as_ref()]);
    let half = ins.call(&full_op, half_args.as_ref())?;

    // one_plus_x = add(1, x) = 1 + x
    let one_plus_args = PyTuple::new_bound(py, &[one.as_ref(), x.as_ref()]);
    let one_plus_x = ins.call(&add_op, one_plus_args.as_ref())?;

    // one_minus_x = sub(1, x) = 1 - x
    let one_minus_args = PyTuple::new_bound(py, &[one.as_ref(), x.as_ref()]);
    let one_minus_x = ins.call(&sub_op, one_minus_args.as_ref())?;

    // ratio = div(one_plus_x, one_minus_x)
    let div_args = PyTuple::new_bound(py, &[one_plus_x.as_ref(), one_minus_x.as_ref()]);
    let ratio = ins.call(&div_op, div_args.as_ref())?;

    // log_ratio = log(ratio)
    let log_args = PyTuple::new_bound(py, &[ratio.as_ref()]);
    let log_ratio = ins.call(&log_op, log_args.as_ref())?;

    // result = mul(0.5, log_ratio)
    let mul_args = PyTuple::new_bound(py, &[half.as_ref(), log_ratio.as_ref()]);
    let result = ins.call(&mul_op, mul_args.as_ref())?;

    // Copy original node's meta (with qparams) onto the final result
    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}

/// stack(tensors, dim) -> unsqueeze each tensor at dim, then cat along dim
fn decompose_stack<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }

    // args[0] is a list/tuple of tensors, args[1] is dim (default 0)
    let tensors_obj = &args[0];
    let dim: i64 = if args.len() > 1 {
        args[1].extract().unwrap_or(0)
    } else {
        0
    };

    // Extract individual tensor nodes from the list
    let tensors: Vec<Bound<'_, PyAny>> = tensors_obj.iter()?.filter_map(|t| t.ok()).collect();
    if tensors.is_empty() {
        return Ok(());
    }

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeOps] decomposing stack: {} ({} tensors, dim={})",
            name,
            tensors.len(),
            dim
        );
    }

    let unsqueeze_op = resolve_aten_op(py, "unsqueeze_copy.default")
        .or_else(|_| resolve_aten_op(py, "unsqueeze.default"))?;
    let cat_op = resolve_aten_op(py, "cat.default")?;

    let dim_py = py_int(py, dim)?;

    let ins = NodeInserter::before(py, graph, node);

    // Unsqueeze each tensor at the stack dimension
    let mut unsqueezed: Vec<Bound<'_, PyAny>> = Vec::with_capacity(tensors.len());
    for tensor in &tensors {
        let unsqueeze_args = PyTuple::new_bound(py, &[tensor.as_ref(), dim_py.as_ref()]);
        let unsq = ins.call(&unsqueeze_op, unsqueeze_args.as_ref())?;
        unsqueezed.push(unsq);
    }

    // Cat the unsqueezed tensors along dim
    let unsqueezed_list = PyList::new_bound(py, &unsqueezed);
    let cat_args = PyTuple::new_bound(py, &[unsqueezed_list.as_ref(), dim_py.as_ref()]);
    let result = ins.call(&cat_op, cat_args.as_ref())?;

    let copy_mod = py.import_bound("copy")?;
    let final_meta = copy_mod.call_method1("copy", (node.getattr("meta")?,))?;
    result.setattr("meta", &final_meta)?;

    FxGraphModule::replace_all_uses_with(node, &result)?;
    gm.erase_node(node)?;
    Ok(())
}

/// unbind(x, dim) -> for each index: slice(x, dim, i, i+1) + squeeze(dim)
///
/// unbind returns a tuple of tensors. In the FX graph, the unbind node is
/// consumed by getitem nodes. We replace each getitem with the corresponding
/// slice+squeeze chain.
fn decompose_unbind<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<()> {
    let args = FxGraphModule::node_args(node)?;
    if args.is_empty() {
        return Ok(());
    }
    let x = &args[0];
    let dim: i64 = if args.len() > 1 {
        args[1].extract().unwrap_or(0)
    } else {
        0
    };

    let input_shape = match FxGraphModule::node_shape(x) {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    let rank = input_shape.len() as i64;
    let norm_dim = if dim < 0 { dim + rank } else { dim } as usize;

    if debug {
        let name = FxGraphModule::node_name(node)?;
        let num_slices = input_shape.get(norm_dim).copied().unwrap_or(0);
        eprintln!(
            "[DecomposeOps] decomposing unbind: {} (dim={}, slices={})",
            name, dim, num_slices
        );
    }

    let slice_op = resolve_aten_op(py, "slice_copy.Tensor")
        .or_else(|_| resolve_aten_op(py, "slice.Tensor"))?;
    let squeeze_op =
        resolve_aten_op(py, "squeeze_copy.dims").or_else(|_| resolve_aten_op(py, "squeeze.dim"))?;

    // Collect getitem users of the unbind node
    let users = FxGraphModule::node_users(node)?;
    let mut getitem_users: Vec<(Bound<'_, PyAny>, i64)> = Vec::new();
    for user in &users {
        let user_op = FxGraphModule::node_op(user)?;
        if user_op == "call_function" {
            let user_target = FxGraphModule::node_target_str(user)?;
            if user_target.contains("getitem") {
                let user_args = FxGraphModule::node_args(user)?;
                if user_args.len() >= 2 {
                    if let Ok(idx) = user_args[1].extract::<i64>() {
                        getitem_users.push((user.clone(), idx));
                    }
                }
            }
        }
    }

    let copy_mod = py.import_bound("copy")?;
    let dim_py = py_int(py, norm_dim as i64)?;

    for (getitem_node, idx) in &getitem_users {
        let idx_py = py_int(py, *idx)?;
        let idx_plus_one_py = py_int(py, idx + 1)?;
        let ins = NodeInserter::before(py, graph, getitem_node).meta_from(node);

        // slice_copy(x, dim, idx, idx+1)
        let slice_args = PyTuple::new_bound(
            py,
            &[
                x.as_ref(),
                dim_py.as_ref(),
                idx_py.as_ref(),
                idx_plus_one_py.as_ref(),
            ],
        );
        let sliced = ins.call(&slice_op, slice_args.as_ref())?;

        // squeeze_copy(sliced, [dim])
        let dims_list = PyList::new_bound(py, &[norm_dim as i64]);
        let squeeze_args = PyTuple::new_bound(py, &[sliced.as_ref(), dims_list.as_ref()]);
        let squeezed = ins.call(&squeeze_op, squeeze_args.as_ref())?;

        // Copy getitem's meta to the squeezed node for correct shape
        let gi_meta = copy_mod.call_method1("copy", (getitem_node.getattr("meta")?,))?;
        squeezed.setattr("meta", &gi_meta)?;

        FxGraphModule::replace_all_uses_with(getitem_node, &squeezed)?;
        gm.erase_node(getitem_node)?;
    }

    // If the unbind node has no remaining users, erase it
    let remaining_users = FxGraphModule::node_users(node)?;
    if remaining_users.is_empty() {
        gm.erase_node(node)?;
    }

    Ok(())
}
