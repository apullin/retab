use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// Ensure the engine directory is on sys.path for importing fx_python_passes.
fn ensure_engine_on_sys_path(py: Python<'_>) -> PyResult<()> {
    let sys = py.import_bound("sys")?;
    let path = sys.getattr("path")?;
    let engine_dir = env!("CARGO_MANIFEST_DIR");
    let already: bool = path
        .call_method1("__contains__", (engine_dir,))?
        .extract()?;
    if !already {
        path.call_method1("insert", (0, engine_dir))?;
    }
    Ok(())
}

/// Swap the graph from `new_gm` into `old_gm` if they differ.
///
/// The Python ExportPass retraces and may produce a different GraphModule.
fn swap_graph_if_needed(old_gm: &Bound<'_, PyAny>, new_gm: &Bound<'_, PyAny>) -> PyResult<()> {
    if !new_gm.is(old_gm) {
        let new_graph = new_gm.getattr("graph")?;
        old_gm.setattr("graph", &new_graph)?;
        old_gm.call_method0("recompile")?;
    }
    Ok(())
}

/// Run Python's FoldAndAnnotateQParamsPass + InsertRescalePass + InsertRescaleInt32Pass.
///
/// These are the battle-tested Python implementations from the ARM backend.
/// They produce QuantArgs namedtuples in node.meta (not plain dicts), which
/// downstream passes (InsertRescalePass, InsertRescaleInt32Pass) depend on.
///
/// The `exported_program` is stashed on construction and forwarded to
/// `FoldAndAnnotateQParamsPass.__init__` so it can resolve parameter tensor
/// nodes (per-channel scales/zero-points stored as graph parameters).
pub struct FxPythonFoldAndRescalePass {
    /// Kept as a raw Python object so we can pass it through to the Python helper.
    /// `None` when running without an ExportedProgram (e.g. unit tests).
    exported_program: Option<PyObject>,
}

impl FxPythonFoldAndRescalePass {
    pub fn new(exported_program: Option<PyObject>) -> Self {
        Self { exported_program }
    }
}

impl FxPass for FxPythonFoldAndRescalePass {
    fn name(&self) -> &str {
        "PythonFoldAndRescale"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        ensure_engine_on_sys_path(py)?;

        let module = py.import_bound("fx_python_passes")?;

        let ep_arg: Bound<'_, PyAny> = match &self.exported_program {
            Some(ep) => ep.bind(py).clone(),
            None => py.None().into_bound(py),
        };

        let new_gm = module.call_method1(
            "run_python_fold_and_rescale",
            (gm.gm_ref(), ep_arg),
        )?;

        swap_graph_if_needed(gm.gm_ref(), &new_gm)?;
        Ok(true)
    }
}

/// Run only the Python rescale passes (InsertRescalePass + InsertRescaleInt32Pass).
///
/// This is used after the Rust `FxFoldAndAnnotateQParamsPass` which stores
/// real QuantArgs namedtuples in node.meta. The Python rescale passes read
/// those QuantArgs and insert TOSA RESCALE ops as needed.
pub struct FxPythonRescaleOnlyPass;

impl FxPass for FxPythonRescaleOnlyPass {
    fn name(&self) -> &str {
        "PythonRescaleOnly"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        ensure_engine_on_sys_path(py)?;

        let module = py.import_bound("fx_python_passes")?;

        let new_gm = module.call_method1(
            "run_python_rescale_only",
            (gm.gm_ref(),),
        )?;

        swap_graph_if_needed(gm.gm_ref(), &new_gm)?;
        Ok(true)
    }
}

/// Run Python's DecomposeConvWithInt16ActivationPass.
///
/// The Arm backend uses this pass to rewrite int16-activation convolutions with
/// bias into no-bias conv + bias add. Without it, the serializer emits an
/// illegal direct int16 conv bias contract that the TOSA reference model
/// rejects.
pub struct FxPythonInt16ActivationConvPass;

impl FxPass for FxPythonInt16ActivationConvPass {
    fn name(&self) -> &str {
        "PythonDecomposeInt16ActivationConv"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        ensure_engine_on_sys_path(py)?;

        let module = py.import_bound("fx_python_passes")?;
        let new_gm = module.call_method1(
            "run_python_decompose_int16_activation_conv",
            (gm.gm_ref(),),
        )?;

        swap_graph_if_needed(gm.gm_ref(), &new_gm)?;
        Ok(true)
    }
}
