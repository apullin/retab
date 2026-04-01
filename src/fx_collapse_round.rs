use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// Collapse the decomposed banker's-rounding subgraph into a single TABLE op.
///
/// The Python DecomposeRoundPass runs during quantization annotation (before
/// the Rust pipeline) and produces a 12-op subgraph that loses precision at
/// the 0.5 tie-break boundary.  After q/dq folding, the nodes have qparams
/// and we can replace the entire subgraph with a TABLE lookup that is exact
/// for INT8 inputs.
///
/// Must run AFTER FxFoldAndAnnotateQParamsPass (needs qparams on nodes)
/// and BEFORE FxInsertTableOpsPass (so it gets priority over individual
/// floor TABLE conversions).
pub struct FxCollapseRoundToTablePass;

impl FxPass for FxCollapseRoundToTablePass {
    fn name(&self) -> &str {
        "FxCollapseRoundToTable"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let result = helpers
            .call_method1("collapse_decomposed_round_to_table", (gm.gm_ref(),))?
            .extract::<bool>()?;
        Ok(result)
    }
}
