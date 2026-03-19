use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;

/// A compiler pass that operates *directly* on a Python FX graph via PyO3.
///
/// Unlike [`crate::passes::Pass`] which works on Rust-owned IR, an `FxPass`
/// reads and mutates the live Python `torch.fx.Graph` — giving us correct
/// graph semantics (user lists, validation) for free while keeping the heavy
/// lifting (pattern matching, iteration) in Rust.
pub trait FxPass {
    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Run the pass.  Returns `true` if the graph was modified.
    fn run(&self, gm: &FxGraphModule) -> PyResult<bool>;
}

/// Run a sequence of `FxPass`es, logging progress to stderr.
pub fn run_fx_passes(gm: &FxGraphModule, passes: &[&dyn FxPass]) -> PyResult<bool> {
    use std::io::Write;
    let mut any_modified = false;
    for (i, pass) in passes.iter().enumerate() {
        let before = gm.node_count()?;
        let _ = writeln!(
            std::io::stderr(),
            "[FxPassRunner] starting pass {}/{}: {}",
            i + 1,
            passes.len(),
            pass.name()
        );
        let _ = std::io::stderr().flush();

        let modified = pass.run(gm)?;

        let after = gm.node_count()?;
        let _ = writeln!(
            std::io::stderr(),
            "[FxPassRunner] {} — modified={}, nodes: {} -> {}",
            pass.name(),
            modified,
            before,
            after,
        );
        let _ = std::io::stderr().flush();

        if modified {
            any_modified = true;
        }
    }

    if any_modified {
        let helpers = gm.py().import_bound("fx_helpers")?;
        helpers.call_method1("refresh_safe_quantized_fake_tensors", (gm.gm_ref(),))?;
    }

    Ok(any_modified)
}
