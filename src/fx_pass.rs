use pyo3::prelude::*;
use std::time::Instant;

use crate::fx_graph::FxGraphModule;
use crate::runner_logging::{log_line, log_pass_result, log_pass_start};

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
    let profile_enabled = crate::fx_profile::enabled();
    if profile_enabled {
        crate::fx_profile::reset();
    }

    let mut any_modified = false;
    let t_total = Instant::now();
    for (i, pass) in passes.iter().enumerate() {
        let before = gm.node_count()?;
        log_pass_start("FxPassRunner", i, passes.len(), pass.name());

        let recompile_count_before = crate::fx_profile::recompile_count();
        let recompile_elapsed_before = crate::fx_profile::recompile_elapsed();
        let t_pass = if profile_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let modified = pass.run(gm)?;
        let pass_elapsed = t_pass.map(|t| t.elapsed());

        let after = gm.node_count()?;
        log_pass_result("FxPassRunner", pass.name(), modified, before, after);
        if let Some(pass_elapsed) = pass_elapsed {
            let recompile_count_after = crate::fx_profile::recompile_count();
            let recompile_elapsed_after = crate::fx_profile::recompile_elapsed();
            let recompile_count_delta = recompile_count_after - recompile_count_before;
            let recompile_elapsed_delta =
                recompile_elapsed_after.saturating_sub(recompile_elapsed_before);
            log_line(format_args!(
                "[FxPassProfile] {} elapsed={:.3}ms recompile_count={} recompile_elapsed={:.3}ms",
                pass.name(),
                pass_elapsed.as_secs_f64() * 1000.0,
                recompile_count_delta,
                recompile_elapsed_delta.as_secs_f64() * 1000.0,
            ));
        }

        if modified {
            any_modified = true;
        }
    }

    if any_modified {
        let t_refresh = if profile_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let helpers = gm.py().import_bound("fx_helpers")?;
        helpers.call_method1("refresh_safe_quantized_fake_tensors", (gm.gm_ref(),))?;
        if let Some(t_refresh) = t_refresh {
            log_line(format_args!(
                "[FxPassProfile] refresh_safe_quantized_fake_tensors elapsed={:.3}ms",
                t_refresh.elapsed().as_secs_f64() * 1000.0,
            ));
        }
    }

    if profile_enabled {
        let total_elapsed = t_total.elapsed();
        let recompile_elapsed = crate::fx_profile::recompile_elapsed();
        let recompile_count = crate::fx_profile::recompile_count();
        log_line(format_args!(
            "[FxPassProfile] total elapsed={:.3}ms, recompile_count={}, recompile_elapsed={:.3}ms",
            total_elapsed.as_secs_f64() * 1000.0,
            recompile_count,
            recompile_elapsed.as_secs_f64() * 1000.0,
        ));
    }

    Ok(any_modified)
}

pub fn run_boxed_fx_passes(gm: &FxGraphModule, passes: &[Box<dyn FxPass>]) -> PyResult<bool> {
    let pass_refs: Vec<&dyn FxPass> = passes.iter().map(|pass| pass.as_ref()).collect();
    run_fx_passes(gm, &pass_refs)
}
