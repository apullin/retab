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

/// Declare an FX pass that collects candidate nodes by target match,
/// then transforms each one.
///
/// Eliminates ~15 lines of boilerplate per pass (struct, impl FxPass,
/// candidate collection loop, recompile).
///
/// # Usage
/// ```ignore
/// fx_pass!(FxDecomposeAddmmPass, "FxDecomposeAddmm",
///     matches: |target, _node| target.contains("addmm"),
///     transform: |py, graph, gm, node, debug| decompose_addmm(py, graph, gm, node, debug)
/// );
/// ```
#[macro_export]
macro_rules! fx_pass {
    (
        $(#[$meta:meta])*
        $struct_name:ident, $name_str:expr,
        matches: |$target:ident, $mnode:ident| $match_expr:expr,
        transform: |$py:ident, $graph:ident, $gm:ident, $node:ident, $debug:ident| $transform_expr:expr
    ) => {
        $(#[$meta])*
        pub struct $struct_name;

        impl $crate::fx_pass::FxPass for $struct_name {
            fn name(&self) -> &str {
                $name_str
            }

            fn run(&self, gm: &$crate::fx_graph::FxGraphModule) -> pyo3::PyResult<bool> {
                let $py = gm.py();
                let $debug = $crate::fx_graph::FxGraphModule::debug();
                let mut modified = false;

                let mut candidates: Vec<pyo3::Bound<'_, pyo3::PyAny>> = Vec::new();

                for ($mnode, $target) in gm.call_functions()? {
                    if $match_expr {
                        candidates.push($mnode.clone());
                    }
                }

                if candidates.is_empty() {
                    return Ok(false);
                }

                let $graph = gm.gm_ref().getattr("graph")?;
                let $gm = gm;

                for $node in &candidates {
                    let result: pyo3::PyResult<()> = (|| { $transform_expr })();
                    result?;
                    modified = true;
                }

                if modified {
                    gm.recompile()?;
                }
                Ok(modified)
            }
        }
    };
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
