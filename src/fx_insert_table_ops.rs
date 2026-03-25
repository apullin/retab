use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of `InsertTableOpsPass`.
///
/// Finds quantized math ops that should become TOSA TABLEs on U55 and
/// replaces them with a TOSA TABLE op backed by a precomputed lookup table.
///
/// The heavy matching logic lives in Rust; the actual table computation and
/// node creation are delegated to `fx_helpers.py` to avoid reimplementing
/// NumPy / FakeTensor bookkeeping.
pub struct FxInsertTableOpsPass;

#[derive(Clone, Copy)]
struct TableOpSpec {
    kind: &'static str,
    scalar_arg: Option<f64>,
}

fn node_scalar_arg(node: &Bound<'_, PyAny>, idx: usize) -> PyResult<Option<f64>> {
    let args = FxGraphModule::node_args(node)?;
    let Some(arg) = args.get(idx) else {
        return Ok(None);
    };
    if let Ok(v) = arg.extract::<f64>() {
        return Ok(Some(v));
    }
    if let Ok(v) = arg.extract::<i64>() {
        return Ok(Some(v as f64));
    }
    Ok(None)
}

/// Recognised targets that can be lowered to TABLE ops.
fn table_op_spec(node: &Bound<'_, PyAny>, target: &str) -> PyResult<Option<TableOpSpec>> {
    if target.contains("rsqrt.default") {
        Ok(Some(TableOpSpec {
            kind: "rsqrt",
            scalar_arg: None,
        }))
    } else if target.contains("reciprocal.default") {
        Ok(Some(TableOpSpec {
            kind: "reciprocal",
            scalar_arg: None,
        }))
    } else if target.contains("exp.default") {
        Ok(Some(TableOpSpec {
            kind: "exp",
            scalar_arg: None,
        }))
    } else if target.contains("log.default") {
        Ok(Some(TableOpSpec {
            kind: "log",
            scalar_arg: None,
        }))
    } else if target.contains("sigmoid.default") {
        Ok(Some(TableOpSpec {
            kind: "sigmoid",
            scalar_arg: None,
        }))
    } else if target.contains("tanh.default") {
        Ok(Some(TableOpSpec {
            kind: "tanh",
            scalar_arg: None,
        }))
    } else if target.contains("pow.Tensor_Scalar") {
        Ok(node_scalar_arg(node, 1)?.map(|scalar_arg| TableOpSpec {
            kind: "pow",
            scalar_arg: Some(scalar_arg),
        }))
    } else if target.contains("gelu.default") {
        Ok(Some(TableOpSpec {
            kind: "gelu",
            scalar_arg: None,
        }))
    } else if target.contains("silu.default") {
        Ok(Some(TableOpSpec {
            kind: "silu",
            scalar_arg: None,
        }))
    } else if target.contains("erf.default") {
        Ok(Some(TableOpSpec {
            kind: "erf",
            scalar_arg: None,
        }))
    } else if target.contains("sinh.default") {
        Ok(Some(TableOpSpec {
            kind: "sinh",
            scalar_arg: None,
        }))
    } else if target.contains("cosh.default") {
        Ok(Some(TableOpSpec {
            kind: "cosh",
            scalar_arg: None,
        }))
    } else if target.contains("tan.default") {
        Ok(Some(TableOpSpec {
            kind: "tan",
            scalar_arg: None,
        }))
    } else if target.contains("acos.default") {
        Ok(Some(TableOpSpec {
            kind: "acos",
            scalar_arg: None,
        }))
    } else if target.contains("asin.default") {
        Ok(Some(TableOpSpec {
            kind: "asin",
            scalar_arg: None,
        }))
    } else if target.contains("atan.default") {
        Ok(Some(TableOpSpec {
            kind: "atan",
            scalar_arg: None,
        }))
    } else if target.contains("expm1.default") {
        Ok(Some(TableOpSpec {
            kind: "expm1",
            scalar_arg: None,
        }))
    } else if target.contains("elu.default") || target.contains("elu_") {
        // elu(x, alpha) = x if x >= 0, alpha * (exp(x) - 1) if x < 0
        // Default alpha = 1.0
        let alpha = node_scalar_arg(node, 1)?.unwrap_or(1.0);
        Ok(Some(TableOpSpec {
            kind: "elu",
            scalar_arg: Some(alpha),
        }))
    } else if target.contains("acosh.default") {
        Ok(Some(TableOpSpec {
            kind: "acosh",
            scalar_arg: None,
        }))
    } else if target.contains("asinh.default") {
        Ok(Some(TableOpSpec {
            kind: "asinh",
            scalar_arg: None,
        }))
    } else if target.contains("atanh.default") {
        // atanh is already handled by FxDecomposeOpsPass, but add TABLE fallback
        Ok(Some(TableOpSpec {
            kind: "atanh",
            scalar_arg: None,
        }))
    } else if target.contains("erfinv.default") {
        Ok(Some(TableOpSpec {
            kind: "erfinv",
            scalar_arg: None,
        }))
    } else if target.contains("log1p.default") {
        Ok(Some(TableOpSpec {
            kind: "log1p",
            scalar_arg: None,
        }))
    } else if target.contains("sqrt.default") {
        Ok(Some(TableOpSpec {
            kind: "sqrt",
            scalar_arg: None,
        }))
    } else {
        Ok(None)
    }
}

impl FxPass for FxInsertTableOpsPass {
    fn name(&self) -> &str {
        "FxInsertTableOps"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let helpers = py.import_bound("fx_helpers")?;
        let mut modified = false;

        let all_nodes = gm.nodes()?;

        struct TableCandidate<'py> {
            node: Bound<'py, PyAny>,
            spec: TableOpSpec,
        }
        let mut candidates: Vec<TableCandidate> = Vec::new();

        for node in &all_nodes {
            let op = FxGraphModule::node_op(node)?;
            if op != "call_function" {
                continue;
            }
            let target = FxGraphModule::node_target_str(node)?;
            let spec = match table_op_spec(node, &target)? {
                Some(spec) => spec,
                None => continue,
            };

            // Only convert when the node has *both* input and output qparams
            // (i.e. it sits inside a quantised region).
            let meta = node.getattr("meta")?;

            let has_input_qp = match meta.call_method1("get", ("input_qparams",)) {
                Ok(v) if !v.is_none() => {
                    let len: usize = py
                        .import_bound("builtins")?
                        .call_method1("len", (&v,))?
                        .extract()
                        .unwrap_or(0);
                    len > 0
                }
                _ => false,
            };

            let has_output_qp = match meta.call_method1("get", ("output_qparams",)) {
                Ok(v) if !v.is_none() => {
                    let len: usize = py
                        .import_bound("builtins")?
                        .call_method1("len", (&v,))?
                        .extract()
                        .unwrap_or(0);
                    len > 0
                }
                _ => false,
            };

            if has_input_qp && has_output_qp {
                candidates.push(TableCandidate {
                    node: node.clone(),
                    spec,
                });
            }
        }

        // Apply replacements.
        for cand in &candidates {
            // Delegate to Python helper which:
            //   1. Reads input_qparams[0] and output_qparams[0]
            //   2. Computes the 513-entry table
            //   3. Creates a const node + TABLE call_function node
            //   4. Replaces all uses of the original node with the TABLE node
            //   5. Erases the original node
            helpers.call_method1(
                "insert_table_op",
                (
                    gm.gm_ref(),
                    &cand.node,
                    cand.spec.kind,
                    cand.spec.scalar_arg,
                ),
            )?;
            modified = true;
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python; Rust unit tests would need a live
    // Python FX graph which is non-trivial to set up.
}
