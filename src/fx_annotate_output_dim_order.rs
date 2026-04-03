use pyo3::prelude::*;

use crate::fx_graph::{meta_val, FxGraphModule};
use crate::fx_pass::FxPass;

pub struct FxAnnotateOutputDimOrderPass;

fn collect_output_nodes<'py>(node_args: &[Bound<'py, PyAny>]) -> Vec<Bound<'py, PyAny>> {
    let mut output_nodes = Vec::new();
    for arg in node_args {
        if FxGraphModule::is_node(arg) {
            output_nodes.push(arg.clone());
            continue;
        }
        if let Ok(iter) = arg.iter() {
            for item in iter.flatten() {
                if FxGraphModule::is_node(&item) {
                    output_nodes.push(item);
                }
            }
        }
    }
    output_nodes
}

fn extract_output_dim_order(node: &Bound<'_, PyAny>) -> PyResult<Option<Vec<usize>>> {
    let meta = node.getattr("meta")?;
    let Some(val) = meta_val(&meta, "val") else { return Ok(None) };
    let dim_order = match val.call_method0("dim_order") {
        Ok(order) => order,
        Err(_) => return Ok(None),
    };
    Ok(dim_order.extract::<Vec<usize>>().ok())
}

impl FxPass for FxAnnotateOutputDimOrderPass {
    fn name(&self) -> &str {
        "FxAnnotateOutputDimOrder"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let py = gm.py();
        let mut modified = false;

        for node in gm.nodes()? {
            if FxGraphModule::node_op(&node)? != "output" {
                continue;
            }

            let meta = node.getattr("meta")?;
            if meta_val(&meta, "original_dim_orders").is_some() {
                continue;
            }

            let args = FxGraphModule::node_args(&node)?;
            let output_nodes = collect_output_nodes(&args);
            let output_dim_orders: Vec<Vec<usize>> = output_nodes
                .iter()
                .filter_map(|output_node| extract_output_dim_order(output_node).ok().flatten())
                .collect();

            if output_dim_orders.is_empty() {
                continue;
            }

            let py_dim_orders = pyo3::types::PyList::empty_bound(py);
            for order in &output_dim_orders {
                py_dim_orders.append(order.clone())?;
            }
            meta.set_item("original_dim_orders", py_dim_orders)?;
            modified = true;
        }

        Ok(modified)
    }
}
