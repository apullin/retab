use pyo3::prelude::*;

use crate::fx_graph::FxGraphModule;

pub trait FxPass {
    fn run(&self, gm: &FxGraphModule) -> PyResult<bool>;
}
