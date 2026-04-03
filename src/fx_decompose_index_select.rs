use pyo3::prelude::*;

use crate::fx_decompose_ops::{propagate_qparams, FxDecomposer};
use crate::fx_graph::{meta_val, FxGraphModule};

crate::fx_pass!(
    /// FX-graph-direct decomposition of `aten.index_select` into
    /// `backend.tosa.GATHER` plus shape-only ops, mirroring the Python Arm pass.
    FxDecomposeIndexSelectToGatherPass, "FxDecomposeIndexSelectToGather",
    matches: |target, _node| target.contains("index_select.default"),
    transform: |py, graph, gm, node, debug| decompose_index_select(py, &graph, gm, node, debug).map(|_| ())
);

fn copy_qparams_only(src_meta: &Bound<'_, PyAny>, dst_meta: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Some(input_qparams) = meta_val(src_meta, "input_qparams") {
        dst_meta.set_item("input_qparams", input_qparams)?;
    }
    if let Some(output_qparams) = meta_val(src_meta, "output_qparams") {
        dst_meta.set_item("output_qparams", output_qparams)?;
    }
    Ok(())
}

fn product(shape: &[i64]) -> i64 {
    if shape.is_empty() {
        1
    } else {
        shape.iter().copied().product()
    }
}

fn decompose_index_select<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    gm: &'py FxGraphModule<'py>,
    node: &Bound<'py, PyAny>,
    debug: bool,
) -> PyResult<bool> {
    let args = FxGraphModule::node_args(node)?;
    if args.len() != 3 {
        return Ok(false);
    }

    let x = &args[0];
    let dim: i64 = args[1].extract()?;
    let index = &args[2];
    if !FxGraphModule::is_node(x) || !FxGraphModule::is_node(index) {
        return Ok(false);
    }

    let x_shape = FxGraphModule::node_shape(x)?;
    let index_shape = FxGraphModule::node_shape(index)?;
    let x_rank = x_shape.len() as i64;
    if x_rank < 1 || index_shape.len() != 1 {
        return Ok(false);
    }

    let dim_norm = dim.rem_euclid(x_rank) as usize;
    let before = x_shape[..dim_norm].to_vec();
    let after = x_shape[dim_norm + 1..].to_vec();
    let n = product(&before);
    let k = x_shape[dim_norm];
    let c = product(&after);
    let w = index_shape[0];

    if debug {
        let name = FxGraphModule::node_name(node)?;
        eprintln!(
            "[DecomposeIndexSelectToGather] {} dim={} input={:?} index={:?} -> before={:?} after={:?} N={} K={} C={} W={}",
            name, dim, x_shape, index_shape, before, after, n, k, c, w
        );
    }

    let x_meta = x.getattr("meta")?;
    let x_dtype_str: String = x_meta.get_item("val")?.getattr("dtype")?.str()?.extract()?;
    let needs_bool_cast = x_dtype_str == "torch.bool";

    let torch = py.import_bound("torch")?;
    let int8_dtype = torch.getattr("int8")?;
    let bool_dtype = torch.getattr("bool")?;

    let d = FxDecomposer::new(py, graph, gm, node);

    // Optional bool -> int8 cast
    let mut x_for_gather = x.clone();
    if needs_bool_cast {
        x_for_gather = d.cast(x, &int8_dtype)?;
    }

    // Flatten input to (N, K, C)
    let values_nkc = d.view(&x_for_gather, &[n, k, c])?;
    propagate_qparams(&x_for_gather, &values_nkc)?;

    // Reshape index: unsqueeze(0) -> expand to (N, W)
    let idx_1w = d.unsqueeze(index, 0)?;
    let indices_nw = d.expand(&idx_1w, &[n, w])?;

    // GATHER
    let gathered_nwc = d.gather(&values_nkc, &indices_nw)?;
    {
        d.set_meta_val_shape(&x_for_gather, &gathered_nwc, &[n, w, c])?;
        copy_qparams_only(&x_for_gather.getattr("meta")?, &gathered_nwc.getattr("meta")?)?;
    }

    // Reshape output to (...before, W, ...after)
    let out_shape: Vec<i64> = before
        .iter()
        .copied()
        .chain(std::iter::once(w))
        .chain(after.iter().copied())
        .collect();
    let out = d.view(&gathered_nwc, &out_shape)?;
    copy_qparams_only(&gathered_nwc.getattr("meta")?, &out.getattr("meta")?)?;

    // Optional int8 -> bool cast back
    let result = if needs_bool_cast {
        d.cast(&out, &bool_dtype)?
    } else {
        out
    };

    d.replace(&result)
}
