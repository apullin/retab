use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use smallvec::SmallVec;

use crate::fx_graph::{meta_get, meta_val};
use crate::graph::ir::*;
use crate::graph::Graph;
use crate::ingest::target_map::parse_op_target;
use crate::store::TensorStore;
use crate::tensor_data::{extract_tensor_data, parse_torch_dtype};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Convert a `torch.fx.GraphModule` + state_dict into the Rust `Graph` and
/// `TensorStore` representations.
pub fn ingest_fx_graph(
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
) -> PyResult<(Graph, TensorStore)> {
    ingest_fx_graph_with_prefix(graph_module, state_dict, graph_signature, None)
}

fn ingest_fx_graph_with_prefix(
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
    name_prefix: Option<&str>,
) -> PyResult<(Graph, TensorStore)> {
    let mut graph = Graph::new();
    let mut store = TensorStore::new();

    // name → NodeId mapping for resolving node references in args
    let mut name_map: HashMap<String, NodeId> = HashMap::new();

    // Build placeholder-name → state-dict-key mapping from graph_signature
    let mut placeholder_to_sd_key: HashMap<String, String> = HashMap::new();
    for attr_name in &[
        "inputs_to_parameters",
        "inputs_to_buffers",
        "inputs_to_lifted_tensor_constants",
    ] {
        if let Ok(mapping) = graph_signature.getattr(*attr_name) {
            if let Ok(dict) = mapping.downcast::<PyDict>() {
                for (k, v) in dict.iter() {
                    if let (Ok(placeholder_name), Ok(sd_key)) =
                        (k.extract::<String>(), v.extract::<String>())
                    {
                        placeholder_to_sd_key
                            .insert(qualify_name(name_prefix, &placeholder_name), sd_key);
                    }
                }
            }
            // Also try calling .items() for dict-like objects
            else if let Ok(items) = mapping.call_method0("items") {
                for item in items.iter()? {
                    let item = item?;
                    if let Ok(tuple) = item.extract::<(String, String)>() {
                        placeholder_to_sd_key.insert(qualify_name(name_prefix, &tuple.0), tuple.1);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // 1. Walk graph_module.graph.nodes
    // -----------------------------------------------------------------
    let fx_graph = graph_module.getattr("graph")?;
    let nodes = fx_graph.getattr("nodes")?;

    for py_node in nodes.iter()? {
        let py_node = py_node?;

        let op_str: String = py_node.getattr("op")?.extract()?;
        let original_name: String = py_node.getattr("name")?.extract()?;
        let name = qualify_name(name_prefix, &original_name);

        let is_get_attr = op_str == "get_attr";
        let op_type = match op_str.as_str() {
            "placeholder" | "get_attr" => OpType::Placeholder,
            "call_function" => OpType::CallFunction,
            "output" => OpType::Output,
            _ => OpType::CallFunction, // call_method, etc.
        };

        // Skip dead placeholder/get_attr nodes (no users in the FX graph).
        // After fold_qdq, scale/zero_point placeholders that fed DQ nodes
        // become dead (no consumers). Serializing them creates spurious
        // FP32 CONST tensors that confuse Vela and reduce NPU op count.
        //
        // Control-flow subgraphs are different: unused region placeholders
        // still participate in the cond/body function signature and must be
        // preserved so WHILE/COND region arity matches the carried inputs.
        if op_type == OpType::Placeholder {
            let users_obj = py_node.getattr("users")?;
            let user_count: usize = users_obj.len()?;
            let preserve_region_placeholder = !is_get_attr && name_prefix.is_some();
            if user_count == 0 && !preserve_region_placeholder {
                continue;
            }
        }

        // For get_attr nodes, extract the tensor data from graph_module
        // and add it to the store. These are constants registered as buffers
        // (bias tensors, table lookup data, etc.) by FX-direct passes.
        if is_get_attr {
            let attr_name: String = py_node.getattr("target")?.str()?.extract()?;
            if let Ok(attr) = graph_module.getattr(attr_name.as_str()) {
                if attr.getattr("graph").is_ok() {
                    // Control-flow branch/body submodules are serialized as
                    // named regions, not tensor placeholders.
                    continue;
                }
            }
            if store.get(&name).is_none() {
                if let Ok(tensor) = graph_module.getattr(attr_name.as_str()) {
                    if let Ok(tensor_data) = extract_tensor_data(&tensor) {
                        store.insert(name.clone(), tensor_data);
                    }
                }
            }
        }

        // --- target ---
        let op_target = if op_type == OpType::CallFunction {
            let target_str: String = py_node
                .getattr("target")?
                .call_method0("__str__")?
                .extract()?;
            parse_op_target(&target_str)
        } else {
            // placeholder and output targets are just string names
            let target_str: String = py_node.getattr("target")?.str()?.extract()?;
            OpTarget::Other(target_str)
        };

        // --- args ---
        let py_args = py_node.getattr("args")?;
        let args = extract_args(
            graph_module,
            &py_args,
            &mut graph,
            &mut name_map,
            name_prefix,
        )?;

        // --- kwargs ---
        let py_kwargs = py_node.getattr("kwargs")?;
        let kwargs = extract_kwargs(
            graph_module,
            &py_kwargs,
            &mut graph,
            &mut name_map,
            name_prefix,
        )?;

        // --- meta ---
        let mut meta = extract_meta(&py_node, op_type)?;

        let meta_dict = py_node.getattr("meta")?;
        enrich_meta_from_meta_dict(&meta_dict, &mut meta)?;

        // --- create node ---
        let node_id = graph.create_node(op_type, op_target, args, kwargs, meta, name.clone());

        // Track placeholder inputs
        if op_type == OpType::Placeholder {
            graph.inputs.push(node_id);
        }

        // Track output node
        if op_type == OpType::Output {
            graph.output = Some(node_id);
        }

        name_map.insert(original_name, node_id);
    }

    // -----------------------------------------------------------------
    // 2. Ingest state_dict into TensorStore
    // -----------------------------------------------------------------
    for (key, value) in state_dict.iter() {
        let tensor_name: String = key.extract()?;
        store.insert(tensor_name, extract_tensor_data(&value)?);
    }

    // -----------------------------------------------------------------
    // 3. Create store aliases: placeholder node name -> state_dict data
    // -----------------------------------------------------------------
    // Placeholder nodes have names like "b__frozen_param0" but state_dict
    // keys are "_frozen_param0". The graph_signature maps between them.
    for (placeholder_name, sd_key) in &placeholder_to_sd_key {
        if placeholder_name != sd_key {
            if let Some(data) = store.get(sd_key).cloned() {
                store.insert(placeholder_name.clone(), data);
            }
        }
    }

    // -----------------------------------------------------------------
    // 4. Extract frozen tensor constants from graph module attributes
    // -----------------------------------------------------------------
    // After pt2e quantization, parameters become frozen constants stored
    // as attributes on the graph module (e.g., graph_module._frozen_param0).
    // These aren't in state_dict, so we extract them directly.
    for (placeholder_name, sd_key) in &placeholder_to_sd_key {
        if store.get(placeholder_name).is_some() {
            continue; // Already have data from state_dict
        }
        // Try to get the tensor from graph_module as an attribute
        // The sd_key typically looks like "_frozen_param0"
        if let Ok(tensor) = graph_module.getattr(sd_key.as_str()) {
            if let Ok(tensor_data) = extract_tensor_data(&tensor) {
                store.insert(placeholder_name.clone(), tensor_data);
            }
        }
    }

    // -----------------------------------------------------------------
    // 5. Fix up output shapes for decomposed-linear nodes
    // -----------------------------------------------------------------
    // _create_node in fx_helpers.py copies meta from the original linear
    // node, so the intermediate view_copy and convolution nodes have the
    // wrong output shape. Fix them by reading the actual shape from args.
    fixup_reshape_shapes(&mut graph);

    // -----------------------------------------------------------------
    // 6. Fix up dtypes based on quantization parameters
    // -----------------------------------------------------------------
    // The FX-direct passes store qparams in meta but don't update
    // meta["val"].dtype (the FakeTensor retains its original FP32 dtype).
    // Now that we have qparams in our Rust NodeMeta, fix up the dtypes:
    //   - Nodes with input_qparams or output_qparams: use the quantized dtype
    //   - Accumulator ops (Conv2d, etc.): output is Int32
    //   - Placeholder nodes with Int64: already handled above during ingestion
    fixup_quantized_dtypes(&mut graph);

    Ok((graph, store))
}

pub fn ingest_fx_graph_hierarchy(
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
) -> PyResult<(Vec<(String, Graph)>, TensorStore)> {
    let (main_graph, mut store) = ingest_fx_graph(graph_module, state_dict, graph_signature)?;
    let mut graphs = vec![(String::from("main"), main_graph)];
    collect_control_flow_subgraphs(
        graph_module,
        state_dict,
        graph_signature,
        &mut graphs,
        &mut store,
        None,
    )?;
    Ok((graphs, store))
}

fn collect_control_flow_subgraphs(
    graph_module: &Bound<'_, PyAny>,
    state_dict: &Bound<'_, PyDict>,
    graph_signature: &Bound<'_, PyAny>,
    graphs: &mut Vec<(String, Graph)>,
    store: &mut TensorStore,
    name_prefix: Option<&str>,
) -> PyResult<()> {
    let fx_graph = graph_module.getattr("graph")?;
    let nodes = fx_graph.getattr("nodes")?;

    for py_node in nodes.iter()? {
        let py_node = py_node?;
        let op_str: String = py_node.getattr("op")?.extract()?;
        if op_str != "get_attr" {
            continue;
        }

        let attr_name: String = py_node.getattr("target")?.str()?.extract()?;
        let Ok(attr) = graph_module.getattr(attr_name.as_str()) else {
            continue;
        };
        let Ok(sub_graph) = attr.getattr("graph") else {
            continue;
        };
        if sub_graph.is_none() {
            continue;
        }

        let graph_name = match name_prefix {
            Some(prefix) => format!("{prefix}{attr_name}"),
            None => attr_name.clone(),
        };
        if graphs.iter().any(|(name, _)| name == &graph_name) {
            continue;
        }

        let child_prefix = format!("{graph_name}::");
        let (submodule_graph, submodule_store) =
            ingest_fx_graph_with_prefix(&attr, state_dict, graph_signature, Some(&child_prefix))?;
        for (key, value) in submodule_store.buffers {
            if store.get(&key).is_none() {
                store.insert(key, value);
            }
        }
        graphs.push((graph_name, submodule_graph));
        collect_control_flow_subgraphs(
            &attr,
            state_dict,
            graph_signature,
            graphs,
            store,
            Some(&child_prefix),
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Arg extraction helpers
// ---------------------------------------------------------------------------

/// Extract positional args from a Python tuple/list of FX node args.
fn extract_args(
    graph_module: &Bound<'_, PyAny>,
    py_args: &Bound<'_, PyAny>,
    graph: &mut Graph,
    name_map: &mut HashMap<String, NodeId>,
    name_prefix: Option<&str>,
) -> PyResult<SmallVec<[Arg; 4]>> {
    let mut args = SmallVec::new();

    // py_args is typically a tuple
    let items: Vec<Bound<'_, PyAny>> =
        if py_args.is_instance_of::<PyTuple>() || py_args.is_instance_of::<PyList>() {
            py_args.iter()?.collect::<PyResult<Vec<_>>>()?
        } else {
            // Single value
            vec![py_args.clone()]
        };

    for item in &items {
        args.push(extract_single_arg(
            graph_module,
            item,
            graph,
            name_map,
            name_prefix,
        )?);
    }

    Ok(args)
}

/// Extract kwargs from a Python dict.
fn extract_kwargs(
    graph_module: &Bound<'_, PyAny>,
    py_kwargs: &Bound<'_, PyAny>,
    graph: &mut Graph,
    name_map: &mut HashMap<String, NodeId>,
    name_prefix: Option<&str>,
) -> PyResult<Vec<(String, Arg)>> {
    let mut kwargs = Vec::new();

    if let Ok(dict) = py_kwargs.downcast::<PyDict>() {
        for (key, value) in dict.iter() {
            let key_str: String = key.extract()?;
            let arg = extract_single_arg(graph_module, &value, graph, name_map, name_prefix)?;
            kwargs.push((key_str, arg));
        }
    }

    Ok(kwargs)
}

/// Convert a single Python value into an `Arg`.
fn extract_single_arg(
    graph_module: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    graph: &mut Graph,
    name_map: &mut HashMap<String, NodeId>,
    name_prefix: Option<&str>,
) -> PyResult<Arg> {
    // Check for None
    if value.is_none() {
        return Ok(Arg::None);
    }

    // Check if it is an FX Node (has .name and .op attributes)
    if let (Ok(node_name), Ok(op_name)) = (value.getattr("name"), value.getattr("op")) {
        let node_name: String = node_name.extract()?;
        if let Some(&node_id) = name_map.get(&node_name) {
            return Ok(Arg::Node(node_id));
        }
        let op_name: String = op_name.extract()?;
        if op_name == "get_attr" {
            let target_name: String = value.getattr("target")?.str()?.extract()?;
            if let Ok(attr) = graph_module.getattr(target_name.as_str()) {
                if attr.getattr("graph").is_ok() {
                    let qualified = match name_prefix {
                        Some(prefix) => format!("{prefix}{node_name}"),
                        None => node_name,
                    };
                    return Ok(Arg::String(qualified));
                }
            }
        }
        return Ok(Arg::Node(materialize_external_placeholder(
            graph_module,
            graph,
            name_map,
            value,
            &node_name,
            name_prefix,
        )?));
    }

    // Check for bool before int (bool is a subtype of int in Python)
    if let Ok(b) = value.extract::<bool>() {
        return Ok(Arg::Bool(b));
    }

    // Check for int
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Arg::Int(i));
    }

    // Check for float
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Arg::Float(f));
    }

    // Check for string
    if value.is_instance_of::<PyString>() {
        let s: String = value.extract()?;
        if let Ok(attr) = graph_module.getattr(s.as_str()) {
            if attr.getattr("graph").is_ok() {
                let qualified = match name_prefix {
                    Some(prefix) => format!("{prefix}{s}"),
                    None => s,
                };
                return Ok(Arg::String(qualified));
            }
        }
        return Ok(Arg::String(s));
    }

    // Check for tuple/list — could be list of nodes, list of ints, etc.
    if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
        let items: Vec<Bound<'_, PyAny>> = value.iter()?.collect::<PyResult<Vec<_>>>()?;

        if items.is_empty() {
            return Ok(Arg::IntList(SmallVec::new()));
        }

        // Check if first element is an FX node
        if items[0].getattr("op").is_ok() {
            let mut node_ids = SmallVec::new();
            for item in &items {
                if let Ok(node_name) = item.getattr("name") {
                    let node_name: String = node_name.extract()?;
                    if let Some(&node_id) = name_map.get(&node_name) {
                        node_ids.push(node_id);
                    } else {
                        node_ids.push(materialize_external_placeholder(
                            graph_module,
                            graph,
                            name_map,
                            item,
                            &node_name,
                            name_prefix,
                        )?);
                    }
                }
            }
            return Ok(Arg::NodeList(node_ids));
        }

        // Try list of ints
        if let Ok(int_list) = items
            .iter()
            .map(|i| i.extract::<i64>())
            .collect::<PyResult<SmallVec<[i64; 6]>>>()
        {
            return Ok(Arg::IntList(int_list));
        }

        // Try list of floats
        if let Ok(float_list) = items
            .iter()
            .map(|i| i.extract::<f64>())
            .collect::<PyResult<SmallVec<[f64; 4]>>>()
        {
            return Ok(Arg::FloatList(float_list));
        }
    }

    // Check for torch dtype
    let type_name = value.get_type().name()?;
    if type_name == "dtype" {
        let dtype_str: String = value.str()?.extract()?;
        return Ok(Arg::Dtype(parse_torch_dtype(&dtype_str)));
    }

    // Fallback: convert to string representation
    let repr: String = value.str()?.extract()?;
    Ok(Arg::String(repr))
}

fn materialize_external_placeholder(
    _graph_module: &Bound<'_, PyAny>,
    graph: &mut Graph,
    name_map: &mut HashMap<String, NodeId>,
    value: &Bound<'_, PyAny>,
    original_name: &str,
    name_prefix: Option<&str>,
) -> PyResult<NodeId> {
    if let Some(&node_id) = name_map.get(original_name) {
        return Ok(node_id);
    }

    let qualified_name = qualify_name(name_prefix, original_name);

    let mut meta = extract_meta(value, OpType::Placeholder)?;
    let meta_dict = value.getattr("meta")?;
    enrich_meta_from_meta_dict(&meta_dict, &mut meta)?;

    let target_name = value
        .getattr("target")
        .ok()
        .and_then(|target| target.str().ok())
        .and_then(|target| target.extract::<String>().ok())
        .unwrap_or_else(|| original_name.to_string());

    let node_id = graph.create_node(
        OpType::Placeholder,
        OpTarget::Other(target_name),
        SmallVec::new(),
        Vec::new(),
        meta,
        qualified_name,
    );
    graph.inputs.push(node_id);
    name_map.insert(original_name.to_string(), node_id);
    Ok(node_id)
}

fn qualify_name(name_prefix: Option<&str>, name: &str) -> String {
    match name_prefix {
        Some(prefix) => format!("{prefix}{name}"),
        None => name.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Meta extraction
// ---------------------------------------------------------------------------

/// Extract node metadata (output shape and dtype) from `node.meta["val"]`.
fn extract_meta(py_node: &Bound<'_, PyAny>, op_type: OpType) -> PyResult<NodeMeta> {
    // Output nodes and some special nodes may not have useful meta["val"]
    let meta_dict = py_node.getattr("meta")?;

    // Try to get the "val" entry
    let Some(val) = meta_val(&meta_dict, "val") else {
        // No val — return a default TensorInfo
        return Ok(NodeMeta::new(TensorInfo::new(&[], DType::Float32)));
    };

    // For output nodes, val is often a tuple of tensors — take the first one
    if op_type == OpType::Output {
        if val.is_instance_of::<PyTuple>() || val.is_instance_of::<PyList>() {
            let items: Vec<Bound<'_, PyAny>> = val.iter()?.collect::<PyResult<Vec<_>>>()?;
            if let Some(first) = items.first() {
                if let Ok(info) = extract_tensor_info(first) {
                    return Ok(NodeMeta::new(info));
                }
            }
        }
        return Ok(NodeMeta::new(TensorInfo::new(&[], DType::Float32)));
    }

    // val might be a single FakeTensor or a tuple of them
    if val.is_instance_of::<PyTuple>() || val.is_instance_of::<PyList>() {
        let items: Vec<Bound<'_, PyAny>> = val.iter()?.collect::<PyResult<Vec<_>>>()?;
        if let Some(first) = items.first() {
            if let Ok(info) = extract_tensor_info(first) {
                return Ok(NodeMeta::new(info));
            }
        }
        return Ok(NodeMeta::new(TensorInfo::new(&[], DType::Float32)));
    }

    match extract_tensor_info(&val) {
        Ok(info) => Ok(NodeMeta::new(info)),
        Err(_) => Ok(NodeMeta::new(TensorInfo::new(&[], DType::Float32))),
    }
}

/// Extract shape and dtype from a Python tensor-like object (FakeTensor or similar).
fn extract_tensor_info(val: &Bound<'_, PyAny>) -> PyResult<TensorInfo> {
    let shape_obj = val.getattr("shape")?;
    let shape: Vec<i64> = shape_obj
        .iter()?
        .map(|s| s.and_then(|v| v.extract::<i64>()))
        .collect::<PyResult<Vec<_>>>()?;

    let dtype_str: String = val.getattr("dtype")?.str()?.extract()?;
    let dtype = parse_torch_dtype(&dtype_str);

    Ok(TensorInfo::new(&shape, dtype))
}

// ---------------------------------------------------------------------------
// DType mapping
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// QParam extraction from Python meta dicts
// ---------------------------------------------------------------------------

/// Extract `input_qparams` and `output_qparams` from a Python node's meta dict
/// into the Rust `NodeMeta`.
///
/// The FX-direct `FxFoldAndAnnotateQParamsPass` stores qparams as Python dicts:
///   meta["input_qparams"]  = {position: {scale, zero_point, qmin, qmax, dtype}}
///   meta["output_qparams"] = {position: {scale, zero_point, qmin, qmax, dtype}}
/// Parse a Python qparams dict `{pos: QuantArgs}` into entries, skipping
/// positions already present in `existing`.
fn parse_qparams_dict(
    py_obj: &Bound<'_, PyAny>,
    existing: &[(u32, QuantArgs)],
) -> Vec<(u32, QuantArgs)> {
    if py_obj.is_none() {
        return vec![];
    }
    let Ok(dict) = py_obj.downcast::<PyDict>() else {
        return vec![];
    };
    let mut out = vec![];
    for (pos_key, qp_val) in dict.iter() {
        let Ok(pos) = pos_key.extract::<u32>() else {
            continue;
        };
        let Some(qa) = extract_quantargs_from_py(&qp_val) else {
            continue;
        };
        if existing.iter().any(|(p, _)| *p == pos) {
            continue;
        }
        out.push((pos, qa));
    }
    out
}

fn enrich_meta_from_meta_dict(
    meta_dict: &Bound<'_, PyAny>,
    node_meta: &mut NodeMeta,
) -> PyResult<()> {
    extract_qparams_from_meta(meta_dict, node_meta)?;
    extract_tosa_dim_order(meta_dict, node_meta)?;
    extract_tosa_special_dtype(meta_dict, node_meta)?;
    extract_control_flow_meta(meta_dict, node_meta)?;
    Ok(())
}

fn extract_qparams_from_meta(
    meta_dict: &Bound<'_, PyAny>,
    node_meta: &mut NodeMeta,
) -> PyResult<()> {
    if let Some(input_qp) = meta_val(meta_dict, "input_qparams") {
        let entries = parse_qparams_dict(&input_qp, &node_meta.input_qparams);
        node_meta.input_qparams.extend(entries);
    }
    if let Some(output_qp) = meta_val(meta_dict, "output_qparams") {
        let entries = parse_qparams_dict(&output_qp, &node_meta.output_qparams);
        node_meta.output_qparams.extend(entries);
    }
    Ok(())
}

/// Extract `tosa_dim_order` from a Python node's meta dict.
///
/// The FX-direct `FxToTosaMemoryFormatPass` stores a list like `[0, 2, 3, 1]`
/// (NHWC) or `[0, 1, 2, 3]` (NCHW) in `node.meta["tosa_dim_order"]`.
fn extract_tosa_dim_order(meta_dict: &Bound<'_, PyAny>, node_meta: &mut NodeMeta) -> PyResult<()> {
    if let Some(dim_order) = meta_val(meta_dict, "tosa_dim_order") {
        if let Ok(order_list) = dim_order.iter().and_then(|it| {
            it.map(|item| item.and_then(|v| v.extract::<u8>()))
                .collect::<PyResult<smallvec::SmallVec<[u8; 6]>>>()
        }) {
            if !order_list.is_empty() {
                node_meta.tosa_dim_order = Some(order_list);
            }
        }
    }
    Ok(())
}

fn extract_tosa_special_dtype(
    meta_dict: &Bound<'_, PyAny>,
    node_meta: &mut NodeMeta,
) -> PyResult<()> {
    if let Some(special_dtype) = meta_val(meta_dict, "tosa_special_dtype") {
        let special_dtype_str = special_dtype.str()?.extract::<String>()?;
        node_meta.tosa_special_dtype = if special_dtype_str.contains("INT48") {
            Some(DType::Int48)
        } else if special_dtype_str.contains("INT4") {
            Some(DType::Int4)
        } else {
            None
        };
    }
    Ok(())
}

fn extract_control_flow_meta(
    meta_dict: &Bound<'_, PyAny>,
    node_meta: &mut NodeMeta,
) -> PyResult<()> {
    node_meta.arm_control_flow_parent_output_user = meta_get(meta_dict, "arm_control_flow_parent_output_user", false);
    Ok(())
}

/// Extract a `QuantArgs` from a Python QuantArgs namedtuple or a plain dict.
///
/// Python's `QuantArgs` (NamedTuple) has fields: scale, zp, qmin, qmax, dtype.
/// The legacy Rust fold pass stored dicts with keys: scale, zero_point, qmin, qmax, dtype.
/// This function handles both formats.
fn extract_quantargs_from_py(py_obj: &Bound<'_, PyAny>) -> Option<QuantArgs> {
    // Helper: try attribute access first (NamedTuple), then dict .get().
    fn get_field<T: for<'a> pyo3::FromPyObject<'a>>(
        obj: &Bound<'_, PyAny>,
        attr: &str,
        fallback_key: Option<&str>,
    ) -> Option<T> {
        if let Ok(val) = obj.getattr(attr) {
            if let Ok(v) = val.extract::<T>() {
                return Some(v);
            }
        }
        if let Some(key) = fallback_key {
            if let Some(val) = meta_val(obj, key) {
                if let Ok(v) = val.extract::<T>() {
                    return Some(v);
                }
            }
        }
        // Also try dict-style get on the primary attr name
        if let Some(val) = meta_val(obj, attr) {
            if let Ok(v) = val.extract::<T>() {
                return Some(v);
            }
        }
        None
    }

    // scale can be a float or a list of floats; extract as f64 first,
    // then try list and take the first element.
    let scale: f64 = get_field(py_obj, "scale", None).or_else(|| {
        // Try extracting as a list and taking the first element
        py_obj
            .getattr("scale")
            .ok()
            .and_then(|v| v.get_item(0).ok())
            .and_then(|v| v.extract().ok())
    })?;

    // QuantArgs uses "zp"; legacy dicts use "zero_point"
    let zero_point: i64 = get_field(py_obj, "zp", Some("zero_point")).unwrap_or(0);
    let qmin: i64 = get_field(py_obj, "qmin", None).unwrap_or(-128);
    let qmax: i64 = get_field(py_obj, "qmax", None).unwrap_or(127);

    // dtype: QuantArgs stores a torch.dtype object, dicts store a string.
    let dtype = py_obj
        .getattr("dtype")
        .ok()
        .or_else(|| meta_val(py_obj, "dtype"))
        .and_then(|v| {
            // First try as string directly
            v.extract::<String>()
                .ok()
                // Otherwise stringify the torch.dtype object
                .or_else(|| v.str().ok().and_then(|s| s.extract().ok()))
        })
        .map(|s| parse_torch_dtype(&s))
        .unwrap_or(DType::Int8);

    Some(QuantArgs {
        scale,
        zero_point,
        dtype,
        qmin,
        qmax,
    })
}

// ---------------------------------------------------------------------------
// Post-ingestion shape fixup
// ---------------------------------------------------------------------------

/// Fix output shapes for Reshape/View/Flatten nodes and their downstream conv.
///
/// The linear decomposition pass (`FxDecomposeLinearPass`) creates nodes whose
/// meta (including `val` FakeTensor) is copied from the original linear node.
/// This means the intermediate view_copy and convolution nodes have the linear
/// output's shape (e.g. `[1, 10]`) instead of their actual output shapes:
///   - view_copy(input, [1, 1568, 1, 1]) should have shape [1, 1568, 1, 1]
///   - view_copy(weights, [10, 1568, 1, 1]) should have shape [10, 1568, 1, 1]
///   - conv2d(input_4d, weight_4d, ...) should have 4D output shape
///   - view_copy(conv_out, [1, 10]) should have shape [1, 10]
///
/// For reshape-like ops: read the target shape from args[1] (IntList).
/// For conv: compute output shape from input shape, weight shape, and params.
fn resolve_inferred_reshape_shape(shape: &[i64], input_shape: &[i64]) -> Option<Vec<i64>> {
    let negative_positions: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (dim < 0).then_some(idx))
        .collect();
    if negative_positions.is_empty() {
        return Some(shape.to_vec());
    }
    if negative_positions.len() > 1 {
        return None;
    }

    let input_numel = input_shape
        .iter()
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim))?;
    let known_product = shape
        .iter()
        .filter(|&&dim| dim >= 0)
        .try_fold(1i64, |acc, &dim| acc.checked_mul(dim))?;
    if known_product == 0 || input_numel % known_product != 0 {
        return None;
    }

    let mut resolved = shape.to_vec();
    resolved[negative_positions[0]] = input_numel / known_product;
    Some(resolved)
}

fn fixup_reshape_shapes(graph: &mut Graph) {
    let all_ids = graph.all_node_ids();
    for node_id in &all_ids {
        let (arg1_shape, input_id) = {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            if node.op != OpType::CallFunction {
                continue;
            }
            let is_reshape = matches!(
                node.target,
                OpTarget::Reshape | OpTarget::View | OpTarget::Flatten
            );
            if !is_reshape {
                continue;
            }
            // Extract shape from args[1] (IntList)
            let shape = match node.args.get(1) {
                Some(Arg::IntList(list)) => {
                    let v: Vec<i64> = list.iter().copied().collect();
                    Some(v)
                }
                _ => None,
            };
            let input_id = match node.args.first() {
                Some(Arg::Node(id)) => Some(*id),
                _ => None,
            };
            (shape, input_id)
        };

        if let Some(new_shape) = arg1_shape {
            let resolved_shape = if new_shape.iter().any(|&dim| dim < 0) {
                input_id
                    .and_then(|id| graph.node(id))
                    .and_then(|input| {
                        resolve_inferred_reshape_shape(&new_shape, input.meta.val.shape_slice())
                    })
                    .unwrap_or(new_shape)
            } else {
                new_shape
            };
            // Check if current shape matches the intended shape from args
            let node = graph.node(*node_id).unwrap();
            let current = node.meta.val.shape_slice();
            if current != resolved_shape.as_slice() {
                let dt = node.meta.val.dtype;
                if let Some(node_mut) = graph.node_mut(*node_id) {
                    node_mut.meta.val = TensorInfo::new(&resolved_shape, dt);
                }
            }
        }
    }

    // Also fix Conv2d nodes whose output shape doesn't match their inputs.
    // The decomposed linear creates conv2d with 4D inputs but inherits 2D
    // output shape from the original linear. Fix by computing output shape:
    //   output = [batch, out_channels, oH, oW] where oH/oW come from
    //   (input_H + 2*pad - dilation*(kernel-1) - 1) / stride + 1
    let all_ids2 = graph.all_node_ids();
    for node_id in &all_ids2 {
        let (is_conv, input_id, weight_id, current_shape_len) = {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            if node.op != OpType::CallFunction {
                continue;
            }
            let is_conv = matches!(node.target, OpTarget::Conv2d);
            if !is_conv {
                continue;
            }
            let input_id = match node.args.get(0) {
                Some(Arg::Node(id)) => Some(*id),
                _ => None,
            };
            let weight_id = match node.args.get(1) {
                Some(Arg::Node(id)) => Some(*id),
                _ => None,
            };
            (true, input_id, weight_id, node.meta.val.shape_len)
        };

        if !is_conv {
            continue;
        }

        // Only fix if the current shape is not 4D (indicates it was not properly set)
        if current_shape_len == 4 {
            continue;
        }

        if let (Some(input_id), Some(weight_id)) = (input_id, weight_id) {
            let (input_shape, weight_shape) = {
                let input_node = graph.node(input_id);
                let weight_node = graph.node(weight_id);
                (
                    input_node.map(|n| n.meta.val.shape_slice().to_vec()),
                    weight_node.map(|n| n.meta.val.shape_slice().to_vec()),
                )
            };

            if let (Some(ishape), Some(wshape)) = (input_shape, weight_shape) {
                if ishape.len() == 4 && wshape.len() == 4 {
                    // Extract conv params from args
                    let node = graph.node(*node_id).unwrap();
                    let stride = match node.args.get(3) {
                        Some(Arg::IntList(l)) => l.iter().copied().collect::<Vec<_>>(),
                        _ => vec![1, 1],
                    };
                    let padding = match node.args.get(4) {
                        Some(Arg::IntList(l)) => l.iter().copied().collect::<Vec<_>>(),
                        _ => vec![0, 0],
                    };
                    let dilation = match node.args.get(5) {
                        Some(Arg::IntList(l)) => l.iter().copied().collect::<Vec<_>>(),
                        _ => vec![1, 1],
                    };

                    // Input is NHWC: [N, H, W, C]
                    // Weight is OHWI: [OC, KH, KW, IC]
                    let batch = ishape[0];
                    let ih = ishape[1];
                    let iw = ishape[2];
                    let out_channels = wshape[0];
                    let kh = wshape[1];
                    let kw = wshape[2];
                    let sh = if stride.len() >= 2 { stride[0] } else { 1 };
                    let sw = if stride.len() >= 2 { stride[1] } else { 1 };
                    let ph = if padding.len() >= 2 { padding[0] } else { 0 };
                    let pw = if padding.len() >= 2 { padding[1] } else { 0 };
                    let dh = if dilation.len() >= 2 { dilation[0] } else { 1 };
                    let dw = if dilation.len() >= 2 { dilation[1] } else { 1 };

                    let oh = (ih + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
                    let ow = (iw + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

                    // Output is NHWC: [N, OH, OW, OC]
                    let new_shape = [batch, oh, ow, out_channels];
                    let dt = node.meta.val.dtype;
                    if let Some(node_mut) = graph.node_mut(*node_id) {
                        node_mut.meta.val = TensorInfo::new(&new_shape, dt);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Post-ingestion dtype fixup
// ---------------------------------------------------------------------------

/// Returns true for ops that produce int32 accumulator outputs.
fn is_accumulator_op(target: &OpTarget) -> bool {
    matches!(
        target,
        OpTarget::Conv2d
            | OpTarget::DepthwiseConv2d
            | OpTarget::TosaConv2d
            | OpTarget::TosaDepthwiseConv2d
            | OpTarget::TosaTransposeConv2d
            | OpTarget::TosaFullyConnected
            | OpTarget::Linear
            | OpTarget::Addmm
            | OpTarget::Matmul
            | OpTarget::Bmm
            | OpTarget::TosaMatmul
    )
}

/// Fix up `meta.val.dtype` on all nodes based on quantization parameters.
///
/// After ingestion, FakeTensor dtypes are still FP32 because the Python
/// pipeline's FX-direct passes don't modify `meta["val"]`. This function:
///   1. Sets dtype to the quantized type (from qparams) for CallFunction nodes
///   2. Sets accumulator ops' output dtype to Int32
///   3. Casts placeholder nodes with Int64 dtype to Int32
///   4. Infers placeholder dtypes from consumer node qparams
fn fixup_quantized_dtypes(graph: &mut Graph) {
    let all_ids = graph.all_node_ids();
    for node_id in all_ids {
        let (
            op,
            target,
            has_input_qp,
            has_output_qp,
            input_qp_dtype,
            output_qp_dtype,
            current_dtype,
        ) = {
            let Some(node) = graph.node(node_id) else {
                continue;
            };
            let iqp_dtype = node.meta.input_qparams.first().map(|(_, qa)| qa.dtype);
            let oqp_dtype = node.meta.output_qparams.first().map(|(_, qa)| qa.dtype);
            (
                node.op,
                node.target.clone(),
                !node.meta.input_qparams.is_empty(),
                !node.meta.output_qparams.is_empty(),
                iqp_dtype,
                oqp_dtype,
                node.meta.val.dtype,
            )
        };

        // Fix 1: CallFunction nodes with qparams should use the quantized dtype
        if op == OpType::CallFunction {
            // Accumulator ops produce Int32 output
            if is_accumulator_op(&target) && (has_input_qp || has_output_qp) {
                if let Some(node) = graph.node_mut(node_id) {
                    node.meta.val.dtype = DType::Int32;
                }
                continue;
            }

            // TOSA TABLE output dtype is determined by its input dtype:
            // INT8 -> INT8, INT16 -> INT32.
            if target == OpTarget::TosaTable {
                let table_dtype = graph.node(node_id).and_then(|n| match n.args.first() {
                    Some(Arg::Node(input_id)) => {
                        graph.node(*input_id).map(|input| input.meta.val.dtype)
                    }
                    _ => input_qp_dtype,
                });
                if let Some(dt) = table_dtype {
                    if let Some(node) = graph.node_mut(node_id) {
                        node.meta.val.dtype = if dt == DType::Int16 { DType::Int32 } else { dt };
                    }
                }
                continue;
            }

            // RESCALE output: use the output qparam dtype, or extract from args.
            // FX-direct rescale nodes have args: [input, out_dtype, scales, input_zp, output_zp]
            // Rust-IR rescale nodes have args: [input, multiplier, shift] with kwargs.
            if target == OpTarget::TosaRescale {
                // FX-direct rescale nodes carry the actual destination dtype in
                // args[1]. Prefer that over output_qparams, which may still
                // describe the eventual quantized consumer rather than this
                // intermediate accumulator rescale.
                let rescale_dtype = graph.node(node_id).and_then(|n| match n.args.get(1) {
                    Some(Arg::Dtype(dt)) => Some(*dt),
                    _ => output_qp_dtype,
                });
                if let Some(dt) = rescale_dtype {
                    if let Some(node) = graph.node_mut(node_id) {
                        node.meta.val.dtype = dt;
                    }
                }
                continue;
            }

            // Dequantize nodes consume quantized tensors but produce float
            // tensors. Preserve the FX-provided float meta dtype here instead
            // of falling through to the generic qparam-based dtype rewrite.
            if matches!(
                target,
                OpTarget::DequantizePerTensor | OpTarget::DequantizePerChannel
            ) {
                continue;
            }

            // General case: prefer output_qparams dtype, then input_qparams dtype
            let new_dtype = if let Some(dt) = output_qp_dtype {
                Some(dt)
            } else if let Some(dt) = input_qp_dtype {
                Some(dt)
            } else {
                None
            };

            if let Some(dt) = new_dtype {
                if let Some(node) = graph.node_mut(node_id) {
                    node.meta.val.dtype = dt;
                }
            }
        }

        // Fix 2: Placeholder nodes with Int64 should be Int32
        if op == OpType::Placeholder && current_dtype == DType::Int64 {
            if let Some(node) = graph.node_mut(node_id) {
                node.meta.val.dtype = DType::Int32;
            }
        }
    }

    // Fix 3: Infer placeholder dtypes from consumer qparams.
    // Runtime input placeholders (no store data) may still have FP32 dtype
    // when they should be Int8 (because the Q node was folded). Check if
    // any consumer has input_qparams for the arg position that references
    // this placeholder.
    let all_ids2 = graph.all_node_ids();
    for node_id in &all_ids2 {
        let (is_placeholder, current_dt, users) = {
            let Some(node) = graph.node(*node_id) else {
                continue;
            };
            if node.op != OpType::Placeholder {
                continue;
            }
            // Only fix FP32 placeholders (Int8/Int32 are already correct)
            if node.meta.val.dtype != DType::Float32 {
                continue;
            }
            (true, node.meta.val.dtype, node.users.to_vec())
        };

        if !is_placeholder {
            continue;
        }

        // Check each user for input_qparams referencing this placeholder
        for user_id in &users {
            let inferred_dtype = {
                let Some(user) = graph.node(*user_id) else {
                    continue;
                };
                // Find which arg position references this placeholder
                let pos = user.args.iter().enumerate().find_map(|(i, arg)| match arg {
                    Arg::Node(id) if *id == *node_id => Some(i as u32),
                    _ => None,
                });
                let Some(pos) = pos else {
                    continue;
                };
                // Get the input_qparams for that position
                user.meta
                    .input_qparams
                    .iter()
                    .find(|(p, _)| *p == pos)
                    .map(|(_, qa)| qa.dtype)
            };

            if let Some(dt) = inferred_dtype {
                if dt != current_dt {
                    if let Some(node) = graph.node_mut(*node_id) {
                        node.meta.val.dtype = dt;
                    }
                }
                break; // Only need one inference
            }
        }
    }

    // Fix 4: Propagate dtype through reshape-like ops.
    // Reshape, View, Flatten, Squeeze, Unsqueeze do not change dtype —
    // they should inherit it from their first (input) argument. After
    // FX-direct passes, these ops may retain the original FP32 dtype
    // from meta["val"] because the passes don't update FakeTensor dtypes.
    // Run multiple iterations to handle chains (e.g. view -> view -> conv).
    for _iter in 0..3 {
        let all_ids4 = graph.all_node_ids();
        let mut changed = false;
        for node_id in &all_ids4 {
            let (is_reshape, current_dt, input_id) = {
                let Some(node) = graph.node(*node_id) else {
                    continue;
                };
                if node.op != OpType::CallFunction {
                    continue;
                }
                let is_reshape = matches!(
                    node.target,
                    OpTarget::Reshape
                        | OpTarget::View
                        | OpTarget::Flatten
                        | OpTarget::Squeeze
                        | OpTarget::Unsqueeze
                        | OpTarget::TosaReshape
                        | OpTarget::Permute
                        | OpTarget::TosaTranspose
                );
                if !is_reshape {
                    continue;
                }
                // Only fix FP32 nodes (quantized dtypes are already correct)
                if node.meta.val.dtype != DType::Float32 {
                    continue;
                }
                let input_id = match node.args.first() {
                    Some(Arg::Node(id)) => Some(*id),
                    _ => None,
                };
                (true, node.meta.val.dtype, input_id)
            };

            if !is_reshape {
                continue;
            }

            if let Some(input_id) = input_id {
                let input_dt = graph.node(input_id).map(|n| n.meta.val.dtype);
                if let Some(dt) = input_dt {
                    if dt != current_dt && dt != DType::Float32 {
                        if let Some(node) = graph.node_mut(*node_id) {
                            node.meta.val.dtype = dt;
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}
