use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use smallvec::SmallVec;

use crate::graph::ir::*;
use crate::graph::Graph;
use crate::ingest::target_map::parse_op_target;
use crate::store::{TensorData, TensorStore};

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
                        placeholder_to_sd_key.insert(placeholder_name, sd_key);
                    }
                }
            }
            // Also try calling .items() for dict-like objects
            else if let Ok(items) = mapping.call_method0("items") {
                for item in items.iter()? {
                    let item = item?;
                    if let Ok(tuple) = item.extract::<(String, String)>() {
                        placeholder_to_sd_key.insert(tuple.0, tuple.1);
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
        let name: String = py_node.getattr("name")?.extract()?;

        let op_type = match op_str.as_str() {
            "placeholder" => OpType::Placeholder,
            "call_function" => OpType::CallFunction,
            "output" => OpType::Output,
            _ => OpType::CallFunction, // get_attr, call_method, etc.
        };

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
        let args = extract_args(&py_args, &name_map)?;

        // --- kwargs ---
        let py_kwargs = py_node.getattr("kwargs")?;
        let kwargs = extract_kwargs(&py_kwargs, &name_map)?;

        // --- meta ---
        let meta = extract_meta(&py_node, op_type)?;

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

        name_map.insert(name, node_id);
    }

    // -----------------------------------------------------------------
    // 2. Ingest state_dict into TensorStore
    // -----------------------------------------------------------------
    for (key, value) in state_dict.iter() {
        let tensor_name: String = key.extract()?;

        // Get shape
        let shape_tuple = value.getattr("shape")?;
        let shape: Vec<i64> = shape_tuple
            .iter()?
            .map(|s| s.and_then(|v| v.extract::<i64>()))
            .collect::<PyResult<Vec<_>>>()?;

        // Get dtype
        let dtype = parse_torch_dtype(&value.getattr("dtype")?.str()?.extract::<String>()?);

        // Get raw bytes: tensor.contiguous().cpu().numpy().tobytes()
        let contiguous = value.call_method0("contiguous")?;
        let cpu = contiguous.call_method0("cpu")?;
        let numpy = cpu.call_method0("numpy")?;
        let bytes_obj = numpy.call_method0("tobytes")?;
        let data: Vec<u8> = bytes_obj.extract()?;

        store.insert(
            tensor_name,
            TensorData { data, shape, dtype },
        );
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
            if let Ok(shape_obj) = tensor.getattr("shape") {
                if let Ok(shape) = shape_obj
                    .iter()
                    .and_then(|it| it.map(|s| s.and_then(|v| v.extract::<i64>())).collect::<PyResult<Vec<_>>>())
                {
                    let dtype = tensor
                        .getattr("dtype")
                        .and_then(|d| d.str()?.extract::<String>())
                        .map(|s| parse_torch_dtype(&s))
                        .unwrap_or(DType::Float32);
                    if let Ok(data) = tensor
                        .call_method0("contiguous")
                        .and_then(|c| c.call_method0("cpu"))
                        .and_then(|c| c.call_method0("numpy"))
                        .and_then(|n| n.call_method0("tobytes"))
                        .and_then(|b| b.extract::<Vec<u8>>())
                    {
                        store.insert(
                            placeholder_name.clone(),
                            TensorData { data, shape, dtype },
                        );
                    }
                }
            }
        }
    }

    Ok((graph, store))
}

// ---------------------------------------------------------------------------
// Arg extraction helpers
// ---------------------------------------------------------------------------

/// Extract positional args from a Python tuple/list of FX node args.
fn extract_args(
    py_args: &Bound<'_, PyAny>,
    name_map: &HashMap<String, NodeId>,
) -> PyResult<SmallVec<[Arg; 4]>> {
    let mut args = SmallVec::new();

    // py_args is typically a tuple
    let items: Vec<Bound<'_, PyAny>> = if py_args.is_instance_of::<PyTuple>() || py_args.is_instance_of::<PyList>() {
        py_args.iter()?.collect::<PyResult<Vec<_>>>()?
    } else {
        // Single value
        vec![py_args.clone()]
    };

    for item in &items {
        args.push(extract_single_arg(item, name_map)?);
    }

    Ok(args)
}

/// Extract kwargs from a Python dict.
fn extract_kwargs(
    py_kwargs: &Bound<'_, PyAny>,
    name_map: &HashMap<String, NodeId>,
) -> PyResult<Vec<(String, Arg)>> {
    let mut kwargs = Vec::new();

    if let Ok(dict) = py_kwargs.downcast::<PyDict>() {
        for (key, value) in dict.iter() {
            let key_str: String = key.extract()?;
            let arg = extract_single_arg(&value, name_map)?;
            kwargs.push((key_str, arg));
        }
    }

    Ok(kwargs)
}

/// Convert a single Python value into an `Arg`.
fn extract_single_arg(
    value: &Bound<'_, PyAny>,
    name_map: &HashMap<String, NodeId>,
) -> PyResult<Arg> {
    // Check for None
    if value.is_none() {
        return Ok(Arg::None);
    }

    // Check if it is an FX Node (has .name and .op attributes)
    if let (Ok(node_name), Ok(_op)) = (value.getattr("name"), value.getattr("op")) {
        let node_name: String = node_name.extract()?;
        if let Some(&node_id) = name_map.get(&node_name) {
            return Ok(Arg::Node(node_id));
        }
        // Node not yet seen — this shouldn't happen in topological order
        return Ok(Arg::None);
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

// ---------------------------------------------------------------------------
// Meta extraction
// ---------------------------------------------------------------------------

/// Extract node metadata (output shape and dtype) from `node.meta["val"]`.
fn extract_meta(py_node: &Bound<'_, PyAny>, op_type: OpType) -> PyResult<NodeMeta> {
    // Output nodes and some special nodes may not have useful meta["val"]
    let meta_dict = py_node.getattr("meta")?;

    // Try to get the "val" entry
    let val = match meta_dict.call_method1("get", ("val",)) {
        Ok(v) if !v.is_none() => v,
        _ => {
            // No val — return a default TensorInfo
            return Ok(NodeMeta::new(TensorInfo::new(&[], DType::Float32)));
        }
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

/// Map a string like `"torch.float32"` to our `DType` enum.
fn parse_torch_dtype(dtype_str: &str) -> DType {
    match dtype_str {
        s if s.contains("float32") => DType::Float32,
        s if s.contains("float16") => DType::Float16,
        s if s.contains("bfloat16") => DType::BFloat16,
        s if s.contains("int8") => DType::Int8,
        s if s.contains("int16") => DType::Int16,
        s if s.contains("int32") => DType::Int32,
        s if s.contains("int64") => DType::Int64,
        s if s.contains("bool") => DType::Bool,
        _ => DType::Float32, // default fallback
    }
}
