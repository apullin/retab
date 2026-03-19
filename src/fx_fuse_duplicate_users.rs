use std::collections::{HashMap, VecDeque};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::fx_graph::FxGraphModule;
use crate::fx_pass::FxPass;

/// FX-graph-direct implementation of ExecuTorch's `FuseDuplicateUsersPass`.
///
/// This only fuses duplicate *sibling* users of the same producer, matching the
/// Python reference pass. A global dedup is unsafe here because repeated
/// layernorm-style blocks legitimately reuse the same op shapes and kwargs.
pub struct FxFuseDuplicateUsersPass;

fn node_is_live(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(!node.getattr("graph")?.is_none())
}

fn target_key(node: &Bound<'_, PyAny>) -> PyResult<String> {
    FxGraphModule::node_target_str(node)
}

fn normalize_value(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if value.is_none() {
        return Ok("None".into());
    }

    if value.getattr("name").is_ok() && value.getattr("op").is_ok() {
        return Ok(format!("Node({})", FxGraphModule::node_name(value)?));
    }

    if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
        let items = value.iter()?.collect::<PyResult<Vec<_>>>()?;
        let mut out = String::from("[");
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&normalize_value(item)?);
        }
        out.push(']');
        return Ok(out);
    }

    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut items: Vec<(String, String)> = Vec::new();
        for (k, v) in dict.iter() {
            items.push((k.repr()?.extract()?, normalize_value(&v)?));
        }
        items.sort_by(|a, b| a.0.cmp(&b.0));
        let body = items
            .into_iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",");
        return Ok(format!("{{{body}}}"));
    }

    if value.is_instance_of::<PyString>() {
        return Ok(format!("str({})", value.extract::<String>()?));
    }

    Ok(value.repr()?.extract()?)
}

fn normalize_node_args(args: &[Bound<'_, PyAny>]) -> PyResult<String> {
    let mut out = String::from("[");
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&normalize_value(arg)?);
    }
    out.push(']');
    Ok(out)
}

fn build_user_signature(node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if FxGraphModule::node_op(node)? != "call_function" {
        return Ok(None);
    }

    let args = FxGraphModule::node_args(node)?;
    let kwargs = FxGraphModule::node_kwargs(node)?;

    let norm_args = normalize_node_args(&args)?;
    let norm_kwargs = normalize_value(&kwargs)?;
    Ok(Some(format!(
        "{}|{}|{}|{}",
        FxGraphModule::node_op(node)?,
        target_key(node)?,
        norm_args,
        norm_kwargs
    )))
}

fn candidate_groups<'py>(
    producer: &Bound<'py, PyAny>,
) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
    let mut users_by_target: HashMap<(String, String), Vec<Bound<'_, PyAny>>> = HashMap::new();
    for user in &FxGraphModule::node_users(producer)? {
        if !node_is_live(user)? {
            continue;
        }
        let op = FxGraphModule::node_op(user)?;
        if op != "call_function" {
            continue;
        }
        let key = (op, target_key(user)?);
        users_by_target.entry(key).or_default().push(user.clone());
    }

    Ok(users_by_target
        .into_values()
        .filter(|group| group.len() > 1)
        .collect())
}

impl FxPass for FxFuseDuplicateUsersPass {
    fn name(&self) -> &str {
        "FxFuseDuplicateUsers"
    }

    fn run(&self, gm: &FxGraphModule) -> PyResult<bool> {
        let mut modified = false;
        let mut producers: VecDeque<Bound<'_, PyAny>> = gm.nodes()?.into();

        while let Some(producer) = producers.pop_front() {
            if !node_is_live(&producer)? {
                continue;
            }

            if FxGraphModule::node_users(&producer)?.len() < 2 {
                continue;
            }

            let mut signature_to_user: HashMap<String, Bound<'_, PyAny>> = HashMap::new();
            for group in candidate_groups(&producer)? {
                for user in group {
                    let Some(signature) = build_user_signature(&user)? else {
                        continue;
                    };

                    let Some(representative) = signature_to_user.get(&signature) else {
                        signature_to_user.insert(signature, user);
                        continue;
                    };

                    if user.is(representative) {
                        continue;
                    }

                    FxGraphModule::replace_all_uses_with(&user, representative)?;
                    gm.erase_node(&user)?;
                    modified = true;

                    producers.push_back(producer.clone());
                    producers.push_back(representative.clone());
                }
            }
        }

        if modified {
            gm.recompile()?;
        }
        Ok(modified)
    }
}

#[cfg(test)]
mod tests {
    // Integration tests live in Python.
}
