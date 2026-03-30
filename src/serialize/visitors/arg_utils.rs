use super::*;

pub(super) fn kwarg_i32_list(node: &Node, name: &str) -> Vec<i32> {
    for (k, v) in &node.kwargs {
        if k == name {
            return v
                .as_int_list()
                .map(|list| list.iter().map(|&x| x as i32).collect())
                .unwrap_or_default();
        }
    }
    Vec::new()
}

pub(super) fn arg_i32_list(node: &Node, idx: usize) -> Vec<i32> {
    match node.args.get(idx) {
        Some(arg) => arg
            .as_int_list()
            .map(|list| list.iter().map(|&x| x as i32).collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

pub(super) fn kwarg_i32(node: &Node, name: &str, default: i32) -> i32 {
    for (k, v) in &node.kwargs {
        if k == name {
            return v.as_i64().map(|value| value as i32).unwrap_or(default);
        }
    }
    default
}

pub(super) fn kwarg_bool(node: &Node, name: &str, default: bool) -> bool {
    for (k, v) in &node.kwargs {
        if k == name {
            return v.as_bool().unwrap_or(default);
        }
    }
    default
}

pub(super) fn kwarg_string(node: &Node, name: &str) -> Option<String> {
    for (k, v) in &node.kwargs {
        if k == name {
            if let Arg::String(value) = v {
                return Some(value.clone());
            }
        }
    }
    None
}

pub(super) fn arg_i64(node: &Node, idx: usize, default: i64) -> i64 {
    match node.args.get(idx) {
        Some(arg) => arg.as_i64().unwrap_or(default),
        _ => default,
    }
}

pub(super) fn arg_bool(node: &Node, idx: usize, default: bool) -> bool {
    match node.args.get(idx) {
        Some(arg) => arg.as_bool().unwrap_or(default),
        _ => default,
    }
}
