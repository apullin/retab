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

pub(super) fn kwarg_or_arg_i32_list(node: &Node, name: &str, idx: usize) -> Vec<i32> {
    let kw = kwarg_i32_list(node, name);
    if kw.is_empty() {
        arg_i32_list(node, idx)
    } else {
        kw
    }
}

pub(super) fn kwarg_or_arg_i32_list_or(
    node: &Node,
    name: &str,
    idx: usize,
    default: Vec<i32>,
) -> Vec<i32> {
    i32_list_or(kwarg_or_arg_i32_list(node, name, idx), default)
}

pub(super) fn kwarg_or_arg_i32_list_or_repeat(
    node: &Node,
    name: &str,
    idx: usize,
    len: usize,
    default: i32,
) -> Vec<i32> {
    i32_list_or_repeat(kwarg_or_arg_i32_list(node, name, idx), len, default)
}

pub(super) fn arg_or_kwarg_i32_list(node: &Node, idx: usize, name: &str) -> Vec<i32> {
    let arg = arg_i32_list(node, idx);
    if arg.is_empty() {
        kwarg_i32_list(node, name)
    } else {
        arg
    }
}

pub(super) fn i32_list_or(values: Vec<i32>, default: Vec<i32>) -> Vec<i32> {
    if values.is_empty() {
        default
    } else {
        values
    }
}

pub(super) fn i32_list_or_repeat(values: Vec<i32>, len: usize, default: i32) -> Vec<i32> {
    i32_list_or(values, vec![default; len])
}

pub(super) fn i32_pair_or_default(values: &[i32], default: i32) -> (i32, i32) {
    (
        values.first().copied().unwrap_or(default),
        values.get(1).copied().unwrap_or(default),
    )
}

pub(super) fn i32_pair_or_first(values: &[i32], default: i32) -> (i32, i32) {
    let first = values.first().copied().unwrap_or(default);
    (first, values.get(1).copied().unwrap_or(first))
}

pub(super) fn i32_quad_or_default(values: &[i32], default: i32) -> (i32, i32, i32, i32) {
    (
        values.first().copied().unwrap_or(default),
        values.get(1).copied().unwrap_or(default),
        values.get(2).copied().unwrap_or(default),
        values.get(3).copied().unwrap_or(default),
    )
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

pub(super) fn arg_f64(node: &Node, idx: usize, default: f64) -> f64 {
    node.args.get(idx).map_or(default, Arg::as_f64)
}

pub(super) fn arg_string(node: &Node, idx: usize) -> Option<String> {
    match node.args.get(idx) {
        Some(Arg::String(value)) => Some(value.clone()),
        _ => None,
    }
}

pub(super) fn arg_bool(node: &Node, idx: usize, default: bool) -> bool {
    match node.args.get(idx) {
        Some(arg) => arg.as_bool().unwrap_or(default),
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::{i32_list_or_repeat, i32_pair_or_default, i32_pair_or_first, i32_quad_or_default};

    #[test]
    fn i32_sequence_defaults_preserve_pair_semantics() {
        assert_eq!(i32_list_or_repeat(Vec::new(), 3, 1), vec![1, 1, 1]);
        assert_eq!(i32_list_or_repeat(vec![2, 3], 3, 1), vec![2, 3]);
        assert_eq!(i32_pair_or_default(&[3], 2), (3, 2));
        assert_eq!(i32_pair_or_first(&[3], 2), (3, 3));
        assert_eq!(i32_pair_or_first(&[], 2), (2, 2));
        assert_eq!(i32_quad_or_default(&[1, 2], 0), (1, 2, 0, 0));
    }
}
