//! rc4 graph dumps as layout-test fixtures and expectations.
//!
//! The upstream ports in `p1_upstream_tests` were confirmed by running rc4's
//! tests with each layout pass's `__call__` instrumented to dump the graph it
//! received and the graph it returned. The dumps were reduced to a canonical
//! text form, one line per node:
//!
//! ```text
//! %x = placeholder [1, 2, 3, 4] float32 dim_order=[0, 2, 3, 1]
//! %0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 2, 3, 4] float32 iq={0:int8@0.5:-128}
//! output [%0]
//! ```
//!
//! - Placeholders keep their names; calls are numbered `%0, %1, ...` in graph
//!   order, so new nodes renumber the calls after them.
//! - A call keeps its tensor operands (a list of tensors stays a list) and,
//!   for `permute_copy` and `view_copy`, the int list. Other literals and all
//!   kwargs are dropped: none of the P1 passes reads them.
//! - The value is rc4's `meta["val"]` shape and dtype, or `-` when it is not a
//!   tensor (a tuple, or a TOSA shape).
//! - Per-tensor `input_qparams`/`output_qparams` maps follow as `iq={...}`
//!   and `oq={...}` (`index:dtype@scale:zp`, scales as Rust prints an `f64`).
//!   A map holding a per-channel entry is left out.
//!
//! Fixture lines may add two annotations that dumps leave out: `param` (rc4's
//! `is_param_node`: a parameter, buffer or lifted constant, which RETAB marks
//! with tensor data) and `dim_order=[...]` (a non-identity
//! `meta["val"].dim_order()`, which RETAB imports as `val_dim_order`).

use std::collections::HashMap;

use crate::fx_ir::{IrArg, IrPass, NodeId, NodeMeta, NodeOp, QParam, QParamMap};
use crate::fx_meta::TensorFacts;
use crate::layout::testing::{ints, LayoutGraph, PERMUTE, VIEW};
use crate::test_ir::{f32_tensor, parameter};

/// Build a graph from fixture lines. Calls are named `n<label>`.
pub(crate) fn parse(lines: &[&str]) -> LayoutGraph {
    let mut g = LayoutGraph::new();
    let mut ids: HashMap<String, NodeId> = HashMap::new();
    for &line in lines {
        if let Some(rest) = line.strip_prefix("output ") {
            let (inner, _) = bracketed(rest.trim(), '[', ']', line);
            let outputs: Vec<NodeId> = split_top(inner)
                .into_iter()
                .map(|item| lookup(&ids, item, line))
                .collect();
            g.output(&outputs);
            continue;
        }
        let (label, rhs) = line
            .split_once(" = ")
            .unwrap_or_else(|| panic!("no ` = ` in {line:?}"));
        let label = label
            .strip_prefix('%')
            .unwrap_or_else(|| panic!("a label starts with % in {line:?}"));
        let id = if let Some(value) = rhs.strip_prefix("placeholder ") {
            let (meta, param) = parse_value(value, line);
            add_placeholder(&mut g, label, meta, param)
        } else {
            let open = rhs
                .find('(')
                .unwrap_or_else(|| panic!("no operands in {line:?}"));
            let close = open
                + rhs[open..]
                    .rfind(')')
                    .unwrap_or_else(|| panic!("unclosed operands in {line:?}"));
            let args = split_top(&rhs[open + 1..close])
                .into_iter()
                .map(|arg| parse_arg(arg, &ids, line))
                .collect();
            let (meta, param) = parse_value(&rhs[close + 1..], line);
            assert!(!param, "only placeholders are params: {line:?}");
            g.graph_mut().add_call_function_with_kwargs(
                format!("n{label}"),
                &rhs[..open],
                args,
                HashMap::new(),
                meta,
            )
        };
        assert!(
            ids.insert(label.to_string(), id).is_none(),
            "label %{label} defined twice"
        );
    }
    g
}

/// The live graph in canonical form.
pub(crate) fn dump(g: &LayoutGraph) -> Vec<String> {
    let graph = g.graph();
    let mut labels: HashMap<NodeId, String> = HashMap::new();
    let mut calls = 0;
    let mut lines = Vec::new();
    for id in graph.topological_live_node_ids() {
        let node = graph.node(id);
        let label_of = |labels: &HashMap<NodeId, String>, input: NodeId| {
            labels
                .get(&input)
                .cloned()
                .unwrap_or_else(|| format!("<unordered {}>", graph.node(input).name))
        };
        match node.op {
            NodeOp::Placeholder => {
                let label = format!("%{}", node.name);
                lines.push(format!("{label} = placeholder {}", value(&node.meta)));
                labels.insert(id, label);
            }
            NodeOp::Output => {
                let mut outputs = Vec::new();
                for arg in &node.args {
                    node_refs(arg, &mut outputs);
                }
                let outputs: Vec<String> =
                    outputs.into_iter().map(|o| label_of(&labels, o)).collect();
                lines.push(format!("output [{}]", outputs.join(", ")));
            }
            _ => {
                let mut operands = Vec::new();
                for (index, arg) in node.args.iter().enumerate() {
                    match arg {
                        IrArg::Node(input) => operands.push(label_of(&labels, *input)),
                        IrArg::List(items) if items.iter().any(|i| i.as_node().is_some()) => {
                            let mut refs = Vec::new();
                            node_refs(arg, &mut refs);
                            let refs: Vec<String> =
                                refs.into_iter().map(|r| label_of(&labels, r)).collect();
                            operands.push(format!("[{}]", refs.join(", ")));
                        }
                        _ if index == 1 && (node.target == PERMUTE || node.target == VIEW) => {
                            let list = arg
                                .as_int_list()
                                .unwrap_or_else(|| panic!("{} has no int list", node.name));
                            operands.push(format!("{list:?}"));
                        }
                        _ => {}
                    }
                }
                let label = format!("%{calls}");
                calls += 1;
                lines.push(format!(
                    "{label} = {}({}) {}",
                    node.target,
                    operands.join(", "),
                    value(&node.meta)
                ));
                labels.insert(id, label);
            }
        }
    }
    lines
}

/// `line` without the fixture-only annotations, as `dump` renders it.
pub(crate) fn without_annotations(line: &str) -> String {
    let mut line = line.replace(" param", "");
    if let Some(start) = line.find(" dim_order=[") {
        let end = start + line[start..].find(']').expect("a closed dim order") + 1;
        line.replace_range(start..end, "");
    }
    line
}

/// Panic with both graphs unless `g` renders as `expected`.
pub(crate) fn assert_graph(g: &LayoutGraph, expected: &[&str], what: &str) {
    let actual = dump(g);
    if actual
        .iter()
        .map(String::as_str)
        .ne(expected.iter().copied())
    {
        panic!(
            "{what} differs from rc4's\n--- RETAB\n{}\n--- rc4\n{}",
            actual.join("\n"),
            expected.join("\n")
        );
    }
}

/// Run `pass` on the graph rc4's pass received (`before`) and check it leaves
/// what rc4's pass returned (`after`), or changes nothing when `after` is
/// `None`. The fixture is first checked to render back as `before`, so the
/// lines double as the transcription check. Returns the graph for further
/// assertions.
pub(crate) fn check_pass(
    pass: &dyn IrPass,
    before: &[&str],
    after: Option<&[&str]>,
) -> LayoutGraph {
    let mut g = parse(before);
    let fixture: Vec<String> = before
        .iter()
        .map(|line| without_annotations(line))
        .collect();
    let fixture: Vec<&str> = fixture.iter().map(String::as_str).collect();
    assert_graph(&g, &fixture, "the fixture");
    let modified = pass.run(g.graph_mut());
    match after {
        Some(after) => {
            assert!(modified, "{} reported no change", pass.name());
            assert_graph(&g, after, "the rewritten graph");
        }
        None => {
            assert!(!modified, "{} reported a change", pass.name());
            assert_graph(&g, &fixture, "the graph");
        }
    }
    g
}

fn add_placeholder(g: &mut LayoutGraph, name: &str, meta: NodeMeta, param: bool) -> NodeId {
    let graph = g.graph_mut();
    let id = if param {
        let shape = meta.val_shape().expect("a param has a shape").to_vec();
        let numel = shape.iter().product::<i64>().max(1) as usize;
        parameter(graph, name, f32_tensor(shape, &vec![0.0; numel]))
    } else {
        graph.add_node(
            NodeOp::Placeholder,
            name.to_string(),
            name.to_string(),
            Vec::new(),
            NodeMeta::default(),
        )
    };
    graph.node_mut(id).meta = meta;
    id
}

fn parse_arg(text: &str, ids: &HashMap<String, NodeId>, line: &str) -> IrArg {
    if text.starts_with('%') {
        return IrArg::Node(lookup(ids, text, line));
    }
    let (inner, _) = bracketed(text, '[', ']', line);
    if inner.contains('%') {
        let items = split_top(inner)
            .into_iter()
            .map(|item| IrArg::Node(lookup(ids, item, line)))
            .collect();
        IrArg::List(items)
    } else {
        ints(&parse_ints(inner, line))
    }
}

fn parse_value(text: &str, line: &str) -> (NodeMeta, bool) {
    let mut meta = NodeMeta::default();
    let mut rest = text.trim();
    if let Some(after) = rest.strip_prefix('-') {
        rest = after.trim_start();
    } else {
        let (shape, after) = bracketed(rest, '[', ']', line);
        let shape = parse_ints(shape, line);
        let after = after.trim_start();
        let (dtype, after) = after.split_once(' ').unwrap_or((after, ""));
        assert!(!dtype.is_empty(), "no dtype in {line:?}");
        meta.set_tensor_facts(TensorFacts::new(shape, format!("torch.{dtype}")));
        rest = after.trim_start();
    }
    let mut param = false;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("param") {
            param = true;
            rest = after.trim_start();
        } else if let Some(after) = rest.strip_prefix("dim_order=") {
            let (order, after) = bracketed(after, '[', ']', line);
            let order = parse_ints(order, line)
                .into_iter()
                .map(|d| d as usize)
                .collect();
            meta.set_val_dim_order(order);
            rest = after.trim_start();
        } else if let Some(after) = rest.strip_prefix("iq=") {
            let (map, after) = bracketed(after, '{', '}', line);
            meta.set_input_qparams(parse_qparams(map, line));
            rest = after.trim_start();
        } else if let Some(after) = rest.strip_prefix("oq=") {
            let (map, after) = bracketed(after, '{', '}', line);
            meta.set_output_qparams(parse_qparams(map, line));
            rest = after.trim_start();
        } else {
            panic!("unknown annotation {rest:?} in {line:?}");
        }
    }
    (meta, param)
}

/// `index:dtype@scale:zp, ...`. qmin and qmax are not rendered, so they are
/// left at int8's range.
fn parse_qparams(text: &str, line: &str) -> QParamMap {
    split_top(text)
        .into_iter()
        .map(|entry| {
            let fields = entry.split_once(':').and_then(|(index, rest)| {
                let (dtype, rest) = rest.split_once('@')?;
                let (scale, zp) = rest.rsplit_once(':')?;
                Some((
                    index.parse().ok()?,
                    dtype,
                    scale.parse().ok()?,
                    zp.parse().ok()?,
                ))
            });
            let (index, dtype, scale, zp): (usize, &str, f64, i64) =
                fields.unwrap_or_else(|| panic!("bad qparam {entry:?} in {line:?}"));
            let qparam = QParam::per_tensor(scale, zp, -128, 127, format!("torch.{dtype}"));
            (index, qparam)
        })
        .collect()
}

fn value(meta: &NodeMeta) -> String {
    let mut text = match meta.tensor_facts() {
        Some(facts) => format!(
            "{:?} {}",
            facts.shape,
            facts.dtype.trim_start_matches("torch.")
        ),
        None => "-".to_string(),
    };
    for (key, map) in [("iq", meta.input_qparams()), ("oq", meta.output_qparams())] {
        let Some(map) = map else { continue };
        if map.values().any(QParam::is_per_channel) {
            continue;
        }
        let entries: Vec<String> = map
            .iter()
            .map(|(index, qparam)| {
                format!(
                    "{index}:{}@{}:{}",
                    qparam.dtype_str.trim_start_matches("torch."),
                    qparam.scale,
                    qparam.zp
                )
            })
            .collect();
        text.push_str(&format!(" {key}={{{}}}", entries.join(", ")));
    }
    text
}

fn node_refs(arg: &IrArg, refs: &mut Vec<NodeId>) {
    match arg {
        IrArg::Node(id) => refs.push(*id),
        IrArg::List(items) => items.iter().for_each(|item| node_refs(item, refs)),
        _ => {}
    }
}

fn lookup(ids: &HashMap<String, NodeId>, text: &str, line: &str) -> NodeId {
    let label = text.trim().strip_prefix('%').unwrap_or(text.trim());
    *ids.get(label)
        .unwrap_or_else(|| panic!("%{label} is used before it is defined in {line:?}"))
}

/// The text inside a leading `open ... close` pair and the text after it.
fn bracketed<'a>(text: &'a str, open: char, close: char, line: &str) -> (&'a str, &'a str) {
    let text = text.trim_start();
    assert!(text.starts_with(open), "expected {open:?} in {line:?}");
    let end = text
        .find(close)
        .unwrap_or_else(|| panic!("expected {close:?} in {line:?}"));
    (&text[1..end], &text[end + 1..])
}

fn parse_ints(text: &str, line: &str) -> Vec<i64> {
    split_top(text)
        .into_iter()
        .map(|item| {
            item.parse()
                .unwrap_or_else(|_| panic!("bad int {item:?} in {line:?}"))
        })
        .collect()
}

/// Split at top-level commas, ignoring empty items.
fn split_top(text: &str) -> Vec<&str> {
    let mut items = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                items.push(text[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    items.push(text[start..].trim());
    items.retain(|item| !item.is_empty());
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAPH: &[&str] = &[
        "%w = placeholder [4] float32 param",
        "%x = placeholder [1, 2, 3, 4] int8 dim_order=[0, 2, 3, 1] iq={} oq={0:int8@0.5:-3}",
        "%0 = aten.permute_copy.default(%x, [0, 3, 1, 2]) [1, 4, 2, 3] int8",
        "%1 = aten.cat.default([%0, %0]) [2, 4, 2, 3] int8",
        "%2 = getitem(%1) -",
        "%3 = aten.mul.Tensor(%1, %w) [2, 4, 2, 3] int32 iq={0:int8@0.00000005960464477539063:0, 1:int8@2048:1}",
        "output [%3, %0]",
    ];

    #[test]
    fn fixtures_round_trip_without_their_annotations() {
        let g = parse(GRAPH);
        let expected: Vec<String> = GRAPH.iter().map(|line| without_annotations(line)).collect();
        assert_eq!(dump(&g), expected);
        let x = g.id("x");
        assert_eq!(
            g.graph().node(x).meta.val_dim_order(),
            Some(&[0, 2, 3, 1][..])
        );
        assert!(g.graph().node(g.id("w")).tensor.is_some());
        assert!(g.graph().node(x).tensor.is_none());
    }

    #[test]
    fn annotations_are_stripped() {
        assert_eq!(
            without_annotations("%x = placeholder [1] int8 param dim_order=[0] oq={}"),
            "%x = placeholder [1] int8 oq={}"
        );
    }
}
