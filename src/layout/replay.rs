//! Replays ExecuTorch v1.6.0-rc4's layout passes against the RETAB ports
//! (APU-135 P1).
//!
//! `tests/fixtures/layout_replay/gen_layout_replay.py` builds edge-dialect
//! torch.fx graphs, runs the real rc4 pass on each and records the graph before
//! and after, one fixture per pass (the format is in the generator's
//! docstring). [`check`] rebuilds every "before" graph as an `IrGraph`, runs
//! the port and compares the result with rc4's "after" graph:
//!
//! - nodes in graph order (`IrGraph::topological_live_node_ids`, which
//!   `LayoutCtx::finish`'s renumbering makes the ctx order), with each node's
//!   op, target, args and kwargs; node references compare by position, so
//!   names don't matter;
//! - each node's dtype and shape, and whether its `input_qparams` and
//!   `output_qparams` maps are absent, empty or hold which keys;
//! - placeholders' constant kinds;
//! - the output node's argument.
//!
//! Dim orders are recorded but compared only on request (`Options::dim_orders`).
//! rc4's passes end in an `ExportPass` retrace that recomputes every val, so a
//! node a pass never touched can still change strides (a broadcast op whose
//! operand became a rank-matched view, anything downstream of a normalized
//! input), while the ports leave untouched meta alone. And a normalized input's
//! val is `input_fake.reshape(boundary_shape)`, which torch returns as a
//! strided view whenever it can: its dim order then stays non-identity, and
//! which one depends on the strides of size-1 dims, which no `IrGraph` keeps.
//!
//! Dead code: rc4 often leaves dead nodes for a later pass to remove (the pass
//! manager runs no DCE between passes, and MatchArgRanks leaves a dead view
//! behind for a repeated argument), while a port may never create them. By
//! default both graphs are compared after dropping every call node the output
//! doesn't reach, so dead code alone is never a mismatch; placeholders always
//! stay. `Options::dead_nodes` compares the whole graphs instead.
//!
//! Import emulation: the "before" graph looks like what RETAB's importer
//! builds. An empty list or tuple becomes `IrArg::None` (`resolve_arg` in
//! src/fx_ir.rs), so a fixture `[]` compares equal to `None`. Every node gets
//! its val's dim order, the identity unless the fixture says otherwise, as
//! `NodeMeta::from_fx` records `val.dim_order()`. A `kind=` placeholder gets a
//! zero-filled `NodeTensor`, which is how RETAB tells parameters, buffers and
//! lifted constants from runtime inputs. QParam maps keep the fixture's keys
//! with placeholder values; empty maps stay empty (the importer drops them,
//! but RETAB passes create them), so the ports' handling of both is visible.
//!
//! Usage: a pass's replay test is one line,
//! `replay::check(&IrFooPass, include_str!("../../tests/fixtures/layout_replay/foo.txt"))`.
//! [`check_known`] names cases still expected to differ, with a reason, and
//! fails once one matches; [`check_cases`] and [`check_cases_with`] replay
//! chosen cases, as the ignored BUG tests in `p1_replay_tests` do. There,
//! `replay_report` prints per-pass statistics under every comparison mode and
//! `replay_external` replays fixtures too large to commit (a fuzzing run).

use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Write as _};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::fx_ir::{
    IrArg, IrGraph, IrPass, NodeId, NodeMeta, NodeOp, NodeTensor, PlaceholderTensorBinding, QParam,
    QParamMap, TensorSource,
};
use crate::graph::ir::DType;
use crate::lowering_effects::LoweringInputKind;
use crate::store::TensorData;

/// An argument, with node references as positions in the graph.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Arg {
    Node(usize),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Dtype(String),
    List(Vec<Arg>),
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Op {
    /// `constant` is `param`, `buffer` or `constant` for a constant placeholder.
    Placeholder {
        name: String,
        constant: Option<String>,
    },
    Call {
        target: String,
        args: Vec<Arg>,
        kwargs: BTreeMap<String, Arg>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    pub(crate) op: Op,
    /// `(dtype, shape)`; only a RETAB node can lack one.
    pub(crate) val: Option<(String, Vec<i64>)>,
    /// `None` is the identity.
    pub(crate) dim_order: Option<Vec<usize>>,
    pub(crate) q_in: Option<Vec<usize>>,
    pub(crate) q_out: Option<Vec<usize>>,
    pub(crate) dead: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) output: Arg,
}

#[derive(Debug, Clone)]
pub(crate) struct Case {
    pub(crate) name: String,
    pub(crate) before: Graph,
    /// The `modified` flag rc4's pass returned.
    pub(crate) modified: bool,
    pub(crate) after: Graph,
}

/// Which dim orders [`compare`] checks; see the module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum DimOrders {
    #[default]
    Ignore,
    Placeholders,
    All,
}

/// What [`compare`] checks beyond the defaults in the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Options {
    pub(crate) dim_orders: DimOrders,
    /// Compare dead nodes too instead of dropping them from both graphs.
    pub(crate) dead_nodes: bool,
}

/// The first difference between rc4's graph and RETAB's.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mismatch {
    /// A short label for grouping: `target`, `args`, `shape`, `panic`, ...
    pub(crate) class: &'static str,
    pub(crate) detail: String,
}

#[derive(Debug, Clone)]
pub(crate) struct CaseResult {
    pub(crate) name: String,
    /// What the port's `run` returned; `None` if it panicked.
    pub(crate) retab_modified: Option<bool>,
    pub(crate) rc4_modified: bool,
    pub(crate) mismatch: Option<Mismatch>,
    /// Before, rc4 after and RETAB after, printed, for a mismatch.
    pub(crate) report: String,
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Replay every case of `fixture` through `pass`; every case must match rc4.
pub(crate) fn check(pass: &dyn IrPass, fixture: &str) {
    check_known(pass, fixture, &[]);
}

/// Like [`check`], except that each `(case, reason)` in `known` must still
/// differ from rc4, so the list shrinks as the port is fixed.
pub(crate) fn check_known(pass: &dyn IrPass, fixture: &str, known: &[(&str, &str)]) {
    let results = replay(pass, fixture, Options::default());
    let mut problems = Vec::new();
    for (name, _) in known {
        match results.iter().find(|result| result.name == *name) {
            None => problems.push(format!("known mismatch {name} is not in the fixture")),
            Some(result) if result.mismatch.is_none() => problems.push(format!(
                "case {name} now matches rc4; remove it from the known mismatches"
            )),
            Some(_) => {}
        }
    }
    let failures: Vec<&CaseResult> = results
        .iter()
        .filter(|result| result.mismatch.is_some())
        .filter(|result| !known.iter().any(|(name, _)| *name == result.name))
        .collect();
    if failures.is_empty() && problems.is_empty() {
        return;
    }
    let mut message = format!(
        "{}: {} of {} rc4 replay cases differ ({})\n",
        pass.name(),
        failures.len(),
        results.len(),
        class_counts(failures.iter().copied())
    );
    for problem in &problems {
        let _ = writeln!(message, "{problem}");
    }
    for failure in failures.iter().take(3) {
        message.push_str(&failure.report);
    }
    if failures.len() > 3 {
        let names: Vec<&str> = failures.iter().map(|result| result.name.as_str()).collect();
        let _ = writeln!(message, "all differing cases: {}", names.join(" "));
    }
    panic!("{message}");
}

/// Replay only the named cases; each must match rc4.
pub(crate) fn check_cases(pass: &dyn IrPass, fixture: &str, names: &[&str]) {
    check_cases_with(pass, fixture, names, Options::default());
}

/// [`check_cases`] with more checks, such as every node's dim order.
pub(crate) fn check_cases_with(pass: &dyn IrPass, fixture: &str, names: &[&str], options: Options) {
    let cases = parse(fixture).unwrap_or_else(|error| panic!("bad replay fixture: {error}"));
    let mut message = String::new();
    for name in names {
        let case = cases
            .iter()
            .find(|case| case.name == *name)
            .unwrap_or_else(|| panic!("no case {name} in the fixture"));
        let result = replay_case(pass, case, options);
        if result.mismatch.is_some() {
            message.push_str(&result.report);
        }
    }
    assert!(
        message.is_empty(),
        "{}: cases differ from rc4\n{message}",
        pass.name()
    );
}

/// Replay every case of `fixture` through `pass`.
pub(crate) fn replay(pass: &dyn IrPass, fixture: &str, options: Options) -> Vec<CaseResult> {
    let cases = parse(fixture).unwrap_or_else(|error| panic!("bad replay fixture: {error}"));
    cases
        .iter()
        .map(|case| replay_case(pass, case, options))
        .collect()
}

pub(crate) fn replay_case(pass: &dyn IrPass, case: &Case, options: Options) -> CaseResult {
    let mut graph = build(&case.before);
    let run = catch_unwind(AssertUnwindSafe(|| pass.run(&mut graph)));
    let (retab_modified, actual) = match run {
        Ok(modified) => (Some(modified), snapshot(&graph)),
        Err(payload) => (None, Err(panic_message(payload.as_ref()))),
    };
    let mismatch = match &actual {
        Ok(actual) => compare(&case.after, actual, options).err(),
        Err(error) => Some(Mismatch {
            class: if retab_modified.is_none() {
                "panic"
            } else {
                "invalid-graph"
            },
            detail: error.clone(),
        }),
    };
    let report = match &mismatch {
        None => String::new(),
        Some(mismatch) => {
            let actual = match &actual {
                Ok(graph) => graph.to_string(),
                Err(error) => format!("    <{error}>\n"),
            };
            format!(
                "case {} ({}): {}\n  before:\n{}  rc4 after (modified={}):\n{}  RETAB after (modified={}):\n{}",
                case.name,
                mismatch.class,
                mismatch.detail,
                case.before,
                case.modified,
                case.after,
                retab_modified.map_or("panic".to_string(), |modified| modified.to_string()),
                actual
            )
        }
    };
    CaseResult {
        name: case.name.clone(),
        retab_modified,
        rc4_modified: case.modified,
        mismatch,
        report,
    }
}

/// `class count` pairs, most frequent first.
pub(crate) fn class_counts<'a>(results: impl Iterator<Item = &'a CaseResult>) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for result in results {
        if let Some(mismatch) = &result.mismatch {
            *counts.entry(mismatch.class).or_default() += 1;
        }
    }
    let mut counts: Vec<(&str, usize)> = counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    counts
        .iter()
        .map(|(class, count)| format!("{class} {count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    let text = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic");
    format!("the pass panicked: {text}")
}

// ---------------------------------------------------------------------------
// Fixture parsing
// ---------------------------------------------------------------------------

/// Parse a fixture into its cases.
pub(crate) fn parse(text: &str) -> Result<Vec<Case>, String> {
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line.trim_end()))
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'));
    let mut cases = Vec::new();
    while let Some((number, line)) = lines.next() {
        let name = line
            .strip_prefix("case ")
            .ok_or_else(|| format!("line {number}: expected `case <name>`, got {line:?}"))?;
        let at = |number: usize, error: String| format!("case {name}, line {number}: {error}");
        match lines.next() {
            Some((_, "before")) => {}
            Some((number, line)) => {
                return Err(at(number, format!("expected `before`, got {line:?}")))
            }
            None => return Err(format!("case {name}: missing `before`")),
        }
        let before = parse_graph(&mut lines).map_err(|(number, error)| at(number, error))?;
        let (number, header) = lines
            .next()
            .ok_or_else(|| format!("case {name}: missing `after`"))?;
        let rest = header.strip_prefix("after modified=").ok_or_else(|| {
            at(
                number,
                format!("expected `after modified=...`, got {header:?}"),
            )
        })?;
        let (flag, same) = match rest.strip_suffix(" same") {
            Some(flag) => (flag, true),
            None => (rest, false),
        };
        let modified = match flag {
            "true" => true,
            "false" => false,
            _ => return Err(at(number, format!("bad modified flag {flag:?}"))),
        };
        let after = if same {
            before.clone()
        } else {
            parse_graph(&mut lines).map_err(|(number, error)| at(number, error))?
        };
        match lines.next() {
            Some((_, "end")) => {}
            Some((number, line)) => {
                return Err(at(number, format!("expected `end`, got {line:?}")))
            }
            None => return Err(format!("case {name}: missing `end`")),
        }
        for (graph, label) in [(&before, "before"), (&after, "after")] {
            check_dead_flags(graph).map_err(|error| format!("case {name} ({label}): {error}"))?;
        }
        cases.push(Case {
            name: name.to_string(),
            before,
            modified,
            after,
        });
    }
    Ok(cases)
}

/// Node lines up to and including the output line.
fn parse_graph<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
) -> Result<Graph, (usize, String)> {
    let mut nodes = Vec::new();
    for (number, line) in lines {
        if let Some(rest) = line.strip_prefix("output(") {
            let mut cursor = Cursor::new(rest);
            let output = cursor.arg().map_err(|error| (number, error))?;
            cursor.expect(")").map_err(|error| (number, error))?;
            cursor.end().map_err(|error| (number, error))?;
            check_refs(&output, nodes.len()).map_err(|error| (number, error))?;
            return Ok(Graph { nodes, output });
        }
        let node = parse_node(line, nodes.len()).map_err(|error| (number, error))?;
        if let Op::Call { args, kwargs, .. } = &node.op {
            for arg in args.iter().chain(kwargs.values()) {
                check_refs(arg, nodes.len()).map_err(|error| (number, error))?;
            }
        }
        nodes.push(node);
    }
    Err((0, "missing output line".to_string()))
}

/// Every node `arg` reads must come before the `defined`-th node.
fn check_refs(arg: &Arg, defined: usize) -> Result<(), String> {
    match arg {
        Arg::Node(position) if *position >= defined => Err(format!(
            "%{position} is used before it is defined ({defined} nodes so far)"
        )),
        Arg::List(items) => items.iter().try_for_each(|item| check_refs(item, defined)),
        _ => Ok(()),
    }
}

fn parse_node(line: &str, position: usize) -> Result<Node, String> {
    let mut cursor = Cursor::new(line);
    cursor.expect("%")?;
    let index = cursor.int()?;
    if index != position as i64 {
        return Err(format!("expected %{position}, got %{index}"));
    }
    cursor.expect(" = ")?;
    let op = if cursor.eat("placeholder ") {
        let name = cursor.until(" : ")?.to_string();
        Op::Placeholder {
            name,
            constant: None,
        }
    } else {
        let target = cursor.until("(")?.to_string();
        cursor.expect("(")?;
        let (args, kwargs) = cursor.call_args()?;
        Op::Call {
            target,
            args,
            kwargs,
        }
    };
    cursor.expect(" : ")?;
    let dtype = cursor.until("[")?.to_string();
    cursor.expect("[")?;
    let mut shape = Vec::new();
    while !cursor.eat("]") {
        if !shape.is_empty() {
            cursor.expect(", ")?;
        }
        shape.push(cursor.int()?);
    }
    let mut node = Node {
        op,
        val: Some((dtype, shape)),
        dim_order: None,
        q_in: None,
        q_out: None,
        dead: false,
    };
    for note in cursor.rest().split(' ').filter(|note| !note.is_empty()) {
        if note == "dead" {
            node.dead = true;
        } else if let Some(order) = note.strip_prefix("dim_order=") {
            node.dim_order = Some(parse_usizes(order, '[', ']')?);
        } else if let Some(kind) = note.strip_prefix("kind=") {
            match &mut node.op {
                Op::Placeholder { constant, .. } => *constant = Some(kind.to_string()),
                Op::Call { .. } => return Err("kind= on a call node".to_string()),
            }
        } else if let Some(spec) = note.strip_prefix("q=") {
            if spec != "none" {
                for map in spec.split('/') {
                    if let Some(keys) = map.strip_prefix("in") {
                        node.q_in = Some(parse_usizes(keys, '{', '}')?);
                    } else if let Some(keys) = map.strip_prefix("out") {
                        node.q_out = Some(parse_usizes(keys, '{', '}')?);
                    } else {
                        return Err(format!("bad qparam spec {spec:?}"));
                    }
                }
            }
        } else {
            return Err(format!("unknown annotation {note:?}"));
        }
    }
    Ok(node)
}

fn parse_usizes(text: &str, open: char, close: char) -> Result<Vec<usize>, String> {
    let inner = text
        .strip_prefix(open)
        .and_then(|rest| rest.strip_suffix(close))
        .ok_or_else(|| format!("expected {open}...{close}, got {text:?}"))?;
    inner
        .split(',')
        .filter(|item| !item.is_empty())
        .map(|item| item.parse().map_err(|_| format!("bad index {item:?}")))
        .collect()
}

/// rc4's `dead` flags must agree with liveness computed from the output.
fn check_dead_flags(graph: &Graph) -> Result<(), String> {
    let reached = reached(graph);
    for (position, node) in graph.nodes.iter().enumerate() {
        let dead = matches!(node.op, Op::Call { .. }) && !reached[position];
        if dead != node.dead {
            return Err(format!(
                "%{position} is marked dead={} but is dead={dead}",
                node.dead
            ));
        }
    }
    Ok(())
}

struct Cursor<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, position: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.text[self.position..]
    }

    fn eat(&mut self, token: &str) -> bool {
        let found = self.rest().starts_with(token);
        if found {
            self.position += token.len();
        }
        found
    }

    fn expect(&mut self, token: &str) -> Result<(), String> {
        if self.eat(token) {
            Ok(())
        } else {
            Err(format!("expected {token:?} at {:?}", self.rest()))
        }
    }

    fn end(&self) -> Result<(), String> {
        if self.rest().is_empty() {
            Ok(())
        } else {
            Err(format!("trailing text {:?}", self.rest()))
        }
    }

    /// The text up to `token`, which stays unread.
    fn until(&mut self, token: &str) -> Result<&'a str, String> {
        let rest = self.rest();
        let length = rest
            .find(token)
            .ok_or_else(|| format!("expected {token:?} in {rest:?}"))?;
        self.position += length;
        Ok(&rest[..length])
    }

    fn int(&mut self) -> Result<i64, String> {
        let rest = self.rest();
        let length = rest
            .char_indices()
            .find(|&(index, c)| !(c.is_ascii_digit() || (index == 0 && c == '-')))
            .map_or(rest.len(), |(index, _)| index);
        let value = rest[..length]
            .parse()
            .map_err(|_| format!("expected an integer at {rest:?}"))?;
        self.position += length;
        Ok(value)
    }

    /// Positional args and then kwargs, up to and including the closing `)`.
    fn call_args(&mut self) -> Result<(Vec<Arg>, BTreeMap<String, Arg>), String> {
        let mut args = Vec::new();
        let mut kwargs = BTreeMap::new();
        let mut first = true;
        while !self.eat(")") {
            if !first {
                self.expect(", ")?;
            }
            first = false;
            match self.keyword() {
                Some(key) => {
                    let value = self.arg()?;
                    kwargs.insert(key.to_string(), value);
                }
                None if kwargs.is_empty() => args.push(self.arg()?),
                None => return Err(format!("positional arg after kwargs at {:?}", self.rest())),
            }
        }
        Ok((args, kwargs))
    }

    /// `name=` at the cursor, consumed.
    fn keyword(&mut self) -> Option<&'a str> {
        let rest = self.rest();
        let length = rest
            .char_indices()
            .find(|&(index, c)| {
                !(c == '_' || c.is_ascii_alphabetic() || (index > 0 && c.is_ascii_digit()))
            })
            .map(|(index, _)| index)?;
        if length == 0 || !rest[length..].starts_with('=') {
            return None;
        }
        self.position += length + 1;
        Some(&rest[..length])
    }

    fn arg(&mut self) -> Result<Arg, String> {
        if self.eat("%") {
            let position = self.int()?;
            return usize::try_from(position)
                .map(Arg::Node)
                .map_err(|_| format!("bad node reference %{position}"));
        }
        if self.eat("[") {
            let mut items = Vec::new();
            while !self.eat("]") {
                if !items.is_empty() {
                    self.expect(", ")?;
                }
                items.push(self.arg()?);
            }
            return Ok(Arg::List(items));
        }
        if self.eat("\"") {
            let text = self.until("\"")?.to_string();
            self.expect("\"")?;
            return Ok(Arg::Str(text));
        }
        let rest = self.rest();
        let length = rest.find([',', ')', ']']).unwrap_or(rest.len());
        let token = &rest[..length];
        self.position += length;
        Ok(match token {
            "True" => Arg::Bool(true),
            "False" => Arg::Bool(false),
            "None" => Arg::None,
            _ if token.starts_with("torch.") => Arg::Dtype(token.to_string()),
            _ => match token.parse::<i64>() {
                Ok(value) => Arg::Int(value),
                Err(_) => Arg::Float(
                    token
                        .parse()
                        .map_err(|_| format!("bad argument {token:?}"))?,
                ),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Building the "before" IrGraph
// ---------------------------------------------------------------------------

/// The `IrGraph` RETAB's importer would build for `graph`.
pub(crate) fn build(graph: &Graph) -> IrGraph {
    let mut ir = IrGraph::empty();
    let mut ids: Vec<NodeId> = Vec::with_capacity(graph.nodes.len());
    for (position, node) in graph.nodes.iter().enumerate() {
        let meta = node_meta(node);
        let id = match &node.op {
            Op::Placeholder { name, constant } => {
                let id = ir.add_node(
                    NodeOp::Placeholder,
                    name.clone(),
                    name.clone(),
                    Vec::new(),
                    meta,
                );
                if let Some(kind) = constant {
                    ir.node_mut(id).tensor = Some(zero_tensor(kind, name, node));
                }
                id
            }
            Op::Call {
                target,
                args,
                kwargs,
            } => {
                let args = args.iter().map(|arg| ir_arg(arg, &ids)).collect();
                let kwargs: HashMap<String, IrArg> = kwargs
                    .iter()
                    .map(|(key, value)| (key.clone(), ir_arg(value, &ids)))
                    .collect();
                ir.add_node_with_kwargs(
                    NodeOp::CallFunction,
                    format!("n{position}"),
                    target.clone(),
                    args,
                    kwargs,
                    meta,
                )
            }
        };
        ids.push(id);
    }
    ir.add_node(
        NodeOp::Output,
        "output".to_string(),
        "output".to_string(),
        vec![ir_arg(&graph.output, &ids)],
        NodeMeta::default(),
    );
    ir
}

fn ir_arg(arg: &Arg, ids: &[NodeId]) -> IrArg {
    match arg {
        Arg::Node(position) => IrArg::Node(ids[*position]),
        Arg::Int(value) => IrArg::Int(*value),
        Arg::Float(value) => IrArg::Float(*value),
        Arg::Bool(value) => IrArg::Bool(*value),
        Arg::Str(text) => IrArg::Str(text.clone()),
        Arg::Dtype(dtype) => IrArg::Dtype(dtype.clone()),
        // `resolve_arg` turns an empty list or tuple into None.
        Arg::List(items) if items.is_empty() => IrArg::None,
        Arg::List(items) => IrArg::List(items.iter().map(|item| ir_arg(item, ids)).collect()),
        Arg::None => IrArg::None,
    }
}

fn node_meta(node: &Node) -> NodeMeta {
    let (dtype, shape) = node.val.clone().expect("fixture nodes have a val");
    let rank = shape.len();
    let mut meta = NodeMeta::tensor(shape, dtype);
    meta.set_val_dim_order(
        node.dim_order
            .clone()
            .unwrap_or_else(|| (0..rank).collect()),
    );
    if let Some(keys) = &node.q_in {
        meta.set_input_qparams(qparam_map(keys));
    }
    if let Some(keys) = &node.q_out {
        meta.set_output_qparams(qparam_map(keys));
    }
    meta
}

fn qparam_map(keys: &[usize]) -> QParamMap {
    QParamMap::from_pairs(keys.iter().map(|&key| {
        (
            key,
            QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string()),
        )
    }))
}

fn zero_tensor(kind: &str, name: &str, node: &Node) -> NodeTensor {
    let input_kind = match kind {
        "param" => LoweringInputKind::Parameter,
        "buffer" => LoweringInputKind::Buffer,
        "constant" => LoweringInputKind::LiftedTensorConstant,
        _ => panic!("unknown constant kind {kind:?}"),
    };
    let (dtype, shape) = node.val.clone().expect("fixture nodes have a val");
    let dtype = DType::from_torch_str(&dtype);
    let numel: i64 = shape.iter().product();
    NodeTensor {
        source: TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
            input_kind,
            target_name: format!("c_{name}"),
        }),
        data: TensorData {
            data: vec![0; numel as usize * dtype.storage_size_bytes()],
            shape,
            dtype,
        },
    }
}

// ---------------------------------------------------------------------------
// Reading the RETAB result
// ---------------------------------------------------------------------------

/// `ir`'s live nodes in graph order, in fixture form.
pub(crate) fn snapshot(ir: &IrGraph) -> Result<Graph, String> {
    let order = ir.topological_live_node_ids();
    let mut positions: HashMap<NodeId, usize> = HashMap::new();
    let mut position = 0;
    for &id in &order {
        if ir.node(id).op != NodeOp::Output {
            positions.insert(id, position);
            position += 1;
        }
    }
    let arg = |arg: &IrArg| fixture_arg(arg, &positions);
    let mut nodes = Vec::new();
    let mut output = None;
    for &id in &order {
        let node = ir.node(id);
        let op = match node.op {
            NodeOp::Output => {
                if output.is_some() {
                    return Err("the graph has two output nodes".to_string());
                }
                let result = node
                    .args
                    .first()
                    .ok_or_else(|| "the output node has no argument".to_string())?;
                output = Some(arg(result)?);
                continue;
            }
            NodeOp::Placeholder => Op::Placeholder {
                name: node.name.clone(),
                constant: node.tensor.as_ref().map(constant_kind),
            },
            NodeOp::CallFunction => Op::Call {
                target: node.target.clone(),
                args: node.args.iter().map(arg).collect::<Result<_, _>>()?,
                kwargs: node
                    .kwargs
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), arg(value)?)))
                    .collect::<Result<_, String>>()?,
            },
            other => return Err(format!("unexpected node op {other:?} ({})", node.name)),
        };
        let val = node
            .meta
            .tensor_facts()
            .map(|facts| (facts.dtype.clone(), facts.shape.clone()));
        let rank = val.as_ref().map_or(0, |(_, shape)| shape.len());
        let dim_order = node
            .meta
            .val_dim_order()
            .filter(|order| !order.iter().copied().eq(0..rank))
            .map(<[usize]>::to_vec);
        nodes.push(Node {
            op,
            val,
            dim_order,
            q_in: node.meta.input_qparams().map(qparam_keys),
            q_out: node.meta.output_qparams().map(qparam_keys),
            dead: false,
        });
    }
    let output = output.ok_or_else(|| "the graph has no output node".to_string())?;
    let mut graph = Graph { nodes, output };
    let reached = reached(&graph);
    for (position, node) in graph.nodes.iter_mut().enumerate() {
        node.dead = matches!(node.op, Op::Call { .. }) && !reached[position];
    }
    Ok(graph)
}

fn fixture_arg(arg: &IrArg, positions: &HashMap<NodeId, usize>) -> Result<Arg, String> {
    Ok(match arg {
        IrArg::Node(id) => Arg::Node(
            *positions
                .get(id)
                .ok_or_else(|| format!("a live node reads node {id}, which is not live"))?,
        ),
        IrArg::Int(value) => Arg::Int(*value),
        IrArg::Float(value) => Arg::Float(*value),
        IrArg::Bool(value) => Arg::Bool(*value),
        IrArg::Str(text) => Arg::Str(text.clone()),
        IrArg::Dtype(dtype) => Arg::Dtype(dtype.clone()),
        IrArg::List(items) => Arg::List(
            items
                .iter()
                .map(|item| fixture_arg(item, positions))
                .collect::<Result<_, _>>()?,
        ),
        IrArg::None => Arg::None,
    })
}

fn constant_kind(tensor: &NodeTensor) -> String {
    match &tensor.source {
        TensorSource::ExportedProgramInput(binding) => match binding.input_kind {
            LoweringInputKind::Parameter => "param",
            LoweringInputKind::Buffer => "buffer",
            LoweringInputKind::LiftedTensorConstant => "constant",
        },
        TensorSource::GraphModuleAttr => "attr",
    }
    .to_string()
}

fn qparam_keys(map: &QParamMap) -> Vec<usize> {
    let mut keys: Vec<usize> = map.iter().map(|(&key, _)| key).collect();
    keys.sort_unstable();
    keys
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Which nodes the output reaches.
fn reached(graph: &Graph) -> Vec<bool> {
    let mut reached = vec![false; graph.nodes.len()];
    let mut stack = Vec::new();
    collect_refs(&graph.output, &mut stack);
    while let Some(position) = stack.pop() {
        if std::mem::replace(&mut reached[position], true) {
            continue;
        }
        if let Op::Call { args, kwargs, .. } = &graph.nodes[position].op {
            for arg in args.iter().chain(kwargs.values()) {
                collect_refs(arg, &mut stack);
            }
        }
    }
    reached
}

fn collect_refs(arg: &Arg, out: &mut Vec<usize>) {
    match arg {
        Arg::Node(position) => out.push(*position),
        Arg::List(items) => items.iter().for_each(|item| collect_refs(item, out)),
        _ => {}
    }
}

/// `graph` without its dead call nodes, with references renumbered.
pub(crate) fn without_dead_nodes(graph: &Graph) -> Graph {
    let mut positions = vec![None; graph.nodes.len()];
    let mut nodes = Vec::new();
    for (position, node) in graph.nodes.iter().enumerate() {
        if !node.dead {
            positions[position] = Some(nodes.len());
            nodes.push(node.clone());
        }
    }
    let renumber = |arg: &Arg| renumbered(arg, &positions);
    for node in &mut nodes {
        if let Op::Call { args, kwargs, .. } = &mut node.op {
            *args = args.iter().map(renumber).collect();
            for value in kwargs.values_mut() {
                *value = renumber(value);
            }
        }
    }
    Graph {
        nodes,
        output: renumber(&graph.output),
    }
}

fn renumbered(arg: &Arg, positions: &[Option<usize>]) -> Arg {
    match arg {
        Arg::Node(position) => {
            Arg::Node(positions[*position].expect("a live node only reads live nodes"))
        }
        Arg::List(items) => Arg::List(
            items
                .iter()
                .map(|item| renumbered(item, positions))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `[]` and `None` are the same value after import.
fn canonical(arg: &Arg) -> Arg {
    match arg {
        Arg::List(items) if items.is_empty() => Arg::None,
        Arg::List(items) => Arg::List(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// Compare RETAB's graph with rc4's.
pub(crate) fn compare(expected: &Graph, actual: &Graph, options: Options) -> Result<(), Mismatch> {
    let (expected, actual) = if options.dead_nodes {
        (expected.clone(), actual.clone())
    } else {
        (without_dead_nodes(expected), without_dead_nodes(actual))
    };
    let mismatch = |class, detail| Err(Mismatch { class, detail });
    for (position, (want, got)) in expected.nodes.iter().zip(&actual.nodes).enumerate() {
        let at = format!("%{position}");
        match (&want.op, &got.op) {
            (Op::Placeholder { constant: a, .. }, Op::Placeholder { constant: b, .. }) => {
                if a != b {
                    return mismatch("constant-kind", format!("{at}: rc4 {a:?}, RETAB {b:?}"));
                }
            }
            (
                Op::Call {
                    target: target_a,
                    args: args_a,
                    kwargs: kwargs_a,
                },
                Op::Call {
                    target: target_b,
                    args: args_b,
                    kwargs: kwargs_b,
                },
            ) => {
                if target_a != target_b {
                    return mismatch("target", format!("{at}: rc4 {target_a}, RETAB {target_b}"));
                }
                let args_a: Vec<Arg> = args_a.iter().map(canonical).collect();
                let args_b: Vec<Arg> = args_b.iter().map(canonical).collect();
                if args_a != args_b {
                    return mismatch(
                        "args",
                        format!(
                            "{at} {target_a}: rc4 ({}), RETAB ({})",
                            list(&args_a),
                            list(&args_b)
                        ),
                    );
                }
                let kwargs_a: BTreeMap<&String, Arg> = kwargs_a
                    .iter()
                    .map(|(key, value)| (key, canonical(value)))
                    .collect();
                let kwargs_b: BTreeMap<&String, Arg> = kwargs_b
                    .iter()
                    .map(|(key, value)| (key, canonical(value)))
                    .collect();
                if kwargs_a != kwargs_b {
                    return mismatch(
                        "kwargs",
                        format!("{at} {target_a}: rc4 {kwargs_a:?}, RETAB {kwargs_b:?}"),
                    );
                }
            }
            (a, b) => {
                return mismatch(
                    "op",
                    format!("{at}: rc4 {}, RETAB {}", op_label(a), op_label(b)),
                )
            }
        }
        let label = format!("{at} {}", op_label(&want.op));
        match (&want.val, &got.val) {
            (Some((dtype_a, shape_a)), Some((dtype_b, shape_b))) => {
                if dtype_a != dtype_b {
                    return mismatch("dtype", format!("{label}: rc4 {dtype_a}, RETAB {dtype_b}"));
                }
                if shape_a != shape_b {
                    return mismatch(
                        "shape",
                        format!("{label}: rc4 {shape_a:?}, RETAB {shape_b:?}"),
                    );
                }
            }
            (a, b) => return mismatch("val", format!("{label}: rc4 {a:?}, RETAB {b:?}")),
        }
        let is_placeholder = matches!(want.op, Op::Placeholder { .. });
        let check_dim_order = match options.dim_orders {
            DimOrders::Ignore => false,
            DimOrders::Placeholders => is_placeholder,
            DimOrders::All => true,
        };
        if check_dim_order && want.dim_order != got.dim_order {
            return mismatch(
                if is_placeholder {
                    "dim-order"
                } else {
                    "dim-order-inner"
                },
                format!(
                    "{label}: rc4 {}, RETAB {}",
                    order_label(&want.dim_order),
                    order_label(&got.dim_order)
                ),
            );
        }
        if (&want.q_in, &want.q_out) != (&got.q_in, &got.q_out) {
            return mismatch(
                "qparams",
                format!(
                    "{label}: rc4 {}, RETAB {}",
                    qparams_label(want),
                    qparams_label(got)
                ),
            );
        }
        if options.dead_nodes && want.dead != got.dead {
            return mismatch(
                "dead",
                format!("{label}: rc4 dead={}, RETAB dead={}", want.dead, got.dead),
            );
        }
    }
    if expected.nodes.len() != actual.nodes.len() {
        return mismatch(
            "node-count",
            format!(
                "rc4 has {} nodes, RETAB {}{}",
                expected.nodes.len(),
                actual.nodes.len(),
                if options.dead_nodes {
                    ""
                } else {
                    " (live nodes and placeholders)"
                }
            ),
        );
    }
    if canonical(&expected.output) != canonical(&actual.output) {
        return mismatch(
            "output",
            format!(
                "rc4 output({}), RETAB output({})",
                arg_text(&expected.output),
                arg_text(&actual.output)
            ),
        );
    }
    Ok(())
}

fn op_label(op: &Op) -> String {
    match op {
        Op::Placeholder { name, .. } => format!("placeholder {name}"),
        Op::Call { target, .. } => target.clone(),
    }
}

fn order_label(order: &Option<Vec<usize>>) -> String {
    order
        .as_ref()
        .map_or_else(|| "identity".to_string(), |order| format!("{order:?}"))
}

fn qparams_label(node: &Node) -> String {
    let mut parts = Vec::new();
    for (label, keys) in [("in", &node.q_in), ("out", &node.q_out)] {
        if let Some(keys) = keys {
            let keys: Vec<String> = keys.iter().map(ToString::to_string).collect();
            parts.push(format!("{label}{{{}}}", keys.join(",")));
        }
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("/")
    }
}

// ---------------------------------------------------------------------------
// Printing, in fixture form
// ---------------------------------------------------------------------------

fn arg_text(arg: &Arg) -> String {
    match arg {
        Arg::Node(position) => format!("%{position}"),
        Arg::Int(value) => value.to_string(),
        Arg::Float(value) => format!("{value:?}"),
        Arg::Bool(true) => "True".to_string(),
        Arg::Bool(false) => "False".to_string(),
        Arg::Str(text) => format!("\"{text}\""),
        Arg::Dtype(dtype) => dtype.clone(),
        Arg::List(items) => format!("[{}]", list(items)),
        Arg::None => "None".to_string(),
    }
}

fn list(items: &[Arg]) -> String {
    items.iter().map(arg_text).collect::<Vec<_>>().join(", ")
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, node) in self.nodes.iter().enumerate() {
            write!(f, "    %{position} = ")?;
            match &node.op {
                Op::Placeholder { name, .. } => write!(f, "placeholder {name}")?,
                Op::Call {
                    target,
                    args,
                    kwargs,
                } => {
                    let mut parts: Vec<String> = args.iter().map(arg_text).collect();
                    parts.extend(
                        kwargs
                            .iter()
                            .map(|(key, value)| format!("{key}={}", arg_text(value))),
                    );
                    write!(f, "{target}({})", parts.join(", "))?;
                }
            }
            match &node.val {
                Some((dtype, shape)) => {
                    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
                    write!(f, " : {dtype}[{}]", dims.join(", "))?;
                }
                None => write!(f, " : <no val>")?,
            }
            if let Some(order) = &node.dim_order {
                let dims: Vec<String> = order.iter().map(ToString::to_string).collect();
                write!(f, " dim_order=[{}]", dims.join(","))?;
            }
            if let Op::Placeholder {
                constant: Some(kind),
                ..
            } = &node.op
            {
                write!(f, " kind={kind}")?;
            }
            write!(f, " q={}", qparams_label(node))?;
            if node.dead {
                write!(f, " dead")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "    output({})", arg_text(&self.output))
    }
}

// ---------------------------------------------------------------------------
// Harness self-tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A pass that changes nothing.
    struct Noop;

    impl IrPass for Noop {
        fn name(&self) -> &str {
            "Noop"
        }

        fn run(&self, _graph: &mut IrGraph) -> bool {
            false
        }
    }

    /// Erases the first unused call node it finds.
    struct EraseOneDeadNode;

    impl IrPass for EraseOneDeadNode {
        fn name(&self) -> &str {
            "EraseOneDeadNode"
        }

        fn run(&self, graph: &mut IrGraph) -> bool {
            let dead = graph
                .live_call_function_ids()
                .into_iter()
                .find(|&id| !graph.has_live_users(id));
            dead.map(|id| graph.erase(id)).is_some()
        }
    }

    const FIXTURE: &str = "\
# a comment
case unchanged
before
%0 = placeholder x : torch.float32[1, 3, 4, 5] dim_order=[0,2,3,1] q=in{}/out{0}
%1 = placeholder w : torch.int8[2] kind=param q=none
%2 = aten.view_copy.default(%0, [60]) : torch.float32[60] q=none
%3 = aten.clamp.default(%2, None, 6.0) : torch.float32[60] q=out{0,1}
%4 = aten.neg.default(%0) : torch.float32[1, 3, 4, 5] dim_order=[0,2,3,1] q=none dead
%5 = aten._to_copy.default(%3, dtype=torch.float16) : torch.float16[60] q=none
%6 = aten.gelu.default(%5, approximate=\"tanh\") : torch.float16[60] q=none
output([%6, [%1, None], %6])
after modified=false same
end
case renamed_and_rewired
before
%0 = placeholder x : torch.float32[2, 3] q=none
%1 = aten.relu.default(%0) : torch.float32[2, 3] q=none
output([%1])
after modified=true
%0 = placeholder x : torch.float32[2, 3] q=none
%1 = aten.neg.default(%0) : torch.float32[2, 3] q=none
output([%1])
end
case rank0_view
before
%0 = placeholder x : torch.float32[1] q=none
%1 = aten.view_copy.default(%0, []) : torch.float32[] q=none
output([%1])
after modified=false same
end
";

    #[test]
    fn parses_every_part_of_the_format() {
        let cases = parse(FIXTURE).unwrap();
        assert_eq!(cases.len(), 3);
        let case = &cases[0];
        assert_eq!(case.name, "unchanged");
        assert!(!case.modified);
        assert_eq!(case.before, case.after);
        let nodes = &case.before.nodes;
        assert_eq!(nodes[0].dim_order, Some(vec![0, 2, 3, 1]));
        assert_eq!(
            (nodes[0].q_in.clone(), nodes[0].q_out.clone()),
            (Some(vec![]), Some(vec![0]))
        );
        assert_eq!(
            nodes[1].op,
            Op::Placeholder {
                name: "w".to_string(),
                constant: Some("param".to_string())
            }
        );
        let Op::Call { args, kwargs, .. } = &nodes[3].op else {
            panic!("expected a call");
        };
        assert_eq!(args, &[Arg::Node(2), Arg::None, Arg::Float(6.0)]);
        assert!(kwargs.is_empty());
        let Op::Call { kwargs, .. } = &nodes[5].op else {
            panic!("expected a call");
        };
        assert_eq!(kwargs["dtype"], Arg::Dtype("torch.float16".to_string()));
        assert!(nodes[4].dead);
        assert_eq!(
            case.before.output,
            Arg::List(vec![
                Arg::Node(6),
                Arg::List(vec![Arg::Node(1), Arg::None]),
                Arg::Node(6)
            ])
        );
    }

    #[test]
    fn printing_round_trips() {
        for case in parse(FIXTURE).unwrap() {
            let text = format!(
                "case {}\nbefore\n{}after modified=false\n{}end\n",
                case.name, case.before, case.after
            )
            .replace("    ", "");
            let reparsed = parse(&text).unwrap();
            assert_eq!(reparsed[0].before, case.before);
            assert_eq!(reparsed[0].after, case.after);
        }
    }

    #[test]
    fn built_graphs_read_back_as_the_fixture() {
        for case in parse(FIXTURE).unwrap() {
            let graph = snapshot(&build(&case.before)).unwrap();
            let strict = Options {
                dim_orders: DimOrders::All,
                dead_nodes: true,
            };
            assert_eq!(compare(&case.before, &graph, strict), Ok(()));
        }
    }

    #[test]
    fn the_build_emulates_the_importer() {
        let cases = parse(FIXTURE).unwrap();
        let graph = build(&cases[2].before);
        // `resolve_arg` turns `[]` into None.
        let view = graph.node(1);
        assert!(matches!(view.args[1], IrArg::None));
        let graph = build(&cases[0].before);
        assert!(graph.node(1).tensor.is_some() && graph.node(0).tensor.is_none());
        assert_eq!(graph.node(2).meta.val_dim_order(), Some(&[0][..]));
        assert_eq!(graph.node(4).meta.val_dim_order(), Some(&[0, 2, 3, 1][..]));
        assert!(graph
            .node(0)
            .meta
            .input_qparams()
            .is_some_and(QParamMap::is_empty));
        assert!(graph.node(1).meta.input_qparams().is_none());
    }

    #[test]
    fn matching_cases_pass_and_differences_are_classified() {
        let results = replay(&Noop, FIXTURE, Options::default());
        let classes: Vec<Option<&str>> = results
            .iter()
            .map(|result| result.mismatch.as_ref().map(|mismatch| mismatch.class))
            .collect();
        assert_eq!(classes, [None, Some("target"), None]);
        assert!(results[1]
            .report
            .contains("rc4 aten.neg.default, RETAB aten.relu.default"));
        check_known(
            &Noop,
            FIXTURE,
            &[("renamed_and_rewired", "Noop doesn't rewrite")],
        );
        check_cases(&Noop, FIXTURE, &["unchanged", "rank0_view"]);
    }

    #[test]
    #[should_panic(expected = "1 of 3 rc4 replay cases differ (target 1)")]
    fn check_reports_unexpected_differences() {
        check(&Noop, FIXTURE);
    }

    #[test]
    #[should_panic(expected = "case unchanged now matches rc4")]
    fn check_known_rejects_stale_entries() {
        check_known(
            &Noop,
            FIXTURE,
            &[
                ("renamed_and_rewired", "Noop doesn't rewrite"),
                ("unchanged", "stale"),
            ],
        );
    }

    #[test]
    fn dead_nodes_only_count_when_asked() {
        let cases = parse(FIXTURE).unwrap();
        let mut graph = build(&cases[0].before);
        assert!(EraseOneDeadNode.run(&mut graph));
        let actual = snapshot(&graph).unwrap();
        assert_eq!(
            compare(&cases[0].after, &actual, Options::default()),
            Ok(())
        );
        let strict = Options {
            dead_nodes: true,
            ..Options::default()
        };
        // rc4's dead neg sits where RETAB's graph has the next node.
        let mismatch = compare(&cases[0].after, &actual, strict).unwrap_err();
        assert_eq!(mismatch.class, "target");
        assert!(
            mismatch.detail.contains("rc4 aten.neg.default"),
            "{}",
            mismatch.detail
        );
    }

    #[test]
    fn dim_orders_only_count_when_asked() {
        let cases = parse(FIXTURE).unwrap();
        let options = |dim_orders| Options {
            dim_orders,
            ..Options::default()
        };
        let class = |graph: &IrGraph, dim_orders| {
            let actual = snapshot(graph).unwrap();
            compare(&cases[0].after, &actual, options(dim_orders))
                .err()
                .map(|mismatch| mismatch.class)
        };
        let mut graph = build(&cases[0].before);
        // A stale order of the wrong rank, as a cloned meta leaves.
        graph.node_mut(2).meta.set_val_dim_order(vec![0, 1]);
        assert_eq!(class(&graph, DimOrders::Ignore), None);
        assert_eq!(class(&graph, DimOrders::Placeholders), None);
        assert_eq!(class(&graph, DimOrders::All), Some("dim-order-inner"));
        graph.node_mut(0).meta.set_val_dim_order(vec![0, 1, 2, 3]);
        assert_eq!(class(&graph, DimOrders::Ignore), None);
        assert_eq!(class(&graph, DimOrders::Placeholders), Some("dim-order"));
    }

    #[test]
    fn bad_fixtures_are_rejected() {
        let bad = [
            "case x\nbefore\n%1 = placeholder x : torch.float32[2] q=none\noutput([%1])\nafter modified=false same\nend\n",
            "case x\nbefore\n%0 = aten.relu.default(%1) : torch.float32[2] q=none\noutput([%0])\nafter modified=false same\nend\n",
            "case x\nbefore\n%0 = placeholder x : torch.float32[2] q=none\n%1 = aten.relu.default(%0) : torch.float32[2] q=none\noutput([%0])\nafter modified=false same\nend\n",
            "case x\nbefore\n%0 = placeholder x : torch.float32[2] q=none\noutput([%0])\nafter modified=maybe same\nend\n",
            "case x\nbefore\n%0 = placeholder x : torch.float32[2] q=none weird\noutput([%0])\nafter modified=false same\nend\n",
        ];
        for text in bad {
            assert!(parse(text).is_err(), "{text}");
        }
    }
}
