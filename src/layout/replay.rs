//! Replays ExecuTorch v1.6.0-rc4's layout passes against the RETAB ports
//! (APU-135 P1 and P2).
//!
//! `tests/fixtures/layout_replay/gen_layout_replay.py` records rc4 graphs
//! before and after a pass, one fixture per pass (the format is in the
//! generator's docstring). The P1 fixtures run the real rc4 pass on hand-built
//! edge graphs; the P2 fixtures (`format 2`) capture the graphs rc4's
//! ArmPassManager pipeline hands to the pass and gets back while it lowers
//! small quantized models. [`check`] rebuilds every "before" graph as an
//! `IrGraph`, runs the port and compares the result with rc4's "after" graph:
//!
//! - nodes in graph order (`IrGraph::topological_live_node_ids`, which
//!   `LayoutCtx::finish`'s renumbering makes the ctx order), with each node's
//!   op, target, args and kwargs; node references compare by position, so
//!   names don't matter (rc4 renames nodes when it retraces). `tosa.X` and
//!   `backend.tosa.X` are the same target, `[]` and None are the same value,
//!   and a kwarg that is None (or a device or layout, which the importer reads
//!   as None) is the same as no kwarg;
//! - each node's dtype and shape, and whether its `input_qparams` and
//!   `output_qparams` maps are absent, empty or hold which keys;
//! - placeholders' constant kinds;
//! - the output node's argument;
//! - in format 2 also every qparam's value, each constant's data, special
//!   dtypes (`tosa_special_dtype`) and the `do_not_fuse_duplicate` flag.
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
//! doesn't reach, so dead code alone is never a mismatch; placeholders and
//! `get_attr` nodes always stay. `Options::dead_nodes` compares the whole
//! graphs instead.
//!
//! Import emulation: the "before" graph looks like what RETAB's importer
//! builds. An empty list or tuple becomes `IrArg::None` (`resolve_arg` in
//! src/fx_ir.rs), so a fixture `[]` compares equal to `None`, and so do the
//! devices, layouts and memory formats the importer can't read. Every node
//! gets its val's dim order, the identity unless the fixture says otherwise,
//! as `NodeMeta::from_fx` records `val.dim_order()`; a val that isn't a
//! tensor (a tuple) imports as no val at all. Constant placeholders and
//! `get_attr` nodes get a `NodeTensor`, which is how RETAB tells parameters,
//! buffers and lifted constants from runtime inputs: the fixture's data in
//! format 2 (int64 data narrowed to int32 the way `extract_tensor_data` does
//! it, and widened back for comparison), zeros otherwise. QParam maps keep the
//! fixture's keys; empty maps stay empty (the importer drops them, but RETAB
//! passes create them), so the ports' handling of both is visible. In format 1
//! the values are placeholders; in format 2 they are rc4's, built the way
//! `QParam::from_py` reads a `QuantArgs` (a per-channel qparam has scale 1.0
//! and zero point 0 besides its lists). Special dtypes import in RETAB's own
//! spelling (`INT48`, as RETAB's passes write it).
//!
//! Dtypes (format 2, design F2 and critique A11): rc4 types a quantized
//! activation int8 or int16 while RETAB's IR keeps it float32 and carries the
//! dtype in its qparams. [`retab_dtypes`] gives a captured graph RETAB's
//! dtypes before it is imported: a call node whose int8/int16 val carries a
//! first input or output qparam of that dtype becomes float32, and so does a
//! data-movement op (permute, view, slice, ...) reading one; placeholders keep
//! their storage dtype. RETAB passes don't retrace, so after a rewrite such a
//! node can sit downstream of an int8 RESCALE and still be float32; the
//! comparison therefore accepts float32 wherever rc4 has int8/int16 on a node
//! that carries a qparam of that dtype or moves data. Every other dtype
//! (accumulators, RESCALE outputs, placeholders, FP graphs) must match exactly.
//!
//! Usage: a pass's replay test is one line,
//! `replay::check(&IrFooPass, include_str!("../../tests/fixtures/layout_replay/foo.txt"))`.
//! [`check_known`] names cases still expected to differ, with a reason, and
//! fails once one matches; [`check_cases`] and [`check_cases_with`] replay
//! chosen cases, as the ignored BUG tests in `p1_replay_tests` do. There,
//! `replay_report` prints per-pass statistics under every comparison mode and
//! `replay_external` replays fixtures too large to commit (a fuzzing run).
//! `p2_replay_tests` holds the P2 fixtures' sanity tests and their replay hooks.

use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Write as _};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::fx_ir::{
    shape_product, IrArg, IrGraph, IrNode, IrPass, MetaValue, NodeId, NodeMeta, NodeOp,
    NodeTensor, PlaceholderTensorBinding, QParam, QParamMap, TensorSource,
};
use crate::graph::ir::DType;
use crate::lowering_effects::LoweringInputKind;
use crate::store::TensorData;

/// The newest fixture format; see the generator's docstring.
pub(crate) const FORMAT: u32 = 2;
/// Constant data up to this many bytes is recorded in full, larger data as a
/// SHA-256 digest.
pub(crate) const DATA_HEX_LIMIT: usize = 4096;

const FLOAT32: &str = "torch.float32";
const SPECIAL_DTYPE: &str = "tosa_special_dtype";
const DO_NOT_FUSE: &str = "do_not_fuse_duplicate";

/// An argument, with node references as positions in the graph.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Arg {
    Node(usize),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Dtype(String),
    /// A device, layout or memory format, as printed (`device:cpu`); RETAB's
    /// importer reads it as None.
    Opaque(String),
    List(Vec<Arg>),
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Op {
    /// `constant` is `param`, `buffer` or `constant` for a constant placeholder
    /// (`attr` for one backed by a graph-module attribute).
    Placeholder {
        name: String,
        constant: Option<String>,
    },
    GetAttr {
        name: String,
    },
    Call {
        target: String,
        args: Vec<Arg>,
        kwargs: BTreeMap<String, Arg>,
    },
}

/// A node's `meta["val"]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Val {
    Tensor {
        dtype: String,
        shape: Vec<i64>,
    },
    /// A val that isn't a tensor, as printed: a tuple of tensors
    /// (`max_pool2d_with_indices`) or of ints (CONST_SHAPE). RETAB's importer
    /// keeps no val for it.
    Other(String),
    /// No val; only a RETAB node can lack one.
    Missing,
}

/// One qparam's value: rc4's `QuantArgs`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QValue {
    pub(crate) dtype: String,
    pub(crate) per_channel: bool,
    /// One scale and zero point per tensor, or one per channel along `axis`.
    pub(crate) scales: Vec<f64>,
    pub(crate) zps: Vec<i64>,
    pub(crate) qmin: i64,
    pub(crate) qmax: i64,
    pub(crate) axis: i64,
}

/// A qparam map: its keys in order, each with its value when the fixture
/// records one (format 2).
pub(crate) type QMap = Vec<(usize, Option<QValue>)>;

/// A constant's data: its bytes, or the SHA-256 digest (lowercase hex) of
/// bytes over `DATA_HEX_LIMIT`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Data {
    Bytes(Vec<u8>),
    Sha256(String),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    pub(crate) op: Op,
    pub(crate) val: Val,
    /// `None` is the identity.
    pub(crate) dim_order: Option<Vec<usize>>,
    pub(crate) q_in: Option<QMap>,
    pub(crate) q_out: Option<QMap>,
    /// The constant's data (format 2), in its own dtype.
    pub(crate) data: Option<Data>,
    /// `tosa_special_dtype`, without the `TosaSpecialDtype.` prefix.
    pub(crate) special: Option<String>,
    /// `do_not_fuse_duplicate` is set.
    pub(crate) no_fuse: bool,
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
    /// The fixture's format: 1 for the P1 fixtures, 2 for the P2 ones.
    pub(crate) format: u32,
    pub(crate) before: Graph,
    /// The `modified` flag rc4's pass returned (format 1) or whether the pass
    /// changed the graph (format 2, whose capture hooks can't see the flag).
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
    /// Accept RETAB's float32 for rc4's quantized activations (design F2);
    /// set for every format-2 case.
    pub(crate) float32_activations: bool,
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

impl Node {
    fn tensor_dtype(&self) -> Option<&str> {
        match &self.val {
            Val::Tensor { dtype, .. } => Some(dtype),
            _ => None,
        }
    }
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

/// The graph a replay hands the port: `case.before` as RETAB's importer
/// would build it, with RETAB's dtypes in format 2.
pub(crate) fn import(case: &Case) -> IrGraph {
    if case.format >= 2 {
        build(&retab_dtypes(&case.before))
    } else {
        build(&case.before)
    }
}

/// Compare RETAB's graph after a replay of `case` with rc4's: [`compare`],
/// with format 2's float32 activations, or without the details format 1
/// doesn't record.
pub(crate) fn compare_case(case: &Case, actual: &Graph, options: Options) -> Result<(), Mismatch> {
    if case.format >= 2 {
        let options = Options {
            float32_activations: true,
            ..options
        };
        compare(&case.after, actual, options)
    } else {
        compare(&case.after, &without_details(actual), options)
    }
}

pub(crate) fn replay_case(pass: &dyn IrPass, case: &Case, options: Options) -> CaseResult {
    let mut graph = import(case);
    let run = catch_unwind(AssertUnwindSafe(|| pass.run(&mut graph)));
    let (retab_modified, actual) = match run {
        Ok(modified) => (Some(modified), snapshot(&graph)),
        Err(payload) => (None, Err(panic_message(payload.as_ref()))),
    };
    let actual = actual.map(|actual| {
        if case.format >= 2 {
            actual
        } else {
            without_details(&actual)
        }
    });
    let mismatch = match &actual {
        Ok(actual) => compare_case(case, actual, options).err(),
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
            let (before, note) = if case.format >= 2 {
                (
                    retab_dtypes(&case.before),
                    " (as imported, with RETAB's float32 activations)",
                )
            } else {
                (case.before.clone(), "")
            };
            format!(
                "case {} ({}): {}\n  before{note}:\n{}  rc4 after (modified={}):\n{}  RETAB after (modified={}):\n{}",
                case.name,
                mismatch.class,
                mismatch.detail,
                before,
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
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'))
        .peekable();
    let mut format = 1;
    if let Some(&(number, line)) = lines.peek() {
        if let Some(version) = line.strip_prefix("format ") {
            format = version
                .parse()
                .ok()
                .filter(|version| (1..=FORMAT).contains(version))
                .ok_or_else(|| format!("line {number}: unknown format {version:?}"))?;
            lines.next();
        }
    }
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
            if format >= 2 {
                check_details(graph)
                    .map_err(|error| format!("case {name} ({label}): {error}"))?;
            }
        }
        cases.push(Case {
            name: name.to_string(),
            format,
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
    } else if cursor.eat("get_attr ") {
        let name = cursor.until(" : ")?.to_string();
        Op::GetAttr { name }
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
    let val = cursor.val()?;
    let mut node = Node {
        op,
        val,
        dim_order: None,
        q_in: None,
        q_out: None,
        data: None,
        special: None,
        no_fuse: false,
        dead: false,
    };
    for note in cursor.rest().split(' ').filter(|note| !note.is_empty()) {
        if note == "dead" {
            node.dead = true;
        } else if note == "no_fuse" {
            node.no_fuse = true;
        } else if let Some(order) = note.strip_prefix("dim_order=") {
            node.dim_order = Some(parse_usizes(order, '[', ']')?);
        } else if let Some(kind) = note.strip_prefix("kind=") {
            match &mut node.op {
                Op::Placeholder { constant, .. } => *constant = Some(kind.to_string()),
                _ => return Err("kind= on a node that isn't a placeholder".to_string()),
            }
        } else if let Some(spec) = note.strip_prefix("q=") {
            (node.q_in, node.q_out) = parse_qparams(spec)?;
        } else if let Some(data) = note.strip_prefix("data=") {
            node.data = Some(parse_data(data)?);
        } else if let Some(special) = note.strip_prefix("special=") {
            node.special = Some(special.to_string());
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

/// `none`, or `in{...}` and/or `out{...}` joined by `/`.
fn parse_qparams(spec: &str) -> Result<(Option<QMap>, Option<QMap>), String> {
    let (mut q_in, mut q_out) = (None, None);
    if spec == "none" {
        return Ok((q_in, q_out));
    }
    let mut rest = spec;
    loop {
        let (slot, body) = if let Some(body) = rest.strip_prefix("in{") {
            (&mut q_in, body)
        } else if let Some(body) = rest.strip_prefix("out{") {
            (&mut q_out, body)
        } else {
            return Err(format!("bad qparam spec {spec:?}"));
        };
        let end = body
            .find('}')
            .ok_or_else(|| format!("unclosed qparam map in {spec:?}"))?;
        if slot.is_some() {
            return Err(format!("repeated qparam map in {spec:?}"));
        }
        *slot = Some(parse_qmap(&body[..end])?);
        rest = &body[end + 1..];
        if rest.is_empty() {
            return Ok((q_in, q_out));
        }
        rest = rest
            .strip_prefix('/')
            .ok_or_else(|| format!("bad qparam spec {spec:?}"))?;
    }
}

/// `key` or `key:value` entries separated by commas.
fn parse_qmap(body: &str) -> Result<QMap, String> {
    let mut map: QMap = Vec::new();
    for entry in split_top_level(body) {
        let (key, value) = match entry.split_once(':') {
            Some((key, value)) => (key, Some(parse_qvalue(value)?)),
            None => (entry, None),
        };
        let key: usize = key.parse().map_err(|_| format!("bad index {key:?}"))?;
        if map.last().is_some_and(|(last, _)| *last >= key) {
            return Err(format!("qparam keys out of order in {body:?}"));
        }
        map.push((key, value));
    }
    Ok(map)
}

/// `dtype(scale,zp,qmin,qmax)` or `dtype([scales],[zps],qmin,qmax,axis=A)`.
fn parse_qvalue(text: &str) -> Result<QValue, String> {
    let bad = || format!("bad qparam value {text:?}");
    let open = text.find('(').ok_or_else(bad)?;
    let dtype = text[..open].to_string();
    let inner = text[open + 1..].strip_suffix(')').ok_or_else(bad)?;
    let mut parts = split_top_level(inner);
    let axis = match parts.last().and_then(|part| part.strip_prefix("axis=")) {
        Some(axis) => {
            let axis = axis.parse().map_err(|_| bad())?;
            parts.pop();
            axis
        }
        None => 0,
    };
    let [scale, zp, qmin, qmax] = parts[..] else {
        return Err(bad());
    };
    let floats = |text: &str| -> Result<Vec<f64>, String> {
        text.split(',')
            .map(|item| item.parse().map_err(|_| bad()))
            .collect()
    };
    let ints = |text: &str| -> Result<Vec<i64>, String> {
        text.split(',')
            .filter(|item| !item.is_empty())
            .map(|item| item.parse().map_err(|_| bad()))
            .collect()
    };
    let per_channel = scale.starts_with('[');
    let (scales, zps) = if per_channel {
        let scales = scale.strip_prefix('[').and_then(|s| s.strip_suffix(']'));
        let zps = zp.strip_prefix('[').and_then(|z| z.strip_suffix(']'));
        match (scales, zps) {
            (Some(scales), Some(zps)) => (floats(scales)?, ints(zps)?),
            _ => return Err(bad()),
        }
    } else {
        (floats(scale)?, ints(zp)?)
    };
    if !per_channel && (scales.len() != 1 || zps.len() != 1) {
        return Err(bad());
    }
    Ok(QValue {
        dtype,
        per_channel,
        scales,
        zps,
        qmin: qmin.parse().map_err(|_| bad())?,
        qmax: qmax.parse().map_err(|_| bad())?,
        axis,
    })
}

/// Split at the commas outside brackets and parentheses.
fn split_top_level(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (index, c) in text.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&text[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if !text.is_empty() {
        parts.push(&text[start..]);
    }
    parts
}

/// `hex:<bytes>` or `sha256:<digest>`.
fn parse_data(text: &str) -> Result<Data, String> {
    if let Some(hex) = text.strip_prefix("hex:") {
        return hex_decode(hex).map(Data::Bytes);
    }
    if let Some(digest) = text.strip_prefix("sha256:") {
        if digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Ok(Data::Sha256(digest.to_ascii_lowercase()));
        }
    }
    Err(format!("bad data {text:?}"))
}

fn hex_decode(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err(format!("odd-length hex data ({} digits)", hex.len()));
    }
    (0..hex.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&hex[index..index + 2], 16)
                .map_err(|_| format!("bad hex byte {:?}", &hex[index..index + 2]))
        })
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
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

/// Format 2 records every qparam's value and every constant's data.
fn check_details(graph: &Graph) -> Result<(), String> {
    for (position, node) in graph.nodes.iter().enumerate() {
        for map in [&node.q_in, &node.q_out].into_iter().flatten() {
            if map.iter().any(|(_, value)| value.is_none()) {
                return Err(format!("%{position} has a qparam without a value"));
            }
        }
        let constant = match &node.op {
            Op::Placeholder { constant, .. } => constant.is_some(),
            Op::GetAttr { .. } => true,
            Op::Call { .. } => false,
        };
        if constant && node.data.is_none() {
            return Err(format!("%{position} is a constant without data"));
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

    /// `dtype[shape]`, a tuple `(...)` of vals and ints, or `<no val>`.
    fn val(&mut self) -> Result<Val, String> {
        if self.eat("<no val>") {
            return Ok(Val::Missing);
        }
        if self.rest().starts_with('(') {
            let start = self.position;
            self.tuple()?;
            return Ok(Val::Other(self.text[start..self.position].to_string()));
        }
        let dtype = self.until("[")?.to_string();
        self.expect("[")?;
        let mut shape = Vec::new();
        while !self.eat("]") {
            if !shape.is_empty() {
                self.expect(", ")?;
            }
            shape.push(self.int()?);
        }
        Ok(Val::Tensor { dtype, shape })
    }

    fn tuple(&mut self) -> Result<(), String> {
        self.expect("(")?;
        let mut first = true;
        while !self.eat(")") {
            if !first {
                self.expect(", ")?;
            }
            first = false;
            if self
                .rest()
                .starts_with(|c: char| c == '-' || c.is_ascii_digit())
            {
                self.int()?;
            } else if matches!(self.val()?, Val::Missing) {
                return Err("<no val> inside a tuple".to_string());
            }
        }
        Ok(())
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
            _ if ["device:", "layout:", "memory_format:"]
                .iter()
                .any(|prefix| token.starts_with(prefix)) =>
            {
                Arg::Opaque(token.to_string())
            }
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
// RETAB's dtypes
// ---------------------------------------------------------------------------

/// Targets that only move data; RETAB gives them their input's dtype.
const DATA_MOVEMENT: &[&str] = &[
    "aten.permute_copy.default",
    "aten.permute.default",
    "aten.view_copy.default",
    "aten.view.default",
    "aten.slice_copy.Tensor",
    "aten.select_copy.int",
    "aten.clone.default",
    "dim_order_ops._clone_dim_order.default",
    "aten.alias_copy.default",
    "aten.unsqueeze_copy.default",
    "aten.squeeze_copy.default",
    "aten.squeeze_copy.dim",
    "aten.squeeze_copy.dims",
    "aten.expand_copy.default",
    "aten.repeat.default",
    "tosa.TRANSPOSE.default",
    "backend.tosa.TRANSPOSE.default",
    "<built-in function getitem>",
];

fn is_data_movement(target: &str) -> bool {
    DATA_MOVEMENT.contains(&target)
}

/// int8 and int16 are rc4's quantized activation dtypes.
fn is_quantized_storage(dtype: &str) -> bool {
    matches!(dtype, "torch.int8" | "torch.int16")
}

/// `node`'s first input or output qparam has dtype `dtype`.
fn carries_qparam_of(node: &Node, dtype: &str) -> bool {
    [&node.q_in, &node.q_out].into_iter().flatten().any(|map| {
        map.iter()
            .find(|(key, _)| *key == 0)
            .and_then(|(_, value)| value.as_ref())
            .is_some_and(|value| value.dtype == dtype)
    })
}

/// rc4's `graph` with the dtypes RETAB's IR gives the same values (design F2,
/// critique A11): a call node whose int8/int16 val carries a first input or
/// output qparam of that dtype is a quantized activation, which RETAB keeps
/// float32, and so is a data-movement op reading one. Placeholders and
/// `get_attr` nodes keep their storage dtype.
pub(crate) fn retab_dtypes(graph: &Graph) -> Graph {
    let mut retab = graph.clone();
    let mut float32 = vec![false; graph.nodes.len()];
    for (position, node) in graph.nodes.iter().enumerate() {
        let Op::Call { target, args, .. } = &node.op else {
            continue;
        };
        let Val::Tensor { dtype, shape } = &node.val else {
            continue;
        };
        if !is_quantized_storage(dtype) {
            continue;
        }
        let follows = is_data_movement(target)
            && match args.first() {
                Some(Arg::Node(input)) => {
                    float32[*input] && graph.nodes[*input].tensor_dtype() == Some(dtype)
                }
                _ => false,
            };
        if follows || carries_qparam_of(node, dtype) {
            retab.nodes[position].val = Val::Tensor {
                dtype: FLOAT32.to_string(),
                shape: shape.clone(),
            };
            float32[position] = true;
        }
    }
    retab
}

/// RETAB's `got` dtype stands for rc4's `want` dtype on `node`.
fn dtype_matches(node: &Node, want: &str, got: &str, float32_activations: bool) -> bool {
    want == got
        || (float32_activations
            && got == FLOAT32
            && is_quantized_storage(want)
            && match &node.op {
                Op::Call { target, .. } => {
                    is_data_movement(target) || carries_qparam_of(node, want)
                }
                _ => false,
            })
}

// ---------------------------------------------------------------------------
// Building the "before" IrGraph
// ---------------------------------------------------------------------------

/// The `IrGraph` RETAB's importer would build for `graph`, with `graph`'s
/// dtypes as they are; [`import`] adds RETAB's dtypes for format 2.
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
                    ir.node_mut(id).tensor = Some(constant_tensor(kind, name, node));
                }
                id
            }
            Op::GetAttr { name } => {
                let id = ir.add_node(NodeOp::GetAttr, name.clone(), name.clone(), Vec::new(), meta);
                ir.node_mut(id).tensor = Some(constant_tensor("attr", name, node));
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
        // `resolve_arg` reads neither a device nor a layout.
        Arg::Opaque(_) => IrArg::None,
        // `resolve_arg` turns an empty list or tuple into None.
        Arg::List(items) if items.is_empty() => IrArg::None,
        Arg::List(items) => IrArg::List(items.iter().map(|item| ir_arg(item, ids)).collect()),
        Arg::None => IrArg::None,
    }
}

fn node_meta(node: &Node) -> NodeMeta {
    let mut meta = match &node.val {
        Val::Tensor { dtype, shape } => {
            let mut meta = NodeMeta::tensor(shape.clone(), dtype.clone());
            meta.set_val_dim_order(
                node.dim_order
                    .clone()
                    .unwrap_or_else(|| (0..shape.len()).collect()),
            );
            meta
        }
        // `TensorFacts::from_py` and `dim_order()` both fail on a tuple.
        Val::Other(_) | Val::Missing => NodeMeta::default(),
    };
    if let Some(map) = &node.q_in {
        meta.set_input_qparams(qparam_map(map));
    }
    if let Some(map) = &node.q_out {
        meta.set_output_qparams(qparam_map(map));
    }
    if let Some(special) = &node.special {
        meta.set_extra(SPECIAL_DTYPE, MetaValue::Str(special.clone()));
    }
    if node.no_fuse {
        meta.set_extra(DO_NOT_FUSE, MetaValue::Bool(true));
    }
    meta
}

fn qparam_map(map: &QMap) -> QParamMap {
    QParamMap::from_pairs(map.iter().map(|(key, value)| {
        let qparam = match value {
            Some(value) => qparam(value),
            None => QParam::per_tensor(0.5, 0, -128, 127, "torch.int8".to_string()),
        };
        (*key, qparam)
    }))
}

/// The `QParam` `QParam::from_py` reads from rc4's `QuantArgs`: a per-channel
/// one keeps its lists, and its scalar scale and zero point fall back to 1.0
/// and 0 because the lists don't extract as numbers.
fn qparam(value: &QValue) -> QParam {
    let mut qparam = if value.per_channel {
        QParam {
            scale: 1.0,
            zp: 0,
            qmin: value.qmin,
            qmax: value.qmax,
            dtype_str: value.dtype.clone(),
            per_channel_scales: value.scales.clone(),
            per_channel_zps: value.zps.clone(),
            axis: 0,
        }
    } else {
        QParam::per_tensor(
            value.scales[0],
            value.zps[0],
            value.qmin,
            value.qmax,
            value.dtype.clone(),
        )
    };
    qparam.axis = value.axis;
    qparam
}

/// The `NodeTensor` the importer gives a constant: the fixture's data, zeros
/// without it, with int64 narrowed to int32 as `extract_tensor_data` does.
fn constant_tensor(kind: &str, name: &str, node: &Node) -> NodeTensor {
    let source = match kind {
        "param" | "buffer" | "constant" => {
            TensorSource::ExportedProgramInput(PlaceholderTensorBinding {
                input_kind: match kind {
                    "param" => LoweringInputKind::Parameter,
                    "buffer" => LoweringInputKind::Buffer,
                    _ => LoweringInputKind::LiftedTensorConstant,
                },
                target_name: format!("c_{name}"),
            })
        }
        "attr" => TensorSource::GraphModuleAttr,
        _ => panic!("unknown constant kind {kind:?}"),
    };
    let Val::Tensor { dtype, shape } = &node.val else {
        panic!("constant {name} has no tensor val");
    };
    let dtype = DType::from_torch_str(dtype);
    let size = usize::try_from(shape_product(shape)).unwrap_or(0) * dtype.storage_size_bytes();
    let bytes = match &node.data {
        Some(Data::Bytes(bytes)) => {
            assert_eq!(
                bytes.len(),
                size,
                "constant {name}: {} bytes of data for {dtype:?}{shape:?}",
                bytes.len()
            );
            bytes.clone()
        }
        // A digest can't be imported; the generator keeps them out of before graphs.
        Some(Data::Sha256(_)) | None => vec![0; size],
    };
    let (data, dtype) = if dtype == DType::Int64 {
        let narrowed = crate::tensor_data::cast_i64_bytes_to_i32(&bytes)
            .unwrap_or_else(|error| panic!("constant {name}: {error}"));
        (narrowed, DType::Int32)
    } else {
        (bytes, dtype)
    };
    NodeTensor {
        source,
        data: TensorData {
            data,
            shape: shape.clone(),
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
            NodeOp::GetAttr => Op::GetAttr {
                name: node.name.clone(),
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
            .map_or(Val::Missing, |facts| Val::Tensor {
                dtype: facts.dtype.clone(),
                shape: facts.shape.clone(),
            });
        let rank = match &val {
            Val::Tensor { shape, .. } => shape.len(),
            _ => 0,
        };
        let dim_order = node
            .meta
            .val_dim_order()
            .filter(|order| !order.iter().copied().eq(0..rank))
            .map(<[usize]>::to_vec);
        nodes.push(Node {
            op,
            val,
            dim_order,
            q_in: node.meta.input_qparams().map(fixture_qmap),
            q_out: node.meta.output_qparams().map(fixture_qmap),
            data: snapshot_data(node),
            special: node.meta.extra_str(SPECIAL_DTYPE).map(|special| {
                special
                    .strip_prefix("TosaSpecialDtype.")
                    .unwrap_or(special)
                    .to_string()
            }),
            no_fuse: node.meta.extra_bool(DO_NOT_FUSE) == Some(true),
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

fn fixture_qmap(map: &QParamMap) -> QMap {
    map.iter()
        .map(|(&key, qparam)| {
            let value = QValue {
                dtype: qparam.dtype_str.clone(),
                per_channel: qparam.is_per_channel(),
                scales: if qparam.is_per_channel() {
                    qparam.per_channel_scales.clone()
                } else {
                    vec![qparam.scale]
                },
                zps: if qparam.is_per_channel() {
                    qparam.per_channel_zps.clone()
                } else {
                    vec![qparam.zp]
                },
                qmin: qparam.qmin,
                qmax: qparam.qmax,
                axis: qparam.axis,
            };
            (key, Some(value))
        })
        .collect()
}

/// A constant's bytes in its val's dtype: int64 data the importer narrowed to
/// int32 is widened back.
fn snapshot_data(node: &IrNode) -> Option<Data> {
    let tensor = node.tensor.as_ref()?;
    let bytes = &tensor.data.data;
    if tensor.data.dtype == DType::Int32 && node.meta.val_dtype() == Some("torch.int64") {
        let widened = bytes
            .chunks_exact(4)
            .flat_map(|chunk| {
                i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).to_le_bytes()
            })
            .collect();
        return Some(Data::Bytes(widened));
    }
    Some(Data::Bytes(bytes.clone()))
}

/// `graph` without what format 1 doesn't record: qparam values, data,
/// special dtypes and flags.
pub(crate) fn without_details(graph: &Graph) -> Graph {
    let mut graph = graph.clone();
    for node in &mut graph.nodes {
        for map in [&mut node.q_in, &mut node.q_out].into_iter().flatten() {
            for (_, value) in map.iter_mut() {
                *value = None;
            }
        }
        node.data = None;
        node.special = None;
        node.no_fuse = false;
    }
    graph
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

/// `[]`, devices and layouts are None after import.
fn canonical(arg: &Arg) -> Arg {
    match arg {
        Arg::List(items) if items.is_empty() => Arg::None,
        Arg::List(items) => Arg::List(items.iter().map(canonical).collect()),
        Arg::Opaque(_) => Arg::None,
        other => other.clone(),
    }
}

/// `backend.tosa.X` is RETAB's other spelling of `tosa.X`.
fn canonical_target(target: &str) -> &str {
    target.strip_prefix("backend.").unwrap_or(target)
}

/// Kwargs after import, without the ones that are None.
fn canonical_kwargs(kwargs: &BTreeMap<String, Arg>) -> BTreeMap<&String, Arg> {
    kwargs
        .iter()
        .map(|(key, value)| (key, canonical(value)))
        .filter(|(_, value)| *value != Arg::None)
        .collect()
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
            (Op::GetAttr { .. }, Op::GetAttr { .. }) => {}
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
                if canonical_target(target_a) != canonical_target(target_b) {
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
                let kwargs_a = canonical_kwargs(kwargs_a);
                let kwargs_b = canonical_kwargs(kwargs_b);
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
            (
                Val::Tensor {
                    dtype: dtype_a,
                    shape: shape_a,
                },
                Val::Tensor {
                    dtype: dtype_b,
                    shape: shape_b,
                },
            ) => {
                if !dtype_matches(want, dtype_a, dtype_b, options.float32_activations) {
                    return mismatch("dtype", format!("{label}: rc4 {dtype_a}, RETAB {dtype_b}"));
                }
                if shape_a != shape_b {
                    return mismatch(
                        "shape",
                        format!("{label}: rc4 {shape_a:?}, RETAB {shape_b:?}"),
                    );
                }
            }
            // The importer keeps no val for one that isn't a tensor.
            (Val::Other(_), Val::Missing) => {}
            (a, b) => {
                return mismatch(
                    "val",
                    format!("{label}: rc4 {}, RETAB {}", val_text(a), val_text(b)),
                )
            }
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
        if !qmaps_match(&want.q_in, &got.q_in) || !qmaps_match(&want.q_out, &got.q_out) {
            return mismatch(
                "qparams",
                format!(
                    "{label}: rc4 {}, RETAB {}",
                    qparams_label(want),
                    qparams_label(got)
                ),
            );
        }
        if want.special != got.special {
            return mismatch(
                "special",
                format!(
                    "{label}: rc4 {:?}, RETAB {:?}",
                    want.special.as_deref(),
                    got.special.as_deref()
                ),
            );
        }
        if want.no_fuse != got.no_fuse {
            return mismatch(
                "no-fuse",
                format!("{label}: rc4 no_fuse={}, RETAB no_fuse={}", want.no_fuse, got.no_fuse),
            );
        }
        if let Some(want_data) = &want.data {
            if let Err(detail) = data_matches(want_data, got.data.as_ref()) {
                return mismatch("data", format!("{label}: {detail}"));
            }
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

/// Same keys, and the same value wherever rc4's map records one.
fn qmaps_match(want: &Option<QMap>, got: &Option<QMap>) -> bool {
    match (want, got) {
        (None, None) => true,
        (Some(want), Some(got)) => {
            want.len() == got.len()
                && want.iter().zip(got).all(|((key_a, value_a), (key_b, value_b))| {
                    key_a == key_b
                        && match (value_a, value_b) {
                            (Some(value_a), Some(value_b)) => value_a == value_b,
                            (Some(_), None) => false,
                            (None, _) => true,
                        }
                })
        }
        _ => false,
    }
}

fn data_matches(want: &Data, got: Option<&Data>) -> Result<(), String> {
    let digest = |data: &Data| match data {
        Data::Bytes(bytes) => sha256_hex(bytes),
        Data::Sha256(digest) => digest.clone(),
    };
    match (want, got) {
        (_, None) => Err("rc4 has data, RETAB's node has none".to_string()),
        (Data::Bytes(a), Some(Data::Bytes(b))) => {
            if a == b {
                return Ok(());
            }
            let first = a.iter().zip(b).position(|(x, y)| x != y);
            Err(match first {
                Some(index) => format!(
                    "data differs at byte {index} of {}: rc4 {:02x}, RETAB {:02x}",
                    a.len(),
                    a[index],
                    b[index]
                ),
                None => format!("rc4 has {} bytes of data, RETAB {}", a.len(), b.len()),
            })
        }
        (want, Some(got)) => {
            if digest(want) == digest(got) {
                Ok(())
            } else {
                Err(format!(
                    "data digests differ: rc4 {}, RETAB {}",
                    digest(want),
                    digest(got)
                ))
            }
        }
    }
}

fn op_label(op: &Op) -> String {
    match op {
        Op::Placeholder { name, .. } => format!("placeholder {name}"),
        Op::GetAttr { name } => format!("get_attr {name}"),
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
    for (label, map) in [("in", &node.q_in), ("out", &node.q_out)] {
        if let Some(map) = map {
            let entries: Vec<String> = map
                .iter()
                .map(|(key, value)| match value {
                    Some(value) => format!("{key}:{}", qvalue_text(value)),
                    None => key.to_string(),
                })
                .collect();
            parts.push(format!("{label}{{{}}}", entries.join(",")));
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
        Arg::Dtype(dtype) | Arg::Opaque(dtype) => dtype.clone(),
        Arg::List(items) => format!("[{}]", list(items)),
        Arg::None => "None".to_string(),
    }
}

fn list(items: &[Arg]) -> String {
    items.iter().map(arg_text).collect::<Vec<_>>().join(", ")
}

fn val_text(val: &Val) -> String {
    match val {
        Val::Tensor { dtype, shape } => {
            let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
            format!("{dtype}[{}]", dims.join(", "))
        }
        Val::Other(text) => text.clone(),
        Val::Missing => "<no val>".to_string(),
    }
}

fn qvalue_text(value: &QValue) -> String {
    let join = |items: Vec<String>| items.join(",");
    let mut text = if value.per_channel {
        format!(
            "{}([{}],[{}],{},{}",
            value.dtype,
            join(value.scales.iter().map(|s| format!("{s:?}")).collect()),
            join(value.zps.iter().map(ToString::to_string).collect()),
            value.qmin,
            value.qmax
        )
    } else {
        format!(
            "{}({:?},{},{},{}",
            value.dtype, value.scales[0], value.zps[0], value.qmin, value.qmax
        )
    };
    if value.per_channel || value.axis != 0 {
        let _ = write!(text, ",axis={}", value.axis);
    }
    text.push(')');
    text
}

fn data_text(data: &Data) -> String {
    match data {
        Data::Bytes(bytes) if bytes.len() <= DATA_HEX_LIMIT => format!("hex:{}", hex_encode(bytes)),
        Data::Bytes(bytes) => format!("sha256:{}", sha256_hex(bytes)),
        Data::Sha256(digest) => format!("sha256:{digest}"),
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, node) in self.nodes.iter().enumerate() {
            write!(f, "    %{position} = ")?;
            match &node.op {
                Op::Placeholder { name, .. } => write!(f, "placeholder {name}")?,
                Op::GetAttr { name } => write!(f, "get_attr {name}")?,
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
            write!(f, " : {}", val_text(&node.val))?;
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
            if let Some(data) = &node.data {
                write!(f, " data={}", data_text(data))?;
            }
            if let Some(special) = &node.special {
                write!(f, " special={special}")?;
            }
            if node.no_fuse {
                write!(f, " no_fuse")?;
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
// SHA-256 (FIPS 180-4), for constant data over DATA_HEX_LIMIT
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The SHA-256 digest of `data` in lowercase hex, as Python's
/// `hashlib.sha256(data).hexdigest()` prints it.
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bits.to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (slot, word) in w.iter_mut().zip(block.chunks_exact(4)) {
            *slot = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for (k, w) in SHA256_K.iter().zip(w) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(*k)
                .wrapping_add(w);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
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

    /// Format 2: real qparams, data, special dtypes, flags, a tuple val, a
    /// get_attr node and a device kwarg.
    const DETAILED: &str = "\
format 2
case conv
before
%0 = placeholder w : torch.int8[2, 1, 1, 1] kind=buffer data=hex:01ff special=INT4 q=none
%1 = placeholder z : torch.int64[2] kind=buffer data=hex:0300000000000000fdffffffffffffff q=none
%2 = get_attr c : torch.float32[1] data=hex:0000803f q=none
%3 = placeholder x : torch.int8[1, 1, 2, 2] q=none
%4 = aten.convolution.default(%3, %0, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1) : torch.int8[1, 2, 2, 2] q=in{0:torch.int8(0.5,-3,-128,127),1:torch.int8([0.25,0.125],[0,0],-7,7,axis=0)}/out{0:torch.int8(0.75,1,-128,127)}
%5 = aten.permute_copy.default(%4, [0, 2, 3, 1]) : torch.int8[1, 2, 2, 2] q=in{}/out{}
%6 = aten.full.default([1], 3, device=device:cpu, dtype=torch.int8) : torch.int8[1] q=none
%7 = aten.max_pool2d_with_indices.default(%5, [1, 1]) : (torch.int8[1, 2, 2, 2], torch.int64[1, 2, 2, 2]) q=in{0:torch.int8(0.75,1,-128,127)}/out{}
%8 = <built-in function getitem>(%7, 0) : torch.int8[1, 2, 2, 2] q=in{}/out{0:torch.int8(0.75,1,-128,127)}
%9 = tosa.CONST_SHAPE.default([2, 2]) : (2, 2) special=SHAPE q=in{}/out{}
%10 = tosa.RESCALE.default(%8, torch.int32, [1.5], 1, 0) : torch.int32[1, 2, 2, 2] special=INT48 no_fuse q=in{}/out{}
output([%10, %6, %9])
after modified=true
%0 = placeholder w : torch.int8[2, 1, 1, 1] kind=buffer data=hex:01ff special=INT4 q=none
%1 = placeholder z : torch.int64[2] kind=buffer data=hex:0300000000000000fdffffffffffffff q=none
%2 = get_attr c : torch.float32[1] data=hex:0000803f q=none
%3 = placeholder x : torch.int8[1, 1, 2, 2] q=none
%4 = placeholder big : torch.int8[2] kind=buffer data=sha256:a7bd7fd8a61b25b3cd1d35b7ce0c7b2c2dd6c1b97f2d20e0e7a3e7aa0ba3c6a1 q=none
%5 = aten.permute_copy.default(%3, [0, 2, 3, 1]) : torch.int8[1, 2, 2, 1] q=in{}/out{}
output([%5])
end
";

    #[test]
    fn parses_every_part_of_the_format() {
        let cases = parse(FIXTURE).unwrap();
        assert_eq!(cases.len(), 3);
        let case = &cases[0];
        assert_eq!(case.name, "unchanged");
        assert_eq!(case.format, 1);
        assert!(!case.modified);
        assert_eq!(case.before, case.after);
        let nodes = &case.before.nodes;
        assert_eq!(nodes[0].dim_order, Some(vec![0, 2, 3, 1]));
        assert_eq!(
            (nodes[0].q_in.clone(), nodes[0].q_out.clone()),
            (Some(vec![]), Some(vec![(0, None)]))
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
    fn parses_the_format_2_details() {
        let cases = parse(DETAILED).unwrap();
        let case = &cases[0];
        assert_eq!(case.format, 2);
        let nodes = &case.before.nodes;
        assert_eq!(nodes[0].data, Some(Data::Bytes(vec![0x01, 0xff])));
        assert_eq!(nodes[0].special.as_deref(), Some("INT4"));
        assert_eq!(nodes[2].op, Op::GetAttr { name: "c".to_string() });
        let weight = nodes[4].q_in.as_ref().unwrap()[1].1.clone().unwrap();
        assert_eq!(
            weight,
            QValue {
                dtype: "torch.int8".to_string(),
                per_channel: true,
                scales: vec![0.25, 0.125],
                zps: vec![0, 0],
                qmin: -7,
                qmax: 7,
                axis: 0,
            }
        );
        let input = nodes[4].q_in.as_ref().unwrap()[0].1.clone().unwrap();
        assert!(!input.per_channel);
        assert_eq!((input.scales, input.zps), (vec![0.5], vec![-3]));
        let Op::Call { kwargs, .. } = &nodes[6].op else {
            panic!("expected a call");
        };
        assert_eq!(kwargs["device"], Arg::Opaque("device:cpu".to_string()));
        assert_eq!(
            nodes[7].val,
            Val::Other("(torch.int8[1, 2, 2, 2], torch.int64[1, 2, 2, 2])".to_string())
        );
        assert_eq!(nodes[9].val, Val::Other("(2, 2)".to_string()));
        assert!(nodes[10].no_fuse && !nodes[9].no_fuse);
        assert!(matches!(case.after.nodes[4].data, Some(Data::Sha256(_))));
    }

    #[test]
    fn printing_round_trips() {
        for fixture in [FIXTURE, DETAILED] {
            for case in parse(fixture).unwrap() {
                let text = format!(
                    "format {}\ncase {}\nbefore\n{}after modified=false\n{}end\n",
                    case.format, case.name, case.before, case.after
                )
                .replace("    ", "");
                let reparsed = parse(&text).unwrap();
                assert_eq!(reparsed[0].before, case.before);
                assert_eq!(reparsed[0].after, case.after);
            }
        }
    }

    #[test]
    fn built_graphs_read_back_as_the_fixture() {
        for fixture in [FIXTURE, DETAILED] {
            for case in parse(fixture).unwrap() {
                let graph = snapshot(&build(&case.before)).unwrap();
                let strict = Options {
                    dim_orders: DimOrders::All,
                    dead_nodes: true,
                    float32_activations: false,
                };
                assert_eq!(compare(&case.before, &graph, strict), Ok(()), "{}", case.name);
            }
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
    fn the_build_imports_format_2_details_like_the_importer() {
        let cases = parse(DETAILED).unwrap();
        let graph = build(&cases[0].before);
        let weight = graph.node(0);
        let tensor = weight.tensor.as_ref().unwrap();
        assert_eq!(tensor.data.data, vec![0x01, 0xff]);
        assert_eq!(weight.meta.extra_str(SPECIAL_DTYPE), Some("INT4"));
        // `extract_tensor_data` narrows int64 to int32.
        let zero_points = graph.node(1).tensor.as_ref().unwrap();
        assert_eq!(zero_points.data.dtype, DType::Int32);
        assert_eq!(zero_points.data.data, vec![3, 0, 0, 0, 0xfd, 0xff, 0xff, 0xff]);
        assert!(matches!(
            graph.node(2).tensor.as_ref().unwrap().source,
            TensorSource::GraphModuleAttr
        ));
        let conv = graph.node(4);
        let input = conv.meta.input_qparam_at(0).unwrap();
        assert_eq!((input.scale, input.zp, input.qmin), (0.5, -3, -128));
        // `QParam::from_py` on a per-channel QuantArgs.
        let weight = conv.meta.input_qparam_at(1).unwrap();
        assert!(weight.is_per_channel());
        assert_eq!((weight.scale, weight.zp), (1.0, 0));
        assert_eq!(weight.per_channel_scales, vec![0.25, 0.125]);
        assert_eq!((weight.qmin, weight.qmax), (-7, 7));
        // A device kwarg imports as None; a tuple val as no val.
        assert!(matches!(graph.node(6).kwargs["device"], IrArg::None));
        assert!(graph.node(7).meta.tensor_facts().is_none());
        assert!(graph.node(7).meta.val_dim_order().is_none());
        assert_eq!(graph.node(10).meta.extra_bool(DO_NOT_FUSE), Some(true));
    }

    #[test]
    fn retab_dtypes_float_quantized_activations() {
        let cases = parse(DETAILED).unwrap();
        let retab = retab_dtypes(&cases[0].before);
        let dtype = |position: usize| retab.nodes[position].tensor_dtype().map(str::to_string);
        // Placeholders keep their storage dtype; the conv and the getitem carry
        // int8 qparams; the permute reads the conv; the full carries none.
        assert_eq!(dtype(0).as_deref(), Some("torch.int8"));
        assert_eq!(dtype(3).as_deref(), Some("torch.int8"));
        assert_eq!(dtype(4).as_deref(), Some(FLOAT32));
        assert_eq!(dtype(5).as_deref(), Some(FLOAT32));
        assert_eq!(dtype(6).as_deref(), Some("torch.int8"));
        assert_eq!(dtype(8).as_deref(), Some(FLOAT32));
        assert_eq!(dtype(10).as_deref(), Some("torch.int32"));
        // A data-movement op of an int8 placeholder stays int8.
        assert_eq!(
            retab_dtypes(&cases[0].after).nodes[5].tensor_dtype(),
            Some("torch.int8")
        );
    }

    #[test]
    fn float32_activations_only_stand_in_for_quantized_values() {
        let cases = parse(DETAILED).unwrap();
        let before = &cases[0].before;
        let lenient = Options {
            float32_activations: true,
            ..Options::default()
        };
        let mismatch = |position: usize, dtype: &str, options: Options| {
            let mut graph = snapshot(&build(before)).unwrap();
            if let Val::Tensor { dtype: slot, .. } = &mut graph.nodes[position].val {
                *slot = dtype.to_string();
            }
            compare(before, &graph, options).err().map(|m| m.class)
        };
        // The conv carries an int8 qparam; the permute moves data.
        assert_eq!(mismatch(4, FLOAT32, lenient), None);
        assert_eq!(mismatch(5, FLOAT32, lenient), None);
        assert_eq!(mismatch(4, FLOAT32, Options::default()), Some("dtype"));
        // The full carries no qparam, the RESCALE is int32 and a placeholder
        // keeps its storage dtype.
        assert_eq!(mismatch(6, FLOAT32, lenient), Some("dtype"));
        assert_eq!(mismatch(10, FLOAT32, lenient), Some("dtype"));
        assert_eq!(mismatch(3, FLOAT32, lenient), Some("dtype"));
        assert_eq!(mismatch(4, "torch.int16", lenient), Some("dtype"));
    }

    #[test]
    fn format_2_details_are_compared() {
        let cases = parse(DETAILED).unwrap();
        let before = &cases[0].before;
        let class = |edit: &dyn Fn(&mut Graph)| {
            let mut graph = snapshot(&build(before)).unwrap();
            edit(&mut graph);
            compare(before, &graph, Options::default())
                .err()
                .map(|mismatch| mismatch.class)
        };
        assert_eq!(class(&|_| {}), None);
        assert_eq!(
            class(&|graph| graph.nodes[0].data = Some(Data::Bytes(vec![1, 0xfe]))),
            Some("data")
        );
        assert_eq!(class(&|graph| graph.nodes[0].data = None), Some("data"));
        assert_eq!(
            class(&|graph| graph.nodes[10].special = None),
            Some("special")
        );
        assert_eq!(
            class(&|graph| graph.nodes[10].no_fuse = false),
            Some("no-fuse")
        );
        assert_eq!(
            class(&|graph| {
                let map = graph.nodes[4].q_in.as_mut().unwrap();
                map[1].1.as_mut().unwrap().scales[1] = 0.5;
            }),
            Some("qparams")
        );
        assert_eq!(
            class(&|graph| graph.nodes[7].val = Val::Tensor {
                dtype: "torch.int8".to_string(),
                shape: vec![1]
            }),
            Some("val")
        );
        // RETAB's own spellings: `backend.tosa.X` and a kwarg left out
        // instead of the device the importer reads as None.
        assert_eq!(
            class(&|graph| {
                if let Op::Call { target, .. } = &mut graph.nodes[10].op {
                    *target = "backend.tosa.RESCALE.default".to_string();
                }
                if let Op::Call { kwargs, .. } = &mut graph.nodes[6].op {
                    kwargs.remove("device");
                }
            }),
            None
        );
    }

    #[test]
    fn digests_stand_in_for_large_data() {
        let digest = sha256_hex(&[7, 7]);
        assert_eq!(data_matches(&Data::Sha256(digest.clone()), Some(&Data::Bytes(vec![7, 7]))), Ok(()));
        assert!(data_matches(&Data::Sha256(digest), Some(&Data::Bytes(vec![7, 8]))).is_err());
        let large = Data::Bytes(vec![0; DATA_HEX_LIMIT + 1]);
        assert!(data_text(&large).starts_with("sha256:"));
        assert!(data_text(&Data::Bytes(vec![0; DATA_HEX_LIMIT])).starts_with("hex:"));
    }

    #[test]
    fn sha256_matches_the_fips_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
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
    fn format_2_reports_show_the_imported_graph() {
        let results = replay(&Noop, DETAILED, Options::default());
        let report = &results[0].report;
        assert_eq!(results[0].mismatch.as_ref().map(|m| m.class), Some("op"));
        assert!(report.contains("before (as imported, with RETAB's float32 activations)"));
        assert!(report.contains("aten.convolution.default(%3, %0, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1) : torch.float32[1, 2, 2, 2]"), "{report}");
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
            // Format 2 needs every qparam's value and every constant's data.
            "format 2\ncase x\nbefore\n%0 = placeholder x : torch.float32[2] q=out{0}\noutput([%0])\nafter modified=false same\nend\n",
            "format 2\ncase x\nbefore\n%0 = placeholder x : torch.float32[2] kind=param q=none\noutput([%0])\nafter modified=false same\nend\n",
            "format 3\ncase x\nbefore\n%0 = placeholder x : torch.float32[2] q=none\noutput([%0])\nafter modified=false same\nend\n",
            "case x\nbefore\n%0 = placeholder x : torch.int8[2] data=hex:0 q=none\noutput([%0])\nafter modified=false same\nend\n",
            "case x\nbefore\n%0 = placeholder x : torch.int8[2] q=in{0:torch.int8(0.5,0,-128)} \noutput([%0])\nafter modified=false same\nend\n",
            "case x\nbefore\n%0 = placeholder x : torch.int8[2] q=in{1,0}\noutput([%0])\nafter modified=false same\nend\n",
            "case x\nbefore\n%0 = get_attr x : torch.int8[2] kind=param q=none\noutput([%0])\nafter modified=false same\nend\n",
        ];
        for text in bad {
            assert!(parse(text).is_err(), "{text}");
        }
    }
}
