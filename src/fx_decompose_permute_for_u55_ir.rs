//! IrPass port of rc4's DecomposePermuteForU55Pass.
//!
//! Ethos-U55's TRANSPOSE has rank-dependent dimension-product limits. A permute that
//! Vela can't run on the NPU becomes `slice_copy -> permute -> cat` chunks, each small
//! enough to pass upstream's worst-case rule. rc4 judges every permute alive at its slot
//! (`arm_pass_manager.py:736`): one that passes the worst-case rule stays, and any other
//! is split when rc4's exact check, a Vela compile of a one-TRANSPOSE probe, rejects it.
//! The answer depends on the installed Vela, so it comes from Python.
//!
//! In explicit mode a node's val is its TOSA shape, and most permutes alive at the slot
//! are ones RETAB's passes created (IrRewriteConv's NHWC and NCHW permutes, the pool and
//! upsample permutes, the propagation's fused permutes). There `run_fx_passes` installs
//! Python's exact check as a [`TransposeProbe`] ([`with_transpose_probe`]), and this pass
//! asks it about each live permute that breaks the worst-case rule, as rc4 does
//! (`preprocess._u55_transpose_probe`).
//!
//! Dim-order mode, and explicit mode without a probe (Rust tests), decide by name instead:
//! Python lists the edge permutes to split (`PipelineConfig::u55_split_permutes`,
//! `preprocess._u55_permutes_to_split`). A pass that hands a listed permute's role to a
//! copy under a new name (names are never reused, design F9) carries the decision on the
//! copy's meta instead ([`mark_split`]), and this pass splits a marked permute as it
//! splits a listed one. Explicit mode's IrRewriteConv marks the copy of an A16W8 conv's
//! source permute on its widened INT32 branch (`layout::rewrite_conv`).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::fx_ir::{IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta};
use crate::layout::dtype::value_dtype;

const MAX_PRODUCT: i64 = 1 << 16;

/// rc4's exact check, `DecomposePermuteForU55Pass._violates_exact_constraint`
/// (`decompose_permute_for_u55_pass.py:113-138`): whether Vela rejects a TRANSPOSE by
/// `perm` of a tensor of `shape` whose dtype has the torch name `dtype` (empty when the
/// IR has none). rc4 rejects every dtype but int8, int16 and bool without asking Vela.
pub(crate) type TransposeProbe = dyn Fn(&[i64], &[i64], &str) -> bool;

/// An installed [`TransposeProbe`] and its verdicts by (shape, dims, dtype): each new
/// question costs a Vela compile, tens of milliseconds.
struct InstalledProbe {
    check: Box<TransposeProbe>,
    verdicts: HashMap<(Vec<i64>, Vec<i64>, String), bool>,
}

thread_local! {
    /// The probe of the running pipeline. Like the stashed IR it lives on the thread
    /// that called `run_fx_passes`, which holds the GIL throughout.
    static PROBE: RefCell<Option<InstalledProbe>> = const { RefCell::new(None) };
}

/// Run `f` with `probe` as this thread's [`TransposeProbe`] (none when `None`), then
/// restore the previous one, also when `f` unwinds.
pub(crate) fn with_transpose_probe<R>(
    probe: Option<Box<TransposeProbe>>,
    f: impl FnOnce() -> R,
) -> R {
    struct Restore(Option<InstalledProbe>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            PROBE.with(|slot| *slot.borrow_mut() = previous);
        }
    }
    let installed = probe.map(|check| InstalledProbe {
        check,
        verdicts: HashMap::new(),
    });
    let _restore = Restore(PROBE.with(|slot| slot.replace(installed)));
    f()
}

fn probe_installed() -> bool {
    PROBE.with(|slot| slot.borrow().is_some())
}

/// The installed probe's verdict, asked once per (shape, dims, dtype) while it is
/// installed; `None` without a probe.
fn probe_verdict(shape: &[i64], perm: &[i64], dtype: &str) -> Option<bool> {
    PROBE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let probe = slot.as_mut()?;
        let key = (shape.to_vec(), perm.to_vec(), dtype.to_owned());
        Some(
            *probe
                .verdicts
                .entry(key)
                .or_insert_with(|| (probe.check)(shape, perm, dtype)),
        )
    })
}

/// The meta key of [`mark_split`]. RETAB's own: rc4 needs no such mark, as it decides
/// at the pass itself.
const SPLIT_MARK: &str = "retab_u55_split";

/// Mark the permute whose meta this is to be split on U55, as if its name were in
/// `split_permutes`.
pub(crate) fn mark_split(meta: &mut NodeMeta) {
    meta.set_extra(SPLIT_MARK, MetaValue::Bool(true));
}

/// Whether [`mark_split`] marked this meta.
pub(crate) fn is_marked_split(meta: &NodeMeta) -> bool {
    meta.extra_bool(SPLIT_MARK) == Some(true)
}

pub struct IrDecomposePermuteForU55Pass {
    /// The edge permutes to split, by name: Python's `_u55_permutes_to_split`. Permutes
    /// marked by [`mark_split`] are split too.
    pub split_permutes: HashSet<String>,
    /// Explicit mode's pass: with a [`TransposeProbe`] installed, the probe alone decides
    /// whether a live permute that breaks the worst-case rule is split, as rc4's exact
    /// check does, and names and marks are not consulted. Dim-order mode's permutes don't
    /// have their TOSA shapes yet, so its pass never asks.
    pub probe_live_permutes: bool,
}

fn combinations(rank: usize, len: usize) -> Vec<Vec<usize>> {
    fn extend(start: usize, rank: usize, len: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == len {
            out.push(current.clone());
            return;
        }
        for axis in start..rank {
            current.push(axis);
            extend(axis + 1, rank, len, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    extend(0, rank, len, &mut Vec::new(), &mut out);
    out
}

fn violating_combinations(shape: &[i64]) -> Vec<Vec<usize>> {
    let rank = shape.len();
    if rank == 0 {
        return Vec::new();
    }
    combinations(rank, rank.saturating_sub(2).max(1))
        .into_iter()
        .filter(|combo| combo.iter().map(|&axis| shape[axis]).product::<i64>() > MAX_PRODUCT)
        .collect()
}

/// Upstream `_find_safe_split_chunk_limits`: a greedy per-axis chunk limit that makes
/// the worst-case rule pass. `None` when no axis can be reduced.
fn split_chunk_limits(shape: &[i64]) -> Option<BTreeMap<usize, i64>> {
    let mut effective = shape.to_vec();
    let mut limits = BTreeMap::new();
    loop {
        let combos = violating_combinations(&effective);
        if combos.is_empty() {
            return Some(limits);
        }
        let mut candidates: Vec<usize> = combos
            .iter()
            .flatten()
            .copied()
            .filter(|&axis| effective[axis] > 1)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        candidates.sort_by_key(|&axis| std::cmp::Reverse(effective[axis]));
        let reduced = candidates.into_iter().find_map(|axis| {
            let max_other = combos
                .iter()
                .filter(|combo| combo.contains(&axis))
                .map(|combo| {
                    combo
                        .iter()
                        .filter(|&&other| other != axis)
                        .map(|&other| effective[other])
                        .product::<i64>()
                })
                .max()
                .unwrap_or(1)
                .max(1);
            let safe = MAX_PRODUCT / max_other;
            (safe >= 1 && effective[axis] > safe).then_some((axis, safe))
        });
        let (axis, safe) = reduced?;
        let limit = limits.entry(axis).or_insert(effective[axis]);
        *limit = (*limit).min(safe);
        effective[axis] = safe;
    }
}

fn chunk_ranges(size: i64, max_chunk: i64) -> Vec<(i64, i64)> {
    (0..size)
        .step_by(max_chunk as usize)
        .map(|start| (start, (start + max_chunk).min(size)))
        .collect()
}

struct Split<'a> {
    name: String,
    perm: &'a [i64],
    axes: Vec<(usize, i64)>,
    meta: crate::fx_ir::NodeMeta,
}

fn emit(
    cursor: &mut crate::fx_ir::IrRewriteCursor<'_>,
    split: &Split<'_>,
    source: NodeId,
    shape: Vec<i64>,
    depth: usize,
    tag: &str,
) -> NodeId {
    let permuted = |shape: &[i64]| -> Vec<i64> {
        split.perm.iter().map(|&axis| shape[axis as usize]).collect()
    };
    let Some(&(axis, limit)) = split.axes.get(depth) else {
        let out_shape = permuted(&shape);
        return cursor.permute_copy(
            format!("{}_u55{tag}_permute", split.name),
            source,
            split.perm,
            split.meta.cloned_with_same_dtype_or_float32(out_shape),
        );
    };
    let chunks: Vec<NodeId> = chunk_ranges(shape[axis], limit)
        .into_iter()
        .enumerate()
        .map(|(index, (start, end))| {
            let mut chunk_shape = shape.clone();
            chunk_shape[axis] = end - start;
            let chunk_tag = format!("{tag}_{index}");
            let slice = cursor.slice_copy(
                format!("{}_u55{chunk_tag}_slice", split.name),
                source,
                axis as i64,
                start,
                end,
                None,
                split.meta.cloned_with_same_dtype_or_float32(chunk_shape.clone()),
            );
            emit(cursor, split, slice, chunk_shape, depth + 1, &chunk_tag)
        })
        .collect();
    if chunks.len() == 1 {
        return chunks[0];
    }
    let cat_dim = split.perm.iter().position(|&p| p as usize == axis).unwrap();
    cursor.call_function(
        format!("{}_u55{tag}_cat", split.name),
        "aten.cat.default",
        vec![
            IrArg::List(chunks.into_iter().map(IrArg::Node).collect()),
            IrArg::Int(cat_dim as i64),
        ],
        split.meta.cloned_with_same_dtype_or_float32(permuted(&shape)),
    )
}

fn violates_worst_case(shape: &[i64]) -> bool {
    !violating_combinations(shape).is_empty()
}

/// Append one line about a permute this pass judged to the `RETAB_U55_PERMUTE_LOG`
/// file: its name, its input's shape and value dtype, its dims, what decided it and
/// whether it is split. Write errors are ignored.
fn log_judgement(path: &str, line: String) {
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}

impl IrDecomposePermuteForU55Pass {
    /// rc4's `call_operator` for the permute `id`, if it is one: split it when its input
    /// breaks the worst-case rule and the probe rejects it, or, without a probe, when it
    /// is listed or marked.
    fn decompose_one(&self, graph: &mut IrGraph, id: NodeId, log: Option<&str>) -> bool {
        let node = &graph.nodes[id];
        if !matches!(
            node.target.as_str(),
            "aten.permute_copy.default" | "aten.permute.default"
        ) {
            return false;
        }
        let (Some(input), Some(perm)) = (node.first_node_arg(), node.arg_int_list(1)) else {
            return false;
        };
        let Some(shape) = graph.node_shape_vec(input) else {
            return false;
        };
        let rank = shape.len() as i64;
        if rank == 0 || perm.len() != shape.len() {
            return false;
        }
        let perm: Vec<i64> = perm.iter().map(|&axis| axis.rem_euclid(rank)).collect();
        // rc4's quick check: a permute that passes the worst-case rule runs on U55.
        if !violates_worst_case(&shape) {
            return false;
        }
        let listed = self.split_permutes.contains(&node.name);
        let marked = is_marked_split(&node.meta);
        // rc4 asks about `args[0].data.dtype`, the integer dtype its retraced vals hold.
        let dtype = value_dtype(graph, input).unwrap_or_default();
        let verdict = if self.probe_live_permutes {
            probe_verdict(&shape, &perm, &dtype)
        } else {
            None
        };
        let split = verdict.unwrap_or(listed || marked);
        if let Some(path) = log {
            let probe = verdict.map_or("-".to_string(), |verdict| verdict.to_string());
            log_judgement(
                path,
                format!(
                    "{}\tshape={shape:?}\tperm={perm:?}\tdtype={dtype}\tlisted={listed}\tmarked={marked}\tprobe={probe}\tsplit={split}\n",
                    node.name
                ),
            );
        }
        if !split {
            return false;
        }
        let Some(limits) = split_chunk_limits(&shape) else {
            return false;
        };
        // The chunks pass the worst-case rule, so none carries the decision on, as none
        // has a listed name.
        let mut meta = node.meta.clone();
        meta.extras.remove(SPLIT_MARK);
        let split = Split {
            name: node.name.clone(),
            perm: &perm,
            axes: limits.into_iter().collect(),
            meta,
        };
        let mut cursor = graph.rewrite_cursor(id);
        let result = emit(&mut cursor, &split, input, shape, 0, "");
        cursor.replace_anchor_with(result);
        true
    }
}

impl IrPass for IrDecomposePermuteForU55Pass {
    fn name(&self) -> &str {
        "IrDecomposePermuteForU55"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let log = crate::config::u55_permute_log_file();
        let probing = self.probe_live_permutes && probe_installed();
        if log.is_none()
            && !probing
            && self.split_permutes.is_empty()
            && !graph.live_nodes().any(|node| is_marked_split(&node.meta))
        {
            return false;
        }
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            modified |= self.decompose_one(graph, id, log.as_deref());
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::fx_ir::{i64_list_arg, NodeOp};
    use crate::test_ir::{call, f32_call, f32_placeholder, output, placeholder, qparam};

    const I8: &str = "torch.int8";

    /// A question to the probe: the permute input's shape, the dims and the dtype.
    type Question = (Vec<i64>, Vec<i64>, String);

    /// A fake [`TransposeProbe`] that rejects the inputs of the shapes in `rejected`,
    /// with the questions it is asked.
    fn fake_probe(rejected: &[&[i64]]) -> (Box<TransposeProbe>, Rc<RefCell<Vec<Question>>>) {
        let questions = Rc::new(RefCell::new(Vec::new()));
        let asked = Rc::clone(&questions);
        let rejected: Vec<Vec<i64>> = rejected.iter().map(|shape| shape.to_vec()).collect();
        let probe = Box::new(move |shape: &[i64], perm: &[i64], dtype: &str| {
            asked.borrow_mut().push((shape.to_vec(), perm.to_vec(), dtype.to_string()));
            rejected.iter().any(|rejected| rejected == shape)
        });
        (probe, questions)
    }

    fn question(shape: &[i64], perm: &[i64], dtype: &str) -> Question {
        (shape.to_vec(), perm.to_vec(), dtype.to_string())
    }

    /// `permute_copy(input, dims)`, its val the input's dims permuted, in the input's
    /// val dtype.
    fn permute(graph: &mut IrGraph, name: &str, input: NodeId, dims: &[i64]) -> NodeId {
        let shape = graph.node_shape_vec(input).unwrap();
        let rank = shape.len() as i64;
        let out: Vec<i64> = dims.iter().map(|&dim| shape[dim.rem_euclid(rank) as usize]).collect();
        let dtype = graph.node_dtype(input).unwrap().to_string();
        call(
            graph,
            name,
            "aten.permute_copy.default",
            vec![IrArg::Node(input), i64_list_arg(dims)],
            out,
            &dtype,
        )
    }

    fn outputs(graph: &mut IrGraph, values: &[NodeId]) {
        graph.add_node(
            NodeOp::Output,
            "output".to_string(),
            "output".to_string(),
            vec![IrArg::List(values.iter().copied().map(IrArg::Node).collect())],
            NodeMeta::default(),
        );
    }

    fn pass(split_permutes: &[&str], probe_live_permutes: bool) -> IrDecomposePermuteForU55Pass {
        IrDecomposePermuteForU55Pass {
            split_permutes: split_permutes.iter().map(|name| name.to_string()).collect(),
            probe_live_permutes,
        }
    }

    /// Whether this pass split the permute `name`: it is gone and its chunks' cat
    /// stands in its place.
    fn was_split(graph: &IrGraph, name: &str) -> bool {
        let cat = format!("{name}_u55_cat");
        graph.live_nodes().all(|node| node.name != name)
            && graph.live_nodes().any(|node| node.name == cat)
    }

    #[test]
    fn plans_match_upstream_for_the_u55_permute_tests() {
        // Shapes from rc4's test_permute.py u55 cases; expected plans computed with
        // rc4's DecomposePermuteForU55Pass._find_safe_split_chunk_limits.
        for (shape, plan) in [
            (vec![65537, 2, 2], vec![(0, 65536)]),
            (vec![2, 2, 257, 257], vec![(2, 255)]),
            (vec![2, 257, 257, 2], vec![(1, 255)]),
            (vec![1, 65537, 2, 2], vec![(1, 32768)]),
            (vec![1, 2, 2, 65537], vec![(3, 32768)]),
            (vec![2, 2, 41, 41, 41], vec![(2, 38)]),
            (vec![2, 3, 4, 5], vec![]),
        ] {
            assert_eq!(split_chunk_limits(&shape), Some(BTreeMap::from_iter(plan)), "{shape:?}");
        }
    }

    #[test]
    fn splits_a_listed_permute_into_sliced_permutes_and_a_cat() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 2, 257, 257]);
        let permute = f32_call(
            &mut graph,
            "permute",
            "aten.permute_copy.default",
            vec![IrArg::Node(x), IrArg::List(vec![IrArg::Int(0), IrArg::Int(2), IrArg::Int(3), IrArg::Int(1)])],
            [2, 257, 257, 2],
        );
        let out = output(&mut graph, permute);
        let pass = IrDecomposePermuteForU55Pass {
            split_permutes: HashSet::from(["permute".to_string()]),
            probe_live_permutes: false,
        };

        assert!(pass.run(&mut graph));
        let result = graph.node(out).first_node_arg().unwrap();
        let cat = graph.node(result);
        assert_eq!(cat.target, "aten.cat.default");
        assert!(matches!(cat.args[1], IrArg::Int(1)));
        let IrArg::List(chunks) = &cat.args[0] else { panic!("cat inputs") };
        let shapes: Vec<_> = chunks
            .iter()
            .map(|chunk| graph.node_shape_vec(chunk.as_node().unwrap()).unwrap())
            .collect();
        assert_eq!(shapes, vec![vec![2, 255, 257, 2], vec![2, 2, 257, 2]]);
        assert_eq!(graph.node_shape_vec(result), Some(vec![2, 257, 257, 2]));
    }

    /// A permute whose role a rewrite handed to it from a listed one carries the split
    /// in its meta: it is split under its own name with no name listed at all, and the
    /// chunks it becomes don't carry the mark on.
    #[test]
    fn splits_a_marked_permute_whatever_its_name() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 2, 257, 257]);
        let permute = f32_call(
            &mut graph,
            "permute_int32",
            "aten.permute_copy.default",
            vec![IrArg::Node(x), IrArg::List(vec![IrArg::Int(0), IrArg::Int(2), IrArg::Int(3), IrArg::Int(1)])],
            [2, 257, 257, 2],
        );
        mark_split(&mut graph.node_mut(permute).meta);
        let out = output(&mut graph, permute);
        let pass = IrDecomposePermuteForU55Pass {
            split_permutes: HashSet::new(),
            probe_live_permutes: false,
        };

        assert!(pass.run(&mut graph));
        assert!(!graph.node(permute).alive);
        let cat = graph.node(out).first_node_arg().unwrap();
        assert_eq!(graph.node(cat).target, "aten.cat.default");
        assert_eq!(graph.node(cat).name, "permute_int32_u55_cat");
        let created: Vec<_> = graph.live_nodes().filter(|node| node.name.starts_with("permute_int32_u55")).collect();
        assert_eq!(created.len(), 5, "two slices, two permutes and the cat");
        assert!(created.iter().all(|node| !is_marked_split(&node.meta)));
        assert!(!pass.run(&mut graph), "the chunks are not split again");
    }

    #[test]
    fn leaves_unlisted_permutes_alone() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 2, 257, 257]);
        let permute = f32_call(
            &mut graph,
            "permute",
            "aten.permute_copy.default",
            vec![IrArg::Node(x), IrArg::List(vec![IrArg::Int(0), IrArg::Int(2), IrArg::Int(3), IrArg::Int(1)])],
            [2, 257, 257, 2],
        );
        output(&mut graph, permute);
        let pass = IrDecomposePermuteForU55Pass {
            split_permutes: HashSet::new(),
            probe_live_permutes: false,
        };
        assert!(!pass.run(&mut graph));
    }

    /// Explicit mode asks the probe about every live permute whose input breaks the
    /// worst-case rule, with the dims normalized and the input's dtype, as rc4's
    /// `call_operator` asks `_violates_exact_constraint`, and splits what it rejects.
    /// A permute that passes the rule is never asked about.
    #[test]
    fn explicit_pass_asks_the_probe_about_each_worst_case_violator() {
        let mut graph = IrGraph::empty();
        let a = placeholder(&mut graph, "a", [2, 2, 257, 257], I8);
        let rejected = permute(&mut graph, "rejected", a, &[0, 2, 3, 1]);
        let b = placeholder(&mut graph, "b", [1, 3, 320, 320], I8);
        let accepted = permute(&mut graph, "accepted", b, &[0, -2, -1, 1]);
        let c = placeholder(&mut graph, "c", [2, 3, 4, 5], I8);
        let small = permute(&mut graph, "small", c, &[0, 2, 3, 1]);
        outputs(&mut graph, &[rejected, accepted, small]);
        let (probe, questions) = fake_probe(&[&[2, 2, 257, 257]]);

        assert!(with_transpose_probe(Some(probe), || pass(&[], true).run(&mut graph)));
        assert_eq!(
            *questions.borrow(),
            vec![
                question(&[2, 2, 257, 257], &[0, 2, 3, 1], I8),
                question(&[1, 3, 320, 320], &[0, 2, 3, 1], I8),
            ]
        );
        assert!(was_split(&graph, "rejected"));
        assert!(graph.node(accepted).alive);
        assert!(graph.node(small).alive);
    }

    /// rc4 decides by its exact check alone: with a probe, a listed or marked permute
    /// that Vela accepts stays whole, and a permute nobody named is split when Vela
    /// rejects it.
    #[test]
    fn the_probe_alone_decides_in_explicit_mode() {
        let mut graph = IrGraph::empty();
        let a = placeholder(&mut graph, "a", [2, 2, 257, 257], I8);
        let listed = permute(&mut graph, "listed", a, &[0, 2, 3, 1]);
        let b = placeholder(&mut graph, "b", [2, 257, 257, 2], I8);
        let marked = permute(&mut graph, "marked", b, &[0, 3, 1, 2]);
        mark_split(&mut graph.node_mut(marked).meta);
        let c = placeholder(&mut graph, "c", [65537, 2, 2], I8);
        let created = permute(&mut graph, "created", c, &[1, 2, 0]);
        outputs(&mut graph, &[listed, marked, created]);
        let (probe, _) = fake_probe(&[&[65537, 2, 2]]);

        assert!(with_transpose_probe(Some(probe), || pass(&["listed"], true).run(&mut graph)));
        assert!(graph.node(listed).alive);
        assert!(graph.node(marked).alive);
        assert!(was_split(&graph, "created"));
    }

    /// Dim-order mode's pass never asks, whatever is installed, and explicit mode's
    /// pass without a probe splits the listed and marked permutes as before.
    #[test]
    fn names_decide_in_dim_order_mode_and_without_a_probe() {
        let build = || {
            let mut graph = IrGraph::empty();
            let a = placeholder(&mut graph, "a", [2, 2, 257, 257], I8);
            let listed = permute(&mut graph, "listed", a, &[0, 2, 3, 1]);
            let b = placeholder(&mut graph, "b", [2, 257, 257, 2], I8);
            let marked = permute(&mut graph, "marked", b, &[0, 3, 1, 2]);
            mark_split(&mut graph.node_mut(marked).meta);
            let c = placeholder(&mut graph, "c", [65537, 2, 2], I8);
            let unnamed = permute(&mut graph, "unnamed", c, &[1, 2, 0]);
            outputs(&mut graph, &[listed, marked, unnamed]);
            (graph, unnamed)
        };

        let (mut graph, unnamed) = build();
        let (probe, questions) = fake_probe(&[&[2, 2, 257, 257], &[2, 257, 257, 2], &[65537, 2, 2]]);
        assert!(with_transpose_probe(Some(probe), || pass(&["listed"], false).run(&mut graph)));
        assert!(questions.borrow().is_empty());
        assert!(was_split(&graph, "listed") && was_split(&graph, "marked"));
        assert!(graph.node(unnamed).alive);

        let (mut graph, unnamed) = build();
        assert!(!probe_installed());
        assert!(pass(&["listed"], true).run(&mut graph));
        assert!(was_split(&graph, "listed") && was_split(&graph, "marked"));
        assert!(graph.node(unnamed).alive);
    }

    /// A Vela compile per question: the probe hears each (shape, dims, dtype) once
    /// while it is installed, and is gone when `with_transpose_probe` returns.
    #[test]
    fn asks_once_per_shape_dims_and_dtype() {
        let mut graph = IrGraph::empty();
        let a = placeholder(&mut graph, "a", [1, 3, 320, 320], I8);
        let first = permute(&mut graph, "first", a, &[0, 2, 3, 1]);
        let second = permute(&mut graph, "second", a, &[0, -2, -1, -3]);
        let b = placeholder(&mut graph, "b", [1, 3, 320, 320], "torch.int16");
        let wider = permute(&mut graph, "wider", b, &[0, 2, 3, 1]);
        outputs(&mut graph, &[first, second, wider]);
        let (probe, questions) = fake_probe(&[]);

        assert!(!with_transpose_probe(Some(probe), || pass(&[], true).run(&mut graph)));
        assert_eq!(
            *questions.borrow(),
            vec![
                question(&[1, 3, 320, 320], &[0, 2, 3, 1], I8),
                question(&[1, 3, 320, 320], &[0, 2, 3, 1], "torch.int16"),
            ]
        );
        assert!(!probe_installed());
    }

    /// The probe hears the dtype rc4's retraced fake tensor holds for the TRANSPOSE's
    /// input (`layout::dtype::value_dtype`): a quantized compute node's output
    /// quantizer dtype though its val is float32, a RESCALE's own dtype, and for a
    /// qparam-less layout copy holding its producer's float32 val, the producer's.
    #[test]
    fn the_probe_hears_the_transpose_storage_dtype() {
        let mut graph = IrGraph::empty();
        let x = f32_placeholder(&mut graph, "x", [2, 2, 257, 257]);
        let relu = f32_call(&mut graph, "relu", "aten.relu.default", vec![IrArg::Node(x)], [2, 2, 257, 257]);
        graph.node_mut(relu).meta.set_output_qparams_from([(0, qparam(0.1, 0, I8))]);
        let quantized = permute(&mut graph, "quantized", relu, &[0, 2, 3, 1]);
        let rescale = call(
            &mut graph,
            "rescale",
            "tosa.RESCALE.default",
            vec![IrArg::Node(x), IrArg::Dtype("torch.int32".to_string())],
            [2, 2, 257, 257],
            "torch.int32",
        );
        let widened = permute(&mut graph, "widened", rescale, &[0, 3, 2, 1]);
        let copy = permute(&mut graph, "copy", relu, &[0, 1, 3, 2]);
        let chained = permute(&mut graph, "chained", copy, &[0, 3, 1, 2]);
        outputs(&mut graph, &[quantized, widened, chained]);
        let (probe, questions) = fake_probe(&[]);

        with_transpose_probe(Some(probe), || pass(&[], true).run(&mut graph));
        assert_eq!(graph.node_dtype(copy), Some("torch.float32"));
        assert_eq!(
            *questions.borrow(),
            vec![
                question(&[2, 2, 257, 257], &[0, 2, 3, 1], I8),
                question(&[2, 2, 257, 257], &[0, 3, 2, 1], "torch.int32"),
                question(&[2, 2, 257, 257], &[0, 1, 3, 2], I8),
                question(&[2, 2, 257, 257], &[0, 3, 1, 2], I8),
            ]
        );
    }

    /// An inner installation shadows the outer probe, and the outer one is back once
    /// the inner `f` returns or unwinds.
    #[test]
    fn with_transpose_probe_restores_the_previous_probe() {
        let verdict = || probe_verdict(&[2, 2, 257, 257], &[0, 2, 3, 1], I8);
        let rejecting: Box<TransposeProbe> = Box::new(|_: &[i64], _: &[i64], _: &str| true);
        with_transpose_probe(Some(rejecting), || {
            assert_eq!(verdict(), Some(true));
            let accepting: Box<TransposeProbe> = Box::new(|_: &[i64], _: &[i64], _: &str| false);
            with_transpose_probe(Some(accepting), || assert_eq!(verdict(), Some(false)));
            assert_eq!(verdict(), Some(true));
            let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_transpose_probe(None, || panic!("pipeline panic"))
            }));
            assert!(unwound.is_err());
            assert_eq!(verdict(), Some(true));
        });
        assert_eq!(verdict(), None);
    }
}
