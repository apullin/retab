//! IrPass port of rc4's DecomposePermuteForU55Pass.
//!
//! Ethos-U55's TRANSPOSE has rank-dependent dimension-product limits. A permute that
//! Vela can't run on the NPU becomes `slice_copy -> permute -> cat` chunks, each small
//! enough to pass upstream's worst-case rule. Whether a permute needs it is decided in
//! Python with rc4's own Vela probe (`PipelineConfig::u55_split_permutes`), because the
//! answer depends on the installed Vela.
//!
//! That decision is keyed by the edge program's node names, while rc4 judges whatever
//! permutes are alive when its pass runs. A pass that hands a listed permute's role to a
//! copy under a new name (names are never reused, design F9) carries the decision on the
//! copy's meta instead ([`mark_split`]), and this pass splits a marked permute as it
//! splits a listed one. Explicit mode's IrRewriteConv marks the copy of an A16W8 conv's
//! source permute on its widened INT32 branch (`layout::rewrite_conv`).

use std::collections::{BTreeMap, HashSet};

use crate::fx_ir::{IrArg, IrGraph, IrPass, MetaValue, NodeId, NodeMeta};
use crate::layout::dtype::value_dtype;

const MAX_PRODUCT: i64 = 1 << 16;

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

fn decompose_one(
    graph: &mut IrGraph,
    id: NodeId,
    split_permutes: &HashSet<String>,
    log: Option<&str>,
) -> bool {
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
    let listed = split_permutes.contains(&node.name);
    let marked = is_marked_split(&node.meta);
    let split = listed || marked;
    if let Some(path) = log {
        let dtype = value_dtype(graph, input).unwrap_or_else(|| "?".to_string());
        log_judgement(
            path,
            format!(
                "{}\tshape={shape:?}\tperm={perm:?}\tdtype={dtype}\tlisted={listed}\tmarked={marked}\tsplit={split}\n",
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

impl IrPass for IrDecomposePermuteForU55Pass {
    fn name(&self) -> &str {
        "IrDecomposePermuteForU55"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let log = crate::config::u55_permute_log_file();
        if log.is_none()
            && self.split_permutes.is_empty()
            && !graph.live_nodes().any(|node| is_marked_split(&node.meta))
        {
            return false;
        }
        let mut modified = false;
        for id in graph.live_call_function_ids() {
            modified |= decompose_one(graph, id, &self.split_permutes, log.as_deref());
        }
        modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ir::{f32_call, f32_placeholder, output};

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
        let pass = IrDecomposePermuteForU55Pass { split_permutes: HashSet::new() };

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
        let pass = IrDecomposePermuteForU55Pass { split_permutes: HashSet::new() };
        assert!(!pass.run(&mut graph));
    }
}
