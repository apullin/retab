//! Port of rc4's `FuseDuplicateUsersPass`
//! (`backends/transforms/fuse_duplicate_users_pass.py`), as Arm configures it
//! (`backends/arm/_passes/fuse_duplicate_users_pass.py`): RESCALE users are
//! never fused (`TOSA_EXCLUDED_TARGETS`, `:18`).
//!
//! Users of one producer that have the same signature (op, target, args and
//! kwargs, with nodes compared by identity and values by Python's `==`) are
//! collapsed onto the earliest of them in graph order:
//!
//! ```text
//! y = producer(x); z0 = add(y, bias); z1 = add(y, bias)   ->   z0 feeds both
//! ```
//!
//! [`fuse_duplicate_users`] is the pass body inside a [`LayoutCtx`], for the
//! propagation driver's `fuse_horizontal`
//! (`backends/transforms/propagate_view_copy_permute_pass.py:569-576`);
//! [`IrFuseDuplicateUsersPass`] is the standalone pass at
//! `arm_pass_manager.py:587` and `:765`.
//!
//! Differences from rc4, none visible in the graph:
//!
//! - rc4 ends with an `ExportPass` retrace when it fused anything (`:147-151`),
//!   which renames nodes and re-derives every value. A fusion changes no value,
//!   so the port keeps names and meta.
//! - rc4 groups a producer's users in `node.users` insertion order; RETAB keeps
//!   no insertion order, so users go in graph order. Group order cannot change
//!   the result: signatures include the target, so groups never share a
//!   representative, and the representative is always the earliest user with
//!   the signature in graph order.
//! - Python compares a NaN argument equal only to the same float object;
//!   [`PyValue`] has no value for one, so such a user is never fused (rc4 fuses
//!   two users that share one NaN object, as with `math.nan`).
//!
//! Complexity: every node is visited once, and again only when a fusion
//! touches it (as the producer or the representative), so a call costs
//! O(N + sum of user-list sizes x signature size) plus the fusions. Across
//! calls on one ctx with rc4's exclusions, the attempt memo skips producers
//! whose users haven't changed since a pass found nothing to fuse there.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};

use crate::fx_ir::{IrGraph, IrNode, IrPass, NodeId, NodeOp};
use crate::layout::ctx::{LayoutCtx, RuleId};
use crate::layout::fx_node::{rc4_target, PyValue};

/// `DO_NOT_FUSE_DUPLICATE_META_KEY` (`:17`): a node whose meta sets it is
/// never fused. RETAB's importer keeps it (`NodeMeta::from_fx`).
pub(crate) const DO_NOT_FUSE_DUPLICATE_META_KEY: &str = "do_not_fuse_duplicate";

/// `TOSA_EXCLUDED_TARGETS` (backends/arm/_passes/fuse_duplicate_users_pass.py:18),
/// the exclusions of every rc4 caller: the Arm pass (`:26-27`) and the TOSA
/// propagation overrides (`duplicate_user_fusion_exclusions`,
/// backends/arm/_passes/propagate_view_copy_permute_pass.py:37-38).
pub(crate) const TOSA_EXCLUDED_TARGETS: &[&str] = &["backend.tosa.RESCALE.default"];

/// Memo rule for a producer whose users had nothing to fuse; only used with
/// [`TOSA_EXCLUDED_TARGETS`], so one ctx never mixes exclusion sets in it.
const NOTHING_TO_FUSE: RuleId = RuleId("fuse_duplicate_users");

/// The Arm `FuseDuplicateUsersPass()` (`arm_pass_manager.py:587,765`).
pub(crate) struct IrFuseDuplicateUsersPass;

impl IrPass for IrFuseDuplicateUsersPass {
    fn name(&self) -> &str {
        // `IrFuseDuplicateUsers` is the dim-order pipeline's pass.
        "IrLayoutFuseDuplicateUsers"
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut ctx = LayoutCtx::new(graph);
        let modified = fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS);
        if modified {
            ctx.finish();
        }
        modified
    }
}

/// `FuseDuplicateUsersPass(excluded_targets).call(graph_module)` (`:115-153`)
/// inside `ctx`; returns rc4's `modified`. Targets match in either of RETAB's
/// spellings ([`rc4_target`]).
pub(crate) fn fuse_duplicate_users(ctx: &mut LayoutCtx<'_>, excluded_targets: &[&str]) -> bool {
    let excluded: Vec<Cow<'_, str>> = excluded_targets.iter().map(|target| rc4_target(target)).collect();
    let memo = excluded_targets == TOSA_EXCLUDED_TARGETS;
    // `producers` and `queued_producers` (`:120-127`). The pass creates no
    // nodes, so the graph's size bounds every id.
    let mut producers: VecDeque<NodeId> = ctx.order().iter().collect();
    let mut queued = vec![false; ctx.graph().nodes.len()];
    for &producer in &producers {
        queued[producer] = true;
    }
    let mut modified = false;
    while let Some(producer) = producers.pop_front() {
        queued[producer] = false;
        if ctx.is_erased(producer) || (memo && !ctx.should_retry(NOTHING_TO_FUSE, producer)) {
            continue;
        }
        let mut users = ctx.graph().live_user_ids(producer);
        ctx.order().sort(&mut users);
        let fused = users.len() >= 2 && {
            let groups = candidate_groups(ctx, &users, &excluded);
            fuse_candidate_groups(ctx, producer, &groups, &mut |id| {
                if !std::mem::replace(&mut queued[id], true) {
                    producers.push_back(id);
                }
            })
        };
        if fused {
            modified = true;
        } else if memo {
            ctx.record_attempt(NOTHING_TO_FUSE, producer, users);
        }
    }
    modified
}

/// `_get_candidate_groups` (`:189-213`): call users not excluded, grouped by
/// target, groups of two or more, each in graph order.
fn candidate_groups(ctx: &LayoutCtx<'_>, users: &[NodeId], excluded: &[Cow<'_, str>]) -> Vec<Vec<NodeId>> {
    let mut groups: Vec<Vec<NodeId>> = Vec::new();
    let mut by_target: HashMap<Cow<'_, str>, usize> = HashMap::new();
    for &user in users {
        let node = ctx.node(user);
        if node.op != NodeOp::CallFunction {
            continue;
        }
        let target = rc4_target(&node.target);
        if excluded.contains(&target) {
            continue;
        }
        let index = *by_target.entry(target).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[index].push(user);
    }
    groups.retain(|group| group.len() > 1);
    for group in &mut groups {
        ctx.order().sort(group);
    }
    groups
}

/// `_fuse_candidate_groups` (`:155-187`). Signatures are built as each user
/// comes up, so a fusion that rewires a later user's args is seen.
fn fuse_candidate_groups(
    ctx: &mut LayoutCtx<'_>,
    producer: NodeId,
    groups: &[Vec<NodeId>],
    enqueue_producer: &mut dyn FnMut(NodeId),
) -> bool {
    let mut modified = false;
    let mut signature_to_user: HashMap<Signature, NodeId> = HashMap::new();
    for group in groups {
        for &user in group {
            let Some(signature) = user_signature(ctx.node(user)) else {
                continue;
            };
            let representative = *signature_to_user.entry(signature).or_insert(user);
            if representative == user {
                continue;
            }
            ctx.replace_all_uses_with(user, representative);
            ctx.erase_node(user);
            modified = true;
            // `:182-185`: the fusion can expose another duplicate pair.
            enqueue_producer(producer);
            enqueue_producer(representative);
        }
    }
    modified
}

/// `build_node_signature(node)` (`:69-83`) without the op, which is always
/// `call_function` here.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Signature {
    target: String,
    args: Vec<PyValue>,
    /// Sorted by key, as `_to_hashable` sorts dict items (`:29-31`).
    kwargs: Vec<(String, PyValue)>,
}

/// `_build_user_signature` (`:215-218`): none for a node marked
/// [`DO_NOT_FUSE_DUPLICATE_META_KEY`], or one holding a NaN (see the module
/// docs).
fn user_signature(node: &IrNode) -> Option<Signature> {
    if node.meta.extra_bool(DO_NOT_FUSE_DUPLICATE_META_KEY) == Some(true) {
        return None;
    }
    let args = node.args.iter().map(PyValue::of).collect::<Option<Vec<_>>>()?;
    let mut kwargs = node
        .kwargs
        .iter()
        .map(|(key, value)| Some((key.clone(), PyValue::of(value)?)))
        .collect::<Option<Vec<_>>>()?;
    kwargs.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    Some(Signature {
        target: rc4_target(&node.target).into_owned(),
        args,
        kwargs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::IrArg;
    use crate::fx_meta::MetaValue;
    use crate::layout::testing::*;

    fn run(g: &mut LayoutGraph) -> bool {
        IrFuseDuplicateUsersPass.run(g.graph_mut())
    }

    fn output_args(g: &LayoutGraph) -> Vec<NodeId> {
        let output = g.graph().output_node().expect("an output node");
        let IrArg::List(items) = &output.args[0] else {
            panic!("the output holds a list");
        };
        items.iter().map(|item| item.as_node().expect("a node")).collect()
    }

    #[test]
    fn identical_users_collapse_onto_the_first() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let b = g.input("b", &[2, 3]);
        let first = g.call(ADD, vec![node(x), node(b)]);
        let second = g.call(ADD, vec![node(x), node(b)]);
        let other = g.call(ADD, vec![node(b), node(x)]);
        g.output(&[first, second, other]);
        let (first, other) = (g.name(first), g.name(other));

        assert!(run(&mut g));

        g.assert_targets(&[ADD, ADD]);
        let outputs = output_args(&g);
        assert_eq!(outputs, vec![g.id(&first), g.id(&first), g.id(&other)]);
    }

    #[test]
    fn values_compare_with_python_equality() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2]);
        let int_scale = g.call(MUL, vec![node(x), IrArg::Int(2)]);
        let float_scale = g.call(MUL, vec![node(x), IrArg::Float(2.0)]);
        let bool_scale = g.call(MUL, vec![node(x), IrArg::Bool(true)]);
        let nan = g.call(MUL, vec![node(x), IrArg::Float(f64::NAN)]);
        let nan_again = g.call(MUL, vec![node(x), IrArg::Float(f64::NAN)]);
        g.output(&[int_scale, float_scale, bool_scale, nan, nan_again]);

        assert!(run(&mut g));

        // 2 == 2.0 fuses; 2 != True, and a NaN equals nothing.
        assert_eq!(g.count(MUL), 4);
        let outputs = output_args(&g);
        assert_eq!(outputs[0], outputs[1]);
        assert_ne!(outputs[0], outputs[2]);
        assert_ne!(outputs[3], outputs[4]);
    }

    #[test]
    fn kwargs_take_part_in_the_signature() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2]);
        let y = g.input("y", &[2]);
        let alpha_two = g.call_with_kwargs(ADD, vec![node(x), node(y)], &[("alpha", IrArg::Int(2))]);
        let alpha_float = g.call_with_kwargs(ADD, vec![node(x), node(y)], &[("alpha", IrArg::Float(2.0))]);
        let no_alpha = g.call(ADD, vec![node(x), node(y)]);
        g.output(&[alpha_two, alpha_float, no_alpha]);

        assert!(run(&mut g));

        assert_eq!(g.count(ADD), 2);
    }

    #[test]
    fn rescales_and_marked_nodes_are_left_alone() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2], "torch.int8");
        let first = g.rescale(x, "torch.int32", 0.5);
        let second = g.rescale(x, "torch.int32", 0.5);
        let tosa_spelling = g.rescale(x, "torch.int32", 0.5);
        g.graph_mut().node_mut(tosa_spelling).target = "tosa.RESCALE.default".to_string();
        let neg = g.call(NEG, vec![node(first)]);
        let marked = g.call(NEG, vec![node(first)]);
        g.graph_mut()
            .node_mut(marked)
            .meta
            .set_extra(DO_NOT_FUSE_DUPLICATE_META_KEY, MetaValue::Bool(true));
        g.output(&[second, tosa_spelling, neg, marked]);

        assert!(!run(&mut g));

        assert_eq!(g.count(RESCALE) + g.count("tosa.RESCALE.default"), 3);
        assert_eq!(g.count(NEG), 2);
    }

    /// A fusion rewires the users of the fused node, which can make them
    /// duplicates in turn: `add(neg(x), neg(x))` twice becomes one `neg` and
    /// one `add`.
    #[test]
    fn fusions_cascade_through_rewired_users() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2]);
        let neg0 = g.call(NEG, vec![node(x)]);
        let neg1 = g.call(NEG, vec![node(x)]);
        let add0 = g.call(ADD, vec![node(neg0), node(neg1)]);
        let add1 = g.call(ADD, vec![node(neg1), node(neg0)]);
        let add2 = g.call(ADD, vec![node(neg0), node(neg0)]);
        g.output(&[add0, add1, add2]);

        assert!(run(&mut g));

        g.assert_targets(&[NEG, ADD]);
        let outputs = output_args(&g);
        assert!(outputs.iter().all(|&output| output == outputs[0]));
    }

    #[test]
    fn excluded_targets_match_either_spelling() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2], "torch.int32");
        let y = g.typed_input("y", &[2], "torch.int32");
        let first = g.add_unrefreshed("tosa.ADD.default", vec![node(x), node(y)]);
        let second = g.add_unrefreshed("backend.tosa.ADD.default", vec![node(x), node(y)]);
        g.output(&[first, second]);
        let mut ctx = LayoutCtx::new(g.graph_mut());
        assert!(!fuse_duplicate_users(&mut ctx, &["backend.tosa.ADD.default"]));
        // Without the exclusion the two spellings are one target.
        assert!(fuse_duplicate_users(&mut ctx, &[]));
        ctx.finish();
        assert_eq!(g.graph().live_call_function_ids().len(), 1);
    }

    /// With rc4's exclusions, a second call on the same ctx skips producers
    /// whose users are unchanged, and still sees a duplicate made since.
    #[test]
    fn the_memo_skips_unchanged_producers_only() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2]);
        let neg = g.call(NEG, vec![node(x)]);
        let relu = g.call(RELU, vec![node(x)]);
        g.output(&[neg, relu]);
        let mut ctx = LayoutCtx::new(g.graph_mut());
        assert!(!fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS));
        assert!(!ctx.should_retry(NOTHING_TO_FUSE, x));
        assert!(!fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS));
        let meta = ctx.node(neg).meta.clone();
        let copy = ctx.call_function("neg_copy", NEG, vec![node(x)], meta);
        assert!(ctx.should_retry(NOTHING_TO_FUSE, x));
        let output = ctx.graph().output_node().map(|output| output.id).unwrap();
        ctx.replace_input_with(output, relu, copy);
        assert!(fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS));
        ctx.finish();
        g.assert_targets(&[NEG, RELU]);
    }
}
