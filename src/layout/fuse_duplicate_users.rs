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
//! (`backends/transforms/propagate_view_copy_permute_pass.py:569-576`), which
//! passes [`TableStage::OwnTableConstants`]. At the pass's own slots,
//! `arm_pass_manager.py:587` and `:765`, RETAB runs its own
//! `fx_fuse_duplicate_users_ir`; the standalone port of those two,
//! `IrFuseDuplicateUsersPass::EARLY` and `::LATE`, and the stage only `LATE`
//! uses, `TableStage::SharedTableConstants`, are for rc4's ported tests and
//! exist only in test builds.
//!
//! **TABLE ops.** RETAB lowers a quantized table op to TABLE only when it
//! serializes (`layout::pending`), while rc4 has replaced it by the time two
//! of these calls run, so how such a pending table compares depends on where
//! the call sits ([`TableStage`]):
//!
//! - At `:587`, before InsertTableOpsPass (`arm_pass_manager.py:708`), rc4
//!   holds the aten op too and fuses two of them on their arguments alone,
//!   whatever their qparams.
//! - In the propagation driver at phase 6 (`:725-727`), each TABLE reads its
//!   own `b_<name>_table_constant` placeholder (insert_table_ops.py:392-399),
//!   so no two TABLEs are duplicates: a pending table has no signature.
//! - At `:765`, FuseEqualPlaceholdersPass (`:755`) has merged constants with
//!   equal bytes, dtype and shape (fuse_equal_placeholders_pass.py:61-72), so
//!   two TABLEs of one input are duplicates when their tables are equal byte
//!   for byte, whatever op or qparams built them (two sigmoids whose output
//!   scales differ keep apart; a ceil and a floor of integer-valued codes
//!   fuse). The port compares the tables `fx_insert_table_ops_ir::planned_table`
//!   builds, which are the ones RETAB serializes, grouped under rc4's TABLE
//!   target. A 16-bit table is followed by a RESCALE unless its left shift is
//!   0 (insert_table_ops.py:410-417); rc4 fuses such TABLEs and keeps one
//!   RESCALE per original, which a pending table cannot express, so the port
//!   keeps both pending tables: one TABLE more than rc4, and nothing fused
//!   downstream that rc4 keeps apart. A table RETAB can't plan never fuses.
//!   rc4's TABLE node never carries `do_not_fuse_duplicate`, so a pending
//!   table's marker doesn't count there.
//!
//! Differences from rc4, none visible in the graph:
//!
//! - rc4 ends with an `ExportPass` retrace when it fused anything (`:147-151`),
//!   which renames nodes and re-derives every value. A fusion changes no value,
//!   so the port keeps names and meta.
//! - rc4 groups a producer's users in `node.users` insertion order; the port
//!   groups them in graph order, though the ctx tracks that order
//!   (`LayoutCtx::users_in_fx_order`). Group order cannot change the result:
//!   signatures include the target, so groups never share a representative,
//!   and the representative is always the earliest user with the signature in
//!   graph order.
//! - Python compares a NaN argument equal only to the same float object;
//!   [`PyValue`] has no value for one, so such a user is never fused (rc4 fuses
//!   two users that share one NaN object, as with `math.nan`).
//!
//! Complexity: every node is visited once, and again only when a fusion
//! touches it (as the producer or the representative), so a call costs
//! O(N + sum of user-list sizes x signature size) plus the fusions; a pending
//! table's signature at `:765` is its table, 256 or 513 entries. Across calls
//! on one ctx with rc4's exclusions, the attempt memo skips producers whose
//! users haven't changed since a call at the same stage found nothing to fuse
//! there.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};

#[cfg(test)]
use crate::fx_insert_table_ops_ir::planned_table;
#[cfg(test)]
use crate::fx_ir::{IrGraph, IrPass};
use crate::fx_ir::{IrNode, NodeId, NodeOp};
#[cfg(test)]
use crate::graph::ir::DType;
use crate::layout::ctx::{LayoutCtx, RuleId};
use crate::layout::fx_node::{rc4_target, PyValue};
use crate::layout::pending::is_pending_table;

/// `DO_NOT_FUSE_DUPLICATE_META_KEY` (`:17`): a node whose meta sets it is
/// never fused. RETAB's importer keeps it (`NodeMeta::from_fx`).
pub(crate) const DO_NOT_FUSE_DUPLICATE_META_KEY: &str = "do_not_fuse_duplicate";

/// `TOSA_EXCLUDED_TARGETS` (backends/arm/_passes/fuse_duplicate_users_pass.py:18),
/// the exclusions of every rc4 caller: the Arm pass (`:26-27`) and the TOSA
/// propagation overrides (`duplicate_user_fusion_exclusions`,
/// backends/arm/_passes/propagate_view_copy_permute_pass.py:37-38).
pub(crate) const TOSA_EXCLUDED_TARGETS: &[&str] = &["backend.tosa.RESCALE.default"];

/// rc4's target for the TABLE node a pending table has become.
const TABLE: &str = "backend.tosa.TABLE.default";

/// Where a FuseDuplicateUsers call runs relative to rc4's TABLE lowering,
/// which decides how a pending table compares (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TableStage {
    /// Before InsertTableOpsPass (`arm_pass_manager.py:708`): the pass at
    /// `:587`. A table op compares like any other node.
    BeforeTables,
    /// After InsertTableOpsPass and before FuseEqualPlaceholdersPass
    /// (`:755`): the propagation driver at phase 6. Every TABLE reads its own
    /// table constant, so a pending table is never a duplicate.
    OwnTableConstants,
    /// After FuseEqualPlaceholdersPass: the pass at `:765`. Pending tables of
    /// one input are duplicates when their tables are equal.
    #[cfg(test)]
    SharedTableConstants,
}

impl TableStage {
    /// The memo rule for a producer whose users had nothing to fuse; used
    /// only with [`TOSA_EXCLUDED_TARGETS`], so one ctx never mixes exclusion
    /// sets in it, and per stage, as the stages compare pending tables
    /// differently.
    fn nothing_to_fuse(self) -> RuleId {
        match self {
            Self::BeforeTables => RuleId("fuse_duplicate_users.before_tables"),
            Self::OwnTableConstants => RuleId("fuse_duplicate_users.own_table_constants"),
            #[cfg(test)]
            Self::SharedTableConstants => RuleId("fuse_duplicate_users.shared_table_constants"),
        }
    }

    /// Whether rc4 holds `node` as a TABLE at this stage.
    fn holds_as_table(self, node: &IrNode) -> bool {
        self != Self::BeforeTables && is_pending_table(node)
    }
}

/// The Arm `FuseDuplicateUsersPass()` at one of its two pipeline positions,
/// for rc4's ported tests.
#[cfg(test)]
pub(crate) struct IrFuseDuplicateUsersPass {
    stage: TableStage,
}

#[cfg(test)]
impl IrFuseDuplicateUsersPass {
    /// `arm_pass_manager.py:587`, before InsertTableOpsPass.
    pub(crate) const EARLY: Self = Self {
        stage: TableStage::BeforeTables,
    };
    /// `arm_pass_manager.py:765`, after FuseEqualPlaceholdersPass.
    pub(crate) const LATE: Self = Self {
        stage: TableStage::SharedTableConstants,
    };
}

#[cfg(test)]
impl IrPass for IrFuseDuplicateUsersPass {
    fn name(&self) -> &str {
        // `IrFuseDuplicateUsers` names RETAB's own pass (`fx_fuse_duplicate_users_ir`).
        match self.stage {
            TableStage::BeforeTables => "IrLayoutFuseDuplicateUsers",
            TableStage::OwnTableConstants => "IrLayoutFuseDuplicateUsersOwnTables",
            TableStage::SharedTableConstants => "IrLayoutFuseDuplicateUsersLate",
        }
    }

    fn run(&self, graph: &mut IrGraph) -> bool {
        let mut ctx = LayoutCtx::new(graph);
        let modified = fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS, self.stage);
        if modified {
            ctx.finish();
        }
        modified
    }
}

/// `FuseDuplicateUsersPass(excluded_targets).call(graph_module)` (`:115-153`)
/// inside `ctx`, at `stage`; returns rc4's `modified`. Targets match in
/// either of RETAB's spellings ([`rc4_target`]).
pub(crate) fn fuse_duplicate_users(ctx: &mut LayoutCtx<'_>, excluded_targets: &[&str], stage: TableStage) -> bool {
    let excluded: Vec<Cow<'_, str>> = excluded_targets.iter().map(|target| rc4_target(target)).collect();
    let memo = (excluded_targets == TOSA_EXCLUDED_TARGETS).then(|| stage.nothing_to_fuse());
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
        if ctx.is_erased(producer) || memo.is_some_and(|rule| !ctx.should_retry(rule, producer)) {
            continue;
        }
        let mut users = ctx.graph().live_user_ids(producer);
        ctx.order().sort(&mut users);
        let fused = users.len() >= 2 && {
            let groups = candidate_groups(ctx, &users, &excluded, stage);
            fuse_candidate_groups(ctx, producer, &groups, stage, &mut |id| {
                if !std::mem::replace(&mut queued[id], true) {
                    producers.push_back(id);
                }
            })
        };
        if fused {
            modified = true;
        } else if let Some(rule) = memo {
            ctx.record_attempt(rule, producer, users);
        }
    }
    modified
}

/// rc4's target for `node` at `stage`: TABLE for a pending table rc4 has
/// replaced, else the target in rc4's spelling.
fn stage_target(node: &IrNode, stage: TableStage) -> Cow<'_, str> {
    if stage.holds_as_table(node) {
        Cow::Borrowed(TABLE)
    } else {
        rc4_target(&node.target)
    }
}

/// `_get_candidate_groups` (`:189-213`): call users not excluded, grouped by
/// target, groups of two or more, each in graph order.
fn candidate_groups(
    ctx: &LayoutCtx<'_>,
    users: &[NodeId],
    excluded: &[Cow<'_, str>],
    stage: TableStage,
) -> Vec<Vec<NodeId>> {
    let mut groups: Vec<Vec<NodeId>> = Vec::new();
    let mut by_target: HashMap<Cow<'_, str>, usize> = HashMap::new();
    for &user in users {
        let node = ctx.node(user);
        if node.op != NodeOp::CallFunction {
            continue;
        }
        let target = stage_target(node, stage);
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
    stage: TableStage,
    enqueue_producer: &mut dyn FnMut(NodeId),
) -> bool {
    let mut modified = false;
    let mut signature_to_user: HashMap<Signature, NodeId> = HashMap::new();
    for group in groups {
        for &user in group {
            let Some(signature) = user_signature(ctx.node(user), stage) else {
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

/// What `build_node_signature(node)` (`:69-83`) compares.
#[derive(Debug, PartialEq, Eq, Hash)]
enum Signature {
    /// The node's own signature, without the op, which is always
    /// `call_function` here.
    Call {
        target: String,
        args: Vec<PyValue>,
        /// Sorted by key, as `_to_hashable` sorts dict items (`:29-31`).
        kwargs: Vec<(String, PyValue)>,
    },
    /// A pending table at `:765`: rc4's `TABLE(input, table_constant)`, with
    /// the constant identified by what FuseEqualPlaceholdersPass merges on.
    #[cfg(test)]
    Table {
        input: NodeId,
        dtype: DType,
        shape: Vec<i64>,
        data: Vec<u8>,
    },
}

/// `_build_user_signature` (`:215-218`) for the node rc4 holds at `stage`:
/// none for a node marked [`DO_NOT_FUSE_DUPLICATE_META_KEY`] or holding a
/// NaN, or for a pending table rc4 never fuses (see the module docs).
fn user_signature(node: &IrNode, stage: TableStage) -> Option<Signature> {
    if stage.holds_as_table(node) {
        return match stage {
            #[cfg(test)]
            TableStage::SharedTableConstants => table_signature(node),
            TableStage::BeforeTables | TableStage::OwnTableConstants => None,
        };
    }
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
    Some(Signature::Call {
        target: rc4_target(&node.target).into_owned(),
        args,
        kwargs,
    })
}

/// A pending table's signature at `:765`: InsertTableOpsPass reads `args[0]`
/// (insert_table_ops.py:403-406) and FuseEqualPlaceholdersPass keys a
/// constant on its dtype, shape and bytes. None for a table followed by a
/// RESCALE, or one RETAB can't plan.
#[cfg(test)]
fn table_signature(node: &IrNode) -> Option<Signature> {
    let input = node.args.first()?.as_node()?;
    let table = planned_table(node)?;
    if table.rescale_lshift != 0 {
        return None;
    }
    Some(Signature::Table {
        input,
        dtype: table.table_dtype,
        shape: table.shape,
        data: table.data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::{IrArg, QParam};
    use crate::fx_meta::MetaValue;
    use crate::layout::testing::*;
    use crate::test_ir::qparam;

    fn run(g: &mut LayoutGraph) -> bool {
        IrFuseDuplicateUsersPass::EARLY.run(g.graph_mut())
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
        assert!(!fuse_duplicate_users(&mut ctx, &["backend.tosa.ADD.default"], TableStage::BeforeTables));
        // Without the exclusion the two spellings are one target.
        assert!(fuse_duplicate_users(&mut ctx, &[], TableStage::BeforeTables));
        ctx.finish();
        assert_eq!(g.graph().live_call_function_ids().len(), 1);
    }

    /// With rc4's exclusions, a second call on the same ctx skips producers
    /// whose users are unchanged, and still sees a duplicate made since.
    #[test]
    fn the_memo_skips_unchanged_producers_only() {
        const STAGE: TableStage = TableStage::OwnTableConstants;
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2]);
        let neg = g.call(NEG, vec![node(x)]);
        let relu = g.call(RELU, vec![node(x)]);
        g.output(&[neg, relu]);
        let mut ctx = LayoutCtx::new(g.graph_mut());
        assert!(!fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS, STAGE));
        assert!(!ctx.should_retry(STAGE.nothing_to_fuse(), x));
        // Another stage compares pending tables differently: its own memo.
        assert!(ctx.should_retry(TableStage::SharedTableConstants.nothing_to_fuse(), x));
        assert!(!fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS, STAGE));
        let meta = ctx.node(neg).meta.clone();
        let copy = ctx.call_function("neg_copy", NEG, vec![node(x)], meta);
        assert!(ctx.should_retry(STAGE.nothing_to_fuse(), x));
        let output = ctx.graph().output_node().map(|output| output.id).unwrap();
        ctx.replace_input_with(output, relu, copy);
        assert!(fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS, STAGE));
        ctx.finish();
        g.assert_targets(&[NEG, RELU]);
    }

    // Pending TABLE ops at each stage. The rc4 results come from the venv
    // (`.venv-et16`): the pass alone for `:587`, InsertTableOpsPass then the
    // pass for the driver, and InsertTableOpsPass, FuseEqualPlaceholdersPass
    // then the pass for `:765`; the replay fixtures
    // fuse_duplicate_users_{driver,late}.txt cover them too.

    const INT16: &str = "torch.int16";

    fn int8(scale: f64) -> QParam {
        qparam(scale, 0, "torch.int8")
    }

    fn int16(scale: f64) -> QParam {
        QParam::per_tensor(scale, 0, -32768, 32767, INT16.to_string())
    }

    /// A quantized table op as RETAB holds it where rc4 may already have a
    /// TABLE (`layout::pending::is_pending_table`).
    fn pending(g: &mut LayoutGraph, target: &str, x: NodeId, input: QParam, output: QParam) -> NodeId {
        let table = g.call(target, vec![node(x)]);
        g.graph_mut()
            .node_mut(table)
            .meta
            .set_unary_input_output_qparams(input, output);
        table
    }

    fn run_at(g: &mut LayoutGraph, stage: TableStage) -> bool {
        let mut ctx = LayoutCtx::new(g.graph_mut());
        let modified = fuse_duplicate_users(&mut ctx, TOSA_EXCLUDED_TARGETS, stage);
        if modified {
            ctx.finish();
        }
        modified
    }

    /// Two sigmoids of one input whose output scales differ: rc4 fuses them at
    /// `:587` (modified=True, one sigmoid) and keeps both TABLEs in the driver
    /// and at `:765` (modified=False).
    #[test]
    fn different_tables_fuse_only_before_insert_table_ops() {
        for (stage, fuses) in [
            (TableStage::BeforeTables, true),
            (TableStage::OwnTableConstants, false),
            (TableStage::SharedTableConstants, false),
        ] {
            let mut g = LayoutGraph::new();
            let x = g.typed_input("x", &[2, 3], "torch.int8");
            let first = pending(&mut g, SIGMOID, x, int8(0.5), int8(0.5));
            let second = pending(&mut g, SIGMOID, x, int8(0.5), int8(0.25));
            g.output(&[first, second]);
            assert_eq!(run_at(&mut g, stage), fuses, "{stage:?}");
            assert_eq!(g.count(SIGMOID), if fuses { 1 } else { 2 }, "{stage:?}");
        }
    }

    /// Equal qparams: one TABLE at `:765`, two in the driver, where each
    /// TABLE reads its own constant. The users of one pending table still
    /// fuse there.
    #[test]
    fn equal_tables_fuse_once_their_constants_are_merged() {
        for (stage, fuses) in [
            (TableStage::BeforeTables, true),
            (TableStage::OwnTableConstants, false),
            (TableStage::SharedTableConstants, true),
        ] {
            let mut g = LayoutGraph::new();
            let x = g.typed_input("x", &[2, 3], "torch.int8");
            let first = pending(&mut g, SIGMOID, x, int8(0.5), int8(0.5));
            let second = pending(&mut g, SIGMOID, x, int8(0.5), int8(0.5));
            let neg0 = g.call(NEG, vec![node(first)]);
            let neg1 = g.call(NEG, vec![node(first)]);
            g.output(&[neg0, neg1, second]);
            assert!(run_at(&mut g, stage), "{stage:?}");
            assert_eq!(g.count(SIGMOID), if fuses { 1 } else { 2 }, "{stage:?}");
            assert_eq!(g.count(NEG), 1, "{stage:?}");
        }
    }

    /// rc4 compares table bytes at `:765`, not ops or qparams: a ceil and a
    /// floor of codes that dequantize to integers build one table, and so do
    /// sigmoids whose input ranges differ only where the output saturates.
    /// The earliest survives, as its TABLE does in rc4.
    #[test]
    fn coinciding_tables_fuse_at_765() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3], "torch.int8");
        let ceil = pending(&mut g, "aten.ceil.default", x, int8(1.0), int8(1.0));
        let floor = pending(&mut g, "aten.floor.default", x, int8(1.0), int8(1.0));
        let wide = pending(&mut g, SIGMOID, x, int8(0.5), int8(1.0));
        let narrow = pending(&mut g, SIGMOID, x, QParam::per_tensor(0.5, 0, -127, 127, "torch.int8".into()), int8(1.0));
        g.output(&[ceil, floor, wide, narrow]);
        let (ceil, wide) = (g.name(ceil), g.name(wide));

        assert!(run_at(&mut g, TableStage::SharedTableConstants));

        g.assert_targets(&["aten.ceil.default", SIGMOID]);
        let outputs = output_args(&g);
        assert_eq!(outputs, vec![g.id(&ceil), g.id(&ceil), g.id(&wide), g.id(&wide)]);
    }

    /// rc4 follows a 16-bit table with a RESCALE (a left shift of -7 for an
    /// int16 output) and keeps one RESCALE per original when it fuses the
    /// TABLEs at `:765`: TABLE, RESCALE, RESCALE. A pending table can't say
    /// that, so the port keeps both, one TABLE more than rc4.
    #[test]
    fn sixteen_bit_tables_keep_apart_at_765() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3], INT16);
        let first = pending(&mut g, SIGMOID, x, int16(1.0 / 1024.0), int16(1.0 / 32768.0));
        let second = pending(&mut g, SIGMOID, x, int16(1.0 / 1024.0), int16(1.0 / 32768.0));
        g.output(&[first, second]);
        let table = crate::fx_insert_table_ops_ir::planned_table(g.graph().node(first)).expect("a planned table");
        assert_eq!(table.rescale_lshift, -7);

        assert!(!run_at(&mut g, TableStage::SharedTableConstants));
        assert_eq!(g.count(SIGMOID), 2);
    }

    /// rc4's TABLE node is new (insert_table_ops.py:403-406) and carries no
    /// `do_not_fuse_duplicate`, so a pending table's marker doesn't count
    /// once tables exist; a quantized table op and a float one never share a
    /// group (rc4 has a TABLE and the aten op).
    #[test]
    fn pending_tables_compare_as_tables_at_765() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3], "torch.int8");
        let marked = pending(&mut g, SIGMOID, x, int8(0.5), int8(0.5));
        g.graph_mut()
            .node_mut(marked)
            .meta
            .set_extra(DO_NOT_FUSE_DUPLICATE_META_KEY, MetaValue::Bool(true));
        let plain = pending(&mut g, SIGMOID, x, int8(0.5), int8(0.5));
        let float = g.call(SIGMOID, vec![node(x)]);
        g.output(&[marked, plain, float]);
        let (marked, float) = (g.name(marked), g.name(float));

        assert!(IrFuseDuplicateUsersPass::LATE.run(g.graph_mut()));

        let outputs = output_args(&g);
        assert_eq!(outputs, vec![g.id(&marked), g.id(&marked), g.id(&float)]);
    }
}
