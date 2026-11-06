//! Meta for the nodes the ported rc4 layout rewrites create (APU-135 P2).
//!
//! rc4 builds a created node's meta in one of two ways, and the P2 rewrites
//! use both (design §2.7, §3; critique A1, A3, A9):
//!
//! - [`from_node`] and [`inheriting_qparams`] are `create_node(graph, target,
//!   args, from_node=source, inherit_qparams=...)`
//!   (`backends/arm/_passes/arm_pass_utils.py:148-192`). It copies every key of
//!   the source's meta and, unless `inherit_qparams`, replaces the qparam maps
//!   the source has with empty ones; a map the source lacks stays absent. The
//!   caller then assigns `meta["val"]`, or the pass's closing retrace does.
//!   - RewriteConvPass builds every node this way. Only the TOSA convolution
//!     inherits the convolution's qparams, output qparams included
//!     (`rewrite_conv_pass.py:1175-1182`), and so does a U55 clone of it
//!     (`:652-663`).
//!   - RewriteUpsamplePass builds its `CONST_SHAPE`s, permutes and RESCALE
//!     from the upsample with empty maps; only the RESIZE inherits
//!     (`rewrite_upsample.py:205-297`).
//!   - SizeAdjustInputPass builds its slices from the convolution or pool
//!     they feed, so they have empty maps (`size_adjust_input_pass.py:396-401`).
//!   - NormalizeDelegateIOLayoutPass builds its permutes from the boundary
//!     node (`normalize_delegate_io_layout_pass.py`).
//! - [`call_operator`] is `ArmPass.call_operator(op, args, kwargs, meta,
//!   updated=True)` (`arm_pass.py:88-111`). It copies every key of the meta it
//!   is given, qparams included, and `ExportPass._fx` sets the value to the
//!   fake kernel's result. RewriteAvgPool2dPass and RewriteMaxPool2dPass build
//!   their permutes and TOSA pools this way from the pool's meta
//!   (`rewrite_avg_pool2d_pass.py:74-106`, `rewrite_max_pool2d_pass.py:118-170`),
//!   so all of them keep the pool's qparams. rc4's max pool has input qparams only:
//!   RemoveGetItemPass made it from the `max_pool2d_with_indices` node, whose
//!   output qparams sit on the getitem (critique A3).
//! - [`shape_node`] is `create_shape_node` (`arm_pass_utils.py:195-216`): a
//!   [`from_node`] copy marked `TosaSpecialDtype.SHAPE`, whose value is an
//!   `int[]` rather than a tensor.
//!
//! RETAB has no retrace, so each helper takes the created node's final value
//! (design §2.1), and [`assign_val`] is `node.meta["val"] = value` on a node
//! that exists. A value fresh from a fake kernel or
//! `permute_fake_tensor_metadata` (`arm_pass_utils.py:820-830`) is
//! contiguous, so its dim order is the identity ([`mark_contiguous`]). RETAB's
//! other layout facts are copied like any other key; explicit mode never
//! writes them.
//!
//! RETAB's importer drops an empty qparam map (`NodeMeta::from_fx`), so a map
//! rc4 holds empty is absent on an imported node. Maps these helpers empty
//! stay present, as rc4's do; the replay harness compares the difference.

use crate::fx_ir::{NodeMeta, QParamMap};
use crate::fx_meta::TensorFacts;
use crate::layout::dtype::{set_special_dtype, SpecialDtype};
use crate::layout::shape::mark_contiguous;

/// `create_node(..., from_node=source)`: `source`'s meta with the qparam maps
/// it has emptied, holding `val`.
pub(crate) fn from_node(source: &NodeMeta, val: TensorFacts) -> NodeMeta {
    let mut meta = source.clone();
    empty_qparam_maps(&mut meta);
    assign_val(&mut meta, val);
    meta
}

/// `create_node(..., from_node=source, inherit_qparams=True)`: `source`'s
/// meta, qparams included, holding `val`.
pub(crate) fn inheriting_qparams(source: &NodeMeta, val: TensorFacts) -> NodeMeta {
    let mut meta = source.clone();
    assign_val(&mut meta, val);
    meta
}

/// `ArmPass.call_operator(..., meta=source, updated=True)`: every key of
/// `source`, qparams included, with the kernel's result `val`. The same
/// operation as [`inheriting_qparams`]; the two names keep each port's rc4
/// call recognisable.
pub(crate) fn call_operator(source: &NodeMeta, val: TensorFacts) -> NodeMeta {
    inheriting_qparams(source, val)
}

/// `create_shape_node(..., from_node=source)`: `source`'s meta with the qparam
/// maps it has emptied, marked SHAPE, and no tensor value, as the
/// `CONST_SHAPE` kernel returns its `int[]` (`backends/arm/tosa/dialect/ops/shape_ops.py:270-275`).
pub(crate) fn shape_node(source: &NodeMeta) -> NodeMeta {
    let mut meta = source.clone();
    empty_qparam_maps(&mut meta);
    meta.clear_val();
    set_special_dtype(&mut meta, SpecialDtype::Shape);
    meta
}

/// `node.meta["val"] = val` on a node that exists: nothing else changes, and
/// the new value is contiguous.
pub(crate) fn assign_val(meta: &mut NodeMeta, val: TensorFacts) {
    meta.set_tensor_facts(val);
    mark_contiguous(meta);
}

/// `create_node`'s `new_meta["input_qparams"] = {}` and
/// `new_meta["output_qparams"] = {}`, each only if the key is present
/// (`arm_pass_utils.py:178-182`).
fn empty_qparam_maps(meta: &mut NodeMeta) {
    if meta.input_qparams().is_some() {
        meta.set_empty_input_qparams();
    }
    if meta.output_qparams().is_some() {
        meta.set_output_qparams(QParamMap::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_meta::MetaValue;
    use crate::layout::dtype::{special_dtype, SPECIAL_DTYPE_KEY};
    use crate::test_ir::qparam;

    const I8: &str = "torch.int8";
    const I32: &str = "torch.int32";

    /// A quantized convolution's meta as FoldAndAnnotateQParamsPass leaves it:
    /// input qparams for x, the weight and the bias, int8 output qparams.
    fn conv_meta() -> NodeMeta {
        let mut meta = NodeMeta::tensor(vec![1, 4, 3, 4], "torch.float32");
        meta.set_val_dim_order(vec![0, 2, 3, 1]);
        meta.set_input_qparams_from([
            (0, qparam(0.1, -3, I8)),
            (1, qparam(0.02, 0, I8)),
            (2, qparam(0.002, 0, I32)),
        ]);
        meta.set_single_output_qparam(qparam(0.05, 4, I8));
        meta.set_extra("arm_quantized_annotation", MetaValue::Bool(true));
        meta
    }

    fn keys(map: Option<&QParamMap>) -> Option<Vec<usize>> {
        map.map(|map| map.iter().map(|(&index, _)| index).collect())
    }

    #[test]
    fn from_node_empties_the_qparam_maps_the_source_has() {
        let source = conv_meta();
        let meta = from_node(&source, TensorFacts::new(vec![1, 3, 8, 10], I8));
        // rc4's slice after SizeAdjustInputPass: `iq={} oq={}`.
        assert_eq!(keys(meta.input_qparams()), Some(vec![]));
        assert_eq!(keys(meta.output_qparams()), Some(vec![]));
        assert_eq!(meta.val_shape(), Some(&[1, 3, 8, 10][..]));
        assert_eq!(meta.val_dtype(), Some(I8));
        assert_eq!(meta.val_dim_order(), Some(&[0, 1, 2, 3][..]));
        assert_eq!(meta.extra_bool("arm_quantized_annotation"), Some(true));
        // The source is untouched.
        assert_eq!(keys(source.input_qparams()), Some(vec![0, 1, 2]));
        assert_eq!(source.val_dim_order(), Some(&[0, 2, 3, 1][..]));

        // A map the source lacks stays absent.
        let mut float = NodeMeta::tensor(vec![2, 3], "torch.float32");
        float.set_single_output_qparam(qparam(0.5, 0, I8));
        let meta = from_node(&float, TensorFacts::new(vec![3, 2], "torch.float32"));
        assert!(meta.input_qparams().is_none());
        assert_eq!(keys(meta.output_qparams()), Some(vec![]));
        let plain = from_node(&NodeMeta::default(), TensorFacts::new(vec![4], I8));
        assert!(!plain.has_qparam_metadata());
    }

    /// Critique A9: the TOSA convolution keeps the convolution's int8 output
    /// qparams along with its input qparams.
    #[test]
    fn inheriting_qparams_keeps_every_map() {
        let source = conv_meta();
        let meta = inheriting_qparams(&source, TensorFacts::new(vec![1, 3, 4, 4], I32));
        assert_eq!(keys(meta.input_qparams()), Some(vec![0, 1, 2]));
        assert_eq!(meta.output_qparam(), source.output_qparam());
        assert_eq!(meta.input_qparam_at(2), source.input_qparam_at(2));
        assert_eq!(
            (meta.val_shape(), meta.val_dtype()),
            (Some(&[1, 3, 4, 4][..]), Some(I32))
        );
        assert_eq!(meta.val_dim_order(), Some(&[0, 1, 2, 3][..]));
    }

    /// The pools' permutes and TOSA ops keep the pool's maps; rc4's max pool
    /// has input qparams only (critique A3), and so does everything built from it.
    #[test]
    fn call_operator_copies_the_pool_meta_qparams_included() {
        let mut pool = NodeMeta::tensor(vec![1, 3, 3, 3], "torch.float32");
        pool.set_single_input_qparam(qparam(0.1, 0, I8));
        let meta = call_operator(&pool, TensorFacts::new(vec![1, 3, 3, 3], "torch.float32"));
        assert_eq!(keys(meta.input_qparams()), Some(vec![0]));
        assert!(meta.output_qparams().is_none());
        assert_eq!(meta.input_qparam(), pool.input_qparam());
    }

    /// The source's special dtype and do-not-fuse flag come along, as rc4's
    /// U55 clone of an INT48 convolution needs (`rewrite_conv_pass.py:651-662`).
    #[test]
    fn extras_are_copied_like_any_other_key() {
        let mut source = conv_meta();
        set_special_dtype(&mut source, SpecialDtype::Int48);
        source.set_extra("do_not_fuse_duplicate", MetaValue::Bool(true));
        for meta in [
            from_node(&source, TensorFacts::new(vec![1], I32)),
            inheriting_qparams(&source, TensorFacts::new(vec![1], I32)),
        ] {
            assert_eq!(special_dtype(&meta), Some(SpecialDtype::Int48));
            assert_eq!(meta.extra_bool("do_not_fuse_duplicate"), Some(true));
        }
    }

    /// rc4's upsample `CONST_SHAPE`s: empty maps, SHAPE, and an `int[]` value.
    #[test]
    fn shape_nodes_hold_no_tensor_value() {
        let mut upsample = NodeMeta::tensor(vec![1, 3, 8, 8], "torch.float32");
        upsample.set_val_dim_order(vec![0, 1, 2, 3]);
        upsample.set_both_qparams(qparam(0.1, 0, I8));
        let meta = shape_node(&upsample);
        assert!(meta.tensor_facts().is_none());
        assert!(meta.val_dim_order().is_none());
        assert_eq!(keys(meta.input_qparams()), Some(vec![]));
        assert_eq!(keys(meta.output_qparams()), Some(vec![]));
        assert_eq!(meta.extra_str(SPECIAL_DTYPE_KEY), Some("SHAPE"));
        assert_eq!(special_dtype(&meta), Some(SpecialDtype::Shape));
    }

    /// Each rewrite family's qparam rule, built from one source: RewriteConv's
    /// permutes and RESCALE empty the maps while its TOSA op inherits them,
    /// output qparams included (critique A9); RewriteUpsample's nodes all
    /// empty them except the RESIZE (A1); the pool rewrites copy them (A3).
    #[test]
    fn rewrite_families_follow_rc4s_qparam_rules() {
        let source = conv_meta();
        let facts = || TensorFacts::new(vec![1, 4, 4, 3], I8);
        let maps = |meta: &NodeMeta| (keys(meta.input_qparams()), keys(meta.output_qparams()));
        let emptied = (Some(vec![]), Some(vec![]));
        let kept = (Some(vec![0, 1, 2]), Some(vec![0]));

        // RewriteConvPass: input permute, TOSA op, RESCALE, output permute.
        assert_eq!(maps(&from_node(&source, facts())), emptied);
        assert_eq!(maps(&inheriting_qparams(&source, facts())), kept);
        // RewriteUpsamplePass: CONST_SHAPEs, permutes and RESCALE; RESIZE.
        assert_eq!(maps(&shape_node(&source)), emptied);
        assert_eq!(maps(&from_node(&source, facts())), emptied);
        assert_eq!(maps(&inheriting_qparams(&source, facts())), kept);
        // RewriteAvgPool2dPass and RewriteMaxPool2dPass: every node.
        assert_eq!(maps(&call_operator(&source, facts())), kept);
    }

    #[test]
    fn assign_val_changes_only_the_value() {
        let mut meta = conv_meta();
        assign_val(&mut meta, TensorFacts::new(vec![1, 3, 4, 4, 1], I32));
        assert_eq!(meta.val_shape(), Some(&[1, 3, 4, 4, 1][..]));
        assert_eq!(meta.val_dim_order(), Some(&[0, 1, 2, 3, 4][..]));
        assert_eq!(keys(meta.input_qparams()), Some(vec![0, 1, 2]));
        assert_eq!(meta.extra_bool("arm_quantized_annotation"), Some(true));
    }
}
