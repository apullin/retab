//! The dtypes rc4 reads from fake tensors and from `tosa_special_dtype`,
//! derived from RETAB's IR (APU-135 P2).
//!
//! rc4 retraces after `FoldAndAnnotateQParamsPass`
//! (`backends/arm/_passes/fold_qdq_with_annotated_qparams_pass.py:464-466`),
//! so a quantized node's `meta["val"]` holds its integer dtype, and its passes
//! branch on that. RETAB's IR keeps a quantized compute node's val in
//! float32 and records the real dtype in its qparams (design F2): a quantized
//! `aten.convolution` is `torch.float32` with int8 output qparams, where rc4's
//! is `torch.int8`. Placeholders and the slices RETAB's passes create keep
//! their storage dtype (critique A11), and a layout copy takes its producer's
//! val dtype, which is float32 after a quantized compute node. Every rc4
//! decision that reads a fake dtype must therefore go through these helpers:
//!
//! | rc4 check | helper |
//! |---|---|
//! | `RewriteConvPass._is_quantized_conv` (`rewrite_conv_pass.py:405-406`), the design's "q" | [`is_quantized_conv`] |
//! | `_is_int16_activation_conv` (`:408-412`): `input_qparams[0].dtype`, else the input's fake dtype | [`activation_dtype`] |
//! | `_insert_output_conversion`'s `is_a8w8_conv`/`is_a16w8_conv` (`:830-841`): the TOSA conv's fake dtype is int32 and the input's is int8/int16 | [`conv_output_dtype`] of [`activation_dtype`] |
//! | the conv's bias test (`:926-927`): `get_first_fake_tensor(bias).dtype != torch.int32` | [`value_dtype`] of the bias |
//! | `MoveDataMovementOpsToSmallerDtypePass._uses_wider_dtype` (`move_data_movement_ops_to_smaller_dtype_pass.py:73-87`): `element_size()` of two fake values | [`element_size`] of [`value_dtype`] |
//! | `_is_per_tensor_rescale` there (`:89-98`) and in the TOSA propagation overrides (`propagate_view_copy_permute_pass.py:132-144`): the RESCALE and its input carry the same special dtype | [`special_dtype`] |
//! | `TosaPropagationOverrides._node_or_inputs_are_int48` (`propagate_view_copy_permute_pass.py:82-89`), behind `blocks_moving` (`:48-60`) | [`node_or_inputs_are_int48`] |
//! | RewriteUpsamplePass's `input_dtype` (`rewrite_upsample.py:258-263`) and RewriteAvgPool2dPass's accumulator choice (`rewrite_avg_pool2d_pass.py:66-72`): the input's fake dtype | [`operand_dtype`] at index 0 |
//! | the TOSA fake kernels' output dtypes (`backends/arm/tosa/dialect/ops/conv2d.py:17-88`, `resize.py:28-55`) | [`conv_output_dtype`], [`resize_output_dtype`] |
//!
//! The values that passes *write* follow RETAB's representation, not these
//! helpers: a created layout copy takes its producer's val dtype
//! (`layout::shape`), and a created TOSA convolution, RESIZE or RESCALE
//! records rc4's dtype in its val (design §2.7, §3.3).
//!
//! A special dtype is a separate fact in rc4 too: an INT48 accumulator's fake
//! dtype is `torch.int32` and an INT4 weight's is `torch.int8`, while TOSA's
//! element type is the special dtype, as RETAB's serializer reads it
//! (`node_output_dtype`, `serialize/visitors/dtype_quant.rs:67-83`).

use crate::fx_ir::{IrArg, IrGraph, IrNode, NodeId, NodeMeta, QParamMap};
use crate::fx_meta::MetaValue;
use crate::layout::pointwise::is_transparent_copy;
use crate::layout::shape::{is_layout_transform, tosa_op_name};

/// The meta key of `TosaSpecialDtype` (`backends/arm/tosa/mapping.py:55-63`).
pub(crate) const SPECIAL_DTYPE_KEY: &str = "tosa_special_dtype";

const SLICE_COPY: &str = "aten.slice_copy.Tensor";

/// rc4's `TosaSpecialDtype` (`backends/arm/tosa/mapping.py:36-44`): a TOSA
/// element type torch has no dtype for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpecialDtype {
    Fp4E2M1,
    Fp6E2M3,
    Fp6E3M2,
    Int48,
    Int4,
    Shape,
}

impl SpecialDtype {
    const NAMES: [(Self, &'static str); 6] = [
        (Self::Fp4E2M1, "FP4E2M1"),
        (Self::Fp6E2M3, "FP6E2M3"),
        (Self::Fp6E3M2, "FP6E3M2"),
        (Self::Int48, "INT48"),
        (Self::Int4, "INT4"),
        (Self::Shape, "SHAPE"),
    ];

    /// The member a recorded `tosa_special_dtype` names. RETAB's importer
    /// records the enum's display, `TosaSpecialDtype.INT48`
    /// (`NodeMeta::from_fx`), while its passes write the bare member name
    /// (`fx_fold_qdq_ir`, `fx_annotate_conv_bias_ir`); both are accepted.
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let name = text.strip_prefix("TosaSpecialDtype.").unwrap_or(text);
        Self::NAMES
            .iter()
            .find_map(|&(dtype, member)| (member == name).then_some(dtype))
    }

    /// The member name, RETAB's spelling, which ingest's
    /// `special_dtype_from_extras` (`ingest/from_ir.rs`) and the IR passes read.
    pub(crate) fn name(self) -> &'static str {
        Self::NAMES
            .iter()
            .find_map(|&(dtype, member)| (dtype == self).then_some(member))
            .expect("every special dtype has a name")
    }
}

/// `node.meta.get(TosaSpecialDtype.meta_key())`.
pub(crate) fn special_dtype(meta: &NodeMeta) -> Option<SpecialDtype> {
    meta.extra_str(SPECIAL_DTYPE_KEY)
        .and_then(SpecialDtype::parse)
}

/// `node.meta[TosaSpecialDtype.meta_key()] = dtype`, in RETAB's spelling.
pub(crate) fn set_special_dtype(meta: &mut NodeMeta, dtype: SpecialDtype) {
    meta.set_extra(SPECIAL_DTYPE_KEY, MetaValue::Str(dtype.name().to_string()));
}

/// `if special := source.meta.get(key): target.meta[key] = special`, as
/// RewriteConvPass does for a rewritten weight (`rewrite_conv_pass.py:377-378`)
/// and for the squeeze view and output permute after the TOSA convolution
/// (`:1204-1214`, `:1226-1229`). The recorded spelling is kept. Returns
/// whether `source` had one.
pub(crate) fn copy_special_dtype(source: &NodeMeta, target: &mut NodeMeta) -> bool {
    let Some(value) = source.extras.get(SPECIAL_DTYPE_KEY).cloned() else {
        return false;
    };
    target.set_extra(SPECIAL_DTYPE_KEY, value);
    true
}

/// Whether `meta` is marked INT48.
pub(crate) fn is_int48(meta: &NodeMeta) -> bool {
    special_dtype(meta) == Some(SpecialDtype::Int48)
}

/// `TosaPropagationOverrides._node_or_inputs_are_int48`
/// (`propagate_view_copy_permute_pass.py:82-89`): `id` or a node it reads is
/// marked INT48.
pub(crate) fn node_or_inputs_are_int48(graph: &IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    is_int48(&node.meta)
        || input_node_ids(node).into_iter().any(|input| {
            graph
                .node_opt(input)
                .is_some_and(|input| is_int48(&input.meta))
        })
}

/// `torch.Tensor.element_size()` for a dtype named as torch prints it.
pub(crate) fn element_size(dtype: &str) -> Option<usize> {
    Some(match dtype.strip_prefix("torch.")? {
        "bool" | "uint8" | "int8" | "float8_e4m3fn" | "float8_e5m2" => 1,
        "int16" | "uint16" | "float16" | "bfloat16" => 2,
        "int32" | "uint32" | "float32" => 4,
        "int64" | "uint64" | "float64" => 8,
        _ => return None,
    })
}

/// `RewriteConvPass._is_quantized_conv` (`rewrite_conv_pass.py:405-406`):
/// `bool(node.meta.get("input_qparams", {}))`.
pub(crate) fn is_quantized_conv(meta: &NodeMeta) -> bool {
    meta.input_qparams()
        .is_some_and(|qparams| !qparams.is_empty())
}

/// The dtype of input qparam `index`, which FoldAndAnnotateQParamsPass keys by
/// argument position (`fold_qdq_with_annotated_qparams_pass.py:409-412`).
pub(crate) fn input_qparam_dtype(meta: &NodeMeta, index: usize) -> Option<&str> {
    meta.input_qparam_at(index)
        .map(|qparam| qparam.dtype_str.as_str())
        .filter(|dtype| !dtype.is_empty())
}

/// The dtype rc4's fake tensor would give argument `index` of `id`: `id`'s
/// input qparam at `index` when it has one, else the argument's own
/// [`value_dtype`]. This is `_is_int16_activation_conv`'s rule
/// (`rewrite_conv_pass.py:408-412`), and the consumer's view of an operand:
/// the operand may be a layout copy or a slice whose qparams were emptied.
pub(crate) fn operand_dtype(graph: &IrGraph, id: NodeId, index: usize) -> Option<String> {
    let node = graph.node(id);
    if let Some(dtype) = input_qparam_dtype(&node.meta, index) {
        return Some(dtype.to_owned());
    }
    match node.args.get(index) {
        Some(IrArg::Node(input)) => value_dtype(graph, *input),
        _ => None,
    }
}

/// A convolution's activation dtype: `input_qparams[0].dtype`, else the fake
/// dtype of its input (`_is_int16_activation_conv`, `rewrite_conv_pass.py:408-412`).
/// RCP:830-841 reads the input's fake dtype directly; once
/// `_get_effective_conv_input_qparams` has filled index 0 (`:452-462`) the
/// two agree.
pub(crate) fn activation_dtype(graph: &IrGraph, conv: NodeId) -> Option<String> {
    operand_dtype(graph, conv, 0)
}

/// The output dtype of rc4's TOSA convolution fake kernels for an input of
/// `input` dtype (`validate_conv2d_args_dtypes`, `backends/arm/tosa/dialect/ops/conv2d.py:17-88`,
/// shared by DEPTHWISE_CONV2D, TRANSPOSE_CONV2D and CONV3D): an int8 or int16
/// input accumulates in int32, an fp8 input in float16, and any other float
/// input in its own dtype. `None` where rc4 raises for the input dtype. The
/// weight and bias dtype checks judge legality and are not ported.
pub(crate) fn conv_output_dtype(input: &str) -> Option<&'static str> {
    Some(match input {
        "torch.int8" | "torch.int16" => "torch.int32",
        "torch.float8_e4m3fn" | "torch.float8_e5m2" | "torch.float16" => "torch.float16",
        "torch.float32" => "torch.float32",
        "torch.bfloat16" => "torch.bfloat16",
        _ => return None,
    })
}

/// The output dtype of rc4's RESIZE fake kernel (`_get_output_dtype`,
/// `backends/arm/tosa/dialect/ops/resize.py:28-55`): int8 stays int8 for
/// nearest and widens to int32 for bilinear; int16 and floats keep their
/// dtype. `None` where rc4 raises.
pub(crate) fn resize_output_dtype(input: &str, bilinear: bool) -> Option<&'static str> {
    Some(match input {
        "torch.int8" if bilinear => "torch.int32",
        "torch.int8" => "torch.int8",
        "torch.int16" => "torch.int16",
        "torch.float16" => "torch.float16",
        "torch.float32" => "torch.float32",
        "torch.bfloat16" => "torch.bfloat16",
        _ => return None,
    })
}

/// The dtype rc4's fake tensor would give `id`'s value, from RETAB's
/// representation, or `None` when the value is not a tensor (a `CONST_SHAPE`'s
/// `int[]`, the tuple of `max_pool2d_with_indices`). In order:
///
/// 1. A TOSA op whose fake kernel sets the dtype records it in its val: RESCALE
///    takes its dtype argument (RETAB vals already hold it), and the passes
///    creating a convolution, RESIZE, MATMUL or TABLE write rc4's dtype into
///    the val. These ops inherit the source's output qparams, which name the
///    dtype after the RESCALE that follows, not theirs (critique A9; design F10).
/// 2. Output qparams: a quantized value has its output quantizer's dtype, as
///    FoldAndAnnotateQParamsPass's retrace gives it.
/// 3. Input qparams, for a value RETAB keeps in a float dtype: a quantized node
///    whose output carries no quantizer computes in its inputs' dtype (rc4's
///    max pool after RemoveGetItemPass has input qparams only, critique A3).
///    Integer and bool vals already hold the real dtype (a comparison's input
///    qparams describe its operands).
/// 4. A data-movement copy without qparams has its input's dtype (a permute,
///    view or slice copy, and a clone or `_to_copy` without a `dtype` kwarg):
///    RETAB's copy may hold its producer's float32 val.
/// 5. Otherwise the val dtype.
///
/// Rules 2-5 follow ingest's dtype fixup (`ingest/finalize.rs:225-415`), which
/// decides the dtypes RETAB serializes, without its RETAB-only heuristics.
pub(crate) fn value_dtype(graph: &IrGraph, id: NodeId) -> Option<String> {
    let mut current = id;
    // A copy chain cannot be longer than the graph; the bound keeps a
    // malformed (cyclic) graph from looping.
    for _ in 0..=graph.nodes.len() {
        let node = graph.node_opt(current)?;
        let facts = node.meta.tensor_facts()?;
        if records_value_dtype(&node.target) {
            return Some(facts.dtype.clone());
        }
        if let Some(dtype) = first_qparam_dtype(node.meta.output_qparams()) {
            return Some(dtype.to_owned());
        }
        if is_float_dtype(&facts.dtype) {
            if let Some(dtype) = first_qparam_dtype(node.meta.input_qparams()) {
                return Some(dtype.to_owned());
            }
        }
        match copied_input(node) {
            Some(input) => current = input,
            None => return Some(facts.dtype.clone()),
        }
    }
    None
}

/// TOSA ops whose fake kernel decides the dtype; see [`value_dtype`].
fn records_value_dtype(target: &str) -> bool {
    matches!(
        tosa_op_name(target),
        Some(
            "RESCALE"
                | "CONV2D"
                | "DEPTHWISE_CONV2D"
                | "TRANSPOSE_CONV2D"
                | "CONV3D"
                | "RESIZE"
                | "MATMUL"
                | "TABLE"
        )
    )
}

/// The input of a data-movement copy that preserves its input's dtype.
fn copied_input(node: &IrNode) -> Option<NodeId> {
    let target = node.target.as_str();
    let copies = is_layout_transform(target)
        || target == SLICE_COPY
        || (is_transparent_copy(target) && matches!(node.kwarg("dtype"), None | Some(IrArg::None)));
    if !copies {
        return None;
    }
    node.args.first().and_then(IrArg::as_node)
}

fn first_qparam_dtype(qparams: Option<&QParamMap>) -> Option<&str> {
    qparams?
        .values()
        .next()
        .map(|qparam| qparam.dtype_str.as_str())
        .filter(|dtype| !dtype.is_empty())
}

fn is_float_dtype(dtype: &str) -> bool {
    matches!(
        dtype,
        "torch.float16"
            | "torch.bfloat16"
            | "torch.float32"
            | "torch.float64"
            | "torch.float8_e4m3fn"
            | "torch.float8_e5m2"
    )
}

fn input_node_ids(node: &IrNode) -> Vec<NodeId> {
    let mut ids = Vec::new();
    for arg in node.args.iter().chain(node.kwargs.values()) {
        arg.collect_node_ids(&mut ids);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::i64_list_arg;
    use crate::fx_quant::QParam;
    use crate::layout::testing::*;
    use crate::test_ir::qparam;

    const F32: &str = "torch.float32";
    const I8: &str = "torch.int8";
    const I16: &str = "torch.int16";
    const I32: &str = "torch.int32";
    const I64: &str = "torch.int64";
    const BOOL: &str = "torch.bool";

    fn meta_of(g: &mut LayoutGraph, id: NodeId) -> &mut NodeMeta {
        &mut g.graph_mut().node_mut(id).meta
    }

    /// A node with an explicit val, as RETAB's importer or a pass leaves it.
    fn raw(
        g: &mut LayoutGraph,
        target: &str,
        args: Vec<IrArg>,
        shape: &[i64],
        dtype: &str,
    ) -> NodeId {
        let id = g.add_unrefreshed(target, args);
        meta_of(g, id).set_tensor_facts(crate::fx_meta::TensorFacts::new(shape.to_vec(), dtype));
        id
    }

    #[test]
    fn special_dtypes_parse_both_spellings_and_write_retabs() {
        for (text, dtype) in [
            ("INT48", SpecialDtype::Int48),
            ("TosaSpecialDtype.INT48", SpecialDtype::Int48),
            ("INT4", SpecialDtype::Int4),
            ("TosaSpecialDtype.INT4", SpecialDtype::Int4),
            ("SHAPE", SpecialDtype::Shape),
            ("TosaSpecialDtype.FP6E3M2", SpecialDtype::Fp6E3M2),
        ] {
            assert_eq!(SpecialDtype::parse(text), Some(dtype), "{text}");
            assert_eq!(SpecialDtype::parse(dtype.name()), Some(dtype));
        }
        for text in ["int48", "INT8", "TosaSpecialDtype.", "", "DType.INT48"] {
            assert_eq!(SpecialDtype::parse(text), None, "{text}");
        }

        let mut meta = NodeMeta::default();
        assert_eq!(special_dtype(&meta), None);
        meta.set_extra(
            SPECIAL_DTYPE_KEY,
            MetaValue::Str("TosaSpecialDtype.INT48".into()),
        );
        assert!(is_int48(&meta));
        set_special_dtype(&mut meta, SpecialDtype::Int4);
        assert_eq!(meta.extra_str(SPECIAL_DTYPE_KEY), Some("INT4"));
        assert_eq!(special_dtype(&meta), Some(SpecialDtype::Int4));
        assert!(!is_int48(&meta));
    }

    /// RCP:377-378 copies an INT4 weight's marker; an unmarked source leaves
    /// the target alone.
    #[test]
    fn copy_special_dtype_copies_the_recorded_value() {
        let mut source = NodeMeta::default();
        source.set_extra(
            SPECIAL_DTYPE_KEY,
            MetaValue::Str("TosaSpecialDtype.INT4".into()),
        );
        let mut target = NodeMeta::default();
        assert!(copy_special_dtype(&source, &mut target));
        assert_eq!(
            target.extra_str(SPECIAL_DTYPE_KEY),
            Some("TosaSpecialDtype.INT4")
        );
        assert_eq!(special_dtype(&target), Some(SpecialDtype::Int4));
        set_special_dtype(&mut target, SpecialDtype::Int48);
        assert!(!copy_special_dtype(&NodeMeta::default(), &mut target));
        assert!(is_int48(&target));
    }

    /// `torch.empty(0, dtype=d).element_size()` in torch 2.15.
    #[test]
    fn element_sizes_match_torch() {
        for (dtype, size) in [
            ("bool", 1),
            ("uint8", 1),
            ("int8", 1),
            ("int16", 2),
            ("int32", 4),
            ("int64", 8),
            ("float16", 2),
            ("bfloat16", 2),
            ("float32", 4),
            ("float64", 8),
            ("float8_e4m3fn", 1),
            ("float8_e5m2", 1),
        ] {
            assert_eq!(
                element_size(&format!("torch.{dtype}")),
                Some(size),
                "{dtype}"
            );
        }
        assert_eq!(element_size("int8"), None);
        assert_eq!(element_size("torch.int48"), None);
    }

    #[test]
    fn is_quantized_conv_needs_a_non_empty_input_map() {
        let mut meta = NodeMeta::default();
        assert!(!is_quantized_conv(&meta));
        meta.set_empty_input_qparams();
        assert!(!is_quantized_conv(&meta));
        meta.set_single_input_qparam(qparam(0.5, 0, I8));
        assert!(is_quantized_conv(&meta));
    }

    /// The TOSA fake kernels' output dtypes, checked against rc4 (the shape
    /// rule tests in `shape/tosa.rs` record the full vectors).
    #[test]
    fn tosa_output_dtypes_follow_rc4s_fake_kernels() {
        for (input, output) in [
            (I8, Some(I32)),
            (I16, Some(I32)),
            (F32, Some(F32)),
            ("torch.float16", Some("torch.float16")),
            ("torch.bfloat16", Some("torch.bfloat16")),
            ("torch.float8_e4m3fn", Some("torch.float16")),
            ("torch.float8_e5m2", Some("torch.float16")),
            (I32, None),
            (BOOL, None),
        ] {
            assert_eq!(conv_output_dtype(input), output, "{input}");
        }
        for (input, bilinear, output) in [
            (I8, false, Some(I8)),
            (I8, true, Some(I32)),
            (I16, false, Some(I16)),
            (I16, true, Some(I16)),
            (F32, true, Some(F32)),
            ("torch.bfloat16", false, Some("torch.bfloat16")),
            (I32, false, None),
        ] {
            assert_eq!(
                resize_output_dtype(input, bilinear),
                output,
                "{input} {bilinear}"
            );
        }
    }

    #[test]
    fn quantized_values_take_their_output_qparam_dtype() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 3, 8, 8], I8);
        // RETAB's quantized compute node: a float32 val, int8 qparams (design F2).
        let relu = raw(&mut g, RELU, vec![node(x)], &[1, 3, 8, 8], F32);
        meta_of(&mut g, relu).set_both_qparams(qparam(0.1, 0, I8));
        assert_eq!(value_dtype(g.graph(), relu).as_deref(), Some(I8));
        // A placeholder keeps its storage dtype.
        assert_eq!(value_dtype(g.graph(), x).as_deref(), Some(I8));
        // An int32 accumulator records int32 in its val and its qparams.
        let acc = g.typed_input("acc", &[2, 3], I32);
        let sum = g.call(SUM, vec![node(acc), ints(&[1])]);
        meta_of(&mut g, sum).set_single_output_qparam(qparam(0.01, 0, I32));
        assert_eq!(value_dtype(g.graph(), sum).as_deref(), Some(I32));
    }

    /// Critique A9 and design F10: the TOSA ops inherit the source's int8
    /// output qparams, yet their fake dtype is the one their val records.
    #[test]
    fn tosa_convolutions_resizes_and_rescales_keep_their_val_dtype() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 8, 8, 3], I8);
        let w = g.typed_input("w", &[4, 3, 3, 3], I8);
        let b = g.typed_input("b", &[4], I32);
        for target in [TOSA_CONV2D, "tosa.CONV2D.default", TOSA_RESIZE] {
            let id = raw(
                &mut g,
                target,
                vec![node(x), node(w), node(b)],
                &[1, 6, 6, 4],
                I32,
            );
            meta_of(&mut g, id).set_both_qparams(qparam(0.1, -3, I8));
            assert_eq!(value_dtype(g.graph(), id).as_deref(), Some(I32), "{target}");
        }
        let rescale = g.rescale(x, I16, 0.5);
        meta_of(&mut g, rescale).set_single_output_qparam(qparam(0.1, 0, I8));
        assert_eq!(value_dtype(g.graph(), rescale).as_deref(), Some(I16));
    }

    #[test]
    fn comparisons_and_integer_vals_ignore_their_input_qparams() {
        let mut g = LayoutGraph::new();
        let a = g.typed_input("a", &[2], I32);
        let b = g.typed_input("b", &[2], I32);
        let gt = g.call("aten.gt.Tensor", vec![node(a), node(b)]);
        meta_of(&mut g, gt)
            .set_input_qparams_from([(0, qparam(1.0, 0, I32)), (1, qparam(1.0, 0, I32))]);
        assert_eq!(value_dtype(g.graph(), gt).as_deref(), Some(BOOL));

        let x = g.typed_input("x", &[2, 3], F32);
        let argmax = raw(
            &mut g,
            "aten.argmax.default",
            vec![node(x), IrArg::Int(1)],
            &[2],
            I64,
        );
        meta_of(&mut g, argmax).set_single_input_qparam(qparam(0.1, 0, I8));
        assert_eq!(value_dtype(g.graph(), argmax).as_deref(), Some(I64));
    }

    /// Critique A3: rc4's max pool keeps the `_with_indices` node's input
    /// qparams only; its fake dtype is its input's.
    #[test]
    fn a_float_val_with_input_qparams_only_takes_their_dtype() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 8, 8]);
        let pool = raw(&mut g, TOSA_MAX_POOL2D, vec![node(x)], &[1, 4, 4, 3], F32);
        meta_of(&mut g, pool).set_single_input_qparam(qparam(0.1, 0, I8));
        meta_of(&mut g, pool).set_output_qparams(QParamMap::new());
        assert_eq!(value_dtype(g.graph(), pool).as_deref(), Some(I8));
    }

    #[test]
    fn copies_without_qparams_take_their_inputs_dtype() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 8, 8]);
        let relu = g.call(RELU, vec![node(x)]);
        meta_of(&mut g, relu).set_both_qparams(qparam(0.1, 0, I8));
        let permute = g.permute(relu, &[0, 2, 3, 1]);
        meta_of(&mut g, permute).set_empty_qparam_maps();
        let view = g.view(permute, &[1, 64, 3]);
        let slice = g.slice(view, 1, 0, 32);
        let clone = g.call(CLONE, vec![node(slice)]);
        for id in [permute, view, slice, clone] {
            assert_eq!(g.dtype(id), F32);
            assert_eq!(value_dtype(g.graph(), id).as_deref(), Some(I8));
        }
        // A cast names its own dtype.
        let cast = g.call_with_kwargs(
            TO_COPY,
            vec![node(clone)],
            &[("dtype", IrArg::Dtype(I32.into()))],
        );
        assert_eq!(value_dtype(g.graph(), cast).as_deref(), Some(I32));
        // A float graph stays float.
        let y = g.input("y", &[2, 3]);
        let y_t = g.permute(y, &[1, 0]);
        assert_eq!(value_dtype(g.graph(), y_t).as_deref(), Some(F32));
    }

    #[test]
    fn values_that_are_not_tensors_have_no_dtype() {
        let mut g = LayoutGraph::new();
        let shape = g.const_shape(&[1, 2, 3, 4]);
        assert_eq!(value_dtype(g.graph(), shape), None);
        let x = g.input("x", &[1, 3, 8, 8]);
        let tuple = g.add_unrefreshed("aten.max_pool2d_with_indices.default", vec![node(x)]);
        assert_eq!(value_dtype(g.graph(), tuple), None);
    }

    /// RCP:408-412 both ways: `input_qparams[0]` decides; without it, the
    /// input's fake dtype does.
    #[test]
    fn activation_dtype_prefers_the_conv_input_qparam() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 3, 8, 8]);
        let relu = g.call(RELU, vec![node(x)]);
        meta_of(&mut g, relu).set_both_qparams(qparam(0.1, 0, I8));
        let w = g.typed_input("w", &[4, 3, 3, 3], I8);
        let conv = raw(
            &mut g,
            CONVOLUTION,
            vec![
                node(relu),
                node(w),
                IrArg::None,
                i64_list_arg(&[1, 1]),
                i64_list_arg(&[0, 0]),
                i64_list_arg(&[1, 1]),
                IrArg::Bool(false),
                i64_list_arg(&[0, 0]),
                IrArg::Int(1),
            ],
            &[1, 4, 6, 6],
            F32,
        );
        assert_eq!(activation_dtype(g.graph(), conv).as_deref(), Some(I8));
        meta_of(&mut g, conv)
            .set_input_qparams_from([(0, qparam(0.1, 0, I16)), (1, qparam(0.2, 0, I8))]);
        assert_eq!(activation_dtype(g.graph(), conv).as_deref(), Some(I16));
        assert_eq!(operand_dtype(g.graph(), conv, 1).as_deref(), Some(I8));
        // RCP:926-927: the bias test reads the bias's own fake dtype.
        assert_eq!(operand_dtype(g.graph(), conv, 2), None);
        assert_eq!(
            conv_output_dtype(&activation_dtype(g.graph(), conv).unwrap()),
            Some(I32)
        );
    }

    #[test]
    fn a_qparam_without_a_dtype_is_ignored() {
        let mut meta = NodeMeta::default();
        meta.set_single_input_qparam(QParam::per_tensor(0.5, 0, -128, 127, String::new()));
        assert_eq!(input_qparam_dtype(&meta, 0), None);
    }

    #[test]
    fn int48_blocks_through_the_node_or_any_input() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 4, 4, 2], I32);
        let rescale = g.rescale(x, I16, 0.5);
        let permute = g.permute(rescale, &[0, 3, 1, 2]);
        assert!(!node_or_inputs_are_int48(g.graph(), permute));
        set_special_dtype(meta_of(&mut g, x), SpecialDtype::Int48);
        assert!(node_or_inputs_are_int48(g.graph(), rescale));
        assert!(!node_or_inputs_are_int48(g.graph(), permute));
        set_special_dtype(meta_of(&mut g, permute), SpecialDtype::Int48);
        assert!(node_or_inputs_are_int48(g.graph(), permute));
    }
}
