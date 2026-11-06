//! torch.fx node facts that the ported rc4 passes read, on RETAB's IR (APU-160).
//!
//! - [`all_input_nodes`]: `Node.all_input_nodes` order, which rc4's passes use
//!   to pick "the first transform" or "the producer".
//! - [`PyValue`]: argument values with Python's `==`, the equality behind
//!   `FuseDuplicateUsersPass`'s signatures and the arg comparisons in
//!   `FuseIdenticalInputTransformsPass`.
//! - [`rc4_target`]: rc4's spelling of a target RETAB spells differently.
//! - The TOSA `RESCALE` facts that `MoveDataMovementOpsToSmallerDtypePass` and
//!   the TOSA propagation overrides share: [`is_rescale`],
//!   [`is_per_tensor_rescale`], [`special_dtype`] and [`element_size`].

use std::borrow::Cow;

use crate::fx_ir::{IrArg, IrGraph, IrNode, NodeId, NodeMeta};
use crate::fx_target::{classify_ir_target, IrTargetKind};

/// `Node.all_input_nodes` (torch/fx/node.py, `_input_nodes`): the nodes `node`
/// reads, each once, in order of first occurrence: positional args (into
/// nested lists), then kwargs. FX keeps kwargs in insertion order; RETAB keeps
/// them in a map, so kwargs go by key, as `layout::shape` does.
pub(crate) fn all_input_nodes(node: &IrNode) -> Vec<NodeId> {
    let mut kwargs: Vec<(&String, &IrArg)> = node.kwargs.iter().collect();
    kwargs.sort_by(|lhs, rhs| lhs.0.cmp(rhs.0));
    let mut ids = Vec::new();
    for arg in node.args.iter().chain(kwargs.into_iter().map(|(_, arg)| arg)) {
        arg.collect_node_ids(&mut ids);
    }
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    ids.retain(|&id| seen.insert(id));
    ids
}

/// rc4's spelling of `target`. RETAB also writes `tosa.X` for
/// `backend.tosa.X`, and `_clone_dim_order.default` and
/// `_to_dim_order_copy.default` without their `dim_order_ops.` namespace
/// (`fx_target`); every other target is already spelled as rc4 prints it.
pub(crate) fn rc4_target(target: &str) -> Cow<'_, str> {
    if let Some(rest) = target.strip_prefix("tosa.") {
        return Cow::Owned(format!("backend.tosa.{rest}"));
    }
    match target {
        "_clone_dim_order.default" | "_to_dim_order_copy.default" => {
            Cow::Owned(format!("dim_order_ops.{target}"))
        }
        _ => Cow::Borrowed(target),
    }
}

/// An argument as rc4's `_to_hashable` (backends/transforms/fuse_duplicate_users_pass.py:24-60)
/// normalizes it, with Python's `==` as equality and a matching hash.
///
/// - Python compares numbers by value across types (`2 == 2.0`, `True == 1`,
///   `0.0 == -0.0`), so ints, bools and integral floats share [`PyValue::Int`].
/// - Lists and tuples become tuples; RETAB keeps both as `IrArg::List`.
/// - A node is its name in rc4 (`_map_leaf_to_key`), which identifies it, so
///   it is its id here.
/// - A torch dtype is `("dtype", str(dtype))`, never equal to a string.
///
/// A NaN equals nothing in Python unless it is the very same float object;
/// RETAB cannot tell objects apart, so [`PyValue::of`] has no value for an
/// argument holding a NaN, and callers treat it as equal to nothing. RETAB's
/// importer also reads an empty list or tuple, and any object it doesn't know
/// (a device, a memory format), as `None` (`fx_ir::resolve_arg`), so those
/// compare equal to `None` here although they don't in rc4. An empty list a
/// pass writes (`IrArg::List(vec![])`, as the layout ports spell a rank-0
/// view's shape) is the same `None`, so it equals an imported `[]`, as the two
/// are equal in rc4.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum PyValue {
    Node(NodeId),
    /// An int, a bool, or a float with an exact integer value.
    Int(i64),
    /// Any other non-NaN float, by its bits.
    Float(u64),
    Str(String),
    Dtype(String),
    Tuple(Vec<PyValue>),
    None,
}

impl PyValue {
    /// `arg`'s value, or `None` if it holds a NaN.
    pub(crate) fn of(arg: &IrArg) -> Option<Self> {
        Some(match arg {
            IrArg::Node(id) => Self::Node(*id),
            IrArg::Int(value) => Self::Int(*value),
            IrArg::Bool(value) => Self::Int(i64::from(*value)),
            IrArg::Float(value) => Self::float(*value)?,
            IrArg::Str(text) => Self::Str(text.clone()),
            IrArg::Dtype(dtype) => Self::Dtype(dtype.clone()),
            IrArg::List(items) if items.is_empty() => Self::None,
            IrArg::List(items) => Self::Tuple(items.iter().map(Self::of).collect::<Option<_>>()?),
            IrArg::None => Self::None,
        })
    }

    fn float(value: f64) -> Option<Self> {
        if value.is_nan() {
            return None;
        }
        // `i64::MIN as f64` is exactly -2^63; `i64::MAX as f64` rounds up to
        // 2^63, which is out of range, hence the strict upper bound.
        let integral = value.fract() == 0.0 && value >= i64::MIN as f64 && value < i64::MAX as f64;
        Some(if integral {
            Self::Int(value as i64)
        } else {
            Self::Float(value.to_bits())
        })
    }
}

/// Python's `lhs == rhs` for two arguments; false when either holds a NaN.
pub(crate) fn py_eq(lhs: &IrArg, rhs: &IrArg) -> bool {
    matches!((PyValue::of(lhs), PyValue::of(rhs)), (Some(lhs), Some(rhs)) if lhs == rhs)
}

/// The meta key of `TosaSpecialDtype` (backends/arm/tosa/mapping.py:56-63).
pub(crate) const SPECIAL_DTYPE_KEY: &str = "tosa_special_dtype";

/// `node.meta.get(TosaSpecialDtype.meta_key())`, by member name. RETAB's
/// importer records the enum's display (`TosaSpecialDtype.INT48`,
/// `NodeMeta::from_fx`) while its passes write the bare name (`INT48`), so
/// the prefix is dropped.
pub(crate) fn special_dtype(meta: &NodeMeta) -> Option<&str> {
    meta.extra_str(SPECIAL_DTYPE_KEY)
        .map(|name| name.strip_prefix("TosaSpecialDtype.").unwrap_or(name))
}

/// `node.target == exir_ops.backend.tosa.RESCALE.default`, in either of
/// RETAB's spellings.
pub(crate) fn is_rescale(target: &str) -> bool {
    classify_ir_target(target) == IrTargetKind::TosaRescale
}

/// `_is_per_tensor_rescale`
/// (backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py:89-98;
/// `TosaPropagationOverrides._is_per_tensor_rescale` at
/// backends/arm/_passes/propagate_view_copy_permute_pass.py:130-142 is the same
/// test). Like rc4 it doesn't check the target: callers already know `id` is a
/// RESCALE.
///
/// The scales argument is per tensor when it is not a sequence or holds one
/// entry. RETAB's importer turns an empty scale list into `IrArg::None`
/// (`fx_ir::resolve_arg`); the schema (`float[] scale`) admits no Python
/// `None`, so `None` is read as the empty list, which is not per tensor.
pub(crate) fn is_per_tensor_rescale(graph: &IrGraph, id: NodeId) -> bool {
    let node = graph.node(id);
    let inputs = all_input_nodes(node);
    if node.args.len() < 3 || inputs.len() != 1 {
        return false;
    }
    if special_dtype(&graph.node(inputs[0]).meta) != special_dtype(&node.meta) {
        return false;
    }
    match &node.args[2] {
        IrArg::List(scales) => scales.len() == 1,
        IrArg::None => false,
        // A Python str is a sequence of characters.
        IrArg::Str(text) => text.chars().count() == 1,
        _ => true,
    }
}

/// `torch.Tensor.element_size()` for a dtype named as torch prints it, or
/// `None` for a name torch doesn't have.
pub(crate) fn element_size(dtype: &str) -> Option<usize> {
    Some(match dtype.strip_prefix("torch.")? {
        "bool" | "uint8" | "int8" | "quint8" | "qint8" | "quint4x2" | "quint2x4" | "float8_e4m3fn"
        | "float8_e4m3fnuz" | "float8_e5m2" | "float8_e5m2fnuz" | "float8_e8m0fnu"
        | "float4_e2m1fn_x2" | "bits8" => 1,
        "int16" | "uint16" | "float16" | "bfloat16" | "bits16" => 2,
        "int32" | "uint32" | "float32" | "qint32" | "complex32" => 4,
        "int64" | "uint64" | "float64" | "complex64" => 8,
        "complex128" => 16,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::fx_ir::{IrGraph, NodeOp};
    use crate::fx_meta::MetaValue;

    fn value(arg: IrArg) -> PyValue {
        PyValue::of(&arg).expect("a value without NaN")
    }

    /// Python: `2 == 2.0`, `True == 1`, `0.0 == -0.0`, `[1, 2] == (1, 2)`;
    /// `2 != True`, `1.5 != 1`, `torch.int8 != "torch.int8"`, `nan != nan`.
    #[test]
    fn values_compare_like_python() {
        assert_eq!(value(IrArg::Int(2)), value(IrArg::Float(2.0)));
        assert_eq!(value(IrArg::Bool(true)), value(IrArg::Int(1)));
        assert_eq!(value(IrArg::Bool(false)), value(IrArg::Float(-0.0)));
        assert_eq!(value(IrArg::Float(0.0)), value(IrArg::Float(-0.0)));
        assert_ne!(value(IrArg::Int(2)), value(IrArg::Bool(true)));
        assert_ne!(value(IrArg::Float(1.5)), value(IrArg::Int(1)));
        assert_ne!(
            value(IrArg::Dtype("torch.int8".into())),
            value(IrArg::Str("torch.int8".into()))
        );
        assert_eq!(
            value(IrArg::List(vec![IrArg::Int(1), IrArg::Float(2.0)])),
            value(IrArg::List(vec![IrArg::Bool(true), IrArg::Int(2)]))
        );
        assert_eq!(value(IrArg::Float(f64::INFINITY)), value(IrArg::Float(f64::INFINITY)));
        // 2^53 + 1 is not a float; Python compares ints and floats exactly.
        assert_ne!(
            value(IrArg::Int((1 << 53) + 1)),
            value(IrArg::Float(((1i64 << 53) + 1) as f64))
        );
        // An imported `[]` is `IrArg::None`; one a pass writes equals it.
        assert_eq!(value(IrArg::List(Vec::new())), value(IrArg::None));
        assert_eq!(PyValue::of(&IrArg::Float(f64::NAN)), None);
        assert_eq!(PyValue::of(&IrArg::List(vec![IrArg::Float(f64::NAN)])), None);
        assert!(!py_eq(&IrArg::Float(f64::NAN), &IrArg::Float(f64::NAN)));
        assert!(py_eq(&IrArg::Float(1e300), &IrArg::Float(1e300)));
        assert!(py_eq(&IrArg::Float(-9.223_372_036_854_776e18), &IrArg::Int(i64::MIN)));
        assert!(!py_eq(&IrArg::Float(9.223_372_036_854_776e18), &IrArg::Int(i64::MAX)));
    }

    #[test]
    fn input_nodes_follow_fx_order_without_repeats() {
        let mut graph = IrGraph::empty();
        let ids: Vec<NodeId> = ["a", "b", "c", "d"]
            .iter()
            .map(|name| {
                graph.add_node(
                    NodeOp::Placeholder,
                    name.to_string(),
                    name.to_string(),
                    Vec::new(),
                    NodeMeta::tensor(vec![1], "torch.float32"),
                )
            })
            .collect();
        let mut kwargs = HashMap::new();
        kwargs.insert("z".to_string(), IrArg::Node(ids[0]));
        kwargs.insert("y".to_string(), IrArg::Node(ids[3]));
        let call = graph.add_call_function_with_kwargs(
            "call",
            "aten.cat.default",
            vec![
                IrArg::List(vec![IrArg::Node(ids[2]), IrArg::Node(ids[0]), IrArg::Node(ids[2])]),
                IrArg::Int(0),
                IrArg::Node(ids[1]),
            ],
            kwargs,
            NodeMeta::default(),
        );
        assert_eq!(all_input_nodes(graph.node(call)), vec![ids[2], ids[0], ids[1], ids[3]]);
    }

    #[test]
    fn targets_use_rc4_spellings() {
        assert_eq!(rc4_target("tosa.RESCALE.default"), "backend.tosa.RESCALE.default");
        assert_eq!(rc4_target("backend.tosa.ADD.default"), "backend.tosa.ADD.default");
        assert_eq!(
            rc4_target("_clone_dim_order.default"),
            "dim_order_ops._clone_dim_order.default"
        );
        assert_eq!(rc4_target("aten._to_copy.default"), "aten._to_copy.default");
        assert!(is_rescale("tosa.RESCALE.default") && is_rescale("backend.tosa.RESCALE.default"));
        assert!(!is_rescale("aten.permute_copy.default"));
    }

    #[test]
    fn special_dtypes_read_both_spellings() {
        let mut meta = NodeMeta::default();
        assert_eq!(special_dtype(&meta), None);
        meta.set_extra(SPECIAL_DTYPE_KEY, MetaValue::Str("TosaSpecialDtype.INT48".into()));
        assert_eq!(special_dtype(&meta), Some("INT48"));
        meta.set_extra(SPECIAL_DTYPE_KEY, MetaValue::Str("INT48".into()));
        assert_eq!(special_dtype(&meta), Some("INT48"));
    }

    #[test]
    fn per_tensor_rescales_have_one_input_one_scale_and_matching_special_dtypes() {
        let mut graph = IrGraph::empty();
        let x = graph.add_node(
            NodeOp::Placeholder,
            "x".into(),
            "x".into(),
            Vec::new(),
            NodeMeta::tensor(vec![2], "torch.int32"),
        );
        let rescale = |graph: &mut IrGraph, name: &str, scales: IrArg| {
            graph.add_call_function(
                name,
                "backend.tosa.RESCALE.default",
                vec![
                    IrArg::Node(x),
                    IrArg::Dtype("torch.int8".into()),
                    scales,
                    IrArg::Int(0),
                    IrArg::Int(0),
                ],
                NodeMeta::tensor(vec![2], "torch.int8"),
            )
        };
        let one = rescale(&mut graph, "one", IrArg::List(vec![IrArg::Float(0.5)]));
        let two = rescale(
            &mut graph,
            "two",
            IrArg::List(vec![IrArg::Float(0.5), IrArg::Float(0.25)]),
        );
        let scalar = rescale(&mut graph, "scalar", IrArg::Float(0.5));
        let empty = rescale(&mut graph, "empty", IrArg::None);
        assert!(is_per_tensor_rescale(&graph, one));
        assert!(!is_per_tensor_rescale(&graph, two));
        assert!(is_per_tensor_rescale(&graph, scalar));
        assert!(!is_per_tensor_rescale(&graph, empty));
        graph
            .node_mut(one)
            .meta
            .set_extra(SPECIAL_DTYPE_KEY, MetaValue::Str("INT48".into()));
        assert!(!is_per_tensor_rescale(&graph, one));
        graph
            .node_mut(x)
            .meta
            .set_extra(SPECIAL_DTYPE_KEY, MetaValue::Str("TosaSpecialDtype.INT48".into()));
        assert!(is_per_tensor_rescale(&graph, one));
        let short = graph.add_call_function(
            "short",
            "backend.tosa.RESCALE.default",
            vec![IrArg::Node(x), IrArg::Dtype("torch.int8".into())],
            NodeMeta::tensor(vec![2], "torch.int8"),
        );
        assert!(!is_per_tensor_rescale(&graph, short));
    }

    #[test]
    fn element_sizes_are_torchs() {
        assert_eq!(element_size("torch.int8"), Some(1));
        assert_eq!(element_size("torch.bool"), Some(1));
        assert_eq!(element_size("torch.int16"), Some(2));
        assert_eq!(element_size("torch.bfloat16"), Some(2));
        assert_eq!(element_size("torch.int32"), Some(4));
        assert_eq!(element_size("torch.float32"), Some(4));
        assert_eq!(element_size("torch.int64"), Some(8));
        assert_eq!(element_size("int8"), None);
        assert_eq!(element_size("torch.int48"), None);
    }
}
