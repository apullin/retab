//! Local metadata refresh for the nodes layout passes move, cross or create.
//!
//! rc4's propagation passes recompute a node's `meta["val"]` right after a
//! rewrite (`PropagateViewCopyPermutePass._refresh_node_meta`,
//! `backends/transforms/propagate_view_copy_permute_pass.py:164-192`). View and
//! permute copies derive their value from their input
//! (`refresh_permute_view_meta`, `backends/transforms/permute_view_meta.py:22-56`);
//! any other op is fake-executed. RETAB has no fake tensors, so this module
//! derives [`TensorFacts`] with one rule per op family the layout passes touch:
//! view and permute copies, pointwise ops with numpy broadcasting, transparent
//! copies, `mean.dim` and `sum.dim_IntList`, `slice_copy`, `cat` and
//! `backend.tosa.RESCALE`. A target without a rule is an error, never a guess.
//!
//! Three rules decide what a refresh changes:
//!
//! - **Shape comes from the operands**, with torch's semantics for each op,
//!   including the argument checks fake execution (or rc4's later retrace)
//!   would make.
//! - **Dtype follows the producer.** Fake execution re-derives dtypes with
//!   torch's promotion rules, and moving a copy never changes an operand's
//!   dtype, so in rc4 a crossed node keeps the dtype it had. RETAB's IR differs
//!   from torch on purpose in places: quantized table ops such as `aten.sigmoid`
//!   keep integer vals and lower to a deferred TABLE (`fx_insert_table_ops_ir`),
//!   and integer accumulators such as `sum` carry `torch.int32` vals without a
//!   `dtype` kwarg (`fx_insert_int32_rescale_ir`). torch's rules (int to float
//!   for sigmoid, int64 for an integer sum) would change dtypes a move cannot
//!   change, so copies, unary ops and reductions take their input's dtype;
//!   multi-operand ops promote their value operands with torch's rules, which
//!   is the identity when they agree, as TOSA requires; comparisons, `isinf`,
//!   `isnan` and logical ops give `torch.bool`; RESCALE and dtype casts take
//!   their dtype argument.
//! - **Moved copies lose stale qparams.** A view or permute that moved no longer
//!   sits where its qparams were recorded; once it crosses a RESCALE, say, its
//!   output qparams describe the other side. Its refreshed meta drops them, as
//!   rc4's `create_node` drops them from copies it makes out of another node's
//!   meta (`backends/arm/_passes/arm_pass_utils.py:174-182`). Crossed ops keep
//!   theirs: a view or permute does not change the values they see.
//!
//! Everything else in the meta is left alone, as rc4 assigns only
//! `meta["val"]`. Layout facts in particular: explicit mode never writes them.

use std::fmt;

use crate::fx_ir::{broadcast_shapes, normalize_dim, IrArg, IrGraph, IrNode, NodeId, NodeMeta};
use crate::fx_meta::TensorFacts;
use crate::fx_target::{classify_ir_target, IrTargetKind};
use crate::layout::pointwise::{is_pointwise, is_transparent_copy};

const PERMUTE_COPY: &str = "aten.permute_copy.default";
const VIEW_COPY: &str = "aten.view_copy.default";
const VIEW_DEFAULT: &str = "aten.view.default";
const MEAN_DIM: &str = "aten.mean.dim";
const SUM_DIM_INT_LIST: &str = "aten.sum.dim_IntList";
const SLICE_COPY: &str = "aten.slice_copy.Tensor";
const CAT_DEFAULT: &str = "aten.cat.default";
const WHERE_SELF: &str = "aten.where.self";
const MASKED_FILL_SCALAR: &str = "aten.masked_fill.Scalar";
const BOOL: &str = "torch.bool";

/// Pointwise targets whose result is `torch.bool` whatever their operands.
/// Each one is in the pointwise table (see the test below).
const BOOL_VALUED_POINTWISE: &[&str] = &[
    "aten.eq.Scalar",
    "aten.eq.Tensor",
    "aten.ge.Scalar",
    "aten.ge.Tensor",
    "aten.gt.Scalar",
    "aten.gt.Tensor",
    "aten.isinf.default",
    "aten.isnan.default",
    "aten.le.Scalar",
    "aten.le.Tensor",
    "aten.logical_and.default",
    "aten.logical_not.default",
    "aten.logical_or.default",
    "aten.logical_xor.default",
    "aten.lt.Scalar",
    "aten.lt.Tensor",
    "aten.ne.Scalar",
    "aten.ne.Tensor",
];

/// Why a node's value facts could not be derived locally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShapeError {
    /// No local rule covers the target.
    Unsupported(String),
    /// A tensor operand is absent or has no value facts.
    MissingOperand(&'static str),
    /// An argument is malformed, or out of range for its operand.
    InvalidArg(&'static str),
    /// Operand shapes or dtypes do not combine.
    Incompatible(&'static str),
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(target) => write!(f, "no local shape rule for {target}"),
            Self::MissingOperand(what) => write!(f, "missing tensor operand: {what}"),
            Self::InvalidArg(what) => write!(f, "invalid argument: {what}"),
            Self::Incompatible(what) => write!(f, "incompatible operands: {what}"),
        }
    }
}

impl std::error::Error for ShapeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rule {
    Permute,
    View,
    Elementwise,
    TransparentCopy,
    Reduction,
    Slice,
    Cat,
    Rescale,
}

fn rule(target: &str) -> Option<Rule> {
    match target {
        PERMUTE_COPY => Some(Rule::Permute),
        VIEW_COPY | VIEW_DEFAULT => Some(Rule::View),
        MEAN_DIM | SUM_DIM_INT_LIST => Some(Rule::Reduction),
        SLICE_COPY => Some(Rule::Slice),
        CAT_DEFAULT => Some(Rule::Cat),
        _ if classify_ir_target(target) == IrTargetKind::TosaRescale => Some(Rule::Rescale),
        _ if is_transparent_copy(target) => Some(Rule::TransparentCopy),
        _ if is_pointwise(target) => Some(Rule::Elementwise),
        _ => None,
    }
}

/// Whether `target` is a view or permute copy, the nodes rc4's propagation
/// moves (`PropagateViewCopyPermutePass._TARGETS`,
/// `backends/transforms/propagate_view_copy_permute_pass.py:59-62`).
pub(crate) fn is_layout_transform(target: &str) -> bool {
    matches!(target, PERMUTE_COPY | VIEW_COPY | VIEW_DEFAULT)
}

/// Whether a local rule covers `target`, so a planner can reject a rewrite
/// before touching the graph.
pub(crate) fn has_local_rule(target: &str) -> bool {
    rule(target).is_some()
}

/// `id`'s meta with its value facts re-derived from its current operands.
///
/// Mirrors rc4's `_refresh_node_meta`
/// (`backends/transforms/propagate_view_copy_permute_pass.py:164-192`): a node
/// without input nodes is returned unchanged (`:165-166`), view and permute
/// copies follow `refresh_permute_view_meta` (`:167-171`) and lose their qparams
/// (see the module docs), and other ops get the facts fake execution would give
/// (`:173-192`). Callers refresh in dependency order, as rc4 does: after moving
/// a copy up, the copy then the crossed nodes in reverse path order (`:560-567`);
/// after moving it down, the crossed nodes in path order then the copy
/// (`:912-919`).
pub(crate) fn refreshed_meta(graph: &IrGraph, id: NodeId) -> Result<NodeMeta, ShapeError> {
    let node = graph.node(id);
    let mut meta = node.meta.clone();
    if !has_input_nodes(node) {
        return Ok(meta);
    }
    meta.set_tensor_facts(infer_facts(graph, id)?);
    if is_layout_transform(&node.target) {
        meta.clear_all_qparams();
    }
    Ok(meta)
}

/// Value facts `id` would have for its current operands.
pub(crate) fn infer_facts(graph: &IrGraph, id: NodeId) -> Result<TensorFacts, ShapeError> {
    infer_node_facts(graph.node(id), &|input| {
        graph
            .node_opt(input)
            .and_then(|node| node.meta.tensor_facts())
    })
}

/// Value facts `node` would have, reading operand facts through `facts_of`.
///
/// `node` need not be in a graph: a planner can pass a node with rewritten
/// arguments, and a lookup that answers for operands it has not created yet.
pub(crate) fn infer_node_facts<'a>(
    node: &IrNode,
    facts_of: &dyn Fn(NodeId) -> Option<&'a TensorFacts>,
) -> Result<TensorFacts, ShapeError> {
    let rule = rule(&node.target).ok_or_else(|| ShapeError::Unsupported(node.target.clone()))?;
    let operands = Operands { node, facts_of };
    match rule {
        Rule::Permute => operands.permute(),
        Rule::View => operands.view(),
        Rule::Elementwise => operands.elementwise(),
        Rule::TransparentCopy => operands.transparent_copy(),
        Rule::Reduction => operands.reduction(),
        Rule::Slice => operands.slice(),
        Rule::Cat => operands.cat(),
        Rule::Rescale => operands.rescale(),
    }
}

fn has_input_nodes(node: &IrNode) -> bool {
    let mut ids = Vec::new();
    for arg in node.args.iter().chain(node.kwargs.values()) {
        arg.collect_node_ids(&mut ids);
    }
    !ids.is_empty()
}

/// A node's arguments, resolved against operand facts.
struct Operands<'n, 'a> {
    node: &'n IrNode,
    facts_of: &'n dyn Fn(NodeId) -> Option<&'a TensorFacts>,
}

impl<'n, 'a> Operands<'n, 'a> {
    /// A schema parameter passed positionally or by name.
    fn arg(&self, index: usize, name: &str) -> Option<&'n IrArg> {
        self.node
            .args
            .get(index)
            .or_else(|| self.node.kwargs.get(name))
    }

    fn facts(&self, id: NodeId, what: &'static str) -> Result<&'a TensorFacts, ShapeError> {
        (self.facts_of)(id).ok_or(ShapeError::MissingOperand(what))
    }

    fn tensor(&self, index: usize, name: &'static str) -> Result<&'a TensorFacts, ShapeError> {
        let id = self
            .arg(index, name)
            .and_then(IrArg::as_node)
            .ok_or(ShapeError::MissingOperand(name))?;
        self.facts(id, name)
    }

    /// Direct tensor operands in FX `all_input_nodes` order: positional
    /// arguments, then keyword arguments (sorted, as RETAB keeps them in a map).
    fn tensor_operands(&self) -> Result<Vec<&'a TensorFacts>, ShapeError> {
        let mut kwargs: Vec<_> = self.node.kwargs.iter().collect();
        kwargs.sort_by(|lhs, rhs| lhs.0.cmp(rhs.0));
        self.node
            .args
            .iter()
            .chain(kwargs.into_iter().map(|(_, arg)| arg))
            .filter_map(IrArg::as_node)
            .map(|id| self.facts(id, "operand"))
            .collect()
    }

    /// rc4 `refresh_permute_view_meta`, permute branch
    /// (`backends/transforms/permute_view_meta.py:48-54`).
    fn permute(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "self")?;
        let dims = required_int_list(self.arg(1, "dims"), "permute dims")?;
        let shape = permute_shape(&input.shape, &dims)?;
        Ok(TensorFacts::new(shape, input.dtype.clone()))
    }

    /// rc4 `refresh_permute_view_meta`, view branch
    /// (`backends/transforms/permute_view_meta.py:40-47`). `aten.view.default`
    /// is fake-executed in rc4, with the same result.
    fn view(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "self")?;
        let size = required_int_list(self.arg(1, "size"), "view size")?;
        let shape = view_shape(&input.shape, &size)?;
        Ok(TensorFacts::new(shape, input.dtype.clone()))
    }

    /// Fake execution of a pointwise op (`:173-192`): operands broadcast; the
    /// dtype rule is in the module docs. In-place variants keep `self`'s shape
    /// and dtype, and reject operands that would broadcast past it, as torch does.
    fn elementwise(&self) -> Result<TensorFacts, ShapeError> {
        let target = self.node.target.as_str();
        let operands = self.tensor_operands()?;
        if operands.is_empty() {
            return Err(ShapeError::MissingOperand("self"));
        }
        let shapes: Vec<Vec<i64>> = operands.iter().map(|facts| facts.shape.clone()).collect();
        let shape = broadcast_shapes(&shapes)
            .ok_or(ShapeError::Incompatible("operand shapes do not broadcast"))?;
        if is_in_place(target) {
            let this = self.tensor(0, "self")?;
            if shape != this.shape {
                return Err(ShapeError::Incompatible(
                    "an in-place op cannot broadcast past self",
                ));
            }
            return Ok(this.clone());
        }
        let dtype = if BOOL_VALUED_POINTWISE.contains(&target) {
            BOOL.to_string()
        } else {
            let values = match target {
                WHERE_SELF => vec![self.tensor(1, "self")?, self.tensor(2, "other")?],
                MASKED_FILL_SCALAR => vec![self.tensor(0, "self")?],
                _ => operands,
            };
            result_dtype(&values)?
        };
        Ok(TensorFacts::new(shape, dtype))
    }

    /// `_clone_dim_order`, `_to_dim_order_copy` and RETAB's `_to_copy`: the
    /// input's shape, and the `dtype` kwarg when one is given.
    fn transparent_copy(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "self")?;
        let dtype = dtype_kwarg(self.node)?.unwrap_or_else(|| input.dtype.clone());
        Ok(TensorFacts::new(input.shape.clone(), dtype))
    }

    /// `mean.dim` and `sum.dim_IntList`: `(self, int[1]? dim, bool keepdim=False,
    /// *, ScalarType? dtype=None)`. A missing or empty `dim` reduces every axis.
    fn reduction(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "self")?;
        let dims = match self.arg(1, "dim") {
            None | Some(IrArg::None) => Vec::new(),
            Some(IrArg::Int(dim)) => vec![*dim],
            Some(arg) => arg
                .as_int_list()
                .ok_or(ShapeError::InvalidArg("reduction dims"))?,
        };
        let keepdim = match self.arg(2, "keepdim") {
            None => false,
            Some(IrArg::Bool(keepdim)) => *keepdim,
            Some(_) => return Err(ShapeError::InvalidArg("reduction keepdim")),
        };
        let shape = reduce_shape(&input.shape, &dims, keepdim)?;
        let dtype = dtype_kwarg(self.node)?.unwrap_or_else(|| input.dtype.clone());
        Ok(TensorFacts::new(shape, dtype))
    }

    /// `slice_copy.Tensor(self, int dim=0, SymInt? start=None, SymInt? end=None,
    /// SymInt step=1)`.
    fn slice(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "self")?;
        let dim = optional_int(self.arg(1, "dim"), "slice dim")?.unwrap_or(0);
        let start = optional_int(self.arg(2, "start"), "slice start")?;
        let end = optional_int(self.arg(3, "end"), "slice end")?;
        let step = optional_int(self.arg(4, "step"), "slice step")?.unwrap_or(1);
        let shape = slice_shape(&input.shape, dim, start, end, step)?;
        Ok(TensorFacts::new(shape, input.dtype.clone()))
    }

    /// `cat(Tensor[] tensors, int dim=0)`; the dtype promotes over every input,
    /// as `torch.cat` does.
    fn cat(&self) -> Result<TensorFacts, ShapeError> {
        let Some(IrArg::List(items)) = self.arg(0, "tensors") else {
            return Err(ShapeError::InvalidArg("cat tensors"));
        };
        let inputs = items
            .iter()
            .map(|item| {
                let id = item
                    .as_node()
                    .ok_or(ShapeError::InvalidArg("cat tensors"))?;
                self.facts(id, "cat tensor")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dim = optional_int(self.arg(1, "dim"), "cat dim")?.unwrap_or(0);
        let shapes: Vec<&[i64]> = inputs.iter().map(|facts| facts.shape.as_slice()).collect();
        let shape = cat_shape(&shapes, dim)?;
        Ok(TensorFacts::new(shape, result_dtype(&inputs)?))
    }

    /// `backend.tosa.RESCALE(Tensor input1, ScalarType dtype, ...)`: rc4's fake
    /// kernel returns `torch.empty_like(x, dtype=dtype)`
    /// (`backends/arm/tosa/dialect/ops/conversion_ops.py:213`). The checks before
    /// it (`:128-212`: integer support, output dtype, signedness, zero points)
    /// judge lowering legality and are not ported.
    fn rescale(&self) -> Result<TensorFacts, ShapeError> {
        let input = self.tensor(0, "input1")?;
        let dtype = match self.arg(1, "dtype") {
            Some(IrArg::Dtype(dtype) | IrArg::Str(dtype)) => dtype.clone(),
            _ => return Err(ShapeError::InvalidArg("RESCALE dtype")),
        };
        Ok(TensorFacts::new(input.shape.clone(), dtype))
    }
}

/// An `int[]` parameter. FX import turns an empty list into `IrArg::None`
/// (`fx_ir::resolve_arg`), which is how a rank-0 permute or view arrives.
fn required_int_list(arg: Option<&IrArg>, what: &'static str) -> Result<Vec<i64>, ShapeError> {
    match arg {
        Some(IrArg::None) => Ok(Vec::new()),
        Some(arg) => arg.as_int_list().ok_or(ShapeError::InvalidArg(what)),
        None => Err(ShapeError::InvalidArg(what)),
    }
}

fn optional_int(arg: Option<&IrArg>, what: &'static str) -> Result<Option<i64>, ShapeError> {
    match arg {
        None | Some(IrArg::None) => Ok(None),
        Some(IrArg::Int(value)) => Ok(Some(*value)),
        Some(_) => Err(ShapeError::InvalidArg(what)),
    }
}

/// The keyword-only `dtype` argument of reductions and casts.
fn dtype_kwarg(node: &IrNode) -> Result<Option<String>, ShapeError> {
    match node.kwarg("dtype") {
        None | Some(IrArg::None) => Ok(None),
        Some(IrArg::Dtype(dtype) | IrArg::Str(dtype)) => Ok(Some(dtype.clone())),
        Some(_) => Err(ShapeError::InvalidArg("dtype")),
    }
}

/// In-place variants such as `aten.add_.Tensor`; dunder ops such as
/// `aten.__lshift__.Scalar` are not in place.
fn is_in_place(target: &str) -> bool {
    target
        .rsplit('.')
        .nth(1)
        .is_some_and(|op| op.ends_with('_') && !op.starts_with("__"))
}

/// rc4's `_normalize_dim` (`backends/transforms/dim_maps.py:40-43`), as
/// `fx_ir::normalize_dim`, with the failure as an error.
fn axis(dim: i64, rank: usize) -> Result<usize, ShapeError> {
    normalize_dim(dim, rank).ok_or(ShapeError::InvalidArg("dim out of range"))
}

fn numel(shape: &[i64]) -> Result<i64, ShapeError> {
    shape.iter().try_fold(1i64, |count, &extent| {
        if extent < 0 {
            return Err(ShapeError::InvalidArg("negative extent"));
        }
        count
            .checked_mul(extent)
            .ok_or(ShapeError::InvalidArg("element count overflows"))
    })
}

/// Output shape of `permute_copy(shape, dims)`: rc4 `refresh_permute_view_meta`
/// (`backends/transforms/permute_view_meta.py:48-54`) with `_normalize_dims`
/// (`backends/transforms/dim_maps.py:46-49`). torch also rejects a `dims` that
/// is not a permutation, which rc4's later retrace would catch.
pub(crate) fn permute_shape(shape: &[i64], dims: &[i64]) -> Result<Vec<i64>, ShapeError> {
    if dims.len() != shape.len() {
        return Err(ShapeError::InvalidArg("permute dims must name every axis"));
    }
    let mut seen = vec![false; shape.len()];
    dims.iter()
        .map(|&dim| {
            let axis = axis(dim, shape.len())?;
            if std::mem::replace(&mut seen[axis], true) {
                return Err(ShapeError::InvalidArg("permute dims repeat an axis"));
            }
            Ok(shape[axis])
        })
        .collect()
}

/// Output shape of `view_copy(source, size)`: rc4's `normalize_view_shape`
/// (`backends/transforms/dim_maps.py:183-205`) infers the one `-1`. torch's
/// view checks are added (one `-1` at most, no other negative extent, the
/// element count kept, no `-1` against zero known elements), since rc4's
/// retrace would make them.
pub(crate) fn view_shape(source: &[i64], size: &[i64]) -> Result<Vec<i64>, ShapeError> {
    let source_numel = numel(source)?;
    let mut inferred = None;
    let mut known = 1i64;
    for (index, &extent) in size.iter().enumerate() {
        if extent == -1 {
            if inferred.replace(index).is_some() {
                return Err(ShapeError::InvalidArg("only one view dim can be inferred"));
            }
        } else if extent < 0 {
            return Err(ShapeError::InvalidArg("negative view extent"));
        } else {
            known = known
                .checked_mul(extent)
                .ok_or(ShapeError::InvalidArg("element count overflows"))?;
        }
    }
    let mut shape = size.to_vec();
    match inferred {
        Some(index) => {
            if known == 0 || source_numel % known != 0 {
                return Err(ShapeError::Incompatible(
                    "view size does not divide the input",
                ));
            }
            shape[index] = source_numel / known;
        }
        None if known != source_numel => {
            return Err(ShapeError::Incompatible("view must keep the element count"));
        }
        None => {}
    }
    Ok(shape)
}

/// Numpy broadcast of `shapes`, as fake execution of a pointwise op computes it.
pub(crate) fn broadcast_shape(shapes: &[&[i64]]) -> Result<Vec<i64>, ShapeError> {
    let shapes: Vec<Vec<i64>> = shapes.iter().map(|shape| shape.to_vec()).collect();
    broadcast_shapes(&shapes).ok_or(ShapeError::Incompatible("operand shapes do not broadcast"))
}

/// Output shape of a `dim`-list reduction with torch's semantics: an empty
/// `dims` reduces every axis, an axis may appear once, and a rank-0 operand
/// accepts dims `0` and `-1`. A full reduction without `keepdim` is rank 0.
///
/// RETAB's `fx_reduction` helpers are not reused: they skip out-of-range dims
/// and give `[1]` for a full reduction, the shape RETAB emits for TOSA.
pub(crate) fn reduce_shape(
    shape: &[i64],
    dims: &[i64],
    keepdim: bool,
) -> Result<Vec<i64>, ShapeError> {
    // torch wraps dims against at least rank 1.
    let mut reduced = vec![dims.is_empty(); shape.len().max(1)];
    for &dim in dims {
        let axis = axis(dim, reduced.len())?;
        if std::mem::replace(&mut reduced[axis], true) {
            return Err(ShapeError::InvalidArg("reduction dims repeat an axis"));
        }
    }
    Ok(shape
        .iter()
        .zip(reduced)
        .filter_map(|(&extent, reduced)| match (reduced, keepdim) {
            (false, _) => Some(extent),
            (true, true) => Some(1),
            (true, false) => None,
        })
        .collect())
}

/// Output shape of `slice_copy.Tensor` with torch's bounds handling: `None`
/// bounds span the axis, negative bounds count from its end, both clamp to
/// it, an end before the start gives an empty slice, and `step` must be
/// positive.
pub(crate) fn slice_shape(
    shape: &[i64],
    dim: i64,
    start: Option<i64>,
    end: Option<i64>,
    step: i64,
) -> Result<Vec<i64>, ShapeError> {
    if step <= 0 {
        return Err(ShapeError::InvalidArg("slice step must be positive"));
    }
    let axis = axis(dim, shape.len())?;
    let extent = shape[axis];
    if extent < 0 {
        return Err(ShapeError::InvalidArg("negative extent"));
    }
    let wrap = |bound: i64| {
        if bound < 0 {
            bound.saturating_add(extent)
        } else {
            bound
        }
    };
    let start = wrap(start.unwrap_or(0)).clamp(0, extent);
    let end = wrap(end.unwrap_or(i64::MAX)).clamp(start, extent);
    let length = end - start;
    let mut out = shape.to_vec();
    out[axis] = if length == 0 {
        0
    } else {
        (length - 1) / step + 1
    };
    Ok(out)
}

/// Output shape of `cat(shapes, dim)` with torch's rules: inputs are at least
/// rank 1; legacy empty inputs (shape `[0]`) are skipped; the rest share a
/// rank and agree off the cat dim, which `dim` names in the first kept input.
pub(crate) fn cat_shape(shapes: &[&[i64]], dim: i64) -> Result<Vec<i64>, ShapeError> {
    if shapes.is_empty() {
        return Err(ShapeError::MissingOperand("cat tensors"));
    }
    if shapes.iter().any(|shape| shape.is_empty()) {
        return Err(ShapeError::InvalidArg(
            "cat inputs must have rank 1 or more",
        ));
    }
    let mut kept = shapes.iter().filter(|shape| **shape != [0]);
    let Some(first) = kept.next() else {
        return Ok(vec![0]);
    };
    let axis = axis(dim, first.len())?;
    let mut out = first.to_vec();
    for shape in kept {
        if shape.len() != out.len() {
            return Err(ShapeError::Incompatible("cat inputs must share a rank"));
        }
        for (index, (&lhs, &rhs)) in out.iter().zip(shape.iter()).enumerate() {
            if index != axis && lhs != rhs {
                return Err(ShapeError::Incompatible(
                    "cat inputs must agree off the cat dim",
                ));
            }
        }
        out[axis] = out[axis]
            .checked_add(shape[axis])
            .ok_or(ShapeError::InvalidArg("element count overflows"))?;
    }
    Ok(out)
}

/// The torch dtypes [`promote_dtypes`] knows, ordered so that outside the
/// special cases the promotion of two is the later one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TorchDtype {
    Bool,
    UInt8,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    BFloat16,
    Float32,
    Float64,
}

impl TorchDtype {
    const NAMES: [(Self, &'static str); 10] = [
        (Self::Bool, "torch.bool"),
        (Self::UInt8, "torch.uint8"),
        (Self::Int8, "torch.int8"),
        (Self::Int16, "torch.int16"),
        (Self::Int32, "torch.int32"),
        (Self::Int64, "torch.int64"),
        (Self::Float16, "torch.float16"),
        (Self::BFloat16, "torch.bfloat16"),
        (Self::Float32, "torch.float32"),
        (Self::Float64, "torch.float64"),
    ];

    fn parse(dtype: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find_map(|&(kind, name)| (name == dtype).then_some(kind))
    }

    fn name(self) -> &'static str {
        Self::NAMES[self as usize].1
    }

    fn is_floating(self) -> bool {
        self >= Self::Float16
    }
}

/// `torch.promote_types` over torch dtype names. Other dtypes, such as the
/// float8 formats, promote only with themselves.
pub(crate) fn promote_dtypes(lhs: &str, rhs: &str) -> Option<String> {
    if lhs == rhs {
        return Some(lhs.to_string());
    }
    let (lhs, rhs) = (TorchDtype::parse(lhs)?, TorchDtype::parse(rhs)?);
    let promoted = match (lhs.min(rhs), lhs.max(rhs)) {
        (TorchDtype::Bool, other) => other,
        (TorchDtype::UInt8, TorchDtype::Int8) => TorchDtype::Int16,
        (TorchDtype::Float16, TorchDtype::BFloat16) => TorchDtype::Float32,
        (_, wider) => wider,
    };
    Some(promoted.name().to_string())
}

/// torch's `result_type` over tensor operands (`at::native::result_type`):
/// dimensioned operands promote together, and rank-0 operands count only when
/// they are of a higher category (`combine_categories`). Python scalars do not
/// take part (the module docs explain why).
fn result_dtype(operands: &[&TensorFacts]) -> Result<String, ShapeError> {
    let mut dimensioned: Option<String> = None;
    let mut zero_dim: Option<String> = None;
    for facts in operands {
        let tier = if facts.shape.is_empty() {
            &mut zero_dim
        } else {
            &mut dimensioned
        };
        let promoted = match tier.take() {
            None => facts.dtype.clone(),
            Some(dtype) => promote_dtypes(&dtype, &facts.dtype)
                .ok_or(ShapeError::Incompatible("operand dtypes do not promote"))?,
        };
        *tier = Some(promoted);
    }
    match (dimensioned, zero_dim) {
        (Some(higher), Some(lower)) => {
            let higher_kind = TorchDtype::parse(&higher);
            let lower_kind = TorchDtype::parse(&lower);
            if higher_kind.is_some_and(TorchDtype::is_floating) {
                Ok(higher)
            } else if higher_kind == Some(TorchDtype::Bool)
                || lower_kind.is_some_and(TorchDtype::is_floating)
            {
                promote_dtypes(&higher, &lower)
                    .ok_or(ShapeError::Incompatible("operand dtypes do not promote"))
            } else {
                Ok(higher)
            }
        }
        (Some(dtype), None) | (None, Some(dtype)) => Ok(dtype),
        (None, None) => Err(ShapeError::MissingOperand("self")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::NodeOp;
    use crate::layout::testing::*;
    use crate::test_ir::qparam;

    const F32: &str = "torch.float32";
    const F16: &str = "torch.float16";
    const I8: &str = "torch.int8";
    const I16: &str = "torch.int16";
    const I32: &str = "torch.int32";
    const RANK0: &[i64] = &[];

    fn ok(result: Result<Vec<i64>, ShapeError>) -> Vec<i64> {
        result.unwrap_or_else(|error| panic!("{error}"))
    }

    fn invalid(what: &'static str) -> Result<Vec<i64>, ShapeError> {
        Err(ShapeError::InvalidArg(what))
    }

    fn incompatible(what: &'static str) -> Result<Vec<i64>, ShapeError> {
        Err(ShapeError::Incompatible(what))
    }

    // Shape helpers. Expected values were checked against torch 2.15 eager ops.

    #[test]
    fn permute_shape_normalizes_dims_and_rejects_non_permutations() {
        assert_eq!(ok(permute_shape(&[2, 3, 4], &[-1, 0, 1])), vec![4, 2, 3]);
        assert_eq!(ok(permute_shape(RANK0, RANK0)), RANK0);
        assert_eq!(
            permute_shape(&[2, 3, 4], &[0, 0, 1]),
            invalid("permute dims repeat an axis")
        );
        assert_eq!(
            permute_shape(&[2, 3, 4], &[1, 0]),
            invalid("permute dims must name every axis")
        );
        assert_eq!(
            permute_shape(&[2, 3, 4], &[0, 1, 3]),
            invalid("dim out of range")
        );
    }

    #[test]
    fn view_shape_infers_one_dim_and_checks_the_element_count() {
        assert_eq!(ok(view_shape(&[2, 3, 4], &[-1, 4])), vec![6, 4]);
        assert_eq!(ok(view_shape(&[2, 3, 4], &[4, -1, 2])), vec![4, 3, 2]);
        assert_eq!(ok(view_shape(RANK0, &[1, 1])), vec![1, 1]);
        assert_eq!(ok(view_shape(&[1, 1], RANK0)), RANK0);
        assert_eq!(ok(view_shape(RANK0, &[-1])), vec![1]);
        assert_eq!(ok(view_shape(&[0, 3], &[3, 0])), vec![3, 0]);
        assert_eq!(ok(view_shape(&[0, 3], &[-1, 3])), vec![0, 3]);
        assert_eq!(
            view_shape(&[2, 3, 4], &[5, -1]),
            incompatible("view size does not divide the input")
        );
        assert_eq!(
            view_shape(&[2, 3, 4], &[-1, -1]),
            invalid("only one view dim can be inferred")
        );
        assert_eq!(
            view_shape(&[2, 3, 4], &[2, 12, 2]),
            incompatible("view must keep the element count")
        );
        assert_eq!(
            view_shape(&[0, 3], &[0, -1]),
            incompatible("view size does not divide the input")
        );
        assert_eq!(
            view_shape(&[2, 3, 4], &[-2, 12]),
            invalid("negative view extent")
        );
    }

    #[test]
    fn broadcast_shape_follows_numpy_rules() {
        assert_eq!(
            ok(broadcast_shape(&[&[3, 1], &[1, 4], &[3, 4]])),
            vec![3, 4]
        );
        assert_eq!(
            ok(broadcast_shape(&[&[3, 1], &[4], &[1, 1, 1]])),
            vec![1, 3, 4]
        );
        assert_eq!(ok(broadcast_shape(&[&[0, 3], &[1, 3]])), vec![0, 3]);
        assert_eq!(ok(broadcast_shape(&[RANK0, &[2]])), vec![2]);
        assert_eq!(
            broadcast_shape(&[&[0, 3], &[2, 3]]),
            incompatible("operand shapes do not broadcast")
        );
    }

    #[test]
    fn reduce_shape_matches_torch_reductions() {
        assert_eq!(ok(reduce_shape(&[2, 3, 4], &[1], false)), vec![2, 4]);
        assert_eq!(ok(reduce_shape(&[2, 3, 4], &[1], true)), vec![2, 1, 4]);
        assert_eq!(ok(reduce_shape(&[2, 3, 4], &[-1, 0], false)), vec![3]);
        assert_eq!(ok(reduce_shape(&[2, 3, 4], &[-1, 0], true)), vec![1, 3, 1]);
        assert_eq!(ok(reduce_shape(&[2, 3, 4], RANK0, false)), RANK0);
        assert_eq!(ok(reduce_shape(&[2, 3, 4], RANK0, true)), vec![1, 1, 1]);
        assert_eq!(ok(reduce_shape(RANK0, &[0], false)), RANK0);
        assert_eq!(ok(reduce_shape(RANK0, &[-1], true)), RANK0);
        assert_eq!(ok(reduce_shape(RANK0, RANK0, false)), RANK0);
        assert_eq!(
            reduce_shape(&[2, 3, 4], &[1, -2], false),
            invalid("reduction dims repeat an axis")
        );
        assert_eq!(
            reduce_shape(RANK0, &[0, -1], false),
            invalid("reduction dims repeat an axis")
        );
        assert_eq!(
            reduce_shape(&[2, 3, 4], &[3], false),
            invalid("dim out of range")
        );
        assert_eq!(
            reduce_shape(RANK0, &[1], false),
            invalid("dim out of range")
        );
    }

    #[test]
    fn slice_shape_clamps_bounds_like_torch() {
        let shape = [2, 3, 4];
        let cases = [
            (1, Some(0), Some(2), 1, [2, 2, 4]),
            (1, Some(-2), Some(i64::MAX), 1, [2, 2, 4]),
            (1, None, None, 1, [2, 3, 4]),
            (1, Some(5), Some(9), 1, [2, 0, 4]),
            (1, Some(2), Some(1), 1, [2, 0, 4]),
            (1, Some(-9), Some(2), 1, [2, 2, 4]),
            (1, Some(0), Some(3), 2, [2, 2, 4]),
            (1, Some(1), Some(3), 2, [2, 1, 4]),
            (1, Some(0), Some(3), 5, [2, 1, 4]),
            (-1, Some(1), Some(-1), 1, [2, 3, 2]),
        ];
        for (dim, start, end, step, expected) in cases {
            assert_eq!(
                ok(slice_shape(&shape, dim, start, end, step)),
                expected,
                "slice(dim={dim}, start={start:?}, end={end:?}, step={step})"
            );
        }
        for step in [0, -1] {
            assert_eq!(
                slice_shape(&shape, 1, Some(0), Some(3), step),
                invalid("slice step must be positive")
            );
        }
        assert_eq!(
            slice_shape(RANK0, 0, None, None, 1),
            invalid("dim out of range")
        );
    }

    #[test]
    fn cat_shape_matches_torch_cat() {
        assert_eq!(ok(cat_shape(&[&[2, 3], &[2, 5]], -1)), vec![2, 8]);
        assert_eq!(ok(cat_shape(&[&[0], &[2, 3]], 1)), vec![2, 3]);
        assert_eq!(ok(cat_shape(&[&[0], &[0]], 0)), vec![0]);
        assert_eq!(ok(cat_shape(&[&[2, 0], &[2, 3]], 1)), vec![2, 3]);
        assert_eq!(
            cat_shape(&[&[2, 3], &[3, 5]], 1),
            incompatible("cat inputs must agree off the cat dim")
        );
        assert_eq!(
            cat_shape(&[&[2, 3], &[2, 3, 1]], 1),
            incompatible("cat inputs must share a rank")
        );
        assert_eq!(
            cat_shape(&[RANK0, &[2]], 0),
            invalid("cat inputs must have rank 1 or more")
        );
    }

    /// `torch.promote_types` for every pair of these dtypes, from torch 2.15.
    #[rustfmt::skip]
    const PROMOTE_TYPES: [[&str; 10]; 10] = [
        ["bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        ["uint8", "uint8", "int16", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        ["int8", "int16", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        ["int16", "int16", "int16", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        ["int32", "int32", "int32", "int32", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        ["int64", "int64", "int64", "int64", "int64", "int64", "float16", "bfloat16", "float32", "float64"],
        ["float16", "float16", "float16", "float16", "float16", "float16", "float16", "float32", "float32", "float64"],
        ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "bfloat16", "float32", "float64"],
        ["float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float64"],
        ["float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64"],
    ];

    #[test]
    fn promote_dtypes_matches_torch_promote_types() {
        let dtypes = PROMOTE_TYPES[0];
        for (row, lhs) in dtypes.iter().enumerate() {
            for (column, rhs) in dtypes.iter().enumerate() {
                assert_eq!(
                    promote_dtypes(&format!("torch.{lhs}"), &format!("torch.{rhs}")),
                    Some(format!("torch.{}", PROMOTE_TYPES[row][column])),
                    "promote_types({lhs}, {rhs})"
                );
            }
        }
        let fp8 = "torch.float8_e4m3fn";
        assert_eq!(promote_dtypes(fp8, fp8), Some(fp8.to_string()));
        assert_eq!(promote_dtypes(fp8, F32), None);
    }

    // Node refresh, on graphs built with the layout test DSL.

    #[test]
    fn copies_take_shape_and_dtype_from_their_input() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3, 4], I8);
        let permute = g.permute(x, &[2, 0, 1]);
        let view = g.view(permute, &[-1, 6]);
        let one = g.input("one", &[1, 1]);
        let squeezed = g.view(one, RANK0);
        assert_eq!((g.shape(permute), g.dtype(permute)), (&[4, 2, 3][..], I8));
        assert_eq!((g.shape(view), g.dtype(view)), (&[4, 6][..], I8));
        assert_eq!(g.shape(squeezed), RANK0);
    }

    /// FX import turns an empty list into `IrArg::None`.
    #[test]
    fn rank0_copies_accept_none_for_their_empty_list() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", RANK0);
        let one = g.input("one", &[1, 1]);
        let permute = g.call(PERMUTE, vec![node(x), IrArg::None]);
        let view = g.call(VIEW, vec![node(one), IrArg::None]);
        assert_eq!(g.shape(permute), RANK0);
        assert_eq!(g.shape(view), RANK0);
    }

    #[test]
    fn pointwise_ops_broadcast_their_operands() {
        let mut g = LayoutGraph::new();
        let a = g.input("a", &[1, 3, 1]);
        let b = g.input("b", &[2, 1, 4]);
        let hi = g.input("hi", &[4]);
        let add = g.call(ADD, vec![node(a), node(b)]);
        let relu = g.call(RELU, vec![node(add)]);
        let clamp = g.call_with_kwargs(CLAMP_TENSOR, vec![node(relu)], &[("max", node(hi))]);
        assert_eq!((g.shape(add), g.dtype(add)), (&[2, 3, 4][..], F32));
        assert_eq!(g.shape(relu), &[2, 3, 4]);
        assert_eq!(g.shape(clamp), &[2, 3, 4]);
        g.assert_targets(&[ADD, RELU, CLAMP_TENSOR]);
    }

    #[test]
    fn where_takes_its_dtype_from_the_values_and_its_shape_from_all_operands() {
        let mut g = LayoutGraph::new();
        let condition = g.typed_input("condition", &[3, 1], BOOL);
        let lhs = g.typed_input("lhs", &[1, 4], I8);
        let rhs = g.typed_input("rhs", &[3, 4], I16);
        let select = g.call(WHERE, vec![node(condition), node(lhs), node(rhs)]);
        assert_eq!((g.shape(select), g.dtype(select)), (&[3, 4][..], I16));
    }

    #[test]
    fn masked_fill_keeps_self_dtype_and_broadcasts_the_mask() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 3], I8);
        let mask = g.typed_input("mask", &[2, 3], BOOL);
        let filled = g.call(MASKED_FILL, vec![node(x), node(mask), IrArg::Int(0)]);
        assert_eq!((g.shape(filled), g.dtype(filled)), (&[2, 3][..], I8));
    }

    #[test]
    fn comparisons_and_logical_ops_give_bool() {
        let mut g = LayoutGraph::new();
        let a = g.typed_input("a", &[2], I8);
        let b = g.typed_input("b", &[2], I8);
        for &target in BOOL_VALUED_POINTWISE {
            assert!(is_pointwise(target), "{target}");
            let args = if target.ends_with(".Scalar") {
                vec![node(a), IrArg::Int(0)]
            } else if matches!(
                target,
                "aten.isinf.default" | "aten.isnan.default" | "aten.logical_not.default"
            ) {
                vec![node(a)]
            } else {
                vec![node(a), node(b)]
            };
            let result = g.call(target, args);
            assert_eq!(g.dtype(result), BOOL, "{target}");
        }
    }

    /// RETAB keeps quantized table ops as aten nodes with integer vals, so a
    /// unary op keeps its input dtype where torch would promote to float.
    #[test]
    fn unary_ops_keep_their_input_dtype() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3], I8);
        let sigmoid = g.call(SIGMOID, vec![node(x)]);
        let abs = g.call(ABS, vec![node(sigmoid)]);
        assert_eq!(g.dtype(sigmoid), I8);
        assert_eq!(g.dtype(abs), I8);
    }

    /// Expected dtypes checked against torch 2.15 `torch.mul`.
    #[test]
    fn rank0_operands_promote_only_from_a_higher_category() {
        let mut g = LayoutGraph::new();
        let half = g.typed_input("half", &[2, 3], F16);
        let wide = g.typed_input("wide", RANK0, F32);
        let small = g.typed_input("small", &[2, 3], I8);
        let flag = g.typed_input("flag", &[2, 3], BOOL);
        let int_scalar = g.typed_input("int_scalar", RANK0, I32);
        let same_category = g.call(MUL, vec![node(half), node(wide)]);
        let higher_category = g.call(MUL, vec![node(small), node(wide)]);
        let lower_category = g.call(MUL, vec![node(small), node(int_scalar)]);
        let from_bool = g.call(MUL, vec![node(flag), node(int_scalar)]);
        let both_rank0 = g.call(MUL, vec![node(wide), node(int_scalar)]);
        assert_eq!(g.dtype(same_category), F16);
        assert_eq!(g.dtype(higher_category), F32);
        assert_eq!(g.dtype(lower_category), I8);
        assert_eq!(g.dtype(from_bool), I32);
        assert_eq!((g.shape(both_rank0), g.dtype(both_rank0)), (RANK0, F32));
    }

    #[test]
    fn in_place_variants_keep_self() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let row = g.input("row", &[3]);
        let add = g.call(ADD_INPLACE, vec![node(x), node(row)]);
        assert_eq!(g.shape(add), &[2, 3]);

        let narrow = g.input("narrow", &[1, 3]);
        let id = g.add_unrefreshed(ADD_INPLACE, vec![node(narrow), node(x)]);
        assert_eq!(
            infer_facts(g.graph(), id),
            Err(ShapeError::Incompatible(
                "an in-place op cannot broadcast past self"
            ))
        );
        assert!(!is_in_place("aten.__lshift__.Scalar"));
    }

    #[test]
    fn reductions_follow_keepdim_and_the_producer_dtype() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let mean_keep = g.call(MEAN, vec![node(x), ints(&[2, 3]), IrArg::Bool(true)]);
        let mean_drop = g.call(MEAN, vec![node(x), ints(&[-1]), IrArg::Bool(false)]);
        let mean_all = g.call(MEAN, vec![node(x), IrArg::None]);
        assert_eq!(g.shape(mean_keep), &[1, 2, 1, 1]);
        assert_eq!(g.shape(mean_drop), &[1, 2, 3]);
        assert_eq!(g.shape(mean_all), RANK0);

        let acc = g.typed_input("acc", &[2, 3], I32);
        let sum = g.call(SUM, vec![node(acc), ints(&[1])]);
        let sum_keep = g.call_with_kwargs(
            SUM,
            vec![node(acc), IrArg::Int(0)],
            &[
                ("keepdim", IrArg::Bool(true)),
                ("dtype", IrArg::Dtype("torch.int64".into())),
            ],
        );
        // torch would give int64; RETAB's int32 accumulators keep int32.
        assert_eq!((g.shape(sum), g.dtype(sum)), (&[2][..], I32));
        assert_eq!(
            (g.shape(sum_keep), g.dtype(sum_keep)),
            (&[1, 3][..], "torch.int64")
        );
    }

    #[test]
    fn slices_follow_torch_bounds() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3, 4]);
        let tail = g.slice(x, 1, -2, i64::MAX);
        let strided = g.call(
            SLICE,
            vec![
                node(x),
                IrArg::Int(2),
                IrArg::None,
                IrArg::None,
                IrArg::Int(3),
            ],
        );
        let whole = g.call(SLICE, vec![node(x)]);
        assert_eq!(g.shape(tail), &[2, 2, 4]);
        assert_eq!(g.shape(strided), &[2, 3, 2]);
        assert_eq!(g.shape(whole), &[2, 3, 4]);
    }

    #[test]
    fn cat_sums_the_cat_dim_and_promotes_dtypes() {
        let mut g = LayoutGraph::new();
        let a = g.typed_input("a", &[2, 3], I8);
        let b = g.typed_input("b", &[2, 5], I16);
        let cat = g.cat(&[a, b], -1);
        assert_eq!((g.shape(cat), g.dtype(cat)), (&[2, 8][..], I16));
    }

    #[test]
    fn rescale_keeps_shape_takes_its_dtype_argument_and_keeps_qparams() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3], I32);
        let rescale = g.rescale(x, I8, 0.5);
        g.graph_mut()
            .node_mut(rescale)
            .meta
            .set_single_output_qparam(qparam(0.5, 0, I8));
        g.refresh(rescale);
        assert_eq!((g.shape(rescale), g.dtype(rescale)), (&[2, 3][..], I8));
        assert!(g.graph().node(rescale).meta.output_qparam().is_some());
    }

    #[test]
    fn transparent_copies_keep_shape_and_take_a_dtype_kwarg() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[2, 3]);
        let clone = g.call(CLONE, vec![node(x)]);
        let cast = g.call_with_kwargs(
            TO_COPY,
            vec![node(clone)],
            &[("dtype", IrArg::Dtype(I32.into()))],
        );
        assert_eq!((g.shape(clone), g.dtype(clone)), (&[2, 3][..], F32));
        assert_eq!((g.shape(cast), g.dtype(cast)), (&[2, 3][..], I32));
    }

    #[test]
    fn a_node_without_inputs_keeps_its_meta() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[2, 3], I8);
        let meta = refreshed_meta(g.graph(), x).unwrap();
        assert_eq!(meta.val_shape(), Some(&[2, 3][..]));
        assert_eq!(meta.val_dtype(), Some(I8));
    }

    #[test]
    fn unsupported_targets_and_bad_arguments_are_errors() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let conv = g.add_unrefreshed("aten.convolution.default", vec![node(x)]);
        let bad_permute = g.add_unrefreshed(PERMUTE, vec![node(x)]);
        let bad_rescale = g.add_unrefreshed(RESCALE, vec![node(x), IrArg::Int(8)]);
        let orphan = g.add_unrefreshed(RELU, vec![node(conv)]);
        let graph = g.graph();
        assert_eq!(
            infer_facts(graph, conv),
            Err(ShapeError::Unsupported("aten.convolution.default".into()))
        );
        assert!(!has_local_rule("aten.convolution.default"));
        assert!(has_local_rule(RESCALE) && has_local_rule(CLONE) && has_local_rule(WHERE));
        assert_eq!(
            infer_facts(graph, bad_permute),
            Err(ShapeError::InvalidArg("permute dims"))
        );
        assert_eq!(
            infer_facts(graph, bad_rescale),
            Err(ShapeError::InvalidArg("RESCALE dtype"))
        );
        assert_eq!(
            infer_facts(graph, orphan),
            Err(ShapeError::MissingOperand("operand"))
        );
        assert!(refreshed_meta(graph, conv).is_err());
    }

    #[test]
    fn planners_can_refresh_a_hypothetical_node() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let relu = g.call(RELU, vec![node(x)]);
        // What would `relu` be if it read a permuted operand?
        let permuted = TensorFacts::new(vec![1, 3, 4, 2], F32);
        let planned =
            infer_node_facts(g.graph().node(relu), &|id| (id == x).then_some(&permuted)).unwrap();
        assert_eq!(planned.shape, vec![1, 3, 4, 2]);
        assert_eq!(g.shape(relu), &[1, 2, 3, 4]);
    }

    #[test]
    fn layout_transforms_are_rc4s_moved_targets() {
        assert!(is_layout_transform(PERMUTE));
        assert!(is_layout_transform(VIEW));
        assert!(is_layout_transform("aten.view.default"));
        assert!(!is_layout_transform(SLICE));
        assert!(!is_layout_transform(CLONE));
    }

    /// Port of rc4's `test_down_propagation_refreshes_metadata_in_dependency_order`
    /// (`backends/arm/test/passes/test_propagate_permutes_views_pass.py`): after
    /// the down pass moves a permute past two pointwise ops, refreshing the
    /// crossed nodes and then the permute gives the moved shapes.
    #[test]
    fn down_move_refreshes_crossed_nodes_then_the_permute() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let permute = g.permute(x, &[0, 2, 3, 1]);
        let relu = g.call(RELU, vec![node(permute)]);
        let neg = g.call(NEG, vec![node(relu)]);
        let out = g.output(&[neg]);

        g.rewire(relu, permute, x);
        g.rewire(permute, x, neg);
        g.rewire(out, neg, permute);
        for id in [relu, neg, permute] {
            g.refresh(id);
        }

        g.assert_targets(&[RELU, NEG, PERMUTE]);
        assert_eq!(g.shape(relu), &[1, 2, 3, 4]);
        assert_eq!(g.shape(neg), &[1, 2, 3, 4]);
        assert_eq!(g.shape(permute), &[1, 3, 4, 2]);
        g.assert_permute_dims(&[&[0, 2, 3, 1]]);
    }

    /// Port of rc4's `test_up_propagation_refreshes_metadata_in_dependency_order`
    /// (`backends/arm/test/passes/test_propagate_permutes_views_pass.py`): the
    /// permute first, then the crossed nodes in reverse path order.
    #[test]
    fn up_move_refreshes_the_permute_then_crossed_nodes() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let relu = g.call(RELU, vec![node(x)]);
        let neg = g.call(NEG, vec![node(relu)]);
        let permute = g.permute(neg, &[0, 2, 3, 1]);
        let out = g.output(&[permute]);

        g.rewire(permute, neg, x);
        g.rewire(relu, x, permute);
        g.rewire(out, permute, neg);
        for id in [permute, relu, neg] {
            g.refresh(id);
        }

        g.assert_targets(&[PERMUTE, RELU, NEG]);
        assert_eq!(g.shape(permute), &[1, 3, 4, 2]);
        assert_eq!(g.shape(relu), &[1, 3, 4, 2]);
        assert_eq!(g.shape(neg), &[1, 3, 4, 2]);
    }

    /// Port of rc4's `test_up_pass_refreshes_permute_meta_before_view_slice_swap`
    /// (`backends/arm/test/passes/test_propagate_permutes_views_pass.py`): the
    /// graph the up pass leaves, `x -> permute -> view -> slice`, gets rc4's shapes.
    #[test]
    fn view_slice_swap_refreshes_to_rc4s_shapes() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[3, 2, 8, 16]);
        let permute = g.permute(x, &[0, 3, 1, 2]);
        let view = g.view(permute, &[3, 32, 8]);
        let slice = g.slice(view, 0, 0, 1);
        g.output(&[slice]);

        assert_eq!(g.shape(permute), &[3, 16, 2, 8]);
        assert_eq!(g.shape(view), &[3, 32, 8]);
        assert_eq!(g.shape(slice), &[1, 32, 8]);
        g.assert_counts(1, 1);
    }

    /// Port of rc4's `test_propagate_scalar_view_down_through_unary_elementwise`
    /// (`backends/arm/test/passes/test_propagate_permutes_views_pass.py`).
    #[test]
    fn scalar_view_moved_below_a_unary_op() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", RANK0);
        let view = g.view(x, &[1]);
        let relu = g.call(RELU, vec![node(view)]);
        let out = g.output(&[relu]);

        g.rewire(relu, view, x);
        g.rewire(view, x, relu);
        g.rewire(out, relu, view);
        for id in [relu, view] {
            g.refresh(id);
        }

        g.assert_targets(&[RELU, VIEW]);
        assert_eq!(g.shape(relu), RANK0);
        assert_eq!(g.shape(view), &[1]);
    }

    /// Port of rc4's `test_moves_permute_after_narrowing_rescale`
    /// (`backends/arm/test/passes/test_move_data_movement_ops_to_smaller_dtype_pass.py`),
    /// rewired as `_move_after_rescale` does
    /// (`backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py:120-137`).
    /// The moved permute also drops its stale qparams; the RESCALE keeps its own.
    #[test]
    fn permute_moved_after_a_narrowing_rescale() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 2, 3, 4], I32);
        let permute = g.permute(x, &[0, 2, 3, 1]);
        let rescale = g.rescale(permute, I8, 1.0);
        let out = g.output(&[rescale]);
        g.graph_mut()
            .node_mut(permute)
            .meta
            .set_both_qparams(qparam(0.25, 0, I32));
        g.graph_mut()
            .node_mut(rescale)
            .meta
            .set_single_output_qparam(qparam(0.5, 0, I8));

        g.rewire(rescale, permute, x);
        g.rewire(permute, x, rescale);
        g.rewire(out, rescale, permute);
        for id in [rescale, permute] {
            g.refresh(id);
        }

        g.assert_targets(&[RESCALE, PERMUTE]);
        assert_eq!(
            (g.shape(rescale), g.dtype(rescale)),
            (&[1, 2, 3, 4][..], I8)
        );
        assert_eq!(
            (g.shape(permute), g.dtype(permute)),
            (&[1, 3, 4, 2][..], I8)
        );
        assert!(!g.graph().node(permute).meta.has_qparam_metadata());
        assert!(g.graph().node(rescale).meta.output_qparam().is_some());
    }

    /// Port of rc4's `test_moves_permute_before_widening_rescale`
    /// (`backends/arm/test/passes/test_move_data_movement_ops_to_smaller_dtype_pass.py`),
    /// rewired as `_move_before_rescale` does
    /// (`backends/arm/_passes/move_data_movement_ops_to_smaller_dtype_pass.py:100-118`).
    #[test]
    fn permute_moved_before_a_widening_rescale() {
        let mut g = LayoutGraph::new();
        let x = g.typed_input("x", &[1, 2, 3, 4], I8);
        let rescale = g.rescale(x, I32, 1.0);
        let permute = g.permute(rescale, &[0, 2, 3, 1]);
        let out = g.output(&[permute]);

        g.rewire(permute, rescale, x);
        g.rewire(rescale, x, permute);
        g.rewire(out, permute, rescale);
        for id in [permute, rescale] {
            g.refresh(id);
        }

        g.assert_targets(&[PERMUTE, RESCALE]);
        assert_eq!(
            (g.shape(permute), g.dtype(permute)),
            (&[1, 3, 4, 2][..], I8)
        );
        assert_eq!(
            (g.shape(rescale), g.dtype(rescale)),
            (&[1, 3, 4, 2][..], I32)
        );
    }

    #[test]
    fn dsl_names_nodes_like_fx_and_counts_transforms() {
        let mut g = LayoutGraph::new();
        let x = g.input("x", &[1, 2, 3, 4]);
        let first = g.permute(x, &[0, 2, 3, 1]);
        let second = g.permute(first, &[0, 3, 1, 2]);
        let flat = g.view(second, &[1, -1]);
        g.output(&[flat]);
        assert_eq!(g.graph().node(first).name, "permute_copy");
        assert_eq!(g.graph().node(second).name, "permute_copy_1");
        assert_eq!(g.graph().node(flat).op, NodeOp::CallFunction);
        assert_eq!(g.count(PERMUTE), 2);
        g.assert_counts(2, 1);
        g.assert_permute_dims(&[&[0, 2, 3, 1], &[0, 3, 1, 2]]);
        assert_eq!(g.shape(flat), &[1, 24]);
    }
}
