//! Which dims correspond across a `view_copy` or a `permute_copy`, so layout passes
//! can move views and permutes past reductions, slices and each other: rc4's
//! `backends/transforms/dim_maps.py`. Line references below are to that file.
//!
//! Integer dims only, since RETAB hands symbolic shapes to stock. The SymInt helpers
//! `_dim_expr`, `_simplify_dim` and `_simplify_shape` (:104-117) have no counterpart,
//! `_dim_equals` (:120-125) is `==`, `_factor_dim` (:143-149) is `factor_int`, and
//! `numel` (:172-176) is `fx_ir::shape_product`.
//!
//! Failures follow Python: `None` where Python returns `None`, `Err` where it raises
//! (`DimMapError` names the exception), and `Result<Option<_>>` where a method can do
//! either. Callers that mirror rc4's `except AssertionError` match on the variant.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::fx_ir::{normalize_dim as checked_dim, shape_product};

/// An exception rc4's dim maps raise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimMapError {
    /// `AssertionError` in `_normalize_dim` (:40-43).
    InvalidDim { dim: i64, rank: usize },
    /// `AssertionError` in `normalize_view_shape` (:195): more than one `-1` dim.
    InvalidViewShape(Vec<i64>),
    /// `ZeroDivisionError` in `normalize_view_shape` (:202-204): a `-1` dim beside a
    /// zero dim.
    ZeroDivision(Vec<i64>),
    /// `AssertionError` in `ViewMap._valid_groups` (:1004-1007): the method needs a
    /// valid map.
    InvalidMap,
    /// `ValueError` in `PermuteMap.__init__` (:1021-1022).
    InvalidPermuteDims(Vec<i64>),
}

/// `ViewMap.map_reduction_after_view`'s result: the view shape and reduction dims
/// once the reduction follows the view.
pub type ReductionAfterView = (Vec<i64>, Vec<usize>);

/// `ViewMap.map_reduction_before_view`'s result: the reduction dims and output view
/// shape once the reduction precedes the view.
pub type ReductionBeforeView = (Vec<usize>, Vec<i64>);

/// `ViewMap.remap_unit_slice`'s result: the view shape, then the slice's dim, start
/// and end once the slice follows the view.
pub type UnitSliceAfterView = (Vec<i64>, usize, i64, i64);

/// `_is_permutation` (:36-37).
pub fn is_permutation(dims: &[usize], rank: usize) -> bool {
    let mut sorted = dims.to_vec();
    sorted.sort_unstable();
    sorted.into_iter().eq(0..rank)
}

/// `_normalize_dim` (:40-43).
pub fn normalize_dim(dim: i64, rank: usize) -> Result<usize, DimMapError> {
    checked_dim(dim, rank).ok_or(DimMapError::InvalidDim { dim, rank })
}

/// `_normalize_dims` (:46-49). Python also takes a bare int; pass `&[dim]`.
pub fn normalize_dims(dims: &[i64], rank: usize) -> Result<Vec<usize>, DimMapError> {
    dims.iter().map(|&dim| normalize_dim(dim, rank)).collect()
}

/// `_normalize_permutation` (:52-59).
fn normalize_permutation(dims: &[i64], rank: usize) -> Option<Vec<usize>> {
    if dims.len() != rank {
        return None;
    }
    let normalized = normalize_dims(dims, rank).ok()?;
    is_permutation(&normalized, rank).then_some(normalized)
}

/// `_extend_permutation_with_singletons` (:62-78): `extend_permutation`, provided every
/// axis it adds is a singleton in `shape`. Callers pass axes, so Python's
/// normalization is the range check in `extend_permutation`.
fn extend_permutation_with_singletons(dims: &[usize], shape: &[i64]) -> Option<Vec<usize>> {
    let extended = extend_permutation(dims, shape.len())?;
    (0..shape.len())
        .filter(|axis| !dims.contains(axis))
        .all(|axis| shape[axis] == 1)
        .then_some(extended)
}

/// `_extend_permutation` (:81-101): complete a partial permutation by inserting each
/// missing axis, highest first, before the first present axis greater than it.
fn extend_permutation(dims: &[usize], rank: usize) -> Option<Vec<usize>> {
    if dims.iter().any(|&dim| dim >= rank) || has_duplicates(dims) {
        return None;
    }
    let missing: Vec<usize> = (0..rank).filter(|dim| !dims.contains(dim)).collect();
    let mut extended = dims.to_vec();
    for &dim in missing.iter().rev() {
        let insert_at = extended
            .iter()
            .position(|&existing| existing > dim)
            .unwrap_or(extended.len());
        extended.insert(insert_at, dim);
    }
    is_permutation(&extended, rank).then_some(extended)
}

fn has_duplicates(dims: &[usize]) -> bool {
    dims.iter().collect::<HashSet<_>>().len() != dims.len()
}

/// `_factor_int` (:128-140): prime factors in increasing order; `None` below 1.
fn factor_int(mut dim: i64) -> Option<Vec<i64>> {
    if dim < 1 {
        return None;
    }
    let mut factors = Vec::new();
    let mut divisor = 2;
    // `divisor * divisor <= dim`, without overflow.
    while divisor <= dim / divisor {
        while dim % divisor == 0 {
            factors.push(divisor);
            dim /= divisor;
        }
        divisor += if divisor == 2 { 1 } else { 2 };
    }
    if dim > 1 {
        factors.push(dim);
    }
    Some(factors)
}

/// `_Factor` (:22-25). On integer dims the factor key is the prime.
#[derive(Debug, Clone, Copy)]
struct Factor {
    prime: i64,
    axis: usize,
}

/// `_factor_shape` (:152-159).
fn factor_shape(shape: &[i64]) -> Option<Vec<Factor>> {
    let mut factors = Vec::new();
    for (axis, &dim) in shape.iter().enumerate() {
        factors.extend(
            factor_int(dim)?
                .into_iter()
                .map(|prime| Factor { prime, axis }),
        );
    }
    Some(factors)
}

/// `_dedupe` (:162-169): first occurrences, in order.
fn dedupe(items: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut seen = HashSet::new();
    items
        .into_iter()
        .filter(|&item| seen.insert(item))
        .collect()
}

/// `same_numel` (:179-180).
pub fn same_numel(first_shape: &[i64], second_shape: &[i64]) -> bool {
    shape_product(first_shape) == shape_product(second_shape)
}

/// `normalize_view_shape` (:183-205): infer a single `-1` dim from the source numel.
pub fn normalize_view_shape(
    source_shape: &[i64],
    target_shape: &[i64],
) -> Result<Vec<i64>, DimMapError> {
    let inferred: Vec<usize> = (0..target_shape.len())
        .filter(|&index| target_shape[index] == -1)
        .collect();
    let mut normalized = target_shape.to_vec();
    match inferred[..] {
        [] => {}
        [inferred_dim] => {
            let source_numel = shape_product(source_shape);
            let known_numel: i64 = (0..target_shape.len())
                .filter(|&index| index != inferred_dim)
                .map(|index| target_shape[index])
                .product();
            normalized[inferred_dim] = if known_numel == 1 {
                source_numel
            } else {
                floor_div(source_numel, known_numel)
                    .ok_or_else(|| DimMapError::ZeroDivision(target_shape.to_vec()))?
            };
        }
        _ => return Err(DimMapError::InvalidViewShape(target_shape.to_vec())),
    }
    Ok(normalized)
}

/// Python's `//`: rounds toward negative infinity; `None` for a zero divisor.
fn floor_div(lhs: i64, rhs: i64) -> Option<i64> {
    if rhs == 0 {
        return None;
    }
    let quotient = lhs.wrapping_div(rhs);
    let inexact = lhs.wrapping_rem(rhs) != 0;
    Some(if inexact && (lhs < 0) != (rhs < 0) {
        quotient - 1
    } else {
        quotient
    })
}

/// `_UnionFind` (:208-222).
struct UnionFind {
    parents: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parents: (0..size).collect(),
        }
    }

    fn find(&mut self, item: usize) -> usize {
        let parent = self.parents[item];
        if parent != item {
            let root = self.find(parent);
            self.parents[item] = root;
        }
        self.parents[item]
    }

    fn union(&mut self, first: usize, second: usize) {
        let first_root = self.find(first);
        let second_root = self.find(second);
        if first_root != second_root {
            self.parents[second_root] = first_root;
        }
    }
}

/// `_ViewGroups` (:28-33). A group is a union-find root over source factor positions.
/// `_build_groups` unions every factor of an axis, so an axis has at most one group
/// where Python keeps a list of length 0 or 1; singleton axes have none.
#[derive(Debug, Clone)]
struct ViewGroups {
    source_axis_to_group: Vec<Option<usize>>,
    target_axis_to_group: Vec<Option<usize>>,
    /// Indexed by group; empty at factor positions that aren't roots.
    group_to_source_axes: Vec<Vec<usize>>,
    group_to_target_axes: Vec<Vec<usize>>,
}

/// rc4 `ViewMap` (:225-1007). It expands both shapes into ordered prime-factor
/// streams, matches them, and groups factors that share a source axis, share a target
/// axis, or cross in the matching. A source dim maps to every target dim sharing a
/// group with it; singleton dims belong to no group. For [4, 3, 10] -> [2, 2, 1, 5, 3,
/// 2] the groups pair source [0] with target [0, 1] and source [1, 2] with target
/// [3, 4, 5].
#[derive(Debug, Clone)]
pub struct ViewMap {
    source_shape: Vec<i64>,
    target_shape: Vec<i64>,
    groups: Option<ViewGroups>,
}

impl ViewMap {
    /// `ViewMap.from_shapes` (:273-286). `ViewMap(view_node)` (:258-271) is this on the
    /// view input's shape and `view_node.args[1]`.
    pub fn from_shapes(source_shape: &[i64], target_shape: &[i64]) -> Result<Self, DimMapError> {
        let target_shape = normalize_view_shape(source_shape, target_shape)?;
        let groups = build_groups(source_shape, &target_shape);
        Ok(Self {
            source_shape: source_shape.to_vec(),
            target_shape,
            groups,
        })
    }

    /// `ViewMap.is_valid_map` (:288-291).
    pub fn is_valid_map(&self) -> bool {
        self.groups.is_some()
    }

    pub fn source_shape(&self) -> &[i64] {
        &self.source_shape
    }

    /// The view shape with any `-1` inferred.
    pub fn target_shape(&self) -> &[i64] {
        &self.target_shape
    }

    /// `ViewMap.source_rank` (:293-296).
    pub fn source_rank(&self) -> usize {
        self.source_shape.len()
    }

    /// `ViewMap.target_rank` (:298-301).
    pub fn target_rank(&self) -> usize {
        self.target_shape.len()
    }

    /// `ViewMap.map_dim` (:303-348): target dims for a reduction over `source_dims`,
    /// `x.op(dims).view(S) == x.view(S').op(mapped_dims)`. Both sides must cover whole
    /// groups; singleton axes may come along.
    pub fn map_dim(&self, source_dims: &[i64]) -> Result<Option<Vec<usize>>, DimMapError> {
        let Ok(source_dims) = normalize_dims(source_dims, self.source_rank()) else {
            return Ok(None);
        };
        let groups = self.valid_groups()?;
        if !is_valid_reduction_or_singleton(&source_dims, &groups.source_axis_to_group) {
            return Ok(None);
        }
        let source_to_target_axes = self.source_to_target_axes()?;
        let mut target_dims: Vec<usize> = source_dims
            .iter()
            .flat_map(|&dim| source_to_target_axes[dim].iter().copied())
            .collect();
        target_dims.sort_unstable();
        target_dims.dedup();
        let covers_their_sources = target_dims.iter().all(|&target_axis| {
            Self::source_axes_for_target_axis(target_axis, &source_to_target_axes)
                .iter()
                .all(|source_axis| source_dims.contains(source_axis))
        });
        if target_dims.is_empty()
            || !covers_their_sources
            || !is_valid_reduction_or_singleton(&target_dims, &groups.target_axis_to_group)
        {
            return Ok(None);
        }
        Ok(Some(target_dims))
    }

    /// `ViewMap.map_dim_inverse` (:350-378): source dims for a reduction over
    /// `target_dims`, `x.view(S).op(dims) == x.op(mapped_dims).view(S')`.
    pub fn map_dim_inverse(&self, target_dims: &[i64]) -> Result<Option<Vec<usize>>, DimMapError> {
        let Ok(target_dims) = normalize_dims(target_dims, self.target_rank()) else {
            return Ok(None);
        };
        let groups = self.valid_groups()?;
        if !is_valid_reduction(&target_dims, &groups.target_axis_to_group) {
            return Ok(None);
        }
        let source_dims = map_dims(
            &target_dims,
            &groups.target_axis_to_group,
            &groups.group_to_source_axes,
        );
        if source_dims.is_empty() || !is_valid_reduction(&source_dims, &groups.source_axis_to_group)
        {
            return Ok(None);
        }
        Ok(Some(source_dims))
    }

    /// `ViewMap.map_reduction_after_view` (:380-400): the view shape and dims for
    /// `view(reduce(x, source_dims), target_shape) == reduce(view(x, shape), dims)`,
    /// where `source_shape` is the shape of `x`.
    pub fn map_reduction_after_view(
        &self,
        source_shape: &[i64],
        source_dims: &[i64],
    ) -> Option<ReductionAfterView> {
        let target_shape = self.remap_target_shape(source_shape)?;
        // `remap_target_shape` only succeeds on a valid map, so `map_dim` can't raise.
        let target_dims = self.map_dim(source_dims).ok()??;
        is_contiguous_nonempty(&target_dims).then_some((target_shape, target_dims))
    }

    /// `ViewMap.map_reduction_before_view` (:402-425): the dims and output view shape
    /// for `reduce(view(x, target_shape), target_dims) == view(reduce(x, dims), shape)`.
    pub fn map_reduction_before_view(
        &self,
        target_dims: &[i64],
    ) -> Result<Option<ReductionBeforeView>, DimMapError> {
        let Some(source_dims) = self.map_dim_inverse(target_dims)? else {
            return Ok(None);
        };
        if !is_contiguous_nonempty(&source_dims) {
            return Ok(None);
        }
        let Ok(target_dims) = normalize_dims(target_dims, self.target_rank()) else {
            return Ok(None);
        };
        Ok(Some((
            source_dims,
            reduce_shape(&self.target_shape, &target_dims),
        )))
    }

    /// `ViewMap.map_permutation` (:427-469): the target permutation equivalent to
    /// `source_permutation` across the view. Each group must stay one contiguous,
    /// increasing block on both sides, and target axes it can't place must be
    /// singletons.
    pub fn map_permutation(
        &self,
        source_permutation: &[i64],
    ) -> Result<Option<Vec<usize>>, DimMapError> {
        let Some(source_permutation) =
            normalize_permutation(source_permutation, self.source_rank())
        else {
            return Ok(None);
        };
        let groups = self.valid_groups()?;
        let mapped = map_dims(
            &source_permutation,
            &groups.source_axis_to_group,
            &groups.group_to_target_axes,
        );
        Ok(
            extend_permutation_with_singletons(&mapped, &self.target_shape).filter(
                |target_permutation| {
                    matching_permuted_group_blocks(&source_permutation, target_permutation, groups)
                },
            ),
        )
    }

    /// `ViewMap.map_permutation_inverse` (:471-507): `map_permutation` from the target
    /// side.
    pub fn map_permutation_inverse(
        &self,
        target_permutation: &[i64],
    ) -> Result<Option<Vec<usize>>, DimMapError> {
        let Some(target_permutation) =
            normalize_permutation(target_permutation, self.target_rank())
        else {
            return Ok(None);
        };
        let groups = self.valid_groups()?;
        let mapped = map_dims(
            &target_permutation,
            &groups.target_axis_to_group,
            &groups.group_to_source_axes,
        );
        Ok(
            extend_permutation_with_singletons(&mapped, &self.source_shape).filter(
                |source_permutation| {
                    matching_permuted_group_blocks(source_permutation, &target_permutation, groups)
                },
            ),
        )
    }

    /// `ViewMap.remap_target_shape` (:509-540): the view shape for a source of shape
    /// `source_shape` (same rank as the map's source) that keeps this map's grouping.
    pub fn remap_target_shape(&self, source_shape: &[i64]) -> Option<Vec<i64>> {
        if !self.is_valid_map() || source_shape.len() != self.source_rank() {
            return None;
        }
        let source_to_target_axes = self.source_to_target_axes().ok()?;
        let target_to_source_axes: Vec<Vec<usize>> = (0..self.target_rank())
            .map(|target_axis| {
                Self::source_axes_for_target_axis(target_axis, &source_to_target_axes)
            })
            .collect();
        let mut target_shape = vec![1; self.target_rank()];
        for (source_axis, target_axes) in source_to_target_axes.iter().enumerate() {
            let updates = self.target_axis_updates_for_source_axis(
                source_shape,
                source_axis,
                target_axes,
                &target_to_source_axes,
            )?;
            for (target_axis, target_dim) in updates {
                target_shape[target_axis] = target_dim;
            }
        }
        (same_numel(source_shape, &target_shape)
            && !target_shape.contains(&0)
            && preserves_source_axis_order(source_shape, &source_to_target_axes))
        .then_some(target_shape)
    }

    /// `ViewMap._target_axis_updates_for_source_axis` (:542-572).
    fn target_axis_updates_for_source_axis(
        &self,
        source_shape: &[i64],
        source_axis: usize,
        target_axes: &[usize],
        target_to_source_axes: &[Vec<usize>],
    ) -> Option<Vec<(usize, i64)>> {
        match *target_axes {
            [] => Some(Vec::new()),
            [target_axis] => {
                let source_axes = &target_to_source_axes[target_axis];
                // Python indexes `source_axes[0]`; `target_axis` came from this source
                // axis, so the list is never empty.
                if source_axes.first() != Some(&source_axis) {
                    return Some(Vec::new());
                }
                let target_dim = source_axes.iter().map(|&axis| source_shape[axis]).product();
                Some(vec![(target_axis, target_dim)])
            }
            _ => {
                if target_axes
                    .iter()
                    .any(|&target_axis| target_to_source_axes[target_axis].len() > 1)
                {
                    return Some(Vec::new());
                }
                let target_dims: Vec<i64> = target_axes
                    .iter()
                    .map(|&target_axis| self.target_shape[target_axis])
                    .collect();
                let unchanged = || {
                    target_axes
                        .iter()
                        .copied()
                        .zip(target_dims.iter().copied())
                        .collect()
                };
                let target_numel = shape_product(&target_dims);
                if source_shape[source_axis] == self.source_shape[source_axis] {
                    Some(unchanged())
                } else if target_numel == 1 {
                    Some(vec![(target_axes[0], source_shape[source_axis])])
                } else if target_numel == self.source_shape[source_axis] {
                    Some(unchanged())
                } else {
                    None
                }
            }
        }
    }

    /// `ViewMap.remap_unit_slice` (:574-648): move the view before a unit slice whose
    /// singleton axis the view drops, `view(slice(x, dim, start, end), target_shape) ==
    /// slice(view(x, shape), new_dim, new_start, new_end)`, by folding the slice into
    /// the next target axis. `None` start or end is a non-int slice arg, which
    /// Python's isinstance guard rejects. Python ints don't overflow; where the i64
    /// index arithmetic would, this returns `None` (the caller's interval check would
    /// reject Python's result anyway).
    pub fn remap_unit_slice(
        &self,
        producer_shape: &[i64],
        slice_dim: i64,
        start: Option<i64>,
        end: Option<i64>,
        step: i64,
    ) -> Result<Option<UnitSliceAfterView>, DimMapError> {
        let (Some(start), Some(end)) = (start, end) else {
            return Ok(None);
        };
        if producer_shape.len() != self.source_rank()
            || step != 1
            || end.checked_sub(start) != Some(1)
        {
            return Ok(None);
        }
        let Ok(slice_dim) = normalize_dim(slice_dim, self.source_rank()) else {
            return Ok(None);
        };
        let source_to_target_axes = self.source_to_target_axes()?;
        if !source_to_target_axes[slice_dim].is_empty() {
            return Ok(None);
        }
        let prev_target_axes: Vec<usize> = source_to_target_axes[..slice_dim]
            .iter()
            .flatten()
            .copied()
            .collect();
        if prev_target_axes.is_empty() {
            return Ok(None);
        }
        let later_axes = &source_to_target_axes[slice_dim + 1..];
        let next_target_axes: Vec<usize> = later_axes.iter().flatten().copied().collect();
        let fold_axis = later_axes
            .iter()
            .filter_map(|target_axes| target_axes.first().copied())
            .find(|&target_axis| {
                prev_target_axes.iter().all(|&prev| prev <= target_axis)
                    && next_target_axes.iter().all(|&next| target_axis <= next)
            });
        let Some(fold_axis) = fold_axis else {
            return Ok(None);
        };
        let chunk = self.target_shape[fold_axis];
        let (Some(folded), Some(new_start), Some(new_end)) = (
            chunk.checked_mul(producer_shape[slice_dim]),
            start.checked_mul(chunk),
            end.checked_mul(chunk),
        ) else {
            return Ok(None);
        };
        let mut target_shape = self.target_shape.clone();
        target_shape[fold_axis] = folded;
        Ok(Some((target_shape, fold_axis, new_start, new_end)))
    }

    /// `ViewMap.source_to_target_axes` (:650-662): the target axes each source axis
    /// maps to, with leftover singleton axes paired up.
    pub fn source_to_target_axes(&self) -> Result<Vec<Vec<usize>>, DimMapError> {
        let groups = self.valid_groups()?;
        let mut source_to_target_axes: Vec<Vec<usize>> = (0..self.source_rank())
            .map(|source_axis| {
                map_dims(
                    &[source_axis],
                    &groups.source_axis_to_group,
                    &groups.group_to_target_axes,
                )
            })
            .collect();
        self.add_singleton_axes(&mut source_to_target_axes);
        Ok(source_to_target_axes)
    }

    /// `ViewMap.map_source_dims_to_target_axes` (:664-676).
    pub fn map_source_dims_to_target_axes(
        &self,
        source_dims: &[i64],
    ) -> Result<Option<Vec<usize>>, DimMapError> {
        let Ok(source_dims) = normalize_dims(source_dims, self.source_rank()) else {
            return Ok(None);
        };
        let source_to_target_axes = self.source_to_target_axes()?;
        Ok(Some(dedupe(source_dims.iter().flat_map(|&dim| {
            source_to_target_axes[dim].iter().copied()
        }))))
    }

    /// `ViewMap.source_axes_for_target_axis` (:678-686).
    pub fn source_axes_for_target_axis(
        target_axis: usize,
        source_to_target_axes: &[Vec<usize>],
    ) -> Vec<usize> {
        (0..source_to_target_axes.len())
            .filter(|&source_axis| source_to_target_axes[source_axis].contains(&target_axis))
            .collect()
    }

    /// `ViewMap._add_singleton_axes` (:688-720): pair unmapped source singletons with
    /// unmapped target singletons, one to one, or all to the single one on a side
    /// that has exactly one.
    fn add_singleton_axes(&self, source_to_target_axes: &mut [Vec<usize>]) {
        let mapped_target_axes: HashSet<usize> =
            source_to_target_axes.iter().flatten().copied().collect();
        let source_singletons: Vec<usize> = (0..self.source_rank())
            .filter(|&axis| source_to_target_axes[axis].is_empty() && self.source_shape[axis] == 1)
            .collect();
        let target_singletons: Vec<usize> = (0..self.target_rank())
            .filter(|&axis| !mapped_target_axes.contains(&axis) && self.target_shape[axis] == 1)
            .collect();
        let zip = |sources: &[usize], targets: &[usize]| -> Vec<(usize, usize)> {
            sources
                .iter()
                .copied()
                .zip(targets.iter().copied())
                .collect()
        };
        let pairs = match (source_singletons.as_slice(), target_singletons.as_slice()) {
            (sources, targets) if sources.len() == targets.len() => zip(sources, targets),
            (&[source_axis], targets) => targets
                .iter()
                .map(|&target_axis| (source_axis, target_axis))
                .collect(),
            (sources, &[target_axis]) => sources
                .iter()
                .map(|&source_axis| (source_axis, target_axis))
                .collect(),
            (sources, targets) => zip(sources, targets),
        };
        for (source_axis, target_axis) in pairs {
            source_to_target_axes[source_axis].push(target_axis);
        }
    }

    /// `ViewMap._valid_groups` (:1004-1007).
    fn valid_groups(&self) -> Result<&ViewGroups, DimMapError> {
        self.groups.as_ref().ok_or(DimMapError::InvalidMap)
    }
}

/// `ViewMap._preserves_source_axis_order` (:722-733): the target axes of non-singleton
/// source axes appear in source order.
fn preserves_source_axis_order(source_shape: &[i64], source_to_target_axes: &[Vec<usize>]) -> bool {
    source_to_target_axes
        .iter()
        .enumerate()
        .filter(|&(source_axis, _)| source_shape[source_axis] != 1)
        .flat_map(|(_, target_axes)| target_axes.iter().copied())
        .collect::<Vec<_>>()
        .is_sorted()
}

/// `ViewMap._map_dims` (:735-746): the axes on the other side that share a group with
/// any of `dims`, first occurrences in order.
fn map_dims(
    dims: &[usize],
    axis_to_group: &[Option<usize>],
    group_to_axes: &[Vec<usize>],
) -> Vec<usize> {
    dedupe(
        dims.iter()
            .filter_map(|&axis| axis_to_group[axis])
            .flat_map(|group| group_to_axes[group].iter().copied()),
    )
}

/// `ViewMap._matching_permuted_group_blocks` (:748-790): both permutations visit the
/// same groups in the same order, each as one increasing block, skipping singletons.
fn matching_permuted_group_blocks(
    source_permutation: &[usize],
    target_permutation: &[usize],
    groups: &ViewGroups,
) -> bool {
    let mut closed_groups = HashSet::new();
    let mut source_index = 0;
    let mut target_index = 0;
    loop {
        let (source_start, source_group) = next_group(
            source_permutation,
            &groups.source_axis_to_group,
            source_index,
        );
        let (target_start, target_group) = next_group(
            target_permutation,
            &groups.target_axis_to_group,
            target_index,
        );
        let (Some(source_group), Some(target_group)) = (source_group, target_group) else {
            return source_group.is_none() && target_group.is_none();
        };
        if source_group != target_group || closed_groups.contains(&source_group) {
            return false;
        }
        let (source_end, source_axes) = consume_group(
            source_permutation,
            &groups.source_axis_to_group,
            source_start,
            source_group,
        );
        let (target_end, target_axes) = consume_group(
            target_permutation,
            &groups.target_axis_to_group,
            target_start,
            target_group,
        );
        if !source_axes.is_sorted() || !target_axes.is_sorted() {
            return false;
        }
        closed_groups.insert(source_group);
        source_index = source_end;
        target_index = target_end;
    }
}

/// `ViewMap._next_group` (:792-807): the index and group of the next grouped axis.
fn next_group(
    permutation: &[usize],
    axis_to_group: &[Option<usize>],
    mut index: usize,
) -> (usize, Option<usize>) {
    while index < permutation.len() {
        if let Some(group) = axis_to_group[permutation[index]] {
            return (index, Some(group));
        }
        index += 1;
    }
    (index, None)
}

/// `ViewMap._consume_group` (:809-829): the axes of one group block, skipping
/// singletons, and the index after it.
fn consume_group(
    permutation: &[usize],
    axis_to_group: &[Option<usize>],
    mut index: usize,
    group: usize,
) -> (usize, Vec<usize>) {
    let mut axes = Vec::new();
    while index < permutation.len() {
        let axis = permutation[index];
        match axis_to_group[axis] {
            Some(axis_group) if axis_group != group => break,
            Some(_) => axes.push(axis),
            None => {}
        }
        index += 1;
    }
    (index, axes)
}

/// `ViewMap._is_valid_reduction` (:831-854): `is_valid_reduction_or_singleton` with no
/// singleton axes.
fn is_valid_reduction(dims: &[usize], axis_to_group: &[Option<usize>]) -> bool {
    dims.iter().all(|&axis| axis_to_group[axis].is_some())
        && is_valid_reduction_or_singleton(dims, axis_to_group)
}

/// `ViewMap._is_valid_reduction_or_singleton` (:856-878): `dims` is non-empty and
/// holds every axis of each group it touches.
fn is_valid_reduction_or_singleton(dims: &[usize], axis_to_group: &[Option<usize>]) -> bool {
    let selected_groups: HashSet<usize> = dims
        .iter()
        .filter_map(|&axis| axis_to_group[axis])
        .collect();
    !dims.is_empty()
        && axis_to_group.iter().enumerate().all(|(axis, group)| {
            !group.is_some_and(|group| selected_groups.contains(&group)) || dims.contains(&axis)
        })
}

/// `ViewMap._is_contiguous_nonempty` (:880-885).
fn is_contiguous_nonempty(dims: &[usize]) -> bool {
    let mut sorted = dims.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    match (sorted.first(), sorted.last()) {
        (Some(&first), Some(&last)) => last - first + 1 == sorted.len(),
        _ => false,
    }
}

/// `ViewMap._reduce_shape` (:887-892).
fn reduce_shape(shape: &[i64], dims: &[usize]) -> Vec<i64> {
    let mut reduced = shape.to_vec();
    for &dim in dims {
        reduced[dim] = 1;
    }
    reduced
}

/// `ViewMap._build_groups` (:898-962).
fn build_groups(source_shape: &[i64], target_shape: &[i64]) -> Option<ViewGroups> {
    let source_factors = factor_shape(source_shape)?;
    let target_factors = factor_shape(target_shape)?;
    let sorted_primes = |factors: &[Factor]| {
        let mut primes: Vec<i64> = factors.iter().map(|factor| factor.prime).collect();
        primes.sort_unstable();
        primes
    };
    if sorted_primes(&source_factors) != sorted_primes(&target_factors) {
        return None;
    }
    let permutation = find_permutation(&source_factors, &target_factors)?;

    let factor_count = source_factors.len();
    let mut union_find = UnionFind::new(factor_count);
    union_factors_sharing_axes(
        &mut union_find,
        source_factors.iter().map(|factor| factor.axis),
    );
    union_factors_sharing_axes(
        &mut union_find,
        permutation
            .iter()
            .map(|&position| target_factors[position].axis),
    );
    union_crossing_factors(&mut union_find, &permutation);

    let mut groups = ViewGroups {
        source_axis_to_group: vec![None; source_shape.len()],
        target_axis_to_group: vec![None; target_shape.len()],
        group_to_source_axes: vec![Vec::new(); factor_count],
        group_to_target_axes: vec![Vec::new(); factor_count],
    };
    for (source_position, source_factor) in source_factors.iter().enumerate() {
        let group = union_find.find(source_position);
        let target_axis = target_factors[permutation[source_position]].axis;
        groups.source_axis_to_group[source_factor.axis] = Some(group);
        groups.target_axis_to_group[target_axis] = Some(group);
        groups.group_to_source_axes[group].push(source_factor.axis);
        groups.group_to_target_axes[group].push(target_axis);
    }
    for axes in groups
        .group_to_source_axes
        .iter_mut()
        .chain(groups.group_to_target_axes.iter_mut())
    {
        axes.sort_unstable();
        axes.dedup();
    }
    Some(groups)
}

/// `ViewMap._find_permutation` (:964-980): match each source factor to the earliest
/// unused target factor with the same prime.
fn find_permutation(source: &[Factor], target: &[Factor]) -> Option<Vec<usize>> {
    let mut positions: HashMap<i64, VecDeque<usize>> = HashMap::new();
    for (position, factor) in target.iter().enumerate() {
        positions
            .entry(factor.prime)
            .or_default()
            .push_back(position);
    }
    source
        .iter()
        .map(|factor| positions.get_mut(&factor.prime)?.pop_front())
        .collect()
}

/// `ViewMap._union_factors_sharing_axes` (:982-992).
fn union_factors_sharing_axes(union_find: &mut UnionFind, axes: impl Iterator<Item = usize>) {
    let mut first_position_by_axis = HashMap::new();
    for (position, axis) in axes.enumerate() {
        match first_position_by_axis.get(&axis) {
            Some(&first) => union_find.union(first, position),
            None => {
                first_position_by_axis.insert(axis, position);
            }
        }
    }
}

/// `ViewMap._union_crossing_factors` (:994-1002).
fn union_crossing_factors(union_find: &mut UnionFind, permutation: &[usize]) {
    for first in 0..permutation.len() {
        for second in first + 1..permutation.len() {
            if permutation[first] > permutation[second] {
                union_find.union(first, second);
            }
        }
    }
}

/// rc4 `PermuteMap` (:1010-1088): maps dims across a permute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermuteMap {
    permute_dims: Vec<usize>,
}

impl PermuteMap {
    /// `PermuteMap.__init__` (:1013-1023) on `permute_node.args[1]`.
    pub fn new(permute_dims: &[i64]) -> Result<Self, DimMapError> {
        normalize_permutation(permute_dims, permute_dims.len())
            .map(|permute_dims| Self { permute_dims })
            .ok_or_else(|| DimMapError::InvalidPermuteDims(permute_dims.to_vec()))
    }

    /// The normalized permutation.
    pub fn permute_dims(&self) -> &[usize] {
        &self.permute_dims
    }

    /// `PermuteMap.map_dims` (:1025-1035): `x.op(dims).permute(P) ==
    /// x.permute(P).op(mapped_dims)`.
    pub fn map_dims(&self, dims: &[i64]) -> Result<Vec<usize>, DimMapError> {
        let dims = normalize_dims(dims, self.permute_dims.len())?;
        let mut inverse_permute = vec![0; self.permute_dims.len()];
        for (target_dim, &source_dim) in self.permute_dims.iter().enumerate() {
            inverse_permute[source_dim] = target_dim;
        }
        Ok(dims.iter().map(|&dim| inverse_permute[dim]).collect())
    }

    /// `PermuteMap.map_dims_inverse` (:1037-1044): `x.permute(P).op(dims) ==
    /// x.op(mapped_dims).permute(P)`.
    pub fn map_dims_inverse(&self, dims: &[i64]) -> Result<Vec<usize>, DimMapError> {
        let dims = normalize_dims(dims, self.permute_dims.len())?;
        Ok(dims.iter().map(|&dim| self.permute_dims[dim]).collect())
    }

    /// `PermuteMap.lift_to_rank` (:1046-1058): this permutation acting on
    /// `retained_axes` of a rank-`rank` tensor, the other axes left in place.
    pub fn lift_to_rank(&self, retained_axes: &[i64], rank: usize) -> Option<Vec<usize>> {
        if retained_axes.len() != self.permute_dims.len() {
            return None;
        }
        let retained_axes = normalize_dims(retained_axes, rank).ok()?;
        if has_duplicates(&retained_axes) {
            return None;
        }
        let lifted: Vec<usize> = self
            .permute_dims
            .iter()
            .map(|&dim| retained_axes[dim])
            .collect();
        extend_permutation(&lifted, rank)
    }

    /// `PermuteMap.project_to_axes` (:1060-1088): this permutation restricted to
    /// `retained_axes`. With `require_other_axes_preserved`, every other axis must be a
    /// fixed point, as when an op recreates those axes after the projected permute.
    pub fn project_to_axes(
        &self,
        retained_axes: &[i64],
        require_other_axes_preserved: bool,
    ) -> Option<Vec<usize>> {
        let rank = self.permute_dims.len();
        let retained_axes = normalize_dims(retained_axes, rank).ok()?;
        if has_duplicates(&retained_axes) {
            return None;
        }
        if require_other_axes_preserved
            && (0..rank)
                .any(|axis| !retained_axes.contains(&axis) && self.permute_dims[axis] != axis)
        {
            return None;
        }
        Some(
            self.permute_dims
                .iter()
                .filter_map(|dim| retained_axes.iter().position(|axis| axis == dim))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    // Ports rc4's backends/arm/test/passes/test_dim_maps.py except its nine SymInt
    // tests, since RETAB hands symbolic shapes to stock:
    //   test_dim_map_preserves_symbolic_dimensions_as_prime_factors
    //   test_normalize_view_shape_infers_concrete_dim_from_symints
    //   test_normalize_view_shape_preserves_symbolic_inferred_dim
    //   test_view_map_simplifies_constant_symint_dims
    //   test_view_map_from_shapes_normalizes_symbolic_view_shape
    //   test_dim_map_permute_view_swap_preserves_symbolic_view_shape_dims
    //   test_dim_map_view_permute_swap_preserves_symbolic_output_shape_dims
    //   test_dim_map_reduction_view_swap_preserves_symbolic_view_shape_dims
    //   test_dim_map_view_reduction_swap_preserves_symbolic_output_shape_dims
    // The randomized tests draw from a seeded SplitMix64 instead of upstream's
    // torch.Generator, so their cases differ while the properties are the same, and
    // they evaluate the index tensors with `Tensor` below instead of torch.

    use super::*;

    fn from_shapes(source_shape: &[i64], target_shape: &[i64]) -> ViewMap {
        ViewMap::from_shapes(source_shape, target_shape).expect("no inferred dims")
    }

    fn permute_map(permutation: &[usize]) -> PermuteMap {
        PermuteMap::new(&as_i64(permutation)).expect("a permutation")
    }

    fn mapped(dims: &[usize]) -> Result<Option<Vec<usize>>, DimMapError> {
        Ok(Some(dims.to_vec()))
    }

    fn as_i64(dims: &[usize]) -> Vec<i64> {
        dims.iter().map(|&dim| dim as i64).collect()
    }

    /// `torch.arange(numel).reshape(shape)` with the ops the tests apply to it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Tensor {
        shape: Vec<i64>,
        data: Vec<i64>,
    }

    impl Tensor {
        /// `_tensor`.
        fn arange(shape: &[i64]) -> Self {
            Self {
                shape: shape.to_vec(),
                data: (0..shape_product(shape)).collect(),
            }
        }

        fn reshape(&self, shape: &[i64]) -> Self {
            assert_eq!(
                shape_product(shape),
                self.data.len() as i64,
                "reshape to {shape:?}"
            );
            Self {
                shape: shape.to_vec(),
                data: self.data.clone(),
            }
        }

        fn permute(&self, permutation: &[usize]) -> Self {
            let shape: Vec<i64> = permutation.iter().map(|&dim| self.shape[dim]).collect();
            let strides = strides(&self.shape);
            let data = (0..self.data.len())
                .map(|flat| {
                    let source: usize = unravel(flat, &shape)
                        .iter()
                        .zip(permutation)
                        .map(|(&index, &dim)| index * strides[dim])
                        .sum();
                    self.data[source]
                })
                .collect();
            Self { shape, data }
        }

        /// `_reduce`: `sum(dim=dims, keepdim=True)`.
        fn sum_keepdim(&self, dims: &[usize]) -> Self {
            let shape = reduce_shape(&self.shape, dims);
            let strides = strides(&shape);
            let mut data = vec![0; shape_product(&shape) as usize];
            for (flat, &value) in self.data.iter().enumerate() {
                let target: usize = unravel(flat, &self.shape)
                    .iter()
                    .enumerate()
                    .filter(|(axis, _)| !dims.contains(axis))
                    .map(|(axis, &index)| index * strides[axis])
                    .sum();
                data[target] += value;
            }
            Self { shape, data }
        }
    }

    fn strides(shape: &[i64]) -> Vec<usize> {
        let mut stride = 1;
        let mut strides: Vec<usize> = shape
            .iter()
            .rev()
            .map(|&dim| {
                let current = stride;
                stride *= dim as usize;
                current
            })
            .collect();
        strides.reverse();
        strides
    }

    fn unravel(mut flat: usize, shape: &[i64]) -> Vec<usize> {
        let mut index = vec![0; shape.len()];
        for (slot, &dim) in index.iter_mut().zip(shape).rev() {
            *slot = flat % dim as usize;
            flat /= dim as usize;
        }
        index
    }

    fn inverse_permutation(permutation: &[usize]) -> Vec<usize> {
        let mut inverse = vec![0; permutation.len()];
        for (index, &dim) in permutation.iter().enumerate() {
            inverse[dim] = index;
        }
        inverse
    }

    /// `_factorizations`: every rank-`rank` shape with `numel` elements, in upstream's
    /// order.
    fn factorizations(numel: i64, rank: usize) -> Vec<Vec<i64>> {
        fn recurse(remaining: i64, rank: usize, shape: &mut Vec<i64>, shapes: &mut Vec<Vec<i64>>) {
            if rank == 0 {
                if remaining == 1 {
                    shapes.push(shape.clone());
                }
                return;
            }
            for dim in (1..=remaining).filter(|dim| remaining % dim == 0) {
                shape.push(dim);
                recurse(remaining / dim, rank - 1, shape, shapes);
                shape.pop();
            }
        }
        let mut shapes = Vec::new();
        recurse(numel, rank, &mut Vec::new(), &mut shapes);
        shapes
    }

    /// `_all_dim_subsets`: `itertools.combinations` of every size from 1, in order.
    fn all_dim_subsets(rank: usize) -> Vec<Vec<usize>> {
        fn recurse(
            start: usize,
            rank: usize,
            size: usize,
            dims: &mut Vec<usize>,
            subsets: &mut Vec<Vec<usize>>,
        ) {
            if dims.len() == size {
                subsets.push(dims.clone());
                return;
            }
            for dim in start..rank {
                dims.push(dim);
                recurse(dim + 1, rank, size, dims, subsets);
                dims.pop();
            }
        }
        let mut subsets = Vec::new();
        for size in 1..=rank {
            recurse(0, rank, size, &mut Vec::new(), &mut subsets);
        }
        subsets
    }

    /// `itertools.permutations(range(rank))`, in lexicographic order.
    fn permutations(rank: usize) -> Vec<Vec<usize>> {
        fn recurse(
            used: &mut [bool],
            permutation: &mut Vec<usize>,
            permutations: &mut Vec<Vec<usize>>,
        ) {
            if permutation.len() == used.len() {
                permutations.push(permutation.clone());
                return;
            }
            for dim in 0..used.len() {
                if !used[dim] {
                    used[dim] = true;
                    permutation.push(dim);
                    recurse(used, permutation, permutations);
                    permutation.pop();
                    used[dim] = false;
                }
            }
        }
        let mut permutations = Vec::new();
        recurse(&mut vec![false; rank], &mut Vec::new(), &mut permutations);
        permutations
    }

    /// Seeded SplitMix64, standing in for upstream's `torch.Generator().manual_seed(0)`.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// `_randint`: uniform on `low..=high`.
        fn randint(&mut self, low: i64, high: i64) -> i64 {
            low + (self.next_u64() % (high - low + 1) as u64) as i64
        }

        /// `_choice`.
        fn choice<T: Clone>(&mut self, choices: &[T]) -> T {
            choices[self.randint(0, choices.len() as i64 - 1) as usize].clone()
        }

        /// `_shuffle`, as a Fisher-Yates shuffle (upstream permutes by `torch.randperm`).
        fn shuffle<T>(&mut self, values: &mut [T]) {
            for index in (1..values.len()).rev() {
                values.swap(index, self.randint(0, index as i64) as usize);
            }
        }

        /// `_random_shape`.
        fn random_shape(&mut self, rank: usize, max_dim: i64) -> Vec<i64> {
            (0..rank).map(|_| self.randint(1, max_dim)).collect()
        }

        /// `_random_view_shape`.
        fn random_view_shape(&mut self, numel: i64, max_rank: i64) -> Vec<i64> {
            let rank = self.randint(1, max_rank) as usize;
            self.choice(&factorizations(numel, rank))
        }

        /// A rank-1..=4 shape of dims up to 3 and a shuffled identity permutation of it.
        fn random_shape_and_permutation(&mut self) -> (Vec<i64>, Vec<usize>) {
            let rank = self.randint(1, 4) as usize;
            let shape = self.random_shape(rank, 3);
            let mut permutation: Vec<usize> = (0..rank).collect();
            self.shuffle(&mut permutation);
            (shape, permutation)
        }
    }

    /// `_propose_permute_view_swap`: `permute(P).view(S)` as `view(S').permute(P')`.
    fn propose_permute_view_swap(
        input_shape: &[i64],
        permutation: &[usize],
        output_shape: &[i64],
    ) -> Option<(Vec<i64>, Vec<usize>)> {
        let permuted_shape: Vec<i64> = permutation.iter().map(|&dim| input_shape[dim]).collect();
        let view_map = from_shapes(&permuted_shape, output_shape);
        if !view_map.is_valid_map() {
            return None;
        }
        let permuted_axis = inverse_permutation(permutation);
        let target_axis_order = view_map.map_permutation(&as_i64(&permuted_axis)).unwrap()?;
        Some((
            target_axis_order
                .iter()
                .map(|&axis| output_shape[axis])
                .collect(),
            inverse_permutation(&target_axis_order),
        ))
    }

    /// `_propose_view_permute_swap`: `view(S).permute(P)` as `permute(P').view(S')`.
    fn propose_view_permute_swap(
        input_shape: &[i64],
        view_shape: &[i64],
        permutation: &[usize],
    ) -> Option<(Vec<usize>, Vec<i64>)> {
        let view_map = from_shapes(input_shape, view_shape);
        if !view_map.is_valid_map() {
            return None;
        }
        let mapped_dims = view_map
            .map_permutation_inverse(&as_i64(permutation))
            .unwrap()?;
        Some((
            mapped_dims,
            permutation.iter().map(|&dim| view_shape[dim]).collect(),
        ))
    }

    /// `_propose_reduction_view_swap`.
    fn propose_reduction_view_swap(
        input_shape: &[i64],
        source_dims: &[usize],
        view_shape: &[i64],
    ) -> Option<(Vec<i64>, Vec<usize>)> {
        let view_map = from_shapes(input_shape, view_shape);
        if !view_map.is_valid_map() {
            return None;
        }
        view_map.map_reduction_after_view(input_shape, &as_i64(source_dims))
    }

    /// `_propose_view_reduction_swap`.
    fn propose_view_reduction_swap(
        input_shape: &[i64],
        view_shape: &[i64],
        target_dims: &[usize],
    ) -> Option<(Vec<usize>, Vec<i64>)> {
        let view_map = from_shapes(input_shape, view_shape);
        if !view_map.is_valid_map() {
            return None;
        }
        view_map
            .map_reduction_before_view(&as_i64(target_dims))
            .unwrap()
    }

    /// `_bruteforce_permute_view_swaps`.
    fn bruteforce_permute_view_swaps(
        x: &Tensor,
        permutation: &[usize],
        output_shape: &[i64],
    ) -> Vec<(Vec<i64>, Vec<usize>)> {
        let original = x.permute(permutation).reshape(output_shape);
        let mut candidates = Vec::new();
        for candidate_permutation in permutations(output_shape.len()) {
            for candidate_shape in factorizations(shape_product(output_shape), output_shape.len()) {
                if x.reshape(&candidate_shape).permute(&candidate_permutation) == original {
                    candidates.push((candidate_shape, candidate_permutation.clone()));
                }
            }
        }
        candidates
    }

    /// `_bruteforce_view_permute_swaps`.
    fn bruteforce_view_permute_swaps(
        x: &Tensor,
        view_shape: &[i64],
        permutation: &[usize],
    ) -> Vec<(Vec<usize>, Vec<i64>)> {
        let output_shape: Vec<i64> = permutation.iter().map(|&dim| view_shape[dim]).collect();
        let original = x.reshape(view_shape).permute(permutation);
        permutations(x.shape.len())
            .into_iter()
            .filter(|candidate| x.permute(candidate).reshape(&output_shape) == original)
            .map(|candidate| (candidate, output_shape.clone()))
            .collect()
    }

    /// `_bruteforce_reduction_view_swaps`.
    fn bruteforce_reduction_view_swaps(
        x: &Tensor,
        source_dims: &[usize],
        view_shape: &[i64],
    ) -> Vec<(Vec<i64>, Vec<usize>)> {
        let reduced = x.sum_keepdim(source_dims);
        all_dim_subsets(view_shape.len())
            .into_iter()
            .filter(|target_dims| {
                let output_shape = reduce_shape(view_shape, target_dims);
                reduced.data.len() as i64 == shape_product(&output_shape)
                    && reduced.reshape(&output_shape)
                        == x.reshape(view_shape).sum_keepdim(target_dims)
            })
            .map(|target_dims| (view_shape.to_vec(), target_dims))
            .collect()
    }

    /// `_bruteforce_view_reduction_swaps`.
    fn bruteforce_view_reduction_swaps(
        x: &Tensor,
        view_shape: &[i64],
        target_dims: &[usize],
    ) -> Vec<(Vec<usize>, Vec<i64>)> {
        let original = x.reshape(view_shape).sum_keepdim(target_dims);
        all_dim_subsets(x.shape.len())
            .into_iter()
            .filter(|source_dims| {
                let reduced = x.sum_keepdim(source_dims);
                reduced.data.len() == original.data.len()
                    && reduced.reshape(&original.shape) == original
            })
            .map(|source_dims| (source_dims, original.shape.clone()))
            .collect()
    }

    /// rc4 `test_dim_map_maps_split_and_merged_prime_factor_groups`.
    #[test]
    fn dim_map_maps_split_and_merged_prime_factor_groups() {
        let view_map = from_shapes(&[1, 2, 3, 4], &[1, 6, 2, 2]);

        assert!(view_map.is_valid_map());
        assert_eq!(view_map.map_dim(&[0]), mapped(&[0]));
        assert_eq!(view_map.map_dim(&[1]), Ok(None));
        assert_eq!(view_map.map_dim(&[2]), Ok(None));
        assert_eq!(view_map.map_dim(&[3]), mapped(&[2, 3]));
        assert_eq!(view_map.map_dim(&[1, 2]), mapped(&[1]));
        assert_eq!(view_map.map_dim(&[3, 1]), Ok(None));
        assert_eq!(view_map.map_dim(&[3, 1, 2]), mapped(&[1, 2, 3]));

        assert_eq!(view_map.map_dim_inverse(&[0]), Ok(None));
        assert_eq!(view_map.map_dim_inverse(&[1]), mapped(&[1, 2]));
        assert_eq!(view_map.map_dim_inverse(&[2]), Ok(None));
        assert_eq!(view_map.map_dim_inverse(&[3, 1, 2]), mapped(&[3, 1, 2]));
        assert_eq!(view_map.map_dim_inverse(&[2, 3, 1]), mapped(&[3, 1, 2]));
    }

    /// rc4 `test_dim_map_groups_reordered_crossing_prime_factors`.
    #[test]
    fn dim_map_groups_reordered_crossing_prime_factors() {
        let view_map = from_shapes(&[2, 3], &[3, 2]);

        assert!(view_map.is_valid_map());
        assert_eq!(view_map.map_dim(&[0]), Ok(None));
        assert_eq!(view_map.map_dim(&[1]), Ok(None));
        assert_eq!(view_map.map_dim(&[0, 1]), mapped(&[0, 1]));
        assert_eq!(view_map.map_dim_inverse(&[0]), Ok(None));
        assert_eq!(view_map.map_dim_inverse(&[1]), Ok(None));
        assert_eq!(view_map.map_dim_inverse(&[0, 1]), mapped(&[0, 1]));
    }

    /// rc4 `test_dim_map_matches_view_map_docstring_example_reduction_dims`.
    #[test]
    fn dim_map_matches_view_map_docstring_example_reduction_dims() {
        let view_map = from_shapes(&[4, 3, 10], &[2, 2, 1, 5, 3, 2]);

        assert!(view_map.is_valid_map());
        assert_eq!(view_map.map_dim(&[0]), mapped(&[0, 1]));
        assert_eq!(view_map.map_dim(&[1, 2]), mapped(&[3, 4, 5]));
        assert_eq!(view_map.map_dim_inverse(&[0, 1]), mapped(&[0]));
        assert_eq!(view_map.map_dim_inverse(&[3, 4, 5]), mapped(&[1, 2]));

        assert_eq!(view_map.map_dim(&[1]), Ok(None));
        assert_eq!(view_map.map_dim(&[2]), Ok(None));
        assert_eq!(view_map.map_dim_inverse(&[2]), Ok(None));
    }

    /// rc4 `test_dim_map_allows_singleton_rank_change_reduction_swap`.
    #[test]
    fn dim_map_allows_singleton_rank_change_reduction_swap() {
        let x = Tensor::arange(&[6, 4]);
        let proposal = propose_reduction_view_swap(&[6, 4], &[0], &[6, 1, 4]);
        let candidates = bruteforce_reduction_view_swaps(&x, &[0], &[6, 1, 4]);

        let expected = (vec![6, 1, 4], vec![0]);
        assert!(candidates.contains(&expected));
        assert_eq!(proposal, Some(expected));
    }

    /// rc4 `test_dim_map_allows_inverse_singleton_rank_change_reduction_swap`.
    #[test]
    fn dim_map_allows_inverse_singleton_rank_change_reduction_swap() {
        let x = Tensor::arange(&[6, 4]);
        let proposal = propose_view_reduction_swap(&[6, 4], &[6, 1, 4], &[0]);
        let candidates = bruteforce_view_reduction_swaps(&x, &[6, 1, 4], &[0]);

        let expected = (vec![0], vec![1, 1, 4]);
        assert!(candidates.contains(&expected));
        assert_eq!(proposal, Some(expected));
    }

    /// rc4 `test_dim_map_allows_output_singleton_rank_change_reduction_swap`.
    #[test]
    fn dim_map_allows_output_singleton_rank_change_reduction_swap() {
        let x = Tensor::arange(&[6, 4]);
        let proposal = from_shapes(&[1, 4], &[1, 1, 4]).map_reduction_after_view(&[6, 4], &[0]);

        assert_eq!(proposal, Some((vec![6, 1, 4], vec![0, 1])));
        let (view_shape, target_dims) = proposal.unwrap();
        let original = x.sum_keepdim(&[0]).reshape(&[1, 1, 4]);
        let candidate = x.reshape(&view_shape).sum_keepdim(&target_dims);
        assert_eq!(original, candidate);
    }

    /// rc4 `test_dim_map_rejects_invalid_zero_dim_remap`.
    #[test]
    fn dim_map_rejects_invalid_zero_dim_remap() {
        let view_map = from_shapes(&[5, 2, 60], &[2, 60]);

        assert_eq!(view_map.remap_target_shape(&[1, 2, 60]), None);
    }

    /// RETAB: the upstream test above uses a map that is already invalid, so it never
    /// reaches the zero-dim check in `remap_target_shape`; this one does.
    #[test]
    fn rejects_zero_dim_remap_on_a_valid_map() {
        let view_map = from_shapes(&[2, 3], &[6]);

        assert!(view_map.is_valid_map());
        assert_eq!(view_map.remap_target_shape(&[2, 3]), Some(vec![6]));
        assert_eq!(view_map.remap_target_shape(&[0, 3]), None);
    }

    /// rc4 `test_dim_map_matches_view_map_docstring_example_permutation_dims`.
    #[test]
    fn dim_map_matches_view_map_docstring_example_permutation_dims() {
        let view_map = from_shapes(&[4, 3, 10], &[2, 2, 1, 5, 3, 2]);

        assert_eq!(
            view_map.map_permutation(&[1, 2, 0]),
            mapped(&[2, 3, 4, 5, 0, 1])
        );
        assert_eq!(
            view_map.map_permutation_inverse(&[2, 3, 4, 5, 0, 1]),
            mapped(&[1, 2, 0])
        );

        assert_eq!(view_map.map_permutation(&[0, 2, 1]), Ok(None));
        assert_eq!(view_map.map_permutation(&[2, 0, 1]), Ok(None));
    }

    /// rc4 `test_dim_map_validates_reductions_by_whole_groups`.
    #[test]
    fn dim_map_validates_reductions_by_whole_groups() {
        let view_map = from_shapes(&[2, 3], &[3, 2]);

        assert_eq!(view_map.map_dim(&[0]), Ok(None));
        assert_eq!(view_map.map_dim_inverse(&[1]), Ok(None));
        assert_eq!(view_map.map_dim(&[0, 1]), mapped(&[0, 1]));
        assert_eq!(view_map.map_dim_inverse(&[0, 1]), mapped(&[0, 1]));
    }

    /// rc4 `test_dim_map_validates_permuted_group_blocks`.
    #[test]
    fn dim_map_validates_permuted_group_blocks() {
        let view_map = from_shapes(&[2, 3, 5], &[3, 2, 5]);

        assert_eq!(view_map.map_permutation(&[0, 1, 2]), mapped(&[0, 1, 2]));
        assert_eq!(view_map.map_permutation(&[2, 0, 1]), mapped(&[2, 0, 1]));
        assert_eq!(view_map.map_permutation_inverse(&[0, 2, 1]), Ok(None));

        let merged_view_map = from_shapes(&[2, 3], &[6]);
        assert_eq!(merged_view_map.map_permutation(&[0, 1]), mapped(&[0]));
        assert_eq!(merged_view_map.map_permutation(&[1, 0]), Ok(None));
    }

    /// rc4 `test_extends_mapped_permutation_with_singletons`.
    #[test]
    fn extends_mapped_permutation_with_singletons() {
        let view_map = from_shapes(&[2, 2], &[2, 1, 2]);
        assert_eq!(view_map.map_permutation(&[0, 1]), mapped(&[0, 1, 2]));
        assert_eq!(view_map.map_permutation(&[1, 0]), mapped(&[1, 2, 0]));

        let singleton_view_map = from_shapes(&[2], &[1, 2]);
        assert_eq!(singleton_view_map.map_permutation(&[0]), mapped(&[0, 1]));
        assert_eq!(
            singleton_view_map.map_permutation_inverse(&[1, 0]),
            mapped(&[0])
        );

        assert_eq!(view_map.map_permutation(&[0, 0]), Ok(None));
    }

    /// rc4 `test_dim_map_uses_strict_no_mapping_for_singletons`.
    #[test]
    fn dim_map_uses_strict_no_mapping_for_singletons() {
        let view_map = from_shapes(&[1, 4], &[4]);

        assert!(view_map.is_valid_map());
        assert_eq!(view_map.map_dim(&[0]), Ok(None));
        assert_eq!(view_map.map_dim(&[1]), mapped(&[0]));
        assert_eq!(view_map.map_dim_inverse(&[0]), mapped(&[1]));

        let split_view_map = from_shapes(&[4], &[2, 1, 2]);
        assert_eq!(split_view_map.map_dim(&[0]), mapped(&[0, 2]));
        assert_eq!(split_view_map.map_dim_inverse(&[1]), Ok(None));
        assert_eq!(split_view_map.map_dim_inverse(&[0, 2]), mapped(&[0]));
    }

    /// rc4 `test_dim_map_maps_reduced_singletons_only_when_unambiguous`.
    #[test]
    fn dim_map_maps_reduced_singletons_only_when_unambiguous() {
        let split_singleton_view_map = from_shapes(&[1, 4], &[1, 1, 4]);
        assert_eq!(split_singleton_view_map.map_dim(&[0]), mapped(&[0, 1]));

        let squeezed_singleton_view_map = from_shapes(&[1, 50, 10, 1], &[1, 50, 10]);
        assert_eq!(squeezed_singleton_view_map.map_dim(&[-1]), Ok(None));
        assert_eq!(squeezed_singleton_view_map.map_dim(&[0, -1]), mapped(&[0]));
    }

    /// rc4 `test_dim_map_remaps_unit_slice_through_view`.
    #[test]
    fn dim_map_remaps_unit_slice_through_view() {
        let view_map = from_shapes(&[5, 2, 1, 4, 6], &[5, 2, 4, 6]);

        assert_eq!(
            view_map.remap_unit_slice(&[5, 2, 3, 4, 6], 2, Some(0), Some(1), 1),
            Ok(Some((vec![5, 2, 12, 6], 2, 0, 4)))
        );
        assert_eq!(
            view_map.remap_unit_slice(&[5, 2, 3, 4, 6], 2, Some(1), Some(2), 1),
            Ok(Some((vec![5, 2, 12, 6], 2, 4, 8)))
        );
    }

    /// rc4 `test_dim_map_remaps_unit_slice_through_flattening_view`.
    #[test]
    fn dim_map_remaps_unit_slice_through_flattening_view() {
        let view_map = from_shapes(&[5, 2, 1, 4, 6], &[5, 2, 24]);

        assert_eq!(
            view_map.remap_unit_slice(&[5, 2, 3, 4, 6], 2, Some(1), Some(2), 1),
            Ok(Some((vec![5, 2, 72], 2, 24, 48)))
        );
    }

    /// rc4 `test_dim_map_does_not_remap_unit_slice_into_previous_axis`.
    #[test]
    fn dim_map_does_not_remap_unit_slice_into_previous_axis() {
        let view_map = from_shapes(&[3, 3, 1], &[3, 3]);

        assert_eq!(
            view_map.remap_unit_slice(&[3, 3, 3], 2, Some(0), Some(1), 1),
            Ok(None)
        );
    }

    /// rc4 `test_permute_map_matches_docstring_reduction_identities`.
    #[test]
    fn permute_map_matches_docstring_reduction_identities() {
        let permutation = [2, 0, 1];
        let permute_map = permute_map(&permutation);
        let x = Tensor::arange(&[2, 3, 5]);

        let source_dims = [0, 2];
        let target_dims = permute_map.map_dims(&as_i64(&source_dims)).unwrap();
        assert_eq!(target_dims, [1, 0]);
        assert_eq!(
            x.sum_keepdim(&source_dims).permute(&permutation),
            x.permute(&permutation).sum_keepdim(&target_dims)
        );

        let target_dims = [0, 2];
        let source_dims = permute_map.map_dims_inverse(&as_i64(&target_dims)).unwrap();
        assert_eq!(source_dims, [2, 1]);
        assert_eq!(
            x.permute(&permutation).sum_keepdim(&target_dims),
            x.sum_keepdim(&source_dims).permute(&permutation)
        );
    }

    /// rc4 `test_permute_map_lifts_and_projects_across_rank_changes`.
    #[test]
    fn permute_map_lifts_and_projects_across_rank_changes() {
        let lower_rank = permute_map(&[1, 0]);
        assert_eq!(lower_rank.lift_to_rank(&[2, 3], 4), Some(vec![0, 1, 3, 2]));
        assert_eq!(lower_rank.lift_to_rank(&[1, 2], 3), Some(vec![0, 2, 1]));

        let higher_rank = permute_map(&[0, 1, 3, 2]);
        assert_eq!(higher_rank.project_to_axes(&[2, 3], true), Some(vec![1, 0]));
        assert_eq!(
            permute_map(&[2, 1, 0, 3]).project_to_axes(&[2, 3], true),
            None
        );
        assert_eq!(
            permute_map(&[1, 0, 2, 3]).project_to_axes(&[2, 3], true),
            None
        );
        assert_eq!(
            permute_map(&[2, 0, 1]).project_to_axes(&[1, 2], false),
            Some(vec![1, 0])
        );
    }

    // The four randomized swap tests below check soundness like upstream: when the map
    // rejects a swap that brute force finds, they check brute force's first candidate
    // instead. RETAB adds that the map itself proposed at least one swap.

    /// rc4 `test_dim_map_randomized_permute_view_swaps_match_bruteforce`.
    #[test]
    fn dim_map_randomized_permute_view_swaps_match_bruteforce() {
        let mut rng = Rng(0);
        let mut proposed = 0;
        for _ in 0..80 {
            let (input_shape, permutation) = rng.random_shape_and_permutation();
            let output_shape = rng.random_view_shape(shape_product(&input_shape), 4);
            let x = Tensor::arange(&input_shape);

            let proposal = propose_permute_view_swap(&input_shape, &permutation, &output_shape);
            let brute_force_swaps = bruteforce_permute_view_swaps(&x, &permutation, &output_shape);
            proposed += usize::from(proposal.is_some());
            let Some(proposal) = proposal.or_else(|| brute_force_swaps.first().cloned()) else {
                continue;
            };
            assert!(brute_force_swaps.contains(&proposal));
            let (view_shape, new_permutation) = proposal;
            let original = x.permute(&permutation).reshape(&output_shape);
            let candidate = x.reshape(&view_shape).permute(&new_permutation);
            assert_eq!(original, candidate);
        }
        assert!(proposed > 0);
    }

    /// rc4 `test_dim_map_randomized_view_permute_swaps_match_bruteforce`.
    #[test]
    fn dim_map_randomized_view_permute_swaps_match_bruteforce() {
        let mut rng = Rng(0);
        let mut proposed = 0;
        for _ in 0..80 {
            let rank = rng.randint(1, 4) as usize;
            let input_shape = rng.random_shape(rank, 3);
            let view_shape = rng.random_view_shape(shape_product(&input_shape), 4);
            let mut permutation: Vec<usize> = (0..view_shape.len()).collect();
            rng.shuffle(&mut permutation);
            let x = Tensor::arange(&input_shape);

            let proposal = propose_view_permute_swap(&input_shape, &view_shape, &permutation);
            let brute_force_swaps = bruteforce_view_permute_swaps(&x, &view_shape, &permutation);
            proposed += usize::from(proposal.is_some());
            let Some(proposal) = proposal.or_else(|| brute_force_swaps.first().cloned()) else {
                continue;
            };
            assert!(brute_force_swaps.contains(&proposal));
            let (new_permutation, output_shape) = proposal;
            let original = x.reshape(&view_shape).permute(&permutation);
            let candidate = x.permute(&new_permutation).reshape(&output_shape);
            assert_eq!(original, candidate);
        }
        assert!(proposed > 0);
    }

    /// rc4 `test_dim_map_randomized_reduction_view_swaps_match_bruteforce`.
    #[test]
    fn dim_map_randomized_reduction_view_swaps_match_bruteforce() {
        let mut rng = Rng(0);
        let mut proposed = 0;
        for _ in 0..80 {
            let rank = rng.randint(1, 4) as usize;
            let input_shape = rng.random_shape(rank, 3);
            let source_dims = rng.choice(&all_dim_subsets(rank));
            let view_shape = rng.random_view_shape(shape_product(&input_shape), 4);
            let x = Tensor::arange(&input_shape);

            let proposal = propose_reduction_view_swap(&input_shape, &source_dims, &view_shape);
            let brute_force_swaps = bruteforce_reduction_view_swaps(&x, &source_dims, &view_shape);
            proposed += usize::from(proposal.is_some());
            let Some(proposal) = proposal.or_else(|| brute_force_swaps.first().cloned()) else {
                continue;
            };
            assert!(brute_force_swaps.contains(&proposal));
            let (new_shape, target_dims) = proposal;
            let output_shape = reduce_shape(&new_shape, &target_dims);
            let original = x.sum_keepdim(&source_dims).reshape(&output_shape);
            let candidate = x.reshape(&new_shape).sum_keepdim(&target_dims);
            assert_eq!(original, candidate);
        }
        assert!(proposed > 0);
    }

    /// rc4 `test_dim_map_randomized_view_reduction_swaps_match_bruteforce`.
    #[test]
    fn dim_map_randomized_view_reduction_swaps_match_bruteforce() {
        let mut rng = Rng(0);
        let mut proposed = 0;
        for _ in 0..80 {
            let rank = rng.randint(1, 4) as usize;
            let input_shape = rng.random_shape(rank, 3);
            let view_shape = rng.random_view_shape(shape_product(&input_shape), 4);
            let target_dims = rng.choice(&all_dim_subsets(view_shape.len()));
            let x = Tensor::arange(&input_shape);

            let proposal = propose_view_reduction_swap(&input_shape, &view_shape, &target_dims);
            let brute_force_swaps = bruteforce_view_reduction_swaps(&x, &view_shape, &target_dims);
            proposed += usize::from(proposal.is_some());
            let Some(proposal) = proposal.or_else(|| brute_force_swaps.first().cloned()) else {
                continue;
            };
            assert!(brute_force_swaps.contains(&proposal));
            let (source_dims, output_shape) = proposal;
            let original = x.reshape(&view_shape).sum_keepdim(&target_dims);
            let candidate = x.sum_keepdim(&source_dims).reshape(&output_shape);
            assert_eq!(original, candidate);
        }
        assert!(proposed > 0);
    }

    /// rc4 `test_permute_map_randomized_reduction_permute_swaps_match_bruteforce`.
    #[test]
    fn permute_map_randomized_reduction_permute_swaps_match_bruteforce() {
        let mut rng = Rng(0);
        for _ in 0..80 {
            let rank = rng.randint(1, 4) as usize;
            let input_shape = rng.random_shape(rank, 3);
            let source_dims = rng.choice(&all_dim_subsets(rank));
            let mut permutation: Vec<usize> = (0..rank).collect();
            rng.shuffle(&mut permutation);
            let mut target_dims = permute_map(&permutation)
                .map_dims(&as_i64(&source_dims))
                .unwrap();
            let x = Tensor::arange(&input_shape);

            let original = x.sum_keepdim(&source_dims).permute(&permutation);
            assert_eq!(original, x.permute(&permutation).sum_keepdim(&target_dims));

            let brute_force_dims: Vec<Vec<usize>> = all_dim_subsets(rank)
                .into_iter()
                .filter(|dims| original == x.permute(&permutation).sum_keepdim(dims))
                .collect();
            target_dims.sort_unstable();
            assert!(brute_force_dims.contains(&target_dims));
        }
    }

    /// rc4 `test_permute_map_randomized_permute_reduction_swaps_match_bruteforce`.
    #[test]
    fn permute_map_randomized_permute_reduction_swaps_match_bruteforce() {
        let mut rng = Rng(0);
        for _ in 0..80 {
            let (input_shape, permutation) = rng.random_shape_and_permutation();
            let rank = input_shape.len();
            let target_dims = rng.choice(&all_dim_subsets(rank));
            let mut source_dims = permute_map(&permutation)
                .map_dims_inverse(&as_i64(&target_dims))
                .unwrap();
            let x = Tensor::arange(&input_shape);

            let original = x.permute(&permutation).sum_keepdim(&target_dims);
            assert_eq!(original, x.sum_keepdim(&source_dims).permute(&permutation));

            let brute_force_dims: Vec<Vec<usize>> = all_dim_subsets(rank)
                .into_iter()
                .filter(|dims| original == x.sum_keepdim(dims).permute(&permutation))
                .collect();
            source_dims.sort_unstable();
            assert!(brute_force_dims.contains(&source_dims));
        }
    }

    /// Python returns `None` -> `Ok(None)` or `None`; Python raises -> `Err`.
    #[test]
    fn mirrors_python_failure_modes() {
        assert_eq!(
            normalize_view_shape(&[4], &[2, -1, -1]),
            Err(DimMapError::InvalidViewShape(vec![2, -1, -1]))
        );
        assert_eq!(
            normalize_view_shape(&[0, 5], &[0, -1]),
            Err(DimMapError::ZeroDivision(vec![0, -1]))
        );
        assert_eq!(normalize_view_shape(&[5], &[-2, -1]), Ok(vec![-2, -3]));

        let invalid = from_shapes(&[2, 3], &[5]);
        assert!(!invalid.is_valid_map());
        assert_eq!(invalid.map_dim(&[2]), Ok(None));
        assert_eq!(invalid.map_dim(&[0]), Err(DimMapError::InvalidMap));
        assert_eq!(invalid.map_permutation(&[0, 0]), Ok(None));
        assert_eq!(
            invalid.map_permutation(&[1, 0]),
            Err(DimMapError::InvalidMap)
        );
        assert_eq!(invalid.remap_target_shape(&[2, 3]), None);

        assert_eq!(
            PermuteMap::new(&[0, 0]),
            Err(DimMapError::InvalidPermuteDims(vec![0, 0]))
        );
        let permute_map = PermuteMap::new(&[1, -2]).unwrap();
        assert_eq!(permute_map.permute_dims(), [1, 0]);
        assert_eq!(
            permute_map.map_dims(&[2]),
            Err(DimMapError::InvalidDim { dim: 2, rank: 2 })
        );
    }

    /// A fixture value: null, a boolean, an integer, an unescaped string or a list.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Json {
        Null,
        Bool(bool),
        Int(i64),
        Str(String),
        List(Vec<Json>),
    }

    impl Json {
        fn parse(text: &str) -> Self {
            let (value, rest) = Self::parse_prefix(text);
            assert!(rest.is_empty(), "trailing {rest:?} in {text:?}");
            value
        }

        fn parse_prefix(text: &str) -> (Self, &str) {
            for (literal, value) in [
                ("null", Json::Null),
                ("true", Json::Bool(true)),
                ("false", Json::Bool(false)),
            ] {
                if let Some(rest) = text.strip_prefix(literal) {
                    return (value, rest);
                }
            }
            if let Some(rest) = text.strip_prefix('"') {
                let end = rest.find('"').expect("a closing quote");
                assert!(!rest[..end].contains('\\'), "escapes are unsupported");
                return (Json::Str(rest[..end].to_owned()), &rest[end + 1..]);
            }
            if let Some(mut rest) = text.strip_prefix('[') {
                let mut items = Vec::new();
                if let Some(after) = rest.strip_prefix(']') {
                    return (Json::List(items), after);
                }
                loop {
                    let (item, after) = Self::parse_prefix(rest);
                    items.push(item);
                    match after.strip_prefix(',') {
                        Some(next) => rest = next,
                        None => {
                            let after = after.strip_prefix(']').expect("a closing bracket");
                            return (Json::List(items), after);
                        }
                    }
                }
            }
            let end = text
                .find(|c: char| c != '-' && !c.is_ascii_digit())
                .unwrap_or(text.len());
            let value = text[..end]
                .parse()
                .unwrap_or_else(|_| panic!("not JSON: {text:?}"));
            (Json::Int(value), &text[end..])
        }

        fn list(&self) -> &[Json] {
            match self {
                Json::List(items) => items,
                other => panic!("expected a list, got {other:?}"),
            }
        }

        fn int(&self) -> i64 {
            match self {
                Json::Int(value) => *value,
                other => panic!("expected an int, got {other:?}"),
            }
        }

        fn optional_int(&self) -> Option<i64> {
            (*self != Json::Null).then(|| self.int())
        }

        fn ints(&self) -> Vec<i64> {
            self.list().iter().map(Json::int).collect()
        }

        fn rank(&self) -> usize {
            usize::try_from(self.int()).expect("a rank")
        }

        fn flag(&self) -> bool {
            match self {
                Json::Bool(value) => *value,
                other => panic!("expected a bool, got {other:?}"),
            }
        }
    }

    trait ToJson {
        fn to_json(&self) -> Json;
    }

    impl ToJson for i64 {
        fn to_json(&self) -> Json {
            Json::Int(*self)
        }
    }

    impl ToJson for usize {
        fn to_json(&self) -> Json {
            Json::Int(*self as i64)
        }
    }

    impl ToJson for bool {
        fn to_json(&self) -> Json {
            Json::Bool(*self)
        }
    }

    impl<T: ToJson> ToJson for Vec<T> {
        fn to_json(&self) -> Json {
            Json::List(self.iter().map(ToJson::to_json).collect())
        }
    }

    impl<A: ToJson, B: ToJson> ToJson for (A, B) {
        fn to_json(&self) -> Json {
            Json::List(vec![self.0.to_json(), self.1.to_json()])
        }
    }

    impl<A: ToJson, B: ToJson, C: ToJson> ToJson for (A, B, C) {
        fn to_json(&self) -> Json {
            Json::List(vec![self.0.to_json(), self.1.to_json(), self.2.to_json()])
        }
    }

    impl<A: ToJson, B: ToJson, C: ToJson, D: ToJson> ToJson for (A, B, C, D) {
        fn to_json(&self) -> Json {
            let (a, b, c, d) = self;
            Json::List(vec![a.to_json(), b.to_json(), c.to_json(), d.to_json()])
        }
    }

    impl<T: ToJson> ToJson for Option<T> {
        fn to_json(&self) -> Json {
            self.as_ref().map_or(Json::Null, ToJson::to_json)
        }
    }

    impl<T: ToJson> ToJson for Result<T, DimMapError> {
        fn to_json(&self) -> Json {
            match self {
                Ok(value) => value.to_json(),
                Err(error) => Json::Str(python_exception(error)),
            }
        }
    }

    /// The exception line Python prints for the exception each error stands for.
    fn python_exception(error: &DimMapError) -> String {
        match error {
            DimMapError::InvalidDim { dim, rank } => {
                format!("AssertionError: Invalid dim {dim} for rank {rank}")
            }
            DimMapError::InvalidViewShape(shape) => {
                format!("AssertionError: Invalid view shape {shape:?}")
            }
            DimMapError::ZeroDivision(_) => {
                "ZeroDivisionError: integer division or modulo by zero".to_owned()
            }
            DimMapError::InvalidMap => "AssertionError".to_owned(),
            DimMapError::InvalidPermuteDims(dims) => {
                format!("ValueError: Invalid permute dims: {dims:?}")
            }
        }
    }

    /// Runs one fixture query. `view` and `permute` lines open the case the lines after
    /// them query.
    fn replay(
        query: &str,
        args: &[Json],
        view: &mut Option<ViewMap>,
        permute: &mut Option<PermuteMap>,
    ) -> Json {
        fn open<T>(case: &Option<T>) -> &T {
            case.as_ref().expect("a query outside an open case")
        }
        match query {
            "view" => {
                let result = ViewMap::from_shapes(&args[0].ints(), &args[1].ints());
                let json = result
                    .as_ref()
                    .map(|map| {
                        (
                            map.source_shape().to_vec(),
                            map.target_shape().to_vec(),
                            map.is_valid_map(),
                        )
                    })
                    .map_err(Clone::clone)
                    .to_json();
                *view = result.ok();
                json
            }
            "source_to_target_axes" => open(view).source_to_target_axes().to_json(),
            "map_dim" => open(view).map_dim(&args[0].ints()).to_json(),
            "map_dim_inverse" => open(view).map_dim_inverse(&args[0].ints()).to_json(),
            "map_permutation" => open(view).map_permutation(&args[0].ints()).to_json(),
            "map_permutation_inverse" => open(view)
                .map_permutation_inverse(&args[0].ints())
                .to_json(),
            "remap_target_shape" => open(view).remap_target_shape(&args[0].ints()).to_json(),
            "map_reduction_after_view" => open(view)
                .map_reduction_after_view(&args[0].ints(), &args[1].ints())
                .to_json(),
            "map_reduction_before_view" => open(view)
                .map_reduction_before_view(&args[0].ints())
                .to_json(),
            "map_source_dims_to_target_axes" => open(view)
                .map_source_dims_to_target_axes(&args[0].ints())
                .to_json(),
            "remap_unit_slice" => open(view)
                .remap_unit_slice(
                    &args[0].ints(),
                    args[1].int(),
                    args[2].optional_int(),
                    args[3].optional_int(),
                    args[4].int(),
                )
                .to_json(),
            "permute" => {
                let result = PermuteMap::new(&args[0].ints());
                let json = result
                    .as_ref()
                    .map(|map| map.permute_dims().to_vec())
                    .map_err(Clone::clone)
                    .to_json();
                *permute = result.ok();
                json
            }
            "map_dims" => open(permute).map_dims(&args[0].ints()).to_json(),
            "map_dims_inverse" => open(permute).map_dims_inverse(&args[0].ints()).to_json(),
            "lift_to_rank" => open(permute)
                .lift_to_rank(&args[0].ints(), args[1].rank())
                .to_json(),
            "project_to_axes" => open(permute)
                .project_to_axes(&args[0].ints(), args[1].flag())
                .to_json(),
            "extend_permutation" => {
                let dims: Vec<usize> = args[0].list().iter().map(Json::rank).collect();
                extend_permutation(&dims, args[1].rank()).to_json()
            }
            _ => panic!("unknown query {query:?}"),
        }
    }

    /// Replays tests/fixtures/dim_maps_rc4_vectors.tsv: rc4's Python dim maps on seeded
    /// random cases up to rank 6, written by dim_maps_rc4_vectors.py next to it (see
    /// that script to regenerate).
    #[test]
    fn matches_rc4_python_vectors() {
        let vectors = include_str!("../../tests/fixtures/dim_maps_rc4_vectors.tsv");
        let mut view = None;
        let mut permute = None;
        let mut replayed: HashMap<&str, usize> = HashMap::new();
        for (index, line) in vectors.lines().enumerate() {
            if line.starts_with('#') {
                continue;
            }
            let columns: Vec<&str> = line.split('\t').collect();
            let [query, args, expected] = columns[..] else {
                panic!("line {}: expected three columns", index + 1);
            };
            let actual = replay(query, Json::parse(args).list(), &mut view, &mut permute);
            assert_eq!(actual, Json::parse(expected), "line {}: {line}", index + 1);
            *replayed.entry(query).or_default() += 1;
        }
        assert_eq!(replayed.len(), 17, "every query kind replays: {replayed:?}");
    }
}
