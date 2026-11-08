//! Permutation helpers shared by the fx passes and the layout passes.
//!
//! A permutation lists, for each output axis, the input axis it reads, as
//! `torch.permute` takes its dims: [`permute_by`]`(shape, p)[i]` is
//! `shape[p[i]]`.

/// The permutation that undoes `permutation`: `inverse[permutation[i]] == i`.
pub(crate) fn inverse_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; permutation.len()];
    for (index, &axis) in permutation.iter().enumerate() {
        inverse[axis] = index;
    }
    inverse
}

/// Whether `permutation` keeps every axis in place. It takes `usize` axes or
/// the `i64` dims of an FX argument; a negative dim is never in place.
pub(crate) fn is_identity_permutation<T: Copy>(permutation: &[T]) -> bool
where
    usize: TryFrom<T>,
{
    permutation
        .iter()
        .enumerate()
        .all(|(index, &axis)| usize::try_from(axis).is_ok_and(|axis| axis == index))
}

/// `values` in `permutation`'s order: element `i` is `values[permutation[i]]`.
/// Panics on an axis outside `values`.
pub(crate) fn permute_by<T: Copy>(values: &[T], permutation: &[usize]) -> Vec<T> {
    permutation.iter().map(|&axis| values[axis]).collect()
}

/// Axes as the `i64` dims an FX argument holds.
pub(crate) fn dims_to_i64(dims: &[usize]) -> Vec<i64> {
    dims.iter().map(|&dim| dim as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NHWC: [usize; 4] = [0, 2, 3, 1];

    #[test]
    fn inverse_undoes_the_permutation() {
        assert_eq!(inverse_permutation(&NHWC), vec![0, 3, 1, 2]);
        assert_eq!(
            inverse_permutation(&inverse_permutation(&NHWC)),
            NHWC.to_vec()
        );
    }

    #[test]
    fn identity_takes_usize_axes_and_i64_dims() {
        assert!(is_identity_permutation::<usize>(&[]));
        assert!(is_identity_permutation(&[0usize, 1, 2]));
        assert!(!is_identity_permutation(&NHWC));
        assert!(is_identity_permutation(&[0i64, 1, 2]));
        assert!(!is_identity_permutation(&[0i64, 2, 1]));
        assert!(!is_identity_permutation(&[-2i64, 1]));
    }

    #[test]
    fn permute_by_reads_each_axis_from_its_source() {
        assert_eq!(permute_by(&[1i64, 3, 4, 5], &NHWC), vec![1, 4, 5, 3]);
        assert_eq!(
            permute_by(&[1i64, 4, 5, 3], &inverse_permutation(&NHWC)),
            vec![1, 3, 4, 5]
        );
    }

    #[test]
    fn dims_convert_to_i64() {
        assert_eq!(dims_to_i64(&NHWC), vec![0i64, 2, 3, 1]);
    }
}
