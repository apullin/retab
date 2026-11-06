//! What the explicit-mode pool rewrites share (APU-135 P2 S7, design §3.1 and
//! §3.2): rc4's argument helpers read from RETAB's IR arguments, the pad
//! adjustment, and the NHWC permute orders both pool rewrites (and the
//! upsample rewrite) wrap their TOSA op in.
//!
//! RETAB's FX import spells an empty list `IrArg::None` (`fx_ir::resolve_arg`),
//! so rc4's falsy `args[2]` (an omitted or `[]` stride) is either spelling here.

use crate::fx_ir::{IrArg, IrNode};

/// rc4's `NHWC_ORDER` (`backends/arm/constants.py`): NCHW to NHWC.
pub(crate) const NHWC_ORDER: [i64; 4] = [0, 2, 3, 1];
/// rc4's `NHWC_INVERSE_ORDER`: NHWC back to NCHW.
pub(crate) const NHWC_INVERSE_ORDER: [i64; 4] = [0, 3, 1, 2];

/// rc4's `to_2tuple` (`backends/arm/_passes/arm_pass_utils.py:811-817`) for a
/// pool's per-axis argument: an int or a one-element list applies to both
/// axes. rc4 returns a longer list unchanged and later unpacks it as two
/// values or validates it as `int[2]` (`_common.py:105-134`), so only a
/// two-element list passes; anything else is `None`, where rc4 raises.
pub(crate) fn to_2tuple(arg: &IrArg) -> Option<[i64; 2]> {
    match arg {
        IrArg::Int(value) => Some([*value; 2]),
        IrArg::List(_) => match arg.as_int_list()?.as_slice() {
            [value] => Some([*value; 2]),
            [h, w] => Some([*h, *w]),
            _ => None,
        },
        _ => None,
    }
}

/// Python truthiness of an argument, as `args[2] if len(args) > 2 and args[2]
/// else kernel` tests it (`rewrite_avg_pool2d_pass.py:39`,
/// `rewrite_max_pool2d_pass.py:81`): None, an empty list, zero and False are
/// falsy.
pub(crate) fn is_truthy(arg: &IrArg) -> bool {
    match arg {
        IrArg::None => false,
        IrArg::List(items) => !items.is_empty(),
        IrArg::Int(value) => *value != 0,
        IrArg::Float(value) => *value != 0.0,
        IrArg::Bool(value) => *value,
        IrArg::Node(_) | IrArg::Str(_) | IrArg::Dtype(_) => true,
    }
}

/// A pool's stride: `to_2tuple(args[2])` when it is truthy, else the kernel.
pub(crate) fn stride_or_kernel(node: &IrNode, kernel: [i64; 2]) -> Option<[i64; 2]> {
    match node.args.get(2) {
        Some(arg) if is_truthy(arg) => to_2tuple(arg),
        _ => Some(kernel),
    }
}

/// `ceil_mode` at `index`, False when absent; rc4 only tests its truthiness.
pub(crate) fn ceil_mode_arg(node: &IrNode, index: usize) -> Option<bool> {
    match node.args.get(index) {
        None | Some(IrArg::None) => Some(false),
        Some(IrArg::Bool(value)) => Some(*value),
        Some(_) => None,
    }
}

/// Whether every keyword argument is None. rc4's pool rewrites read
/// positional arguments only (`call_operator`'s `args`), so a keyword
/// argument with a value would be dropped silently there; exported graphs
/// never carry one, and the ports decline instead.
pub(crate) fn has_no_valued_kwargs(node: &IrNode) -> bool {
    node.kwargs
        .values()
        .all(|value| matches!(value, IrArg::None))
}

/// rc4's `adjust_pooling_pad_if_needed`
/// (`backends/arm/operators/operator_validation_utils.py:199-239`): the post
/// padding that makes TOSA's explicit pre/post padding reproduce aten's
/// symmetric `pad` and its ceil/floor output rounding. `stride` must be
/// positive (Python's `//` raises for zero; aten rejects it earlier).
pub(crate) fn adjust_pooling_pad_if_needed(
    input_size: i64,
    kernel_size: i64,
    stride: i64,
    pad: i64,
    ceil_mode: bool,
) -> i64 {
    let numerator = input_size - kernel_size + 2 * pad;
    let mut output_size = if ceil_mode {
        (numerator + stride - 1).div_euclid(stride) + 1
    } else {
        numerator.div_euclid(stride) + 1
    };
    if ceil_mode && (output_size - 1) * stride >= input_size + pad {
        output_size -= 1;
    }
    (output_size - 1) * stride - input_size + kernel_size - pad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::i64_list_arg;

    #[test]
    fn to_2tuple_expands_ints_and_one_element_lists() {
        assert_eq!(to_2tuple(&IrArg::Int(3)), Some([3, 3]));
        assert_eq!(to_2tuple(&i64_list_arg(&[2])), Some([2, 2]));
        assert_eq!(to_2tuple(&i64_list_arg(&[3, 2])), Some([3, 2]));
        // rc4 raises unpacking these as two values.
        assert_eq!(to_2tuple(&i64_list_arg(&[1, 2, 3])), None);
        assert_eq!(to_2tuple(&IrArg::List(vec![])), None);
        assert_eq!(to_2tuple(&IrArg::None), None);
        assert_eq!(to_2tuple(&IrArg::Float(2.0)), None);
    }

    #[test]
    fn falsy_strides_are_python_falsy_values() {
        for arg in [
            IrArg::None,
            IrArg::List(vec![]),
            IrArg::Int(0),
            IrArg::Bool(false),
        ] {
            assert!(!is_truthy(&arg), "{arg:?}");
        }
        for arg in [i64_list_arg(&[2, 2]), IrArg::Int(2), IrArg::Bool(true)] {
            assert!(is_truthy(&arg), "{arg:?}");
        }
    }

    /// rc4's function, run in .venv-et16: `(input, kernel, stride, pad, ceil)
    /// -> post pad`.
    #[test]
    fn adjust_pooling_pad_matches_rc4() {
        for (input, kernel, stride, pad, ceil, expected) in [
            (8, 3, 2, 1, false, 0),
            (8, 3, 2, 0, true, 1),
            (8, 2, 2, 0, false, 0),
            (8, 3, 1, 1, false, 1),
            (7, 2, 1, 0, false, 0),
            (5, 2, 2, 1, true, 0),
            (112, 3, 2, 1, true, 2),
            (56, 3, 1, 1, false, 1),
            (14, 3, 2, 1, true, 2),
            (5, 3, 3, 1, true, 0),
            // Negative when the input is one too long; rc4's
            // SizeAdjustInputPass slices it first (53 gives 0).
            (54, 3, 3, 1, false, -1),
        ] {
            assert_eq!(
                adjust_pooling_pad_if_needed(input, kernel, stride, pad, ceil),
                expected,
                "({input}, {kernel}, {stride}, {pad}, {ceil})"
            );
        }
    }
}
