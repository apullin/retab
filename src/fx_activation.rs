use crate::fx_ir::IrNode;
use crate::fx_target::EluTargetKind;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EluParams {
    pub(crate) alpha: f64,
    pub(crate) scale: f64,
    pub(crate) input_scale: f64,
}

impl EluParams {
    pub(crate) fn from_node(node: &IrNode, kind: EluTargetKind) -> Self {
        match kind {
            EluTargetKind::Elu => Self {
                alpha: node.arg_or_kwarg_numeric_f64(1, "alpha").unwrap_or(1.0),
                scale: node.arg_or_kwarg_numeric_f64(2, "scale").unwrap_or(1.0),
                input_scale: node
                    .arg_or_kwarg_numeric_f64(3, "input_scale")
                    .unwrap_or(1.0),
            },
            EluTargetKind::Selu => Self {
                alpha: 1.673_263_242_354_377_2,
                scale: 1.050_700_987_355_480_5,
                input_scale: 1.0,
            },
            EluTargetKind::Celu => {
                let alpha = node.arg_or_kwarg_numeric_f64(1, "alpha").unwrap_or(1.0);
                Self {
                    alpha,
                    scale: 1.0,
                    input_scale: 1.0 / alpha,
                }
            }
        }
    }
}
