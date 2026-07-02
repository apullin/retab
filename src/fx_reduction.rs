use crate::fx_ir::{normalize_dim, IrArg};
use crate::fx_meta::{NodeMeta, QParamMap};
use crate::fx_quant::QParam;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReductionStep {
    pub(crate) dim: i64,
    pub(crate) output_shape: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReductionDomain {
    input_shape: Vec<i64>,
    dims: Vec<usize>,
    keepdim: bool,
    output_shape: Vec<i64>,
}

impl ReductionDomain {
    pub(crate) fn from_arg(
        input_shape: Vec<i64>,
        dim_arg: Option<&IrArg>,
        keepdim: bool,
        skip_unit_dims: bool,
        output_shape: Option<Vec<i64>>,
    ) -> Self {
        let dims = normalized_dims(dim_arg, &input_shape, skip_unit_dims);
        let output_shape =
            output_shape.unwrap_or_else(|| reduced_shape_for_dims(&input_shape, &dims, keepdim));
        Self {
            input_shape,
            dims,
            keepdim,
            output_shape,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub(crate) fn dim_count(&self) -> usize {
        self.dims.len()
    }

    pub(crate) fn keepdim(&self) -> bool {
        self.keepdim
    }

    pub(crate) fn input_shape(&self) -> &[i64] {
        &self.input_shape
    }

    pub(crate) fn output_shape(&self) -> &[i64] {
        &self.output_shape
    }

    pub(crate) fn dims_i64(&self) -> Vec<i64> {
        self.dims.iter().map(|&dim| dim as i64).collect()
    }

    pub(crate) fn shape_with_keepdim(&self, keepdim: bool) -> Vec<i64> {
        reduced_shape_for_dims(&self.input_shape, &self.dims, keepdim)
    }

    pub(crate) fn input_rank_unit_shape(&self) -> Vec<i64> {
        vec![1; self.input_shape.len()]
    }

    pub(crate) fn coefficient_shape(&self) -> Vec<i64> {
        vec![1; self.output_shape.len().max(1)]
    }

    pub(crate) fn element_count_f64(&self) -> f64 {
        self.dims
            .iter()
            .map(|&dim| self.input_shape.get(dim).copied().unwrap_or(1) as f64)
            .product()
    }

    pub(crate) fn element_count_min_one_i64(&self) -> i64 {
        self.dims
            .iter()
            .map(|&dim| self.input_shape[dim].max(1))
            .product()
    }

    pub(crate) fn chained_single_axis_steps(&self) -> Vec<ReductionStep> {
        let mut dims = self.dims_i64();
        dims.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));

        let mut current_shape = self.input_shape.clone();
        dims.into_iter()
            .map(|dim| {
                let rank = current_shape.len();
                let normalized_dim = (dim as usize) % rank;
                let mut output_shape = current_shape.clone();
                if self.keepdim {
                    output_shape[normalized_dim] = 1;
                } else {
                    output_shape.remove(normalized_dim);
                }
                current_shape = output_shape.clone();
                ReductionStep { dim, output_shape }
            })
            .collect()
    }
}

pub(crate) fn normalized_dims(
    dim_arg: Option<&IrArg>,
    input_shape: &[i64],
    skip_unit_dims: bool,
) -> Vec<usize> {
    let rank_i64 = input_shape.len() as i64;
    let dims = dim_arg
        .and_then(IrArg::as_int_list_or_scalar)
        .unwrap_or_else(|| (0..rank_i64).collect());
    let mut normalized = Vec::new();

    for dim in dims {
        let Some(dim) = normalize_dim(dim, input_shape.len()) else {
            continue;
        };
        if skip_unit_dims && input_shape[dim] == 1 {
            continue;
        }
        if !normalized.contains(&dim) {
            normalized.push(dim);
        }
    }

    normalized.sort_unstable();
    normalized
}

pub(crate) fn reduced_shape_for_dims(
    input_shape: &[i64],
    dims: &[usize],
    keepdim: bool,
) -> Vec<i64> {
    if keepdim {
        let mut shape = input_shape.to_vec();
        for &dim in dims {
            shape[dim] = 1;
        }
        return shape;
    }

    let mut shape: Vec<i64> = input_shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &extent)| (!dims.contains(&idx)).then_some(extent))
        .collect();
    if shape.is_empty() {
        shape.push(1);
    }
    shape
}

pub(crate) struct SumQParamChain<'a> {
    source_meta: &'a NodeMeta,
    input_meta: &'a NodeMeta,
    current_input_qparams: Option<QParamMap>,
    output_qparams: Option<QParamMap>,
}

impl<'a> SumQParamChain<'a> {
    pub(crate) fn new(source_meta: &'a NodeMeta, input_meta: &'a NodeMeta) -> Self {
        Self {
            source_meta,
            input_meta,
            current_input_qparams: source_meta.cloned_input_qparams(),
            output_qparams: source_meta.cloned_output_qparams(),
        }
    }

    pub(crate) fn step_meta(&mut self, output_shape: &[i64], is_final: bool) -> NodeMeta {
        let mut step_meta = self
            .source_meta
            .cloned_with_dtype_of_or_float32(output_shape.to_vec(), self.input_meta);
        step_meta.clear_tosa_dim_order_if_rank_mismatch(output_shape.len());
        if let Some(ref qparams) = self.current_input_qparams {
            step_meta.set_input_qparams(qparams.clone());
        }
        if let Some(ref qparams) = self.output_qparams {
            step_meta.set_output_qparams(qparams.clone());
        }
        if is_final {
            step_meta.copy_qparams_from(self.source_meta);
        }
        self.current_input_qparams = self.output_qparams.clone();
        step_meta
    }
}

pub(crate) struct MeanQParamPlan {
    input_qparam: QParam,
    sum_qparam: QParam,
    coefficient_qparam: QParam,
    output_qparam: QParam,
}

impl MeanQParamPlan {
    pub(crate) fn from_meta(meta: &NodeMeta, count: f64) -> Option<Self> {
        if count == 0.0 {
            return None;
        }
        let input_qparam = meta.input_qparam()?.clone();
        let output_qparam = meta.output_qparam()?.clone();
        if meta
            .val_dtype_kind()
            .is_some_and(|dtype| !dtype.is_quantized_integer())
        {
            return None;
        }
        if !input_qparam.is_quantized_integer_dtype() || !output_qparam.is_quantized_integer_dtype()
        {
            return None;
        }
        let sum_qparam = QParam::per_tensor(
            input_qparam.scale * count,
            input_qparam.zp,
            input_qparam.qmin,
            input_qparam.qmax,
            input_qparam.dtype_str.clone(),
        );
        let coefficient_qparam = QParam::per_tensor(
            1.0 / (count * input_qparam.qmax as f64),
            0,
            input_qparam.qmin,
            input_qparam.qmax,
            input_qparam.dtype_str.clone(),
        );
        Some(Self {
            input_qparam,
            sum_qparam,
            coefficient_qparam,
            output_qparam,
        })
    }

    pub(crate) fn stamp_sum(&self, meta: &mut NodeMeta) {
        meta.set_unary_input_output_qparams(self.input_qparam.clone(), self.sum_qparam.clone());
    }

    pub(crate) fn stamp_coefficient(&self, meta: &mut NodeMeta) {
        meta.set_single_output_qparam(self.coefficient_qparam.clone());
    }

    pub(crate) fn stamp_mul(&self, meta: &mut NodeMeta) {
        meta.set_binary_input_output_qparams(
            self.sum_qparam.clone(),
            self.coefficient_qparam.clone(),
            self.output_qparam.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx_ir::IrArg;

    fn qparam(scale: f64, zp: i64) -> QParam {
        QParam::per_tensor(scale, zp, -128, 127, "torch.int8".to_string())
    }

    #[test]
    fn reduction_domain_normalizes_dims_and_shapes() {
        let domain = ReductionDomain::from_arg(
            vec![2, 1, 4, 5],
            Some(&IrArg::List(vec![IrArg::Int(-1), IrArg::Int(1)])),
            false,
            true,
            None,
        );

        assert_eq!(domain.dims_i64(), vec![3]);
        assert_eq!(domain.output_shape(), &[2, 1, 4]);
        assert_eq!(domain.shape_with_keepdim(true), vec![2, 1, 4, 1]);
        assert_eq!(domain.element_count_f64(), 5.0);
    }

    #[test]
    fn reduction_domain_builds_descending_chained_steps() {
        let domain = ReductionDomain::from_arg(
            vec![2, 3, 4, 5],
            Some(&IrArg::List(vec![IrArg::Int(1), IrArg::Int(3)])),
            false,
            false,
            None,
        );
        let steps = domain.chained_single_axis_steps();

        assert_eq!(
            steps,
            vec![
                ReductionStep {
                    dim: 3,
                    output_shape: vec![2, 3, 4],
                },
                ReductionStep {
                    dim: 1,
                    output_shape: vec![2, 4],
                },
            ]
        );
    }

    #[test]
    fn mean_qparams_stamp_sum_coefficient_and_mul_domains() {
        let mut meta = NodeMeta::tensor([2, 4], "torch.int8");
        meta.set_unary_input_output_qparams(qparam(0.25, 3), qparam(0.5, 5));
        let plan = MeanQParamPlan::from_meta(&meta, 4.0).unwrap();

        let mut sum_meta = NodeMeta::default();
        let mut coefficient_meta = NodeMeta::default();
        let mut mul_meta = NodeMeta::default();
        plan.stamp_sum(&mut sum_meta);
        plan.stamp_coefficient(&mut coefficient_meta);
        plan.stamp_mul(&mut mul_meta);

        assert_eq!(sum_meta.output_qparam().unwrap().scale, 1.0);
        assert_eq!(coefficient_meta.output_qparam().unwrap().zp, 0);
        assert_eq!(mul_meta.output_qparam().unwrap().scale, 0.5);
    }

    #[test]
    fn mean_qparams_ignore_float_output_domain() {
        let mut meta = NodeMeta::tensor([2, 4], "torch.float32");
        meta.set_unary_input_output_qparams(qparam(0.25, 3), qparam(0.5, 5));

        assert!(MeanQParamPlan::from_meta(&meta, 4.0).is_none());
    }
}
