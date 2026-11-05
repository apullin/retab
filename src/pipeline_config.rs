use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftmaxDecompositionConfig {
    Masked,
    Stable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeakyReluLoweringConfig {
    Decompose,
    Table,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdpaSafeSoftmaxGuardPolicy {
    Preserve,
    Remove,
    Auto,
}

/// The layout model the pipeline lowers with. APU-135 runs the two side by side until
/// explicit mode reaches parity, then deletes `DimOrder`.
///
/// - `DimOrder` (RETAB's model): nodes keep logical shapes, and `IrToTosaMemoryFormat`
///   gives them a channels-last `tosa_dim_order` that the serializer applies.
/// - `Explicit` (rc4's model): a node's val is its TOSA shape, and every layout change
///   is a permute node, lowered one-to-one to TRANSPOSE
///   (`backends/arm/_passes/aten_to_tosa_data_layout.py:60-68`).
///
/// Until P2 ports the explicit-mode passes, both modes build the same pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayoutMode {
    #[default]
    DimOrder,
    Explicit,
}

impl std::str::FromStr for LayoutMode {
    type Err = String;

    /// Parse the pipeline-config `layout` value.
    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "dim_order" => Ok(Self::DimOrder),
            "explicit" => Ok(Self::Explicit),
            _ => Err(format!(
                "unknown layout config: {name:?} (expected \"dim_order\" or \"explicit\")"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizeInfConfig {
    pub neg_inf: f64,
    pub pos_inf: f64,
}

impl Default for QuantizeInfConfig {
    fn default() -> Self {
        Self {
            neg_inf: -256.0,
            pos_inf: 255.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineConfig {
    pub tosa_spec: String,
    pub is_u55_subset: bool,
    pub int16: bool,
    pub int4: bool,
    pub cf: bool,
    pub support_float: bool,
    pub tosa_dev_mode: bool,
    pub softmax: SoftmaxDecompositionConfig,
    pub leaky_relu: LeakyReluLoweringConfig,
    pub quantize_inf: QuantizeInfConfig,
    pub sdpa_safe_softmax_guard: SdpaSafeSoftmaxGuardPolicy,
    /// Permutes that Vela can't run on U55 (rc4's probe, run in Python); split by
    /// `IrDecomposePermuteForU55Pass`.
    pub u55_split_permutes: std::collections::HashSet<String>,
    /// Pipeline-config key `layout`, which Python sets from `RETAB_LAYOUT`. The parent's
    /// config is stashed with its IR, so `ingest_and_serialize` (ingest and the
    /// serializer) gets the mode the passes ran in.
    pub layout: LayoutMode,
}

fn required<'py, T: FromPyObjectOwned<'py>>(config: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    config
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing pipeline config field: {key}")))?
        .extract()
        .map_err(Into::into)
}

impl PipelineConfig {
    pub fn from_tosa_spec(tosa_spec: &str) -> Self {
        let features: Vec<_> = tosa_spec.split('+').skip(1).collect();
        let is_u55_subset = features.contains(&"u55");
        Self {
            tosa_spec: tosa_spec.to_owned(),
            is_u55_subset,
            int16: features.contains(&"int16"),
            int4: features.contains(&"int4"),
            cf: features.contains(&"cf"),
            support_float: features.contains(&"FP"),
            tosa_dev_mode: false,
            softmax: if is_u55_subset {
                SoftmaxDecompositionConfig::Stable
            } else {
                SoftmaxDecompositionConfig::Masked
            },
            leaky_relu: LeakyReluLoweringConfig::Decompose,
            quantize_inf: QuantizeInfConfig::default(),
            sdpa_safe_softmax_guard: SdpaSafeSoftmaxGuardPolicy::Preserve,
            u55_split_permutes: Default::default(),
            layout: LayoutMode::default(),
        }
    }

    pub fn from_py(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let tosa_spec: String = required(config, "tosa_spec")?;
        let mut result = Self::from_tosa_spec(&tosa_spec);
        result.tosa_dev_mode = required(config, "tosa_dev_mode")?;
        result.softmax = match required::<String>(config, "softmax")?.as_str() {
            "MASKED" => SoftmaxDecompositionConfig::Masked,
            "STABLE" => SoftmaxDecompositionConfig::Stable,
            value => {
                return Err(PyValueError::new_err(format!(
                    "unknown softmax config: {value}"
                )))
            }
        };
        result.leaky_relu = match required::<String>(config, "leaky_relu")?.as_str() {
            "DECOMPOSE" => LeakyReluLoweringConfig::Decompose,
            "TABLE" => LeakyReluLoweringConfig::Table,
            value => {
                return Err(PyValueError::new_err(format!(
                    "unknown leaky_relu config: {value}"
                )))
            }
        };
        result.sdpa_safe_softmax_guard =
            match required::<String>(config, "sdpa_safe_softmax_guard")?.as_str() {
                "PRESERVE" => SdpaSafeSoftmaxGuardPolicy::Preserve,
                "REMOVE" => SdpaSafeSoftmaxGuardPolicy::Remove,
                "AUTO" => SdpaSafeSoftmaxGuardPolicy::Auto,
                value => {
                    return Err(PyValueError::new_err(format!(
                        "unknown sdpa_safe_softmax_guard config: {value}"
                    )))
                }
            };
        let quantize_inf: Bound<'_, PyDict> = required(config, "quantize_inf")?;
        result.quantize_inf = QuantizeInfConfig {
            neg_inf: required(&quantize_inf, "neg_inf")?,
            pos_inf: required(&quantize_inf, "pos_inf")?,
        };
        if let Some(names) = config.get_item("u55_split_permutes")? {
            result.u55_split_permutes = names.extract::<Vec<String>>()?.into_iter().collect();
        }
        if let Some(layout) = config.get_item("layout")? {
            result.layout = layout
                .extract::<String>()?
                .parse()
                .map_err(PyValueError::new_err)?;
        }
        result.validate().map_err(PyValueError::new_err)?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), String> {
        let guard = match self.sdpa_safe_softmax_guard {
            SdpaSafeSoftmaxGuardPolicy::Preserve => None,
            SdpaSafeSoftmaxGuardPolicy::Remove => Some("REMOVE"),
            SdpaSafeSoftmaxGuardPolicy::Auto => Some("AUTO"),
        };
        if let Some(guard) = guard {
            return Err(format!(
                "sdpa_safe_softmax_guard={guard} is not supported by RETAB; use stock preprocess"
            ));
        }
        if self.leaky_relu == LeakyReluLoweringConfig::Table {
            return Err("leaky_relu=TABLE is not supported by RETAB; use stock preprocess".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ethos_u55_and_u85_specs_keep_target_features_and_defaults() {
        let u55 = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+int16+int4+u55");
        let u85 = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+int16+int4+cf");

        assert!(u55.is_u55_subset);
        assert!(!u55.cf);
        assert!(!u85.is_u55_subset);
        assert!(u85.cf);
        assert!(u55.int16 && u85.int16);
        assert!(u55.int4 && u85.int4);
        assert!(!u55.support_float && !u85.support_float);
        assert_eq!(u55.tosa_spec, "TOSA-1.0+INT+int16+int4+u55");
        assert_eq!(u85.tosa_spec, "TOSA-1.0+INT+int16+int4+cf");
        assert_eq!(u55.softmax, SoftmaxDecompositionConfig::Stable);
        assert_eq!(u85.softmax, SoftmaxDecompositionConfig::Masked);
        assert_eq!(u55.quantize_inf.neg_inf, -256.0);
        assert_eq!(u55.quantize_inf.pos_inf, 255.0);
        assert_ne!(u55, u85);
    }

    #[test]
    fn extensions_and_float_support_require_exact_spec_tokens() {
        let int = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+cf");
        assert!(!int.int16 && !int.int4 && !int.is_u55_subset && !int.support_float);
        assert!(int.cf);
        let fp = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+FP+int16");
        assert!(fp.support_float && fp.int16);
        assert!(!fp.cf && !fp.int4 && !fp.is_u55_subset);
    }

    #[test]
    fn refuses_unported_guard_and_leaky_relu_policies() {
        let mut config = PipelineConfig::from_tosa_spec("TOSA-1.0+INT");
        assert!(config.validate().is_ok());
        for (policy, name) in [
            (SdpaSafeSoftmaxGuardPolicy::Remove, "REMOVE"),
            (SdpaSafeSoftmaxGuardPolicy::Auto, "AUTO"),
        ] {
            config.sdpa_safe_softmax_guard = policy;
            let error = config.validate().unwrap_err();
            assert!(error.contains(&format!("sdpa_safe_softmax_guard={name}")));
            assert!(error.contains("use stock preprocess"));
        }
        config.sdpa_safe_softmax_guard = SdpaSafeSoftmaxGuardPolicy::Preserve;
        config.leaky_relu = LeakyReluLoweringConfig::Table;
        assert!(config.validate().unwrap_err().contains("leaky_relu=TABLE"));
    }

    #[test]
    fn layout_defaults_to_dim_order_and_parses_only_exact_names() {
        for spec in ["TOSA-1.0+INT+int16+int4+u55", "TOSA-1.0+INT+int16+int4+cf"] {
            let config = PipelineConfig::from_tosa_spec(spec);
            assert_eq!(config.layout, LayoutMode::DimOrder);
            let explicit = PipelineConfig {
                layout: LayoutMode::Explicit,
                ..config
            };
            assert!(explicit.validate().is_ok());
        }
        assert_eq!("dim_order".parse(), Ok(LayoutMode::DimOrder));
        assert_eq!("explicit".parse(), Ok(LayoutMode::Explicit));
        for name in ["", "DIM_ORDER", "Explicit", "dim-order", " explicit"] {
            let error = name.parse::<LayoutMode>().unwrap_err();
            assert!(
                error.starts_with(&format!("unknown layout config: {name:?}")),
                "{error}"
            );
        }
    }
}
