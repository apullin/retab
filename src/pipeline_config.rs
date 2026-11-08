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
    pub support_integer: bool,
    pub support_float: bool,
    pub tosa_dev_mode: bool,
    pub softmax: SoftmaxDecompositionConfig,
    pub leaky_relu: LeakyReluLoweringConfig,
    pub quantize_inf: QuantizeInfConfig,
    pub sdpa_safe_softmax_guard: SdpaSafeSoftmaxGuardPolicy,
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
            support_integer: features.contains(&"INT"),
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
        result.validate().map_err(PyValueError::new_err)?;
        Ok(result)
    }

    /// Whether the spec lists `extension` (`bf16`, `int16`, ...). rc4's
    /// `support_extension` also needs a profile that defines the extension.
    pub fn lists_extension(&self, extension: &str) -> bool {
        self.tosa_spec
            .split('+')
            .skip(1)
            .any(|token| token == extension)
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
        assert!(int.cf && int.support_integer);
        assert!(int.lists_extension("cf") && !int.lists_extension("int16"));
        let fp = PipelineConfig::from_tosa_spec("TOSA-1.0+INT+FP+int16");
        assert!(fp.support_float && fp.support_integer && fp.int16);
        assert!(!fp.cf && !fp.int4 && !fp.is_u55_subset);
        let fp_only = PipelineConfig::from_tosa_spec("TOSA-1.0+FP+bf16");
        assert!(fp_only.support_float && !fp_only.support_integer);
        assert!(fp_only.lists_extension("bf16") && !fp_only.lists_extension("bf"));
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
}
