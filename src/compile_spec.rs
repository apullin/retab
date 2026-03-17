/// Compilation target specification for the Arm Ethos-U backend.
#[derive(Debug, Clone)]
pub struct CompileSpec {
    /// Target hardware, e.g. "ethos-u55-128".
    pub target: String,
    /// System config name, e.g. "Ethos_U55_High_End_Embedded".
    pub system_config: String,
    /// Memory mode, e.g. "Shared_Sram".
    pub memory_mode: String,
    /// Any additional Vela compiler flags.
    pub extra_flags: Vec<String>,
}

impl CompileSpec {
    pub fn new(target: &str, system_config: &str, memory_mode: &str) -> Self {
        Self {
            target: target.to_owned(),
            system_config: system_config.to_owned(),
            memory_mode: memory_mode.to_owned(),
            extra_flags: Vec::new(),
        }
    }

    /// Convenience constructor for the common Ethos-U55-128 configuration.
    pub fn ethos_u55_128() -> Self {
        Self::new(
            "ethos-u55-128",
            "Ethos_U55_High_End_Embedded",
            "Shared_Sram",
        )
    }
}
