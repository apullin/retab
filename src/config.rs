fn env_is_set(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn env_any_is_set(names: &[&str]) -> bool {
    names.iter().any(|name| env_is_set(name))
}

fn env_equals(name: &str, value: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some(value)
}

fn env_any_equals(names: &[&str], value: &str) -> bool {
    names.iter().any(|name| env_equals(name, value))
}

fn env_any_var(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| std::env::var(name).ok())
}

pub(crate) fn profile_ingest_serialize_enabled() -> bool {
    env_is_set("RETAB_PROFILE_INGEST_SERIALIZE")
}

pub(crate) fn fold_fallback_file() -> Option<String> {
    std::env::var("ARM_FOLD_FALLBACK_FILE").ok()
}

pub(crate) fn u55_permute_log_file() -> Option<String> {
    std::env::var("RETAB_U55_PERMUTE_LOG").ok()
}

pub(crate) fn from_ir_debug_enabled() -> bool {
    env_any_equals(&["RETAB_FROM_IR_DEBUG", "ARM_FROM_IR_DEBUG"], "1")
}

pub(crate) fn from_ir_debug_is_set() -> bool {
    env_any_is_set(&["RETAB_FROM_IR_DEBUG", "ARM_FROM_IR_DEBUG"])
}

pub(crate) fn runner_logging_enabled() -> bool {
    env_any_is_set(&[
        "RETAB_DEBUG",
        "RETAB_LOG",
        "TOSA_ENGINE_DEBUG",
        "TOSA_ENGINE_LOG",
    ])
}

pub(crate) fn retab_debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_any_is_set(&["RETAB_DEBUG", "TOSA_ENGINE_DEBUG"]))
}

pub(crate) fn table_debug_enabled() -> bool {
    env_any_is_set(&["RETAB_TABLE_DEBUG", "ARM_TABLE_DEBUG"])
}

pub(crate) fn debug_slice_enabled() -> bool {
    env_any_equals(&["RETAB_DEBUG_SLICE", "ARM_TOSA_ENGINE_DEBUG_SLICE"], "1")
}

pub(crate) fn debug_cat_enabled() -> bool {
    env_any_equals(&["RETAB_DEBUG_CAT", "ARM_TOSA_ENGINE_DEBUG_CAT"], "1")
}

pub(crate) fn debug_node_probe() -> Option<String> {
    env_any_var(&["RETAB_DEBUG_NODE", "ARM_TOSA_ENGINE_DEBUG_NODE"])
}
