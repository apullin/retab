fn env_is_set(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn env_equals(name: &str, value: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some(value)
}

pub(crate) fn profile_ingest_serialize_enabled() -> bool {
    env_is_set("RETAB_PROFILE_INGEST_SERIALIZE")
}

pub(crate) fn fold_fallback_file() -> Option<String> {
    std::env::var("ARM_FOLD_FALLBACK_FILE").ok()
}

pub(crate) fn from_ir_debug_enabled() -> bool {
    env_equals("ARM_FROM_IR_DEBUG", "1")
}

pub(crate) fn from_ir_debug_is_set() -> bool {
    env_is_set("ARM_FROM_IR_DEBUG")
}

pub(crate) fn runner_logging_enabled() -> bool {
    env_is_set("TOSA_ENGINE_DEBUG") || env_is_set("TOSA_ENGINE_LOG")
}

pub(crate) fn tosa_engine_debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_is_set("TOSA_ENGINE_DEBUG"))
}

pub(crate) fn table_debug_enabled() -> bool {
    env_is_set("ARM_TABLE_DEBUG")
}

pub(crate) fn debug_slice_enabled() -> bool {
    env_equals("ARM_TOSA_ENGINE_DEBUG_SLICE", "1")
}

pub(crate) fn debug_cat_enabled() -> bool {
    env_equals("ARM_TOSA_ENGINE_DEBUG_CAT", "1")
}

pub(crate) fn debug_layout_enabled() -> bool {
    env_equals("ARM_TOSA_ENGINE_DEBUG_LAYOUT", "1")
}

pub(crate) fn debug_node_probe() -> Option<String> {
    std::env::var("ARM_TOSA_ENGINE_DEBUG_NODE").ok()
}
