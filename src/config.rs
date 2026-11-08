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
    std::env::var("RETAB_FOLD_FALLBACK_FILE").ok()
}

pub(crate) fn u55_permute_log_file() -> Option<String> {
    std::env::var("RETAB_U55_PERMUTE_LOG").ok()
}

pub(crate) fn from_ir_debug_enabled() -> bool {
    env_equals("RETAB_FROM_IR_DEBUG", "1")
}

pub(crate) fn from_ir_debug_is_set() -> bool {
    env_is_set("RETAB_FROM_IR_DEBUG")
}

pub(crate) fn runner_logging_enabled() -> bool {
    env_is_set("RETAB_DEBUG") || env_is_set("RETAB_LOG")
}

pub(crate) fn retab_debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_is_set("RETAB_DEBUG"))
}

/// Test hook: the name of the Python entry point that panics on entry, so tests can
/// check that a panic falls back to stock. Read on every call.
pub(crate) fn test_panic_entry_point() -> Option<String> {
    std::env::var("RETAB_TEST_PANIC").ok()
}

pub(crate) fn table_debug_enabled() -> bool {
    env_is_set("RETAB_TABLE_DEBUG")
}

pub(crate) fn debug_slice_enabled() -> bool {
    env_equals("RETAB_DEBUG_SLICE", "1")
}

pub(crate) fn debug_cat_enabled() -> bool {
    env_equals("RETAB_DEBUG_CAT", "1")
}

pub(crate) fn debug_node_probe() -> Option<String> {
    std::env::var("RETAB_DEBUG_NODE").ok()
}
