use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

static FX_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static RECOMPILE_COUNT: AtomicU64 = AtomicU64::new(0);
static RECOMPILE_NS: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    *FX_PROFILE_ENABLED.get_or_init(|| std::env::var_os("ARM_TOSA_ENGINE_PROFILE_FX").is_some())
}

pub fn reset() {
    RECOMPILE_COUNT.store(0, Ordering::Relaxed);
    RECOMPILE_NS.store(0, Ordering::Relaxed);
}

pub fn record_recompile(elapsed: Duration) {
    RECOMPILE_COUNT.fetch_add(1, Ordering::Relaxed);
    RECOMPILE_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

pub fn recompile_count() -> u64 {
    RECOMPILE_COUNT.load(Ordering::Relaxed)
}

pub fn recompile_elapsed() -> Duration {
    Duration::from_nanos(RECOMPILE_NS.load(Ordering::Relaxed))
}
