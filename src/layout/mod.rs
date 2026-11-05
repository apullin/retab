//! ExecuTorch v1.6.0-rc4's layout design, ported to the IR (APU-135).
//!
//! P0 adds the infrastructure without changing the default pipeline; P1-P3
//! build the channels-last rewrites and the permute/view propagation passes on
//! top of it.

// P1-P3 use this; until then only its tests do.
#[allow(dead_code)]
pub(crate) mod ctx;
