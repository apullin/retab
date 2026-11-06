//! ExecuTorch v1.6.0-rc4's explicit layout model, ported to the IR (APU-135):
//! every layout change is a permute or view node.
//!
//! P0 adds the infrastructure without changing the default pipeline; P1-P3
//! build the channels-last rewrites and the permute/view propagation passes on
//! top of it. Until those passes land, only tests reach these modules.

#[allow(dead_code)]
pub(crate) mod canonicalize;
#[allow(dead_code)]
pub(crate) mod ctx;
#[allow(dead_code)]
pub(crate) mod dim_maps;
#[allow(dead_code)]
pub(crate) mod fuse_view_copy;
#[allow(dead_code)]
pub(crate) mod high_rank_singleton_permute;
#[allow(dead_code)]
pub(crate) mod match_arg_ranks;
#[allow(dead_code)]
pub(crate) mod normalize_delegate_io;
#[allow(dead_code)]
pub(crate) mod pointwise;
#[allow(dead_code)]
pub(crate) mod shape;
// Helpers beyond those shape.rs's tests use are for the P1-P3 pass tests.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod testing;
