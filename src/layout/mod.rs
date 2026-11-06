//! ExecuTorch v1.6.0-rc4's explicit layout model, ported to the IR (APU-135):
//! every layout change is a permute or view node.
//!
//! P0 adds the infrastructure without changing the default pipeline; P1-P3
//! build the channels-last rewrites and the permute/view propagation passes on
//! top of it. Until those passes land, only tests reach these modules.

#[allow(dead_code)]
pub(crate) mod canonicalize;
// rc4's ComputeConstantOpsAOT/FuseConstantArgs slot for explicit mode (P2 S0).
#[allow(dead_code)]
pub(crate) mod constants;
#[cfg(test)]
mod canonicalize_rc4_tests;
#[cfg(test)]
mod canonicalize_upstream_tests;
#[allow(dead_code)]
pub(crate) mod ctx;
// Explicit mode's stock fallback for control-flow outputs whose contract target
// its copies hide, until S8.
pub(crate) mod control_flow_guard;
// rc4's round, sqrt and sign decompositions before phase 6 (APU-135 P2 S9).
pub(crate) mod decompositions;
#[allow(dead_code)]
pub(crate) mod dim_maps;
// The dtypes rc4 reads from fake tensors, from RETAB's qparams (APU-135 P2).
#[allow(dead_code)]
pub(crate) mod dtype;
#[allow(dead_code)]
pub(crate) mod fuse_duplicate_users;
#[allow(dead_code)]
pub(crate) mod fuse_identical_input_transforms;
#[allow(dead_code)]
pub(crate) mod fuse_view_copy;
#[allow(dead_code)]
pub(crate) mod fx_node;
#[allow(dead_code)]
pub(crate) mod high_rank_singleton_permute;
#[allow(dead_code)]
pub(crate) mod match_arg_ranks;
// rc4's meta rules for created nodes (`create_node`, `call_operator`).
#[allow(dead_code)]
pub(crate) mod meta;
#[allow(dead_code)]
pub(crate) mod move_data_movement;
#[allow(dead_code)]
pub(crate) mod normalize_delegate_io;
#[allow(dead_code)]
pub(crate) mod pending;
#[allow(dead_code)]
pub(crate) mod pointwise;
// rc4's pool argument helpers; IrDecomposeAvgPoolPad shares the pad adjustment.
#[allow(dead_code)]
pub(crate) mod pool;
#[allow(dead_code)]
pub(crate) mod remove_permutes;
// rc4's RewriteAvgPool2dPass for explicit mode (APU-156, P2 S7).
#[allow(dead_code)]
pub(crate) mod rewrite_avg_pool2d;
// rc4's RewriteConvPass for explicit mode (APU-135 P2 S3, APU-155).
#[allow(dead_code)]
pub(crate) mod rewrite_conv;
#[cfg(test)]
mod rewrite_conv_tests;
// rc4's RewriteMaxPool2dPass and RewriteUpsamplePass for explicit mode
// (APU-156, P2 S7).
#[allow(dead_code)]
pub(crate) mod rewrite_max_pool2d;
#[allow(dead_code)]
pub(crate) mod rewrite_upsample;
#[allow(dead_code)]
pub(crate) mod shape;

// rc4's own tests for the P1 passes, ported (APU-135 P1).
#[cfg(test)]
mod p1_upstream_tests;
// rc4's own tests for the pool and upsample rewrites, ported (APU-156, P2 S7).
#[cfg(test)]
mod p2_pool_upstream_tests;
// rc4 graph dumps as test fixtures and expectations.
#[cfg(test)]
pub(crate) mod rc4_dump;
// Helpers beyond those shape.rs's tests use are for the P1-P3 pass tests.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod testing;
// Fidelity review of the P1 ports against rc4.
#[cfg(test)]
mod p1_review_tests;
// rc4 replays: the harness, the P1 passes' replay tests and the P2 fixtures'
// sanity tests and replay hooks.
#[cfg(test)]
mod p1_replay_tests;
#[cfg(test)]
mod p2_replay_tests;
#[cfg(test)]
pub(crate) mod replay;
// rc4 replays of FuseDuplicateUsers, FuseIdenticalInputTransforms and
// MoveDataMovementOpsToSmallerDtype (APU-160).
#[cfg(test)]
mod fuse_helpers_replay_tests;
// rc4's own tests for those three passes, ported.
#[cfg(test)]
mod fuse_helpers_upstream_tests;
// Large-graph checks and timings for those three passes.
#[cfg(test)]
mod fuse_helpers_scaling_tests;
// Adversarial review of those three ports: rc4 behaviours they miss.
#[cfg(test)]
mod fuse_helpers_review_tests;
// rc4 replays of RemovePermutesAroundElementwiseTosaOps (APU-158).
#[cfg(test)]
mod remove_permutes_replay_tests;
// rc4's own tests for RemovePermutesAroundElementwise(Tosa)Ops (APU-158).
#[cfg(test)]
mod remove_permutes_upstream_tests;
