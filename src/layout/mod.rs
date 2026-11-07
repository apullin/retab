//! ExecuTorch v1.6.0-rc4's explicit layout model, ported to the IR (APU-135):
//! every layout change is a permute or view node. It is RETAB's only layout
//! model: the lowering pipelines (`fx_pipeline`) run these passes in rc4's
//! slots.
//!
//! Some items exist only for the ported upstream tests and the rc4 replays,
//! and are `#[cfg(test)]`: the standalone forms of passes the pipeline runs
//! inside another pass, the pass configurations rc4's own tests construct,
//! and accessors the tests read.

pub(crate) mod canonicalize;
// rc4's ComputeConstantOpsAOT/FuseConstantArgs slot (P2 S0).
pub(crate) mod constants;
#[cfg(test)]
mod canonicalize_rc4_tests;
#[cfg(test)]
mod canonicalize_upstream_tests;
pub(crate) mod ctx;
// The stock fallback for control-flow outputs whose contract target the layout
// copies hide and no record holds.
pub(crate) mod control_flow_guard;
// rc4's round, sqrt and sign decompositions before phase 6 (APU-135 P2 S9).
pub(crate) mod decompositions;
pub(crate) mod dim_maps;
// The dtypes rc4 reads from fake tensors, from RETAB's qparams (APU-135 P2).
pub(crate) mod dtype;
pub(crate) mod fuse_duplicate_users;
pub(crate) mod fuse_identical_input_transforms;
pub(crate) mod fuse_view_copy;
pub(crate) mod fx_node;
pub(crate) mod high_rank_singleton_permute;
pub(crate) mod match_arg_ranks;
// rc4's meta rules for created nodes (`create_node`, `call_operator`).
pub(crate) mod meta;
pub(crate) mod move_data_movement;
pub(crate) mod normalize_delegate_io;
pub(crate) mod pending;
pub(crate) mod pointwise;
// rc4's pool argument helpers; IrDecomposeAvgPoolPad shares the pad adjustment.
pub(crate) mod pool;
pub(crate) mod propagate;
pub(crate) mod remove_permutes;
// rc4's RewriteAvgPool2dPass (APU-156, P2 S7).
pub(crate) mod rewrite_avg_pool2d;
// rc4's RewriteConvPass (APU-135 P2 S3 and S4, APU-155).
pub(crate) mod rewrite_conv;
#[cfg(test)]
mod rewrite_conv_a16w8_tests;
#[cfg(test)]
mod rewrite_conv_tests;
// rc4's RewriteMaxPool2dPass and RewriteUpsamplePass (APU-156, P2 S7).
pub(crate) mod rewrite_max_pool2d;
pub(crate) mod rewrite_upsample;
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
// Graph builders and checks for the pass tests.
#[cfg(test)]
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
// rc4 replays of the propagation passes (APU-159).
#[cfg(test)]
mod propagate_replay_tests;
// rc4's own tests for the propagation passes, ported (APU-159).
#[cfg(test)]
mod propagate_upstream_tests;
// Large-graph checks and timings for the propagation passes.
#[cfg(test)]
mod propagate_scaling_tests;
// Adversarial review of the propagation port: rc4 behaviours it misses.
#[cfg(test)]
mod propagate_review_tests;
