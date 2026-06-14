# Rust Lowering Refactor Backlog

This backlog comes from a holistic audit of the Rust lowering engine after the
round-trip deletion. It is ranked for maximum engineering leverage with minimum
risk to the now-working lowering path.

Do not treat this as permission to start with a broad rewrite. The safest path
is to introduce shared grammar and helpers first, then migrate call sites in
small batches with full validation.

## Tracking Policy

Track every refactor slice with both validation evidence and LOC movement.
LOC is not the only objective, but it is a useful proxy for whether a refactor
is actually collapsing duplicated structure rather than just moving it around.

Starting Rust source checkpoint after the first target-classifier slice:

```text
2026-06-12
src/**/*.rs total: 39,940 lines
command: rg --files src | xargs wc -l
```

The first target-classifier slice is intentionally LOC-positive because it adds
the shared grammar and focused tests before migrating many call sites:

```text
tracked Rust diff: +30 / -23
new src/fx_target.rs: 123 lines
approximate net Rust LOC: +130
```

Future slices should record:

- Rust source LOC before and after.
- Patch-level additions and deletions.
- Which duplicated helpers or predicates were deleted.
- Exact tests run.
- Whether any LOC increase is deliberate setup for a later net deletion.

Slice log:

```text
2026-06-12, target classifier slice 1
Rust LOC: 39,940
Movement: approx +130 Rust LOC
Deleted: local Q/DQ helpers in fx_insert_qdq_rescale_ir.rs, local mm/bmm helper in fx_rewrite_matmul_ir.rs
Added: src/fx_target.rs, classifier-backed IrNode predicates, classifier tests
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib fx_insert_qdq_rescale_ir::tests; cargo test --lib fx_rewrite_matmul_ir::tests; cargo test --lib
Note: setup-positive slice; expected to pay down as more local target helpers migrate

2026-06-12, target classifier slice 2
Rust LOC: 39,969
Movement: +29 Rust LOC from previous checkpoint
Deleted: local Q/DQ target helpers in fx_fold_qdq_ir.rs
Added: per-channel Q/DQ classification and any-Q/DQ IrNode predicates
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_fold_qdq_ir::tests; cargo test --lib fx_target; cargo test --lib
Note: still setup-positive, but fold-q/dq now shares the Q/DQ grammar with the earlier rescale pass

2026-06-12, target classifier slice 3
Rust LOC: 39,992
Movement: +23 Rust LOC from previous checkpoint
Deleted: local relu/hardtanh helpers in fx_convert_clamp_ir.rs; local relu/hardtanh/q/dq/conv-or-linear helpers in fx_fuse_quant_activation_ir.rs
Added: activation and conv/linear producer target categories; classifier-backed IrNode predicates
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_relu.py -k tosa; pytest backends/arm/test/ops/test_clamp.py -k "tosa and (FP or INT)"
Note: still setup-positive, but activation passes now share the same target grammar and Python smoke covered the touched behavior

2026-06-12, target classifier slice 4
Rust LOC: 39,992
Movement: 0 Rust LOC from previous checkpoint
Deleted: local convolution target helper in fx_decompose_grouped_conv_ir.rs
Added: classifier-backed IrNode::is_convolution
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_decompose_grouped_conv_ir::tests; cargo test --lib
Note: kept boundary-rescale matching untouched because its old quantize/dequantize substring behavior needs separate semantic audit

2026-06-12, target classifier slice 5
Rust LOC: 39,983
Movement: -9 Rust LOC from previous checkpoint
Deleted: local per-tensor Q/DQ helpers in fx_decompose_quant_ir.rs
Added: no new classifier categories
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_decompose_quant_ir::tests; cargo test --lib
Note: first net-negative slice after the classifier setup; the common Q/DQ grammar now covers fold-q/dq, qdq-rescale insertion, and quant decomposition

2026-06-12, broad validation after target classifier slices 1-5
Rust LOC: 39,983
Command: scripts/coverage_arm_tests.sh --jobs auto --out target/coverage/refactor-target-classifier
Ops INT: 132 failed, 2192 passed, 3 skipped, 59 xfailed
Ops FP: 2 failed, 1909 passed, 13 skipped, 66 xfailed
Models INT: 58 passed, 17 skipped, 5 xfailed
Models FP: 81 passed, 2 skipped, 3 xfailed
Coverage: TOTAL regions 82.83%, functions 85.49%, lines 83.00%
Reports: target/coverage/refactor-target-classifier/summary.txt; target/coverage/refactor-target-classifier/html/index.html; target/coverage/refactor-target-classifier/lcov.info
Post-run restore: maturin develop --release was run separately and printed Installed retab-0.1.0
Note: ops failures match the current expected baseline buckets: debug flatc failures, a8w4 conv/linear coverage, and erfinv infinities

2026-06-12, target classifier slice 6
Rust LOC: 39,987
Movement: +4 Rust LOC from previous checkpoint
Deleted: raw `linear.default` target check in fx_decompose_linear_ir.rs
Added: classifier-backed IrNode::is_linear
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_linear.py -k "tosa and not a8w4"
Note: focused Python lane passed 72 tests with known a8w4 baseline-broken cases excluded

2026-06-12, target classifier slice 7
Rust LOC: 39,987
Movement: 0 Rust LOC from previous checkpoint
Deleted: raw `target.contains("quantize")` transparency check in fx_memory_format_ir.rs
Added: no new classifier categories
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py -k "tosa and not a8w4"
Note: focused conv2d lane passed 114 tests with known a8w4 baseline-broken cases excluded

2026-06-12, target classifier slice 8
Rust LOC: 39,984
Movement: -3 Rust LOC from previous checkpoint
Deleted: raw per-tensor Q/DQ target checks in fx_decompose_ops_ir.rs quantized-envelope guard
Added: no new classifier categories
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_decompose_ops_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_pow.py backends/arm/test/ops/test_tan.py -k tosa
Note: focused Python lane passed 52 tests with 3 expected xfails

2026-06-12, target classifier slice 9
Rust LOC: 39,996
Movement: +12 Rust LOC from previous checkpoint
Deleted: raw cond/while target checks in lib.rs IR-native control-flow boundary and region-metadata collection
Added: classifier-backed IrNode::is_cond, IrNode::is_while_loop, and IrNode::is_control_flow
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py
Note: cond/while reported 44 passed, 30 skipped, 10 xfailed, and the expected 6 U85 while failures from missing ethos-u-vela

2026-06-12, tensor-facts helper slice 1
Rust LOC: 39,966
Movement: -30 Rust LOC from previous checkpoint
Deleted: duplicate local `clear_qparams` helpers in fx_decompose_grouped_conv_ir.rs, fx_decompose_linear_ir.rs, and fx_conv1d_unsqueeze_ir.rs; duplicate local `copy_qparams_from_node` helpers in fx_decompose_linear_ir.rs and fx_conv1d_unsqueeze_ir.rs
Added: NodeMeta::set_empty_qparam_maps
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_decompose_grouped_conv_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_linear.py -k "tosa and not a8w4"; pytest backends/arm/test/ops/test_conv1d.py backends/arm/test/ops/test_conv2d.py -k "tosa and not a8w4"
Note: focused Python lanes passed 72 linear tests and 159 conv1d/conv2d tests with known a8w4 baseline-broken cases excluded; this is groundwork, not a structural LOC win

2026-06-12, target classifier slice 10
Rust LOC: 39,969
Movement: +3 Rust LOC from previous checkpoint
Deleted: duplicate zero-input factory and shape-only constant target helpers in fx_fuse_constant_inputs_ir.rs, fx_fuse_shape_only_constants_ir.rs, and fx_memory_format_ir.rs
Added: shared fx_target helpers for zero-input constants, shape-only constants, and broad view-like targets
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib fx_fuse; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_full.py backends/arm/test/ops/test_arange.py backends/arm/test/ops/test_eye.py backends/arm/test/ops/test_view.py backends/arm/test/ops/test_permute.py backends/arm/test/ops/test_unsqueeze.py -k tosa; pytest backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_addmm.py -k tosa
Note: focused Python lanes passed 151 factory/view tests and 67 mm/bmm/addmm tests

2026-06-12, constant graph helper slice 1
Rust LOC: 39,925
Movement: -44 Rust LOC from previous checkpoint
Deleted: duplicate constant-placeholder detection, recursive constant-arg traversal, constant input seed collection, and first-constant-input tensor lookup in fx_fuse_constant_inputs_ir.rs and fx_fuse_shape_only_constants_ir.rs
Added: shared fx_constant_tensor_eval helpers for constant input seeds, constant-input graph checks, and first constant input tensor lookup
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_fuse; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_full.py backends/arm/test/ops/test_arange.py backends/arm/test/ops/test_eye.py backends/arm/test/ops/test_view.py backends/arm/test/ops/test_permute.py backends/arm/test/ops/test_unsqueeze.py -k tosa; pytest backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_addmm.py -k tosa
Note: focused Python lanes again passed 151 factory/view tests and 67 mm/bmm/addmm tests

2026-06-12, zero-input factory slice 1
Rust LOC: 39,898
Movement: -27 Rust LOC from previous checkpoint
Deleted: repeated output-int-qparam/dtype selection and scalar-byte materialization logic in zero-input arange, linspace, and eye planners
Added: shared zero-input scalar-domain helpers in fx_fuse_zero_input_constants_ir.rs
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_fuse_zero_input_constants_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_full.py backends/arm/test/ops/test_arange.py backends/arm/test/ops/test_eye.py -k tosa
Note: focused Python lane passed 39 tests with 7 skips; linspace is covered by Rust unit tests because there is no focused Python ops file

2026-06-12, zero-input factory slice 2
Rust LOC: 39,879
Movement: -19 Rust LOC from previous checkpoint
Deleted: duplicate local arg-int-list and tensor-numel helpers in fx_fuse_zero_input_constants_ir.rs
Added: reuse of shared fx_constant_tensor_eval::arg_int_list and tensor_numel
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_fuse_zero_input_constants_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_full.py backends/arm/test/ops/test_arange.py backends/arm/test/ops/test_eye.py -k tosa
Note: focused Python lane again passed 39 tests with 7 skips

2026-06-12, constant graph helper slice 2
Rust LOC: 39,869
Movement: -10 Rust LOC from previous checkpoint
Deleted: duplicate local recursive tensor-input checks in fx_fuse_constant_inputs_ir.rs and fx_fuse_zero_input_constants_ir.rs
Added: shared fx_constant_tensor_eval::arg_has_node and node_has_tensor_inputs helpers
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_fuse; cargo test --lib
Note: no Python rebuild required for this Rust-only helper extraction until the next Python smoke checkpoint

2026-06-12, orphan rewrite pass deletion
Rust LOC: 39,313
Movement: -556 Rust LOC from previous checkpoint
Deleted: unreferenced src/fx_fuse_consecutive_rescales_ir.rs and src/fx_rewrite_high_rank_singleton_permute_ir.rs
Added: no new code
Validation: orphan-module scan over src/*.rs; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_constant_pad_nd.py backends/arm/test/ops/test_conv_constant_pad_nd.py -k tosa
Note: both files were tracked but not declared as modules, so they were not compiled, covered, or active in any pipeline

2026-06-12, serializer emitter dead-api cleanup
Rust LOC: 39,276
Movement: -37 Rust LOC from previous checkpoint
Deleted: unused EmitContext convenience wrappers and unused make_bool_output_tensor helper
Added: no new code
Validation: cargo fmt; cargo check --lib; cargo test --lib serialize::visitors
Note: removing the dead wrappers allowed the serializer emitter impl to drop #[allow(dead_code)] without warnings

2026-06-12, target classifier slice 11 and tuple-user helper
Rust LOC: 39,272
Movement: -4 Rust LOC from previous checkpoint
Deleted: raw rewrite-side getitem target checks; duplicate tuple-output getitem user collection in fx_decompose_ops_ir.rs; duplicate list-or-scalar int parsers in layernorm and var decomposition
Added: IrTargetKind::GetItem; IrNode::is_getitem; shared layout target predicates; IrArg::as_int_list_or_scalar; fx_decompose_ops_ir::getitem_users_of
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib fx_insert_conv_rescale_ir::tests; cargo test --lib fx_decompose_ops_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_layer_norm.py backends/arm/test/ops/test_var.py backends/arm/test/ops/test_batch_norm.py backends/arm/test/ops/test_group_norm.py backends/arm/test/ops/test_unbind.py backends/arm/test/ops/test_split.py backends/arm/test/ops/test_max_pool.py backends/arm/test/ops/test_amax.py backends/arm/test/ops/test_amin.py -k tosa
Note: focused Python lane passed 272 tests, 1 skipped, and 2 expected xfails

2026-06-12, graph tuple-user query slice
Rust LOC: 39,258
Movement: -14 Rust LOC from previous checkpoint
Deleted: pass-local getitem user collection in fx_decompose_ops_ir.rs, fx_decompose_layernorm_ir.rs, and fx_convert_split_to_slice_ir.rs
Added: IrGraph::live_getitem_users
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_decompose_ops_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_layer_norm.py backends/arm/test/ops/test_var.py backends/arm/test/ops/test_batch_norm.py backends/arm/test/ops/test_group_norm.py backends/arm/test/ops/test_unbind.py backends/arm/test/ops/test_split.py backends/arm/test/ops/test_max_pool.py backends/arm/test/ops/test_amax.py backends/arm/test/ops/test_amin.py -k tosa
Note: focused Python lane again passed 272 tests, 1 skipped, and 2 expected xfails

2026-06-12, target classifier slice 12 and constant matmul cleanup
Rust LOC: 39,252
Movement: -6 Rust LOC from previous checkpoint
Deleted: raw addmm/mm/bmm target checks and repeated addmm argument-offset expressions in constant matmul materialization; pass-local constant-matmul wrapper in fx_fuse_constant_inputs_ir.rs
Added: IrNode::is_mm, IrNode::is_bmm, and IrNode::is_addmm predicates
Validation: cargo fmt; cargo test --lib fx_fuse_constant_inputs_ir::tests; cargo test --lib fx_target; cargo check --lib; maturin develop --release; pytest backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_addmm.py -k tosa; cargo test --lib
Note: focused Python matmul family passed 67 tests; the slice keeps the existing addmm bias/alpha/beta behavior but moves op identity to the shared target grammar

2026-06-12, reduction shape helper slice
Rust LOC: 39,247
Movement: -5 Rust LOC from previous checkpoint
Deleted: pass-local reduction dim normalization and reduced-shape construction in fx_decompose_var_ir.rs; duplicated mean sum-shape construction in fx_decompose_mean_ir.rs
Added: shared fx_ir::normalized_reduction_dims and fx_ir::reduced_shape helpers
Validation: cargo fmt; cargo check --lib; maturin develop --release; pytest backends/arm/test/ops/test_mean_dim.py backends/arm/test/ops/test_var.py backends/arm/test/ops/test_layer_norm.py -k tosa; cargo test --lib
Note: focused reduction lane passed 112 tests; layernorm was included as an adjacent shape-sensitive smoke but its trailing-dim semantics were left unchanged

2026-06-12, graph call-user query slice
Rust LOC: 39,224
Movement: -23 Rust LOC from previous checkpoint
Deleted: duplicate pass-local call-function user scans in fx_fuse_quant_activation_ir.rs and fx_fuse_duplicate_users_ir.rs
Added: IrGraph::live_call_function_users, built on the existing live-user query so args, kwargs, and nested list references share one graph-query grammar
Validation: cargo fmt; cargo test --lib fx_fuse_duplicate_users_ir::tests; cargo check --lib; maturin develop --release; pytest backends/arm/test/ops/test_relu.py backends/arm/test/ops/test_clamp.py backends/arm/test/misc/test_conv_relu_residual_add.py -k tosa; cargo test --lib
Note: focused activation lane passed 72 tests; the dedicated duplicate-users filter currently selects no tests, so full Rust unit coverage is the meaningful Rust validation for this slice

2026-06-12, graph single-user query slice
Rust LOC: 39,223
Movement: -1 Rust LOC from previous checkpoint
Deleted: pass-local single-live-user helper in fx_insert_conv_rescale_ir.rs and exact-one user length/index checks in fx_fuse_quant_activation_ir.rs
Added: IrGraph::single_live_user and IrGraph::single_live_call_function_user
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_relu.py backends/arm/test/ops/test_clamp.py backends/arm/test/ops/test_conv2d.py backends/arm/test/misc/test_conv_relu_residual_add.py -k "tosa and not a8w4"
Note: focused conv/activation lane passed 186 tests; this is a small LOC win but moves exact-one user logic into the graph API instead of leaving each pass to spell it out

2026-06-12, IrArg node-reference helper slice
Rust LOC: 39,214
Movement: -9 Rust LOC from previous checkpoint
Deleted: duplicate recursive arg_contains_node helpers in fx_ir.rs live-user discovery and fx_insert_conv_rescale_ir.rs consumer-qparam recovery
Added: IrArg::contains_node
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_relu.py backends/arm/test/ops/test_clamp.py backends/arm/test/ops/test_conv2d.py backends/arm/test/misc/test_conv_relu_residual_add.py -k "tosa and not a8w4"
Note: focused conv/activation lane passed 186 tests; this moves recursive argument reference logic into the argument type instead of local graph/pass helpers

2026-06-12, convolution target grammar slice
Rust LOC: 39,240
Movement: +26 Rust LOC from previous checkpoint
Deleted: pass-local convolution/pool target classifiers in fx_annotate_conv_bias_ir.rs, fx_insert_conv_rescale_ir.rs, fx_size_adjust_input_ir.rs, and one transpose-conv predicate spelling in fx_memory_format_ir.rs
Added: shared fx_target helpers for conv-bias annotation targets, conv accumulator targets, raw ATen convolution targets, transpose-conv2d targets, and size-adjust targets
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib fx_insert_conv_rescale_ir::tests; cargo test --lib fx_annotate_conv_bias_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_max_pool.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/misc/test_conv_relu_residual_add.py -k "tosa and not a8w4"
Note: focused conv/pool lane passed 398 tests with 2 expected xfails; this is grammar consolidation rather than a LOC win

2026-06-12, NodeMeta tensor constructor slice
Rust LOC: 39,162
Movement: -78 Rust LOC from previous checkpoint
Deleted: repeated fresh NodeMeta + FakeTensorMeta construction in pixel shuffle, scalar shift rewrite, and several Rust test helpers
Added: NodeMeta::tensor(shape, dtype) for fresh tensor-valued metadata
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_pixel_shuffling.py backends/arm/test/ops/test_lshift.py backends/arm/test/ops/test_rshift.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_cumsum.py backends/arm/test/ops/test_ne.py backends/arm/test/ops/test_slice_scatter.py backends/arm/test/ops/test_constant_pad_nd.py -k "tosa and not a8w4"
Note: focused metadata-adjacent lane passed 175 tests; this is the first Tensor Facts step that names fresh tensor metadata construction without changing qparam or dirty-flag semantics

2026-06-13, NodeMeta tensor alias cleanup slice
Rust LOC: 39,104
Movement: -58 Rust LOC from previous checkpoint
Deleted: one-line local metadata aliases in scalar shift, ne, cumsum, addmm, qdq-rescale, pad, matmul, conv-bias, quant, slice-scatter, bool-mask index-put, and grouped-conv tests
Added: no new code
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_lshift.py backends/arm/test/ops/test_rshift.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_cumsum.py backends/arm/test/ops/test_ne.py backends/arm/test/ops/test_slice_scatter.py backends/arm/test/ops/test_constant_pad_nd.py backends/arm/test/ops/test_index_put.py -k "tosa and not a8w4"; git diff --check
Note: focused metadata-adjacent lane passed 206 tests with 2 expected xfails; larger fixture aliases were intentionally deferred and later removed in the NodeMeta test fixture alias cleanup slice

2026-06-13, decompose-op target matcher slice
Rust LOC: 39,060
Movement: -44 Rust LOC from previous checkpoint
Deleted: tiny `is_*` target predicate wrappers for the large decompose-op rule table
Added: local TargetMatcher value type so rule entries carry their string-match grammar directly, including explicit any/any-except matching
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_decompose_ops_ir::tests; cargo test --lib fx_target; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_tan.py backends/arm/test/ops/test_sinh.py backends/arm/test/ops/test_cosh.py backends/arm/test/ops/test_asinh.py backends/arm/test/ops/test_asin.py backends/arm/test/ops/test_acos.py backends/arm/test/ops/test_atan.py backends/arm/test/ops/test_acosh.py backends/arm/test/ops/test_elu.py backends/arm/test/ops/test_gelu.py backends/arm/test/ops/test_softmax.py backends/arm/test/ops/test_logsoftmax.py backends/arm/test/ops/test_batch_norm.py backends/arm/test/ops/test_group_norm.py backends/arm/test/ops/test_stack.py backends/arm/test/ops/test_unbind.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_sub.py backends/arm/test/ops/test_pow.py backends/arm/test/ops/test_amax.py backends/arm/test/ops/test_amin.py -k "tosa and not a8w4"; git diff --check
Note: focused decompose-rule lane passed 525 tests, 1 skipped, and 5 expected xfails; the rule order remains significant for log_softmax before softmax

2026-06-13, NodeMeta qparam accessor slice
Rust LOC: 39,045
Movement: -15 Rust LOC from previous checkpoint
Deleted: local indexed input-qparam helpers in matmul and conv rescale passes, local sorted-input-qparam helper in int32 rescale insertion, and direct input-qparam map access in constant matmul folding
Added: NodeMeta::input_qparam_at(index) and NodeMeta::sorted_input_qparams()
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_rewrite_matmul_ir::tests; cargo test --lib fx_insert_conv_rescale_ir::tests; cargo test --lib fx_insert_int32_rescale_ir::tests; cargo test --lib fx_fuse_constant_inputs_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_mul.py backends/arm/test/ops/test_where.py backends/arm/test/ops/test_pow.py -k "tosa and not a8w4"; git diff --check
Note: focused qparam-accessor lane passed 412 tests with 4 expected xfails; this is a Tensor Facts step that centralizes qparam map access without changing qparam propagation semantics

2026-06-13, decomposition target grammar slice
Rust LOC: 39,079
Movement: +34 Rust LOC from previous checkpoint
Deleted: pass-local target predicate helpers for split, tril, layernorm, var, avg_pool2d padding, as_strided_copy, non-constant pad rewrite, pixel shuffle/unshuffle, squeeze/unsqueeze, and transpose-copy/int rewrite
Added: shared fx_target helpers and focused classifier tests for those decomposition/rewrite target families
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_target; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_split.py backends/arm/test/ops/test_tril.py backends/arm/test/ops/test_layer_norm.py backends/arm/test/ops/test_var.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_as_strided_copy.py backends/arm/test/ops/test_constant_pad_nd.py backends/arm/test/ops/test_pixel_shuffling.py backends/arm/test/ops/test_squeeze.py backends/arm/test/ops/test_unsqueeze.py backends/arm/test/ops/test_transpose_copy.py -k tosa; git diff --check
Note: focused decomposition/rewrite lane passed 270 tests; this is a setup-positive grammar slice, so the next target should be deletion-heavy rather than another helper expansion

2026-06-13, broad coverage after decomposition target grammar slice
Rust LOC: 39,079
Command: scripts/coverage_arm_tests.sh --jobs auto --model-jobs 4 --out target/coverage/refactor-20260613-003638
Ops INT: 132 failed, 2192 passed, 3 skipped, 59 xfailed
Ops FP: 2 failed, 1909 passed, 13 skipped, 66 xfailed
Models INT: 58 passed, 17 skipped, 5 xfailed
Models FP: 81 passed, 2 skipped, 3 xfailed
Coverage: TOTAL regions 82.71%, functions 85.58%, lines 82.91%
Reports: target/coverage/refactor-20260613-003638/summary.txt; target/coverage/refactor-20260613-003638/html/index.html; target/coverage/refactor-20260613-003638/lcov.info
Post-run restore: maturin develop --release was run separately and printed Installed retab-0.1.0
Note: ops failures match the current expected baseline buckets; model lanes were explicitly capped at 4 workers to avoid OOM/swap pressure

2026-06-13, NodeMeta test fixture alias cleanup slice
Rust LOC: 39,057
Movement: -22 Rust LOC from previous checkpoint
Deleted: remaining one-line `meta(...)` fixture helpers in conv-rescale, int32-rescale, large decompose-op, and None-index-put tests
Added: no new code; tests now call `NodeMeta::tensor` directly for fresh tensor metadata
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_insert_conv_rescale_ir::tests; cargo test --lib fx_insert_int32_rescale_ir::tests; cargo test --lib fx_decompose_ops_ir::tests; cargo test --lib fx_normalize_index_put_none_indices_ir::tests; cargo test --lib; git diff --check
Note: this is Rust-test-only cleanup, so no Python rebuild was required; `MetaFactory::tensor_meta` remains because it carries dtype-from-source intent inside `ApproxCtx`

2026-06-13, config/debug env parsing slice
Rust LOC: 39,100
Movement: +43 Rust LOC from previous checkpoint
Deleted: scattered Rust env parsing in lib orchestration, runner logging, FX graph debug, from-IR debug, table debug, memory-format debug, and serializer debug probes
Added: `src/config.rs` as the single Rust owner for env flag names and exact matching semantics
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_insert_table_ops_ir::tests; cargo test --lib serialize::visitors; cargo test --lib fx_target::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85" with ARM_FROM_IR_DEBUG=1; pytest backends/arm/test/ops/test_add.py -k "tosa and INT" with ARM_TOSA_ENGINE_DEBUG_NODE=add; git diff --check
Note: control-flow debug smoke passed 29 tests with 1 expected xfail; INT debug-node smoke passed 25 tests; `rg 'std::env::var|std::env::var_os|env::var' src` now reports env reads only in `src/config.rs`

2026-06-13, stale write-back vocabulary cleanup slice
Rust LOC: 39,092
Movement: -8 Rust LOC from previous checkpoint
Deleted: stale source comments that described the removed flush/re-ingest path as live architecture
Added: no behavior code
Validation: cargo fmt; cargo check --lib; cargo test --lib; rg write_back/flush_ir/round-trip/re-ingest audit excluding untracked dSYM artifact; git diff --check
Note: remaining tracked-source `write_back` hits are live structural Python compatibility helpers; tensor-data round-trip comments are encode/decode tests, not lowering architecture

2026-06-13, from-IR architecture invariant tests slice
Rust LOC: 39,246
Movement: +154 Rust LOC from previous checkpoint
Deleted: no production code
Added: two `ingest::from_ir` tests covering TensorStore key prefixing with subgraph node names and topological ordering of IR-inserted nodes before rewired users
Validation: cargo fmt; cargo test --lib ingest::from_ir::tests; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py; git diff --check
Note: Rust tests passed 129/129. Cond/while reported 44 passed, 30 skipped, 10 xfailed, and the expected 6 U85 failures because ethos-u-vela is unavailable in this environment. This is a deliberate setup-positive test slice before more aggressive cleanup.

2026-06-13, RESCALE metadata Tensor Facts slice
Rust LOC: 39,220
Movement: -26 Rust LOC from previous checkpoint
Deleted: local RESCALE metadata builder helpers in q/dq-rescale, matmul-rescale, int32-rescale, and conv-rescale passes
Added: `NodeMeta::rescale_output_meta`, `NodeMeta::rescale_output_meta_with_shape`, and `NodeMeta::rescale_meta_with_input_qparam`
Validation: cargo fmt; cargo test --lib fx_insert_qdq_rescale_ir::tests; cargo test --lib fx_rewrite_matmul_ir::tests; cargo test --lib fx_insert_int32_rescale_ir::tests; cargo test --lib fx_insert_conv_rescale_ir::tests; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_where.py backends/arm/test/ops/test_conv2d.py -k "tosa and not a8w4"; git diff --check
Note: focused Python smoke passed 209 tests with 552 deselected; this keeps qparam/dtype/layout restamping semantics in one `NodeMeta` operation instead of pass-local clones.

2026-06-13, plain cloned metadata Tensor Facts slice
Rust LOC: 39,213
Movement: -7 Rust LOC from previous checkpoint
Deleted: pass-local `meta_plain` helper in unfold decomposition and repeated `cloned_with_val(...).without_qparams()` chains in unfold and avg-pool padding decomposition
Added: `NodeMeta::cloned_without_qparams(shape, dtype)` for cloned shape/dtype metadata that must explicitly drop qparam domains
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_unfold_copy.py backends/arm/test/ops/test_avg_pool2d.py -k "tosa and not u85 and not u55"; git diff --check
Note: focused Python smoke passed 89 tests with 189 deselected; the slice is intentionally narrow and preserves branches that copy qparams from source tensors.

2026-06-13, plain cloned metadata migration slice
Rust LOC: 39,174
Movement: -39 Rust LOC from previous checkpoint
Deleted: remaining production `cloned_with_val(...).without_qparams()` call sites outside the `NodeMeta` helper itself, plus the pass-local quant-decompose metadata wrapper
Added: no new code beyond reuse of `NodeMeta::cloned_without_qparams`
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_index_tensor.py backends/arm/test/ops/test_index_put.py backends/arm/test/ops/test_to_copy.py -k "tosa and not u85 and not u55"; git diff --check
Note: focused Python smoke passed 114 tests with 21 expected xfails and 111 deselected; `rg -U 'cloned_with_val...without_qparams' src` now reports only the helper implementation.

2026-06-13, clone-restamp metadata migration slice
Rust LOC: 39,134
Movement: -40 Rust LOC from previous checkpoint
Deleted: direct `FakeTensorMeta` restamping in sum conversion, split-to-slice conversion, as_strided_copy decomposition, addmm decomposition, linear decomposition, conv1d unsqueeze, and cumsum decomposition
Added: no new code beyond reuse of `NodeMeta::cloned_with_val`
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_sum.py backends/arm/test/ops/test_as_strided_copy.py backends/arm/test/ops/test_split.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_linear.py backends/arm/test/ops/test_conv1d.py backends/arm/test/ops/test_cumsum.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 242 tests with 569 deselected; qparam clearing/copying remains explicit after shape/dtype restamping where those passes need it.

2026-06-13, where/repeat clone-restamp metadata slice
Rust LOC: 39,120
Movement: -14 Rust LOC from previous checkpoint
Deleted: direct `FakeTensorMeta` restamping in where-mask expansion and repeat_interleave decomposition
Added: no new code beyond reuse of `NodeMeta::cloned_with_val`
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_where.py backends/arm/test/ops/test_repeat.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 50 tests with 82 deselected.

2026-06-13, metadata helper restamp cleanup slice
Rust LOC: 39,104
Movement: -16 Rust LOC from previous checkpoint
Deleted: direct `FakeTensorMeta` restamping in bool-mask index_put normalization, grouped-conv slice metadata, memory-format transpose metadata, and dim-extrema decomposition metadata
Added: no new code beyond reuse of `NodeMeta::cloned_with_val` and existing qparam/layout restamping helpers
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_index_put.py backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_permute.py backends/arm/test/ops/test_batch_norm.py backends/arm/test/ops/test_group_norm.py backends/arm/test/ops/test_amax.py backends/arm/test/ops/test_amin.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 310 tests, 1 skipped, and 4 expected xfails with 744 deselected.

2026-06-13, rescale and size-adjust metadata cleanup slice
Rust LOC: 39,084
Movement: -20 Rust LOC from previous checkpoint
Deleted: direct `FakeTensorMeta` restamping in size-adjust inserted slice metadata, conv-rescale plain tensor metadata, and one test fixture in index_put rewrite
Added: no new code beyond reuse of `NodeMeta::cloned_with_val` and `NodeMeta::tensor`
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_max_pool.py backends/arm/test/ops/test_avg_pool2d.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 370 tests with 1 expected xfail and 991 deselected; remaining `FakeTensorMeta` uses are effectful existing-node metadata updates or comments rather than simple clone-restamp helpers.

2026-06-13, index/mask target grammar slice
Rust LOC: 39,105
Movement: +21 Rust LOC from previous checkpoint
Deleted: local `where` predicate in mask expansion, raw index_put/default-index-put target checks in the index_put normalization/rewrite passes, raw clamp-default target check in clamp-arg quantization, raw addmm target check in addmm decomposition, and duplicate sum dim-list parsing
Added: shared `fx_target` helpers for index_put family/default targets, where targets, and clamp targets; reused existing `IrNode::is_addmm` and `IrArg::as_int_list_or_scalar`
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_index_put.py backends/arm/test/ops/test_where.py backends/arm/test/ops/test_clamp.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_sum.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 188 tests with 2 expected xfails and 303 deselected. This is a small setup-positive target grammar slice; the next slice should prefer deletion-only reuse over adding new helper families.

2026-06-13, effectful dtype-restamp Tensor Facts slice
Rust LOC: 39,097
Movement: -8 Rust LOC from previous checkpoint
Deleted: remaining production pass-local `FakeTensorMeta` dtype restamp blocks in int64 buffer casting, INT32 rescale repair, and control-flow placeholder/restamp paths; one raw where target check in INT32 output rescale repair
Added: `NodeMeta::set_val_dtype_for_existing_shape` for in-place dtype restamps that preserve the existing tensor shape
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_where.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_mul.py backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55 and not a8w4"; rg set_val(FakeTensorMeta/ref mut ftm/dirty.val direct-restamp audit; git diff --check
Note: Rust tests passed 130/130 and focused Python smoke passed 137 tests with 1 expected xfail and 196 deselected. Direct `FakeTensorMeta` construction now remains in `NodeMeta` internals rather than production pass dtype-restamp code.

2026-06-13, constant arg-list parser cleanup slice
Rust LOC: 39,088
Movement: -9 Rust LOC from previous checkpoint
Deleted: duplicate `fx_constant_tensor_eval::arg_int_list` helper and its import from zero-input constant materialization
Added: no new helper surface; callers now reuse `IrArg::as_int_list`
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_constant_tensor_eval::tests; cargo test --lib fx_fuse_zero_input_constants_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_full.py backends/arm/test/ops/test_arange.py backends/arm/test/ops/test_eye.py backends/arm/test/ops/test_view.py backends/arm/test/ops/test_permute.py backends/arm/test/ops/test_unsqueeze.py -k "tosa and not u85 and not u55 and not a8w4"; rg arg_int_list audit; git diff --check
Note: focused Python smoke passed 151 tests with 7 skips and 254 deselected. This removes a redundant argument-list grammar from constant materialization and leaves list parsing owned by `IrArg`.

2026-06-13, scalar IrArg parser cleanup slice
Rust LOC: 39,059
Movement: -29 Rust LOC from previous checkpoint
Deleted: duplicate int/float scalar parsers in clamp conversion, clamp-arg quantization, addmm decomposition, zero-input constant materialization, constant matmul materialization, table-op insertion, and quantized activation fusion
Added: `IrArg::as_numeric_f64` for int/float-only scalar parsing and `IrArg::as_scalar_literal_f64` for int/float/bool literal parsing where prior code accepted bool
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_fuse_zero_input_constants_ir::tests; cargo test --lib fx_decompose_addmm_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_clamp.py backends/arm/test/ops/test_relu.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_full.py backends/arm/test/ops/test_arange.py backends/arm/test/ops/test_eye.py backends/arm/test/ops/test_pow.py backends/arm/test/ops/test_remainder.py backends/arm/test/ops/test_elu.py -k "tosa and not u85 and not u55 and not a8w4"; targeted scalar-parser audit; git diff --check
Note: focused Python smoke passed 236 tests with 7 skips and 6 expected xfails. Bool-to-f64 semantics were preserved only for zero-input/constant scalar literals, while clamp/addmm/table paths keep int/float-only parsing.

2026-06-13, decompose-op scalar parser cleanup slice
Rust LOC: 39,041
Movement: -18 Rust LOC from previous checkpoint
Deleted: repeated int/float scalar parser chains in large decompose-op helper functions for pow, add/sub alpha, elu alpha, batch-norm eps, and group-norm eps
Added: no new helper surface; callers reuse `IrArg::as_numeric_f64`
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_decompose_ops_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_add.py backends/arm/test/ops/test_sub.py backends/arm/test/ops/test_pow.py backends/arm/test/ops/test_elu.py backends/arm/test/ops/test_batch_norm.py backends/arm/test/ops/test_group_norm.py -k "tosa and not u85 and not u55 and not a8w4"; decompose-op scalar-parser audit; git diff --check
Note: Rust decompose-op tests passed 21/21, full Rust tests passed 130/130, and focused Python smoke passed 222 tests with 1 skip and 3 expected xfails. This keeps the scalar grammar centralized without changing decomposition rule matching.

2026-06-13, optional-int helper cleanup slice
Rust LOC: 39,032
Movement: -9 Rust LOC from previous checkpoint
Deleted: duplicate optional-int wrapper helpers in slice_scatter and strided_slice decomposition
Added: no new helper surface; callers reuse `IrArg::as_int` directly
Validation: cargo fmt; cargo check --lib; cargo test --lib fx_decompose_slice_scatter_ir::tests; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_slice_scatter.py backends/arm/test/ops/test_slice.py -k "tosa and not u85 and not u55 and not a8w4"; optional-int helper audit; git diff --check
Note: focused Python smoke passed 54 tests with 107 deselected. The similar-looking slice bound fixup helpers were intentionally not unified because slice_scatter and strided_slice differ on negative-bound semantics.

2026-06-13, NN serializer zero-bias helper slice
Rust LOC: 38,980
Movement: -52 Rust LOC from previous checkpoint
Deleted: duplicate synthetic zero-bias dtype, output-channel, zero-byte, and tensor emission helpers from conv and transpose-conv visitors
Added: one shared NN visitor helper cluster for synthetic conv bias tensors
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 240 tests with 2 expected xfails and 785 deselected. Transpose-conv output dtype now calls the same shared FP-or-accumulator dtype rule it previously duplicated inline.

2026-06-13, NN serializer conv-input helper slice
Rust LOC: 38,966
Movement: -14 Rust LOC from previous checkpoint
Deleted: duplicate input/weight zero-point lookup and conv input assembly branches in conv and transpose-conv visitors
Added: shared NN visitor helpers for input/weight zero-point recovery and bias/zp input preparation
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 240 tests with 2 expected xfails and 785 deselected. The helper preserves the prior distinction between ATen conv zero points from qparams only and TOSA/transpose-conv zero points with kwarg fallback.

2026-06-13, serializer i32-list fallback helper slice
Rust LOC: 38,906
Movement: -60 Rust LOC from previous checkpoint
Deleted: repeated kwarg-i32-list-or-positional-i32-list fallback parsing in conv, transpose-conv, rescale shape planning, avg-pool emission, and pool shape inference
Added: `kwarg_or_arg_i32_list` in serializer arg utilities
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py -k "tosa and not u85 and not u55 and not a8w4"; fallback-parser audit; git diff --check
Note: focused Python smoke passed 395 tests with 2 expected xfails and 1062 deselected. Avg-pool pad parsing remains local because kwarg padding is already full TOSA pad while positional padding still needs symmetric expansion.

2026-06-13, NN serializer symmetric-pad helper slice
Rust LOC: 38,889
Movement: -17 Rust LOC from previous checkpoint
Deleted: duplicate spatial symmetric-pad expansion helpers in conv emission and rescale input-shape planning
Added: shared private `symmetric_spatial_pad` helper in the NN serializer module
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py -k "tosa and not u85 and not u55 and not a8w4"; symmetric-pad helper audit; git diff --check
Note: focused Python smoke passed 395 tests with 2 expected xfails and 1062 deselected. The helper preserves the prior behavior for already-expanded pads and for short symmetric pad lists.

2026-06-13, serializer i32-list default helper slice
Rust LOC: 38,838
Movement: -51 Rust LOC from previous checkpoint
Deleted: local empty-list default helpers and repeated empty-list default branches in conv emission, rescale shape planning, max/avg pool parsing, and pool shape inference
Added: `i32_list_or` in serializer arg utilities
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py -k "tosa and not u85 and not u55 and not a8w4"; default-list helper audit; git diff --check
Note: focused Python smoke passed 395 tests with 2 expected xfails and 1062 deselected. The helper only encodes the common empty-Vec fallback; each caller still supplies its own domain default.

2026-06-13, pool symmetric-pad reuse slice
Rust LOC: 38,810
Movement: -28 Rust LOC from previous checkpoint
Deleted: duplicate positional pool padding expansion in avg-pool emission and max-pool adjusted padding
Added: wider `symmetric_pool_pad` reuse from shape inference into pool emission
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py -k "tosa and not u85 and not u55 and not a8w4"; pool-pad helper audit; git diff --check
Note: focused Python smoke passed 155 tests with 277 deselected. Keyword avg-pool padding remains passed through unchanged, while positional avg-pool padding uses the shared symmetric expansion.

2026-06-13, serializer arg-first i32-list fallback slice
Rust LOC: 38,809
Movement: -1 Rust LOC from previous checkpoint
Deleted: local arg-i32-list-or-kwarg-i32-list fallback branches in transpose permutation shape handling and transpose-conv output-padding parsing
Added: `arg_or_kwarg_i32_list` in serializer arg utilities
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_transpose_copy.py backends/arm/test/ops/test_permute.py -k "tosa and not u85 and not u55 and not a8w4"; arg-list fallback audit
Note: focused Python smoke passed 76 tests with 1 expected xfail and 153 deselected. The new helper is deliberately separate from `kwarg_or_arg_i32_list` because these two call sites preserve arg-first precedence.

2026-06-13, pool post-pad formula reuse slice
Rust LOC: 38,799
Movement: -10 Rust LOC from previous checkpoint
Deleted: duplicate max-pool post-padding formula in pool emission
Added: wider `pool_post_pad` reuse from shape inference into pool emission
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py -k "tosa and not u85 and not u55 and not a8w4"; pool post-pad helper audit
Note: focused Python smoke passed 155 tests with 277 deselected. Shape inference and emission now share the same ceil-mode post-padding math.

2026-06-13, NN serializer ATen conv-group parser slice
Rust LOC: 38,794
Movement: -5 Rust LOC from previous checkpoint
Deleted: duplicate ATen convolution groups-argument parser in conv emission and rescale input-shape planning
Added: shared private `aten_conv_groups` helper in the NN serializer module
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py -k "tosa and not u85 and not u55 and not a8w4"; conv-group helper audit
Note: focused Python smoke passed 240 tests with 2 expected xfails and 785 deselected. Depthwise detection remains split because conv emission and rescale shape planning intentionally use different evidence.

2026-06-13, NN serializer input-rank helper slice
Rust LOC: 38,784
Movement: -10 Rust LOC from previous checkpoint
Deleted: duplicate first-input rank probing in conv emission and rescale input-shape planning
Added: shared private `input_rank` helper in the NN serializer module
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py -k "tosa and not u85 and not u55 and not a8w4"; input-rank helper audit
Note: focused Python smoke passed 240 tests with 2 expected xfails and 785 deselected. This only centralizes first-input rank lookup; rank-specific branch behavior is unchanged.

2026-06-13, NN serializer conv default-list grammar slice
Rust LOC: 38,772
Movement: -12 Rust LOC from previous checkpoint
Deleted: repeated conv stride/dilation/pad default-list construction in conv emission and rescale input-shape planning
Added: shared private `i32_list_or_repeat`, `conv_stride_arg`, `conv_dilation_arg`, and `conv_pad_arg` helpers in the NN serializer module
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py -k "tosa and not u85 and not u55 and not a8w4"; conv default-list helper audit
Note: focused Python smoke passed 240 tests with 2 expected xfails and 785 deselected. The helper scope is intentionally conv-only; non-conv list defaults keep their existing parsing grammar.

2026-06-13, NN serializer conv zero-point wrapper cleanup slice
Rust LOC: 38,762
Movement: -10 Rust LOC from previous checkpoint
Deleted: one-line `ParsedConvSpec` zero-point forwarding methods
Added: no new helper surface; conv parsing now calls `input_weight_zero_points` directly with the existing kwarg-fallback mode
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py -k "tosa and not u85 and not u55 and not a8w4"; zero-point helper audit
Note: focused Python smoke passed 240 tests with 2 expected xfails and 785 deselected. This keeps ATen conv on qparam-only zero points and TOSA/transpose-conv on qparam-with-kwarg-fallback zero points.

2026-06-13, NN serializer zero-point tensor emission slice
Rust LOC: 38,734
Movement: -28 Rust LOC from previous checkpoint
Deleted: duplicate rank-1 zero-point tensor construction in conv input preparation and avg-pool emission
Added: shared private `push_zp_tensor` helper in the NN serializer module
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py -k "tosa and not u85 and not u55 and not a8w4"; NN zero-point tensor helper audit
Note: focused Python smoke passed 395 tests with 2 expected xfails and 1062 deselected. The remaining `TosaTensor::with_data` use in `nn.rs` is zero-bias tensor data, not zero-point emission.

2026-06-13, serializer zero-point const helper slice
Rust LOC: 38,704
Movement: -30 Rust LOC from previous checkpoint
Deleted: duplicate rank-1 encoded zero-point const construction in unary negate, matmul, tensor direct-rescale parameter emission, and the NN helper
Added: serializer-wide `push_zero_point_const` helper beside `push_rank1_const`
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_neg.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_matmul.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_upsample_nearest2d.py backends/arm/test/ops/test_upsample_bilinear2d.py backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py -k "tosa and not u85 and not u55 and not a8w4"; serializer zero-point helper audit
Note: focused Python smoke passed 766 tests with 12 expected xfails and 1581 deselected. `nn/rescale` keeps its own helper because RESCALE zero points are normalized for unsigned lanes before encoding.

2026-06-13, serializer qparam zero-point accessor slice
Rust LOC: 38,690
Movement: -14 Rust LOC from previous checkpoint
Deleted: repeated metadata-local input/output qparam zero-point scans in unary negate, matmul, and NN conv input parsing
Added: `input_qparam_zero_point`, `input_qparam_zero_point_or_kwarg`, and `output_qparam_zero_point` accessors in serializer dtype/qparam utilities
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_neg.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_matmul.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py -k "tosa and not u85 and not u55 and not a8w4"; qparam zero-point accessor audit
Note: focused Python smoke passed 390 tests with 12 expected xfails and 1096 deselected. Producer-aware `node_input_zero_point` remains separate; this slice only centralizes metadata-local qparam lookup and qparam-or-kwarg fallback.

2026-06-13, serializer scalar qparam quantization helper slice
Rust LOC: 38,692
Movement: +2 Rust LOC from previous checkpoint
Deleted: repeated scalar quantize-and-clamp arithmetic in constant fill, indexing broadcast constants, and pad constants
Added: `quantize_scalar_with_qparam` in serializer const utilities
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_full.py backends/arm/test/ops/test_zeros.py backends/arm/test/ops/test_ones.py backends/arm/test/ops/test_index_tensor.py backends/arm/test/ops/test_index_put.py backends/arm/test/ops/test_index_select.py backends/arm/test/ops/test_index_copy.py backends/arm/test/ops/test_constant_pad_nd.py backends/arm/test/ops/test_conv_constant_pad_nd.py -k "tosa and not u85 and not u55 and not a8w4"; scalar qparam quantization helper audit
Note: focused Python smoke passed 239 tests with 2 skips, 20 expected xfails, and 280 deselected. This is a small setup-positive grammar slice; dtype eligibility remains at the original call sites.

2026-06-13, NN serializer zero-point const wrapper removal slice
Rust LOC: 38,688
Movement: -4 Rust LOC from previous checkpoint
Deleted: NN-local `push_zp_tensor` wrapper after the serializer-wide zero-point const helper absorbed the duplicated construction
Added: direct NN and pool use of `push_zero_point_const`
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_conv3d.py backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 395 tests with 2 expected xfails and 1062 deselected. No model lane was run in this slice.

2026-06-13, mean serializer shift-const helper slice
Rust LOC: 38,682
Movement: -6 Rust LOC from previous checkpoint
Deleted: ad hoc rank-1 Int8 zero shift tensor construction in mean reduction emission
Added: direct use of the existing `push_shift_const` serializer helper
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_mean_dim.py backends/arm/test/ops/test_sum.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 102 tests with 167 deselected. No model lane was run in this slice.

2026-06-13, pad serializer inline-const helper slice
Rust LOC: 38,676
Movement: -6 Rust LOC from previous checkpoint
Deleted: manual pad const tensor plus `Op::Const` operator emission and direct fallback `TosaTensor::with_data` construction
Added: direct use of `make_rank1_const_tensor` for fallback tensor data and `push_inline_const` for const emission
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_constant_pad_nd.py backends/arm/test/ops/test_conv_constant_pad_nd.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 67 tests with 50 deselected. No model lane was run in this slice.

2026-06-13, serializer EmitResult helper-use slice
Rust LOC: 38,672
Movement: -4 Rust LOC from previous checkpoint
Deleted: direct simple operator emission in avg-pool, pad, quantized negate, and matmul visitors where existing `EmitResult` helpers are equivalent
Added: direct use of `push_op` and `push_output_op` at those serializer call sites
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_neg.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_matmul.py backends/arm/test/ops/test_addmm.py backends/arm/test/ops/test_avg_pool2d.py backends/arm/test/ops/test_max_pool.py backends/arm/test/ops/test_constant_pad_nd.py backends/arm/test/ops/test_conv_constant_pad_nd.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 372 tests with 10 expected xfails and 638 deselected. No model lane was run in this slice.

2026-06-13, serializer EmitResult helper-use slice 2
Rust LOC: 38,672
Movement: 0 Rust LOC from previous checkpoint
Deleted: direct simple operator emission in control-flow, RESCALE, and ConstShape helper paths where existing `EmitResult::push_op` is equivalent
Added: direct use of `push_op` at those serializer call sites
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_sub.py backends/arm/test/ops/test_mul.py backends/arm/test/ops/test_relu.py backends/arm/test/ops/test_clamp.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 236 tests with 1 expected xfail and 353 deselected. No model lane was run in this slice. Remaining direct `TosaOperator::new` sites are now concentrated in transpose-conv, broadcast/reshape helpers, indexing, shape utils, and tensor ops.

2026-06-13, serializer shape/broadcast helper-use slice
Rust LOC: 38,604
Movement: -68 Rust LOC from previous checkpoint
Deleted: manual output tensor plus operator emission in transpose-conv, binary-broadcast transpose/reshape, and reshape helper paths; repeated local ConstShape byte construction in broadcast, shape utils, and reshape emission
Added: direct use of `push_output_op` and `push_const_shape` at those serializer call sites
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_transpose_conv2d.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_sub.py backends/arm/test/ops/test_mul.py backends/arm/test/ops/test_view.py backends/arm/test/ops/test_expand.py backends/arm/test/ops/test_permute.py backends/arm/test/ops/test_squeeze.py backends/arm/test/ops/test_unsqueeze.py backends/arm/test/ops/test_cat.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 343 tests with 1 expected xfail and 584 deselected. No model lane was run in this slice. Remaining direct `TosaOperator::new` sites are down to two specialized cast-rewrite helpers in `op_utils` and `indexing`.

2026-06-13, serializer final direct-operator cleanup slice
Rust LOC: 38,594
Movement: -10 Rust LOC from previous checkpoint
Deleted: final direct `result.operators.push(TosaOperator::new(...))` sites in serializer visitor code
Added: direct use of `push_output_op` for the remaining bool-binary and indexing cast normalization helpers
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_add.py backends/arm/test/ops/test_sub.py backends/arm/test/ops/test_mul.py backends/arm/test/ops/test_div.py backends/arm/test/ops/test_floor_div.py backends/arm/test/ops/test_index_tensor.py backends/arm/test/ops/test_index_put.py backends/arm/test/ops/test_index_select.py backends/arm/test/ops/test_index_copy.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed 301 tests with 21 expected xfails and 441 deselected. No model lane was run in this slice. `rg 'result\.operators\.push\(TosaOperator::new\(' src/serialize/visitors -g '*.rs'` now returns no matches.

2026-06-13, control-flow subgraph assembly module slice
Rust LOC: 38,608
Movement: +14 Rust LOC from previous checkpoint
Deleted: control-flow region metadata collection, scalar cond-output marking, while-body passthrough output appending, and subgraph prefix/assembly helpers from `src/lib.rs`
Added: `src/control_flow.rs` as the owner for IR control-flow region metadata and subgraph assembly rules
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55" twice; git diff --check
Note: both control-flow runs passed `29 passed, 1 expected xfail, 60 deselected`. No model lane was run. This is a Refactor 4 ownership slice; LOC is slightly positive because the module boundary adds explicit crate-qualified calls.

2026-06-13, control-flow boundary contract module slice
Rust LOC: 38,604
Movement: -4 Rust LOC from previous checkpoint
Deleted: remaining IR control-flow boundary contract collection, parent-to-submodule qparam propagation, output normalization, quantize/dequantize boundary rewrite, and while-cond rescale stripping helpers from `src/lib.rs`
Added: `src/control_flow.rs` now owns the complete IR control-flow boundary contract engine; `src/lib.rs` keeps only PyO3 entrypoints, stash orchestration, serialization orchestration, and a thin pre-submodule propagation wrapper
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55" twice; git diff --check
Note: both control-flow runs passed `29 passed, 1 expected xfail, 60 deselected`. No model lane was run. Net LOC barely moved because this was an ownership extraction, but `src/lib.rs` dropped roughly 1,000 lines of control-flow domain logic and is now 340 lines.

2026-06-13, neutral QParam owner slice
Rust LOC: 38,593
Movement: -11 Rust LOC from previous checkpoint
Deleted: `QParam` ownership from `src/fx_control_flow_metadata.rs` and direct `fx_control_flow_metadata::QParam` imports in non-control-flow modules
Added: `src/fx_quant.rs` as the neutral owner for quantization parameter parsing/construction, with `src/fx_ir.rs` continuing to re-export `QParam` for IR code
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_tanh.py backends/arm/test/ops/test_sigmoid.py -k "tosa and not u85 and not u55 and not a8w4"; pytest backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55"; git diff --check
Note: focused quantization-heavy smoke passed `239 passed, 611 deselected`; control-flow smoke passed `29 passed, 1 expected xfail, 60 deselected`. No model lane was run. This removes another accidental coupling where generic quantization facts were owned by the control-flow module.

2026-06-13, Python attr helper slice 1
Rust LOC: 38,594
Movement: +1 Rust LOC from previous checkpoint
Deleted: `py_attr` and `meta_val` ownership from `src/fx_graph.rs`; repeated `getattr().and_then(extract).unwrap_or(...)` qparam parsing in `QParam::from_py`
Added: `src/fx_py_attr.rs` as the neutral owner for lossy/defaulting Python attr and metadata-dict reads; `QParam::from_py` now uses `attr_or` and `attr_first_or` for scalar, per-channel, and zero-point fallback fields
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_tanh.py backends/arm/test/ops/test_sigmoid.py backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: combined focused smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This is a small setup-positive slice; the intended payoff is in migrating `NodeMeta::from_fx` and structural control-flow metadata reads next.

2026-06-13, Python attr helper slice 2
Rust LOC: 38,576
Movement: -18 Rust LOC from previous checkpoint
Deleted: repeated optional metadata-dict read/extract/default spelling in `NodeMeta::from_fx` for qparam maps, tensor value metadata, dim-order fields, table qparams, known boolean extras, custom metadata, and special dtype metadata
Added: `meta_opt` helper for typed optional metadata-dict reads; `NodeMeta::from_fx` now uses `meta_val`/`meta_opt` for lossy optional metadata reads while leaving nested custom parsing explicit
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_tanh.py backends/arm/test/ops/test_sigmoid.py backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: combined focused smoke again passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This touches central metadata import, so the next attr migration should be similarly isolated or followed by a broader ops lane.

2026-06-13, Python attr helper slice 3
Rust LOC: 38,586
Movement: +10 Rust LOC from previous checkpoint
Deleted: local duck-typed Python node checks and optional shape/dtype attr extraction in `FxGraphModule::is_node`, `resolve_arg`, `FakeTensorMeta::from_py`, `QParam::from_py`, and two control-flow qparam walkers
Added: `attr_display_opt`, `has_attr`, and `has_attrs` helpers in `src/fx_py_attr.rs` for optional display attrs and duck-typed existence checks
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_tanh.py backends/arm/test/ops/test_sigmoid.py backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: combined focused smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This is setup-positive, but it makes missing/optional Python attr semantics explicit enough to support a future typed wrapper rather than ad-hoc indexing sugar.

2026-06-13, Python attr wrapper slice
Rust LOC: 38,603
Movement: +17 Rust LOC from previous checkpoint
Deleted: the compatibility free-function attr API (`attr_opt`, `attr_or`, `attr_first_opt`, `attr_first_or`, `attr_display_opt`, `has_attr`, `has_attrs`, `meta_val`, `meta_opt`) after migrating current call sites
Added: `AttrView` and `MetaView` in `src/fx_py_attr.rs`, with call sites now using `attrs(obj).optional/first_or/has` and `meta(dict).value/optional`
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_tanh.py backends/arm/test/ops/test_sigmoid.py backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: combined focused smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This is setup-positive in LOC, but it turns the attr boundary from a bag of helper functions into a typed view API with explicit lossy/optional semantics.

2026-06-13, Python required attr wrapper slice
Rust LOC: 38,614
Movement: +11 Rust LOC from previous checkpoint
Deleted: raw required/error-propagating `getattr`/`hasattr` spelling in `FxGraphModule`, `IrGraph` graph/nodes selection, `FakeTensorMeta::to_py_from`, and several control-flow graph helpers
Added: `AttrView::object`, `AttrView::required`, `AttrView::required_str`, and `AttrView::try_has` so required Python-boundary reads are explicit instead of being folded into lossy optional helpers
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_tanh.py backends/arm/test/ops/test_sigmoid.py backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This kept the Rust boundary honest: required attrs still return `PyResult`, while optional/defaulted reads remain visibly lossy.

2026-06-13, Python item-view attr slice
Rust LOC: 38,630
Movement: +16 Rust LOC from previous checkpoint
Deleted: repeated raw Python `items()` loops, tuple-index extraction, keyed `get_item` probes, nested annotation dict reads, FX metadata attr reads, and dynamic GraphModule attr reads from graph-signature ingestion, kwargs import, qparam dict extraction, FakeTensor base recovery, tensor extraction, metadata import/write-back, and structural control-flow helper code
Added: `ItemView` in `src/fx_py_attr.rs`, with keyed lookup, indexed lookup, required indexed extraction, and lossy pair iteration for dict-like Python objects
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest backends/arm/test/ops/test_conv2d.py backends/arm/test/ops/test_mm.py backends/arm/test/ops/test_bmm.py backends/arm/test/ops/test_add.py backends/arm/test/ops/test_tanh.py backends/arm/test/ops/test_sigmoid.py backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py -k "tosa and not u85 and not u55 and not a8w4"; git diff --check
Note: focused Python smoke passed twice at `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This is a grammar slice, not a deletion slice; the payoff is that dict-like, tuple-like, metadata, and dynamic GraphModule Python boundary reads now have the same explicit required-vs-lossy shape as attr reads. Remaining raw `getattr`/`get_item` hits in the touched files are concentrated in `src/fx_py_attr.rs` itself plus true Python import/constructor paths.

2026-06-13, control-flow structural inlet split
Rust LOC: 38,564
Movement: -66 Rust LOC from previous checkpoint
Deleted: Python-facing structural control-flow pre-pass code from `src/fx_control_flow_metadata.rs`, including live-FX submodule traversal, pre-pass qparam snapshotting, structural placeholder stamping, parent rank-4 metadata stamping, temporary submodule IR write-back wrappers, table qparam snapshotting, and fake-tensor refresh
Added: `src/fx_control_flow_structural.rs` as the explicit Python inlet compatibility layer, plus `StructuralInlet` so `Python<'py>`/GraphModule access is created once at the boundary instead of threaded through general-looking phase signatures
Validation: cargo fmt; cargo check --lib; cargo test --lib; maturin develop --release; pytest via `/tmp/et.sh` over conv2d/mm/bmm/add/tanh/sigmoid/cond/while focused smoke; git diff --check
Note: Rust tests passed `130 passed`. Focused Python smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. `src/fx_control_flow_metadata.rs` is now about 325 lines and owns IR-oriented control-flow metadata/pass helpers; the unavoidable live-Python structural stamping is quarantined in the inlet module. This is not a Python round-trip path after Rust IR lowering.

2026-06-13, control-flow structural IR completion
Rust LOC: 37,563
Movement: large deletion slice; source `cargo check --lib` is warning-clean
Deleted: the temporary `src/fx_control_flow_structural.rs` inlet module, the exported `_native.stamp_control_flow_structural` hook, `src/fx_pass.rs`, old `FxGraphModule` wrapper methods, `IrGraph::write_back`, `NodeMeta::write_back`, Python FakeTensor reconstruction, qparam-to-Python conversion, and duplicate `tosa_spatial_rank` write-back extras
Added: IR-side pre-pass qparam snapshots, pre-submodule CF placeholder layout propagation via `val_dim_order`, and a serializer rule that treats Q/DQ/RESCALE as true layout-preserving shape pass-through ops
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `/tmp/et.sh`; `git diff --check`
Note: Rust tests passed `130 passed`. Focused Python smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This removes the separate Python structural mutation seam; control-flow structural facts now enter with the IR import/pre-submodule contract path, and the old write-back stack is deleted rather than quarantined.

2026-06-13, control-flow boundary RESCALE grammar slice
Rust LOC: 37,565
Movement: +2 Rust LOC from previous checkpoint
Deleted: three hand-built control-flow boundary RESCALE argument/metadata blocks for placeholder input rescale, quantize-boundary rewrite, and output rescale normalization
Added: local control-flow boundary RESCALE helpers that build the canonical RESCALE args and source/target qparam metadata while preserving each call site's existing anchoring and rewiring semantics
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `bash /tmp/et.sh`; `git diff --check`
Note: Rust tests passed `130 passed`. Focused Python smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This is a small grammar slice, not a deletion win; it gives the CF boundary code one named RESCALE construction operation before any larger module split.

2026-06-13, control-flow placeholder contract helper slice
Rust LOC: 37,551
Movement: -14 Rust LOC from previous checkpoint
Deleted: duplicate control-flow placeholder ID collection and repeated placeholder metadata/qparam stamping blocks shared by pre-submodule qparam propagation and final boundary contract application
Added: local helpers for boundary placeholder lookup, external qparam recovery, and placeholder metadata stamping with the prior marker-count behavior preserved explicitly
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `bash /tmp/et.sh`; `git diff --check`
Note: Rust tests passed `130 passed`. Focused Python smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This reduces CF contract duplication without moving module boundaries or changing boundary ordering.

2026-06-13, deleted IR write-back bookkeeping state
Rust LOC: 37,425
Movement: -126 Rust LOC from previous checkpoint
Deleted: dead `IrNode` arg-dirty/write-back fields, dead inserted-node metadata-copy source tracking, dead `NodeMeta::DirtyFlags`, no-op `copy_meta_from` calls, and import-only saved table qparam fields
Added: no replacement state; graph mutation now updates only live IR topology/metadata, and reverse-use lists remain the authoritative dependency structure
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `bash /tmp/et.sh`; source sweep for deleted write-back bookkeeping symbols; `git diff --check`
Note: `cargo check --lib` was warning-clean, Rust tests passed `130 passed`, and focused Python smoke passed `268 passed, 1 expected xfail, 671 deselected`. No model lane was run. This is a direct Refactor 9 payoff: after the Python write-back path was deleted, the write-back bookkeeping fields were provably write-only and could be removed.

2026-06-13, broad coverage after write-back bookkeeping deletion
Rust LOC: 37,425
Command: `bash /tmp/et.sh` using `scripts/coverage_arm_tests.sh --jobs auto --model-jobs 4`
Ops INT: 2 failed, 2323 passed, 2 skipped, 59 xfailed
Ops FP: 1911 passed, 13 skipped, 66 xfailed
Models INT: 72 passed, 2 skipped, 6 xfailed
Models FP: 81 passed, 2 skipped, 3 xfailed
Coverage: TOTAL regions 83.33%, functions 86.47%, lines 83.05%
Reports: `target/coverage/arm-tests-20260613-133542/summary.txt`; `target/coverage/arm-tests-20260613-133542/html/index.html`; `target/coverage/arm-tests-20260613-133542/lcov.info`
Note: the only nonzero lane was `ops-int`, with the known `erfinv(±1)` failures. The coverage wrapper now prefers a native 24.x `flatc` discovered in ExecuTorch `pip-out` before Homebrew/latest flatc or the venv shim, avoiding both the Linux-binary exec-format failure and the FlatBuffers 25 generated-API mismatch.

2026-06-13, lowering-effect naming cleanup slice
Rust LOC: 37,438
Movement: +13 Rust LOC from previous checkpoint
Deleted: stale `FxEp*`, `IrEp*`, `*EpPass`, `fx_ep_effects`, and EP-effect naming from active Rust source and architecture docs
Added: `src/lowering_effects.rs`, `LoweringEffect`, `LoweringEffectBatch`, `LoweringInputKind`, `EffectfulIrPass`, `EffectfulIrPassResult`, and `PipelineStep::Effectful` naming over the same effect model
Validation: `cargo fmt`; stale-name grep over `src` and `docs`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh`; `git diff --check`
Note: Rust tests passed `130 passed`, and focused Python smoke passed `268 passed, 1 expected xfail, 671 deselected`. This is Refactor 6 conceptual cleanup, not a LOC-reduction slice; behavior and effect application are unchanged.

2026-06-13, lowering-effect batch field cleanup slice
Rust LOC: 37,438
Movement: 0 Rust LOC from previous checkpoint
Deleted: the `effects.effects` field-access stutter from active source and tests
Added: `LoweringEffectBatch::items` as the neutral vector field while keeping outer pass results named `effects`
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh`; stutter/stale-name grep; `git diff --check`
Note: Rust tests passed `130 passed`, and focused Python smoke passed `268 passed, 1 expected xfail, 671 deselected`. This is a readability cleanup on top of Refactor 6; behavior and effect ordering are unchanged.

2026-06-13, per-channel qparam axis-slice grammar
Rust LOC: 37,453
Movement: +15 Rust LOC from previous checkpoint
Deleted: the only non-generated `#[allow(dead_code)]` in active Rust source and the grouped-conv pass's raw per-channel vector bounds checks
Added: `QParam::supports_axis0_channel_slice` and `QParam::slice_axis0_channels`, making grouped-conv's output-channel slicing assumption explicit and checked
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib fx_decompose_grouped_conv_ir`; `cargo test --lib fx_fold_qdq_ir`; source dead-code suppression audit
Note: targeted Rust tests passed `2 passed` for grouped-conv decomposition and `8 passed` for folded Q/DQ metadata. Only generated TOSA bindings retain `#[allow(dead_code)]`.

2026-06-13, pass-local metadata wrapper cleanup slice
Rust LOC: 37,441
Movement: -12 Rust LOC from previous checkpoint
Deleted: a one-line size-adjust metadata wrapper and redundant masked-fill output-qparam clone/reapply blocks
Added: direct use of `NodeMeta::cloned_with_val`; masked-fill metadata now relies on clone-preserved output qparams while only restamping input qparams
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over conv2d, conv3d, avg_pool2d, max_pool, and masked_fill
Note: Rust tests passed `130 passed`. Focused Python smoke passed `386 passed, 1 expected xfail, 1023 deselected`.

2026-06-13, pooling pair-argument parser cleanup slice
Rust LOC: 37,429
Movement: -12 Rust LOC from previous checkpoint
Deleted: duplicate scalar-or-pair argument parsers in avg-pool pad decomposition and max-pool dilation decomposition
Added: `IrArg::as_int_pair_or_scalar`, keeping `(scalar | [x] | [h, w, ...]) -> (h, w)` parsing on the IR argument type
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over conv2d, conv3d, avg_pool2d, max_pool, and masked_fill
Note: Rust tests passed `130 passed`. Focused Python smoke passed `386 passed, 1 expected xfail, 1023 deselected`.

2026-06-13, normalization helper cleanup slice
Rust LOC: 37,419
Movement: -10 Rust LOC from previous checkpoint
Deleted: local layernorm float-arg helper, local layernorm anchor wrapper, and local var positional/kwarg scalar default helpers
Added: direct use of `IrArg::as_numeric_f64`, `IrArg::as_bool`, `IrArg::as_int`, and `IrGraph::anchor_to`
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; helper-name audit; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over layer_norm and var
Note: Rust tests passed `130 passed`. Focused Python smoke passed `42 passed, 82 deselected`.

2026-06-13, serializer infrastructure split
Rust LOC: 37,388
Movement: -31 Rust LOC from previous checkpoint
Deleted: monolithic `src/serialize/mod.rs` ownership of tensor materialization, output layout normalization, region collection, flatbuffer graph/table emission, and attribute-union table emission
Added: `src/serialize/collection.rs`, `src/serialize/tensor_layout.rs`, `src/serialize/output_layout.rs`, `src/serialize/flatbuffer.rs`, and `src/serialize/attribute_flatbuffer.rs`; `src/serialize/mod.rs` is now the 35-line public orchestration entry point
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused serializer-sensitive Python smoke via `/tmp/et.sh` over conv2d, conv3d, transpose_conv2d, avg_pool2d, max_pool, permute, view, cond, and while
Note: Rust tests passed `130 passed`. Focused Python smoke passed `526 passed, 3 expected xfails, 1306 deselected`. This is mostly an ownership/grammar split, not a behavior change: semantic region collection, physical tensor declarations, external output layout normalization, flatbuffer tables, and attribute unions now have separate modules.

2026-06-13, IR RESCALE rewrite helper slice
Rust LOC: 37,420
Movement: +32 Rust LOC from previous checkpoint
Deleted: manual RESCALE op construction and user-rewire loops in q/dq folding, matmul output rescale, INT32 input/output/where rescale, and conv accumulator/int16-bias rescale paths
Added: `RescaleRewrite`, `IrGraph::add_rescale`, `IrGraph::add_anchored_rescale`, `IrGraph::add_after_rescale`, and `IrGraph::replace_live_uses_except`
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib rescale`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over mm, bmm, addmm, conv2d, conv3d, transpose_conv2d, add, mul, where, cond, and while; Rust LOC check
Note: Rust tests passed `130 passed`. Focused Python smoke passed `444 passed, 3 expected xfails, 1115 deselected`. This is a setup-positive grammar slice that centralizes canonical RESCALE construction across eight insertion sites while intentionally leaving control-flow boundary RESCALEs for a control-flow-specific follow-up.

2026-06-13, IR graph insertion grammar migration slice
Rust LOC: 37,321
Movement: -99 Rust LOC from previous checkpoint
Deleted: the control-flow-owned `NodeOp` definition, remaining control-flow RESCALE arg construction helpers, and production `add_node(NodeOp::CallFunction, ...)` followed by immediate `anchor_to`/`anchor_after` spelling in the audited rewrite/decomposition paths
Added: `fx_ir::NodeOp` ownership, `IrGraph::add_after_call_function`, and broader use of `add_anchored_call_function`, `add_anchored_call_function_with_kwargs`, `add_after_call_function`, and `RescaleRewrite`
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over layer_norm, var, mean_dim, repeat, ne, where, linear, conv1d, conv2d, cumsum, slice, select_scatter, tril, masked_fill, constant_pad_nd, max_pool, adaptive_avg_pool2d, gather, index_select, embedding, index_put, cond, and while; targeted source-pattern audits; Rust LOC check
Note: Rust tests passed `130 passed`. Focused Python smoke passed `820 passed, 6 skipped, 3 expected xfails, 1810 deselected`. This applies the RESCALE trial's semantic-collapse pattern more broadly: core IR node kind ownership now lives in `fx_ir`, CF boundary RESCALEs share the canonical RESCALE builder, and common call-function insertion with anchoring now has one graph API spelling instead of pass-local boilerplate.

2026-06-13, constant materialization grammar slice
Rust LOC: 37,196
Movement: -125 Rust LOC from previous checkpoint
Deleted: duplicate `MaterializeConstantNodeEffect` vs `MaterializeZeroInputConstantNodeEffect` payloads, duplicate ingest materialization match arms, local scalar/f16/bf16 byte encoders in serializer const utilities, local scalar/repeated-byte helpers in zero-input constant folding, and the float-value-to-tensor mini-trait in constant tensor evaluation
Added: `src/constant_materialization.rs`, unified `MaterializeTensorNodeEffect`, shared scalar byte encoding with explicit storage-vs-serializer `Int4` behavior, shared repeated scalar fill helpers, and `MaterializeTensorNodeEffect::from_tensor`
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over full, zeros, ones, arange, eye, view, permute, unsqueeze, mm, bmm, matmul, and addmm; `git diff --check`; Rust LOC check
Note: Rust tests passed `130 passed`. Focused Python smoke passed `307 passed, 7 skipped, 10 expected xfails, 553 deselected`. This is the first constant-materialization semantic collapse: precomputed tensor nodes now have one lowering effect, and scalar/fill byte construction has one Rust owner while preserving zero-input `Int4` storage behavior separately from serializer-supported inline constants.

2026-06-13, legacy constant fallback effect deletion
Rust LOC: 37,111
Movement: -85 Rust LOC from previous checkpoint
Deleted: `MaterializeZeroInputConstantEffect`, `FuseConstantNodeEffect`, the `fx_fuse_constant_args` module, `retarget_to_fused_const`, and the ingest match arms that looked for Python-created `{node}_fused_const` buffers
Added: no new production API; unsupported constant-fold opportunities now only emit `fold_fallback_log` diagnostics instead of deferred no-op lowering effects
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over full, zeros, ones, arange, eye, view, permute, unsqueeze, mm, bmm, matmul, and addmm; `git diff --check`; Rust LOC check
Note: Rust tests passed `130 passed`. Focused Python smoke passed `307 passed, 7 skipped, 10 expected xfails, 553 deselected`. This removes the remaining constant-fold effect vocabulary that only made sense when Python helper code could materialize `{node}_fused_const` buffers during the deleted round-trip.

2026-06-13, dead fused-const Python helper deletion
Python helper LOC: `python/retab/_helpers.py` 1,221 -> 991
Movement: -230 Python LOC
Deleted: unreferenced Python fused-constant materialization helpers (`materialize_constant_node`, `materialize_zero_input_constant_node`, `fuse_constant_node`, `_get_param_tensor`, `_materialize_constant_placeholder`, and `_materialize_tensor_like_node`)
Added: no new production API; `_torch_dtype_from_name` and `_tensor_from_bytes` remain because precomputed TABLE insertion still uses them
Validation: `python3 -m py_compile python/retab/_helpers.py`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over full, zeros, ones, arange, eye, view, permute, unsqueeze, mm, bmm, matmul, and addmm; `git diff --check`; fused-const helper reference audit
Note: Focused Python smoke passed `307 passed, 7 skipped, 10 expected xfails, 553 deselected`. This removes the Python half of the legacy fused-constant fallback path after the Rust effect model stopped referencing it.

2026-06-13, legacy TABLE fallback deletion
Rust LOC: 37,077
Python helper LOC: `python/retab/_helpers.py` 991 -> 788
Movement: -34 Rust LOC and -203 Python LOC
Deleted: no-op `InsertTableOpEffect`, the from-IR `InsertTableOp` logging arm, the Python `insert_table_op` fallback, Python-side TABLE computation helpers (`QuantSpec`, `_TABLE_TORCH_FNS`, `_table_torch_fn`, `_compute_table`), and stale compute-in-Python table comments
Added: no new production API; non-precomputable TABLE candidates now only emit `fold_fallback_log` diagnostics while the active path remains `InsertPrecomputedTableOpEffect`
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `python3 -m py_compile python/retab/_helpers.py`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over sigmoid, hardsigmoid, hardswish, silu, tanh, tan, sin, cos, sinh, cosh, asin, acos, atan, asinh, acosh, atanh, exp, expm1, log, log1p, sqrt, rsqrt, reciprocal, erf, erfinv, pow, elu, and gelu; `git diff --check`; TABLE fallback helper reference audit
Note: Rust tests passed `130 passed`. Focused Python smoke passed `543 passed, 3 expected xfails, 1056 deselected`. This removes the final Python-computed TABLE fallback now that Rust precomputes active TABLE data directly.

2026-06-13, dead placeholder-deletion helper cleanup
Rust LOC: 37,077 -> 37,079
Python helper LOC: `python/retab/_helpers.py` 788 -> 716
Movement: +2 Rust LOC and -72 Python LOC
Deleted: unreferenced Python `delete_constant_placeholder` and `delete_constant_placeholder_by_name` helpers from the old EP-effect/write-back era
Added: no new production API; constant-fold fallback diagnostics now use skip-oriented names instead of deleted effect names
Validation: `python3 -m py_compile python/retab/_helpers.py`; `cargo fmt`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over sigmoid, hardsigmoid, hardswish, silu, tanh, tan, sin, cos, sinh, cosh, asin, acos, atan, asinh, acosh, atanh, exp, expm1, log, log1p, sqrt, rsqrt, reciprocal, erf, erfinv, pow, elu, and gelu; helper/effect-vocabulary reference audit
Note: This continues the Python helper export audit. The only references to the deleted helpers were their own definitions and internal fallback call, so the deletion narrows `_helpers.py` to the still-called FX insertion and metadata utilities.

2026-06-13, Python helper module deletion
Python helper LOC: `python/retab/_helpers.py` 716 -> deleted
Python package LOC: 2,354 -> 1,663
Movement: -691 Python LOC
Deleted: the remaining `_helpers.py` module, including historical FX node insertion, RESCALE/TRANSPOSE restamping, and Python TABLE application helpers that had no active Rust/PyO3 callers
Added: the one live graph-signature ordering helper is now local to `python/retab/preprocess.py`, and stale helper-path docs/comments were removed
Validation: `python3 -m compileall -q python/retab`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; `maturin develop --release`; focused Python smoke via `/tmp/et.sh` over sigmoid, hardsigmoid, hardswish, silu, tanh, tan, sin, cos, sinh, cosh, asin, acos, atan, asinh, acosh, atanh, exp, expm1, log, log1p, sqrt, rsqrt, reciprocal, erf, erfinv, pow, elu, and gelu; helper-symbol reference audit
Note: This completes the helper-export audit: Python remains the ExecuTorch boundary seam, but no standalone `_helpers.py` graph-surgery module is left for future passes to call back into.

2026-06-13, queued cleanup/builder/tensor-facts slice
Rust LOC: 37,077 -> 37,063
Movement: -14 Rust LOC
Deleted: stale `src/graph/builder.rs` extension module whose remaining behavior was only `Graph::all_node_ids` plus a test-only node constructor; the misleading graph-builder file boundary and remaining production `copy_qparams_from` call sites
Added: `IrGraph::replace_with_call_function`, a small rewrite-builder primitive for one-node replacement rewrites; the `Graph` helpers now live with `Graph`; qparam copying now flows through `NodeMeta::with_qparams_from`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over as_strided_copy, clamp, conv1d, embedding, full/full_like, linear, masked_fill, ne, permute, relu, repeat_interleave, squeeze, transpose_copy, and unsqueeze; `git diff --check`; Rust LOC check
Note: Rust tests passed `130 passed`. Focused Python smoke passed `367 passed, 8 skipped, 782 deselected`. This executed the requested 1/2/3 queue: a small dead-surface deletion, a bounded IR rewrite-builder migration, and a conservative tensor-facts cleanup that makes mutable qparam copying private to `NodeMeta`.

2026-06-13, rewrite-builder expansion slice
Rust LOC: 37,063 -> 37,059
Movement: -4 Rust LOC
Deleted: local final-node `add_node` plus `replace_and_erase` sequences in logit, addmm, cumsum, and mean decomposition paths; several temporary-helper `add_node` spellings in logit and addmm now use the anchored insertion grammar directly
Added: no new API; this expands use of `IrGraph::add_anchored_call_function` and `IrGraph::replace_with_call_function`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over logit, addmm, mm, bmm, cumsum, and mean_dim; Rust LOC check
Note: Rust tests passed `130 passed`. Focused Python smoke passed `171 passed, 331 deselected`. This is a small LOC reduction but a useful grammar migration: these decomposition passes now express helper insertion and final replacement through shared `IrGraph` rewrite primitives instead of pass-local boilerplate.

2026-06-13, replacement-finalizer grammar slice
Rust LOC: 37,059 -> 37,068
Movement: +9 Rust LOC
Deleted: direct final-node add/replace pairs in var, strided-slice, index-select, index-tensor, linear, select-scatter, unfold, index-put, and selected decompose-ops approximation paths; unanchored final view creation in index-tensor/index-put/linear
Added: `IrGraph::replace_with_call_function_with_kwargs` for kwargs-bearing final replacements and `ApproxCtx::replace_output` for approximation-pass final outputs
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over var, slice, unfold_copy, index_select, index_tensor, index_put, linear, select_scatter, softmax, logsoftmax, tan, atan, atanh, sinh, cosh, asinh, acosh, elu, gelu, and erfinv excluding known erfinv infinities; Rust LOC check
Note: Rust tests passed `130 passed`. Focused Python smoke passed `537 passed, 20 xfailed, 1115 deselected`. This is a deliberate LOC-positive grammar slice: it fills the kwargs final-replacement gap and gives approximation decompositions a shared final-output spelling, reducing ownership ambiguity around "create final op, then replace original".

2026-06-13, arithmetic target-family grammar slice
Rust LOC: 37,068 -> 37,088
Movement: +20 Rust LOC
Deleted: local int32-rescale comparison/minmax/mul target helpers, repeated raw add/sub/sum target checks in fold-q/dq, and the now-single-use local `target_matches_any` helper
Added: shared `fx_target` helpers for abs, tensor comparisons, min/max, add/sub, mul, and sum.dim_IntList target families, with direct classifier tests
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over abs, add, sub, mul, sum, eq, ge, gt, le, lt, maximum, minimum, and where; Rust LOC check
Note: Rust tests passed `131 passed`. Focused Python smoke passed `393 passed, 562 deselected`. This is setup-positive because it adds central tests, but it moves arithmetic qparam-domain decisions out of local string predicates and into the shared target grammar.

2026-06-13, approximation finalizer cleanup slice
Rust LOC: 37,088 -> 37,082
Movement: -6 Rust LOC
Deleted: remaining conditional `ApproxCtx::output` plus common `replace_and_erase` finalization patterns in softmax/logsoftmax, asin/acos, and gelu decomposition
Added: no new API; this pays down the earlier `ApproxCtx::replace_output` helper by making conditional approximation decompositions use the same direct final-output grammar as the simple paths
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_decompose_ops_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over softmax, logsoftmax, asin, acos, and gelu; Rust LOC check
Note: Focused Rust decompose-op tests passed `21 passed`; full Rust tests passed `131 passed`; focused Python smoke passed `94 passed, 192 deselected`.

2026-06-13, control-flow assembly child-module split
Rust LOC: 37,082 -> 37,091
Movement: +9 Rust LOC
Deleted: serialize-Graph assembly responsibilities from the parent `src/control_flow.rs`, including scalar-output marking, while-body passthrough output appending, subgraph input-store cleanup, and the subgraph assembly loop
Added: `src/control_flow/assembly.rs` as the owner for converting stashed control-flow submodule IR into prefixed serializer graphs, re-exported through `src/control_flow.rs`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused control-flow smoke via `/tmp/et.sh` twice over cond/while with U55/U85 excluded; Rust LOC check
Note: Rust tests passed `131 passed`; both control-flow smokes passed `29 passed, 1 expected xfail, 60 deselected`. No model lane was run. This is LOC-positive because the child module boundary adds imports and a re-export, but it narrows `control_flow.rs` to contract/orchestration logic and gives serialize-Graph assembly a dedicated owner.

2026-06-13, control-flow contract child-module split
Rust LOC: 37,091 -> 37,103
Movement: +12 Rust LOC
Deleted: boundary contract machinery from the `src/control_flow.rs` module root, including boundary input/output contract collection, pre-submodule qparam propagation, parent rank-4 annotation, quantize/dequantize boundary rewrites, while-cond output RESCALE stripping, and output RESCALE normalization
Added: `src/control_flow/contracts.rs` as the owner for IR control-flow boundary contracts; `src/control_flow.rs` now owns only region metadata plus re-exports for the contracts and assembly modules
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused control-flow smoke via `/tmp/et.sh` twice over cond/while with U55/U85 excluded; Rust LOC check
Note: Rust tests passed `131 passed`; both control-flow smokes passed `29 passed, 1 expected xfail, 60 deselected`. No model lane was run. This is another LOC-positive module-boundary slice, but it cuts the control-flow module root from roughly 880 lines to roughly 80 lines and leaves each CF responsibility with a clearer owner.

2026-06-13, control-flow metadata namespace slice
Rust LOC: 37,103 -> 37,098
Movement: -5 Rust LOC
Deleted: stale top-level `src/fx_control_flow_metadata.rs` module name and direct `crate::fx_control_flow_metadata` references
Added: `src/control_flow/metadata.rs` under the control-flow domain module, with crate-visible re-exports for the while submodule pipeline passes and sibling-module use from boundary contracts
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused control-flow smoke via `/tmp/et.sh` twice over cond/while with U55/U85 excluded; Rust LOC check
Note: Rust tests passed `131 passed`; both control-flow smokes passed `29 passed, 1 expected xfail, 60 deselected`. No model lane was run. This finishes the Refactor 5 naming/module split: active control-flow lowering code now sits under `src/control_flow/` instead of a legacy `fx_*` metadata module.

2026-06-13, constant-materialization target grammar slice
Rust LOC: 37,098 -> 37,126
Movement: +28 Rust LOC
Deleted: local zero-input constant dispatch over `full`/`arange`/`linspace`/`eye`, local shape-only tensor dispatch over `reshape`/`view`/`unsqueeze`/`permute`, and one local add/sub user-domain substring check
Added: typed `ZeroInputConstantTarget` and `ShapeOnlyConstantTarget` classifiers in `fx_target`, with zero-input constant materialization and shape-only tensor evaluation using the shared target grammar
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over full, arange, eye, view, permute, and unsqueeze; Rust LOC check
Note: Rust tests passed `131 passed`; focused Python smoke passed `151 passed, 7 skipped, 254 deselected`. No model lane was run. This is setup-positive because the central enum/tests add lines, but it removes another local target-classification dialect from constant materialization.

2026-06-13, serializer debug config decoupling slice
Rust LOC: 37,126 -> 37,122
Movement: -4 Rust LOC
Deleted: `fx_graph::debug()` as a serializer debug-config wrapper and the remaining serializer dependency on the Python graph helper module for debug checks
Added: cached `config::tosa_engine_debug_enabled()` as the direct owner for `TOSA_ENGINE_DEBUG`, with serializer modules reading debug config through `config`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; debug-enabled Python smoke via `/tmp/et.sh` over `test_full_tosa_FP_only` with `TOSA_ENGINE_DEBUG=1`; focused control-flow smoke via `/tmp/et.sh`; Rust LOC check; grep for removed `fx_graph::debug`
Note: Rust tests passed `131 passed`; debug-enabled smoke passed `1 passed, 37 deselected`; control-flow smoke passed `29 passed, 1 expected xfail, 60 deselected`. No model lane was run. This completes the Rust-side Refactor 8 coupling cleanup; Python env parsing remains intentionally separate from Rust lowering config.

2026-06-13, architecture invariant completion slice
Rust LOC: 37,122 -> 37,256
Movement: +134 Rust LOC
Deleted: no production code
Added: Rust invariant tests for deterministic nested while-loop region metadata qualification/sorting and for absence of deleted Python boundary/round-trip symbols in active Rust/Python source
Validation: `cargo fmt --check`; `cargo check --lib`; focused `cargo test --lib control_flow::tests`; `cargo test --lib`; focused control-flow smoke via `/tmp/et.sh`; Rust LOC check
Note: Focused invariant tests passed `2 passed`; full Rust tests passed `133 passed`; control-flow smoke passed `29 passed, 1 expected xfail, 60 deselected`. No model lane was run. This is intentionally LOC-positive test coverage and completes the remaining Refactor 10 invariant targets.

2026-06-13, noop target grammar slice
Rust LOC: 37,256 -> 37,289
Movement: +33 Rust LOC
Deleted: local noop target list, copy-target discriminator, and dtype-check target substring checks from `fx_remove_noop_ir`
Added: typed `NoopTarget` classifier in `fx_target` with direct target tests; `IrRemoveNoopPass` now dispatches through the shared target grammar for replacement source and dtype-check semantics
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib fx_target::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over copy, to_copy, clone, detach_copy, and alias_copy; Rust LOC check
Note: Target Rust tests passed `12 passed`; full Rust tests passed `134 passed`; focused Python smoke passed `68 passed, 1 expected xfail, 80 deselected`. No model lane was run. This is setup-positive because central classifier tests add lines, but it removes a local target-classification dialect from the noop rewrite pass.

2026-06-13, memory-format target grammar slice
Rust LOC: 37,289 -> 37,300
Movement: +11 Rust LOC
Deleted: local constant-only shape-transform classifier, local raw ATen/Edge convolution target check, and local TOSA-transpose target check from `fx_memory_format_ir`
Added: shared `fx_target` helpers for constant-only shape transforms and TOSA transpose targets; `is_raw_aten_convolution_target` now also recognizes the Edge convolution spelling used by memory-format planning
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over view, permute, unsqueeze, slice, and transpose_conv2d; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `178 passed, 7 expected xfails, 326 deselected`. No model lane was run. This is setup-positive but leaves `fx_memory_format_ir` with no direct `target.contains` calls.

2026-06-13, indexing/scatter target grammar slice
Rust LOC: 37,300 -> 37,340
Movement: +40 Rust LOC
Deleted: active raw target checks for gather, unfold_copy, slice_scatter, strided slice/slice_copy, and constant-input slice folding across six passes
Added: shared `fx_target` predicates for slice-copy, slice tensor, slice-scatter, unfold-copy, gather, and test-only cat target checks, with overload tests that keep slice-scatter out of the broader slice-tensor family
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over slice, slice_scatter, gather, index_put, unfold_copy, and constant_pad_nd; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `171 passed, 2 expected xfails, 279 deselected`. No model lane was run. This is setup-positive because the shared grammar tests add lines, but it removes another cluster of indexing/scatter target spellings from pass bodies.

2026-06-13, live call-function traversal query slice
Rust LOC: 37,340 -> 37,209
Movement: -131 Rust LOC
Deleted: repeated `0..graph.nodes.len()` snapshots plus alive/call-function guards from 27 pass loops
Added: `IrGraph::live_call_function_ids()` as the shared graph query for passes that snapshot live call-function nodes before mutating the graph
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; broad focused Python smoke via `/tmp/et.sh` over as_strided_copy, avg_pool2d, clamp, constant_pad_nd, cumsum, full, gather, index_put, index_select, linear, logit, masked_fill, mean_dim, pixel_shuffling, repeat, slice, slice_scatter, squeeze, sum, to_copy, tril, unfold_copy, and where; Rust LOC check
Note: Full Rust tests passed `134 passed`; broad focused Python smoke passed `685 passed, 2 skipped, 3 expected xfails, 1254 deselected`. No model lane was run. This is the first Refactor 3 graph-query payoff: it does not build new rewrites yet, but it gives traversal a shared grammar and removes the dominant boilerplate in many simple passes.

2026-06-13, live call-function traversal query slice 2
Rust LOC: 37,209 -> 37,128
Movement: -81 Rust LOC
Deleted: remaining straightforward local live-call traversal snapshots from 19 additional pass loops, leaving memory-format's mixed placeholder/output traversal for a separate query design
Added: no new API; this pays down the existing `IrGraph::live_call_function_ids()` helper across conv1d, clamp, split, adaptive avg-pool, addmm, grouped conv, index tensor, layernorm, maxpool dilation, not-equal, decompose-ops, quant decomposition, var, quant activation fusion, conv/int32 rescale insertion, scalar replacement, matmul rescale, and size-adjust input passes
Validation: `cargo fmt`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over adaptive_avg_pool2d, addmm, amax, amin, batch_norm, conv1d, conv2d, conv3d, eq, gelu, group_norm, hardtanh, layer_norm, max_pool, ne, pow, relu, softmax, tan, var, and conv_relu_residual_add; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `787 passed, 1 skipped, 6 expected xfails, 1789 deselected`. No model lane was run. Together with slice 1, 46 pass loops now share the same live-call traversal grammar.

2026-06-13, qparam slot-stamping helper slice
Rust LOC: 37,128 -> 37,096
Movement: -32 Rust LOC
Deleted: manual one-entry input/output qparam map construction from mean decomposition, cumsum decomposition, grouped-conv decomposition, conv/int32/matmul rescale fixtures, size-adjust input metadata, and table-op test fixtures
Added: `NodeMeta::set_input_qparam_at(index, qp)` and `NodeMeta::set_output_qparam_at(index, qp)`, with the existing single-slot helpers expressed through them
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over add, addmm, conv1d, conv2d, conv3d, cumsum, gelu, max_pool, mean_dim, pow, sigmoid, tan, where, and conv_relu_residual_add; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `642 passed, 4 expected xfails, 1487 deselected`. No model lane was run. Dynamic and multi-input qparam maps remain explicit; this slice only names indexed single-slot stamping.

2026-06-13, single-op decomposition target grammar slice
Rust LOC: 37,096 -> 37,158
Movement: +62 Rust LOC
Deleted: raw target gates from embedding, logit, index_select, index.Tensor, cumsum, sum, mean, adaptive_avg_pool2d, masked_fill, select_scatter, and full_like pass bodies; one remaining local node-length traversal in embedding decomposition
Added: shared `fx_target` predicates and classifier tests for those single-op decomposition target families
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_target`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over adaptive_avg_pool2d, cumsum, embedding, full, index_select, index_tensor, logit, masked_fill, mean_dim, select_scatter, and sum; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `335 passed, 8 skipped, 18 expected xfails, 537 deselected`. No model lane was run. This is setup-positive; the next target-grammar slice should pay it down by migrating adjacent raw target checks without adding many new predicates.

2026-06-14, mixed live-node graph query slice
Rust LOC: 37,158 -> 37,153
Movement: -5 Rust LOC
Deleted: local mixed-node live traversal snapshots from memory-format rank/dim-order annotation, boundary transpose insertion, output transpose insertion, permute-arg remapping, debug layout logging, output-dim annotation, and int64-buffer cast planning
Added: `IrGraph::live_node_ids()`, `IrGraph::live_placeholder_ids()`, and `IrGraph::live_output_ids()` as graph-query siblings to `live_call_function_ids()`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over clamp, clone, conv2d, permute, to_copy, transpose_conv2d, view, dim_order, and int64; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `355 passed, 3 expected xfails, 809 deselected`. No model lane was run. This pays down the previous graph-query setup in memory-format without changing traversal order; reverse propagation still uses the live-node snapshot in reverse.

2026-06-14, live-node query reuse slice
Rust LOC: 37,153 -> 37,151
Movement: -2 Rust LOC
Deleted: remaining targeted local alive-node iteration from control-flow region metadata collection, control-flow boundary contract collection, duplicate-placeholder fusion, fold-q/dq candidate collection, and table-op precompute candidate collection
Added: no new API; this reuses `IrGraph::live_node_ids()`, `live_placeholder_ids()`, and `live_call_function_ids()`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over add, cond, conv2d, gelu, linear, pow, sigmoid, tan, and while; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `370 passed, 4 expected xfails, 951 deselected`. No model lane was run. This is a small consistency slice, not a major LOC reduction.

2026-06-14, conv target spelling grammar slice
Rust LOC: 37,151 -> 37,170
Movement: +19 Rust LOC
Deleted: raw conv1d.default, conv2d.default, and convolution.default target checks from conv1d unsqueeze, grouped-conv decomposition, and conv-bias annotation
Added: shared `fx_target` helpers for exact conv1d, conv2d, and ATen convolution default spellings, plus classifier tests and reuse inside the target classifier itself
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_target`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over conv1d, conv2d, conv3d, and transpose_conv2d; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `285 passed, 2 expected xfails, 920 deselected`. No model lane was run. This is setup-positive but reduces repeated exact convolution spelling in pass bodies.

2026-06-14, fp-to-int cast target grammar reuse slice
Rust LOC: 37,170 -> 37,171
Movement: +1 Rust LOC
Deleted: pass-local `_to_dim_order_copy` / `aten._to_copy` target spelling from fp-to-int cast decoration
Added: no new classifier API; `IrDecorateFpToIntCastPass` now reuses `NoopTarget::DtypeCheckedPassthrough`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over clamp, to_copy, and int64; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `94 passed, 2 expected xfails, 99 deselected`. No model lane was run. This is a tiny semantic-collapse slice that gives the `_to_copy` target family one owner.

2026-06-14, defaulted tensor dtype grammar slice
Rust LOC: 37,171 -> 37,144
Movement: -27 Rust LOC
Deleted: repeated `val_dtype().unwrap_or("torch.float32")` spelling from decomposition, indexing, pooling, sum, quant, and size-adjust passes; direct `NodeMeta::default().cloned_with_val(vec![], dtype)` scalar metadata construction from tril decomposition
Added: `NodeMeta::val_dtype_or(default)`, `NodeMeta::val_dtype_or_float32()`, and `IrGraph::replace_with_call_function_with_val()`; reduction decompositions now reuse the existing anchored-with-value builder grammar
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over mean_dim, var, tril, cumsum, sum, addmm, linear, split, slice, slice_scatter, index_select, index_tensor, index_put, unfold_copy, as_strided_copy, repeat, pixel_shuffling, avg_pool2d, adaptive_avg_pool2d, and max_pool; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `754 passed, 20 expected xfails, 1420 deselected`. No model lane was run. The lone remaining direct float32 dtype fallback is layernorm's multi-source `output_meta -> input_meta -> float32` rule, which remains explicit because it encodes a different fallback chain.

2026-06-14, empty qparam map metadata grammar slice
Rust LOC: 37,144 -> 37,156
Movement: +12 Rust LOC
Deleted: remaining `cloned_with_val(...); set_empty_qparam_maps()` two-step spellings from conv1d unsqueeze, linear decomposition, and grouped-conv slice metadata
Added: `NodeMeta::cloned_with_empty_qparam_maps(shape, dtype)` and generalized `NodeMeta::{cloned_with_val,cloned_without_qparams}` shape inputs to accept `Into<Vec<i64>>`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over conv1d, conv2d, conv3d, and linear; Rust LOC check
Note: Full Rust tests passed `134 passed`; focused Python smoke passed `332 passed, 1 expected xfail, 1077 deselected`. No model lane was run. This is a setup-positive grammar slice: it preserves the distinction between qparam fields cleared and qparam maps present-but-empty, but only five current production call sites use it.

2026-06-14, serializer target-map grammar slice
Rust LOC: 37,156 -> 36,928
Movement: -228 Rust LOC
Deleted: monolithic `parse_op_target` ladder of repeated `target.contains(...)` returns
Added: ordered control-flow, TOSA, quantized-decomposed, and ATen target rule tables plus target-map unit tests for ordering-sensitive cases
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib ingest::target_map`; `cargo test --lib`; broad focused Python serializer smoke via `/tmp/et.sh`; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `1213 passed, 8 skipped, 1 expected xfail, 2321 deselected`. No model lane was run. The table conversion also fixed an existing TOSA ordering bug where `GREATER_EQUAL` was classified as `TosaEqual` because `equal` was matched first.

2026-06-14, live-node iterator graph-query slice
Rust LOC: 36,928 -> 36,925
Movement: -3 Rust LOC
Deleted: active direct `nodes.iter().filter(|n| n.alive)` and `0..nodes.len()` live-node scans from from-IR conversion, control-flow contracts, duplicate-user fusion, pipeline node-count logging, and `lib.rs` control-flow detection
Added: `IrGraph::live_nodes()`, `IrGraph::live_node_count()`, and `IrGraph::node_opt(id)` as shared graph-query/access helpers
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over cond, while, add, view, permute, conv2d, mm, and bmm; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `321 passed, 1 expected xfail, 769 deselected`. No model lane was run. This is a small LOC slice, but it removes more direct liveness traversal from central conversion/control-flow code and makes live-node traversal a graph API concept.

2026-06-14, zero-input scalar materialization helper slice
Rust LOC: 36,925 -> 36,923
Movement: -2 Rust LOC
Deleted: repeated scalar-value-to-domain-byte loops from arange, linspace, and eye materialization; duplicate full-shape extraction and repeated full-fill/dtype unwraps in full bailout diagnostics
Added: `scalar_sequence_bytes(...)` and `full_shape(...)` helpers inside zero-input constant planning
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_fuse_zero_input_constants_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over full, arange, and eye; Rust LOC check
Note: Focused zero-input Rust tests passed `9 passed`; full Rust tests passed `137 passed`; focused Python smoke passed `39 passed, 7 skipped, 50 deselected`. No model lane was run. This is a small local semantic-collapse slice: the planning code now names scalar sequence materialization rather than spelling byte extension loops per factory op.

2026-06-14, target marker grammar helper slice
Rust LOC: 36,923 -> 36,913
Movement: -10 Rust LOC
Deleted: repeated target-marker OR chains across `fx_target` helper predicates
Added: local `contains_any!` marker macro with comma form for short marker sets and pipe form for longer readable sets
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_target::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over view, permute, squeeze, unsqueeze, conv2d, max_pool, avg_pool2d, split, slice, tril, layer_norm, var, where, eq, ge, and gt; Rust LOC check
Note: Target-family Rust tests passed `12 passed`; full Rust tests passed `137 passed`; focused Python smoke passed `643 passed, 1330 deselected`. No model lane was run. This keeps target matching behavior unchanged while giving the shared target grammar a named "contains any marker" operation.

2026-06-14, serializer conv spatial helper and graph-arg node lookup slice
Rust LOC: 36,913 -> 36,835
Movement: -78 Rust LOC
Deleted: duplicate conv2d/conv3d emitted-pad adjustment algorithms, duplicate raw ATen conv kernel-layout inference, and repeated serializer single-node `Arg::Node` graph lookup adapters
Added: `graph::ir::Arg::as_node()`, `conv_spatial_inputs_and_kernels(...)`, and a shared `adjusted_tosa_pad_for_emitted_conv_spatial(...)` helper behind the existing conv2d/conv3d public wrappers
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over conv2d, conv3d, transpose_conv2d, max_pool, avg_pool2d, mean_dim, sum, add, sub, mul, div, floor_div, index_tensor, index_put, index_select, index_copy, constant_pad_nd, conv_constant_pad_nd, cond, and while; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `894 passed, 24 expected xfails, 1780 deselected`. No model lane was run. This slice keeps serializer behavior unchanged while giving conv spatial shape/pad inference and graph-arg node extraction one shared grammar.

2026-06-14, max-pool serializer pad helper reuse slice
Rust LOC: 36,835 -> 36,791
Movement: -44 Rust LOC
Deleted: duplicate max-pool symmetric-pad plus ceil-mode post-pad helper from NN pool emission, and one raw avg-pool input-node dtype match
Added: wider reuse of `adjusted_max_pool_pad(...)` from serializer shape inference in max-pool emission
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over max_pool, avg_pool2d, and conv2d; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `269 passed, 700 deselected`. No model lane was run. Shape inference and emission now share one max-pool pad-adjustment rule.

2026-06-14, pool output-shape formula helper slice
Rust LOC: 36,791 -> 36,787
Movement: -4 Rust LOC
Deleted: duplicate NHWC pool output-shape arithmetic from emitted tensor planning and serializer shape inference
Added: shared `pool_nhwc_shape(...)` helper used by both actual-shape inference and emitted pool output tensors
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over max_pool, avg_pool2d, and conv2d; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `269 passed, 700 deselected`. No model lane was run. This is a small LOC movement but removes one more independent pool shape formula.

2026-06-14, serializer filled-const emission helper slice
Rust LOC: 36,787 -> 36,808
Movement: +21 Rust LOC
Deleted: repeated `make_filled_const_tensor(...)` plus `push_inline_const(...)` guard blocks from derived math, reductions, binary literal broadcasting, indexing bool constants, and quant/dequant serializer paths
Added: `push_filled_const_if_supported(...)` as the shared serializer helper for optional scalar-filled CONST emission
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over round, div, floor_div, remainder, sqrt, full, sign, expm1, log1p, leaky_relu, mean_dim, sum, add, sub, mul, index_tensor, index_put, index_select, and index_copy; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `501 passed, 2 skipped, 24 expected xfails, 784 deselected`. No model lane was run. This is intentionally setup-positive: direct filled-const construction is now isolated to the helper itself and the `nn/pad` fallback case.

2026-06-14, late target-grammar cleanup slice
Rust LOC: 36,808 -> 36,851
Movement: +43 Rust LOC
Deleted: pass-local target predicate/replacement helpers for not-equal decomposition, scalar-shift tensor replacement, repeat_interleave.self_int decomposition, and max_pool2d dilation decomposition
Added: shared `fx_target` helpers for not-equal replacement target selection, scalar-shift replacement target selection, repeat_interleave.self_int matching, max_pool2d matching, and max_pool2d-with-indices matching
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_target::tests`; `cargo test --lib fx_decompose_ne_ir::tests`; `cargo test --lib fx_replace_scalar_with_tensor_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over ne, lshift, rshift, repeat, and max_pool; Rust LOC check
Note: Target-family Rust tests passed `12 passed`; touched pass tests passed `3 passed`; full Rust tests passed `137 passed`; focused Python smoke passed `155 passed, 271 deselected`. No model lane was run. This is setup-positive because the classifier tests are larger than the pass-local deletions, but four more operator-identity rules now have one owner.

2026-06-14, IR positional-argument grammar slice
Rust LOC: 36,851 -> 36,840
Movement: -11 Rust LOC
Deleted: repeated `args.get(index).and_then(...).unwrap_or(default)` and first-arg node-read spellings across decomposition, pooling, slicing, cast-decoration, q/dq, memory-format, and noop-removal passes
Added: `IrNode` positional argument helpers for optional args, cloned-or-None args, node args, integer args, defaulted integer args, numeric-f64 args, boolean args, defaulted boolean args, int-list args, and int-pair-or-scalar args
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over conv1d, conv2d, depthwise_conv, avg_pool2d, adaptive_avg_pool2d, max_pool, embedding, layer_norm, logit, tril, pixel_shuffling, slice, slice_scatter, index_put, relu, hardtanh, and clamp; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `686 passed, 6 skipped, 2 expected xfails, 1609 deselected`. No model lane was run. This is a Refactor 3 grammar slice: graph rewrites now name common positional-argument reads rather than open-coding heterogeneous `IrArg` access in each pass.

2026-06-14, IR keyword-argument grammar slice
Rust LOC: 36,840 -> 36,828
Movement: -12 Rust LOC
Deleted: repeated positional-or-keyword `IrArg` reads for addmm alpha/beta, constant-fold alpha/beta, add/sub alpha, ELU alpha, batch-norm training/eps, max/stack dimensions, group-norm kwargs, slice-scatter bounds, pad mode, convolution transposed flag, and table-op string/scalar kwargs
Added: `IrNode` keyword and arg-or-keyword helpers for cloned-or-None, integer, numeric-f64, scalar-literal-f64, boolean, string, and raw `IrArg` reads
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over conv2d, transpose_conv2d, addmm, batch_norm, gelu, elu, pow, stack, logsoftmax, softmax, rsqrt, floor, ceil, round, erf, constant_pad_nd, conv_constant_pad_nd, and slice_scatter; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `518 passed, 1 skipped, 4 expected xfails, 1160 deselected`. No model lane was run. This finishes the initial IR argument grammar pair: positional and keyword/defaulting behavior are now named on `IrNode` instead of open-coded in passes.

2026-06-14, ingest finalization naming cleanup slice
Rust LOC: 36,828 -> 36,812
Movement: -16 Rust LOC
Deleted: stale `ingest::fx_bridge` module name, redundant finalizer guard booleans, and repeated serialize-graph `Arg` match blocks inside post-conversion shape/dtype fixups
Added: `ingest::finalize` as the owner of post-conversion serialize-graph finalization, plus direct reuse of `Arg::as_node` and `Arg::as_int_list`
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over cond, while, conv2d, transpose_conv2d, addmm, batch_norm, gelu, elu, pow, stack, logsoftmax, softmax, rsqrt, floor, ceil, round, erf, constant_pad_nd, conv_constant_pad_nd, and slice_scatter; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `547 passed, 1 skipped, 5 expected xfails, 1220 deselected`. No model lane was run. This is a Refactor 9 cleanup: the active ingest path no longer contains a bridge-named module now that Python re-ingest is gone.

2026-06-14, qparam input-map grammar slice
Rust LOC: 36,812 -> 36,817
Movement: +5 Rust LOC
Deleted: mutable input-qparam map construction from masked_fill where metadata, mean decomposition multiply metadata, int32 rescale input metadata, while-body output qparam stamping, carried-output rescale metadata, and conv-rescale plain tensor metadata
Added: `NodeMeta::set_input_qparams_from(...)` and `NodeMeta::set_empty_input_qparams()` for named construction of present input-qparam maps
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over cond, while, masked_fill, mean_dim, sum, add, sub, mul, where, pow, cumsum, conv2d, conv3d, and transpose_conv2d; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `580 passed, 6 expected xfails, 1321 deselected`. No model lane was run. This is setup-positive in LOC but moves input-qparam map construction into `NodeMeta`, preserving the distinction between absent qparams and present-but-empty maps.

2026-06-14, index linearization rewrite-builder slice
Rust LOC: 36,817 -> 36,787
Movement: -30 Rust LOC
Deleted: duplicate broadcast-align, expand, flatten, stride-scale, and sum rewrite chains from index.Tensor decomposition and index_put rewrite
Added: `src/fx_index_linear.rs` as the shared owner for contiguous index strides, packed linear scales, and generated flat-index IR construction
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over index_tensor, index_put, index_select, index_copy, gather, and unfold_copy; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `155 passed, 20 expected xfails, 252 deselected`. No model lane was run. This is a Refactor 3 payoff slice: two indexing passes now share the same linear-index rewrite grammar instead of carrying independent copies of the algorithm.

2026-06-14, contiguous stride helper reuse slice
Rust LOC: 36,787 -> 36,777
Movement: -10 Rust LOC
Deleted: duplicate contiguous-stride calculation from as_strided_copy decomposition
Added: no new API beyond reuse of `fx_index_linear::contiguous_strides`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over index_tensor, index_put, index_select, index_copy, gather, unfold_copy, and as_strided_copy; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `161 passed, 20 expected xfails, 260 deselected`. No model lane was run. This keeps contiguous stride semantics in one helper while preserving as_strided_copy's relaxed last-dimension stride check.

2026-06-14, typed target-kind cleanup slice
Rust LOC: 36,777 -> 36,829
Movement: +52 Rust LOC
Deleted: pass-local layer_norm.default-vs-native view-target selection and size-adjust conv-vs-pool substring classification
Added: `LayerNormTargetKind`, `SizeAdjustTargetKind`, `layer_norm_param_view_target(...)`, and typed target tests for layernorm and size-adjust semantics
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_target::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over layer_norm, conv2d, conv3d, depthwise_conv, max_pool, and avg_pool2d; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `432 passed, 1 expected xfail, 1202 deselected`. No model lane was run. This is setup-positive in LOC because the target grammar gained explicit typed tests, but it removes two remaining local target-string dialects from production passes.

2026-06-14, qparam copy and boundary target cleanup slice
Rust LOC: 36,829 -> 36,838
Movement: +9 Rust LOC
Deleted: local grouped-conv boundary-rescale target predicate; direct whole-map qparam clone-and-set blocks in split-to-slice conversion, sum conversion, grouped-conv decomposition, and control-flow boundary placeholder stamping
Added: `NodeMeta::set_output_qparams_from`, `NodeMeta::copy_input_qparams_from`, `NodeMeta::copy_output_qparams_from`, and shared `is_boundary_rescale_target(...)`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_fold_qdq_ir::tests`; `cargo test --lib fx_target::tests`; `cargo test --lib fx_decompose_grouped_conv_ir::tests`; `cargo test --lib control_flow::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over cond, while, conv2d, conv3d, sum, split, add, and mul; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `399 passed, 2 expected xfails, 975 deselected`. No model lane was run. This is a small setup-positive Tensor Facts/target grammar slice; it names qparam fact copying and removes several field-level clone/set idioms from production passes.

2026-06-14, table-op arg wrapper cleanup slice
Rust LOC: 36,838 -> 36,830
Movement: -8 Rust LOC
Deleted: pass-local `node_str_kwarg` and `node_scalar_arg` wrappers in table-op insertion
Added: no new helper surface; table-op parsing now calls the existing `IrNode` arg/kwarg grammar directly
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_insert_table_ops_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over pow, remainder, elu, gelu, sigmoid, exp, log, and sqrt; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `186 passed, 6 expected xfails, 346 deselected`. No model lane was run. This is a deletion-only cleanup in the table-op precompute path.

2026-06-14, shape-only constant planner cleanup slice
Rust LOC: 36,830 -> 36,820
Movement: -10 Rust LOC
Deleted: duplicate pass-local shape-only constant planner wrappers in constant-input fusion and shape-only constant fusion
Added: `fx_constant_tensor_eval::plan_shape_only_tensor_from_constants(...)`, naming the shared composition of first-constant-input lookup plus shape-only tensor planning
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_fuse_shape_only_constants_ir::tests`; `cargo test --lib fx_fuse_constant_inputs_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over full, arange, eye, view, permute, unsqueeze, mm, bmm, matmul, and addmm; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `283 passed, 7 skipped, 10 expected xfails, 529 deselected`. No model lane was run. This keeps shape-only constant-chain planning in the shared constant evaluator instead of duplicating wrapper logic in effectful passes.

2026-06-14, constant slice planner cleanup slice
Rust LOC: 36,820 -> 36,820
Movement: 0 Rust LOC
Deleted: pass-local constant-slice planner wrapper in constant-input fusion
Added: `fx_constant_tensor_eval::plan_constant_slice_tensor_from_constants(...)`, naming the shared composition of first-constant-input lookup plus slice tensor planning
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_fuse_constant_inputs_ir::tests`; `cargo test --lib fx_constant_tensor_eval::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over full, arange, eye, view, permute, unsqueeze, mm, bmm, matmul, and addmm; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `283 passed, 7 skipped, 10 expected xfails, 529 deselected`. No model lane was run. This keeps constant-slice planning in the shared constant evaluator instead of duplicating wrapper logic in effectful passes.

2026-06-14, qparam metadata predicate slice
Rust LOC: 36,820 -> 36,820
Movement: 0 Rust LOC
Deleted: table-op local input/output qparam presence helper and direct qparam-field presence check in inf-literal replacement
Added: `NodeMeta::has_qparam_metadata()` and `NodeMeta::has_nonempty_input_and_output_qparams()`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_insert_table_ops_ir::tests`; `cargo test --lib fx_replace_inf_values_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over pow, remainder, elu, gelu, sigmoid, exp, log, sqrt, rsqrt, and masked_fill; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `216 passed, 6 expected xfails, 403 deselected`. No model lane was run. This names the important Tensor Facts distinction between qparam metadata being present at all and both input/output qparam maps being non-empty.

2026-06-14, nonempty qparam metadata predicate slice
Rust LOC: 36,820 -> 36,818
Movement: -2 Rust LOC
Deleted: local quantized-metadata helper in derived op decomposition
Added: `NodeMeta::has_nonempty_qparam_metadata()`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_decompose_ops_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over pow, remainder, elu, gelu, sigmoid, exp, log, sqrt, rsqrt, and masked_fill; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `216 passed, 6 expected xfails, 403 deselected`. No model lane was run. This keeps quantized-envelope detection in the metadata grammar instead of a decomposition-pass-local map scan.

2026-06-14, serializer input-name grammar slice
Rust LOC: 36,818 -> 36,821
Movement: +3 Rust LOC
Deleted: repeated serializer `input_names.first().cloned()` and `collect_input_names(...).first()` spellings from math, arithmetic, indexing, matmul, pad, rescale, and tensor-op visitors
Added: `graph_utils::input_name(...)` and `EmitContext::input_name(...)` as the shared grammar for resolving one serializer input name
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib serialize::visitors`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over add, clamp, index_tensor, index_put, slice, mm, bmm, matmul, constant_pad_nd, conv_constant_pad_nd, expand, repeat, upsample_nearest2d, and upsample_bilinear2d; Rust LOC check
Note: Serializer Rust tests passed `10 passed`; full Rust tests passed `137 passed`; focused Python smoke passed `656 passed, 30 expected xfails, 831 deselected`. No model lane was run. This is a small setup-positive serializer grammar slice that gives single-input emitters the same input-name vocabulary as multi-input emitters.

2026-06-14, grouped-conv qparam lookup cleanup slice
Rust LOC: 36,821 -> 36,817
Movement: -4 Rust LOC
Deleted: pass-local `single_output_qparam(...)` wrapper in grouped-conv decomposition
Added: no new helper surface; grouped-conv redundant-boundary cleanup now uses `NodeMeta::output_qparam()` directly
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_decompose_grouped_conv_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over conv_constant_pad_nd, conv1d, conv2d, conv3d, depthwise_conv, linear, and transpose_conv2d; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `419 passed, 2 expected xfails, 1341 deselected`. No model lane was run. This is a deletion-only Tensor Facts cleanup: a one-line local alias was weaker than the shared metadata vocabulary and did not deserve promotion.

2026-06-14, exact-slot input qparam cleanup slice
Rust LOC: 36,817 -> 36,811
Movement: -6 Rust LOC
Deleted: direct `input_qparams.as_ref().get(...)` slot reads in conv-bias annotation and int32 where-output rescale insertion
Added: no new helper surface; these passes now use the existing `NodeMeta::input_qparam_at(...)` grammar
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_annotate_conv_bias_ir::tests`; `cargo test --lib fx_insert_int32_rescale_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over abs, add, conv2d, conv3d, linear, mul, sum, and where; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `445 passed, 1 expected xfail, 1146 deselected`. No model lane was run. This is a deletion-only Tensor Facts cleanup that keeps exact-slot qparam reads out of production pass field access.

2026-06-14, tensor-data attr-boundary cleanup slice
Rust LOC: 36,811 -> 36,811
Movement: 0 Rust LOC
Deleted: raw tensor `shape`/`dtype` attr reads from tensor-data extraction
Added: no new helper surface; tensor-data extraction now uses `AttrView::object` and `AttrView::display_optional`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib tensor_data::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over addmm, arange, bmm, constant_pad_nd, conv_constant_pad_nd, embedding, eye, full, matmul, mm, ones, and zeros; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `269 passed, 13 skipped, 10 expected xfails, 411 deselected`. No model lane was run. This keeps dynamic Python attr reads isolated in `fx_py_attr.rs`; the remaining raw `getattr` outside that module is the intentional `torch.getattr("uint16")` API lookup for BF16 byte extraction.

2026-06-14, dtype owner grammar slice
Rust LOC: 36,811 -> 36,825
Movement: +14 Rust LOC
Deleted: `tensor_data::parse_torch_dtype` as the semantic owner for dtype-string parsing; ad hoc qparam dtype-string parsing in constant tensor evaluation, zero-input constant materialization, table planning, and from-IR ingestion
Added: `DType::from_torch_str`, `DType::is_integer`, `QParam::dtype`, and `QParam::is_integer_dtype`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib tensor_data::tests`; `cargo test --lib fx_constant_tensor_eval::tests`; `cargo test --lib fx_fuse_zero_input_constants_ir::tests`; `cargo test --lib fx_insert_table_ops_ir::tests`; `cargo test --lib ingest::from_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over addmm, arange, bmm, constant_pad_nd, conv_constant_pad_nd, embedding, eye, full, matmul, mm, ones, and zeros; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `269 passed, 13 skipped, 10 expected xfails, 411 deselected`. No model lane was run. This is setup-positive in LOC, but it moves dtype-string semantics to the graph dtype type and gives qparam consumers a typed accessor instead of reaching into `dtype_str`.

2026-06-14, conv dtype predicate cleanup slice
Rust LOC: 36,825 -> 36,827
Movement: +2 Rust LOC
Deleted: raw `int16`/`int32` qparam and tensor dtype substring checks from conv-bias annotation and conv-rescale insertion
Added: no new helper surface; those paths now use `DType::from_torch_str` and `QParam::dtype`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_annotate_conv_bias_ir::tests`; `cargo test --lib fx_insert_conv_rescale_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over constant_pad_nd, conv_constant_pad_nd, conv1d, conv2d, conv3d, depthwise_conv, linear, and transpose_conv2d; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `468 passed, 2 expected xfails, 1373 deselected`. No model lane was run. This is the first follow-through slice after moving dtype parsing to `DType`, removing stringly dtype predicates from fragile quantized conv paths.

2026-06-14, int32-rescale dtype predicate cleanup slice
Rust LOC: 36,827 -> 36,826
Movement: -1 Rust LOC
Deleted: pass-local `is_integer_quantized_dtype(...)`, cloned dtype-string propagation through int32 rescale scale computation, and raw `int16`/`int32` substring checks in int32 rescale insertion
Added: `DType::is_quantized_integer` and `QParam::is_quantized_integer_dtype`
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_insert_int32_rescale_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over abs, add, conv2d, conv3d, linear, mul, sum, and where; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `445 passed, 1 expected xfail, 1146 deselected`. No model lane was run. This pays down the dtype owner setup by removing the int32-rescale pass's local quantized-integer dtype dialect.

2026-06-14, fold-qdq dtype predicate cleanup slice
Rust LOC: 36,826 -> 36,826
Movement: 0 Rust LOC
Deleted: raw `int32` qparam dtype substring checks in Fold-QDQ's FP+INT32 preservation rule
Added: no new helper surface; Fold-QDQ now uses `QParam::dtype() == DType::Int32`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_fold_qdq_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over abs, add, conv2d, conv3d, linear, mul, sum, and where; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `445 passed, 1 expected xfail, 1146 deselected`. No model lane was run. This is another follow-through slice after moving dtype parsing to `DType`, keeping qparam dtype tests typed in a numerically sensitive fold pass.

2026-06-14, metadata dtype-kind accessor slice
Rust LOC: 36,826 -> 36,831
Movement: +5 Rust LOC
Deleted: repeated `val_dtype().map(DType::from_torch_str)` spellings from zero-input constant fusion, conv-rescale insertion, and int32-rescale insertion
Added: `NodeMeta::val_dtype_kind()`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib fx_fuse_zero_input_constants_ir::tests`; `cargo test --lib fx_insert_conv_rescale_ir::tests`; `cargo test --lib fx_insert_int32_rescale_ir::tests`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over abs, add, arange, conv2d, conv3d, eye, full, linear, mul, ones, sum, where, and zeros; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `508 passed, 7 skipped, 1 expected xfail, 1220 deselected`. No model lane was run. This is setup-positive, but it moves typed tensor-metadata dtype parsing behind `NodeMeta` instead of making passes parse dtype strings themselves.

2026-06-14, bool dtype predicate cleanup slice
Rust LOC: 36,831 -> 36,837
Movement: +6 Rust LOC
Deleted: raw bool dtype substring checks from index-tensor decomposition, strided-slice decomposition, tril decomposition, and control-flow while-cond output rescale stripping
Added: `NodeMeta::val_dtype_is(...)`
Validation: `cargo fmt --check`; `cargo check --lib`; focused Rust filters for index tensor, strided slice, tril, and control flow; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over index tensor, slice, tril, cond, and while; Rust LOC check
Note: Full Rust tests passed `137 passed`; focused Python smoke passed `121 passed, 19 expected xfails, 163 deselected`. No model lane was run. This is setup-positive by LOC, but it collapses the bool dtype grammar behind the metadata accessor and leaves only two intentional raw dtype-range parsing sites for a later dtype-domain helper.

2026-06-14, dtype-domain range helper slice
Rust LOC: 36,837 -> 36,878
Movement: +41 Rust LOC
Deleted: pass-local dtype substring parsing for maxpool pad sentinels and control-flow reverse-engineered RESCALE qparam bounds
Added: `DType::torch_str_is_float_like(...)`, `DType::quantized_integer_bounds(...)`, `DType::quantized_integer_bounds_from_torch_str(...)`, and focused dtype-domain tests
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib dtype_helpers_preserve_torch_string_domains`; `cargo test --lib fx_decompose_maxpool_dilation_ir`; `cargo test --lib control_flow`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over maxpool, cond, and while; Rust LOC check
Note: Full Rust tests passed `138 passed`; focused Python smoke passed `111 passed, 1 expected xfail, 192 deselected`. No model lane was run. This is setup-positive, but the raw dtype `contains(...)` grammar is now owned by `DType` instead of individual passes in the scanned IR/control-flow area.

2026-06-14, serializer i32 sequence-default helper slice
Rust LOC: 36,878 -> 36,894
Movement: +16 Rust LOC
Deleted: repeated `first/get/unwrap_or` pair and quad default spellings from serializer layout and shape helpers
Added: `i32_pair_or_default(...)`, `i32_pair_or_first(...)`, `i32_quad_or_default(...)`, and focused helper tests
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib i32_sequence_defaults_preserve_pair_semantics`; `cargo test --lib serialize::visitors::shape_utils`; `cargo test --lib serialize::tensor_layout`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over avg-pool2d, max-pool, and conv2d; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `269 passed, 700 deselected`. No model lane was run. This is test-positive by LOC but reduces serializer shape/layout indexing noise and preserves the important distinction between "second value defaults to literal" and "second value defaults to first".

2026-06-14, serializer input/list helper reuse slice
Rust LOC: 36,894 -> 36,888
Movement: -6 Rust LOC
Deleted: NN-local `i32_list_or_repeat(...)` helper and two pool-local first-input graph lookup chains
Added: no new production helper surface; `arg_utils` now owns the repeat-list default helper and the existing `first_input_node(...)` grammar is reused in pool parsing
Validation: `cargo fmt`; `cargo fmt --check`; `cargo check --lib`; `cargo test --lib i32_sequence_defaults_preserve_pair_semantics`; `cargo test --lib serialize::visitors::shape_utils`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over avg-pool2d, max-pool, and conv2d; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `269 passed, 700 deselected`. No model lane was run. This is a small cleanup, but it keeps serializer argument defaults in one helper module and removes duplicated first-input traversal from pool emission.

2026-06-14, derived-math emit-builder reuse slice
Rust LOC: 36,888 -> 36,906
Movement: +18 Rust LOC
Deleted: repeated `temp_tensor(...)` plus ad hoc input-vector construction from derived math temp emissions
Added: no new production helper surface; `emit_aten_div`, `sqrt`, `sign`, `expm1`, `log1p`, `remainder`, and `leaky_relu` now reuse `EmitResult::unary/binary/ternary` for temp one-output emissions
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over div, div tensor mode, sqrt, sign, expm1, log1p, floor-div, remainder, and leaky-relu; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `108 passed, 4 expected xfails, 218 deselected`. No model lane was run. This is LOC-positive due to Rust formatting, but it reduces serializer emission dialect count by using the existing arity-aware builder for derived math temp nodes.

2026-06-14, serializer combined i32 list-default helper slice
Rust LOC: 36,906 -> 36,923
Movement: +17 Rust LOC
Deleted: repeated `kwarg_or_arg_i32_list(...)` plus `i32_list_or(...)`/`i32_list_or_repeat(...)` composition from conv and pool serializer paths
Added: `kwarg_or_arg_i32_list_or(...)` and `kwarg_or_arg_i32_list_or_repeat(...)`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib i32_sequence_defaults_preserve_pair_semantics`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over avg-pool2d, max-pool, conv2d, and conv3d; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `370 passed, 1 expected xfail, 991 deselected`. No model lane was run. This is setup-positive, but it gives the serializer one spelling for kwargs-first i32-list defaulting instead of open-coded two-step composition.

2026-06-14, Python-boundary typed item-pair helper slice
Rust LOC: 36,923 -> 36,932
Movement: +9 Rust LOC
Deleted: manual key/value extraction loops from graph-signature tensor binding import, FX kwargs import, and qparam metadata dict import
Added: `ItemView::typed_key_pairs(...)` and `ItemView::typed_pairs(...)`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over add, conv2d, linear, cond, and while; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `263 passed, 1 expected xfail, 757 deselected`. No model lane was run. This is a small setup-positive attr-boundary slice, but it centralizes lossy typed item extraction instead of repeating `pairs()` plus `extract()` filters at ingestion sites.

2026-06-14, qparam fixture construction cleanup slice
Rust LOC: 36,932 -> 36,923
Movement: -9 Rust LOC
Deleted: fixture-local `HashMap` imports and mutable input-qparam map construction in conv-bias, grouped-conv, constant-input fusion, conv-rescale, int32-rescale, and matmul-rescale tests
Added: no new helper surface; tests now reuse `NodeMeta::set_input_qparams_from(...)`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over add, mm, bmm, matmul, conv2d, conv3d, and sum; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `388 passed, 11 expected xfails, 995 deselected`. No model lane was run. This is a small Tensor Facts follow-through slice: it removes incidental qparam-map construction ceremony where the operation is just keyed metadata stamping, while leaving production map mutation explicit where the algorithm actually edits qparam slots.

2026-06-14, live call-function rewrite traversal slice
Rust LOC: 36,923 -> 36,871
Movement: -52 Rust LOC
Deleted: repeated snapshot-loop/modified-flag scaffolding from 21 IR passes, including clamp, noop removal, qparam rescale insertion, conv1d unsqueeze, cumsum, layernorm, logit, var, repeat_interleave, as_strided_copy, index_put normalization, pad rewrite, embedding, not-equal, and quantized activation fusion
Added: `IrGraph::rewrite_live_call_functions(...)`, preserving the existing snapshot-before-mutate traversal semantics
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over add, mm, bmm, matmul, conv1d, conv2d, clamp, relu, hardtanh, cumsum, layer_norm, logit, var, repeat, squeeze, index_put, and constant_pad_nd; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `604 passed, 12 expected xfails, 1294 deselected`. No model lane was run. This is a Refactor 3 graph traversal grammar slice: passes now name the common "snapshot live call-functions and OR rewrite results" operation instead of carrying local loop scaffolding.

2026-06-14, compact live call rewrite follow-through slice
Rust LOC: 36,871 -> 36,867
Movement: -4 Rust LOC
Deleted: remaining local snapshot-loop scaffolding from scalar-shift tensorization, where-mask expansion, masked_fill decomposition, tril decomposition, and pixel shuffle/unshuffle decomposition
Added: no new helper surface; these passes now reuse `IrGraph::rewrite_live_call_functions(...)`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over where, masked_fill, tril, lshift, and pixel_shuffling; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `71 passed, 127 deselected`. No model lane was run. This is intentionally a small follow-through tranche: it converts compact loop bodies without forcing the larger decomposition passes into a noisy closure rewrite.

2026-06-14, dispatcher live call rewrite follow-through slice
Rust LOC: 36,867 -> 36,862
Movement: -5 Rust LOC
Deleted: local snapshot-loop scaffolding from grouped-conv decomposition and the large decompose-ops dispatch table
Added: no new helper surface; both dispatchers now reuse `IrGraph::rewrite_live_call_functions(...)`
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over conv2d, depthwise_conv, conv_combos, pow, tan, sinh, cosh, asinh, acosh, elu, gelu, batch_norm, stack, unbind, add, and sub; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `552 passed, 1 skipped, 3 expected xfails, 1335 deselected`. No model lane was run. This keeps traversal policy centralized without changing grouped-conv or decompose-op rewrite algorithms.

2026-06-14, decompose primary-result user helper slice
Rust LOC: 36,862 -> 36,846
Movement: -16 Rust LOC
Deleted: duplicated batch_norm/group_norm logic for native tuple-output rewrites through `getitem[0]`
Added: `replace_primary_result_users(...)`, naming the invariant that a decomposition may replace the source op directly or replace only primary-result getitem users
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over batch_norm and group_norm; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `43 passed, 1 skipped, 98 deselected`. No model lane was run. This is a small Refactor 3 grammar slice inside the largest remaining file: the tuple-output user contract is now centralized instead of repeated in each decomposition.

2026-06-14, serializer scalar-arg helper slice
Rust LOC: 36,846 -> 36,850
Movement: +4 Rust LOC
Deleted: repeated direct scalar-f64 and zero-point i64 argument reads from tensor-op serializer visitors
Added: `arg_f64(...)` alongside the existing serializer `arg_i64(...)`, so quantize/dequantize/factory emitters use the same argument grammar
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib serialize::visitors::arg_utils`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over full, zeros, ones, scalar_tensor, add, mul, and to_copy; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `167 passed, 2 skipped, 1 expected xfail, 175 deselected`. No model lane was run. This is a small Refactor 7/3 readability slice, not a LOC-reduction slice.

2026-06-14, redundant qparam clone cleanup slice
Rust LOC: 36,850 -> 36,820
Movement: -30 Rust LOC
Deleted: same-source `.cloned_with_val(...).with_qparams_from(...)` chains and the now-dead `NodeMeta::with_qparams_from` / aggregate `copy_qparams_from` helpers
Added: no replacement helper; `cloned_with_val(...)` already preserves qparam metadata from the cloned base
Validation: `cargo fmt --check`; `cargo check --lib`; `cargo test --lib`; focused Python smoke via `/tmp/et.sh` over index_tensor, index_put, unfold_copy, conv1d, and linear; Rust LOC check
Note: Full Rust tests passed `139 passed`; focused Python smoke passed `221 passed, 20 expected xfails, 503 deselected`. No model lane was run. This is a Refactor 2 metadata grammar cleanup: pass code no longer re-states a qparam copy that already follows from the `NodeMeta` cloning contract.
```

## Refactor 1: Add A Rewrite-IR Target Grammar

Status:

Initial slice started. `src/fx_target.rs` owns the first rewrite-IR target
classifier, and `IrNode` exposes classifier-backed predicates for Q/DQ,
TOSA RESCALE, and ATen mm/bmm accumulator cases. The Q/DQ rescale pass and
mm/bmm rescale pass have been migrated to those predicates. Fold-q/dq now uses
the same grammar for per-tensor and per-channel Q/DQ classification. Clamp and
quantized activation fusion now share classifier-backed activation and
conv/linear producer predicates. Grouped-conv and quant decomposition also use
the shared convolution and per-tensor Q/DQ predicates. Linear decomposition now
uses the shared linear predicate. IR-native control-flow boundary and region
metadata collection now use shared cond/while predicates. Conv bias
annotation, conv accumulator rescale insertion, typed size-adjust conv/pool
classification, and transpose-conv2d weight detection now share named
target-family helpers.
The large decompose-op pass now uses a local `TargetMatcher` grammar in its
rule table instead of many one-line predicate wrappers. Smaller decomposition
and rewrite passes now also use shared `fx_target` helpers for split, tril,
layernorm, var, avg_pool2d padding, as_strided_copy, pad, pixel shuffle,
squeeze/unsqueeze, dim-transpose, index_put, where, and clamp target families.
Layernorm parameter view target selection now also lives in `fx_target`, so
plain and native layernorm decomposition no longer spell local target-string
rules.
Addmm decomposition now uses the classifier-backed node predicate instead of a
raw substring check. Noop removal and fp-to-int cast decoration now use the
typed `NoopTarget` classifier.
Constant materialization and memory-format planning now use shared target
grammar for constant-only shape transforms, TOSA transpose targets, and Edge
convolution spellings. Indexing/scatter passes now also share target grammar
for slice-copy, broad slice tensor, slice-scatter, unfold-copy, and gather
families. Single-op decomposition gates for embedding, logit, index_select,
index.Tensor, cumsum, sum, mean, adaptive_avg_pool2d, masked_fill,
select_scatter, and full_like now also live in the shared target grammar. Exact
conv1d, conv2d, and ATen convolution default spellings now also have shared
target helpers. Grouped-conv redundant-boundary cleanup now also uses shared
boundary-rescale target grammar instead of a pass-local substring predicate.

Remaining work:

Migrate target-string helpers in small batches by domain when the helper will
replace multiple call sites or clarify a risky substring match. Avoid further
setup-only classifier growth until a deletion-heavy slice has paid down the
latest helper expansion.

Problem:

Rewrite passes mostly identify operators by raw string predicates such as
`target.contains(...)`. The serializer side already has a typed operation
grammar through `OpTarget`, but that grammar appears after IR conversion, too
late for most passes.

Proposal:

Introduce an `IrTargetKind` or similar classifier for rewrite IR nodes. Keep
the original raw target string on each node, but expose typed predicates for
pass logic.

Useful categories:

- Quantize and dequantize.
- Rescale.
- Conv, depthwise conv, transpose conv.
- Linear, matmul, bmm, addmm.
- Layout ops.
- Shape/view ops.
- Constant-producing ops.
- Table ops.
- Control-flow ops.
- Arithmetic and elementwise ops.

Expected payoff:

- Removes repeated target-string checks.
- Makes pass behavior easier to audit.
- Reduces ordering bugs in substring matching.
- Gives future agents one place to inspect op identity rules.

Risk:

- Medium. A wrong classification can silently change many passes.
- Ambiguous targets need explicit tests.
- The classifier must preserve raw target detail where overloads matter.

Touched modules:

- `src/fx_ir.rs`
- `src/graph/ir.rs`
- `src/ingest/from_ir.rs`
- `src/fx_*_ir.rs`
- `src/serialize/visitors/*` only if shared with `OpTarget`

Validation required:

```bash
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
cargo check --lib
```

Then rebuild with `maturin develop --release` and run:

```bash
cd /Users/andrewpullin/personal/executorch-work/executorch
python -m pytest -q backends/arm/test/ops/test_cond.py backends/arm/test/ops/test_while.py
python -m pytest -q --numprocesses=auto backends/arm/test/ --ignore=backends/arm/test/models \
  -k 'tosa and not FP and (INT or 16a8w or a16w8 or quantized or qat)'
python -m pytest -q --numprocesses=auto backends/arm/test/ --ignore=backends/arm/test/models -k 'tosa and FP'
python -m pytest -q --numprocesses=auto backends/arm/test/models -k tosa
```

Add focused Rust tests or Python smoke tests for ambiguous target strings.

## Refactor 2: Introduce Tensor Facts For Metadata

Status:

Initial slice started. `NodeMeta::set_empty_qparam_maps` names the common
"present but empty" qparam-map operation, and `NodeMeta::tensor(shape, dtype)`
now names fresh tensor-valued metadata construction. This is deliberately
smaller than a full `TensorFacts` abstraction because qparam propagation and
dirty-flag semantics need more focused migration. Most one-line local aliases
for fresh tensor metadata have been removed so new tests and passes should call
`NodeMeta::tensor` directly unless a helper adds real domain meaning. Indexed
input-qparam lookups and deterministic sorted input-qparam traversal now also
live on `NodeMeta`. RESCALE output metadata construction now also lives on
`NodeMeta`, covering output-qparam stamping, stale input-qparam clearing,
special-dtype removal, dtype restamping, and layout/spatial metadata retention
for inserted RESCALE nodes. Plain cloned metadata that explicitly drops qparam
domains is now named by `NodeMeta::cloned_without_qparams`, and production
passes no longer spell the old chained operation directly. Several production
passes now also use `NodeMeta::cloned_with_val` directly instead of hand-written
`FakeTensorMeta` clone-and-restamp blocks. Effectful in-place dtype restamps
that preserve existing tensor shape now use
`NodeMeta::set_val_dtype_for_existing_shape`, leaving direct `FakeTensorMeta`
construction in `NodeMeta` internals rather than production pass restamp code.
Indexed qparam slot stamping now also lives on `NodeMeta`, so simple single-slot
input/output qparam assignment no longer requires ad hoc `HashMap` construction.
Defaulted tensor dtype reads now also live on `NodeMeta` via
`val_dtype_or(default)` and `val_dtype_or_float32()`, removing the repeated
decomposition-pass spelling for "missing tensor dtype means float32". Metadata
clones that should preserve the fact that qparam maps are present but empty now
use `NodeMeta::cloned_with_empty_qparam_maps`, distinct from
`cloned_without_qparams`. Present input-qparam maps with multiple slots now use
`NodeMeta::set_input_qparams_from`, and present-but-empty input maps use
`NodeMeta::set_empty_input_qparams`. Present output-qparam maps now also have
the symmetric `NodeMeta::set_output_qparams_from`, and whole-map qparam copying
from one metadata object to another is named by `copy_input_qparams_from` and
`copy_output_qparams_from`. Qparam fact predicates now also live on `NodeMeta`,
including the distinction between "any qparam metadata is present", "any qparam
map is non-empty", and "both input/output qparam maps are non-empty". One-line
pass-local wrappers around these accessors should be inlined unless they name a
real pass-specific invariant. `DType` now owns torch dtype-string parsing via
`DType::from_torch_str`, and `QParam` exposes typed dtype access via
`QParam::dtype()`/`is_integer_dtype()` so constant and serializer conversion
paths no longer treat `tensor_data` as the dtype grammar owner. Tensor metadata
also exposes `NodeMeta::val_dtype_kind()` and `NodeMeta::val_dtype_is(...)`, so
passes that need typed dtype logic should not parse `val_dtype()` strings
directly. Dtype string-domain parsing for float-like and quantized-integer
range semantics now also lives on `DType`, so pass code should not grow local
`contains("int8")`/`contains("float")` trees for qmin/qmax or sentinel values.

Problem:

Shape, dtype, qparams, dim order, spatial rank, table qparams, and extras are
manipulated directly across many passes. The rules for copying, clearing,
restamping, and deriving metadata are distributed.

Proposal:

Add a `TensorFacts` or `TensorDomain` layer over `NodeMeta`. It should expose
named operations instead of direct field mutation for common metadata actions.

Useful operations:

- Copy all facts from another node.
- Copy only dtype and shape.
- Copy qparams from a specific input or output slot.
- Clear stale qparams.
- Stamp output qparams.
- Stamp rank-4 NHWC layout.
- Derive qparams from a RESCALE node.
- Preserve table qparams through table insertion.

Expected payoff:

- Makes the most fragile lowering state explicit.
- Reduces qparam and dtype drift.
- Makes control-flow boundary rules easier to reason about.
- Gives a common language to quantization, layout, and rescale passes.

Risk:

- Medium-high. Metadata regressions can appear as numerical failures far away
  from the edited code.
- Existing helpers may encode subtle pass-specific behavior.

Touched modules:

- `src/fx_ir.rs`
- `src/fx_quant_utils.rs`
- `src/fx_control_flow_metadata.rs`
- `src/fx_insert_*_ir.rs`
- `src/fx_fold_qdq_ir.rs`
- `src/fx_memory_format_ir.rs`
- `src/fx_insert_table_ops_ir.rs`

Validation required:

Run `cargo check --lib`, rebuild with `maturin develop --release`, run
control-flow twice, then run full INT ops, FP ops, and models. Add focused
reruns for conv, linear, layernorm, table ops, and while-loop INT whenever
metadata behavior changes.

## Refactor 3: Add An IR Rewrite Builder

Status:

Initial graph-query slice started. `IrGraph::live_call_function_ids()` now
owns the common "snapshot live call_function nodes before mutating" traversal,
and 46 pass loops use it instead of spelling a local `0..nodes.len()` snapshot
plus alive/op guard. Mixed traversal now has explicit graph-query siblings:
`live_node_ids`, `live_placeholder_ids`, and `live_output_ids`. Memory-format,
output-dim annotation, and int64-buffer cast planning use those helpers instead
of local mixed-node snapshots. Control-flow metadata/contracts, duplicate
placeholder fusion, fold-q/dq candidate collection, and table-op precompute
candidate collection also reuse the shared graph queries. `IrGraph` also has
value-metadata builder siblings for add, anchored add, and replacement
call-function rewrites. `IrNode` now owns common positional-argument reads for
node, integer, numeric, boolean, list, pair-or-scalar, and cloned-or-None args,
plus keyword and arg-or-keyword reads for mixed Python-style schemas. The
table-op insertion pass now calls that `IrNode` arg/kwarg grammar directly
instead of carrying pass-local wrapper aliases. The
indexing family now also shares `fx_index_linear::build_linear_index`, which
owns the broadcast-align, expand, flatten, stride-scale, and sum rewrite chain
used by index.Tensor and index_put; the same module owns contiguous stride
calculation for indexing and as_strided_copy. Constant-effect passes now share
shape-only and slice constant-chain planning through `fx_constant_tensor_eval`
instead of wrapping first-constant-input lookup locally.
Remaining work should keep adding small builder/query operations only when
they replace repeated rewrite structure across several passes.

Problem:

Many passes repeat this pattern:

```text
add a node
copy metadata
set fake tensor value
anchor after a producer
replace old uses
erase old node
```

The repeated boilerplate makes it easy to get topology, metadata, or anchoring
slightly wrong.

Proposal:

Add an `IrBuilder`, `RewriteCursor`, or small family of helper methods for
common graph rewrites.

Useful helpers:

- Insert unary op after producer.
- Insert binary op.
- Replace node with new op.
- Insert TOSA RESCALE with qparam-derived args.
- Insert reshape/view/slice-like shape op.
- Materialize constant-like op.
- Replace uses while preserving output metadata.

Expected payoff:

- Collapses repeated graph mutation code.
- Reduces topological ordering mistakes.
- Makes pass intent more readable.
- Provides a natural home for comments about non-obvious rewrite patterns.

Risk:

- Medium. The builder must preserve current ordering, liveness, anchoring, and
  user-map behavior.
- A too-large builder API can become another kitchen sink.

Touched modules:

- `src/fx_ir.rs`
- `src/fx_decompose_ops_ir.rs`
- `src/fx_insert_conv_rescale_ir.rs`
- `src/fx_insert_int32_rescale_ir.rs`
- `src/fx_insert_qdq_rescale_ir.rs`
- `src/fx_fold_qdq_ir.rs`
- `src/fx_memory_format_ir.rs`
- Other passes that create and replace nodes.

Validation required:

Run `cargo check --lib`, rebuild, run control-flow twice, then run full
INT/FP/models. If any test passes alone but fails in-suite, inspect builder
iteration order and generated node names first.

## Refactor 4: Move Control-Flow Contracts Out Of lib.rs

Status:

Ownership extraction complete. `src/control_flow.rs` now owns IR control-flow
region metadata collection, scalar cond-output marking, while-body passthrough
output appending, subgraph prefix/assembly from stashed IR, parent-to-submodule
qparam propagation, boundary rescale insertion, output normalization contracts,
and quantize/dequantize boundary rewrites. `src/lib.rs` is back to PyO3
entrypoints, stash orchestration, serialization orchestration, and one thin
thread-local wrapper for pre-submodule qparam propagation.

Problem:

`src/lib.rs` should be mostly PyO3 entrypoints and orchestration, but it also
contains substantial control-flow boundary contract logic. That makes ownership
unclear and makes the control-flow system harder to test in isolation.

Proposal:

Create a dedicated module such as:

```text
src/control_flow/contracts.rs
```

Move IR control-flow contract types and helpers there:

- Boundary contract collection.
- Parent-to-submodule qparam propagation.
- Output normalization contracts.
- While-body passthrough output rules.
- Submodule graph prefix/assembly rules if not better owned by ingest.

Expected payoff:

- Makes `lib.rs` easier to read.
- Gives control-flow behavior a single domain module.
- Reduces risk that future PyO3 orchestration edits accidentally alter CF
  lowering semantics.

Risk:

- Low-medium if the move is mechanical.
- Medium if behavior is changed while moving.

Touched modules:

- `src/lib.rs`
- `src/fx_control_flow_metadata.rs`
- `src/ingest/from_ir.rs`
- New `src/control_flow/*`

Validation required:

Run `cargo check --lib`, rebuild, and run control-flow twice. Run models if the
assembly path changes. Run full INT/FP/models if any contract behavior changes.

## Refactor 5: Split Structural Python CF Helpers From IR CF Metadata

Status:

Complete. The temporary
`src/fx_control_flow_structural.rs` inlet module was deleted after its
load-bearing behavior moved into Rust IR import/pre-submodule contracts:
pre-pass qparam snapshots are stored on CF nodes, parent boundary layout facts
are propagated into submodule placeholders before submodule IR passes, and
Q/DQ/RESCALE shape planning now preserves input layout. Active control-flow
lowering code now lives under `src/control_flow/`: region metadata in the
module root, boundary contracts in `contracts.rs`, subgraph assembly in
`assembly.rs`, and while-submodule metadata passes in `metadata.rs`.

Problem:

Before the initial split, `src/fx_control_flow_metadata.rs` mixed
Python-facing structural metadata logic with IR metadata helpers. That module
is now gone from active source, and the control-flow code is split by
responsibility under `src/control_flow/`.

Proposal:

Completed by the control-flow assembly, contracts, and metadata namespace
slices.

Expected payoff:

- Clarifies that control-flow lowering is Rust IR after import.
- Keeps the deleted Python structural seam from returning under another name.
- Gives future CF changes a clear owner.

Risk:

- Medium. Control-flow is fragile and some structural metadata is still
  required at the boundary.

Touched modules:

- `src/fx_control_flow_metadata.rs`
- `src/lib.rs`
- New `src/control_flow/*`

Validation required:

Run `cargo check --lib`, `cargo test --lib`, rebuild, run control-flow twice,
then grep for old round-trip symbols:

```bash
rg 'flush_ir|write_back|ingest_fx_graph_hierarchy|collect_control_flow_subgraphs|re-ingest' \
  /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
```

Any remaining hits should be intentional and documented.

## Refactor 5A: Add A Python Attribute Boundary Grammar

Status:

Initial slices complete. `src/fx_py_attr.rs` now owns `AttrView` and
`MetaView`, the first typed wrapper API for Python boundary reads.
`AttrView` covers both required/error-propagating attrs and lossy/defaulting
attrs. `MetaView` covers metadata-dict reads. `ItemView` now covers keyed and
indexed Python item access plus lossy typed and untyped pair iteration for
dict-like objects.
`QParam::from_py`, `NodeMeta::from_fx`, graph-signature ingestion, kwargs
import, qparam dict extraction, FakeTensor base recovery, tensor-data
shape/dtype extraction, and legacy structural control-flow inlet code all use
this grammar or have been deleted. The old `fx_graph::py_attr`/`meta_val`
helpers and the temporary free-function attr API have been removed. The larger
opportunity is still open in Python import/object construction and any future
dynamic GraphModule attr reads that cross the inlet boundary.

Design note:

Rust can support a larger "reactive" attr interface, but the likely safe shape
is an explicit typed wrapper rather than `[]`-style indexing. Missing attrs,
invalid extractions, and required attrs mean different things in this codebase.
The next subsystem design should therefore preserve separate APIs:

```text
AttrView::required<T>("name") -> PyResult<T>
AttrView::optional<T>("name") -> Option<T>
AttrView::or<T>("name", default) -> T
AttrView::first_or<T>(&["a", "b"], default) -> T
AttrView::has("name") -> bool
MetaView::optional<T>("key") -> Option<T>
ItemView::required_at<T>(0) -> PyResult<T>
ItemView::key("name") -> Option<PyAny>
ItemView::pairs() -> Vec<(PyAny, PyAny)>
```

That wrapper could eventually absorb the current free functions in
`src/fx_py_attr.rs`, but it should not collapse required/error-propagating reads
into lossy/defaulting reads.

Problem:

Rust-side Python ingestion still spells the same semantic action many times:

```text
getattr("name").and_then(|v| v.extract::<T>()).unwrap_or(default)
get_item("name").ok().and_then(|v| v.extract::<T>().ok())
getattr("a").or_else(|_| getattr("b")).and_then(extract)
```

A quick grep after the neutral `QParam` extraction found roughly 300 raw
getattr/extract/default-style hits, concentrated in `src/fx_ir.rs`,
`src/fx_control_flow_metadata.rs`, `src/fx_graph.rs`, and `src/fx_quant.rs`.
This is a migration artifact from the Python-married design: the boundary does
not have a common grammar for heterogeneous attrs, dict metadata, optional
fields, alternate field names, and fallback defaults.

Proposal:

Create a small Python-boundary helper module, likely:

```text
src/fx_py_attr.rs
```

Start with explicit helpers, not a macro system:

```text
attr_opt<T>(obj, name) -> Option<T>
attr_or<T>(obj, name, default) -> T
attr_first_opt<T>(obj, &[names]) -> Option<T>
attr_first_or<T>(obj, &[names], default) -> T
item_opt<T>(dict_like, key) -> Option<T>
item_or<T>(dict_like, key, default) -> T
```

Migrate one high-density caller first, probably `QParam::from_py`, because its
current per-channel/zero-point parsing is the exact repeated pattern and has
focused Rust/Python validation. Then migrate `NodeMeta::from_fx` dict metadata
access and structural control-flow helpers in small batches.

Expected payoff:

- Collapses repeated Python-boundary parsing boilerplate.
- Makes defaulting behavior explicit and searchable.
- Separates legitimate Python seam code from lowering-domain logic.
- Reveals remaining places where Rust should stop reading Python attrs
  entirely and consume already-normalized IR metadata instead.

Risk:

- Medium. Some call sites deliberately distinguish missing attrs from failed
  extraction, and a helper that erases that distinction would hide bugs.
- Medium. `PyResult`-propagating reads and lossy/defaulting reads must stay as
  separate helper families.

Touched modules:

- `src/fx_quant.rs`
- `src/fx_ir.rs`
- `src/fx_graph.rs`
- `src/fx_control_flow_metadata.rs`
- New `src/fx_py_attr.rs`

Validation required:

For a `QParam`-only first slice, run `cargo check --lib`, `cargo test --lib`,
rebuild, and the focused quantization smoke covering conv/matmul/table-style
qparams. If `NodeMeta::from_fx` is migrated, run control-flow twice and a
broader INT ops lane because metadata import feeds nearly every lowering pass.

## Refactor 6: Rename Lowering Effects Away From FX Round-Trip Language

Status:

Complete. `src/lowering_effects.rs` owns the effect model, the effect batch,
and the input-kind enum. `LoweringEffectBatch` stores its vector as `items` to
avoid `effects.effects` stutter at call sites. Effect-producing passes now implement
`EffectfulIrPass`, return `EffectfulIrPassResult`, and enter the pipeline
through `PipelineStep::Effectful`. Active Rust source and architecture docs no
longer use the old `FxEp` / `IrEp` / `EpPass` naming.

Problem:

The old `FxEpEffect` naming reflected the migration history more than the
current architecture. Effects are post-IR lowering effects applied to the
converted serialize graph and tensor store.

Proposal:

Rename concepts toward:

```text
LoweringEffect
LoweringEffectBatch
apply_lowering_effects
```

Keep the data model stable while renaming. Update comments that still describe
effects as something applied to Python FX at flush time.

Expected payoff:

- Removes stale conceptual baggage.
- Makes `ingest/from_ir.rs` easier to understand.
- Helps future agents avoid looking for a deleted Python path.

Risk:

- Low-medium. Mostly naming, but references are broad.

Touched modules:

- `src/lowering_effects.rs`
- `src/ingest/from_ir.rs`
- Constant and table passes that emit effects.
- `src/lib.rs`

Validation required:

Run `cargo check --lib`, rebuild, then run full INT ops because constants,
tables, and materialization are heavily exercised there. Run FP/models after
the rename if any application logic changes.

## Refactor 7: Split Serializer Infrastructure From Visitor Emission

Status:

Complete for central serializer infrastructure. `src/serialize/mod.rs` now only
computes cache/preallocation state and calls region collection plus flatbuffer
emission. Region collection, physical tensor layout/materialization, root-output
layout normalization, flatbuffer region/tensor/operator table writing, and
attribute-union table writing are split into focused modules. Visitor emission
remains under `src/serialize/visitors/*`. Visitor-side input-name lookup now
has a shared single-slot helper beside full input-name collection.

Problem:

The visitor dispatch model is strong, but `src/serialize/mod.rs` still owns too
many concerns at once.

Proposal:

Split central serializer utilities into smaller modules:

```text
src/serialize/flatbuffer.rs
src/serialize/tensors.rs
src/serialize/attrs.rs
src/serialize/layout_outputs.rs
src/serialize/debug.rs
```

Keep `src/serialize/visitors/*` focused on per-op emission.

Expected payoff:

- Makes serializer failures easier to localize.
- Reduces the size of the central serializer module.
- Clarifies the boundary between op-specific emission and shared flatbuffer
  mechanics.

Risk:

- Medium. Serializer edits can affect almost every test.
- Mechanical moves are safer than behavior changes.

Touched modules:

- `src/serialize/mod.rs`
- `src/serialize/visitors/*`
- `src/serialize/tensor_store.rs`
- New serializer helper modules.

Validation required:

Run `cargo check --lib`, rebuild, and run full INT ops, FP ops, and models.
Treat any new failure outside the known baseline set as a regression.

## Refactor 8: Centralize Config And Debug Environment Parsing

Status:

Complete for active Rust source. `src/config.rs` owns Rust env flag names for
from-IR debug, runner logging, fold fallback logging, table debug, memory
layout/slice/cat probes, serializer node probes, ingest/serialize profiling,
and serializer debug checks. Python env parsing remains separate because it
configures Python integration and process orchestration rather than Rust
lowering behavior.

Problem:

Debug and behavior env vars were parsed in multiple modules. Serializer code no
longer reaches through the Python graph wrapper for debug checks.

Proposal:

Add a small config/debug module that owns env parsing and exposes typed config:

```text
src/config.rs
src/debug_config.rs
```

Candidates to centralize are complete for tracked Rust source.

Expected payoff:

- Removes scattered env handling.
- Makes debug behavior reproducible.
- Reduces unnecessary dependencies on `FxGraphModule`.

Risk:

- Low.

Touched modules:

- `src/lib.rs`
- `src/runner_logging.rs`
- `src/serialize/*`
- Visitors using debug flags.
- `src/fx_graph.rs`

Validation required:

Run `cargo check --lib`, rebuild, run one control-flow smoke test, run one INT
test with debug env enabled, and run full suites only if behavior code changes.

## Refactor 9: Quarantine Or Delete Stale Write-Back Vocabulary

Status:

Complete for active Rust source. Tracked source no longer references the
deleted `flush_ir*`, `ingest_fx_graph_hierarchy`,
`collect_control_flow_subgraphs`, or Python write-back architecture. The
structural compatibility helpers that kept `write_back` alive were replaced by
IR import/pre-submodule contracts, and the Python write-back conversion stack
was deleted. The remaining serialize-graph finalization code now lives under
`src/ingest/finalize.rs`, not the stale `fx_bridge` module name. The numeric
round-trip comments in `src/tensor_data.rs` are unrelated encode/decode tests.
Untracked `python/retab/libretab.dylib.dSYM/` may still contain old linker
symbol names and should be treated as build-artifact noise unless explicitly
cleaning artifacts.

Problem:

Some comments and names still reference old concepts such as write-back,
flush, and re-ingest. Even when harmless, these names mislead future work.

Proposal:

Audit and update remaining references. If any write-back-like data fields are
still live for structural reasons, document why they are not a round-trip path.
If they are dead, delete them.

Expected payoff:

- Prevents future agents from chasing deleted architecture.
- Reduces the chance of accidentally reintroducing the Python round-trip.

Risk:

- Low for comments.
- Medium for fields or helpers that still have subtle live uses.

Touched modules:

- `src/fx_ir.rs`
- `src/ingest/from_ir.rs`
- `src/lib.rs`
- `src/fx_control_flow_metadata.rs`
- `python/retab/_helpers.py`

Validation required:

Run `cargo check --lib`, then:

```bash
rg 'write_back|flush_ir|round-trip|re-ingest|ingest_fx_graph_hierarchy|collect_control_flow_subgraphs' \
  /Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine
```

Run control-flow smoke tests after deleting any code.

## Refactor 10: Add Architecture-Level Invariant Tests

Status:

Complete for the listed architecture invariants. `src/ingest/from_ir.rs`
guards prefixed control-flow subgraph node/store-key movement and topological
ordering of late IR insertions before rewired users. `src/control_flow.rs`
guards deterministic nested while-loop region qualification/sorting and checks
active Rust/Python source for deleted Python boundary/round-trip symbols.
Target-classification invariants live in `src/fx_target.rs`.

Problem:

The most important properties are currently validated by large integration
suites, but a few architectural invariants could be checked faster.

Proposal:

Add targeted tests or assertions for:

- No Python re-ingest path is referenced by default lowering.
- Control-flow submodule region names are deterministic.
- TensorStore keys and prefixed node names move together.
- Inserted IR nodes are topologically valid before conversion.
- Target classification handles ambiguous names.

Expected payoff:

- Catches architecture regressions before full battery runs.
- Makes future refactors safer.

Risk:

- Low-medium. Tests must not overfit implementation details that should remain
  flexible.

Touched modules:

- `tests/*`
- `src/ingest/from_ir.rs`
- `src/fx_ir.rs`
- Future target classifier module.

Validation required:

Run `cargo test` for Rust tests if added, `cargo check --lib`, rebuild, and
run the control-flow suite. Run full batteries when tests accompany behavior
changes.

## Recommended Order

The safest order is:

1. Add target grammar with focused tests.
2. Add tensor facts helpers and migrate one or two metadata-heavy passes.
3. Add an IR rewrite builder and migrate one low-risk pass.
4. Move control-flow contracts out of `lib.rs` mechanically.
5. Split control-flow metadata modules.
6. Rename lowering effects.
7. Split serializer infrastructure.
8. Centralize config/debug parsing.
9. Clean stale write-back vocabulary.
10. Add architecture invariant tests as the grammar stabilizes.

The goal is not to minimize file count. The goal is to establish one common
grammar for targets, one common grammar for tensor metadata, and one common
grammar for graph rewrites.
