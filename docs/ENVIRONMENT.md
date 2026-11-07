# Environment Variables

This inventory covers environment reads in `python/retab/`, `src/`,
`build.rs`, and the repository's shell helpers. It includes names supplied
through multi-line helper calls and legacy aliases. RETAB does not load an
environment file automatically.

## Python runtime

The `_env_get` helpers in
[preprocess.py](../python/retab/preprocess.py) and
[_decomp_hooks.py](../python/retab/_decomp_hooks.py) return the first defined
name, checking the primary name before its aliases. An empty primary value
still overrides an alias. The effects below depend on the returned value,
not simply on whether the variable exists.

| Variable | Alias | Read location | Effect and default |
| --- | --- | --- | --- |
| `RETAB_DUMP_TOSA` | `ARM_TOSA_ENGINE_DUMP_TOSA` | `preprocess.py`: `_maybe_dump_tosa_artifacts`, `_compile_specs_safe_for_parallel` | A nonempty value is a file path. Creates parent directories and writes the current delegate's TOSA bytes, replacing an existing file. Also forces serial backend preprocessing to avoid dump collisions. Unset/empty disables this dump. |
| `RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS` | `ARM_TOSA_ENGINE_PARALLEL_BACKEND_PREPROCESS_WORKERS` | `preprocess.py`: `_parallel_backend_preprocess_workers`, called during installation and job scheduling | Integer worker count, default `0`. Negative and invalid values become `0`; invalid values log a warning. Values above `1` enable the experimental fork-based multi-delegate path. Debug artifacts and unavailable fork multiprocessing still cause serial execution. |
| `RETAB_FORCE_CF_RECURSIVE_PASSES` | `ARM_TOSA_ENGINE_FORCE_CF_RECURSIVE_PASSES` | `preprocess.py`: `_run_control_flow_submodule_rust_passes` | Exactly `1` runs the full parent pass entry point on each immediate control-flow submodule before the normal named cond/while pipelines. Unset otherwise. This looks like a stale debug path; see the audit notes below. |
| `RETAB_DIRECT_LOWER_NODE_LIMIT` | `ARM_TOSA_ENGINE_DIRECT_LOWER_NODE_LIMIT` | `_decomp_hooks.py`: `_direct_lower_node_limit`, via `_install_direct_lower_size_guard`'s wrapper | Default `128`; invalid integers warn and use `128`. A positive limit skips a direct-lowering attempt when the recursive FX node count exceeds it and the compile configuration enables direct lowering. Zero/negative disables the size guard. Dormant on stock rc4, which has no `TOSAPartitioner.direct_lower`. |
| `ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING` | None | `_decomp_hooks.py`: optimized `_gen_edge_manager_for_partitioners` wrapper installed by `_install_edge_decomp_fast_path` | Exactly `1` attempts `_sync_direct_delegate_output_meta` after generating an edge program. Missing helpers are caught and logged. This legacy helper is absent from stock rc4, so the setting has no successful metadata-sync effect there. |

[integration/executorch_arm.py](../python/retab/integration/executorch_arm.py)
also reads, sets, and restores `RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS`
when `install()` or `activated()` receives a worker argument. Nested
activation saves the previous primary value and restores it on uninstall.
The public argument therefore temporarily takes precedence over the alias.

Backend preprocess counters are process-local. Worker processes do not
automatically aggregate their `preprocess_stats()` results into the parent.

## Rust runtime

All runtime reads except `RETAB_PASS_TIMING` are centralized in
[src/config.rs](../src/config.rs).
Boolean presence checks use `std::env::var_os`: even an empty value or `0`
enables those settings. Exact-value checks use `std::env::var` and require
`1`. A boolean setting with multiple names accepts any matching name;
the primary name does not suppress a matching legacy alias. The node probe
instead takes the first readable string value.

| Variable(s) | Read helper and consumers | Effect and default |
| --- | --- | --- |
| `RETAB_PASS_TIMING` | None (read directly); `src/fx_ir.rs`: `run_ir_pipeline`; `src/lib.rs`: `ingest_and_serialize_tosa_pybytes` | Presence prints each IR pipeline run's total time and its 15 slowest passes, plus the ingest-and-serialize time, to stderr. Unset disables it. |
| `RETAB_PROFILE_INGEST_SERIALIZE` | `profile_ingest_serialize_enabled`; `src/lib.rs`: `ingest_and_serialize_tosa_pybytes` | Presence prints the final from-IR graph, node, and tensor counts to stderr after serialization. Unset disables it. |
| `ARM_FOLD_FALLBACK_FILE` | `fold_fallback_file`; `src/lib.rs`: `fold_fallback_log`, called by `fx_fuse_constant_inputs_ir.rs`, `fx_fuse_zero_input_constants_ir.rs`, `fx_insert_table_ops_ir.rs`, and `ingest/from_ir.rs` | File path for appended constant-materialization and missing-constant diagnostics. Unset disables logging; open/write errors are ignored. These records describe Rust constant handling, not stock backend fallback counts. |
| `RETAB_U55_PERMUTE_LOG` | `u55_permute_log_file`; `src/fx_decompose_permute_for_u55_ir.rs`: `IrDecomposePermuteForU55Pass` | File path. A run of the pass that has a Vela probe to ask (on U55) appends one tab-separated line per live permute whose input breaks U55's worst-case TRANSPOSE rule: its name, input shape, dims, input value dtype, `probe=` whether the probe rejects it, and `split=` `true`, `false`, or `fallback` (the pass sends the graph to stock when the probe rejects a permute that has no split plan). A run without a probe, such as any run off U55, writes nothing. Read on every run, so a test harness can point it at a file per test. Unset disables logging; open/write errors are ignored. |
| `RETAB_FROM_IR_DEBUG`, `ARM_FROM_IR_DEBUG` | `from_ir_debug_enabled`, `from_ir_debug_is_set`; `src/lib.rs`, `ingest/from_ir.rs`, `control_flow/assembly.rs`, `control_flow/contracts.rs` | Exactly `1` enables converted-graph dumps and control-flow boundary diagnostics. Presence alone also enables submodule-reference diagnostics in `graph_from_ir_ext`. Unset disables both. |
| `RETAB_DEBUG`, `RETAB_LOG`, `TOSA_ENGINE_DEBUG`, `TOSA_ENGINE_LOG` | `runner_logging_enabled`; `src/runner_logging.rs`, called by the pass runner in `src/fx_ir.rs` | Presence of any name enables pass-start and pass-result logging to stderr. The result is cached on first use. |
| `RETAB_DEBUG`, `TOSA_ENGINE_DEBUG` | `retab_debug_enabled`; `serialize/mod.rs`, `serialize/collection.rs`, `serialize/flatbuffer.rs`, `serialize/visitors/mod.rs`, `serialize/visitors/reductions/sum_ops.rs` | Presence enables serializer allocation, collection, assembly, missing-emitter, and sum diagnostics. It also checks that every node serializes at its val's shape (the layout invariant, `check_layout_invariants`) and fails serialization with an error, not a stock fallback, on a mismatch. Cached independently on first use. |
| `RETAB_TABLE_DEBUG`, `ARM_TABLE_DEBUG` | `table_debug_enabled`; `src/fx_insert_table_ops_ir.rs` | Presence enables table-generation diagnostics to stderr. Unset disables it. |
| `RETAB_DEBUG_SLICE`, `ARM_TOSA_ENGINE_DEBUG_SLICE` | `debug_slice_enabled`; `serialize/visitors/indexing.rs` | Exactly `1` enables slice/indexing diagnostics. Unset disables it. |
| `RETAB_DEBUG_CAT`, `ARM_TOSA_ENGINE_DEBUG_CAT` | `debug_cat_enabled`; `serialize/visitors/tensor_ops.rs` | Exactly `1` enables concatenation diagnostics. Unset disables it. |
| `RETAB_DEBUG_NODE`, `ARM_TOSA_ENGINE_DEBUG_NODE` | `debug_node_probe`; `src/fx_ir.rs`, `serialize/visitors/mod.rs` | Name substring to trace matching live nodes after passes and at serialization. The first readable value wins; an empty substring matches every node. Unset disables the probe. |

Set cached logging/debug variables before the first lowering call. Changing
them later in the same process does not refresh the cached flags.

## Build and shell helpers

| Variable | Read location | Effect |
| --- | --- | --- |
| `CARGO_MANIFEST_DIR` | `src/control_flow.rs`: the compile-time `option_env!` in `active_source_has_no_legacy_python_boundary_symbols` | Cargo-provided crate root; the Rust invariant test uses it to locate source files. |
| `VIRTUAL_ENV` | [scripts/_common.sh](../scripts/_common.sh): `resolve_venv_dir` | Selects an activated existing environment when `--venv` is absent. An explicit `--venv` wins. Missing both is an error; helpers never create a venv implicitly. `build_retab.sh` exports the selected value for maturin. |
| `EXECUTORCH_SRC` | `scripts/_common.sh`: `resolve_executorch_src` | Selects the source checkout for the source-overlay helpers when `--source`/`--executorch-src` is absent. Missing both is an error. The stock-wheel setup does not need it. |
| `RETAB_TORCH_PACKAGES` | [scripts/install_executorch.sh](../scripts/install_executorch.sh) | Whitespace-separated uv install arguments. Default: `torch torchvision torchao`. This legacy install helper does not pin the tested stack by default. |
| `RETAB_ARM_EXTRA_PACKAGES` | `scripts/install_executorch.sh` | Whitespace-separated uv install arguments for additional Arm packages. Default: `ethos-u-vela==5.2.0`. |
| `HOME` | [scripts/build_retab.sh](../scripts/build_retab.sh) | Locates `.cargo/bin` when extending the build command's `PATH`. |
| `PATH` | `scripts/build_retab.sh`; shell command lookup in the helpers | The build helper prepends `/opt/homebrew/bin`, `$HOME/.cargo/bin`, and the selected venv's `bin`. |

The build helper unsets `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, and
`CONDA_PROMPT_MODIFIER`; these are not RETAB runtime switches.
`PYO3_PYTHON` in the build examples is consumed by the PyO3 dependency to
select the interpreter, rather than by RETAB's own source.
`RETAB_ROOT` and `VENV` in documentation are caller-chosen path placeholders,
not environment configuration read by the package.

## Settings that need a separate audit

- `RETAB_DIRECT_LOWER_NODE_LIMIT` and its alias are dormant on stock rc4:
  their only consumer is a wrapper for the absent `direct_lower` method.
- `ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING` targets an absent stock rc4
  metadata-sync helper. It remains useful only if a compatible checkout
  supplies that helper.
- `RETAB_FORCE_CF_RECURSIVE_PASSES` and its alias are live reads but look
  stale. The normal path already runs named cond/while pipelines, and its
  extra `run_fx_passes` call replaces the parent `STASHED_IR` hierarchy
  before serialization. Keep this legacy debug setting unset pending a
  control-flow audit.

The Rust configuration helpers all have active call sites. This inventory
does not remove or change any runtime switch.
