# Environment Variables

This inventory covers environment reads in `python/retab/`, `src/` and the
repository's shell helpers. Every runtime setting has a single `RETAB_` name:
the legacy `ARM_TOSA_ENGINE_*`, `TOSA_ENGINE_*`, `ARM_FROM_IR_DEBUG` and
`ARM_TABLE_DEBUG` aliases are gone, and `ARM_FOLD_FALLBACK_FILE` is now
`RETAB_FOLD_FALLBACK_FILE` (Linear APU-116). RETAB does not load an
environment file automatically.

## Python runtime

[preprocess.py](../python/retab/preprocess.py) reads these with
`os.environ.get`. The effects below depend on the value, not simply on
whether the variable exists.

| Variable | Read location | Effect and default |
| --- | --- | --- |
| `RETAB_DUMP_TOSA` | `preprocess.py`: `_maybe_dump_tosa_artifacts`, `_compile_specs_safe_for_parallel` | A nonempty value is a file path. Creates parent directories and writes the current delegate's TOSA bytes, replacing an existing file. Also forces serial backend preprocessing to avoid dump collisions. Unset/empty disables this dump. |
| `RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS` | `preprocess.py`: `_parallel_backend_preprocess_workers`, called during installation and job scheduling | Integer worker count, default `0`. Negative and invalid values become `0`; invalid values log a warning. Values above `1` enable the experimental fork-based multi-delegate path. Debug artifacts and unavailable fork multiprocessing still cause serial execution. |

[integration/executorch_arm.py](../python/retab/integration/executorch_arm.py)
also reads, sets, and restores `RETAB_PARALLEL_BACKEND_PREPROCESS_WORKERS`
when `install()` or `activated()` receives a worker argument. Nested
activation saves the previous value and restores it on uninstall.

Backend preprocess counters are process-local. Worker processes do not
automatically aggregate their `preprocess_stats()` results into the parent.

## Rust runtime

All runtime reads except `RETAB_PASS_TIMING` are centralized in
[src/config.rs](../src/config.rs).
Boolean presence checks use `std::env::var_os`: even an empty value or `0`
enables those settings. Exact-value checks use `std::env::var` and require
`1`.

| Variable(s) | Read helper and consumers | Effect and default |
| --- | --- | --- |
| `RETAB_PASS_TIMING` | None (read directly); `src/fx_ir.rs`: `run_ir_pipeline`; `src/lib.rs`: `ingest_and_serialize_tosa_pybytes` | Presence prints each IR pipeline run's total time and its 15 slowest passes, plus the ingest-and-serialize time, to stderr. Unset disables it. |
| `RETAB_PROFILE_INGEST_SERIALIZE` | `profile_ingest_serialize_enabled`; `src/lib.rs`: `ingest_and_serialize_tosa_pybytes` | Presence prints the final from-IR graph, node, and tensor counts to stderr after serialization. Unset disables it. |
| `RETAB_FOLD_FALLBACK_FILE` | `fold_fallback_file`; `src/lib.rs`: `fold_fallback_log`, called by `fx_fuse_constant_inputs_ir.rs`, `fx_fuse_zero_input_constants_ir.rs`, `fx_insert_table_ops_ir.rs`, and `ingest/from_ir.rs` | File path for appended constant-materialization and missing-constant diagnostics. Unset disables logging; open/write errors are ignored. These records describe Rust constant handling, not stock backend fallback counts. |
| `RETAB_U55_PERMUTE_LOG` | `u55_permute_log_file`; `src/fx_decompose_permute_for_u55_ir.rs`: `IrDecomposePermuteForU55Pass` | File path. A run of the pass that has a Vela probe to ask (on U55) appends one tab-separated line per live permute whose input breaks U55's worst-case TRANSPOSE rule: its name, input shape, dims, input value dtype, `probe=` whether the probe rejects it, and `split=` `true`, `false`, or `fallback` (the pass sends the graph to stock when the probe rejects a permute that has no split plan). A run without a probe, such as any run off U55, writes nothing. Read on every run, so a test harness can point it at a file per test. Unset disables logging; open/write errors are ignored. |
| `RETAB_FROM_IR_DEBUG` | `from_ir_debug_enabled`, `from_ir_debug_is_set`; `src/lib.rs`, `ingest/from_ir.rs`, `control_flow/assembly.rs`, `control_flow/contracts.rs` | Exactly `1` enables converted-graph dumps and control-flow boundary diagnostics. Presence alone also enables submodule-reference diagnostics in `graph_from_ir_ext`. Unset disables both. |
| `RETAB_DEBUG`, `RETAB_LOG` | `runner_logging_enabled`; `src/runner_logging.rs`, called by the pass runner in `src/fx_ir.rs` | Presence of either name enables pass-start and pass-result logging to stderr. The result is cached on first use. |
| `RETAB_DEBUG` | `retab_debug_enabled`; `src/lib.rs`: `entry_point`, `serialize/mod.rs`, `serialize/collection.rs`, `serialize/flatbuffer.rs`, `serialize/visitors/mod.rs`, `serialize/visitors/reductions/sum_ops.rs` | Presence enables serializer allocation, collection, assembly, missing-emitter, and sum diagnostics. It also checks that every node serializes at its val's shape (the layout invariant, `check_layout_invariants`) and fails serialization with an error, not a stock fallback, on a mismatch. And it lets a panic in RETAB's Rust code propagate to Python as PyO3's `PanicException`, with Rust's usual panic message (and a backtrace under `RUST_BACKTRACE`). Without it, every Python entry point turns a panic into a stock fallback with the reason `Rust panic`. Cached independently on first use. |
| `RETAB_TEST_PANIC` | `test_panic_entry_point`; `src/lib.rs`: `catch_entry_point_panic` | Test hook. The name of a Python entry point (`run_fx_passes`, `run_fx_cond_passes`, `run_fx_submodule_passes`, `ingest_and_serialize` or `last_unused_inputs`) that panics on entry, so tests can check that a panic falls back to stock (`tests/test_rust_panic_fallback.py`). Read on every call. Unset disables it. |
| `RETAB_TABLE_DEBUG` | `table_debug_enabled`; `src/fx_insert_table_ops_ir.rs` | Presence enables table-generation diagnostics to stderr. Unset disables it. |
| `RETAB_DEBUG_SLICE` | `debug_slice_enabled`; `serialize/visitors/indexing.rs` | Exactly `1` enables slice/indexing diagnostics. Unset disables it. |
| `RETAB_DEBUG_CAT` | `debug_cat_enabled`; `serialize/visitors/tensor_ops.rs` | Exactly `1` enables concatenation diagnostics. Unset disables it. |
| `RETAB_DEBUG_NODE` | `debug_node_probe`; `src/fx_ir.rs`, `serialize/visitors/mod.rs` | Name substring to trace matching live nodes after passes and at serialization. An empty substring matches every node. Unset disables the probe. |

Set cached logging/debug variables before the first lowering call. Changing
them later in the same process does not refresh the cached flags.

## Build and shell helpers

| Variable | Read location | Effect |
| --- | --- | --- |
| `CARGO_MANIFEST_DIR` | `src/control_flow.rs`: the compile-time `option_env!` in `active_source_has_no_legacy_python_boundary_symbols` | Cargo-provided crate root; the Rust invariant test uses it to locate source files. |
| `VIRTUAL_ENV` | [scripts/_common.sh](../scripts/_common.sh): `resolve_venv_dir` | Selects an activated existing environment when `--venv` is absent. An explicit `--venv` wins. Missing both is an error; helpers never create a venv implicitly. `build_retab.sh` exports the selected value for maturin. |
| `HOME` | [scripts/build_retab.sh](../scripts/build_retab.sh) | Locates `.cargo/bin` when extending the build command's `PATH`. |
| `PATH` | `scripts/build_retab.sh`; shell command lookup in the helpers | The build helper prepends `/opt/homebrew/bin`, `$HOME/.cargo/bin`, and the selected venv's `bin`. |

The build helper unsets `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, and
`CONDA_PROMPT_MODIFIER`; these are not RETAB runtime switches.
`PYO3_PYTHON` in the build examples is consumed by the PyO3 dependency to
select the interpreter, rather than by RETAB's own source.
`RETAB_ROOT` and `VENV` in documentation are caller-chosen path placeholders,
not environment configuration read by the package.

The Rust configuration helpers all have active call sites.
