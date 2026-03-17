# Ticket 1B: FX Graph Ingestion (PyO3 Bridge) — Writeup

## Files Created/Modified

### Created
- **`src/ingest/mod.rs`** — Module declaration exposing `fx_bridge` and `target_map` submodules.
- **`src/ingest/fx_bridge.rs`** — Core ingestion logic: walks the FX graph, extracts args/kwargs/meta, builds the Rust `Graph` and `TensorStore`.
- **`src/ingest/target_map.rs`** — Standalone `parse_op_target()` function mapping Python target strings to `OpTarget` enum variants.
- **`tests/test_ingest.py`** — End-to-end test that exports and quantizes `tiny_cnn`, then calls `arm_tosa_engine.ingest_graph()`.

### Modified
- **`src/lib.rs`** — Added `mod ingest;`, the `#[pyfunction] ingest_graph` entry point, and registered it on the `arm_tosa_engine` PyO3 module.

## How Target Mapping Works

`target_map::parse_op_target()` takes the `str(node.target)` output from Python (e.g., `"aten.conv2d.default"`) and matches against known substrings using an ordered chain of `if`/`contains` checks. Order matters for overlapping names — `conv_transpose2d` is checked before `conv2d`, `batch_norm` catches `_native_batch_norm`, etc. Unrecognized targets fall through to `OpTarget::Other(target_string)`.

Quantized decomposed ops (`quantized_decomposed.quantize_per_tensor`, etc.) are checked first to avoid false positives.

## How Args Are Resolved

The `extract_single_arg()` function handles each Python arg value with this priority:

1. **None** — `value.is_none()` → `Arg::None`
2. **FX Node reference** — detected by the presence of `.name` and `.op` attributes. The node's name is looked up in the `name_map: HashMap<String, NodeId>` to resolve to `Arg::Node(id)`.
3. **Bool** — checked before int since Python `bool` is a subclass of `int`.
4. **Int** (`i64`) → `Arg::Int`
5. **Float** (`f64`) → `Arg::Float`
6. **String** → `Arg::String`
7. **Tuple/List** — inspects the first element to decide:
   - If elements are FX nodes → `Arg::NodeList`
   - If elements are ints → `Arg::IntList`
   - If elements are floats → `Arg::FloatList`
8. **torch.dtype** — detected via `type_name == "dtype"`, converted to string and parsed → `Arg::Dtype`
9. **Fallback** — `str()` representation stored as `Arg::String`

Kwargs follow the same logic but are extracted from a `PyDict` and stored as `Vec<(String, Arg)>`.

## PyO3 Type Handling Notes

- **Node detection**: FX Node objects don't have a single reliable Python type name to check, so we use duck-typing — checking for `.name` and `.op` attributes.
- **Bool vs Int**: Python `True`/`False` extract successfully as both `bool` and `i64` via PyO3, so `extract::<bool>()` must be tried first.
- **Meta extraction**: `node.meta["val"]` can be a single FakeTensor, a tuple of FakeTensors, or `None`. The code handles all cases, falling back to a zero-dimensional Float32 `TensorInfo` when no valid tensor metadata is found.
- **Output node**: Its `args[0]` is a tuple of node references. These are handled by the standard tuple/list → `NodeList` path in `extract_single_arg`.
- **State dict bytes**: Tensors are converted via `.contiguous().cpu().numpy().tobytes()` and the resulting `bytes` object is extracted to `Vec<u8>` via PyO3.

## Build/Test Results

- **`cargo check`**: Passes cleanly. The only warnings are dead-code warnings for IR types and graph methods that will be used by later pipeline stages (Tickets 2+).
- **`maturin develop --release`**: Not attempted in this session (environment permission). The code compiles and should link correctly against the Python interpreter in the venv.
- **`tests/test_ingest.py`**: Ready to run once `maturin develop` installs the native module into the venv.
