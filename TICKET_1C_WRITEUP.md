# Ticket 1C: TOSA Flatbuffer Codegen Investigation + Schema Analysis

## 1. TOSA Flatbuffer Schema Structure

**Schema file:** `/Users/andrewpullin/personal/executorch-work/tosa-tools/serialization/schema/tosa.fbs`

The TOSA flatbuffer uses the file identifier `"TOSA"` and file extension `.tosa`.

### Root Table: `TosaGraph`

```
TosaGraph (root_type)
  version: Version (required)
  regions: [TosaRegion]
  software_version: SoftwareVersion
```

### Nesting Hierarchy

```
TosaGraph
 +-- Version { _major, _minor, _patch, _draft }
 +-- SoftwareVersion { _major, _minor, _micro, _modifier }
 +-- TosaRegion[]
      +-- name: string
      +-- TosaBasicBlock[]
           +-- name: string
           +-- operators: [TosaOperator]
           +-- tensors: [TosaTensor]
           +-- shapes: [TosaShape]
           +-- inputs: [string]    (tensor names)
           +-- outputs: [string]   (tensor names)
```

### Key Tables

**TosaOperator:**
- `op`: Op enum (uint32)
- `attribute`: Attribute union -- one per op type
- `inputs`: [string] -- tensor names
- `outputs`: [string] -- tensor names
- `location`: OpLocation (optional debug metadata)

**TosaTensor:**
- `name`: string (used for dependency resolution between operators)
- `shape`: [int32]
- `type`: DType enum
- `data`: [ubyte] (force_align: 8) -- raw bytes for constants
- `variable`: bool
- `is_unranked`: bool
- `variable_name`: string
- `offset`/`size`: ulong -- for >2GB models (external data)

**TosaShape:**
- `name`: string
- `rank`: uint32
- `data`: [ubyte] (force_align: 8)

### Key Enums

**DType** (uint32): UNKNOWN=0, BOOL=1, INT4=2, INT8=3, INT16=4, INT32=5, INT48=6, FP32=7, FP16=8, BF16=9, SHAPE=10, FP8E4M3=11, FP8E5M2=12, ..., MXINT8=18

**Op** (uint32): 53 operations total. Key ones for our pipeline:
- CONV2D=3, DEPTHWISE_CONV2D=5, ADD=14, MUL=27, SUB=29
- CLAMP=11, MAX_POOL2D=8, AVG_POOL2D=2
- RESHAPE=31, TRANSPOSE=35, CONCAT=29, PAD=30, SLICE=33
- CAST=39, RESCALE=40, CONST=41, CONST_SHAPE=49
- TABLE=30

**Attribute** (union): One attribute table per op. Each op has a mandatory attribute even if it is empty (e.g., `AddAttribute {}`, `MulAttribute {}`). Attributes with actual fields include:
- `Conv2dAttribute { pad, stride, dilation: [int32]; local_bound: bool; acc_type: DType }`
- `RescaleAttribute { scale32: bool; rounding_mode: RoundingMode; per_channel, input_unsigned, output_unsigned: bool }`
- `ClampAttribute { min_val, max_val: [ubyte]; nan_mode: NanPropagationMode }`
- `TransposeAttribute { perms: [int32] }`
- `ConcatAttribute { axis: int32 }`
- `ResizeAttribute { mode: ResizeMode }`
- Many others are empty tables (e.g., `AddAttribute {}`, `SubAttribute {}`, `ReshapeAttribute {}`)

### Important Schema Design Notes

1. **Every op requires an attribute** -- even ops like ADD that have no parameters still require an empty `AddAttribute` table in the union. The Python serializer auto-populates these if not explicitly set.
2. **Tensor data is inline** -- constant tensor data is embedded directly in the flatbuffer as `[ubyte]` in `TosaTensor.data`, using little-endian byte encoding.
3. **Tensor references use string names** -- operators reference inputs/outputs by string name, not by index. This is different from many ML frameworks.
4. **CONST ops are special** -- constant tensors are added both as tensor entries (with data) AND as operators (`Op.CONST` with empty inputs and one output name).

## 2. Python Serializer Step-by-Step

**Source:** `/Users/andrewpullin/personal/executorch-work/u55-pipeline/.venv/lib/python3.12/site-packages/serializer/tosa_serializer.py`

Note: The ARM backend actually imports `tosa_serializer` which is a C++ extension (`_tosa_serializer.cpython-312-darwin.so`) wrapping the Python serializer. The API is the same.

### Construction Flow

1. **Create TosaSerializer** -- initializes a flatbuffers.Builder, creates a "main" region with a "main" basic block:
   ```python
   tosa_graph = ts.TosaSerializer(artifact_path)
   # Auto-creates: regions["main"] -> basicBlocks["main"]
   ```

2. **Add input tensors** -- via `addInputTensor(tensor)`:
   - Creates a `TosaSerializerTensor` (name, shape, dtype, no data)
   - Adds to `currBasicBlock.tensors` dict
   - Adds name to `currBasicBlock.inputs` list

3. **Add constant tensors** -- via `addConst(shape, dtype, vals, name=)`:
   - Creates a `TosaSerializerTensor` with data bytes embedded
   - Adds a `CONST` operator (inputs=[], outputs=[name])
   - Constants are stored as numpy arrays, converted to uint8 byte vectors for flatbuffer embedding

4. **Add intermediate tensors** -- via `addTensor(name, shape, dtype)` on the basic block:
   - Output tensors from operators (no data, just metadata)

5. **Add operators** -- via `addOperator(op, inputs, outputs, attributes)`:
   - Creates `TosaSerializerOperator` with op enum, input/output name lists, and serialized attribute
   - Attributes are built via `TosaSerializerAttribute` which stores builder functions for later serialization

6. **Mark outputs** -- via `addOutputTensor(tensor)`:
   - Adds tensor name to `currBasicBlock.outputs`

7. **Serialize** -- `tosa_graph.serialize()`:
   - Builds Version and SoftwareVersion tables
   - Serializes all regions -> blocks -> operators/tensors/shapes bottom-up
   - Calls `builder.Finish(graph, "TOSA")` and returns bytes

### Serialization Detail

The `TosaSerializerAttribute` class uses a deferred builder pattern:
- Each attribute method (e.g., `Conv2dAttribute(pad, stride, ...)`) stores function references and values
- At serialize time, it calls the flatbuffer Start/Add/End functions in order
- Strings and vectors are pre-serialized, then primitives and bools are added

Data encoding in `convertDataToUint8Vec`:
- BOOL: 1 byte per value
- INT4: nibble-packed (2 values per byte)
- INT8: 1 byte per value (cast via np.uint8)
- INT16: 2 bytes little-endian
- INT32: 4 bytes little-endian
- INT48: 6 bytes little-endian
- INT64/SHAPE: 8 bytes little-endian
- FP32: 4 bytes (np.float32 tobytes)
- FP16: 2 bytes (np.float16 tobytes)
- BF16: 2 bytes (via ml_dtypes)

## 3. What Each Visitor Produces

### Conv2d Visitor (`op_tosa_conv2d.py`)

**Target:** `tosa.CONV2d.default`
**TOSA Op:** `Op.CONV2D`

**Inputs (5):** `[input, weight, bias, input_zp, weight_zp]`
- `input`: activation tensor name
- `weight`: weight tensor name (already const from process_placeholder)
- `bias`: bias tensor name (already const from process_placeholder)
- `input_zp`: scalar zero-point const added inline (shape=[1], dtype=input.dtype)
- `weight_zp`: scalar zero-point const added inline (shape=[1], dtype=weight.dtype)

**Outputs (1):** `[conv2d_output_name]`

**Attribute:** `Conv2dAttribute`
- `pad`: from input args (list of int32)
- `stride`: from input args (list of int32)
- `dilation`: from input args (list of int32)
- `local_bound`: False
- `acc_type`: output dtype (or FP32 if BF16/FP16 inputs)

**Const tensors created:** 2 -- `{output_name}_input_zp` and `{output_name}_weight_zp` via `add_input_weight_zp_consts()`. These are scalar tensors holding zero-point values.

### Add Visitor (`op_add.py`)

**Target:** `aten.add.Tensor`
**TOSA Op:** `Op.ADD`

Uses `SimpleNodeVisitor` -- the simplest pattern.

**Inputs (2):** `[lhs_name, rhs_name]`
**Outputs (1):** `[output_name]`
**Attribute:** `AddAttribute` (empty table, no fields)
**Const tensors created:** None

### Rescale Visitor (`op_tosa_rescale.py`)

**Target:** `tosa.RESCALE.default`
**TOSA Op:** `Op.RESCALE`

This is the most complex visitor. RESCALE is used for requantization between quantized operators.

**Inputs (5):** `[input_name, multipliers_name, shifts_name, input_zp_name, output_zp_name]`

**Attribute:** `RescaleAttribute`
- `scale32`: bool (True for INT8/INT16/INT32 inputs, False for INT48)
- `rounding_mode`: RoundingMode.SINGLE_ROUND
- `per_channel`: bool (True if len(scales) > 1)
- `input_unsigned`: False
- `output_unsigned`: False

**Const tensors created (4):**
1. `{output_name}_multipliers` -- shape=(N,), dtype=INT32 or INT16, computed from float scales via `_compute_multiplier_and_shift()`
2. `{output_name}_shifts` -- shape=(N,), dtype=INT8
3. `{output_name}_input_zp` -- shape=(1,), dtype=input.dtype, zero-point value
4. `{output_name}_output_zp` -- shape=(1,), dtype=output.dtype, zero-point value

**Scale computation:** Floating-point scales are decomposed into (multiplier, shift) pairs using `math.frexp()`. The multiplier is scaled to fit in 31 bits (or 15 bits for INT48), and the shift is adjusted so that `result = (input * multiplier) >> shift`.

## 4. flatc Availability and Rust Codegen

### flatc Location
`flatc` is available in the project venv at:
```
/Users/andrewpullin/personal/executorch-work/u55-pipeline/.venv/bin/flatc
```

The flatbuffers Python package version is 24.3.25 (tosa-tools requires 25.2.10). The `flatc` binary ships with the `tosa-tools` package.

### Rust Codegen Support

`flatc --rust` has been supported since flatbuffers v2.0 (2021). The command to generate Rust code would be:

```bash
flatc --rust -o src/tosa_generated/ \
  /path/to/tosa-tools/serialization/schema/tosa.fbs
```

This generates:
- One `.rs` file per table/enum
- Uses the `flatbuffers` Rust crate (runtime)
- Generates zero-copy read accessors (for deserialization)
- Generates builder types (for serialization)

### Alternatives

1. **`flatbuffers` crate runtime-only**: Read raw bytes using the `flatbuffers` crate's `root::<T>()` without codegen. Requires manually defining all types. Not practical for a schema this large.

2. **`planus` crate**: Alternative Rust flatbuffers implementation with better ergonomics. Generates Rust from `.fbs` at build time via `build.rs`. Worth considering but less battle-tested.

3. **Manual byte building**: Write raw flatbuffer bytes. Possible but error-prone and not recommended given the schema complexity.

## 5. Reference TOSA File

### Script Location
A script to generate the reference TOSA flatbuffer is at:
```
/tmp/dump_tosa_reference.py
```

### How to Run

```bash
cd /Users/andrewpullin/personal/executorch-work/u55-pipeline
VIRTUAL_ENV=.venv PATH="$VIRTUAL_ENV/bin:/Users/andrewpullin/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" \
  python /tmp/dump_tosa_reference.py
```

This will produce:
- `/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/test_data/tiny_cnn_reference.tosa`
- `/Users/andrewpullin/personal/executorch-work/u55-pipeline/arm-tosa-engine/test_data/tiny_cnn_reference_summary.txt`

The script runs stages 1-3 of `lower_cnn.py` with `intermediate_path` set so the TOSA backend dumps the `.tosa` file, then parses and dumps a human-readable summary.

### Expected Content for tiny_cnn

The `TinyCNN` model (Conv2d->ReLU->MaxPool2d->Conv2d->ReLU->MaxPool2d->Flatten->Linear) will produce approximately:
- **Input:** 1 tensor, shape [1,1,28,28], INT8 (quantized)
- **CONV2D** x2 (each with 5 inputs: activation, weight, bias, input_zp, weight_zp)
- **CLAMP** x2 (implementing ReLU)
- **MAX_POOL2D** x2
- **RESHAPE** x1 (flatten)
- **FULLY_CONNECTED** or **MATMUL** x1 (linear layer) -- or it may be lowered as CONV2D with reshape
- **RESCALE** ops between quantized operators (requantization)
- Multiple **CONST** ops for weights, biases, zero-points, multipliers, shifts

## 6. Recommendation: flatc Codegen vs Manual Flatbuffer Building

### Recommendation: Use `flatc --rust` codegen

**Reasons:**

1. **Schema complexity**: 53 ops, ~53 attribute types, nested tables. Hand-writing this would take days and be error-prone.

2. **flatc is already available**: The venv ships `flatc` and the `.fbs` schema. Integration is straightforward.

3. **Zero-copy deserialization for free**: The generated code gives us efficient readers for test verification -- we can deserialize reference `.tosa` files to compare against our output.

4. **Serialization builders**: `flatc --rust` generates `FlatBufferBuilder`-based builder types. We call `TosaGraphBuilder`, `TosaRegionBuilder`, etc. in bottom-up order, mirroring the Python serializer's approach.

5. **Type safety**: The generated union types ensure we attach the right attribute to each op at compile time.

### Suggested Integration

Add to `Cargo.toml`:
```toml
[dependencies]
flatbuffers = "24.3"  # runtime crate

[build-dependencies]
# Optional: run flatc in build.rs
```

Add a `build.rs` that runs:
```
flatc --rust -o src/tosa_generated/ <path>/tosa.fbs
```

Or check in the generated code and regenerate manually when the schema changes.

### What we will NOT need

- The `TosaShape` table (only used for `CONST_SHAPE` ops, which appear in dynamic shape scenarios)
- The `offset`/`size` fields on `TosaTensor` (only for >2GB models)
- The `variable`/`variable_name` fields (only for variable tensors)
- Most attribute tables -- for the tiny_cnn model we only need: `Conv2dAttribute`, `ClampAttribute`, `MaxPool2dAttribute`, `RescaleAttribute`, `ReshapeAttribute`, `AddAttribute`, `ConstAttribute`, and possibly `TransposeAttribute`
- The `OpLocation` table (debug-only metadata)
