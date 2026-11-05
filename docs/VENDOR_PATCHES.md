# Vendor Patches

`retab` depends on the local path crate `../vendor/tosa-rs`. That vendor
directory is outside this Git repository, so local edits there are not
tracked with RETAB.

## `tosa-rs` FlatBuffers `size` Field

Patch:

```text
vendor_patches/tosa-rs-flatbuffers-size-field.patch
```

Apply:

```sh
scripts/apply_vendor_patches.sh
```

Reason:

Fresh FlatBuffers-generated Rust for the TOSA schema exposes the tensor size
field as `size`, not `size_`. Older vendored `tosa-rs` source referenced
`TosaTensor::size_()` and `TosaTensorArgs { size_: ... }`, which fails a clean
release build after generated code is rebuilt.

Guard:

`build.rs` checks the vendored source and emits a clear error directing the
developer to run `scripts/apply_vendor_patches.sh` if the vendor tree regresses.
