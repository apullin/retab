#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
vendor_root="$repo_root/../vendor/tosa-rs"
patch_file="$repo_root/vendor_patches/tosa-rs-flatbuffers-size-field.patch"
serialize_rs="$vendor_root/src/serialize.rs"

if [[ ! -f "$serialize_rs" ]]; then
  echo "missing vendored tosa-rs source: $serialize_rs" >&2
  exit 1
fi

if grep -q "size: t.size()" "$serialize_rs" \
  && grep -q "size: tensor.size" "$serialize_rs" \
  && ! grep -q "size: t.size_()" "$serialize_rs" \
  && ! grep -q "size_: tensor.size" "$serialize_rs"; then
  echo "tosa-rs size-field patch already applied"
  exit 0
fi

if [[ ! -f "$patch_file" ]]; then
  echo "missing patch file: $patch_file" >&2
  exit 1
fi

patch -p0 -d "$vendor_root" < "$patch_file"

grep -q "size: t.size()" "$serialize_rs"
grep -q "size: tensor.size" "$serialize_rs"
! grep -q "size: t.size_()" "$serialize_rs"
! grep -q "size_: tensor.size" "$serialize_rs"

echo "applied tosa-rs size-field patch"
